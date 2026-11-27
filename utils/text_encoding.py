"""Captions encoded per batch during training, for a model whose text embeddings are otherwise
always cached (cache_text_embeddings = false on a model without encodes_text_on_the_fly).

The model's own text encoder functions -- the ones that fill the cache -- encode each batch's
captions in the training process, and their results go into the batch as the cache's rows would
have, so prepare_inputs sees the same keys, shapes and dtypes either way.

With pipeline parallelism the first and the last stage both pull every batch through prepare_inputs
(the last stage keeps only the label), so both would load the text encoders and encode each caption.
Only the first stage does: it sends the encoded rows to the last stage, and the stages that pull no
batch drop the encoders they were built with.
"""

from inspect import signature

import torch
import torch.distributed as torch_dist
from torch import nn

# Put into each sample by the dataset for TextEncodingOnTheFly, which takes them out of the batch
# again before it reaches the model.
IS_VIDEO_KEY = '_text_encoder_is_video'
CONTROL_FILE_KEY = '_text_encoder_control_file'


def encodes_text_in_loader(model):
    """Whether this model's captions are encoded per batch by TextEncodingOnTheFly."""
    model_config = (getattr(model, 'config', None) or {}).get('model', {})
    return (not model_config.get('cache_text_embeddings', True)
            and not getattr(model, 'encodes_text_on_the_fly', False)
            and len(model.get_text_encoders()) > 0)


def _row_to_cpu(value):
    # A copy, as the cache stores it: a row of a batched result would otherwise keep the whole
    # batch's memory alive, and the text encoder's device is the GPU.
    if torch.is_tensor(value):
        return value.to('cpu', copy=True)
    if isinstance(value, (list, tuple)):
        return [_row_to_cpu(x) for x in value]
    return value


def _batched(features):
    # The rule utils/dataset.py's _collate applies to cached rows.
    if torch.is_tensor(features[0]):
        shape = features[0].shape
        if all(torch.is_tensor(f) and f.shape == shape for f in features):
            return torch.stack(features)
    return features


class TextEncodingOnTheFly:
    """Encodes the captions of a batch with every text encoder of the model.

    Captions go to the encoder in groups of caching_batch_size, as caching sends them; each row is
    then split out and the rows batched as the cache's rows are. At the default of 1 every caption
    is encoded alone, so a batch holds exactly what the cache would; a larger group pads a caption
    to the longest one of its group, which is a group of the batch here and of the caption files'
    order when caching. The text encoders move to the GPU at the first batch and stay there.
    """

    def __init__(self, model, caching_batch_size=1):
        self.model = model
        self.text_encoders = list(model.get_text_encoders())
        self.call_fns = [model.get_call_text_encoder_fn(text_encoder) for text_encoder in self.text_encoders]
        self.takes_control_file = [len(signature(fn).parameters) == 3 for fn in self.call_fns]
        self.caching_batch_size = caching_batch_size
        self.loaded = False
        self.encoders_loaded = False
        # None: this process encodes and nobody else needs the rows. ('send', rank): it encodes and sends
        # each batch's rows to the last stage. ('recv', rank): it takes them from the first stage.
        self.relay = None

    def bind(self, engine):
        """Say which stage of the pipeline this process is; the data loader calls it once with its engine.

        The first stage encodes and sends its rows to the last, which has no encoder to run; a stage in
        between pulls no batch at all. Neither of those two keeps the text encoders it was built with."""
        self.relay = None
        if not getattr(engine, 'is_pipe_parallel', False):
            return
        grid = engine.grid
        if engine.is_first_stage():
            self.relay = ('send', grid.stage_to_global(engine.num_stages - 1))
            # Now rather than at the first batch: the last stage is already waiting for that batch's rows, and
            # a long read of the encoders (a big one from a network volume) would count against its timeout.
            self._load_encoders()
            return
        if engine.is_last_stage():
            self.relay = ('recv', grid.stage_to_global(0))
        # The encoders were built with the model on every rank; this one never runs them.
        from utils.model_teacher import empty_weights
        empty_weights(self.text_encoders)
        self.text_encoders, self.call_fns, self.takes_control_file = [], [], []

    @staticmethod
    def _object_device():
        return torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else None

    def _send(self, value):
        torch_dist.send_object_list([value], dst=self.relay[1], device=self._object_device())

    def _receive(self):
        received = [None]
        torch_dist.recv_object_list(received, src=self.relay[1], device=self._object_device())
        return received[0]

    def _load_encoders(self):
        if self.encoders_loaded:
            return
        self.encoders_loaded = True
        for text_encoder in self.text_encoders:
            if isinstance(text_encoder, nn.Module):
                # eval: an encoder built from its config starts in train mode, with dropout on.
                text_encoder.to('cuda').eval()
            else:
                # A ComfyUI model behind a ModelWrapper; ComfyUI moves it to the GPU per call.
                text_encoder.load_model_if_needed()

    def _load(self):
        self.loaded = True
        if self.relay is not None and self.relay[0] == 'recv':
            self.model.uncond_dict = self._receive()
            return
        self._load_encoders()
        # The empty caption's embeddings, which caching leaves in model.uncond_dict for a model
        # that reads them (MiniMax H3 with cfg > 1). Encoded as caching encodes it.
        self.model.uncond_dict = self._encode([''], [False], None)[0]
        if self.relay is not None:
            self._send(self.model.uncond_dict)

    @torch.no_grad()
    def __call__(self, batch):
        if not self.loaded:
            self._load()
        captions = list(batch['caption'])
        is_video = list(batch.pop(IS_VIDEO_KEY))
        control_files = list(batch.pop(CONTROL_FILE_KEY))
        if all(control_file is None for control_file in control_files):
            # As caching passes it for a dataset without control images.
            control_files = None
        if self.relay is not None and self.relay[0] == 'recv':
            batch.update(self._receive())
            return batch
        rows = self._encode(captions, is_video, control_files)
        encoded = {key: _batched([row[key] for row in rows]) for key in rows[0]}
        if self.relay is not None:
            self._send(encoded)
        batch.update(encoded)
        return batch

    def _encode(self, captions, is_video, control_files):
        """One dict per caption, holding each text encoder's results for it."""
        rows = [{} for _ in captions]
        # Caching runs with no autocast; the training step may have one active.
        with torch.autocast('cuda', enabled=False):
            for call_fn, takes_control_file in zip(self.call_fns, self.takes_control_file):
                for start in range(0, len(captions), self.caching_batch_size):
                    end = min(start + self.caching_batch_size, len(captions))
                    args = [captions[start:end], is_video[start:end]]
                    if takes_control_file:
                        args.append(None if control_files is None else control_files[start:end])
                    result = call_fn(*args)
                    for key, value in result.items():
                        for i in range(end - start):
                            rows[start + i][key] = _row_to_cpu(value[i])
        return rows


def free_text_encoder_off_first_stage(text_encoder, engine):
    """For a model whose first pipeline layer runs its text encoder: the other stages were built with the
    encoder too (the model loads it before the stages exist) and never run it, so its weights go from their
    memory. Only the weights: the tokenizer, which prepare_inputs of the first and last stage uses, stays.
    Returns whether anything was freed."""
    if text_encoder is None or not getattr(engine, 'is_pipe_parallel', False) or engine.is_first_stage():
        return False
    from utils.model_teacher import empty_weights, free_gpu_cache
    empty_weights([text_encoder])
    free_gpu_cache()
    return True
