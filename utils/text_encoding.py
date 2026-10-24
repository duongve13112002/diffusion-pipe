"""Captions encoded per batch during training, for a model whose text embeddings are otherwise
always cached (cache_text_embeddings = false on a model without encodes_text_on_the_fly).

The model's own text encoder functions -- the ones that fill the cache -- encode each batch's
captions in the training process, and their results go into the batch as the cache's rows would
have, so prepare_inputs sees the same keys, shapes and dtypes either way.
"""

from inspect import signature

import torch
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

    def _load(self):
        for text_encoder in self.text_encoders:
            if isinstance(text_encoder, nn.Module):
                # eval: an encoder built from its config starts in train mode, with dropout on.
                text_encoder.to('cuda').eval()
            else:
                # A ComfyUI model behind a ModelWrapper; ComfyUI moves it to the GPU per call.
                text_encoder.load_model_if_needed()
        self.loaded = True
        # The empty caption's embeddings, which caching leaves in model.uncond_dict for a model
        # that reads them (MiniMax H3 with cfg > 1). Encoded as caching encodes it.
        self.model.uncond_dict = self._encode([''], [False], None)[0]

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
        rows = self._encode(captions, is_video, control_files)
        for key in rows[0]:
            batch[key] = _batched([row[key] for row in rows])
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
