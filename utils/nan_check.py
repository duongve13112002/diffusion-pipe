"""train.py --nan_check: stop at the first non-finite value of a real training run and say where.

Every layer output, every parameter gradient, the loss and the weights after each optimizer step
are checked, on every rank. The first non-finite one is reported with the step, the rank, the
layer or parameter, and the images of the batch, and training stops (an exception on one rank
makes the deepspeed launcher stop the others). Each check reads a value back from the GPU, so this
slows training down: it is for finding a NaN, not for leaving on.

A non-finite gradient is then replayed: the layer it belongs to is run again, forward and backward,
on the inputs and output gradients it had in this step, for the whole batch with each attention
kernel and for every sample alone. That says whether the NaN is reproducible, whether it depends
on the attention kernel, and which samples cause it.
"""

from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path

import torch
from torch import nn


class NonFiniteError(RuntimeError):
    pass


def _non_finite(tensor):
    return torch.is_tensor(tensor) and torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all())


def _describe_tensor(tensor):
    bad = ~torch.isfinite(tensor)
    nan = int(torch.isnan(tensor).sum())
    finite = tensor[~bad]
    peak = float(finite.float().abs().max()) if finite.numel() else float('nan')
    return (f'{int(bad.sum())} of {tensor.numel()} non-finite ({nan} NaN), shape {tuple(tensor.shape)}, '
            f'dtype {tensor.dtype}, max |finite| {peak:.4g}')


def batch_names(inputs):
    specs = inputs.get('image_spec')
    if specs is None:
        return None
    return [str(Path(spec[1]).name) if isinstance(spec, (list, tuple)) else str(spec) for spec in specs]


def _in_backward():
    try:
        return torch._C._current_graph_task_id() != -1
    except AttributeError:
        return False


@contextmanager
def _sdpa_only(backend):
    # With [model] attention = 'flash3' / 'flash4', the SDPA replacement would ignore sdpa_kernel
    # and run FlashAttention again; bypass it so each replay runs the PyTorch kernel it names.
    from torch.nn.attention import sdpa_kernel
    from utils import attention
    with attention.bypass(), sdpa_kernel(backend):
        yield


def _attention_kernels():
    """(name, context) for each SDPA kernel to replay with: the default choice first."""
    from utils import attention
    default = f'default kernel ({attention.label()} where it can)' if attention.flash_enabled() else 'default kernel'
    kernels = [(default, nullcontext)]
    try:
        from torch.nn.attention import SDPBackend
    except ImportError:
        return kernels
    for name in ('FLASH_ATTENTION', 'CUDNN_ATTENTION', 'EFFICIENT_ATTENTION', 'MATH'):
        if hasattr(SDPBackend, name):
            kernels.append(('PyTorch SDPA ' + name.lower().replace('_attention', '') + ' kernel',
                            partial(_sdpa_only, getattr(SDPBackend, name))))
    return kernels


def describe_batch(inputs, tokenizer=None):
    """A short, printable account of a dataloader batch: what to look at when it is the cause.
    With the model's tokenizer, caption lengths are also given in tokens (before truncation)."""
    lines = []
    latents = inputs.get('latents')
    if torch.is_tensor(latents):
        lines.append(f'latents {tuple(latents.shape)} {latents.dtype}')
    mask = inputs.get('mask')
    if torch.is_tensor(mask):
        lines.append(f'mask {tuple(mask.shape)} min {float(mask.min()):.4g} max {float(mask.max()):.4g}')
    captions = inputs.get('caption')
    if captions is not None:
        lengths = [len(c) for c in captions]
        lines.append(f'caption lengths (chars) {lengths}')
        if tokenizer is not None:
            try:
                tokens = [len(ids) for ids in tokenizer(list(captions), add_special_tokens=False)['input_ids']]
                lines.append(f'caption lengths (tokens, before truncation) {tokens}')
            except Exception as error:  # a description must never be what stops training
                lines.append(f'caption tokens unavailable: {error}')
    if (names := batch_names(inputs)) is not None:
        lines.append(f'images {names}')
    return lines


class NanCheck:
    def __init__(self, pipeline_model, model, rank):
        self.rank = rank
        self.step = 0
        self.batches = []
        self.names = None
        self.handles = []
        self.layers = list(pipeline_model.forward_funcs)
        self.layer_inputs = {}  # layer index -> its inputs this step (kept alive by autograd anyway)
        self.output_grads = {}  # layer index -> {output position: gradient}, latest layer only
        self.replaying = False
        self.param_layer = {}
        for index, layer in enumerate(self.layers):
            if isinstance(layer, nn.Module):
                self.handles.append(layer.register_forward_hook(partial(self._after_layer, index)))
                for p in layer.parameters():
                    self.param_layer.setdefault(id(p), index)
        for name, p in pipeline_model.named_parameters():
            if p.requires_grad:
                self.handles.append(p.register_post_accumulate_grad_hook(partial(self._after_grad, name)))
        self.parameters = [(name, p) for name, p in pipeline_model.named_parameters()]
        original = model.prepare_inputs
        tokenizer = getattr(getattr(model, 'text_encoder', None), 'tokenizer', None)

        def prepare_inputs(inputs, *args, **kwargs):
            # Kept per pulled batch: with gradient accumulation a step consumes several.
            self.batches = (self.batches + [describe_batch(inputs, tokenizer)])[-4:]
            self.names = batch_names(inputs)
            return original(inputs, *args, **kwargs)
        model.prepare_inputs = prepare_inputs

    def _fail(self, what, detail, extra=()):
        lines = [f'[nan_check] rank {self.rank}, step {self.step}: {what}', f'[nan_check]   {detail}']
        lines += [f'[nan_check]   {line}' for line in extra]
        for i, batch in enumerate(self.batches[::-1]):
            lines.append(f'[nan_check]   batch {"(latest)" if i == 0 else f"(-{i})"}:')
            lines += [f'[nan_check]     {line}' for line in batch]
        message = '\n'.join(lines)
        print(message, flush=True)
        raise NonFiniteError(message)

    def _after_layer(self, index, module, inputs, output):
        if self.replaying:
            return
        outputs = output if isinstance(output, (tuple, list)) else (output,)
        if not _in_backward():  # not an activation checkpointing recomputation
            args = inputs[0] if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)) else inputs
            self.layer_inputs[index] = tuple(args)
            if torch.is_grad_enabled():
                for j, tensor in enumerate(outputs):
                    # A pass-through output is the next layer's input object, whose gradient
                    # also sums what earlier layers do with it: not this layer's to replay.
                    if (torch.is_tensor(tensor) and tensor.requires_grad
                            and not any(tensor is a for a in args)):
                        tensor.register_hook(partial(self._output_grad, index, j))
        for j, tensor in enumerate(outputs):
            if _non_finite(tensor):
                inputs_ok = not any(_non_finite(t) for t in (inputs[0] if inputs and isinstance(inputs[0], tuple) else inputs))
                self._fail(f'layer {index} {type(module).__name__} output {j} is non-finite'
                           + (' (its inputs were finite)' if inputs_ok else ' (its inputs already were not)'),
                           _describe_tensor(tensor))

    def _output_grad(self, index, j, grad):
        # Backward has finished every layer after this one: what was kept for them can go.
        for other in [i for i in self.output_grads if i != index]:
            del self.output_grads[other]
        for other in [i for i in self.layer_inputs if i > index]:
            del self.layer_inputs[other]
        self.output_grads.setdefault(index, {})[j] = grad

    def _after_grad(self, name, p):
        if self.replaying or not _non_finite(p.grad):
            return
        index = self.param_layer.get(id(p))
        try:
            extra = self._replay_report(index)
        except Exception as error:  # the report must not hide the finding
            extra = [f'replay failed: {type(error).__name__}: {error}']
        self._fail(f'gradient of {getattr(p, "original_name", name)} is non-finite '
                   '(backward runs last layer first: this is the first parameter it reached)',
                   _describe_tensor(p.grad), extra)

    def _replay(self, index, kernel=nullcontext, sample=None):
        """Forward and backward of layer `index` on its recorded inputs and output gradients.
        Returns the names of the parameters and inputs whose gradient is non-finite."""
        layer = self.layers[index]
        args = self.layer_inputs[index]
        batch = next(t.shape[0] for t in args if torch.is_tensor(t) and t.ndim)

        def pick(t):
            # Samples are the leading dim of every per-sample tensor; a non-floating 2-D tensor
            # (Iris's patch grid marker) is shared by the batch.
            if (sample is not None and torch.is_tensor(t) and t.ndim and t.shape[0] == batch
                    and (torch.is_floating_point(t) or t.ndim >= 3)):
                t = t[sample:sample + 1]
            if torch.is_tensor(t) and torch.is_floating_point(t):
                return t.detach().requires_grad_()
            return t
        replay_args = tuple(pick(t) for t in args)
        grads = {j: (g[sample:sample + 1] if sample is not None else g)
                 for j, g in self.output_grads[index].items()}
        named = [(getattr(p, 'original_name', n), p) for n, p in layer.named_parameters() if p.requires_grad]
        named += [(f'input {i}', t) for i, t in enumerate(replay_args) if torch.is_tensor(t) and t.requires_grad]
        self.replaying = True
        try:
            with torch.enable_grad(), kernel():
                outputs = layer(replay_args)
                outputs = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
                pairs = [(outputs[j], g) for j, g in grads.items()]
                found = torch.autograd.grad([o for o, _ in pairs], [t for _, t in named],
                                            [g for _, g in pairs], allow_unused=True)
        finally:
            self.replaying = False
        out_bad = [f'output {j}' for j, (o, _) in enumerate(pairs) if _non_finite(o)]
        return out_bad + [n for (n, _), g in zip(named, found) if g is not None and _non_finite(g)]

    def _replay_report(self, index):
        if index is None or index not in self.layer_inputs or index not in self.output_grads:
            return ['replay: the inputs or output gradients of this layer were not recorded']
        lines = [f'replay of layer {index} {type(self.layers[index]).__name__} with its recorded inputs:']
        bad_grads = [j for j, g in self.output_grads[index].items() if _non_finite(g)]
        if bad_grads:
            return lines + [f'  the gradient arriving at its outputs {bad_grads} is already non-finite']

        def attempt(kernel, sample=None):
            try:
                bad = self._replay(index, kernel, sample)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return 'out of memory'
            except RuntimeError as error:
                return f'unavailable ({str(error).strip().splitlines()[0][:80]})'
            return f'NaN/inf in {bad[:4]}{" ..." if len(bad) > 4 else ""}' if bad else 'finite'

        for name, kernel in _attention_kernels():
            lines.append(f'  whole batch, {name}: {attempt(kernel)}')
        batch = next(t.shape[0] for t in self.layer_inputs[index] if torch.is_tensor(t) and t.ndim)
        bad_samples = []
        for sample in range(batch):
            result = attempt(nullcontext, sample)
            if result != 'finite':
                name = self.names[sample] if self.names and len(self.names) == batch else '?'
                bad_samples.append(f'{sample} {name}: {result}')
        lines.append(f'  each sample alone, default kernel: {len(bad_samples)} of {batch} non-finite')
        lines += [f'    sample {line}' for line in bad_samples[:16]]
        return lines

    def before_step(self, step):
        self.step = step
        self.layer_inputs.clear()
        self.output_grads.clear()

    def after_step(self, step, loss):
        self.layer_inputs.clear()
        self.output_grads.clear()
        if not torch.isfinite(torch.tensor(loss)):
            self._fail(f'loss is {loss} although no layer output or gradient on this rank was non-finite '
                       '(another rank, or the loss reduction)', 'see the other ranks\' output')
        for name, p in self.parameters:
            if _non_finite(p.data):
                self._fail(f'weight {getattr(p, "original_name", name)} became non-finite in the optimizer step',
                           _describe_tensor(p.data))
