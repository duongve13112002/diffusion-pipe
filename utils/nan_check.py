"""train.py --nan_check: stop at the first non-finite value of a real training run and say where.

Every layer output, every parameter gradient, the loss and the weights after each optimizer step
are checked, on every rank. The first non-finite one is reported with the step, the rank, the
layer or parameter, and the images of the batch, and training stops (an exception on one rank
makes the deepspeed launcher stop the others). Each check reads a value back from the GPU, so this
slows training down: it is for finding a NaN, not for leaving on.
"""

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
    specs = inputs.get('image_spec')
    if specs is not None:
        names = [str(Path(spec[1]).name) if isinstance(spec, (list, tuple)) else str(spec) for spec in specs]
        lines.append(f'images {names}')
    return lines


class NanCheck:
    def __init__(self, pipeline_model, model, rank):
        self.rank = rank
        self.step = 0
        self.batches = []
        self.handles = []
        for index, layer in enumerate(pipeline_model.forward_funcs):
            if isinstance(layer, nn.Module):
                self.handles.append(layer.register_forward_hook(partial(self._after_layer, index)))
        for name, p in pipeline_model.named_parameters():
            if p.requires_grad:
                self.handles.append(p.register_post_accumulate_grad_hook(partial(self._after_grad, name)))
        self.parameters = [(name, p) for name, p in pipeline_model.named_parameters()]
        original = model.prepare_inputs
        tokenizer = getattr(getattr(model, 'text_encoder', None), 'tokenizer', None)

        def prepare_inputs(inputs, *args, **kwargs):
            # Kept per pulled batch: with gradient accumulation a step consumes several.
            self.batches = (self.batches + [describe_batch(inputs, tokenizer)])[-4:]
            return original(inputs, *args, **kwargs)
        model.prepare_inputs = prepare_inputs

    def _fail(self, what, detail):
        lines = [f'[nan_check] rank {self.rank}, step {self.step}: {what}', f'[nan_check]   {detail}']
        for i, batch in enumerate(self.batches[::-1]):
            lines.append(f'[nan_check]   batch {"(latest)" if i == 0 else f"(-{i})"}:')
            lines += [f'[nan_check]     {line}' for line in batch]
        message = '\n'.join(lines)
        print(message, flush=True)
        raise NonFiniteError(message)

    def _after_layer(self, index, module, inputs, output):
        outputs = output if isinstance(output, (tuple, list)) else (output,)
        for j, tensor in enumerate(outputs):
            if _non_finite(tensor):
                inputs_ok = not any(_non_finite(t) for t in (inputs[0] if inputs and isinstance(inputs[0], tuple) else inputs))
                self._fail(f'layer {index} {type(module).__name__} output {j} is non-finite'
                           + (' (its inputs were finite)' if inputs_ok else ' (its inputs already were not)'),
                           _describe_tensor(tensor))

    def _after_grad(self, name, p):
        if _non_finite(p.grad):
            self._fail(f'gradient of {getattr(p, "original_name", name)} is non-finite '
                       '(backward runs last layer first: this is the first parameter it reached)',
                       _describe_tensor(p.grad))

    def before_step(self, step):
        self.step = step

    def after_step(self, step, loss):
        if not torch.isfinite(torch.tensor(loss)):
            self._fail(f'loss is {loss} although no layer output or gradient on this rank was non-finite '
                       '(another rank, or the loss reduction)', 'see the other ranks\' output')
        for name, p in self.parameters:
            if _non_finite(p.data):
                self._fail(f'weight {getattr(p, "original_name", name)} became non-finite in the optimizer step',
                           _describe_tensor(p.data))
