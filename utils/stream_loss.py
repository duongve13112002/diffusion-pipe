"""Residual-stream penalties for MOSS-Image (opt-in with [model.stream_loss.<term>] tables).

A fine-tune of MOSS-Image can concentrate the energy of its residual stream into a few "massive"
channels. Three penalties, added to the flow loss, push against it (the ideas and the defaults come
from KohakuMossTrainer's stream-healing notes; docs/note/stream-healing-implementation-plan.md has
the mapping to this code):

    [model.stream_loss.write]          # the main term
    weight = 5e-3                      # 0 turns the term off
    target = 0.5                       # required: the share the hinge settles on
    anneal = 2000                      # steps the hinge share takes to fall from 1 to target
    start = 0                          # step the fall starts at
    k = 32                             # channels counted as "the top"
    parts = 'both'                     # 'attn', 'ffn' or 'both'
    dilute = false                     # false: only the top channels are shrunk (total detached)

    [model.stream_loss.rms]
    weight = 5e-5
    target = 1.0                       # whole-stream RMS per sample, pulled to this from both sides

    [model.stream_loss.concentration]
    weight = 1e-2
    k = 32
    dilute = true                      # true: the rest may also grow

For one block with output x [batch, tokens, D], channel energy e_c = sum over samples and tokens of
x_c^2, and the top-k share s = (energy of the k largest channels) / (energy of all):

- concentration: s of the block output, mean over the blocks;
- rms: per sample R_i = sqrt(sum of x^2 over the sample's valid tokens / (T_i D)), the term is
  mean_i (R_i / target - 1)^2, mean over the blocks;
- write: s of each sublayer's write (y * gate for the attention and for the FFN, the part the
  sublayer adds to the stream), hinge relu(s - tau(step))^2, mean over the blocks and sublayers;
  tau is 1 before `start`, then falls linearly to `target` over `anneal` steps.

The shares are taken over the micro-batch, as one block sees it. The per-block numbers travel
through the pipeline as one float32 vector (see StreamLayout), so the penalty can be formed on the
last stage, where the loss is.
"""

import math

import torch
import torch.distributed as dist

KEY = 'stream_loss'
TERMS = {
    'concentration': {'weight', 'k', 'dilute'},
    'rms': {'weight', 'target'},
    'write': {'weight', 'k', 'parts', 'target', 'anneal', 'start', 'dilute'},
}
DEFAULTS = {
    'concentration': {'k': 32, 'dilute': True},
    'rms': {'target': 1.0},
    'write': {'k': 32, 'parts': 'both', 'anneal': 0, 'start': 0, 'dilute': False},
}
WRITE_PARTS = {'attn': ('attn',), 'ffn': ('ffn',), 'both': ('attn', 'ffn')}
# Slot groups of the vector a block adds to, each num_blocks long.
CONCENTRATION, RMS_TERM, RMS_VALUE, SHARE_ATTN, SHARE_FFN = range(5)
GROUPS = 5
SHARE_SLOT = {'attn': SHARE_ATTN, 'ffn': SHARE_FFN}


def _bad(where, name, value, wanted):
    return ValueError(f'{where}.{name} must be {wanted}, got {value!r}')


def _real(term, name, where, low, strict=False):
    value = term[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or value < low or (strict and value == low):
        raise _bad(where, name, value, f"a number {'>' if strict else '>='} {low}")


def _whole(term, name, where, low):
    value = term[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise _bad(where, name, value, f'a whole number >= {low}')


def parse_stream_loss(model_config):
    """The active terms of model_config['stream_loss'] with their defaults filled in
    ({term: fields}), or None when no term has a weight above zero.

    Refused: a non-table, an unknown term or field, a missing weight, a write term without a
    target, and values out of range.
    """
    if KEY not in model_config:
        return None
    table = model_config[KEY]
    if not isinstance(table, dict):
        raise ValueError(f'model.{KEY} must be a table of terms, got {table!r}')
    unknown = sorted(set(table) - set(TERMS))
    if unknown:
        raise ValueError(f'model.{KEY} has unknown terms {unknown}; valid: {sorted(TERMS)}. '
                         '(KohakuMoss\'s peak term guards fp16 headroom and does not apply to bf16 / fp32 training.)')
    spec = {}
    for name, term in table.items():
        where = f'model.{KEY}.{name}'
        if not isinstance(term, dict):
            raise ValueError(f'{where} must be a table, got {term!r}')
        unknown = sorted(set(term) - TERMS[name])
        if unknown:
            raise ValueError(f'{where} has unknown fields {unknown}; valid: {sorted(TERMS[name])}')
        if 'weight' not in term:
            raise ValueError(f'{where} needs weight (0 turns the term off)')
        term = {**DEFAULTS[name], **term}
        _real(term, 'weight', where, 0.0)
        if name in ('concentration', 'write'):
            _whole(term, 'k', where, 1)
            if not isinstance(term['dilute'], bool):
                raise _bad(where, 'dilute', term['dilute'], 'true or false')
        if name == 'rms':
            _real(term, 'target', where, 0.0, strict=True)
        if name == 'write':
            if 'target' not in table[name]:
                raise ValueError(f'{where} needs target (the share the hinge settles on, in (0, 1])')
            _real(term, 'target', where, 0.0, strict=True)
            if term['target'] > 1:
                raise _bad(where, 'target', term['target'], 'in (0, 1]')
            if term['parts'] not in WRITE_PARTS:
                raise _bad(where, 'parts', term['parts'], f'one of {sorted(WRITE_PARTS)}')
            _whole(term, 'anneal', where, 0)
            _whole(term, 'start', where, 0)
        if term['weight'] > 0:
            spec[name] = term
    return spec or None


def check_width(spec, dim):
    """The top-k terms cannot count more channels than the model has."""
    for name, term in spec.items():
        if 'k' in term and term['k'] > dim:
            raise ValueError(f'model.{KEY}.{name}.k = {term["k"]} exceeds the model width {dim}')


def channel_energy(h):
    """Each channel's energy of h [..., D], summed over every leading position: fp32 [D]. h is read
    in its own dtype (no fp32 copy of it); the reduction is in fp32."""
    return torch.linalg.vector_norm(h.reshape(-1, h.shape[-1]), dim=0, dtype=torch.float32).square()


def top_shares(energy, k, dilute=True):
    """Per row of the channel energies [L, D], the share held by its k largest channels: fp32 [L].
    dilute = False detaches the total, so the gradient only shrinks the top channels."""
    total = energy.sum(-1).clamp_min(torch.finfo(torch.float32).tiny)
    return energy.topk(k, dim=-1).values.sum(-1) / (total if dilute else total.detach())


def write_target(term, step):
    """The write term's hinge share at `step`: 1 before `start`, then linear to `target` over
    `anneal` steps (at once for anneal 0)."""
    t = step - term['start']
    if t < 0:
        return 1.0
    fraction = 1.0 if term['anneal'] <= 0 else min(t / term['anneal'], 1.0)
    return 1.0 - (1.0 - term['target']) * fraction


class StreamLayout:
    """The vector the blocks fill in: GROUPS groups of num_blocks float32 slots, block i adding its
    numbers to slot group * num_blocks + i. A group the spec does not use stays zero."""

    def __init__(self, spec, num_blocks):
        self.spec = spec
        self.num_blocks = num_blocks
        self.size = GROUPS * num_blocks
        self.write_parts = WRITE_PARTS[spec['write']['parts']] if 'write' in spec else ()

    def new(self, like):
        return like.new_zeros(self.size, dtype=torch.float32)

    def block_values(self, block, hidden, tokens, write_energy):
        """The GROUPS numbers of one block, as a [GROUPS] fp32 tensor.

        hidden [batch, total, D]: the block's output, rows past a sample's tokens are zero.
        tokens [batch]: each sample's valid token count. write_energy {part: channel energy [D]}
        of the sublayer writes, summed over the micro-batch's length groups.
        """
        spec = self.spec
        zero = hidden.new_zeros((), dtype=torch.float32)
        values = [zero] * GROUPS
        if 'concentration' in spec:
            term = spec['concentration']
            values[CONCENTRATION] = top_shares(channel_energy(hidden)[None], term['k'], term['dilute'])[0]
        if 'rms' in spec:
            squares = torch.linalg.vector_norm(hidden, dim=(1, 2), dtype=torch.float32).square()
            rms = (squares / (tokens.float() * hidden.shape[-1])).clamp_min(1e-30).sqrt()
            values[RMS_TERM] = (rms / spec['rms']['target'] - 1.0).square().mean()
            values[RMS_VALUE] = rms.detach().mean()
        if 'write' in spec:
            term = spec['write']
            for part in self.write_parts:
                if part not in write_energy:
                    raise RuntimeError(f'stream_loss.write: block {block} produced no {part} write')
                values[SHARE_SLOT[part]] = top_shares(write_energy[part][None], term['k'], term['dilute'])[0]
        return torch.stack(values)

    def add(self, stream, block, values):
        """`stream` with block `block`'s numbers added; a new tensor, so autograd and
        checkpointing see an ordinary function of its inputs."""
        index = torch.arange(GROUPS, device=stream.device) * self.num_blocks + block
        return stream.index_add(0, index, values.to(stream.dtype))

    def penalty(self, stream, step):
        """(the weighted penalty sum, {term: unweighted value}, {metric: value}) from the finished
        vector; the first is attached to the graph, the others are detached."""
        v = stream.float().reshape(GROUPS, self.num_blocks)
        terms, metrics = {}, {}
        spec = self.spec
        if 'concentration' in spec:
            terms['concentration'] = v[CONCENTRATION].mean()
        if 'rms' in spec:
            terms['rms'] = v[RMS_TERM].mean()
            metrics['stream_rms'] = v[RMS_VALUE].detach().mean()
        if 'write' in spec:
            term = spec['write']
            shares = torch.cat([v[SHARE_SLOT[part]] for part in self.write_parts])
            tau = write_target(term, step)
            terms['write'] = (shares - tau).clamp_min(0.0).square().mean()
            for part in self.write_parts:
                metrics[f'write_share_{part}'] = v[SHARE_SLOT[part]].detach().mean()
            metrics['write_target'] = shares.new_tensor(tau).detach()
        total = sum(spec[name]['weight'] * value for name, value in terms.items())
        return total, {name: value.detach() for name, value in terms.items()}, metrics

    def metric_names(self):
        """The logged scalars in a fixed order every rank agrees on."""
        names = ['flow_loss', 'stream_penalty']
        names += [f'penalty_{name}' for name in self.spec]
        if 'rms' in self.spec:
            names.append('stream_rms')
        if 'write' in self.spec:
            names += [f'write_share_{part}' for part in self.write_parts] + ['write_target']
        return names


class StreamMetrics:
    """Sums of the per-micro-batch numbers the last stage's loss function sees, averaged over the
    step's micro-batches and data-parallel ranks when it is logged. Every rank joins the
    reduction (the ranks of the other stages with zeros), like Iris's REPA loss."""

    def __init__(self, names):
        self.names = list(names)
        self.sums = None

    def record(self, values):
        """Add one micro-batch: `values` maps every name to a number or a scalar tensor. Kept on the
        values' device, so recording does not wait for the GPU."""
        row = torch.stack([torch.as_tensor(values[name]).detach().double().reshape(()) for name in self.names])
        row = torch.cat([row, row.new_ones(1)])
        self.sums = row if self.sums is None else self.sums + row.to(self.sums.device)

    def reduce(self):
        """{'train/<name>': mean} for the step so far, and start the next one. A collective when
        torch.distributed is up: every rank calls it on every step."""
        totals, self.sums = self.sums, None
        if dist.is_available() and dist.is_initialized():
            device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'
            totals = (torch.zeros(len(self.names) + 1, dtype=torch.float64, device=device) if totals is None
                      else totals.to(device))
            dist.all_reduce(totals)
        if totals is None or totals[-1] == 0:
            return {}
        count = float(totals[-1])
        return {f'train/{name}': float(totals[i]) / count for i, name in enumerate(self.names)}
