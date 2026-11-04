"""Outlier decay for full fine-tunes (opt-in with an [outlier_decay] table).

    [outlier_decay]
    rate = 6.0          # lambda: an excess shrinks with a time constant of 1 / (lambda * lr) steps
    cap = 16.0          # the threshold, in units of each tensor's RMS at the start of the run
    cap_start = 64.0    # 0 = cap; else the cap falls geometrically from here to `cap`
    cap_anneal = 10000  # steps that fall takes
    warmup = 1000       # steps over which rate ramps linearly from 0
    targets = ''        # regex over the parameter names; empty = every 2-D trained weight
    exclude = ''        # regex of names to leave out
    include_1d = false  # also gains and biases
    seed = 0            # seeds the rounding of the update, see below

After each optimizer step, every targeted weight element is pulled towards zero by its excess over
the cap:

    p -= lr * rate(t) * sign(p) * relu(|p| - cap(t) * r0)

`r0` is the tensor's RMS when the run started. The decay is decoupled from the optimizer, like
AdamW's own, so the move is proportional to the excess instead of the lr-sized step Adam would
give an outlier whatever its gradient. With cap = 0 it would be plain weight decay, which shrinks
the whole model and leaves the concentration where it was; the threshold is what makes this an
outlier decay. It acts on the weights a full fine-tune inherits, which a stream penalty alone
does not reach (docs/note/kohakumoss-vs-diffusion-pipe-moss-training.md).

bf16 weights. lr * rate * excess is about 1e-4 of the excess, below one bf16 step, so rounding to
nearest would never move anything. The update is computed in fp32 and rounded back at random, up
or down with the probabilities that keep its expectation exact; an element the update leaves
alone is exactly representable in bf16 and is returned unchanged. The noise comes from a generator
seeded with (seed, step), the same on every data-parallel replica of a stage, so the replicas keep
identical weights. The optimizer's own Kahan buffer for bf16 weights is not adjusted: the decay
changes the stored weight, and the error that adds is unbiased and below one bf16 step per
element.
"""

import math
import re

import torch
import torch.distributed as dist

KEY = 'outlier_decay'
DEFAULTS = {
    'rate': None,
    'cap': 16.0,
    'cap_start': 0.0,
    'cap_anneal': 0,
    'warmup': 0,
    'targets': '',
    'exclude': '',
    'include_1d': False,
    'seed': 0,
}
SUPPORTED_DTYPES = (torch.float32, torch.bfloat16)
# Optimizers that step the weights in place at the learning rate of their parameter group, which is
# what the decay's lr * rate assumes. Left out: Prodigy and the pytorch_optimizer family (they step
# with d * lr or other internal scales), Automagic (a learning rate per parameter), GenericOptim
# with automagic on, and the torchao offload optimizer (not checked to leave an in-place edit alone).
STEP_LR_OPTIMIZERS = ('adamw', 'adamw8bit', 'adamw_optimi', 'stableadamw', 'sgd', 'adamw8bitkahan', 'genericoptim')


def _number(table, name, low, strict=False):
    value = table[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{KEY}.{name} must be a number, got {value!r}')
    if value < low or (strict and value == low):
        raise ValueError(f"{KEY}.{name} must be {'>' if strict else '>='} {low}, got {value!r}")


def _whole(table, name):
    value = table[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{KEY}.{name} must be a whole number >= 0, got {value!r}')


def apply_outlier_decay_config_defaults(config):
    """Validate config['outlier_decay'] and fill in its defaults; a no-op when the table is absent.

    Refused: unknown fields, a missing rate (a table with no rate would silently do nothing),
    values out of range, a bad regex, and a run with an [adapter], whose base weights never change.
    """
    if KEY not in config:
        return
    table = config[KEY]
    if not isinstance(table, dict):
        raise ValueError(f'{KEY} must be a table, got {table!r}')
    unknown = sorted(set(table) - set(DEFAULTS))
    if unknown:
        raise ValueError(f'{KEY} has unknown fields {unknown}; valid: {sorted(DEFAULTS)}')
    if 'rate' not in table:
        raise ValueError(f'{KEY} needs rate (the decay strength lambda; 0 turns it off)')
    table = {**DEFAULTS, **table}
    _number(table, 'rate', 0.0)
    _number(table, 'cap', 0.0, strict=True)
    _number(table, 'cap_start', 0.0)
    if 0 < table['cap_start'] < table['cap']:
        raise ValueError(f"{KEY}.cap_start must be 0 (= cap) or >= cap, got {table['cap_start']!r}")
    for name in ('cap_anneal', 'warmup', 'seed'):
        _whole(table, name)
    if (table['cap_start'] > 0) != (table['cap_anneal'] > 0):
        raise ValueError(f"{KEY}.cap_start and {KEY}.cap_anneal go together: the cap falls from cap_start to cap over "
                         f"cap_anneal steps, so setting only one changes nothing (got cap_start={table['cap_start']!r}, "
                         f"cap_anneal={table['cap_anneal']!r})")
    for name in ('targets', 'exclude'):
        try:
            re.compile(table[name])
        except (re.error, TypeError) as error:
            raise ValueError(f'{KEY}.{name} must be a regex, got {table[name]!r}: {error}') from error
    if not isinstance(table['include_1d'], bool):
        raise ValueError(f"{KEY}.include_1d must be true or false, got {table['include_1d']!r}")
    if 'adapter' in config and table['rate'] > 0:
        raise ValueError(f'{KEY} decays the base weights, which an adapter run does not train: '
                         'remove the [adapter] table for a full fine-tune, or remove [outlier_decay]')
    optimizer = config.get('optimizer')
    if table['rate'] > 0 and isinstance(optimizer, dict) and optimizer.get('type'):
        kind = str(optimizer['type']).lower()
        if kind not in STEP_LR_OPTIMIZERS or (kind == 'genericoptim' and optimizer.get('automagic', False)):
            raise ValueError(f"{KEY} needs an optimizer that steps at its group's learning rate; "
                             f"{optimizer['type']!r} {'with automagic ' if kind == 'genericoptim' else ''}is not one. "
                             f'Supported: {", ".join(STEP_LR_OPTIMIZERS)} (genericoptim without automagic)')
    config[KEY] = table


def cap_at(spec, t):
    """The cap at optimizer step `t` (0-based): cap_start to cap, geometrically, over cap_anneal steps."""
    cap, start, span = float(spec['cap']), float(spec['cap_start']), int(spec['cap_anneal'])
    if start <= 0 or span <= 0 or t >= span:
        return cap
    return start * (cap / start) ** (t / span)


def rate_at(spec, t):
    """The rate at optimizer step `t` (0-based), ramped linearly from 0 over `warmup` steps."""
    rate, warmup = float(spec['rate']), int(spec['warmup'])
    return rate if warmup <= 0 else rate * min((t + 1) / warmup, 1.0)


def targeted(spec, name, param):
    """Whether the decay acts on parameter `name`: 2-D (any rank with include_1d), matching
    `targets` (all when empty) and not matching `exclude`."""
    if param.ndim < 2 and not spec['include_1d']:
        return False
    if spec['targets'] and not re.search(spec['targets'], name):
        return False
    return not (spec['exclude'] and re.search(spec['exclude'], name))


def round_to_bfloat16(x, generator):
    """fp32 -> bf16, rounding up or down at random with the probabilities that make the result
    equal x in expectation. bf16 is the top 16 bits of fp32: add uniform noise below them and
    truncate. A value that is exactly representable in bf16 comes back unchanged."""
    bits = x.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, device=bits.device, generator=generator)
    return ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)


class OutlierDecay:
    """The decay over the targeted trained parameters of this process (one pipeline stage's).

    `params` and `names` are parallel lists, `r0` each tensor's RMS at the start of the run.
    `contributes`: whether this process reports the metrics (one data-parallel replica per stage,
    since the replicas hold the same weights).
    """

    def __init__(self, spec, optimizer, names, params, r0, contributes=True):
        self.spec = spec
        self.optimizer = optimizer
        self.names = names
        self.params = params
        self.r0 = r0
        self.contributes = contributes
        self._lrs = [0.0] * len(params)
        self._generators = {}
        self._over = None
        self._largest = 1.0

    @classmethod
    @torch.no_grad()
    def build(cls, named_parameters, optimizer, spec, contributes=True):
        """The decay for the trainable parameters among `named_parameters` that `optimizer` steps
        and `spec` targets, with their RMS taken from the weights as they are now: call it before
        a resume replaces them. Names are the `original_name` the saver and the EMA use."""
        stepped = {id(p) for group in optimizer.param_groups for p in group['params']}
        names, params, seen = [], [], set()
        for name, p in named_parameters:
            if not p.requires_grad or id(p) in seen or id(p) not in stepped:
                continue
            key = getattr(p, 'original_name', name)
            if not targeted(spec, key, p):
                continue
            if p.dtype not in SUPPORTED_DTYPES:
                raise ValueError(f'{KEY} handles float32 and bfloat16 weights; {key} is {p.dtype}')
            seen.add(id(p))
            names.append(key)
            params.append(p)
        rms = []
        for p in params:
            rms.append(torch.linalg.vector_norm(p.detach(), dtype=torch.float32) / math.sqrt(p.numel()))
        r0 = torch.stack(rms).tolist() if rms else []
        # An all-zero tensor has no scale to measure outliers against.
        keep = [i for i, value in enumerate(r0) if value > 0]
        return cls(spec, optimizer, [names[i] for i in keep], [params[i] for i in keep],
                   [r0[i] for i in keep], contributes)

    def __len__(self):
        return len(self.params)

    def require_match(self):
        """Raise unless some process holds a parameter the decay acts on. A regex that matches
        nothing would otherwise leave a run that logs outlier_over = 0 for ever. Every rank must
        call it: it is a collective when torch.distributed is up."""
        total = len(self)
        too_strong = int(self.peak_pull() > 1.0)
        if dist.is_available() and dist.is_initialized():
            device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'
            count = torch.tensor([total, too_strong], dtype=torch.int64, device=device)
            dist.all_reduce(count)
            total, too_strong = int(count[0]), int(count[1])
        if too_strong:
            raise ValueError(f'{KEY}: lr * rate is above 1 for some tensor at the highest learning rate of the '
                             'schedule; the update would pass the cap. Lower rate or the learning rate (checked at '
                             'startup on every rank so that no stage is left waiting).')
        if total == 0:
            raise ValueError(f"{KEY}: no trained parameter of any pipeline stage matches targets "
                             f"{self.spec['targets']!r} / exclude {self.spec['exclude']!r} "
                             f"(include_1d = {self.spec['include_1d']})")

    def describe(self):
        spec = self.spec
        if not self.params:
            return f'{KEY}: no trained parameter of this process matches the targets'
        elements = sum(p.numel() for p in self.params)
        return (f'{KEY}: {len(self.params)} tensors ({elements / 1e6:.1f}M elements), rate {spec["rate"]:g}, '
                f'cap {spec["cap_start"] or spec["cap"]:g} -> {spec["cap"]:g} over {spec["cap_anneal"]} steps, '
                f'warmup {spec["warmup"]} steps')

    def peak_pull(self):
        """The largest lr * rate over the parameters, at the highest learning rate the schedule reaches."""
        peak = {id(p): max(float(group.get('initial_lr', group['lr'])), float(group['lr']))
                for group in self.optimizer.param_groups for p in group['params']}
        return max((peak.get(id(p), 0.0) for p in self.params), default=0.0) * float(self.spec['rate'])

    def reset_metrics(self):
        """Forget the last step's numbers: after a step that was dropped there are none, and the logged
        ones would repeat the step before."""
        self._over, self._largest = None, 1.0

    def before_step(self):
        """Record each parameter's learning rate for the step about to run: DeepSpeed advances the
        scheduler inside train_batch, so after it the groups hold the next step's."""
        lr = {id(p): float(group['lr']) for group in self.optimizer.param_groups for p in group['params']}
        self._lrs = [lr.get(id(p), 0.0) for p in self.params]

    def _generator(self, device, step):
        generator = self._generators.get(device)
        if generator is None:
            generator = self._generators[device] = torch.Generator(device=device)
        generator.manual_seed(int(self.spec['seed']) * 1_000_003 + int(step))
        return generator

    @torch.no_grad()
    def apply(self, step):
        """The decay of optimizer step `step` (1-based, as train.py counts), at the learning rates
        before_step recorded. One host read per call: the largest |p| of every active tensor."""
        t = int(step) - 1
        cap, rate = cap_at(self.spec, t), rate_at(self.spec, t)
        self._over, self._largest = None, 1.0
        active = [i for i, lr in enumerate(self._lrs) if lr * rate > 0]
        if not active:
            return
        strongest = max(self._lrs[i] for i in active) * rate
        if strongest > 1.0:
            raise ValueError(f'{KEY}: lr * rate is {strongest:g} at step {step}; above 1 the update passes '
                             'the cap and flips the sign of the weights it acts on. Lower rate.')
        extremes = torch.stack([torch.stack(torch.aminmax(self.params[i].detach())) for i in active])
        largest = extremes.float().abs().amax(1).tolist()
        counts = []
        generators = {}
        for i, biggest in zip(active, largest):
            limit = cap * self.r0[i]
            if biggest <= limit:
                continue
            p = self.params[i]
            self._largest = max(self._largest, biggest / limit)
            if p.device not in generators:
                generators[p.device] = self._generator(p.device, step)
            counts.append(self._shrink(p, limit, self._lrs[i] * rate, generators[p.device]))
        if counts:
            self._over = torch.stack(counts).sum()

    @staticmethod
    def _shrink(p, limit, k, generator):
        """Decay one tensor in place; the number of elements that were over the limit, on its device."""
        w = p.detach().float()
        excess = (w.abs() - limit).clamp_min_(0.0)
        count = torch.count_nonzero(excess)
        new = w - excess.copysign_(w).mul_(k)
        p.copy_(round_to_bfloat16(new, generator) if p.dtype == torch.bfloat16 else new)
        return count

    def log_scalars(self):
        """train/outlier_over (elements above the cap, over every stage) and train/outlier_max (the
        largest |p| / (cap * r0), at least 1) of the last step. A collective when torch.distributed
        is up: call it on every rank on the same steps."""
        over = 0.0 if self._over is None else float(self._over)
        values = [over if self.contributes else 0.0, self._largest if self.contributes else 1.0]
        if dist.is_available() and dist.is_initialized():
            device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'
            totals = torch.tensor([values[0]], dtype=torch.float64, device=device)
            largest = torch.tensor([values[1]], dtype=torch.float64, device=device)
            dist.all_reduce(totals)
            dist.all_reduce(largest, op=dist.ReduceOp.MAX)
            values = [float(totals[0]), float(largest[0])]
        return {'train/outlier_over': values[0], 'train/outlier_max': values[1]}
