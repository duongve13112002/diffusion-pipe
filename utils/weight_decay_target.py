"""Weight decay toward the weights a full fine-tune started from, and toward 1 for chosen tensors.

The decay of AdamW is `p -= lr * wd * p`: every step pulls each weight toward 0, which for a
pretrained model is away from what it learned. With `weight_decay_target = 'start'` the pull is
toward the weights the run started from instead,

    p -= lr * wd * (p - a)        a = the launch weights,

(KohakuMossTrainer's WD_TARGET = "start"). `weight_decay_to_one = '<regex>'` names parameters, by the
name the saver uses, that decay toward 1 instead (`a = 1`): the norm gains of the last block, which
the MOSS-Image recipe pulls to 1 rather than 0 (its WD_RULES = "moss").

The optimizer's own decay is switched off for the tensors this takes over, so nothing decays twice,
and the decay runs after the optimizer step with the same learning rate the step used. Its size,
lr * wd * (p - a), is far below one bf16 step for bf16 weights, so the update is computed in fp32
and rounded back at random, up or down with the probabilities that keep its expectation exact (the
rounding utils/outlier_decay.py uses), from a generator seeded by the step alone: the data-parallel
replicas of a stage, and the nodes of a multi-node run, draw the same numbers and stay identical.

The launch weights are kept in the parameter's own dtype on its device: 2 bytes per bf16 parameter,
4 per fp32 one. They are taken when the decay is built, which train.py does before a resume replaces
the weights, so they are the weights of the model file named in the config.

Full fine-tune only, with an optimizer whose decay is exactly `p -= lr * wd * p` (checked below).
"""

import re

import torch

from utils.outlier_decay import SUPPORTED_DTYPES, round_to_bfloat16

TARGET = 'weight_decay_target'
TO_ONE = 'weight_decay_to_one'
TARGETS = ('zero', 'start')
# Optimizers whose weight decay is decoupled and scaled by the learning rate, `p *= 1 - lr * wd`.
# genericoptim adds `-group_lr * wd * p` to its update, Muon groups included (test_weight_decay_target.py
# checks it against the real optimizer).
DECOUPLED_OPTIMIZERS = ('adamw', 'adamw_optimi', 'adamw8bit', 'genericoptim')
_SALT = 7919
SAVED_DECAY = 'weight_decay_before_target'


def restore_optimizer_decay(optimizer):
    """Give the optimizer groups back the weight decay a run with weight_decay_target had switched off.

    A checkpoint saved by such a run holds those groups with a decay of 0 and the decay they had under
    SAVED_DECAY. Call it after loading a checkpoint when the feature is off, or the resumed run would train
    those tensors without any decay. Returns the number of groups restored."""
    restored = 0
    for group in optimizer.param_groups:
        if SAVED_DECAY in group:
            group['weight_decay'] = group.pop(SAVED_DECAY)
            restored += 1
    return restored


def weight_decay_target_active(config):
    return config.get(TARGET, 'zero') != 'zero' or bool(config.get(TO_ONE, ''))


def apply_weight_decay_target_config_defaults(config):
    """Validate the two top-level keys and write their defaults into config."""
    target = config.setdefault(TARGET, 'zero')
    if target not in TARGETS:
        raise ValueError(f'{TARGET} must be one of {TARGETS}, got {target!r}')
    pattern = config.setdefault(TO_ONE, '')
    if not isinstance(pattern, str):
        raise ValueError(f'{TO_ONE} must be a regular expression string, got {pattern!r}')
    try:
        re.compile(pattern)
    except re.error as error:
        raise ValueError(f'{TO_ONE} must be a regular expression, got {pattern!r}: {error}') from error
    if not weight_decay_target_active(config):
        return
    if 'adapter' in config:
        raise ValueError(f'{TARGET} and {TO_ONE} act on the weights of a full fine-tune; remove the [adapter] '
                         'table, or set weight_decay_target = \'zero\' and weight_decay_to_one = \'\'')
    if config.get('blocks_to_swap', 0):
        raise ValueError(f'{TARGET} and {TO_ONE} keep a copy of the launch weights next to each weight, which block '
                         'swapping moves between CPU and GPU without it; remove blocks_to_swap')
    optimizer = config.get('optimizer')
    if not isinstance(optimizer, dict):
        raise ValueError(f'{TARGET} needs an [optimizer] table')
    if optimizer.get('gradient_release', False):
        raise ValueError(f'{TARGET} acts after the optimizer step, which gradient_release does not have')
    kind = str(optimizer.get('type', '')).lower()
    if kind not in DECOUPLED_OPTIMIZERS:
        raise ValueError(f'{TARGET} needs an optimizer whose weight decay is exactly lr * wd * p; '
                         f'{optimizer.get("type")!r} is not known to be. Supported: {", ".join(DECOUPLED_OPTIMIZERS)}')
    if optimizer.get('decouple_lr', False):
        raise ValueError(f'{TARGET} assumes the optimizer decays by lr * weight_decay * p; decouple_lr = true '
                         'changes the decay to depend on max_lr. Remove decouple_lr.')
    decay = optimizer.get('weight_decay', 0.0)
    if isinstance(decay, bool) or not isinstance(decay, (int, float)) or decay <= 0:
        raise ValueError(f'{TARGET} and {TO_ONE} need [optimizer] weight_decay above 0, got {decay!r}')


class WeightDecayTarget:
    """The decay over the tensors of this process (one pipeline stage's) it takes over.

    `params`, `names`, `anchors` (None = toward 1) and `decays` are parallel lists. `claimed` holds
    the ids of the parameters whose optimizer group had a decay of its own, which is switched off.
    """

    def __init__(self, optimizer, names, params, anchors, decays, claimed, wants_to_one=False):
        self.wants_to_one = wants_to_one
        self.optimizer = optimizer
        self.names = names
        self.params = params
        self.anchors = anchors
        self.decays = decays
        self.claimed = claimed
        self._lrs = [0.0] * len(params)
        self._generators = {}

    @classmethod
    @torch.no_grad()
    def build(cls, named_parameters, optimizer, config):
        target = config[TARGET]
        pattern = re.compile(config[TO_ONE]) if config[TO_ONE] else None
        decay = float(config['optimizer']['weight_decay'])
        group_of = {id(p): group for group in optimizer.param_groups for p in group['params']}
        names, params, anchors, decays, claimed, seen = [], [], [], [], set(), set()
        for name, p in named_parameters:
            group = group_of.get(id(p))
            if group is None or not p.requires_grad or id(p) in seen:
                continue
            key = getattr(p, 'original_name', name)
            to_one = pattern is not None and pattern.search(key) is not None
            decayed = group.get('weight_decay', 0.0) > 0
            if to_one and decayed:
                raise ValueError(f'{TO_ONE} matches {key}, which the optimizer already decays toward 0: '
                                 'train.py gives weight decay to the tensors with two or more dimensions only')
            if not to_one and not (target == 'start' and decayed):
                continue
            if p.dtype not in SUPPORTED_DTYPES:
                raise ValueError(f'{TARGET} handles float32 and bfloat16 weights; {key} is {p.dtype}')
            seen.add(id(p))
            names.append(key)
            params.append(p)
            anchors.append(None if to_one else p.detach().clone())
            decays.append(decay)
            if decayed:
                claimed.update(id(q) for q in group['params'])
        result = cls(optimizer, names, params, anchors, decays, claimed, wants_to_one=pattern is not None)
        result.claim_groups()
        return result

    def claim_groups(self):
        """Switch off the optimizer's own decay on the groups taken over. Call it again after a
        checkpoint is loaded: loading the optimizer state replaces the groups with the saved ones,
        decay included, so the groups are found again by the parameters they hold."""
        for group in self.optimizer.param_groups:
            if any(id(p) in self.claimed for p in group['params']):
                if group.get('weight_decay', 0.0) > 0:
                    # Kept in the group, so a checkpoint saved now still knows the decay the optimizer had
                    # (restore_optimizer_decay gives it back when a later run does not use this feature).
                    group.setdefault(SAVED_DECAY, group['weight_decay'])
                group['weight_decay'] = 0.0

    def __len__(self):
        return len(self.params)

    def describe(self):
        if not self.params:
            return f'{TARGET}: no trained parameter of this process is taken over'
        elements = sum(p.numel() for p in self.params)
        to_one = sum(a is None for a in self.anchors)
        return (f'{TARGET}: {len(self.params)} tensors ({elements / 1e6:.1f}M elements), '
                f'{len(self.params) - to_one} toward the launch weights, {to_one} toward 1')

    def require_match(self):
        """Raise unless some process holds a tensor to act on, and, when weight_decay_to_one is set,
        unless some process holds a tensor it matches (a regex that matches nothing would otherwise
        change nothing silently). A collective when torch.distributed is up: every rank calls it."""
        total = len(self)
        to_one = sum(a is None for a in self.anchors)
        too_strong = int(self.peak_pull() > 1.0)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'
            count = torch.tensor([total, to_one, too_strong], dtype=torch.int64, device=device)
            torch.distributed.all_reduce(count)
            total, to_one, too_strong = int(count[0]), int(count[1]), int(count[2])
        if too_strong:
            raise ValueError(f'{TARGET}: lr * weight_decay is above 1 for some tensor at the highest learning rate of '
                             'the schedule; the pull would pass its target. Lower the learning rate or weight_decay '
                             '(checked at startup on every rank so that no stage is left waiting).')
        if total == 0:
            raise ValueError(f'{TARGET}: no trained parameter of any pipeline stage is taken over '
                             f'(weight_decay_target must be \'start\' or {TO_ONE} must match a parameter)')
        if self.wants_to_one and to_one == 0:
            raise ValueError(f'{TO_ONE} matches no trained parameter of any pipeline stage; '
                             'check the regular expression against the parameter names as saved')

    def peak_pull(self):
        """The largest lr * weight_decay over the tensors taken over, at the highest learning rate the
        schedule reaches (a group's initial_lr, set by the scheduler)."""
        peak = {id(p): max(float(group.get('initial_lr', group['lr'])), float(group['lr']))
                for group in self.optimizer.param_groups for p in group['params']}
        return max((peak.get(id(p), 0.0) * decay for p, decay in zip(self.params, self.decays)), default=0.0)

    def before_step(self):
        """Record each tensor's learning rate for the step about to run: DeepSpeed advances the
        scheduler inside train_batch, so after it the groups hold the next step's."""
        lr = {id(p): float(group['lr']) for group in self.optimizer.param_groups for p in group['params']}
        self._lrs = [lr.get(id(p), 0.0) for p in self.params]

    def _generator(self, device, step):
        generator = self._generators.get(device)
        if generator is None:
            generator = self._generators[device] = torch.Generator(device=device)
        generator.manual_seed(int(step) * 1_000_003 + _SALT)
        return generator

    @torch.no_grad()
    def apply(self, step):
        """The decay of optimizer step `step` (1-based, as train.py counts), at the learning rates
        before_step recorded."""
        generators = {}
        for p, anchor, decay, lr in zip(self.params, self.anchors, self.decays, self._lrs):
            k = lr * decay
            if k <= 0.0:
                continue
            if k > 1.0:
                raise ValueError(f'{TARGET}: lr * weight_decay is {k:g} at step {step}; above 1 the pull passes '
                                 'its target and flips the sign of the distance. Lower the learning rate or weight_decay.')
            if p.device not in generators:
                generators[p.device] = self._generator(p.device, step)
            w = p.detach().float()
            new = w - k * (w - (1.0 if anchor is None else anchor.float()))
            p.copy_(round_to_bfloat16(new, generators[p.device]) if p.dtype == torch.bfloat16 else new)
