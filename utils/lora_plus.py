"""LoRA+: a larger learning rate for the up-projection (`lora_B`) of every LoRA pair.

In W + B @ A the factor A is random and B starts at zero. Hayou et al. (2024, "LoRA+: Efficient Low
Rank Adaptation of Large Models") show that training both factors at one learning rate leaves B
under-trained for wide layers, and that B should learn faster by a fixed ratio (they use 2 to 16).
`[adapter] lora_plus_ratio = 16` gives every `lora_B` tensor `ratio` times the learning rate its
group has; `lora_A` and everything else keep the group's own. The default 1.0 changes nothing.

Only the learning rate differs. The schedule, warmup, weight decay split and OPLoRA projection all
act per group as they did, and the schedules scale each group relative to its own starting rate, so
the B groups follow the same curve at `ratio` times the height.

PEFT LoRA only (adapter type 'lora'). LoKr has no A/B pair to speak of, and gradient_release builds
one optimizer per parameter and is not supported.
"""

import math

NAME = 'lora_plus_ratio'
_B_MARKER = '.lora_B.'


def lora_plus_active(adapter_config):
    return bool(adapter_config) and adapter_config.get(NAME, 1.0) != 1.0


def apply_lora_plus_config_defaults(config):
    """Validate `lora_plus_ratio` in the [adapter] table of config and write its default (1.0)."""
    adapter_config = config['adapter']
    optimizer_config = config.get('optimizer')
    ratio = adapter_config.setdefault(NAME, 1.0)
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio < 1.0:
        raise ValueError(f'{NAME} must be a number >= 1 (1 turns LoRA+ off; the paper uses 2 to 16), got {ratio!r}')
    if ratio == 1.0:
        return
    if adapter_config.get('type') != 'lora':
        raise ValueError(f"{NAME} is only supported for adapter type 'lora', not '{adapter_config.get('type')}': "
                         'it raises the learning rate of the lora_B factor, which other adapters do not have')
    if isinstance(optimizer_config, dict) and optimizer_config.get('gradient_release', False):
        raise ValueError(f'{NAME} needs one optimizer over the parameter groups; gradient_release builds one per '
                         'parameter. Turn one of them off.')
    if isinstance(optimizer_config, dict):
        kind = str(optimizer_config.get('type', '')).lower()
        if kind == 'prodigy' or (kind == 'genericoptim' and optimizer_config.get('automagic', False)):
            raise ValueError(f"{NAME} needs an optimizer that steps each group at its own learning rate; "
                             f"{optimizer_config.get('type')!r}{' with automagic' if kind == 'genericoptim' else ''} "
                             'adapts its own rates and would ignore the ratio. Use another optimizer, or remove '
                             f'{NAME}.')
        if 'lr' not in optimizer_config:
            raise ValueError(f'{NAME} multiplies the optimizer learning rate, so [optimizer] needs an explicit lr')
    if 'force_constant_lr' in config:
        raise ValueError(f'force_constant_lr sets every group to one learning rate, which would undo {NAME}. '
                         'Remove one of them.')


def is_lora_b(name):
    """Whether a parameter name (`p.original_name`) is the B factor of a PEFT LoRA layer."""
    return _B_MARKER in name


def split_lora_plus_groups(param_groups, ratio, default_lr):
    """Return param_groups with each group's lora_B tensors moved to a group of their own whose
    learning rate is `ratio` times the group's (`default_lr` for a group without one).

    The group keeps its other options (weight decay, name...). A group's B group follows it, named
    '<name>+B' when it has a name. Raises when no tensor is a lora_B: LoRA+ would otherwise change
    nothing without saying so.
    """
    result, moved = [], 0
    for group in param_groups:
        params = group['params']
        rest = [p for p in params if not is_lora_b(getattr(p, 'original_name', ''))]
        b = [p for p in params if is_lora_b(getattr(p, 'original_name', ''))]
        if not b:
            result.append(group)
            continue
        moved += len(b)
        b_group = {key: value for key, value in group.items() if key != 'params'}
        b_group['params'] = b
        b_group['lr'] = float(group.get('lr', default_lr)) * ratio
        if group.get('name') is not None:
            b_group['name'] = f"{group['name']}+B"
        if rest:
            kept = dict(group)
            kept['params'] = rest
            result.append(kept)
        result.append(b_group)
    if not moved:
        raise ValueError(f'{NAME} = {ratio:g} found no lora_B tensor among the trainable parameters of this '
                         'process. LoRA+ raises the learning rate of those; with none present it would '
                         f'change nothing. Set {NAME} = 1 if this is intended.')
    return result
