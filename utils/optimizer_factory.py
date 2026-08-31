"""Resolution of the `optimizer.type` name to a class, shared by train.py and the tools.

Kept free of training-stack imports (no deepspeed, no models) so it can be unit-tested on CPU,
the same reason utils/lr_schedule.py is.

Only the *name to class* mapping lives here. Everything train.py does around it -- parameter
groups from the model, the weight-decay split keyed on `p.original_name`, gradient release with
its pipeline-engine monkeypatching, the GenericOptim 2-d/other split -- stays in train.py,
because all of it depends on the pipeline model. What is shared is the answer to "what does
`type = 'adamw8bit'` mean", and that answer should not be written down twice.

Optimizer imports are deliberately lazy: bitsandbytes, optimi, torchao and pytorch_optimizer are
optional, and a user of plain AdamW should not need any of them installed.
"""


def resolve_optimizer_class(optim_config):
    """Return (klass, args, kwargs) for an [optimizer] config table.

    `type` selects the class; every other key is passed through to its constructor, except the
    ones train.py consumes itself (`type`, `gradient_release`). `args` is non-empty only for
    wrapper optimizers that take an inner class positionally.
    """
    optim_type = optim_config['type']
    optim_type_lower = optim_type.lower()

    args = []
    kwargs = {k: v for k, v in optim_config.items() if k not in ('type', 'gradient_release')}

    if optim_type_lower == 'adamw':
        import torch
        # TODO: fix this. Building DeepSpeed's fused Adam extension fails with
        # "fatal error: cuda_runtime.h: No such file or directory".
        # klass = deepspeed.ops.adam.FusedAdam
        klass = torch.optim.AdamW
    elif optim_type_lower == 'adamw8bit':
        import bitsandbytes
        klass = bitsandbytes.optim.AdamW8bit
    elif optim_type_lower == 'adamw_optimi':
        import optimi
        klass = optimi.AdamW
    elif optim_type_lower == 'stableadamw':
        import optimi
        klass = optimi.StableAdamW
    elif optim_type_lower == 'sgd':
        import torch
        klass = torch.optim.SGD
    elif optim_type_lower == 'adamw8bitkahan':
        from optimizers import adamw_8bit
        klass = adamw_8bit.AdamW8bitKahan
    elif optim_type_lower == 'offload':
        import torch
        from torchao.prototype.low_bit_optim import CPUOffloadOptimizer
        klass = CPUOffloadOptimizer
        args.append(torch.optim.AdamW)
        kwargs['fused'] = True
    elif optim_type_lower == 'automagic':
        from optimizers import automagic
        klass = automagic.Automagic
    elif optim_type_lower == 'genericoptim':
        from optimizers import generic_optim
        klass = generic_optim.GenericOptim
    else:
        import pytorch_optimizer
        klass = getattr(pytorch_optimizer, optim_type)

    return klass, args, kwargs
