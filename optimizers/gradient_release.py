import torch

# Simple wrapper for use with gradient release. Grad hooks do the optimizer steps, so this no-ops
# the step() and zero_grad() methods. It also handles state_dict.
class GradientReleaseOptimizerWrapper(torch.optim.Optimizer):
    def __init__(self, optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        ret = []
        for opt in self.optimizers:
            ret.extend(opt.param_groups)
        return ret

    @param_groups.setter
    def param_groups(self, groups):
        # train.py's --reset_optimizer_params puts back the groups it read before loading a
        # checkpoint: the per-parameter optimizers' groups, in the order the getter lists them.
        groups = list(groups)
        if len(groups) != sum(len(opt.param_groups) for opt in self.optimizers):
            raise ValueError(f'Expected one param group per wrapped optimizer group, got {len(groups)}')
        i = 0
        for opt in self.optimizers:
            n = len(opt.param_groups)
            opt.param_groups = groups[i:i + n]
            i += n

    def state_dict(self):
        return {i: opt.state_dict() for i, opt in enumerate(self.optimizers)}

    def load_state_dict(self, state_dict):
        for i, sd in state_dict.items():
            self.optimizers[i].load_state_dict(sd)

    def step(self):
        pass

    def zero_grad(self, set_to_none=True):
        pass