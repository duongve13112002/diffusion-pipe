"""Optimizer param group handling shared by train.py, kept free of training-stack imports so it
can be unit-tested on CPU."""

PROJECTION_KEYS = ('rank', 'proj_type', 'update_proj_gap')


def split_2d_param_groups(param_groups, kwargs):
    """genericoptim's groups: each group split into its 2D parameters and the rest.

    The low-rank projection settings (rank, proj_type, update_proj_gap) are per-group options in
    genericoptim, so they move out of the optimizer kwargs and onto every 2D group.
    """
    projection = {key: kwargs.pop(key) for key in PROJECTION_KEYS if key in kwargs}
    new_param_groups = []
    for pg in param_groups:
        params = pg.pop('params')
        params_2d = []
        params_other = []
        for p in params:
            if p.ndim == 2:
                params_2d.append(p)
            else:
                params_other.append(p)
        pg_2d = pg.copy()
        pg_2d['params'] = params_2d
        if kwargs.get('second_moment_type', None) == 'sn':
            pg_2d['subset_size'] = 'heuristics'
        pg_2d.update(projection)
        new_param_groups.append(pg_2d)
        pg_other = pg
        pg_other['params'] = params_other
        new_param_groups.append(pg_other)
    return new_param_groups


def group_names(param_groups):
    return [pg.get('name') for pg in param_groups]


def check_resumed_param_groups(before, after):
    """Refuse optimizer state loaded into other param groups.

    torch maps a saved optimizer state onto the current groups by position, checking only their
    sizes, so groups that hold other parameters than when the checkpoint was saved would get
    each other's moments without an error. A model that names its groups ('name', which the
    saved state carries) is checked here: before is the names of the groups the run built,
    after the names the loaded state brought.
    """
    if before != after:
        raise RuntimeError(
            'The checkpoint\'s optimizer state belongs to other param groups than this run builds '
            f'(saved: {after}; now: {before}), so loading it would hand parameters each other\'s '
            'optimizer state. Was a learning rate setting that changes the groups edited? Resume '
            'with --reset_optimizer to start the optimizer state afresh, or restore the settings.')
