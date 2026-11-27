"""Which global ranks make one pipeline.

DeepSpeed numbers ranks stage-major by default: stage s holds ranks s*dp .. s*dp+dp-1, so the ranks of one
pipeline are dp apart and a pipeline reaches across nodes as soon as dp*stages is more than one node's GPUs.
'node_local' numbers them replica-major: pipeline d holds the consecutive ranks d*stages .. d*stages+stages-1.
Ranks fill a node before the next one, so a pipeline sits on one node whenever its stages fit and crosses a node
boundary only when it has to; the data parallel all-reduce, which happens once a step, is what crosses nodes.
"""

import os

PLACEMENTS = ('node_local', 'stage_major')
DEFAULT_PLACEMENT = 'node_local'


def check_placement(value):
    if value not in PLACEMENTS:
        raise ValueError(f'pipeline_placement must be one of {PLACEMENTS}, got {value!r}')
    return value


def stage_of_rank(rank, world, stages, placement=DEFAULT_PLACEMENT):
    """The pipeline stage a global rank holds."""
    stages = max(1, int(stages))
    if stages == 1:
        return 0
    if placement == 'stage_major':
        return rank // max(1, world // stages)
    return rank % stages


def data_rank_of_rank(rank, world, stages, placement=DEFAULT_PLACEMENT):
    """The data parallel index (which pipeline replica) of a global rank."""
    stages = max(1, int(stages))
    if stages == 1:
        return rank
    if placement == 'stage_major':
        return rank % max(1, world // stages)
    return rank // stages


def pipeline_topology(stages, world, placement=DEFAULT_PLACEMENT):
    """The DeepSpeed topology of this placement, or None for DeepSpeed's own (stage-major)."""
    if stages <= 1 or placement == 'stage_major':
        return None
    if world % stages != 0:
        raise RuntimeError(f'pipeline_stages ({stages}) must divide the number of GPUs ({world})')
    from deepspeed.runtime.pipe.topology import ProcessTopology
    return ProcessTopology(axes=['data', 'pipe'], dims=[world // stages, stages])


def straddling_note(stages, world, local_world_size):
    """A warning for pipelines that cross a node boundary although they would fit in one, or None."""
    if stages <= 1 or not local_world_size or world <= local_world_size:
        return None
    if stages <= local_world_size and local_world_size % stages != 0:
        return (f'pipeline_stages = {stages} does not divide the {local_world_size} GPUs of a node, so some pipelines '
                f'cross a node boundary. {stages} stages per pipeline fit a node when it holds a multiple of '
                f'{stages} GPUs (for {local_world_size} GPUs: a divisor such as '
                f'{max(d for d in range(1, local_world_size + 1) if local_world_size % d == 0 and d <= stages)}).')
    return None


def local_world_size():
    return int(os.environ.get('LOCAL_WORLD_SIZE', 0) or 0)
