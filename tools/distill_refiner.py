"""Stage 1 for the anima_refiner architecture: distil a ContextRefiner from Anima's LLMAdapter.

anima_refiner replaces Anima's LLMAdapter with a ContextRefiner (models/text_refiner.py). The
DiT's cross-attention was trained to consume whatever the LLMAdapter emits, so a freshly
initialised refiner produces features the frozen DiT cannot read. Training it from a random
init with the diffusion loss works, but it needs images, the VAE and a full DiT forward pass
for every step.

This script gets the refiner into roughly the right space using captions only: no images, no
VAE, no diffusion. It is a warm start, not a finished model -- follow it with a diffusion-loss
stage (see docs/anima_refiner/README.md).

The loss is measured at the cross-attention output, not position by position. The obvious
objective, a position-wise MSE between teacher and student features, does not work here. The teacher's output sequence is indexed by *T5* tokens (the LLMAdapter embeds T5
ids as its query sequence) while the student's is indexed by the source LLM's own tokens. Both
are (B, L, 1024) so the shapes match and the code would run, but position i means different
things on each side, and the loss plateaus without ever converging.

Cross-attention is a weighted sum over text positions, so its output does not depend on how
that text is indexed. Pushing both feature sets through the DiT's own frozen cross-attention
modules and comparing there sidesteps the mismatch entirely, and it optimises exactly the
quantity the DiT will consume.

Usage:
    python -m tools.distill_refiner --config examples/anima_refiner/distill.toml
    deepspeed --num_gpus=4 tools/distill_refiner.py --config examples/anima_refiner/distill.toml

The launcher and the parallelism strategy are separate choices. Either launcher works -- both
export RANK/LOCAL_RANK/WORLD_SIZE, which is all this script reads -- and the strategy is set by
`distributed_strategy` under [distill]: 'ddp' (the default) or 'zero1'/'zero2'. See
build_strategy and docs/anima_refiner/README.md for why DDP is the default here.
"""

import argparse
import datetime
import contextlib
import math
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import safetensors.torch
import toml
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, T5TokenizerFast

from models.cosmos_predict2 import (get_dit_config, _tokenize as cosmos_tokenize,
                                    normalise_hidden_layer)
from models.cosmos_predict2_modeling import MiniTrainDIT
from models.text_refiner import ContextRefiner, extract_refiner_state_dict
from utils.common import iterate_safetensors, load_state_dict
from utils.caption_corpus import read_corpus
# The batch fill keys mean the same thing here as in a dataset config and are validated by the
# same function, so the two cannot drift into accepting different spellings. It costs 1.6s of
# import on a ~20s startup, measured, which is not the 50s case utils/captions.py was split out
# to avoid.
from utils.dataset import resolve_batch_fill_config
from utils.lr_schedule import create_lr_scheduler
from utils.optimizer_factory import resolve_optimizer_class
from utils.captions import enumerate_captions, preprocess_caption, tag_markers

MAX_TEXT_LENGTH_DEFAULT = 512


def load_captions(config):
    """Get the captions to distil on, preferring the ordinary dataset.toml flow.

    `dataset` points at the same dataset.toml every other training mode uses, so the captions
    seen here are exactly the ones training will see -- same directories, same captions.json /
    .txt resolution, same caption_prefix and tag shuffling. Images are never opened: this stage
    trains only the text frontend.

    `caption_corpus` points at a file produced by tools/export_caption_corpus.py: the same
    captions, already flattened, so a few million images do not have to be walked again. It is
    the same set of strings either way -- the corpus is a cache of the walk, not a different
    source of truth.

    `captions` is a fallback for when there is no dataset.toml to hand: either a file with one
    caption per line, or a directory of .txt files.
    """
    distill_config = config['distill']
    sources = [k for k in ('dataset', 'caption_corpus', 'captions') if k in distill_config]
    if len(sources) > 1:
        raise RuntimeError(
            f"[distill] sets {' and '.join(repr(s) for s in sources)}; they are alternative "
            f'caption sources, so set exactly one.'
        )

    if 'dataset' in distill_config:
        dataset_config = toml.load(distill_config['dataset'])
        # apply_shuffle=False: shuffling and dropout are applied per sample below instead of
        # being baked into a fixed set of variants here, so every epoch sees a fresh one.
        return enumerate_captions(
            dataset_config,
            apply_num_repeats=distill_config.get('apply_num_repeats', False),
            apply_shuffle=False,
        )

    if 'caption_corpus' in distill_config:
        # apply_num_repeats has no effect here: a corpus stores no num_repeats, because the
        # dataset's semantics are per directory, not per caption. Expand it at export time with
        # export_caption_corpus.py --apply-num-repeats instead.
        if distill_config.get('apply_num_repeats', False):
            print(
                "WARNING: [distill] apply_num_repeats is set but the caption source is a corpus, "
                "which stores no num_repeats. Re-export with --apply-num-repeats to honour it."
            )
        return read_corpus(
            distill_config['caption_corpus'],
            fmt=distill_config.get('caption_corpus_format', None),
        )

    if 'captions' not in distill_config:
        raise RuntimeError(
            "set one of 'dataset' (a dataset.toml), 'caption_corpus' (a file from "
            "tools/export_caption_corpus.py) or 'captions' under [distill]"
        )

    path = Path(distill_config['captions'])
    if path.is_dir():
        return [
            text for txt in sorted(path.rglob('*.txt'))
            if (text := txt.read_text(encoding='utf-8-sig').strip())
        ]
    return [line.strip() for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def load_captions_once(config, rank, world_size, is_main):
    """Resolve the captions on rank 0 and hand them to everyone else.

    load_captions walks every directory and opens every tar when the source is a dataset.toml --
    its own docstring quotes about three minutes for three million captions. Every rank was
    doing that walk independently against the same tree, which on a network filesystem is the
    difference between one client and eight hammering it at startup.

    The corpus source (tools/export_caption_corpus.py) is cheap enough that broadcasting is
    roughly a wash, but it costs nothing to take the same path, and one code path is easier to
    reason about than two.

    Single process: no broadcast, no serialisation, exactly what it did before.
    """
    if world_size < 2:
        return load_captions(config)

    payload = [load_captions(config) if is_main else None]
    dist.broadcast_object_list(payload, src=0)
    captions = payload[0]
    if captions is None:
        raise RuntimeError(
            f'rank {rank} received no captions from rank 0. The broadcast failed, or rank 0 '
            'raised while resolving them.'
        )
    return captions


class EpochSampler:
    """One epoch is one pass over every caption, sharded across ranks.

    The loop used to draw `random.sample(captions, batch_size)` per micro batch, which samples
    with replacement across steps: some captions appear many times before others appear once,
    and "epoch" has no meaning. train.py has always been epoch-driven, and a distillation run
    should be describable the same way.

    Each epoch shuffles the FULL list with a seed derived from the epoch number -- identical on
    every rank -- and then takes this rank's stride. Shuffling before sharding is what makes the
    shards differ between epochs; sharding a fixed order would pin each rank to the same
    captions forever. Every caption is seen exactly once per epoch across the job, which is the
    property that makes the word mean anything.

    What happens to the tail that does not fill a whole global batch depends on
    batch_fill_strategy. Under 'drop' it is discarded, the same rounding SizeBucketDataset does.
    Under 'fill' the batch is completed instead, preferring captions borrowed from the next
    epoch's permutation and falling back to masked-out repeats. Either way every rank runs the
    same number of steps, so no collective is left waiting on a rank that finished early.
    """

    def __init__(self, captions, batch_size, grad_accum, rank, world_size, seed,
                 fill_strategy='drop', undersized='pad_masked', min_real_fraction=0.25):
        self.captions = captions
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.fill_strategy = fill_strategy
        self.undersized = undersized
        self.min_real_fraction = min_real_fraction
        self.global_batch = batch_size * grad_accum * world_size
        self.num_masked_per_epoch = 0

        too_small = len(captions) < self.global_batch
        if fill_strategy == 'fill' and not (too_small and undersized == 'drop'):
            self.steps_per_epoch = math.ceil(len(captions) / self.global_batch)
            if too_small:
                fraction = len(captions) / self.global_batch
                if fraction < min_real_fraction:
                    raise RuntimeError(
                        f'{len(captions)} captions against a global batch of '
                        f'{self.global_batch} is a real fraction of {fraction:.3f}, below '
                        f'min_real_fraction {min_real_fraction}: most of every step would be '
                        'masked-out padding. Lower batch_size or gradient_accumulation_steps, '
                        'supply more captions, or lower min_real_fraction.'
                    )
        else:
            self.steps_per_epoch = len(captions) // self.global_batch
        if self.steps_per_epoch < 1:
            raise RuntimeError(
                f'{len(captions)} captions cannot fill one global batch of '
                f'{self.global_batch} (batch_size {batch_size} x '
                f'gradient_accumulation_steps {grad_accum} x world_size {world_size}). '
                'Lower batch_size or gradient_accumulation_steps.'
            )

    def epoch_order(self, epoch):
        """This rank's captions for `epoch`, already shuffled and sharded."""
        return [caption for caption, _ in self.epoch_order_weighted(epoch)]

    def epoch_order_weighted(self, epoch):
        """This rank's (caption, loss weight) pairs for `epoch`.

        The single source of truth; epoch_order is the caption-only view of it. Two methods
        computing the order separately would be two things to keep in step, which is the same
        trap resolve_schedule below exists to avoid.

        The weight is 1.0 for every caption under 'drop', so that path is unchanged. Under
        'fill' a masked repeat gets 0.0 and the real captions of a batch that contains one get
        G/G_real, which undoes the dilution `.mean()` introduces by dividing by the padded
        count. The scale is computed over the WHOLE global batch, never over this rank's shard
        or one micro batch: the ranks average their gradients and the micro batches average
        their losses, so one shared constant is what makes the result equal a mean over the
        real captions no matter how the padding lands.
        """
        order = list(self.captions)
        random.Random(self.seed + epoch).shuffle(order)
        usable = self.steps_per_epoch * self.global_batch
        weights = [1.0] * len(order)

        if len(order) < usable:
            order, weights = self._extend(order, weights, usable, epoch)
        order, weights = order[:usable], weights[:usable]

        for start in range(0, usable, self.global_batch):
            block = weights[start:start + self.global_batch]
            num_real = sum(1 for w in block if w > 0)
            if 0 < num_real < len(block):
                scale = len(block) / num_real
                for k in range(start, start + self.global_batch):
                    if weights[k] > 0:
                        weights[k] = scale

        pairs = list(zip(order, weights))
        # Strided sharding, so a contiguous slice of this rank's list corresponds to a
        # contiguous block of the global order: rank r's local element k is global element
        # k*world_size + r, and one step's slice across all ranks is exactly the global batch.
        return pairs[self.rank:usable:self.world_size]

    def _extend(self, order, weights, usable, epoch):
        """Complete the final global batch of `epoch`.

        Captions come from the NEXT epoch's permutation, which keeps this a pure function of
        (seed, epoch) -- resume needs the order to be reconstructible from the epoch number and
        nothing else. Anything already in the part of the final batch that comes from this
        epoch is skipped, so the batch holds no caption twice.

        Duplicates are detected by caption text, not by index: a corpus that genuinely contains
        the same string twice would otherwise put two identical captions in one batch, and the
        relational term reads a pair of identical captions as a distance of zero and pushes the
        model toward exactly the collapse it exists to prevent.
        """
        num_missing = usable - len(order)
        from_this_epoch = order[len(order) - (self.global_batch - num_missing):] if num_missing < self.global_batch else order
        taken = set(from_this_epoch)

        borrowed = list(self.captions)
        random.Random(self.seed + epoch + 1).shuffle(borrowed)
        added, added_weights = [], []
        for caption in borrowed:
            if len(added) == num_missing:
                break
            if caption in taken:
                continue
            taken.add(caption)
            added.append(caption)
            added_weights.append(1.0)
        # Nothing distinct left. A repeat is the only way to fill the batch, and it is masked
        # so it teaches nothing -- the constructor already refused the case where that would be
        # most of the batch.
        while len(added) < num_missing:
            added.append(order[len(added) % len(order)])
            added_weights.append(0.0)

        self.num_masked_per_epoch = added_weights.count(0.0)
        return order + added, weights + added_weights


def resolve_schedule(epochs, steps, captions, batch_size, grad_accum, rank, world_size, seed,
                     fill_strategy='drop', undersized='pad_masked', min_real_fraction=0.25):
    """Build the sampler and settle the final step count in one place.

    They are one decision rather than two: with `epochs` the step count is a property of the
    sampler, so every consumer of `steps` -- the LR scheduler's total, the resume guard, the
    progress bar -- reads a wrong value if it runs before this does. That is not hypothetical.
    The derivation used to sit inline in main() below the optimizer, and the LR scheduler was
    built one screen too early holding total_steps=None: it constructed fine and raised
    TypeError the moment warmup ended, 500 steps into a four-GPU run. Returning both together
    is what makes reading one without the other impossible.

    Returns (sampler, steps, description) -- the description is the line worth printing.
    """
    sampler = EpochSampler(captions, batch_size, grad_accum, rank, world_size, seed,
                           fill_strategy=fill_strategy, undersized=undersized,
                           min_real_fraction=min_real_fraction)
    if epochs is not None:
        steps = epochs * sampler.steps_per_epoch
        description = (
            f'{epochs} epochs over {len(captions)} captions = {steps} steps '
            f'({sampler.steps_per_epoch} steps/epoch at a global batch of {sampler.global_batch})'
        )
    else:
        description = (
            f'{steps} steps over {len(captions)} captions '
            f'({steps / max(sampler.steps_per_epoch, 1):.2f} epochs at a global batch of '
            f'{sampler.global_batch})'
        )
    if fill_strategy == 'fill':
        # steps_per_epoch rounds up here rather than down, so say why an epoch is one step
        # longer than the division suggests instead of leaving it to be worked out.
        #
        # How many of those slots end up masked is not stated: it depends on how many distinct
        # captions are left once the batch's own are excluded, which is a property of the
        # epoch's permutation. Computing it here would mean building an epoch's order before
        # the run starts, and on a multi-million caption corpus that is a real cost for a log
        # line.
        tail = sampler.steps_per_epoch * sampler.global_batch - len(captions)
        description += (
            f"\nbatch_fill_strategy = 'fill': the last batch of each epoch is completed with "
            f"{tail} more caption slot(s), taken from the next epoch's order where distinct "
            'captions remain and from masked-out repeats otherwise'
        )
    return sampler, steps, description


def warmup_advice(warmup_steps, steps):
    """What to say about a warmup that is too long for the run, or None if it is fine.

    A separate function so it can be tested, and because it is about the same trap
    resolve_schedule exists for: with `epochs` the step count is derived from the corpus size
    rather than written down, so the default 500 -- chosen against a 20,000-step run, where it is
    2.5% -- silently becomes a third of the schedule on a smaller corpus.
    """
    if warmup_steps >= steps:
        return (
            f'WARNING: warmup_steps ({warmup_steps}) is not shorter than the run ({steps} '
            'steps). SequentialLR never reaches its milestone, so the learning rate ramps for '
            'the whole run and the decay phase never starts. Lower warmup_steps, or raise '
            'epochs/steps.'
        )
    if warmup_steps > steps // 4:
        return (
            f'WARNING: warmup_steps ({warmup_steps}) is {100 * warmup_steps / steps:.0f}% of '
            f'this {steps}-step run. The default of 500 assumes a 20,000-step run; with '
            '`epochs` the step count follows the corpus size. Roughly 5% is the usual choice, '
            f'so about {max(1, steps // 20)} here.'
        )
    return None


def caption_augment_config(config):
    """Resolve the caption augmentation applied to each sampled batch.

    Settings are read from [distill] first, then from the dataset.toml when that is the caption
    source, so a run driven by a dataset.toml augments the way the diffusion stages do without
    having to restate it. Only top-level dataset.toml keys are picked up: a batch here is a
    flat sample of captions with no directory attached, so per-directory overrides have nothing
    to attach to and must be restated under [distill] if they matter. Unlike the diffusion stages, which bake a fixed number of shuffled
    variants into the embedding cache, distillation re-embeds every step and so can augment per
    sample -- each epoch sees a different tag order and a different dropout draw.
    """
    distill_config = config['distill']
    fallback = {}
    if 'dataset' in distill_config:
        dataset_config = toml.load(distill_config['dataset'])
        fallback = {k: v for k, v in dataset_config.items() if not isinstance(v, (list, dict))}
        # prefix_tag_caption is resolved per directory, so a single top-level value cannot
        # represent a dataset that annotates its directories differently. Collect every marker
        # in use and hand the whole list down; split_tag_prefix accepts a list. Read from the
        # config rather than by walking the dataset: the markers are config values, and a
        # second full enumeration would re-open every image and every tar for nothing.
        markers = set()
        for directory_config in dataset_config.get('directory', []):
            markers.update(tag_markers(
                directory_config.get('prefix_tag_caption', dataset_config.get('prefix_tag_caption', ''))
            ))
        if markers:
            fallback['prefix_tag_caption'] = sorted(markers)

    def setting(key, default):
        if key in distill_config:
            return distill_config[key]
        return fallback.get(key, default)

    if 'dataset' not in distill_config:
        # A corpus or a bare caption file carries no dataset config to fall back on, so
        # anything not restated under [distill] silently defaults to off. Say so, rather than
        # quietly training on a distribution that differs from the diffusion stages'.
        settings = ('prefix_tag_caption', 'caption_prefix', 'cache_shuffle_num',
                    'shuffle_tags', 'tag_dropout_rate')
        restated = [k for k in settings if k in distill_config]
        if not restated:
            print(
                'WARNING: the caption source is not a dataset.toml, so no caption settings can '
                'be inherited. Tag shuffling, tag dropout, caption_prefix and prefix_tag_caption '
                'are all off. If the dataset uses any of them, restate them under [distill] -- '
                'export_caption_corpus.py prints the prefix_tag_caption line to use. Without '
                'prefix_tag_caption the tag marker is trained as if it were a tag.'
            )
        elif 'prefix_tag_caption' not in distill_config:
            # Restating any one setting used to silence the warning entirely, so a config that
            # set caption_prefix and forgot the marker got no warning at all -- and the marker
            # is the one whose absence corrupts the caption rather than merely differing from
            # the diffusion stages.
            print(
                'WARNING: the caption source is not a dataset.toml and prefix_tag_caption is not '
                f'restated under [distill], though {", ".join(restated)} is. If the dataset uses '
                'a tag marker, the marker is being trained as if it were a tag -- '
                'export_caption_corpus.py prints the line to use.'
            )

    shuffle = setting('cache_shuffle_num', 0) > 0 or setting('shuffle_tags', False)
    # caption_prefix is applied HERE, not by the caption source. Training builds a caption as
    # caption_prefix + augment(strip_marker(raw)), so the prefix has to go on after the marker
    # comes off; a source that baked it in would hide the marker behind it.
    return {
        'delimiter': setting('cache_shuffle_delimiter', ', '),
        'caption_prefix': setting('caption_prefix', ''),
        'prefix_tag_caption': setting('prefix_tag_caption', ''),
        'shuffle': shuffle,
        'tag_dropout_rate': setting('tag_dropout_rate', 0.0),
    }


def build_teacher(config, dtype, device):
    """Qwen3-0.6B plus the LLMAdapter and cross-attention modules from an Anima checkpoint."""
    llm_path = config['teacher']['llm_path']
    tokenizer = AutoTokenizer.from_pretrained('configs/qwen3_06b', local_files_only=True)
    llm_config = transformers.Qwen3Config.from_pretrained('configs/qwen3_06b', local_files_only=True)
    with init_empty_weights():
        llm = transformers.Qwen3ForCausalLM(llm_config)
    for key, tensor in iterate_safetensors(llm_path):
        set_module_tensor_to_device(llm, key, device='cpu', dtype=dtype, value=tensor)
    text_encoder = llm.model
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_encoder.config.use_cache = False

    state_dict = load_state_dict(config['teacher']['transformer_path'])
    state_dict = {(k[len('net.'):] if k.startswith('net.') else k): v for k, v in state_dict.items()}
    dit_config = get_dit_config(state_dict)
    if 'llm_adapter.out_proj.weight' not in state_dict:
        raise RuntimeError(
            'teacher.transformer_path has no llm_adapter weights. Distillation needs a trained '
            'Anima checkpoint to act as the teacher.'
        )
    dit_config['use_llm_adapter'] = True
    with init_empty_weights():
        dit = MiniTrainDIT(**dit_config)
    for name, p in dit.named_parameters():
        if name in state_dict:
            set_module_tensor_to_device(dit, name, device='cpu', dtype=dtype, value=state_dict[name])

    llm_adapter = dit.llm_adapter
    # Probe through a spread of blocks rather than all of them: adjacent blocks give highly
    # correlated signal, so a subset covers the same ground for less compute.
    num_probe_blocks = config.get('probe', {}).get('num_blocks', 8)
    if num_probe_blocks < 1:
        raise RuntimeError(
            f'[probe] num_blocks must be >= 1, got {num_probe_blocks}. It selects how many of '
            "the DiT's frozen cross-attention blocks the loss is measured through; zero blocks "
            'is not a valid objective.'
        )
    num_probe_blocks = min(num_probe_blocks, len(dit.blocks))
    stride = max(1, len(dit.blocks) // num_probe_blocks)
    block_indices = list(range(0, len(dit.blocks), stride))[:num_probe_blocks]
    cross_attns = nn.ModuleList([dit.blocks[i].cross_attn for i in block_indices])
    # Two different dimensions, easy to confuse: model_channels is the image side, which
    # the probe queries live in, while crossattn_emb_channels is the text side the refiner
    # must output. Passing the former as the refiner's model_dim would build a refiner
    # twice the required width and fail at the first cross-attention.
    model_channels = dit_config['model_channels']
    crossattn_emb_channels = dit_config['crossattn_emb_channels']

    # Everything else in the DiT is dead weight for this stage -- unless the denoising rollout
    # is on, which needs the whole thing to predict with. Keeping it costs several GB, which is
    # why it is opt-in and why the default path still throws it away.
    rollout_config = config.get('rollout', {})
    keep_dit = rollout_config.get('loss_weight', 0.0) > 0
    dit.llm_adapter = None
    # Released BEFORE the device transfer. load_state_dict materialises every tensor, so holding
    # it across dit.to(device) means the checkpoint and the CPU-side DiT are both resident at
    # once -- roughly 8 GB of host RAM for Anima, and only when the rollout is on.
    del state_dict
    if keep_dit:
        missing = [name for name, p in dit.named_parameters() if p.is_meta]
        if missing:
            # Parameters absent from the checkpoint stay on the meta device, and Module.to()
            # does not materialise them. Only cross_attn and llm_adapter were ever used before,
            # so this could not bite; the rollout runs the whole DiT.
            raise RuntimeError(
                f'The teacher checkpoint is missing {len(missing)} DiT parameters that the '
                f'denoising rollout needs, starting with {missing[:5]}. They would fail as meta '
                'tensors at the first rollout forward.'
            )
        if dit_config['in_channels'] != dit_config['out_channels']:
            # The Euler step x <- x - dt*v needs v to have x's shape.
            raise RuntimeError(
                f"The rollout needs in_channels == out_channels, got "
                f"{dit_config['in_channels']} and {dit_config['out_channels']}."
            )
        dit.to(device).eval().requires_grad_(False)
    else:
        dit.blocks = None
        dit = None

    text_encoder.to(device).eval().requires_grad_(False)
    llm_adapter.to(device).eval().requires_grad_(False)
    cross_attns.to(device).eval().requires_grad_(False)
    return (tokenizer, text_encoder, llm_adapter, cross_attns, model_channels,
            crossattn_emb_channels, dit, dit_config)


def build_student(config, dtype, device, model_dim):
    """Source LLM plus a fresh (or resumed) ContextRefiner.

    model_dim comes from the teacher DiT's crossattn_emb_channels rather than the config:
    the refiner's output feeds that cross-attention, so the DiT is the authority on the size.

    When resuming, the layer count is taken FROM THE RESUMED WEIGHTS for the same reason the
    training pipeline derives it from the checkpoint -- a config that disagrees with the
    weights is a mistake, not an instruction.
    """
    llm_path = config['student']['llm_path']
    tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
    llm_config = transformers.AutoConfig.from_pretrained(llm_path, local_files_only=True)
    if hasattr(llm_config, 'text_config'):
        full_model = transformers.AutoModelForImageTextToText.from_pretrained(
            llm_path, dtype=dtype, local_files_only=True
        )
        text_encoder = full_model.model.language_model
        del full_model
        cap_feat_dim = llm_config.text_config.hidden_size
    else:
        text_encoder = transformers.AutoModelForCausalLM.from_pretrained(
            llm_path, dtype=dtype, local_files_only=True
        ).model
        cap_feat_dim = llm_config.hidden_size
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_encoder.config.use_cache = False
    text_encoder.to(device).eval().requires_grad_(False)

    resume = config['student'].get('resume_from', None)
    resumed_state_dict = extract_refiner_state_dict(load_state_dict(resume)) if resume else None

    num_layers = config['student'].get('n_refiner_layers', 6)
    if resumed_state_dict is not None:
        derived_layers = 1 + max(
            (int(k.split('.')[1]) for k in resumed_state_dict if k.startswith('blocks.')), default=-1
        )
        derived_cap_feat_dim = resumed_state_dict['cap_embedder.1.weight'].shape[1]
        if config['student'].get('n_refiner_layers', None) not in (None, derived_layers):
            raise RuntimeError(
                f"n_refiner_layers={config['student']['n_refiner_layers']} in the config, but "
                f'{resume} holds {derived_layers} layers. Remove n_refiner_layers to use what '
                'the weights carry.'
            )
        if derived_cap_feat_dim != cap_feat_dim:
            raise RuntimeError(
                f'{resume} expects a text encoder with hidden size {derived_cap_feat_dim}, but '
                f'llm_path provides {cap_feat_dim}. These weights were trained against a '
                'different text encoder.'
            )
        num_layers = derived_layers

    refiner = ContextRefiner(
        cap_feat_dim=cap_feat_dim,
        model_dim=model_dim,
        num_layers=num_layers,
    )
    refiner.init_weights()
    if resumed_state_dict is not None:
        refiner.load_state_dict(resumed_state_dict)
        print(f'Resumed refiner weights from {resume} ({num_layers} layers)')
    # Trained in fp32 for stable optimisation; it is small enough that this is cheap.
    refiner.to(device=device, dtype=torch.float32).train()
    return tokenizer, text_encoder, refiner, cap_feat_dim


def encode(text_encoder, input_ids, attn_mask, hidden_layer):
    if hidden_layer is None:
        out = text_encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
    else:
        out = text_encoder(
            input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True
        ).hidden_states[hidden_layer].clone()
    out[~attn_mask.bool()] = 0
    return out


def _velocity(dit, x, t, cond_feats, uncond_feats, guidance_scale):
    """One frozen-DiT prediction, optionally with classifier-free guidance.

    Anima is a rectified-flow model: the DiT predicts a velocity, and the sampling ODE is
    dx/dt = v. That is why the caller advances with a plain Euler step rather than a DDPM
    posterior -- the paper this follows is written for DDPM and the schedule does not carry
    over unchanged.
    """
    padding_mask = torch.zeros(x.shape[0], 1, x.shape[3], x.shape[4], dtype=x.dtype, device=x.device)
    v = dit(x, t, cond_feats, padding_mask=padding_mask)
    if guidance_scale > 0 and uncond_feats is not None:
        v_uncond = dit(x, t, uncond_feats, padding_mask=padding_mask)
        v = v_uncond + guidance_scale * (v - v_uncond)
    return v


def shifted_schedule(steps, shift, device):
    """The timesteps the trajectory visits, warped the way training and sampling warp them.

    A uniform walk from 1 to 0 is not where the model is actually asked to predict: both
    prepare_inputs and the sampler apply t <- (t*shift) / (1 + (shift-1)*t), which concentrates
    steps near the noisy end. The loss is unbiased either way -- teacher and student see the
    same t -- but the visited x_t only lie on the real sampling path if the same warp is used.

    shift = 1 (the default) leaves the schedule uniform, which is exactly what an unset shift
    means everywhere else in the repo.
    """
    schedule = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    if shift and shift != 1.0:
        schedule = (schedule * shift) / (1 + (shift - 1) * schedule)
    return schedule


@torch.no_grad()
def teacher_trajectory(dit, teacher_feats, teacher_uncond, shape, steps, guidance_scale,
                       generator, device, dtype, shift=1.0):
    """Walk from pure noise toward clean, driven entirely by the teacher.

    The student never advances the trajectory and never sees its own output as input, so there
    is no error accumulation and none of the exposure-bias machinery that the phrase "rollout"
    usually implies applies here. That is not a simplification of the method -- it is what the
    method does (Scaling Down Text Encoders of Text-to-Image Diffusion Models, CVPR 2025,
    Algorithm 1: both models are evaluated at the SAME x_t, and x_t is advanced with the
    teacher's prediction).

    The whole walk therefore runs under no_grad. It exists to produce x_t that lie on a real
    sampling trajectory, which matters because this stage has no images: without x_0 there is
    no way to build x_t = (1-t)*x_0 + t*noise the way training normally would.

    Returns (x_t, t, v) triples. t runs from 1 (pure noise) to just above 0, and v is the
    teacher's velocity AT that point -- kept rather than discarded, because it is exactly the
    target the loss needs there. Recomputing it in rollout_loss was 2 redundant full-DiT
    forwards per micro-batch at the shipped settings, 4 with guidance on.
    """
    x = torch.randn(shape, generator=generator, device='cpu').to(device=device, dtype=dtype)
    schedule = shifted_schedule(steps, shift, device)
    visited = []
    for i in range(steps):
        t = schedule[i].to(dtype).expand(shape[0], 1)
        v = _velocity(dit, x, t, teacher_feats, teacher_uncond, guidance_scale)
        visited.append((x, t, v))
        if i + 1 < steps:
            # The final step would produce an x nothing ever reads.
            x = x - (schedule[i] - schedule[i + 1]).to(dtype) * v
    return visited


def rollout_loss(dit, visited, teacher_feats, student_feats, teacher_uncond, student_uncond,
                 guidance_scale, loss_points, rng, weights=None):
    """Compare what the frozen DiT predicts from the two text frontends, on the trajectory.

    This is the objective the probe loss approximates. The probe measures agreement at the
    cross-attention output for synthetic queries; this measures it where it actually matters,
    at the model's own prediction, for queries the model itself produced.

    Only the student side carries gradient. The teacher side and the trajectory are frozen, so
    the cost of a longer rollout is inference cost, not backward cost -- `steps` and
    `loss_points` are independent knobs on purpose.
    """
    if (student_uncond is None) != (teacher_uncond is None):
        # One side guided and the other not is a silently biased comparison, not a degraded
        # mode. _velocity drops guidance when its uncond argument is None, so this would be
        # invisible.
        raise RuntimeError(
            'rollout_loss needs unconditional features for both sides or neither; got '
            f'teacher_uncond={teacher_uncond is not None}, student_uncond={student_uncond is not None}.'
        )
    chosen = rng.sample(visited, min(loss_points, len(visited)))
    loss = 0.0
    for x, t, target in chosen:
        # target came from teacher_trajectory, which already ran this exact forward.
        prediction = _velocity(dit, x, t, student_feats, student_uncond, guidance_scale)
        loss = loss + weighted_mse(prediction.float(), target.float(), weights)
    return loss / len(chosen)


@torch.no_grad()
def build_unconditional_features(teacher_tok, t5_tokenizer, teacher_llm, llm_adapter,
                                 student_tok, student_llm, max_text_length, device,
                                 llm_hidden_layer):
    """Everything the guided rollout needs for the unconditional branch, computed once.

    Classifier-free guidance needs an unconditional prediction from both frontends. All of it is
    constant: the caption is always '', both LLMs are frozen, and the teacher's adapter is
    frozen. Only the REFINER's view of the student's hidden state changes as training proceeds,
    and that stays in the training loop.

    The two sides tokenize differently on purpose. The student path passes
    keep_one_real_token=True, because Qwen pads with its own eos and adds no bos, so '' becomes
    an all-padding row -- the refiner would emit zeros and hand the frozen DiT a context its
    original training never produced. The teacher path does not, because its QUERY sequence is
    old T5's, which already yields </s> for '' and needs no help. That asymmetry mirrors
    _tokenize(keep_one_real_token=self.use_context_refiner) in models/cosmos_predict2.py.

    That argument covers target_attention_mask only. source_attention_mask is a different mask,
    built from the teacher LLM's own tokenizer, and for '' it is entirely zero -- so every query
    attends over a fully masked key set. What keeps that finite is allow_fully_masked_rows in
    models/llm_adapter.py, not anything here.

    Returns (teacher_uncond, student_ids, student_mask, student_hidden), each with batch size 1;
    the caller expands to the real batch.
    """
    t_uncond = teacher_tok([''], return_tensors='pt', truncation=True,
                           padding='max_length', max_length=max_text_length)
    t5_uncond = t5_tokenizer([''], return_tensors='pt', truncation=True,
                             padding='max_length', max_length=max_text_length)
    uncond_hidden = encode(teacher_llm, t_uncond.input_ids.to(device),
                           t_uncond.attention_mask.to(device), None)
    teacher_uncond = llm_adapter(
        source_hidden_states=uncond_hidden,
        target_input_ids=t5_uncond.input_ids.to(device),
        target_attention_mask=t5_uncond.attention_mask.to(device),
        source_attention_mask=t_uncond.attention_mask.to(device),
    )
    t5_uncond_mask = t5_uncond.attention_mask.to(device)
    teacher_uncond = teacher_uncond * t5_uncond_mask.unsqueeze(-1).to(teacher_uncond.dtype)

    s_uncond = cosmos_tokenize(student_tok, [''], max_text_length, keep_one_real_token=True)
    student_mask = s_uncond['attention_mask'].to(device)
    student_ids = s_uncond['input_ids'].to(device)
    student_hidden = encode(student_llm, student_ids, student_mask, llm_hidden_layer)
    return teacher_uncond, student_ids, student_mask, student_hidden


def weighted_mse(prediction, target, weights):
    """Mean squared error that ignores the samples batch fill masked out.

    weights is None whenever the batch holds no padding, and then this is exactly
    F.mse_loss(prediction, target): every sample carries the same element count, so the mean of
    the per-sample means equals the mean over all elements. That equality is what keeps the
    'drop' path unchanged rather than merely close.

    Dividing by the batch size rather than by the weight sum is deliberate. The real samples
    already carry G/G_real, which EpochSampler computed over the whole global batch, and both
    gradient accumulation and the data-parallel all-reduce are averages -- so one shared
    constant lands on the mean over real captions however unevenly the padding is spread, while
    a per-micro-batch normalisation does not, and would divide by zero on a micro batch that is
    entirely padding.
    """
    if weights is None:
        return F.mse_loss(prediction, target)
    per_sample = F.mse_loss(prediction, target, reduction='none').flatten(1).mean(dim=1)
    return (per_sample * weights).mean()


def relational_loss(student_pooled, teacher_pooled):
    """Match the teacher's pairwise distance structure, not just its per-sample values.

    The probe objective compares captions one at a time, so it constrains where each caption
    lands but says nothing about how captions sit relative to each other. A student that maps
    every caption to the same point satisfies it about as well as one that keeps them apart --
    which is exactly the mode collapse reported for naive text-encoder distillation
    (Scaling Down Text Encoders of Text-to-Image Diffusion Models, CVPR 2025: "rat", "cat" and
    "man" receiving identical embeddings).

    This is Relational Knowledge Distillation's distance-wise term (Park et al., 2019): take the
    pairwise distances within the batch, normalise each side by its own mean so absolute scale
    does not matter, and match the two structures. Collapse drives every student distance toward
    zero, which this penalises directly and the probe loss does not see at all.

    It costs nothing extra -- both feature sets are already computed for the probe term -- and
    needs no images, so the stage keeps the property that is its whole reason to exist.
    """
    if student_pooled.shape[0] < 2:
        # A single sample has no pairs, so there is no structure to preserve.
        return student_pooled.sum() * 0.0

    teacher_distances = torch.cdist(teacher_pooled, teacher_pooled)
    student_distances = torch.cdist(student_pooled, student_pooled)
    # BOTH sides are divided by the teacher's mean distance, not each by its own. Dividing each
    # by its own -- which is how RKD is usually written -- makes the loss scale invariant, and
    # scale is precisely what collapse destroys: a student whose captions all drift toward one
    # point has uniformly smaller distances, and normalising that away hides it. Measured on a
    # synthetic teacher, per-side normalisation scored 0.0000 at every collapse fraction from
    # 25% to 90%. A shared teacher-side scale keeps the comparison unit-free while leaving
    # shrinkage visible.
    n = teacher_pooled.shape[0]
    scale = (teacher_distances.sum() / (n * (n - 1))).clamp_min(1e-8)
    return F.smooth_l1_loss(student_distances / scale, teacher_distances / scale)


def mean_pairwise_cosine_distance(x):
    """Collapse diagnostic: how far apart distinct captions sit, on average.

    Falling toward zero means the refiner is mapping different captions to the same feature.
    Logged rather than acted on, because the number is only meaningful as a trend.
    """
    if x.shape[0] < 2:
        return 0.0
    normed = F.normalize(x.float(), dim=-1)
    similarity = normed @ normed.T
    n = x.shape[0]
    off_diagonal = (similarity.sum() - similarity.diagonal().sum()) / (n * (n - 1))
    return float(1.0 - off_diagonal)


def padded_mean(x, mask, length):
    """Sum over real tokens divided by the PADDED length, not by the number of real tokens.

    This has to match how the probe-attention term behaves. Padded context rows project to
    k = 0 (k_proj has no bias and RMSNorm(0) = 0), so every padded position still contributes
    exp(0) = 1 to the attention softmax denominator while contributing nothing to the
    numerator. At max_text_length = 512 with a 20-token caption, ~490 pad terms dominate that
    denominator, which makes the cross-attention output a strong function of the token COUNT.

    Teacher and student tokenize the same caption into different numbers of real tokens, so a
    mean over real tokens (the obvious choice) and the attention term pull in different
    directions and cannot both be satisfied. Dividing by the shared padded length keeps the two
    terms consistent.
    """
    mask = mask.unsqueeze(-1).to(x.dtype)
    return (x * mask).sum(dim=1) / length


def setup_distributed():
    """Initialise torch.distributed from the launcher's env vars, otherwise run single process.

    Deliberately launcher-agnostic. Both `deepspeed` and `torchrun` export RANK, LOCAL_RANK and
    WORLD_SIZE, and those are the only variables read here, so either one drives this script
    through the same path. Nothing launcher-specific belongs in this function.

    The launcher is a separate choice from the parallelism strategy; see build_strategy.
    """
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        return 0, 1, 0
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    if not dist.is_initialized():
        # A generous timeout, because the final barrier waits on rank 0 writing a full
        # model checkpoint: save_full_model re-reads the entire teacher and writes a multi-GB
        # file, which on networked storage can exceed the 10-minute default and abort a job
        # that had already finished training.
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size,
                                timeout=datetime.timedelta(hours=1))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


DISTRIBUTED_STRATEGIES = ('ddp', 'zero1', 'zero2')

# ZeRO 1 and 2 partition the optimizer state across ranks: deepspeed.initialize replaces the
# client optimizer's param_groups with this rank's flat fp32 partition, so optimizer.state_dict()
# afterwards describes a shard, not the whole thing. DDP never touches the optimizer, so rank 0's
# copy is complete and is the only one worth writing.
SHARDED_STATE_STRATEGIES = ('zero1', 'zero2')

def build_optimizer(config, refiner, is_main):
    """Build the optimizer from an [optimizer] table, or from the legacy flat [distill] keys.

    The names come from utils/optimizer_factory, the same table train.py uses, so
    `type = 'adamw8bit'` means here what it means everywhere else. That matters more at this
    stage than it looks: the refiner is 77.64M parameters, so AdamW's two fp32 moments are
    592 MB, and the 8-bit variants cut that to 148 MB -- on a single GPU, with no extra ranks.

    Without an [optimizer] table the behaviour is exactly what it was before this existed:
    torch.optim.AdamW read from lr / betas / weight_decay under [distill].
    """
    distill_config = config['distill']
    flat_keys = [k for k in ('lr', 'betas', 'weight_decay') if k in distill_config]
    optim_config = dict(config.get('optimizer', {}))

    if optim_config and flat_keys:
        # Not merged, because merging silently is worse than either option. An [optimizer] table
        # holding only `type` would inherit lr but take torch's defaults for betas and
        # weight_decay -- the user's configured values would vanish with nothing said. Same
        # rule load_captions applies to its three caption sources: set exactly one.
        raise RuntimeError(
            f"[distill] sets {', '.join(flat_keys)} and an [optimizer] table is also present. "
            'They are alternative ways to configure the same optimizer, so set exactly one: '
            f"move {', '.join(flat_keys)} into [optimizer], or drop the [optimizer] table. "
            '(warmup_steps, lr_scheduler and max_grad_norm stay under [distill] either way -- '
            'they configure the schedule and clipping, not the optimizer.)'
        )

    if not optim_config:
        optim_config = {
            'type': 'adamw',
            'lr': distill_config.get('lr', 1e-4),
            'betas': list(distill_config.get('betas', [0.9, 0.99])),
            'weight_decay': distill_config.get('weight_decay', 0.01),
        }
    optim_config.setdefault('type', 'adamw')
    optim_config.setdefault('lr', 1e-4)

    if optim_config.pop('gradient_release', False):
        raise RuntimeError(
            'gradient_release is not available here. It steps the optimizer from a '
            'post-accumulate-grad hook and requires a data-parallel world size of 1, which is '
            'incompatible with the DDP and ZeRO strategies this script uses.'
        )
    if optim_config['type'].lower() == 'genericoptim':
        raise RuntimeError(
            'genericoptim needs a pipeline mpu and per-parameter-group splitting that only '
            'train.py builds. Use adamw, adamw8bit, adamw8bitkahan, adamw_optimi, stableadamw '
            'or automagic here.'
        )

    klass, args, kwargs = resolve_optimizer_class(optim_config)

    quantised = optim_config['type'].lower() in ('adamw8bit', 'adamw8bitkahan')
    if quantised and str(distill_config.get('distributed_strategy', 'ddp')).lower() != 'ddp' and is_main:
        print(
            'WARNING: an 8-bit optimizer under ZeRO is untested. ZeRO replaces the optimizer\'s '
            'param groups with its own flat fp32 partitions after deepspeed.initialize, and the '
            'quantisation blocks are then laid out over those partitions rather than over whole '
            'parameters. It may well be fine; it has not been verified. Note also that the two '
            'save the same memory in different places, so combining them buys less than the sum '
            'of their separate figures. Prefer one or the other until this is checked.'
        )

    # Weight decay on 1-d parameters -- LayerNorm/RMSNorm gains and biases -- shrinks them
    # toward zero for no benefit. train.py splits them into a no-decay group; this does too
    # when asked. Off by default so an existing run's numbers do not move underneath it.
    if distill_config.get('no_weight_decay_on_1d', False) and kwargs.get('weight_decay', 0) > 0:
        decay = [p for p in refiner.parameters() if p.ndim > 1]
        no_decay = [p for p in refiner.parameters() if p.ndim == 1]
        param_groups = [{'params': decay}, {'params': no_decay, 'weight_decay': 0.0}]
        if is_main:
            print(f'Weight decay applied to {len(decay)} tensors, skipped on {len(no_decay)} 1-d tensors')
        return klass(param_groups, *args, **kwargs)

    return klass(refiner.parameters(), *args, **kwargs)


def build_lr_scheduler(config, optimizer, steps):
    """Use the repo's shared scheduler rather than a private cosine.

    utils/lr_schedule is deliberately free of training-stack imports so it can be used outside
    train.py; this stage hand-rolling its own cosine was duplication, and it limited the choice
    to cosine when constant, linear and cosine_with_restarts were already sitting there.
    """
    return create_lr_scheduler(
        optimizer,
        config['distill'].get('lr_scheduler', 'cosine'),
        total_steps=steps,
        warmup_steps=config['distill'].get('warmup_steps', 500),
        num_cycles=config['distill'].get('lr_scheduler_num_cycles', 1),
    )


PRECISIONS = ('fp32', 'bf16-mixed', 'fp16-mixed', 'bf16-full')


class Precision:
    """How the *trainable* refiner is stored and computed. Orthogonal to [distill] dtype.

    Worth separating carefully, because the config already has a `dtype` and it is easy to read
    it as "the precision this trains in". It is not. `dtype` applies only to the frozen modules
    -- both LLMs, the LLMAdapter, the cross-attention probes -- while the refiner has always
    been fp32 regardless. This class is the knob for the refiner itself.

    Three axes, and the reason they are one object rather than three settings: they are not
    independently valid. fp16 without a loss scaler underflows; a scaler on top of DeepSpeed's
    own loss scaling double-scales; autocast over bf16 parameters is a no-op that reads as if
    it were doing something.
    """

    def __init__(self, name, param_dtype, autocast_dtype, needs_scaler, deepspeed_section, note):
        self.name = name
        self.param_dtype = param_dtype
        self.autocast_dtype = autocast_dtype
        self.needs_scaler = needs_scaler
        self.deepspeed_section = deepspeed_section
        self.note = note

    def autocast(self, device_type):
        if self.autocast_dtype is None or device_type != 'cuda':
            # autocast is a CUDA/CPU-specific context and buys nothing on CPU here; keeping it
            # off there means a CPU smoke run exercises the same numerics it always did.
            return contextlib.nullcontext()
        return torch.autocast(device_type, dtype=self.autocast_dtype)


def resolve_precision(config):
    """Map the `precision` setting onto parameter storage, autocast and DeepSpeed config.

    The mapping depends on the strategy, because DeepSpeed will not share the job with
    torch.amp: under ZeRO the engine owns loss scaling and the fp32 master weights, so the
    same user-facing name has to be implemented through the engine's config instead of through
    autocast and a GradScaler. The observable behaviour is the same; the mechanism is not, and
    pretending otherwise is how double-scaling bugs get written.
    """
    name = str(config['distill'].get('precision', 'fp32')).lower()
    strategy = str(config['distill'].get('distributed_strategy', 'ddp')).lower()
    zero = strategy.startswith('zero')

    if name == 'fp16-full':
        raise RuntimeError(
            "precision='fp16-full' is refused. Pure fp16 parameters with no fp32 master copy "
            'lose AdamW updates to underflow -- an update smaller than about 6e-8 relative to '
            'the weight rounds to nothing, and at this learning rate most of them are. Use '
            "'fp16-mixed' (fp16 compute, fp32 master weights) or 'bf16-full'."
        )
    if name not in PRECISIONS:
        raise RuntimeError(f'precision={name!r} under [distill] is not one of {PRECISIONS}.')

    if name == 'fp32':
        return Precision(name, torch.float32, None, False, {},
                         'refiner parameters and compute in fp32')

    if name == 'bf16-mixed':
        # fp32 parameters, bf16 compute. Saves activation memory, not parameter memory, and
        # needs no loss scaler: bf16 has fp32's exponent range.
        return Precision(name, torch.float32, torch.bfloat16, False, {},
                         'fp32 refiner parameters, bf16 compute via autocast')

    if name == 'bf16-full':
        if zero:
            # DeepSpeed's bf16 mode keeps fp32 master weights in the optimizer, so this saves
            # less parameter memory than the DDP path below -- and is numerically better. The
            # asymmetry is real; see docs/anima_refiner/README.md.
            return Precision(name, torch.float32, None, False, {'bf16': {'enabled': True}},
                             'bf16 refiner compute, fp32 master weights held by the ZeRO engine')
        return Precision(name, torch.bfloat16, None, False, {},
                         'refiner parameters and compute in bf16, no master weights')

    # fp16-mixed
    if zero:
        # loss_scale 0 selects DeepSpeed's dynamic scaler. A torch.amp GradScaler on top of
        # this would scale the loss twice.
        return Precision(name, torch.float32, None, False,
                         {'fp16': {'enabled': True, 'loss_scale': 0}},
                         'fp16 refiner compute with DeepSpeed dynamic loss scaling')
    return Precision(name, torch.float32, torch.float16, True, {},
                     'fp32 refiner parameters, fp16 compute via autocast + GradScaler')


class DDPStrategy:
    """Plain DistributedDataParallel with a hand-rolled accumulation loop. The default.

    Every rank keeps a full copy of the optimizer state. For this stage that is the right
    trade: the teacher, both LLMs and the cross-attention probes are frozen, so the only thing
    carrying optimizer state is the refiner (77.64M params -> 1.24 GB of AdamW state, fp32
    master weights and gradients, or 0.93 GB per GPU saved across four if it were sharded),
    while the ~5.20 GB of frozen bf16 LLM weights that actually dominate residency is exactly
    what ZeRO-1/2 cannot touch. See docs/anima_refiner/README.md for the full arithmetic.
    """

    name = 'ddp'

    def __init__(self, refiner, world_size, local_rank, device, grad_accum, optimizer, scheduler,
                 max_grad_norm, precision):
        self.refiner = refiner
        self.world_size = world_size
        self.grad_accum = grad_accum
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.precision = precision
        # Enabled only for fp16-mixed. bf16 needs no scaler (fp32's exponent range) and fp32
        # obviously not; an always-on scaler would just be a no-op that looks load-bearing.
        self.scaler = torch.amp.GradScaler('cuda', enabled=precision.needs_scaler)
        if world_size > 1:
            self.module = torch.nn.parallel.DistributedDataParallel(
                refiner,
                device_ids=[local_rank] if device.type == 'cuda' else None,
                output_device=local_rank if device.type == 'cuda' else None,
            )
        else:
            self.module = refiner

    def micro_batch_context(self, is_last):
        """DDP all-reduces on every backward by default. Only the last micro batch pays for
        that; the others just accumulate into .grad locally."""
        if self.world_size == 1 or is_last:
            return contextlib.nullcontext()
        return self.module.no_sync()

    def backward(self, loss):
        # Scale so the gradient matches one batch of batch_size * grad_accum, rather than
        # growing with the number of micro batches. (The ZeRO engine does this internally,
        # which is why DeepSpeedZeROStrategy.backward must not repeat it.)
        grad_accum = self.grad_accum
        self.scaler.scale(loss / grad_accum).backward()

    def scale_side_branch(self, tensor):
        """No-op: dividing the whole loss above already covers every branch of it."""
        return tensor

    def step(self):
        # Unscale before clipping, or max_grad_norm would be compared against a gradient
        # inflated by the loss scale and would effectively never trigger.
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.refiner.parameters(), self.max_grad_norm)
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # GradScaler.step() skips the update when it finds inf or nan in the gradients, and
        # update() lowers the scale when it does. Advancing the schedule anyway would walk the
        # LR curve further than the number of updates that actually happened. DeepSpeed gates
        # its own scheduler on exactly this condition, so leaving it ungated here made the two
        # strategies disagree under the same `precision` setting.
        if not self.scaler.is_enabled() or self.scaler.get_scale() >= scale_before:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(grad_norm)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def last_lr(self):
        return self.scheduler.get_last_lr()[0]


class DeepSpeedZeROStrategy:
    """Opt-in ZeRO stage 1 or 2 through deepspeed.initialize.

    Off by default, because the arithmetic in DDPStrategy says it buys about 0.93 GB per GPU on
    four ranks and cannot touch the frozen weights that dominate. It is here for the cases where
    that 0.93 GB is what stands between a run starting and OOMing: a large `batch_size`, a wider
    student LLM, or a deeper refiner all raise the trainable share and shift the balance.

    Stage 3 is refused rather than supported. It would shard the frozen weights too, but it
    all-gathers them on every forward pass -- paying bandwidth to save memory that is not the
    constraint here -- and it leaves refiner.state_dict() holding shards rather than weights, so
    the save path would need a gather that exists for no benefit.

    DeepSpeed owns accumulation, clipping and stepping once its config carries
    gradient_accumulation_steps and gradient_clipping: engine.backward applies the 1/N scaling
    internally and engine.step is a no-op except on an accumulation boundary. So this class does
    NOT rescale the loss and does NOT clip by hand -- doing either on top of the engine would
    double-apply it.

    The engine does NOT work out that boundary for itself under this loop, which is the one
    thing here that has to be driven by hand. DeepSpeed advances micro_steps inside step(), not
    inside backward(), and derives the boundary from it; a loop that calls backward() N times
    and step() once therefore advances the counter once per OUTER step, and the boundary lands
    every Nth outer step instead of every Nth micro batch. The measured effect with N=2 was
    half the configured optimizer updates, an LR schedule that never finished, and gradients
    reduced across ranks on only one outer step in two. set_gradient_accumulation_boundary is
    DeepSpeed's documented remedy for exactly this call shape, and it must be set before each
    forward/backward, which is why micro_batch_context does it.
    """

    name = 'zero'

    def __init__(self, refiner, stage, world_size, batch_size, grad_accum, optimizer, scheduler,
                 max_grad_norm, precision):
        import deepspeed

        if stage not in (1, 2):
            raise RuntimeError(
                f'distributed_strategy asks for ZeRO stage {stage}; only 1 and 2 are supported '
                'here. Stage 3 shards the frozen weights and all-gathers them every forward '
                'pass, which trades bandwidth for memory that is not the bottleneck at this '
                'stage. See docs/anima_refiner/README.md.'
            )
        if world_size < 2 or not dist.is_initialized():
            # Sharding across one rank shards nothing, so this is always a mistake rather than
            # a degraded mode. Catch it here: left to DeepSpeed, a missing process group sends
            # it down its MPI discovery path and it dies on `No module named 'mpi4py'`, which
            # says nothing about the actual problem.
            raise RuntimeError(
                f'distributed_strategy asks for ZeRO but world_size={world_size}. ZeRO shards '
                'optimizer state across ranks, so on a single process it costs a dependency '
                'and saves nothing. Launch with `deepspeed --num_gpus=N` (N > 1), or leave '
                'distributed_strategy at its default of ddp.'
            )
        if precision.needs_scaler:
            raise RuntimeError(
                'a torch.amp GradScaler cannot drive a DeepSpeed engine, which does its own '
                'loss scaling. resolve_precision is meant to route fp16-mixed through the '
                "engine's fp16 config under ZeRO; reaching here means that routing broke."
            )
        self.stage = stage
        self.grad_accum = grad_accum
        self.precision = precision
        # Size the communication buckets to the model. DeepSpeed defaults both to 5e8
        # ELEMENTS, which contiguous_gradients turns into a 2 GB fp32 buffer and overlap_comm
        # into a second one -- several GB spent to save the 0.93 GB the docstring above quotes
        # as the whole reason for using ZeRO here. A bucket only has to be large enough to
        # pipeline communication, never larger than the gradients it carries.
        trainable = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
        bucket = max(1, min(2 * 10 ** 7, trainable))
        ds_config = {
            'train_micro_batch_size_per_gpu': batch_size,
            'gradient_accumulation_steps': grad_accum,
            'gradient_clipping': max_grad_norm,
            'zero_optimization': {
                'stage': stage,
                'contiguous_gradients': True,
                'overlap_comm': True,
                'reduce_bucket_size': bucket,
                'allgather_bucket_size': bucket,
            },
            'zero_allow_untested_optimizer': True,
            'steps_per_print': 10 ** 9,
            'wall_clock_breakdown': False,
            # Empty for fp32 (the default), which is what keeps the engine from casting the
            # refiner's master weights out from under the optimizer.
            **precision.deepspeed_section,
        }
        # The process group already exists -- setup_distributed built it from the launcher's
        # env vars, and it is the same group either launcher exports. Reuse it.
        self.engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=refiner,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=ds_config,
            dist_init_required=False,
        )
        self.module = self.engine

    def micro_batch_context(self, is_last):
        # Tell the engine whether this micro batch closes the accumulation window. It has to be
        # set before the forward, which is what this context manager wraps. Without it the
        # engine infers the boundary from a counter that only advances in step(), so most
        # optimizer updates never happen -- see the class docstring.
        self.engine.set_gradient_accumulation_boundary(is_last)
        return contextlib.nullcontext()

    def backward(self, loss):
        self.engine.backward(loss)

    def scale_side_branch(self, tensor):
        """Apply the 1/N accumulation scaling to a tensor that did not come from the engine.

        DeepSpeed does not divide inside backward(). engine.py:2490 computes gas_scaled_loss
        only as a return value; the real division is a hook the engine registers on the OUTPUT
        of its own forward (engine.py:2237-2243 -> _backward_prologue_per_tensor). So a forward
        that deliberately bypasses the engine -- the unconditional rollout branch, which calls
        the bare refiner to avoid building a second backward hook manager -- also bypasses the
        only place the scaling happens, and contributes grad_accum times its intended gradient.
        Measured on a real engine at grad_accum=4: the engine path lands at 1.0x a single
        un-accumulated batch, the bypassing path at 4.0x.
        """
        if self.grad_accum > 1 and tensor.requires_grad:
            tensor.register_hook(lambda grad, n=self.grad_accum: grad / n)
        return tensor

    def step(self):
        self.engine.step()
        grad_norm = self.engine.get_global_grad_norm()
        # None until the first accumulation boundary completes.
        return float(grad_norm) if grad_norm is not None else 0.0

    def zero_grad(self):
        # engine.step() zeroes on the boundary; calling it from outside would drop gradients
        # mid-accumulation.
        pass

    def last_lr(self):
        return self.scheduler.get_last_lr()[0]


def validate_config_early(config, world_size, is_main):
    """Refuse an unusable config before anything expensive happens.

    Every refusal in this script used to fire after load_captions had walked the whole dataset
    and both LLMs plus the teacher DiT were resident -- minutes and tens of GB of I/O per rank
    before being told about a typo in `precision`. The checks themselves are pure config, so
    there is no reason for them to wait.

    Returns the resolved Precision, since resolving it is most of the work.
    """
    strategy = str(config['distill'].get('distributed_strategy', 'ddp')).lower()
    if strategy not in DISTRIBUTED_STRATEGIES:
        raise RuntimeError(
            f'distributed_strategy={strategy!r} under [distill] is not one of '
            f'{DISTRIBUTED_STRATEGIES}.'
        )
    if strategy != 'ddp':
        stage = int(strategy[-1])
        if stage not in (1, 2):
            raise RuntimeError(
                f'distributed_strategy asks for ZeRO stage {stage}; only 1 and 2 are supported. '
                'See docs/anima_refiner/README.md.'
            )
        if world_size < 2:
            raise RuntimeError(
                f'distributed_strategy asks for ZeRO but world_size={world_size}. ZeRO shards '
                'optimizer state across ranks, so on a single process it costs a dependency and '
                'saves nothing. Launch with `deepspeed --num_gpus=N` (N > 1), or leave '
                'distributed_strategy at its default of ddp.'
            )

    # Raises on an unknown name and on fp16-full.
    precision = resolve_precision(config)

    # The optimizer table's shape, without building anything.
    optim_config = config.get('optimizer', None)
    if optim_config is not None:
        flat = [k for k in ('lr', 'betas', 'weight_decay') if k in config['distill']]
        if flat:
            raise RuntimeError(
                f'[optimizer] is an alternative to the flat {flat} under [distill], not an '
                'addition to them. Remove one or the other.'
            )
        if optim_config.get('gradient_release', False):
            raise RuntimeError(
                'gradient_release is refused here: it needs pipeline machinery only train.py '
                'builds.'
            )
        if str(optim_config.get('type', '')).lower() == 'genericoptim':
            raise RuntimeError(
                'genericoptim is refused here: it needs the 2-d/other parameter split only '
                'train.py builds.'
            )
        if (str(optim_config.get('type', '')).lower() == 'offload'
                and config['distill'].get('no_weight_decay_on_1d', False)):
            # CPUOffloadOptimizer takes the inner class positionally and historically rejects
            # parameter groups, which is exactly what the weight-decay split produces.
            raise RuntimeError(
                "optimizer type 'offload' cannot be combined with no_weight_decay_on_1d: the "
                'weight-decay split passes parameter groups, which CPUOffloadOptimizer does not '
                'accept. Use one or the other.'
            )
    if precision.name == 'bf16-full' and strategy == 'ddp' and is_main:
        # bf16 parameters with no master copy, and -- because the cast happens before the
        # optimizer is built -- bf16 Adam moments too. That is the same failure fp16-full is
        # refused for, with fewer mantissa bits (8 against fp16's 11): an update small enough
        # relative to the weight rounds to nothing, and nothing reports it. Not refused, because
        # it is a legitimate way to fit a larger batch and the shipped 4-GPU configs use it --
        # but they pair it with a Kahan-compensated optimizer, which is what makes it safe.
        optim_type = str((optim_config or {}).get('type', 'adamw')).lower()
        if 'kahan' not in optim_type:
            print(
                f"WARNING: precision='bf16-full' with optimizer type={optim_type!r} keeps the "
                'refiner parameters AND the Adam moments in bf16 with no fp32 master copy, so '
                'small updates round away and the loss barely moves. Use '
                "type = 'adamw8bitkahan' (Kahan summation compensates for exactly this), or "
                "precision = 'bf16-mixed', which keeps fp32 parameters."
            )
    if is_main:
        print(f'Config validated: strategy={strategy}, precision={precision.name}')
    return precision


def build_strategy(config, refiner, world_size, local_rank, device, batch_size, grad_accum,
                   optimizer, scheduler, precision):
    """Pick the parallelism strategy. DDP unless the config asks for ZeRO.

    This is independent of how the job was launched: `deepspeed --num_gpus=4` can drive DDP and
    `torchrun --nproc_per_node=4` can drive ZeRO. The launcher supplies the process group; the
    strategy decides what is sharded inside it.
    """
    strategy = config['distill'].get('distributed_strategy', 'ddp').lower()
    if strategy not in DISTRIBUTED_STRATEGIES:
        raise RuntimeError(
            f'distributed_strategy={strategy!r} under [distill] is not one of '
            f'{DISTRIBUTED_STRATEGIES}.'
        )
    max_grad_norm = config['distill'].get('max_grad_norm', 1.0)
    if strategy == 'ddp':
        return DDPStrategy(refiner, world_size, local_rank, device, grad_accum, optimizer,
                           scheduler, max_grad_norm, precision)
    return DeepSpeedZeROStrategy(refiner, int(strategy[-1]), world_size, batch_size, grad_accum,
                                 optimizer, scheduler, max_grad_norm, precision)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='Path to the TOML config.')
    # Accepted but unused: deepspeed's default launcher appends --local_rank=N to every spawned
    # process's argv (a legacy PyTorch DDP convention) in addition to setting the LOCAL_RANK
    # env var, so parsing has to tolerate it. setup_distributed() stays launcher-agnostic and
    # reads LOCAL_RANK from the environment instead, since torchrun does not inject this flag.
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Unused; accepted so deepspeed\'s default launcher does not fail argument parsing.')
    args = parser.parse_args()
    config = toml.load(args.config)

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    configured_device = config['distill'].get('device', None)
    if configured_device is not None:
        device = torch.device(configured_device)
    elif torch.cuda.is_available():
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    dtype = getattr(torch, config['distill'].get('dtype', 'bfloat16'))
    max_text_length = config['distill'].get('max_text_length', MAX_TEXT_LENGTH_DEFAULT)
    if max_text_length != MAX_TEXT_LENGTH_DEFAULT and is_main:
        # The DiT's cross-attention carries no attention mask: padding is handled by padded keys
        # being exactly zero, so each pad position still contributes exp(0) = 1 to the softmax
        # denominator. The gain of every text pathway therefore scales roughly as
        # 1/(max_text_length - n). The DiT is frozen and was trained at 512, so any other value
        # amplifies or attenuates the text signal into a network calibrated for 512.
        print(
            f'\nWARNING: max_text_length = {max_text_length}, not {MAX_TEXT_LENGTH_DEFAULT}.\n'
            '  Padded keys are exactly zero and the DiT cross-attention has no mask, so every\n'
            '  padding position still adds 1 to the softmax denominator. Changing this length\n'
            '  rescales the text signal reaching a DiT that was frozen at '
            f'{MAX_TEXT_LENGTH_DEFAULT}.\n'
            '  It must match [model] max_text_length in the training config, and both should\n'
            f'  stay at {MAX_TEXT_LENGTH_DEFAULT} unless you are also unfreezing the DiT.\n'
        )
    batch_size = config['distill'].get('batch_size', 8)
    grad_accum = max(1, config['distill'].get('gradient_accumulation_steps', 1))
    # epochs and steps are alternatives. epochs is what train.py uses and what a run is
    # normally described in; steps is kept because every config written before this used it.
    epochs = config['distill'].get('epochs', None)
    steps = config['distill'].get('steps', None)
    if epochs is not None and steps is not None:
        raise RuntimeError(
            '[distill] sets both epochs and steps. They are alternative ways of saying how long '
            'to train: epochs counts passes over the caption set, steps counts optimizer '
            'updates. Set one.'
        )
    if epochs is not None and epochs < 1:
        raise RuntimeError(f'[distill] epochs must be >= 1, got {epochs}')
    if epochs is None and steps is None:
        steps = 20000
    seed = config['distill'].get('seed', 42)
    pooled_weight = config['distill'].get('pooled_loss_weight', 0.1)
    relational_weight = config['distill'].get('relational_loss_weight', 1.0)

    rollout_config = config.get('rollout', {})
    rollout_weight = rollout_config.get('loss_weight', 0.0)
    rollout_steps = rollout_config.get('steps', 8)
    rollout_points = rollout_config.get('loss_points', 2)
    rollout_resolution = rollout_config.get('resolution', 256)
    rollout_guidance = rollout_config.get('guidance_scale', 0.0)
    # Should match [model] shift in the training config: the trajectory is only a stand-in for
    # the sampler's path if it is warped the same way. 1.0 leaves it uniform.
    rollout_shift = rollout_config.get('shift', 1.0)
    for key, value in (('pooled_loss_weight', pooled_weight),
                       ('relational_loss_weight', relational_weight),
                       ('[rollout] loss_weight', rollout_weight)):
        if value < 0:
            # Each is gated on `> 0`, so a negative value silently disables the term rather
            # than doing anything. Say so instead of ignoring it.
            raise RuntimeError(
                f'{key} must be >= 0, got {value}. A negative weight silently disables the '
                'term; use 0 if that is what you meant.'
            )

    if rollout_weight > 0:
        for key, value, minimum in (('steps', rollout_steps, 1),
                                    ('loss_points', rollout_points, 1),
                                    ('resolution', rollout_resolution, 16)):
            if value < minimum:
                raise RuntimeError(f'[rollout] {key} must be >= {minimum}, got {value}')
        if rollout_shift <= 0:
            raise RuntimeError(f'[rollout] shift must be > 0, got {rollout_shift}')
        if rollout_guidance != 0 and rollout_guidance <= 1:
            # v = (1-g)*v_uncond + g*v_cond, guarded by `if guidance_scale > 0`. So 0 short
            # circuits to pure conditional while 0.001 is essentially pure UNconditional -- the
            # opposite -- and 1.0 is pure conditional again while paying a second DiT forward
            # whose gradient contribution is exactly zero. The useful range starts above 1.
            raise RuntimeError(
                f'[rollout] guidance_scale must be 0 (disabled) or greater than 1, got '
                f'{rollout_guidance}. Between 0 and 1 the guidance formula weights the '
                'unconditional branch more heavily than the conditional one, which is not what '
                'the setting is for; at exactly 1 the extra forward has no effect.'
            )
        # The VAE downsamples by 8 and the DiT patches by 2, so the latent side has to be a
        # multiple of 16 in pixels for the patch grid to come out whole.
        if rollout_resolution % 16 != 0:
            raise RuntimeError(
                f'[rollout] resolution must be a multiple of 16, got {rollout_resolution}: the '
                'VAE downsamples by 8 and the DiT patches by 2.'
            )
    output_dir = Path(config['distill']['output_dir'])
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Every rank draws different captions, so the effective batch is
    # batch_size * gradient_accumulation_steps * world_size. The model seed stays shared, so
    # the refiner and the probe are identical everywhere; only the data stream differs.
    torch.manual_seed(seed)
    random.seed(seed + rank)
    if world_size > 1 and is_main:
        print(
            f'Distributed: {world_size} ranks. Effective batch = '
            f'{batch_size} x {grad_accum} accum x {world_size} ranks = '
            f'{batch_size * grad_accum * world_size} captions per optimizer step.'
        )

    # Before load_captions walks the dataset and before any model is loaded.
    precision = validate_config_early(config, world_size, is_main)

    captions = load_captions_once(config, rank, world_size, is_main)
    if not captions:
        raise RuntimeError('No captions found. Check the dataset / captions path under [distill].')
    if is_main:
        print(f'Loaded {len(captions)} captions')
    augment = caption_augment_config(config)
    if is_main and (augment['shuffle'] or augment['tag_dropout_rate'] > 0):
        print(
            f"Caption augmentation: shuffle={augment['shuffle']} "
            f"tag_dropout_rate={augment['tag_dropout_rate']} "
            f"prefix_tag_caption={augment['prefix_tag_caption']!r}"
        )

    # The sampler and the final step count both have to exist before the optimizer does. The LR
    # scheduler is built from `steps`, and with `epochs` set `steps` is not known until the
    # sampler has worked out how many steps an epoch takes. Deriving it after the scheduler was
    # built left it holding total_steps=None, which survives construction and then raises
    # TypeError the moment warmup ends -- 500 steps into a multi-GPU run.
    # Same four keys the dataset side reads, with the same names and the same defaults, so a
    # distillation config and a dataset config say this the same way. fill_rotate_per_epoch is
    # not among them: this sampler already re-permutes on every epoch, so the tail rotates by
    # construction and there is nothing to switch on.
    distill_fill = resolve_batch_fill_config(config['distill'])
    if 'fill_rotate_per_epoch' in config['distill'] and is_main:
        print('\nWARNING: fill_rotate_per_epoch has no effect under [distill]. This sampler '
              're-permutes the whole caption list on every epoch, so which captions complete '
              'the last batch already changes from epoch to epoch.\n')
    sampler, steps, schedule_description = resolve_schedule(
        epochs, steps, captions, batch_size, grad_accum, rank, world_size, seed,
        fill_strategy=distill_fill['batch_fill_strategy'],
        undersized=distill_fill['undersized_bucket'],
        min_real_fraction=distill_fill['min_real_fraction'])
    if is_main:
        print(schedule_description)

    # With `epochs`, the step count is derived rather than written down, so a warmup longer than
    # the whole run is easy to arrive at without noticing. SequentialLR never reaches its
    # milestone in that case: the LR ramps for the entire run and the decay phase never starts.
    advice = warmup_advice(config['distill'].get('warmup_steps', 500), steps)
    if advice and is_main:
        print(advice)

    if is_main:
        print('Building teacher...')
    t5_tokenizer = T5TokenizerFast(
        vocab_file='configs/t5_old/spiece.model',
        tokenizer_file='configs/t5_old/tokenizer.json',
    )
    (teacher_tok, teacher_llm, llm_adapter, cross_attns, model_channels,
     crossattn_emb_channels, teacher_dit, teacher_dit_config) = build_teacher(config, dtype, device)

    if is_main:
        print('Building student...')
    student_tok, student_llm, refiner, cap_feat_dim = build_student(config, dtype, device, crossattn_emb_channels)
    if is_main:
        print(f'Student LLM hidden size: {cap_feat_dim}')

    # -1 asks for the last hidden state, which is what None already means -- and asking by
    # index takes the output_hidden_states branch, materialising every layer's output to then
    # throw all but one away. Every shipped anima_refiner config uses -1.
    student_hidden_layer = normalise_hidden_layer(
        config['student'].get('llm_hidden_layer', None))

    # Resolved and applied BEFORE the optimizer is built. AdamW allocates its exp_avg and
    # exp_avg_sq to match each parameter's dtype at construction time, so casting the refiner
    # afterwards would leave the optimizer holding state of the wrong dtype.
    if precision.param_dtype != torch.float32:
        refiner.to(dtype=precision.param_dtype)
    if is_main:
        print(f'Precision: {precision.name} -- {precision.note}')

    # Fixed probe queries. The cross-attention modules are frozen and shared by both paths, so
    # any query set works as a measuring stick; a fixed one keeps the objective stationary
    # across steps. Matching the output for many random queries is a strong proxy for matching
    # the key/value content itself, without ever comparing individual token positions.
    # Derived from head_dim, not hardcoded. Within one attention head the probe only ever
    # constrains the span of its own projected queries, so num_queries below head_dim leaves
    # directions of the key space that the student can fill with anything. Anima has
    # head_dim = model_channels // num_heads = 2048 // 16 = 128, so the default here is 256.
    # Measured on a reduced-scale replica, held-out probe error against num_queries/head_dim:
    # 0.25 -> 2.8x the training-probe error, 0.5 -> 1.63x, 2.0 -> 1.16x. num_blocks recovers
    # most of it (8 blocks bring 0.5 down to 1.14x) because each block projects differently,
    # but more queries are cheap and strictly help.
    # The head count comes from the checkpoint, not from a literal: get_dit_config derives it
    # from model_channels (16 at 2048, 40 at 5120, 20 at 1280), so a hardcoded 16 would size
    # head_dim -- and therefore the default num_queries -- wrongly for anything but Anima.
    head_dim = model_channels // config.get('probe', {}).get(
        'num_heads', teacher_dit_config['num_heads'])
    num_queries = config.get('probe', {}).get('num_queries', 2 * head_dim)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    probe = torch.randn(1, num_queries, model_channels, generator=generator).to(device=device, dtype=dtype)

    # The probe seed is deliberately NOT rank-offset: every rank must measure against the same
    # queries, or their gradients describe different objectives.

    # Rollout setup. Everything here is skipped entirely when the feature is off, including
    # keeping the DiT resident, so the default path is byte for byte what it was.
    rollout_generator = None
    rollout_rng = None
    teacher_uncond = None
    uncond_student_ids = uncond_student_mask = uncond_student_hidden = None
    if rollout_weight > 0:
        if teacher_dit is None:
            raise RuntimeError(
                '[rollout] loss_weight > 0 but the teacher DiT was not retained. This is a bug: '
                'build_teacher keys the decision off the same config value.'
            )
        # Rank-offset, so every rank walks a different trajectory and the effective sample count
        # scales with world size, exactly as the caption stream already does.
        rollout_generator = torch.Generator().manual_seed(seed + 10_000 + rank)
        # Its own stream: drawing loss points from the global `random` would shift the caption
        # sampling sequence for a given seed, so turning the rollout on would silently change
        # which captions a run sees.
        rollout_rng = random.Random(seed + 20_000 + rank)
        latent_side = rollout_resolution // 8
        rollout_shape = (batch_size, teacher_dit_config['in_channels'], 1, latent_side, latent_side)
        if is_main:
            graphs = rollout_points * (2 if rollout_guidance > 0 else 1)
            print(
                f'Denoising rollout: weight={rollout_weight}, steps={rollout_steps}, '
                f'loss_points={rollout_points}, latent={rollout_shape[1:]}, '
                f'shift={rollout_shift}'
                + (f', guidance={rollout_guidance}' if rollout_guidance > 0 else ', no guidance')
            )
            # The losses are summed and backward runs once, so every student forward's graph
            # stays alive until then. This is the number that decides whether raising
            # loss_points OOMs, and it is not obvious from the config.
            print(
                f'  Peak activations hold {graphs} simultaneous full-DiT backward graph(s). '
                'Raise loss_points one step at a time.'
            )
        if rollout_guidance > 0:
            teacher_uncond, uncond_student_ids, uncond_student_mask, uncond_student_hidden = (
                build_unconditional_features(
                    teacher_tok, t5_tokenizer, teacher_llm, llm_adapter,
                    student_tok, student_llm, max_text_length, device,
                    student_hidden_layer,
                )
            )

    provenance = refiner_provenance(config, cap_feat_dim, max_text_length)

    optimizer = build_optimizer(config, refiner, is_main)
    scheduler = build_lr_scheduler(config, optimizer, steps)
    if is_main:
        print(f'Optimizer: {type(optimizer).__name__} | '
              f"lr_scheduler: {config['distill'].get('lr_scheduler', 'cosine')}")

    # ZeRO partitions the optimizer state across ranks: deepspeed.initialize replaces the client
    # optimizer's param_groups with this rank's flat fp32 partition, so its state_dict describes
    # a shard and only has that shape once the engine exists. DDP leaves the optimizer alone and
    # could restore on either side of the wrap; both restore after it, so there is one order to
    # reason about and so the fp16 GradScaler exists to be restored along with everything else.
    sharded_state = config['distill'].get('distributed_strategy', 'ddp').lower() in         SHARDED_STATE_STRATEGIES
    state_rank = rank if sharded_state else None
    resume_path = config['student'].get('resume_from', None)

    # Built after the optimizer and scheduler: ZeRO wraps both, so they have to exist first.
    strategy = build_strategy(config, refiner, world_size, local_rank, device, batch_size,
                              grad_accum, optimizer, scheduler, precision)
    train_module = strategy.module
    if is_main and strategy.name != 'ddp':
        print(f'Parallelism: DeepSpeed ZeRO stage {strategy.stage} (optimizer state sharded).')

    # Every tag this process writes, so the prune never deletes one of them. Tags are ordered
    # by number, which does not increase across runs sharing an output_dir.
    tags_written_here = set()

    start_step = 0
    if resume_path:
        start_step = load_training_state(
            resume_path, optimizer, scheduler, is_main,
            rank=state_rank, world_size=world_size,
            own_python_rng=sharded_state or is_main,
            rollout_generator=rollout_generator, rollout_rng=rollout_rng,
            scaler=getattr(strategy, 'scaler', None),
            batch_size=batch_size, grad_accum=grad_accum, precision_name=precision.name,
            batch_fill=distill_fill,
        )
        if start_step >= steps:
            raise RuntimeError(
                f'{resume_path} was already trained for {start_step} steps and this config asks '
                f'for {steps}. Raise steps, or point resume_from at an earlier checkpoint.'
            )
        if not (sharded_state or is_main):
            # A DDP checkpoint carries rank 0's `random` stream only. Re-offset the other ranks
            # rather than leaving them all on rank 0's, which would correlate the caption
            # augmentation across the job for the rest of the run.
            random.seed(seed + rank + start_step)

    save_every = config['distill'].get('save_every', 2000)
    log_every = config['distill'].get('log_every', 50)
    for key, value in (('save_every', save_every), ('log_every', log_every)):
        if value < 1:
            # Both are used as `(step + 1) % value`. At 0 that is a ZeroDivisionError on rank 0
            # only, so the other ranks sit at the next barrier until the watchdog kills the job
            # -- a hang, not an error message.
            raise RuntimeError(
                f'[distill] {key} must be >= 1, got {value}. To effectively disable it, set it '
                'larger than `steps`.'
            )
    save_full_model_enabled = config['distill'].get('save_full_model', False)
    keep_last_n = config['distill'].get('keep_last_n_checkpoints', None)
    if keep_last_n is not None and keep_last_n < 1:
        raise RuntimeError(
            f'[distill] keep_last_n_checkpoints must be >= 1, got {keep_last_n}. Omit it to keep '
            'every checkpoint.'
        )

    save_every_n_epochs = config['distill'].get('save_every_n_epochs', None)
    if save_every_n_epochs is not None:
        if save_every_n_epochs < 1:
            raise RuntimeError(
                f'[distill] save_every_n_epochs must be >= 1, got {save_every_n_epochs}')
        save_every = save_every_n_epochs * sampler.steps_per_epoch
        if is_main:
            print(f'Saving every {save_every_n_epochs} epoch(s) = every {save_every} steps')

    running = 0.0
    last_spread = last_teacher_spread = 0.0
    last_terms = {}
    progress_bar = tqdm(range(start_step, steps), initial=start_step, total=steps,
                        desc='distill', disable=not is_main)

    strategy.zero_grad()
    # Resuming mid-epoch resumes at the right point in the right epoch's shuffle, because the
    # order is a pure function of (seed, epoch) rather than of how many draws have happened.
    current_epoch = start_step // sampler.steps_per_epoch
    epoch_captions = sampler.epoch_order(current_epoch)
    for step in progress_bar:
        epoch = step // sampler.steps_per_epoch
        if epoch != current_epoch:
            current_epoch = epoch
            epoch_captions = sampler.epoch_order_weighted(epoch)
        offset = (step % sampler.steps_per_epoch) * grad_accum * batch_size

        accum_loss = 0.0
        strategy.zero_grad()
        for micro in range(grad_accum):
            start = offset + micro * batch_size
            entries = epoch_captions[start:start + batch_size]
            batch = [preprocess_caption(c, **augment) for c, _ in entries]
            # None whenever nothing in this global batch is padding, which is every batch under
            # 'drop' and every batch but the last under 'fill'. weighted_mse then takes the
            # plain F.mse_loss path, so the ordinary run is not merely equivalent, it is the
            # same call.
            sample_weights = None
            real_index = None
            if any(w != 1.0 for _, w in entries):
                sample_weights = torch.tensor([w for _, w in entries], device=device, dtype=torch.float32)
                real_index = sample_weights > 0

            with torch.no_grad():
                t_enc = teacher_tok(batch, return_tensors='pt', truncation=True, padding='max_length', max_length=max_text_length)
                t5_enc = t5_tokenizer(batch, return_tensors='pt', truncation=True, padding='max_length', max_length=max_text_length)
                teacher_hidden = encode(
                    teacher_llm, t_enc.input_ids.to(device), t_enc.attention_mask.to(device), None
                )
                teacher_feats = llm_adapter(
                    source_hidden_states=teacher_hidden,
                    target_input_ids=t5_enc.input_ids.to(device),
                    target_attention_mask=t5_enc.attention_mask.to(device),
                    source_attention_mask=t_enc.attention_mask.to(device),
                )
                t5_mask = t5_enc.attention_mask.to(device)
                teacher_feats = teacher_feats * t5_mask.unsqueeze(-1).to(teacher_feats.dtype)

                # keep_one_real_token, like every other path that feeds the refiner. A caption
                # that tag_dropout_rate reduced to '' tokenizes to an all-padding row, which the
                # refiner zeroes -- no gradient from that sample, and a degenerate rollout point.
                s_enc = cosmos_tokenize(student_tok, batch, max_text_length,
                                        keep_one_real_token=True)
                s_mask = s_enc['attention_mask'].to(device)
                student_hidden = encode(
                    student_llm, s_enc['input_ids'].to(device), s_mask,
                    student_hidden_layer
                )

            # Whether this micro batch synchronises gradients is the strategy's business: DDP
            # suppresses all-reduce until the last one, the ZeRO engine tracks the boundary
            # itself.
            with strategy.micro_batch_context(is_last=(micro == grad_accum - 1)):
                # The input must match the refiner's parameter dtype: under 'bf16-full' the
                # parameters are bf16 and an fp32 input is a hard dtype error, not a silent
                # upcast. Autocast, where it is on, then handles the compute dtype.
                with precision.autocast(device.type):
                    student_feats = train_module(student_hidden.to(precision.param_dtype), s_mask)

                q = probe.expand(len(batch), -1, -1)
                loss = 0.0
                # Tracked separately because only the sum was ever logged, and the terms are on
                # different scales -- the probe is an MSE over cross-attention outputs, the
                # rollout an MSE over velocities. If one silently dominates, the others stop
                # training the refiner and the summed number does not show it.
                terms = {}
                for cross_attn in cross_attns:
                    with torch.no_grad():
                        target = cross_attn(q, context=teacher_feats.to(dtype))
                    pred = cross_attn(q, context=student_feats.to(dtype))
                    loss = loss + weighted_mse(pred.float(), target.float(), sample_weights)
                loss = loss / len(cross_attns)
                terms['probe'] = float(loss.detach())

                # Auxiliary global term. Also permutation invariant, and it gives a useful
                # gradient early on while the probe-attention term is still dominated by noise.
                student_pooled = padded_mean(student_feats.float(), s_mask, max_text_length)
                teacher_pooled = padded_mean(teacher_feats.float(), t5_mask, max_text_length)
                if pooled_weight > 0:
                    pooled_term = pooled_weight * weighted_mse(student_pooled, teacher_pooled, sample_weights)
                    terms['pooled'] = float(pooled_term.detach())
                    loss = loss + pooled_term
                if relational_weight > 0:
                    # Keeps distinct captions distinct. See relational_loss.
                    #
                    # Padding is REMOVED here rather than weighted to zero, and the difference
                    # matters more than anywhere else in this loop. A masked repeat is a copy
                    # of a caption already in the batch, so it contributes a pairwise distance
                    # of exactly zero -- and this term's whole job is to punish distances that
                    # collapse toward zero. Left in at weight zero its own term would vanish
                    # but the teacher's matching zero distance would still be there to be
                    # matched, teaching the model that two identical captions belong on top of
                    # each other, which is true but is also the shape of the collapse this
                    # exists to prevent. No scale is applied either: this is already a mean
                    # over real pairs, so it never suffered the dilution G/G_real undoes.
                    if real_index is not None:
                        relational_term = relational_weight * relational_loss(
                            student_pooled[real_index], teacher_pooled[real_index])
                    else:
                        relational_term = relational_weight * relational_loss(student_pooled, teacher_pooled)
                    terms['relational'] = float(relational_term.detach())
                    loss = loss + relational_term
                if rollout_weight > 0:
                    student_uncond = None
                    if rollout_guidance > 0:
                        # With gradient: the unconditional branch depends on the refiner too,
                        # and it is the branch every CFG sample actually uses.
                        #
                        # Through `refiner`, not `train_module`, deliberately. A second call
                        # into a DeepSpeed engine within one backward builds a second backward
                        # hook manager and fires _backward_prologue twice, which leaks
                        # _backward_active_depth on every micro-batch and would trip the timer
                        # assertion if wall_clock_breakdown were ever enabled. The parameters
                        # are shared either way, so the gradients still reduce correctly. ZeRO
                        # stage 3 would break this, and stage 3 is refused.
                        #
                        # Under the same autocast as the conditional forward, or this branch
                        # silently runs in fp32 while its sibling honours `precision`.
                        with precision.autocast(device.type):
                            student_uncond = refiner(
                                uncond_student_hidden.to(precision.param_dtype),
                                uncond_student_mask,
                            )
                        # Bypassing the engine also bypasses the engine's 1/N accumulation
                        # scaling, so ask the strategy to put it back. No-op under DDP, which
                        # divides the whole loss instead.
                        student_uncond = strategy.scale_side_branch(student_uncond)
                        student_uncond = student_uncond.to(dtype).expand(len(batch), -1, -1)
                    visited = teacher_trajectory(
                        teacher_dit, teacher_feats.to(dtype),
                        teacher_uncond.expand(len(batch), -1, -1).to(dtype) if teacher_uncond is not None else None,
                        (len(batch),) + tuple(rollout_shape[1:]), rollout_steps,
                        rollout_guidance, rollout_generator, device, dtype, rollout_shift,
                    )
                    rollout_term = rollout_weight * rollout_loss(
                        teacher_dit, visited, teacher_feats.to(dtype), student_feats.to(dtype),
                        teacher_uncond.expand(len(batch), -1, -1).to(dtype) if teacher_uncond is not None else None,
                        student_uncond, rollout_guidance, rollout_points, rollout_rng,
                        weights=sample_weights,
                    )
                    terms['rollout'] = float(rollout_term.detach())
                    loss = loss + rollout_term
                    del visited
                    # MiniTrainDIT.forward assigns crossattn_emb and affline_emb onto itself.
                    # That was harmless while the DiT was thrown away; it is long-lived now, so
                    # without this it pins the last student feature tensor and its autograd
                    # graph until the next forward overwrites them.
                    teacher_dit.crossattn_emb = None
                    teacher_dit.affline_emb = None

                if is_main:
                    # Diagnostic only, and free: the pooled features already exist.
                    #
                    # Real samples only. A masked repeat sits exactly on top of the caption it
                    # was copied from, so counting it drags the mean pairwise distance down and
                    # the number reads as collapse starting -- which is the one thing this
                    # figure exists to warn about, so a false reading of it is worse than no
                    # figure at all.
                    spread_student = student_pooled.detach()
                    spread_teacher = teacher_pooled.detach().float()
                    if real_index is not None:
                        spread_student = spread_student[real_index]
                        spread_teacher = spread_teacher[real_index]
                    last_spread = mean_pairwise_cosine_distance(spread_student)
                    last_teacher_spread = mean_pairwise_cosine_distance(spread_teacher)
                    last_terms = terms

                strategy.backward(loss)
            accum_loss += loss.item() / grad_accum

        grad_norm = strategy.step()

        running += accum_loss
        if is_main and (step + 1) % log_every == 0:
            progress_bar.set_postfix({
                'loss': f'{running / log_every:.5f}',
                'grad': f'{grad_norm:.3f}',
                'lr': f'{strategy.last_lr():.2e}',
                # spread is the mean pairwise cosine distance between captions in the batch.
                # It should track the teacher's. Falling toward 0 means collapse: distinct
                # captions are being mapped to the same feature.
                'ep': f'{step // sampler.steps_per_epoch + 1}',
                'spread': f'{last_spread:.3f}/{last_teacher_spread:.3f}',
                # Each weighted term, so a dominant one is visible rather than buried in the sum.
                **{name: f'{value:.4f}' for name, value in last_terms.items()},
            })
            running = 0.0

        if (step + 1) % save_every == 0 or step + 1 == steps:
            # Tagged by whichever unit drives the saving, so the two kinds are countable apart
            # the way train.py's epoch<N>/ and step<N>/ are. Computed outside the rank guards
            # because under ZeRO every rank writes its own piece of this checkpoint.
            if save_every_n_epochs is not None:
                tag = f'_epoch{step // sampler.steps_per_epoch + 1}'
            else:
                tag = f'_step{step + 1}'
            tags_written_here.add(tag)
            tagged_path = output_dir / f'context_refiner{tag}.safetensors'
            refiner_path = output_dir / 'context_refiner.safetensors'

            # Weights: every rank holds the full, identical set -- DDP all-reduces them, and
            # ZeRO 1/2 shard optimizer state and gradients but never the parameters. One writer.
            if is_main:
                save_refiner(refiner, tagged_path, dtype,
                             metadata=dict(provenance, step=str(step + 1)))
                # And the stable name, so every config that points at
                # context_refiner.safetensors keeps working without knowing about tags.
                _copy_atomically(tagged_path, refiner_path)

                if save_full_model_enabled:
                    # A complete anima_refiner checkpoint per save, not only at the end: the
                    # point of keeping N checkpoints is being able to go back to one, and going
                    # back to a refiner without the model it belongs in is half a checkpoint.
                    full_path = output_dir / f'model{tag}.safetensors'
                    save_full_model(config['teacher']['transformer_path'], refiner,
                                    full_path, dtype)
                    _copy_atomically(full_path, output_dir / 'model.safetensors')

            # Optimizer state: whole on rank 0 under DDP, one shard per rank under ZeRO. Writing
            # only rank 0's shard produced a file that looked valid, cost nothing to write, and
            # could not be resumed -- the failure landed hours later, on the resume.
            if is_main or sharded_state:
                save_training_state(tagged_path, optimizer, scheduler, step + 1,
                                    rank=state_rank, world_size=world_size,
                                    rollout_generator=rollout_generator,
                                    rollout_rng=rollout_rng,
                                    scaler=getattr(strategy, 'scaler', None),
                                    batch_size=batch_size, grad_accum=grad_accum,
                                    precision_name=precision.name,
                                    batch_fill=distill_fill,
                                    master_weights=sharded_state and bool(precision.deepspeed_section))
                _copy_atomically(training_state_path(tagged_path, state_rank),
                                 training_state_path(refiner_path, state_rank))

            if world_size > 1:
                # Before the prune, so no rank is still writing a shard of an older tag when
                # rank 0 starts deleting the files that belong to it.
                dist.barrier()
            if is_main:
                for gone in prune_distill_checkpoints(output_dir, keep_last_n,
                                                      protect_tag=tag,
                                                      protect_tags=tags_written_here):
                    print(f'keep_last_n_checkpoints: removed {gone.name}')

    if is_main:
        # The loop's `step + 1 == steps` branch already wrote this file; no need to write the
        # same tensors again, just say where they are.
        print(f'Done. Point context_refiner_path at {output_dir / "context_refiner.safetensors"}')

    if is_main and save_full_model_enabled:
        # The loop's final save already wrote this; say where it is rather than rebuilding a
        # multi-GB file that is already on disk.
        path = output_dir / 'model.safetensors'
        print(f'Also wrote a full anima_refiner checkpoint to {path}. Use it as transformer_path.')

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def save_full_model(teacher_transformer_path, refiner, path, dtype):
    """Write a complete anima_refiner checkpoint: the teacher's DiT with the refiner attached.

    The DiT weights are unchanged, so this is exactly equivalent to passing the Anima
    checkpoint as transformer_path alongside context_refiner_path. It exists so distillation
    produces the same kind of artefact every other mode does -- a single file that is a whole
    model -- instead of a component that has to be paired up by hand.

    llm_adapter is dropped: this architecture does not build it, and leaving it in would be
    dead weight in every future checkpoint.
    """
    state_dict = {}
    for k, v in load_state_dict(teacher_transformer_path).items():
        name = k[len('net.'):] if k.startswith('net.') else k
        if name.startswith('llm_adapter.'):
            continue
        state_dict['net.' + name] = v.to(dtype).cpu().contiguous()
    for k, v in refiner.state_dict().items():
        # fp32 for the refiner, for the reason save_refiner gives: dtype describes the frozen
        # modules, the trainable refiner is fp32 regardless, and this file is a valid
        # resume_from source. Rounding it to bf16 here threw away sixteen mantissa bits on
        # every save and again on every resume from it, while the sibling artefact written in
        # the same block kept them. Mixed dtypes in one safetensors file are legal, and
        # load_diffusion_model casts on load anyway.
        state_dict['net.context_refiner.' + k] = v.detach().float().cpu().contiguous()
    _save_file_atomically(state_dict, path, {'format': 'pt'})


def _save_file_atomically(state_dict, path, metadata):
    """Write beside the target, then rename over it.

    save_every rewrites one fixed filename, so a plain in-place write destroys the last good
    checkpoint the moment it starts. An interruption there -- Ctrl-C, OOM, preemption -- leaves
    a truncated file and nothing to fall back on. os.replace is atomic on NTFS and on POSIX, so
    the target is either the old file or the new one, never half of either.
    """
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    safetensors.torch.save_file(state_dict, str(tmp), metadata=metadata)
    os.replace(tmp, path)


def _copy_atomically(src, dst):
    """Copy onto a fixed filename without ever leaving it truncated.

    The same hazard _save_file_atomically exists for. shutil.copy2 opens the destination 'wb'
    and streams, so an interruption partway through a multi-gigabyte model.safetensors leaves
    the stable name -- the one every shipped config points at -- as a truncated file, while the
    tagged copy beside it is fine.
    """
    dst = Path(dst)
    tmp = dst.with_name(dst.name + '.tmp')
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def refiner_provenance(config, cap_feat_dim, max_text_length, step=None):
    """What a refiner file has to record about the text encoder it was distilled against.

    _resolve_context_refiner can check the layer count and cap_feat_dim from the tensor shapes,
    but shapes cannot tell two 2048-wide LLMs apart, and they say nothing about which hidden
    layer was read. A refiner distilled on layer 20 and used with the default last layer has
    identical shapes and a completely different input distribution -- the last hidden state is
    post-final-RMSNorm and roughly 50x larger in RMS than a raw residual-stream layer.

    `step` pins these weights to the optimizer state saved alongside them. The two are separate
    files written by separate calls, so an interruption between them leaves weights from one
    step beside moments from another, and the resume succeeds silently -- the loss bump reads as
    ordinary resume noise. Recording it lets load_training_state refuse the mismatch.

    Written as safetensors metadata, which is plain string-to-string, so every value is a str.
    """
    metadata = {
        'format': 'pt',
        'llm_path': str(config['student'].get('llm_path', '')),
        'llm_hidden_layer': str(config['student'].get('llm_hidden_layer', '')),
        'cap_feat_dim': str(cap_feat_dim),
        'max_text_length': str(max_text_length),
    }
    if step is not None:
        metadata['step'] = str(step)
    return metadata


def save_refiner(refiner, path, dtype, metadata=None):
    """Write the refiner.

    fp32, not `dtype`. dtype describes the FROZEN modules -- both LLMs, the adapter, the probes
    -- and the trainable refiner has always been fp32 regardless, which the example config says
    in as many words. Saving through bf16 threw away sixteen mantissa bits on every periodic
    save, and again on every resume, for no benefit: the file is ~310 MB either way.
    """
    state_dict = {k: v.detach().float().cpu().contiguous() for k, v in refiner.state_dict().items()}
    _save_file_atomically(state_dict, path, metadata or {'format': 'pt'})


def training_state_path(refiner_path, rank=None):
    """The training state that belongs to this particular weights file.

    Derived from the weights filename rather than fixed, so a tagged checkpoint carries its own
    optimizer state: context_refiner_epoch5.safetensors pairs with distill_state_epoch5.pt.
    Resuming from an older tag would otherwise pick up the newest state and pair mismatched
    moments with older weights.

    `rank` names one shard of a ZeRO run's optimizer state, which is rank-local and has to be
    written by every rank. The unsuffixed name means whole state, which is what DDP produces.
    Keeping the two apart by filename is what stops a DDP checkpoint from being fed to a ZeRO
    resume, where the shapes differ and the failure is a confusing ValueError deep in torch.
    """
    path = Path(refiner_path)
    # Both names a save writes carry the same tag, and `model_epoch7.safetensors` is documented
    # as a first-class rollback target. Matching only 'context_refiner' sent it to the untagged
    # distill_state.pt -- epoch-7 weights paired with the newest moments and the newest step
    # counter, silently, which is the exact mispairing this function exists to prevent.
    for prefix in ('context_refiner', 'model'):
        if path.stem.startswith(prefix):
            suffix = path.stem[len(prefix):]
            break
    else:
        # A name this script never wrote. There is no state file to find; the untagged name is
        # the honest guess, and load_training_state reports the miss rather than inventing one.
        suffix = ''
    shard = '' if rank is None else f'_rank{rank}'
    return path.with_name(f'distill_state{suffix}{shard}.pt')


def prune_distill_checkpoints(output_dir, keep, protect_tag=None, protect_tags=()):
    """Keep the newest `keep` tagged checkpoints of each kind, with everything that belongs to them.

    Counted separately per kind, matching train.py: N epoch-tagged and N step-tagged, because the
    two are produced by different triggers at different rates. Each tag owns three files -- the
    refiner weights, its optimizer state, and the full model if save_full_model is on -- and they
    are removed together, so a surviving tag is always complete.

    The untagged names are never pruned. They are the stable ones every config points at.

    `protect_tag` is the tag this save just wrote, and `protect_tags` every tag this process has
    written. Tags are ordered by their number, which only increases within one uninterrupted run
    -- but not across runs sharing an output_dir, and not when a resume onto fewer ranks raises
    steps_per_epoch and lowers the epoch number. Protecting only the newest is not enough: a
    second run into a directory holding epoch18/19/20 with keep=3 writes epoch1 (protected, so
    nothing goes), then epoch2 -- at which point epoch1 is the lowest number present and is
    deleted, and so on. The run would end holding the previous run's three checkpoints and one
    of its own. Protecting every tag this process wrote prunes the old ones instead, which is
    what keep_last_n_checkpoints is for.
    """
    if not keep or keep < 1:
        return []
    output_dir = Path(output_dir)
    removed = []
    for kind in ('epoch', 'step'):
        tagged = []
        for path in output_dir.glob(f'context_refiner_{kind}*.safetensors'):
            number = path.stem[len(f'context_refiner_{kind}'):]
            if number.isdigit():
                tagged.append((int(number), number, path))
        def is_ours(number):
            tag = f'_{kind}{number}'
            return tag == protect_tag or tag in protect_tags

        # Anything this process wrote sorts after everything it did not, so the oldest run's
        # checkpoints are the ones pruned. Within each group the number orders them, which is
        # correct because numbers only increase inside one run.
        tagged.sort(key=lambda triple: (is_ours(triple[1]), triple[0]))
        for _, number, path in tagged[:-keep] if len(tagged) > keep else []:
            # Only the tag just written is exempt outright. This run's OLDER tags are ordinary
            # prune candidates -- a long run with keep=3 must still drop its own early
            # checkpoints, which is the whole point of the setting. protect_tags orders them,
            # it does not immunise them.
            if protect_tag is not None and f'_{kind}{number}' == protect_tag:
                continue
            companions = [
                path,
                training_state_path(path),
                output_dir / f'model_{kind}{number}.safetensors',
            ]
            # A ZeRO run writes one state shard per rank, so the tag owns N of them rather than
            # one. Globbed on the '_rank' separator rather than on the number alone: a bare
            # 'distill_state_epoch1*' would also match distill_state_epoch10.pt and delete a
            # checkpoint nine epochs newer than the one being pruned.
            companions.extend(sorted(output_dir.glob(f'distill_state_{kind}{number}_rank*.pt')))
            for victim in companions:
                if victim.exists():
                    victim.unlink()
                    removed.append(victim)
    return removed


def save_training_state(refiner_path, optimizer, scheduler, step, rank=None, world_size=1,
                        rollout_generator=None, rollout_rng=None, scaler=None,
                        batch_size=None, grad_accum=None, precision_name=None,
                        master_weights=False, batch_fill=None):
    """Save what a resume needs beyond the weights.

    Without this, resume_from restarted Adam's moments at zero and rebuilt the LR schedule from
    step 0, so a run resumed at 15,000 of 20,000 steps re-ran its warmup at peak LR and then the
    whole cosine again. That regresses the model visibly and costs more than the interruption
    did. train.py checkpoints full state for every other mode; this brings distillation in line.

    Under ZeRO every rank calls this with its own `rank`, because each holds a different shard of
    the moments and rank 0's alone is not the state. `world_size` is recorded so that a resume
    into a differently sized job is refused rather than silently loading another rank's shard.
    """
    path = training_state_path(refiner_path, rank)
    tmp = path.with_name(path.name + '.tmp')
    payload = {
        'step': step,
        'world_size': world_size,
        # Recorded so a resume can say when it will not line up with the run it continues.
        # EpochSampler.steps_per_epoch divides by batch_size * grad_accum * world_size, so
        # changing any of them moves the epoch boundary the saved `step` is counted against,
        # and with `epochs` it also moves the LR schedule's total. None means a checkpoint
        # written before this was recorded; nothing is claimed about those.
        'batch_size': batch_size,
        'grad_accum': grad_accum,
        # batch_fill_strategy moves the epoch boundary the same way the three above do:
        # steps_per_epoch rounds down under 'drop' and up under 'fill', so the saved step
        # counts a different number of captions on each side of a switch. Absent in a
        # checkpoint written before this existed, which is read as 'drop' -- what it was.
        'batch_fill_strategy': (batch_fill or {}).get('batch_fill_strategy', None),
        'undersized_bucket': (batch_fill or {}).get('undersized_bucket', None),
        # AdamW allocates exp_avg/exp_avg_sq to match the parameter dtype at construction, so
        # moments saved under one precision are the wrong dtype for another.
        'precision': precision_name,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        # Under ZeRO the engine casts the module to bf16/fp16 and keeps the only full-precision
        # copy of the weights in the optimizer's flat fp32 partition -- which deepspeed.initialize
        # installed as this optimizer's param_groups. save_refiner writes the MODULE, so it writes
        # the bit16 view; optimizer.state_dict() carries the moments but never the parameter
        # values. Without this a resumed bf16-full run restarts its masters from 8-mantissa-bit
        # values and silently discards every sub-bf16 increment since the last save.
        'master_weights': (
            [p.detach().float().cpu().clone()
             for group in optimizer.param_groups for p in group['params']]
            if master_weights else None
        ),
        # The caption order does not need saving -- EpochSampler is a pure function of
        # (seed, epoch), so resuming lands on the right captions by construction. The
        # augmentation draws do: caption shuffling and tag_dropout_rate pull from the global
        # `random` stream, so without this a resumed run sees different variants than an
        # uninterrupted one would have.
        'python_rng': random.getstate(),
        'torch_rng': torch.get_rng_state(),
    }
    # The rollout draws its timesteps and its initial noise from streams of their own, seeded
    # apart from the global one so that turning the rollout on does not shift the caption
    # augmentation. They need saving for the same reason the global stream does.
    if rollout_generator is not None:
        payload['rollout_generator'] = rollout_generator.get_state()
    if rollout_rng is not None:
        payload['rollout_rng'] = rollout_rng.getstate()
    # Only fp16-mixed has one. Restarting it at init_scale costs a handful of skipped steps
    # after every resume, which is small but is also entirely avoidable.
    if scaler is not None and scaler.is_enabled():
        payload['scaler'] = scaler.state_dict()
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_training_state(refiner_path, optimizer, scheduler, is_main, rank=None, world_size=1,
                        own_python_rng=True, rollout_generator=None, rollout_rng=None,
                        scaler=None, batch_size=None, grad_accum=None, precision_name=None,
                        batch_fill=None):
    """Restore optimizer, scheduler and step. Returns the step to resume from.

    A missing file is not an error: it is a refiner distilled before this existed, or one
    produced by another mode. Say so and start the schedule from zero, rather than pretending.

    Under ZeRO, `rank` selects this rank's shard and the file must have come from a job of the
    same size -- a shard describes one partition of a particular world, and loading a shard from
    a differently sized job pairs moments with the wrong parameters.

    `own_python_rng` says whether this file's `random` stream belongs to this rank. It does under
    ZeRO, where every rank writes its own; under DDP only rank 0 writes, and pushing rank 0's
    stream onto every rank would undo the `random.seed(seed + rank)` offset that keeps the ranks
    drawing different caption augmentations. The caller re-offsets instead.
    """
    path = training_state_path(refiner_path, rank)
    if not path.exists():
        if is_main:
            # Naming the other layout explicitly, because switching distributed_strategy between
            # runs is the ordinary way to arrive here and "no file" alone does not point at it.
            other = training_state_path(refiner_path, None if rank is not None else 0)
            switched = (
                f' A {other.name} is there instead, which is a '
                + ('DDP' if rank is not None else 'ZeRO')
                + ' checkpoint; optimizer state does not carry across that switch.'
                if other.exists() else ''
            )
            print(
                f'No {path.name} beside {Path(refiner_path).name}: resuming the weights only.'
                + switched +
                ' The optimizer moments restart at zero and the LR schedule restarts from step '
                '0, so expect a visible bump in the loss.'
            )
        return 0
    state = torch.load(path, map_location='cpu', weights_only=False)
    saved_world = state.get('world_size', 1)
    if rank is not None and saved_world != world_size:
        raise RuntimeError(
            f'{path.name} was written by a {saved_world}-rank job and this one has {world_size} '
            'ranks. ZeRO optimizer state is partitioned across ranks, so a shard from a '
            'differently sized job describes the wrong parameters. Resume with '
            f'--num_gpus={saved_world}, or drop the distill_state files to resume the weights '
            'only.'
        )
    if is_main:
        # Not refused: DDP replicates the optimizer state rather than partitioning it, so it
        # loads correctly at any world size, and resuming onto a different number of GPUs is an
        # ordinary thing to want. What does not carry across is the alignment -- the saved step
        # counts global batches, so a different global batch puts it at a different point in the
        # corpus, and under `epochs` the schedule's total moves with it. Say so and continue.
        drifted = [
            (name, state.get(key, None), current)
            for name, key, current in (('world_size', 'world_size', world_size),
                                       ('batch_size', 'batch_size', batch_size),
                                       ('gradient_accumulation_steps', 'grad_accum', grad_accum),
                                       ('batch_fill_strategy', 'batch_fill_strategy',
                                        (batch_fill or {}).get('batch_fill_strategy', None)),
                                       ('undersized_bucket', 'undersized_bucket',
                                        (batch_fill or {}).get('undersized_bucket', None)))
            if state.get(key, None) is not None and current is not None and state[key] != current
        ]
        if drifted:
            changes = ', '.join(f'{name} {was} -> {now}' for name, was, now in drifted)
            print(
                f'WARNING: {path.name} was written with {changes}. The optimizer state still '
                'loads, but the global batch changed, so the resumed step lands at a different '
                'point in the corpus than it did in the interrupted run, and with `epochs` the '
                'LR schedule is rebuilt against a different total. Match the original values to '
                'continue the same schedule.'
            )
        saved_precision = state.get('precision', None)
        if saved_precision is not None and precision_name is not None and saved_precision != precision_name:
            print(
                f'WARNING: {path.name} was written under precision {saved_precision!r} and this '
                f'run is {precision_name!r}. Adam moments were allocated in the old parameter '
                'dtype and are being loaded into an optimizer expecting the new one.'
            )
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    # load_state_dict restores the schedule's position but leaves the optimizer holding whatever
    # learning rate it was constructed with until the next step(). That is one step at the
    # initial LR -- at the start of a warmup, near zero. Push the restored value through now.
    for group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
        group['lr'] = lr
    # Absent in checkpoints written before the RNG was recorded; those still resume, they just
    # cannot reproduce the augmentation stream.
    if 'python_rng' in state and own_python_rng:
        random.setstate(state['python_rng'])
    if 'torch_rng' in state:
        # Rank-uniform by construction (torch.manual_seed(seed), no rank offset), so restoring
        # one rank's copy everywhere is what an uninterrupted run would have had.
        torch.set_rng_state(state['torch_rng'])
    if rollout_generator is not None and 'rollout_generator' in state:
        rollout_generator.set_state(state['rollout_generator'])
    if rollout_rng is not None and 'rollout_rng' in state:
        rollout_rng.setstate(state['rollout_rng'])
    if scaler is not None and 'scaler' in state:
        scaler.load_state_dict(state['scaler'])
    if is_main:
        print(f'Resumed optimizer and LR schedule from {path} at step {state["step"]}')
    _restore_master_weights(state, optimizer, is_main)
    _check_weights_match_training_state(refiner_path, int(state['step']))
    return int(state['step'])


def _restore_master_weights(state, optimizer, is_main):
    """Put the fp32 masters back, for the modes where the module alone does not hold them.

    Only ZeRO with a bit16 section writes these. Copied in place rather than assigned, because
    deepspeed.initialize already installed these exact tensors as the optimizer's param_groups
    and the engine holds references to them.
    """
    masters = state.get('master_weights', None)
    if not masters:
        return
    live = [p for group in optimizer.param_groups for p in group['params']]
    if len(live) != len(masters):
        if is_main:
            print(
                f'WARNING: the checkpoint holds {len(masters)} fp32 master tensors but this run '
                f'has {len(live)}. Skipping the master restore; training continues from the '
                'bit16 weights, which loses precision accumulated since the last save.'
            )
        return
    # Every shape checked before anything is written: bailing out halfway would leave some
    # partitions restored and the rest not, which is worse than not restoring at all.
    mismatched = [i for i, (live_tensor, saved) in enumerate(zip(live, masters))
                  if live_tensor.shape != saved.shape]
    if mismatched:
        if is_main:
            print(
                f'WARNING: fp32 master shape mismatch at partition {mismatched[0]} on resume. '
                'Skipping the restore; training continues from the bit16 weights.'
            )
        return
    with torch.no_grad():
        for live_tensor, saved in zip(live, masters):
            live_tensor.copy_(saved.to(live_tensor.dtype))
    if is_main:
        print(f'Restored {len(masters)} fp32 master weight tensors.')


def _check_weights_match_training_state(refiner_path, state_step):
    """Refuse weights and optimizer moments that came from different steps.

    They are separate files written by separate calls, so an interruption between the two --
    or hand-copying one of them -- leaves a pair that loads without complaint and resumes an
    already-annealed schedule against weights from further ahead. Nothing else would report it.

    A file with no recorded step predates this check, or came from another mode. It claims
    nothing, so it is accepted, the same way a cache with no manifest is.
    """
    try:
        with safetensors.safe_open(str(refiner_path), framework='pt') as f:
            recorded = (f.metadata() or {}).get('step', None)
    except Exception:
        return
    if recorded is None:
        return
    if int(recorded) != state_step:
        raise RuntimeError(
            f'{refiner_path} holds weights from step {recorded}, but the training state beside '
            f'it is from step {state_step}. These are two halves of different checkpoints: '
            'resuming would pair optimizer moments with weights they never saw. Point '
            'resume_from at a tagged checkpoint whose two files agree.'
        )


if __name__ == '__main__':
    main()
