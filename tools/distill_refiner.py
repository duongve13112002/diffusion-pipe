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
import contextlib
import os
import random
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

from models.cosmos_predict2 import get_dit_config
from models.cosmos_predict2_modeling import MiniTrainDIT
from models.text_refiner import ContextRefiner, extract_refiner_state_dict
from utils.common import iterate_safetensors, load_state_dict
from utils.caption_corpus import read_corpus
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
            if (text := txt.read_text(encoding='utf-8').strip())
        ]
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


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
        missing = [k for k in ('prefix_tag_caption', 'caption_prefix', 'cache_shuffle_num',
                               'shuffle_tags', 'tag_dropout_rate') if k in distill_config]
        if not missing:
            print(
                'WARNING: the caption source is not a dataset.toml, so no caption settings can '
                'be inherited. Tag shuffling, tag dropout, caption_prefix and prefix_tag_caption '
                'are all off. If the dataset uses any of them, restate them under [distill] -- '
                'export_caption_corpus.py prints the prefix_tag_caption line to use. Without '
                'prefix_tag_caption the tag marker is trained as if it were a tag.'
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

    # Everything else in the DiT is dead weight for this stage.
    dit.blocks = None
    dit.llm_adapter = None
    del dit, state_dict

    text_encoder.to(device).eval().requires_grad_(False)
    llm_adapter.to(device).eval().requires_grad_(False)
    cross_attns.to(device).eval().requires_grad_(False)
    return tokenizer, text_encoder, llm_adapter, cross_attns, model_channels, crossattn_emb_channels


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
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


DISTRIBUTED_STRATEGIES = ('ddp', 'zero1', 'zero2')

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

    def step(self):
        # Unscale before clipping, or max_grad_norm would be compared against a gradient
        # inflated by the loss scale and would effectively never trigger.
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.refiner.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
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
        self.precision = precision
        ds_config = {
            'train_micro_batch_size_per_gpu': batch_size,
            'gradient_accumulation_steps': grad_accum,
            'gradient_clipping': max_grad_norm,
            'zero_optimization': {
                'stage': stage,
                'contiguous_gradients': True,
                'overlap_comm': True,
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
        # The engine tracks the accumulation boundary itself and only reduces there, so there
        # is nothing to suppress from out here.
        return contextlib.nullcontext()

    def backward(self, loss):
        self.engine.backward(loss)

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
    batch_size = config['distill'].get('batch_size', 8)
    grad_accum = max(1, config['distill'].get('gradient_accumulation_steps', 1))
    steps = config['distill'].get('steps', 20000)
    seed = config['distill'].get('seed', 42)
    pooled_weight = config['distill'].get('pooled_loss_weight', 0.1)
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

    captions = load_captions(config)
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

    if is_main:
        print('Building teacher...')
    t5_tokenizer = T5TokenizerFast(
        vocab_file='configs/t5_old/spiece.model',
        tokenizer_file='configs/t5_old/tokenizer.json',
    )
    teacher_tok, teacher_llm, llm_adapter, cross_attns, model_channels, crossattn_emb_channels = build_teacher(config, dtype, device)

    if is_main:
        print('Building student...')
    student_tok, student_llm, refiner, cap_feat_dim = build_student(config, dtype, device, crossattn_emb_channels)
    if is_main:
        print(f'Student LLM hidden size: {cap_feat_dim}')

    # Resolved and applied BEFORE the optimizer is built. AdamW allocates its exp_avg and
    # exp_avg_sq to match each parameter's dtype at construction time, so casting the refiner
    # afterwards would leave the optimizer holding state of the wrong dtype.
    precision = resolve_precision(config)
    if precision.param_dtype != torch.float32:
        refiner.to(dtype=precision.param_dtype)
    if is_main:
        print(f'Precision: {precision.name} -- {precision.note}')

    # Fixed probe queries. The cross-attention modules are frozen and shared by both paths, so
    # any query set works as a measuring stick; a fixed one keeps the objective stationary
    # across steps. Matching the output for many random queries is a strong proxy for matching
    # the key/value content itself, without ever comparing individual token positions.
    num_queries = config.get('probe', {}).get('num_queries', 64)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    probe = torch.randn(1, num_queries, model_channels, generator=generator).to(device=device, dtype=dtype)

    # The probe seed is deliberately NOT rank-offset: every rank must measure against the same
    # queries, or their gradients describe different objectives.

    optimizer = build_optimizer(config, refiner, is_main)
    scheduler = build_lr_scheduler(config, optimizer, steps)
    if is_main:
        print(f'Optimizer: {type(optimizer).__name__} | '
              f"lr_scheduler: {config['distill'].get('lr_scheduler', 'cosine')}")

    # Built after the optimizer and scheduler: ZeRO wraps both, so they have to exist first.
    strategy = build_strategy(config, refiner, world_size, local_rank, device, batch_size,
                              grad_accum, optimizer, scheduler, precision)
    train_module = strategy.module
    if is_main and strategy.name != 'ddp':
        print(f'Parallelism: DeepSpeed ZeRO stage {strategy.stage} (optimizer state sharded).')

    save_every = config['distill'].get('save_every', 2000)
    log_every = config['distill'].get('log_every', 50)
    running = 0.0
    progress_bar = tqdm(range(steps), desc='distill', disable=not is_main)

    strategy.zero_grad()
    for step in progress_bar:
        accum_loss = 0.0
        strategy.zero_grad()
        for micro in range(grad_accum):
            batch = [
                preprocess_caption(c, **augment)
                for c in random.sample(captions, min(batch_size, len(captions)))
            ]

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

                s_enc = student_tok(batch, return_tensors='pt', truncation=True, padding='max_length', max_length=max_text_length)
                s_mask = s_enc.attention_mask.to(device)
                student_hidden = encode(
                    student_llm, s_enc.input_ids.to(device), s_mask, config['student'].get('llm_hidden_layer', None)
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
                for cross_attn in cross_attns:
                    with torch.no_grad():
                        target = cross_attn(q, context=teacher_feats.to(dtype))
                    pred = cross_attn(q, context=student_feats.to(dtype))
                    loss = loss + F.mse_loss(pred.float(), target.float())
                loss = loss / len(cross_attns)

                # Auxiliary global term. Also permutation invariant, and it gives a useful
                # gradient early on while the probe-attention term is still dominated by noise.
                if pooled_weight > 0:
                    pooled_loss = F.mse_loss(
                        padded_mean(student_feats.float(), s_mask, max_text_length),
                        padded_mean(teacher_feats.float(), t5_mask, max_text_length),
                    )
                    loss = loss + pooled_weight * pooled_loss

                strategy.backward(loss)
            accum_loss += loss.item() / grad_accum

        grad_norm = strategy.step()

        running += accum_loss
        if is_main and (step + 1) % log_every == 0:
            progress_bar.set_postfix({
                'loss': f'{running / log_every:.5f}',
                'grad': f'{grad_norm:.3f}',
                'lr': f'{strategy.last_lr():.2e}',
            })
            running = 0.0

        if (step + 1) % save_every == 0 or step + 1 == steps:
            # Every rank holds the full, identical weights -- DDP all-reduces them, and ZeRO
            # 1/2 shard optimizer state and gradients but never the parameters. Only one writes.
            if is_main:
                save_refiner(refiner, output_dir / 'context_refiner.safetensors', dtype)
            if world_size > 1:
                dist.barrier()

    if is_main:
        save_refiner(refiner, output_dir / 'context_refiner.safetensors', dtype)
        print(f'Done. Point context_refiner_path at {output_dir / "context_refiner.safetensors"}')

    if is_main and config['distill'].get('save_full_model', False):
        path = output_dir / 'model.safetensors'
        save_full_model(config['teacher']['transformer_path'], refiner, path, dtype)
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
        state_dict['net.context_refiner.' + k] = v.detach().to(dtype).cpu().contiguous()
    safetensors.torch.save_file(state_dict, str(path), metadata={'format': 'pt'})


def save_refiner(refiner, path, dtype):
    state_dict = {k: v.detach().to(dtype).cpu().contiguous() for k, v in refiner.state_dict().items()}
    safetensors.torch.save_file(state_dict, str(path), metadata={'format': 'pt'})


if __name__ == '__main__':
    main()
