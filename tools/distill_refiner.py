"""Stage 1 for the anima_refiner architecture: distil a ContextRefiner from Anima's LLMAdapter.

Why this exists
---------------
anima_refiner replaces Anima's LLMAdapter with a ContextRefiner (models/text_refiner.py). The
DiT's cross-attention was trained to consume whatever the LLMAdapter emits, so a freshly
initialised refiner produces features the frozen DiT cannot read. Training it from a random
init with the diffusion loss works, but it needs images, the VAE and a full DiT forward pass
for every step.

This script gets the refiner into roughly the right space using captions only: no images, no
VAE, no diffusion. It is a warm start, not a finished model -- follow it with a diffusion-loss
stage (see docs/anima_refiner.md).

Why the loss is measured at the cross-attention output
------------------------------------------------------
The obvious objective, a position-wise MSE between teacher and student features, does not
work here. The teacher's output sequence is indexed by *T5* tokens (the LLMAdapter embeds T5
ids as its query sequence) while the student's is indexed by the source LLM's own tokens. Both
are (B, L, 1024) so the shapes match and the code would run, but position i means different
things on each side, and the loss plateaus without ever converging.

Cross-attention is a weighted sum over text positions, so its output does not depend on how
that text is indexed. Pushing both feature sets through the DiT's own frozen cross-attention
modules and comparing there sidesteps the mismatch entirely, and it optimises exactly the
quantity the DiT will consume.

Usage:
    python -m tools.distill_refiner --config examples/anima_refiner_distill.toml
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import safetensors.torch
import toml
import torch
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
from utils.captions import enumerate_captions, preprocess_caption

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
        # actually in use and hand the whole list down; split_tag_prefix accepts a list.
        markers = set()
        enumerate_captions(dataset_config, apply_shuffle=False, markers_seen=markers)
        if markers:
            fallback['prefix_tag_caption'] = sorted(markers)

    def setting(key, default):
        if key in distill_config:
            return distill_config[key]
        return fallback.get(key, default)

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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='Path to the TOML config.')
    args = parser.parse_args()
    config = toml.load(args.config)

    device = torch.device(config['distill'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    dtype = getattr(torch, config['distill'].get('dtype', 'bfloat16'))
    max_text_length = config['distill'].get('max_text_length', MAX_TEXT_LENGTH_DEFAULT)
    batch_size = config['distill'].get('batch_size', 8)
    steps = config['distill'].get('steps', 20000)
    seed = config['distill'].get('seed', 42)
    pooled_weight = config['distill'].get('pooled_loss_weight', 0.1)
    output_dir = Path(config['distill']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    random.seed(seed)

    captions = load_captions(config)
    if not captions:
        raise RuntimeError('No captions found. Check the dataset / captions path under [distill].')
    print(f'Loaded {len(captions)} captions')
    augment = caption_augment_config(config)
    if augment['shuffle'] or augment['tag_dropout_rate'] > 0:
        print(
            f"Caption augmentation: shuffle={augment['shuffle']} "
            f"tag_dropout_rate={augment['tag_dropout_rate']} "
            f"prefix_tag_caption={augment['prefix_tag_caption']!r}"
        )

    print('Building teacher...')
    t5_tokenizer = T5TokenizerFast(
        vocab_file='configs/t5_old/spiece.model',
        tokenizer_file='configs/t5_old/tokenizer.json',
    )
    teacher_tok, teacher_llm, llm_adapter, cross_attns, model_channels, crossattn_emb_channels = build_teacher(config, dtype, device)

    print('Building student...')
    student_tok, student_llm, refiner, cap_feat_dim = build_student(config, dtype, device, crossattn_emb_channels)
    print(f'Student LLM hidden size: {cap_feat_dim}')

    # Fixed probe queries. The cross-attention modules are frozen and shared by both paths, so
    # any query set works as a measuring stick; a fixed one keeps the objective stationary
    # across steps. Matching the output for many random queries is a strong proxy for matching
    # the key/value content itself, without ever comparing individual token positions.
    num_queries = config.get('probe', {}).get('num_queries', 64)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    probe = torch.randn(1, num_queries, model_channels, generator=generator).to(device=device, dtype=dtype)

    lr = config['distill'].get('lr', 1e-4)
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=lr,
        betas=tuple(config['distill'].get('betas', [0.9, 0.99])),
        weight_decay=config['distill'].get('weight_decay', 0.01),
    )
    warmup = config['distill'].get('warmup_steps', 500)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    save_every = config['distill'].get('save_every', 2000)
    log_every = config['distill'].get('log_every', 50)
    running = 0.0
    progress_bar = tqdm(range(steps), desc='distill')

    for step in progress_bar:
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

        student_feats = refiner(student_hidden.to(torch.float32), s_mask)

        q = probe.expand(len(batch), -1, -1)
        loss = 0.0
        for cross_attn in cross_attns:
            with torch.no_grad():
                target = cross_attn(q, context=teacher_feats.to(dtype))
            pred = cross_attn(q, context=student_feats.to(dtype))
            loss = loss + F.mse_loss(pred.float(), target.float())
        loss = loss / len(cross_attns)

        # Auxiliary global term. Also permutation invariant, and it gives a useful gradient
        # early on while the probe-attention term is still dominated by noise.
        if pooled_weight > 0:
            pooled_loss = F.mse_loss(
                padded_mean(student_feats.float(), s_mask, max_text_length),
                padded_mean(teacher_feats.float(), t5_mask, max_text_length),
            )
            loss = loss + pooled_weight * pooled_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(refiner.parameters(), config['distill'].get('max_grad_norm', 1.0))
        optimizer.step()
        scheduler.step()

        running += loss.item()
        if (step + 1) % log_every == 0:
            progress_bar.set_postfix({
                'loss': f'{running / log_every:.5f}',
                'grad': f'{grad_norm:.3f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            })
            running = 0.0

        if (step + 1) % save_every == 0 or step + 1 == steps:
            save_refiner(refiner, output_dir / 'context_refiner.safetensors', dtype)

    save_refiner(refiner, output_dir / 'context_refiner.safetensors', dtype)
    print(f'Done. Point context_refiner_path at {output_dir / "context_refiner.safetensors"}')

    if config['distill'].get('save_full_model', False):
        path = output_dir / 'model.safetensors'
        save_full_model(config['teacher']['transformer_path'], refiner, path, dtype)
        print(f'Also wrote a full anima_refiner checkpoint to {path}. Use it as transformer_path.')


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
