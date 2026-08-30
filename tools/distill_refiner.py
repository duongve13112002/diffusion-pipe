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
from models.text_refiner import ContextRefiner
from utils.common import iterate_safetensors, load_state_dict
from utils.dataset import enumerate_captions

MAX_TEXT_LENGTH_DEFAULT = 512


def load_captions(config):
    """Get the captions to distil on, preferring the ordinary dataset.toml flow.

    `dataset` points at the same dataset.toml every other training mode uses, so the captions
    seen here are exactly the ones training will see -- same directories, same captions.json /
    .txt resolution, same caption_prefix and tag shuffling. Images are never opened: this stage
    trains only the text frontend.

    `captions` is a fallback for when there is no dataset.toml to hand: either a file with one
    caption per line, or a directory of .txt files.
    """
    distill_config = config['distill']
    if 'dataset' in distill_config:
        dataset_config = toml.load(distill_config['dataset'])
        return enumerate_captions(dataset_config, apply_num_repeats=distill_config.get('apply_num_repeats', False))

    if 'captions' not in distill_config:
        raise RuntimeError("set either 'dataset' (a dataset.toml) or 'captions' under [distill]")

    path = Path(distill_config['captions'])
    if path.is_dir():
        return [
            text for txt in sorted(path.rglob('*.txt'))
            if (text := txt.read_text(encoding='utf-8').strip())
        ]
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def extract_refiner_state_dict(path):
    """Pull refiner weights out of either a bare refiner file or a full model checkpoint.

    This is what makes distillation re-runnable at any point: a model saved by refiner_only,
    refiner_crossattn or a full fine tune keys its refiner as `net.context_refiner.*`, and
    that has to be loadable back into a bare ContextRefiner so the same refiner can be taken
    back to the distillation objective.
    """
    state_dict = load_state_dict(path)
    refiner = {}
    for k, v in state_dict.items():
        if k.startswith('net.'):
            k = k[len('net.'):]
        if k.startswith('context_refiner.'):
            k = k[len('context_refiner.'):]
        elif any(key.startswith(('context_refiner.', 'net.context_refiner.')) for key in state_dict):
            # A full checkpoint: keep only the refiner half of it.
            continue
        refiner[k] = v
    if not refiner:
        raise RuntimeError(f'No context_refiner weights found in {path}')
    return refiner


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
    num_probe_blocks = config['probe'].get('num_blocks', 8)
    num_probe_blocks = min(num_probe_blocks, len(dit.blocks))
    stride = max(1, len(dit.blocks) // num_probe_blocks)
    block_indices = list(range(0, len(dit.blocks), stride))[:num_probe_blocks]
    cross_attns = nn.ModuleList([dit.blocks[i].cross_attn for i in block_indices])
    model_channels = dit_config['model_channels']

    # Everything else in the DiT is dead weight for this stage.
    dit.blocks = None
    dit.llm_adapter = None
    del dit, state_dict

    text_encoder.to(device).eval().requires_grad_(False)
    llm_adapter.to(device).eval().requires_grad_(False)
    cross_attns.to(device).eval().requires_grad_(False)
    return tokenizer, text_encoder, llm_adapter, cross_attns, model_channels


def build_student(config, dtype, device):
    """Source LLM plus a fresh (or resumed) ContextRefiner."""
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

    refiner = ContextRefiner(
        cap_feat_dim=cap_feat_dim,
        model_dim=config['student'].get('model_dim', 1024),
        num_layers=config['student'].get('n_refiner_layers', 6),
    )
    refiner.init_weights()
    if resume := config['student'].get('resume_from', None):
        refiner.load_state_dict(extract_refiner_state_dict(resume))
        print(f'Resumed refiner weights from {resume}')
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


def masked_mean(x, mask):
    mask = mask.unsqueeze(-1).to(x.dtype)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


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

    print('Building teacher...')
    t5_tokenizer = T5TokenizerFast(
        vocab_file='configs/t5_old/spiece.model',
        tokenizer_file='configs/t5_old/tokenizer.json',
    )
    teacher_tok, teacher_llm, llm_adapter, cross_attns, model_channels = build_teacher(config, dtype, device)

    print('Building student...')
    student_tok, student_llm, refiner, cap_feat_dim = build_student(config, dtype, device)
    print(f'Student LLM hidden size: {cap_feat_dim}')

    # Fixed probe queries. The cross-attention modules are frozen and shared by both paths, so
    # any query set works as a measuring stick; a fixed one keeps the objective stationary
    # across steps. Matching the output for many random queries is a strong proxy for matching
    # the key/value content itself, without ever comparing individual token positions.
    num_queries = config['probe'].get('num_queries', 64)
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
        batch = random.sample(captions, min(batch_size, len(captions)))

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
                masked_mean(student_feats.float(), s_mask),
                masked_mean(teacher_feats.float(), t5_mask),
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


def save_refiner(refiner, path, dtype):
    state_dict = {k: v.detach().to(dtype).cpu().contiguous() for k, v in refiner.state_dict().items()}
    safetensors.torch.save_file(state_dict, str(path), metadata={'format': 'pt'})


if __name__ == '__main__':
    main()
