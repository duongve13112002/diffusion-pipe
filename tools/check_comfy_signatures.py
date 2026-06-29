"""Guard against ComfyUI submodule signature drift.

The ComfyUI-backed models in ``models/`` do not call ComfyUI's own ``forward()``.
They re-implement the forward pass (to split it into pipeline-parallel layers) and
call ComfyUI leaf submodules and helper functions directly. When the ComfyUI
submodule is updated, a leaf's signature can change (e.g. a new required argument),
and the re-implemented call silently breaks at runtime. That is exactly what happened
when ``Ideogram4EmbedScalar.forward`` gained a required ``dtype`` argument.

This script imports the pinned ComfyUI and checks, for every cross-boundary call the
models rely on, that the current signature still accepts what the model code passes.
It does NOT need a GPU, but it DOES need ComfyUI importable (torch + comfy_aimdo and
the other ComfyUI runtime deps), so run it on a machine where training works:

    python tools/check_comfy_signatures.py

Exit code is non-zero if any check fails, so it can be wired into CI or a pre-update
hook. Run it after every ``git submodule update`` (see CLAUDE.md and
docs/note/comfyui-submodule-signature-audit.md).

When a check legitimately needs to change because ComfyUI changed on purpose, update
both the model code and the corresponding check here in the same commit.
"""

import inspect
import os
import sys

# Make the vendored ComfyUI importable the same way train.py does.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'submodules', 'ComfyUI'))


class CheckError(Exception):
    pass


def _params(func):
    return inspect.signature(func).parameters


def accepts(func, name):
    """True if func accepts a keyword argument `name` (explicitly or via **kwargs)."""
    params = _params(func)
    if name in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def has_default(func, name):
    """True if param `name` exists and has a default (so callers may omit it)."""
    params = _params(func)
    return name in params and params[name].default is not inspect.Parameter.empty


def required_positional_names(func):
    """Names of parameters with no default and no *args/**kwargs, excluding self."""
    out = []
    for n, p in _params(func).items():
        if n == 'self':
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p.default is inspect.Parameter.empty:
            out.append(n)
    return out


def check(description, condition):
    status = 'PASS' if condition else 'FAIL'
    print(f'[{status}] {description}')
    return condition


def run_checks():
    """Return a list of (description, ok) for every signature contract we depend on."""
    results = []

    def record(description, condition):
        results.append((description, bool(condition)))
        return condition

    # ----- shared flux layers (flux2, hunyuan_video_15, krea2, chroma) -----
    from comfy.ldm.flux.layers import (
        timestep_embedding, ModulationOut, DoubleStreamBlock, SingleStreamBlock,
    )
    record('flux.layers.timestep_embedding accepts (t, dim)',
           accepts(timestep_embedding, 't') and accepts(timestep_embedding, 'dim'))
    record('flux.layers.ModulationOut has fields shift/scale/gate',
           all(f in ModulationOut.__dataclass_fields__ for f in ('shift', 'scale', 'gate')))
    # Every model omits transformer_options when calling blocks, so it MUST stay optional.
    record('flux.DoubleStreamBlock.forward keeps transformer_options optional',
           has_default(DoubleStreamBlock.forward, 'transformer_options'))
    record('flux.DoubleStreamBlock.forward accepts img/txt/vec/pe',
           all(accepts(DoubleStreamBlock.forward, n) for n in ('img', 'txt', 'vec', 'pe')))
    record('flux.SingleStreamBlock.forward keeps transformer_options optional',
           has_default(SingleStreamBlock.forward, 'transformer_options'))

    # ----- common_dit (z_image, ltx2, hunyuan_video_15, flux2, ernie_image, krea2) -----
    import comfy.ldm.common_dit as common_dit
    record('common_dit.pad_to_patch_size exists', callable(getattr(common_dit, 'pad_to_patch_size', None)))
    record('common_dit.rms_norm exists', callable(getattr(common_dit, 'rms_norm', None)))

    # ----- ideogram4 -----
    import comfy.ldm.ideogram4.model as ideo
    # The original bug: this gained a required `dtype`. Model passes it as a kwarg now.
    record('ideogram4.Ideogram4EmbedScalar.forward accepts dtype',
           accepts(ideo.Ideogram4EmbedScalar.forward, 'dtype'))
    record('ideogram4.Ideogram4FinalLayer.forward accepts (x, c)',
           accepts(ideo.Ideogram4FinalLayer.forward, 'x') and accepts(ideo.Ideogram4FinalLayer.forward, 'c'))
    record('ideogram4.Ideogram4TransformerBlock.forward keeps transformer_options optional',
           has_default(ideo.Ideogram4TransformerBlock.forward, 'transformer_options'))
    from comfy.text_encoders.llama import precompute_freqs_cis
    record('llama.precompute_freqs_cis accepts head_dim/position_ids/theta + rope_dims/interleaved_mrope',
           all(accepts(precompute_freqs_cis, n)
               for n in ('head_dim', 'position_ids', 'theta', 'rope_dims', 'interleaved_mrope')))

    # operations.Embedding must accept out_dtype (model now mirrors ComfyUI and passes it).
    import comfy.ops as ops
    emb_cast = ops.disable_weight_init.Embedding.forward_comfy_cast_weights
    record('ops.Embedding.forward_comfy_cast_weights accepts out_dtype', accepts(emb_cast, 'out_dtype'))

    # ----- z_image (lumina backbone) -----
    import comfy.ldm.lumina.model as lumina
    record('lumina.JointTransformerBlock.forward(x, x_mask, freqs_cis) + adaln_input optional',
           all(accepts(lumina.JointTransformerBlock.forward, n) for n in ('x', 'x_mask', 'freqs_cis'))
           and has_default(lumina.JointTransformerBlock.forward, 'adaln_input'))

    # ----- ltx2 (lightricks av_model) -----
    from comfy.ldm.lightricks.av_model import BasicAVTransformerBlock
    from comfy.ldm.lightricks.model import CrossAttention
    # ltx2 monkey-patches BasicAVTransformerBlock.forward; the patch must accept the same
    # keyword arguments ComfyUI's model passes to its blocks.
    av_forward_params = ('x', 'v_context', 'a_context', 'attention_mask', 'v_timestep', 'a_timestep',
                         'v_pe', 'a_pe', 'v_cross_pe', 'a_cross_pe', 'transformer_options')
    record('av_model.BasicAVTransformerBlock.forward accepts the AV keyword set',
           all(accepts(BasicAVTransformerBlock.forward, n) for n in av_forward_params))
    record('av_model.BasicAVTransformerBlock._apply_text_cross_attention is present',
           callable(getattr(BasicAVTransformerBlock, '_apply_text_cross_attention', None)))
    record('lightricks.CrossAttention.forward accepts context/mask/pe/k_pe',
           all(accepts(CrossAttention.forward, n) for n in ('context', 'mask', 'pe', 'k_pe')))

    # ----- ernie_image -----
    import comfy.ldm.ernie.model as ernie
    # The transformer block is called positionally as (x, rotary_pos_emb, temb, attention_mask=...).
    ernie_block = None
    for _name, obj in vars(ernie).items():
        if inspect.isclass(obj) and hasattr(obj, 'forward'):
            names = list(_params(obj.forward))
            if names[:4] == ['self', 'x', 'rotary_pos_emb', 'temb']:
                ernie_block = obj
                break
    record('ernie block.forward(x, rotary_pos_emb, temb, attention_mask) found',
           ernie_block is not None and has_default(ernie_block.forward, 'attention_mask'))

    # ----- krea2 -----
    import comfy.ldm.krea2.model as krea2
    krea_block = None
    for _name, obj in vars(krea2).items():
        if inspect.isclass(obj) and hasattr(obj, 'forward'):
            names = list(_params(obj.forward))
            if names[:4] == ['self', 'x', 'vec', 'freqs']:
                krea_block = obj
                break
    record('krea2 block.forward(x, vec, freqs, mask=...) found with optional mask',
           krea_block is not None and has_default(krea_block.forward, 'mask'))

    # ----- loading / adapter / VAE / text-encoder APIs (all ComfyUI-backed models) -----
    import comfy.sd as sd
    record('sd.load_clip accepts ckpt_paths/clip_type/disable_dynamic',
           all(accepts(sd.load_clip, n) for n in ('ckpt_paths', 'clip_type', 'disable_dynamic')))
    record('sd.load_diffusion_model accepts model_options/disable_dynamic',
           all(accepts(sd.load_diffusion_model, n) for n in ('model_options', 'disable_dynamic')))
    record('sd.load_checkpoint_guess_config accepts output_vae/output_clip',
           all(accepts(sd.load_checkpoint_guess_config, n) for n in ('output_vae', 'output_clip')))
    record('sd.load_lora_for_models takes (model, clip, lora, strength_model, strength_clip)',
           len(required_positional_names(sd.load_lora_for_models)) <= 5
           and all(accepts(sd.load_lora_for_models, n)
                   for n in ('model', 'clip', 'lora', 'strength_model', 'strength_clip')))
    record('sd.VAE.__init__ accepts sd and metadata',
           accepts(sd.VAE.__init__, 'sd') and accepts(sd.VAE.__init__, 'metadata'))
    import comfy.utils as cu
    record('utils.load_torch_file accepts safe_load', accepts(cu.load_torch_file, 'safe_load'))
    import comfy.sd1_clip as sd1
    record('sd1_clip.gen_empty_tokens accepts (special_tokens, length)',
           accepts(sd1.gen_empty_tokens, 'special_tokens') and accepts(sd1.gen_empty_tokens, 'length'))
    record('sd1_clip.ClipTokenWeightEncoder.encode_token_weights(self, token_weight_pairs)',
           required_positional_names(sd1.ClipTokenWeightEncoder.encode_token_weights) == ['token_weight_pairs'])

    return results


def main():
    try:
        results = run_checks()
    except ImportError as e:
        print('Could not import ComfyUI. Run this on a machine where training works '
              f'(torch + ComfyUI runtime deps installed). Underlying error: {e}')
        return 2
    for description, ok in results:
        check(description, ok)
    failed = [d for d, ok in results if not ok]
    print()
    if failed:
        print(f'{len(failed)} signature check(s) FAILED. ComfyUI drifted; fix the matching '
              'model code (and update this script if the change is intended):')
        for d in failed:
            print(f'  - {d}')
        return 1
    print(f'All {len(results)} ComfyUI signature checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
