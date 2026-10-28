"""DoRA (use_dora in a LoRA [adapter] table): config checks and the saved form of the magnitude vector.

PEFT keeps a DoRA layer's magnitude vector at <module>.lora_magnitude_vector.default.weight, shape
(out_features,). It is saved as <module>.dora_scale, shape (out_features, 1): the name and shape
ComfyUI and kohya read. Without that name ComfyUI skips the magnitude and applies the file as a
plain LoRA. Free of training-stack imports, so it can be unit-tested on CPU.
"""

DORA_MAGNITUDE = '.lora_magnitude_vector.weight'
DORA_SCALE = '.dora_scale'


def apply_dora_config_defaults(adapter_config):
    """Validate use_dora in an [adapter] table and fill its default."""
    if adapter_config.get('type') != 'lora':
        if 'use_dora' in adapter_config:
            raise ValueError(f"use_dora is only supported for adapter type 'lora', not '{adapter_config.get('type')}'")
        return
    adapter_config.setdefault('use_dora', False)
    if not isinstance(adapter_config['use_dora'], bool):
        raise ValueError(f"use_dora must be true or false, got {adapter_config['use_dora']!r}")


def check_model_dtypes(adapter_config, model_config):
    """Refuse use_dora with float8 base weights: PEFT measures the row norms of the base weight
    to build the magnitude, and float8 tensors have no arithmetic for that."""
    if not adapter_config.get('use_dora', False):
        return
    for key in ('transformer_dtype', 'diffusion_model_dtype'):
        if 'float8' in str(model_config.get(key, '')):
            raise ValueError(f'use_dora cannot be combined with {key} = float8: DoRA needs the norm of each '
                             f'base weight row, which float8 tensors cannot compute. Remove {key} or set use_dora = false.')


def saved_key(key):
    """The saved name of a parameter name that has lost '.default'."""
    if key.endswith(DORA_MAGNITUDE):
        return key[:-len(DORA_MAGNITUDE)] + DORA_SCALE
    return key


def saved_value(key, value):
    """The saved tensor for a parameter name that has lost '.default'."""
    return value.reshape(-1, 1) if key.endswith(DORA_MAGNITUDE) else value


def has_dora_scale(state_dict):
    return any(k.endswith(DORA_SCALE) for k in state_dict)


def check_dora_matches_model(model, state_dict, source):
    """Refuse a plain LoRA file for a model whose LoRA layers are DoRA (use_dora = true): its
    magnitude would stay the one computed from the base weights alone, and the loaded LoRA would
    not reproduce its own output. A DoRA file for a plain LoRA model fails on its dora_scale keys."""
    if not has_dora_scale(state_dict) and any('lora_magnitude_vector' in name for name, _ in model.named_parameters()):
        raise ValueError(f'The adapter is DoRA (use_dora = true), but {source} is a plain LoRA (no {DORA_SCALE} '
                         'entries). Set use_dora = false to continue it as a LoRA, or start the DoRA from scratch.')


def check_init_matches(adapter_config, state_dict, source):
    """Refuse an init_from_existing file whose kind (DoRA or plain LoRA) is not the one use_dora
    asks for. A plain LoRA loaded into a DoRA model keeps the magnitude computed from the base
    weights alone, so the loaded LoRA would not reproduce its own output."""
    use_dora = bool(adapter_config.get('use_dora', False))
    if use_dora and not has_dora_scale(state_dict):
        raise ValueError(f'use_dora = true, but init_from_existing {source} is a plain LoRA (no {DORA_SCALE} '
                         'entries). Set use_dora = false to continue it as a LoRA, or start the DoRA from scratch.')
    if not use_dora and has_dora_scale(state_dict):
        raise ValueError(f'init_from_existing {source} is a DoRA adapter ({DORA_SCALE} entries), but use_dora '
                         'is not set. Set use_dora = true in [adapter] to continue it.')


def kohya_module_name(saved_name):
    """The kohya name of the module a saved SDXL adapter key belongs to.

    `saved_name` is '<unet|text_encoder|text_encoder_2>.<module path>.<lora_A|lora_B|dora_scale>...';
    kohya writes 'lora_unet_', 'lora_te1_' or 'lora_te2_' and every '.' of the module path as '_'.
    """
    for prefix, kohya in (('text_encoder_2.', 'lora_te2_'), ('text_encoder.', 'lora_te1_'), ('unet.', 'lora_unet_')):
        if saved_name.startswith(prefix):
            rest = saved_name[len(prefix):]
            break
    else:
        raise ValueError(f'Not an SDXL adapter key: {saved_name}')
    for suffix in ('.lora_A.weight', '.lora_B.weight', '.dora_scale'):
        if rest.endswith(suffix):
            rest = rest[:-len(suffix)]
            break
    return kohya + rest.replace('.', '_')
