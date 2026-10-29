from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func  # noqa: F401

_flash_attn_forward = None
