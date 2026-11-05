"""FP8 training with NVIDIA Transformer Engine (TE): the `[fp8_training]` table.

    [fp8_training]
    recipe = 'block'               # 'block' (1x128 / 128x128 block scales) or 'current' (one scale per tensor)
    weights = 'dequantize'         # or 'direct': base weights stored as e4m3 go into the fp8 GEMM as they are
    exclude = ['adaLN', 'proj_out']  # regexes on module names that stay in bf16
    cache_frozen_weights = false   # keep the fp8 copy of frozen base weights between steps (adapters only)
    attention = false              # fp8 attention through TE's cuDNN kernels (opt-in)
    attention_amax_history = 16    # steps of amax history behind each attention scale

Linear layers
-------------
Every nn.Linear inside the model's transformer blocks (the classes in `adapter_target_modules`,
or `fp8_target_modules` where a model sets its own) is replaced by Fp8Linear before the adapter is
built. Fp8Linear keeps the layer's weight and bias Parameters, so LoRA, LoKr, DoRA, OPLoRA and
full fine-tuning wrap or train it as before, and checkpoints and saved adapters do not change. Its
forward hands the GEMM to a te.Linear, which quantizes input and weight to fp8, runs the fp8 GEMM
and, in the backward, the dgrad and (for trained weights) wgrad GEMMs in fp8 as well. Both recipes
compute their scales from the tensor being quantized, so a recomputation under activation
checkpointing gives the forward's numbers again.

TE runs a GEMM in fp8 only when every dimension it reduces over is a multiple of 16: in_features
(forward), out_features (input gradient) and the tokens, all dimensions but the last (weight
gradient). A smaller multiple passes the forward and fails in the backward (block scaling:
"Leading dimension requirement on NVTE_BLOCK_SCALING GEMM. Caller must pad."). Layers whose
weight does not fit stay bf16 for good; a call whose token count does not fit runs bf16 for that call only. Both are
counted and reported.

`weights` decides what happens to base weights stored in fp8 or another quantized format:
'dequantize' (the default) turns them into the model dtype at load, so they take the same path as
bf16 weights. 'direct' keeps an e4m3 weight with one scale per tensor (ComfyUI `float8_e4m3fn`
checkpoints, or `transformer_dtype = 'float8'`, scale 1) as it is and gives it to the fp8 GEMM
without quantizing it again; those layers use the 'current' recipe whatever `recipe` says (TE
pairs a per-tensor fp8 weight with per-tensor scaling only). Other formats (mxfp8, nvfp4, int8,
w4a4, e5m2) are dequantized in either mode.

Attention
---------
`attention = true` sends attention calls (PyTorch SDPA, and the FlashAttention entry points of the
models that call flash-attn themselves) to TE's DotProductAttention with fp8 inputs. On Hopper
(H100 / H200) TE trains fp8 attention only with the DelayedScaling recipe: each call site keeps a
history of the largest values (amax) it saw and takes its scales from that history, so the scales
are state. Two things keep that state from breaking training:

* Scales change only between optimizer steps. TE normally updates them every time a call ends;
  here the whole step runs inside one outer TE context, so every forward of the step and its
  recomputation under activation checkpointing (PyTorch's, DeepSpeed's or unsloth's) use the same
  scales. step_scope() does that and updates the scales once, when the step is done.
* A call site's first step runs bf16 and measures what fp8 needs: the amax of q, k, v and the
  output in the forward, and through gradient hooks the amax of the output gradient and of the q,
  k, v gradients, plus a bound on the softmax gradient. Its first fp8 step then starts from those
  scales instead of TE's initial 1, at which the softmax probabilities and the small gradients of
  a diffusion loss would mostly round to zero.

Call sites are told apart by the transformer block they run in (a forward hook on each block)
and their order within it, so each layer has its own scales and a recomputation finds the same
site. Calls outside the blocks, with an arbitrary mask or bias, dropout, float32 inputs or head
sizes TE's fp8 kernel does not take run the existing path (FlashAttention or PyTorch SDPA) and are
counted. Sampling previews run bf16.
"""

import collections
import contextlib
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

TABLE = 'fp8_training'
RECIPES = ('block', 'current')
WEIGHT_MODES = ('dequantize', 'direct')
DEFAULTS = {
    'recipe': 'block',
    'weights': 'dequantize',
    'exclude': [],
    'cache_frozen_weights': False,
    'attention': False,
    'attention_amax_history': 16,
}
INSTALL_HINT = ("pip install --no-build-isolation 'transformer_engine[pytorch]' (it compiles against the installed "
                "PyTorch and needs the CUDA toolkit (nvcc) and cuDNN 9.12 or newer)")
# TE's own minimum (docs/installation.rst).
MIN_CUDNN = 91200
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
HALF = (torch.bfloat16, torch.float16)

# Settings once configure_settings() ran; None means fp8 training is off.
_settings = None
# The TE wrapper (TransformerEngineBackend); tests put a stand-in here.
_backend = None
# Inside disabled() (sampling): everything runs bf16.
_disabled = 0
# The active step_scope(): None outside one, else True for a training step, False for eval.
_scope = None
_linear_counts = collections.Counter()
_attention_counts = collections.Counter()
_noted = set()


class Settings:
    def __init__(self, table):
        self.recipe = table['recipe']
        self.weights = table['weights']
        self.exclude = [re.compile(p) for p in table['exclude']]
        self.cache_frozen_weights = table['cache_frozen_weights']
        self.attention = table['attention']
        self.attention_amax_history = table['attention_amax_history']


def apply_fp8_training_config_defaults(config):
    """Checks the [fp8_training] table and fills in its defaults; refuses what does not apply."""
    if TABLE not in config:
        return None
    table = config[TABLE]
    if not isinstance(table, dict):
        raise ValueError(f'[{TABLE}] must be a table, got {table!r}')
    unknown = sorted(set(table) - set(DEFAULTS))
    if unknown:
        raise ValueError(f'[{TABLE}] has unknown keys {unknown}; the keys are {sorted(DEFAULTS)}')
    given = set(table)
    for key, value in DEFAULTS.items():
        table.setdefault(key, list(value) if isinstance(value, list) else value)
    if table['recipe'] not in RECIPES:
        raise ValueError(f"[{TABLE}] recipe must be one of {', '.join(repr(r) for r in RECIPES)}, got {table['recipe']!r}")
    if table['weights'] not in WEIGHT_MODES:
        raise ValueError(f"[{TABLE}] weights must be one of {', '.join(repr(w) for w in WEIGHT_MODES)}, "
                         f"got {table['weights']!r}")
    exclude = table['exclude']
    if isinstance(exclude, str):
        exclude = table['exclude'] = [exclude]
    if not isinstance(exclude, list) or not all(isinstance(p, str) for p in exclude):
        raise ValueError(f'[{TABLE}] exclude must be a list of regular expressions, got {exclude!r}')
    for pattern in exclude:
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(f'[{TABLE}] exclude: {pattern!r} is not a regular expression: {error}') from None
    for key in ('cache_frozen_weights', 'attention'):
        if not isinstance(table[key], bool):
            raise ValueError(f'[{TABLE}] {key} must be true or false, got {table[key]!r}')
    history = table['attention_amax_history']
    if isinstance(history, bool) or not isinstance(history, int) or not 1 <= history <= 1024:
        raise ValueError(f'[{TABLE}] attention_amax_history must be a whole number from 1 to 1024, got {history!r}')
    if 'attention_amax_history' in given and not table['attention']:
        raise ValueError(f'[{TABLE}] attention_amax_history applies only with attention = true; remove it or turn '
                         'attention on')

    has_adapter = 'adapter' in config
    if table['weights'] == 'direct' and not has_adapter:
        raise ValueError(f"[{TABLE}] weights = 'direct' needs an [adapter]: it keeps base weights in fp8, and a "
                         "full fine-tune trains the base weights, which cannot stay fp8. Use weights = 'dequantize'")
    if table['weights'] == 'direct' and config['adapter'].get('use_dora', False):
        raise ValueError(f"[{TABLE}] weights = 'direct' cannot be combined with [adapter] use_dora = true: DoRA "
                         "needs the norm of each base weight row, which stored fp8 weights cannot compute. Use "
                         "weights = 'dequantize' or use_dora = false")
    if table['cache_frozen_weights']:
        if not has_adapter:
            raise ValueError(f'[{TABLE}] cache_frozen_weights applies only to frozen base weights under an '
                             '[adapter]; a full fine-tune changes every weight each step')
        if config.get('blocks_to_swap', 0):
            raise ValueError(f'[{TABLE}] cache_frozen_weights cannot be combined with blocks_to_swap: the cached fp8 '
                             'copies would stay on the GPU while their blocks are swapped out')
    if table['attention']:
        if config.get('compile', False):
            raise ValueError(f'[{TABLE}] attention = true cannot be combined with compile = true: torch.compile would '
                             'trace through the attention replacement and its per-call checks; turn one off')
        if config['model']['type'] == 'cosmos':
            raise ValueError(f'[{TABLE}] attention = true does not apply to cosmos: Cosmos runs its attention '
                             'through its own Transformer Engine modules, not PyTorch SDPA or flash-attn')
    return table


def configure_settings(config):
    """Turns fp8 training on for this process (after apply_fp8_training_config_defaults)."""
    global _settings
    _settings = Settings(config[TABLE]) if TABLE in config else None
    return _settings


def enabled():
    return _settings is not None


def attention_enabled():
    return _settings is not None and _settings.attention


def _is_main():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _print(line):
    if _is_main():
        print(f'[fp8] {line}')


def check_environment(config):
    """Imports TE and checks this GPU and software can train with the configured recipes; called
    once per process before anything loads (not by the runs that only cache)."""
    global _backend
    settings = configure_settings(config)
    if settings is None:
        return
    if _backend is None:
        try:
            _backend = TransformerEngineBackend()
        except ImportError as error:
            raise RuntimeError(f'[{TABLE}] needs NVIDIA Transformer Engine, which did not import ({error}). Install '
                               f'it with {INSTALL_HINT}') from error
    if not torch.cuda.is_available():
        raise RuntimeError(f'[{TABLE}] needs a CUDA GPU')
    _backend.check(settings)


class TransformerEngineBackend:
    """Everything that touches Transformer Engine, so the rest of this module can run on a stand-in."""

    def __init__(self):
        import transformer_engine
        import transformer_engine.pytorch as te
        import transformer_engine_torch as tex
        from transformer_engine.common import recipe
        from transformer_engine.pytorch.quantization import FP8GlobalStateManager
        self.te, self.tex, self.recipe_module, self.state = te, tex, recipe, FP8GlobalStateManager
        self.version = getattr(transformer_engine, '__version__', '?')
        self._recipes = {}
        self._scope_contexts = []

    def device_ok(self, tensor):
        return tensor.is_cuda

    def check(self, settings):
        major, minor = torch.cuda.get_device_capability()
        gpu = torch.cuda.get_device_name()
        cudnn = torch.backends.cudnn.version() or 0
        _print(f'Transformer Engine {self.version}, PyTorch {torch.__version__} (CUDA {torch.version.cuda}), '
               f'cuDNN {cudnn}, {gpu} (compute capability {major}.{minor})')
        ok, reason = self.te.is_fp8_available(return_reason=True)
        if not ok:
            raise RuntimeError(f'[{TABLE}] {gpu} cannot run fp8: {reason}. FP8 needs Ada (8.9), Hopper (9.0) or newer')
        if cudnn < MIN_CUDNN:
            raise RuntimeError(f'[{TABLE}] Transformer Engine needs cuDNN 9.12 or newer; PyTorch here has cuDNN {cudnn}')
        if settings.recipe == 'block':
            ok, reason = self.te.is_fp8_block_scaling_available(return_reason=True)
            if not ok:
                raise RuntimeError(
                    f"[{TABLE}] recipe = 'block' cannot run here: {reason} (PyTorch's CUDA is {torch.version.cuda}; "
                    "block scaling needs Hopper and CUDA 12.9 or newer). Set recipe = 'current' (one scale per "
                    "tensor), which runs on every fp8 GPU")
        if settings.attention and (major, minor) < (9, 0):
            raise RuntimeError(f"[{TABLE}] attention = true: {gpu} (compute capability {major}.{minor}) has no "
                               "cuDNN fp8 attention for training; it needs Hopper (9.0) or newer. Turn attention off")

    def linear_recipe(self, name):
        if name not in self._recipes:
            r = self.recipe_module
            if name == 'block':
                self._recipes[name] = r.Float8BlockScaling(fp8_format=r.Format.E4M3)
            elif name == 'current':
                self._recipes[name] = r.Float8CurrentScaling(fp8_format=r.Format.HYBRID)
            else:
                raise ValueError(name)
        return self._recipes[name]

    def attention_recipe(self, history):
        key = ('attention', history)
        if key not in self._recipes:
            r = self.recipe_module
            # reduce_amax False: an all-reduce over the world group at every scale update would
            # hang pipeline stages that hold different layers; each rank keeps its own scales.
            self._recipes[key] = r.DelayedScaling(fp8_format=r.Format.HYBRID, amax_history_len=history,
                                                  amax_compute_algo='max', reduce_amax=False, fp8_dpa=True,
                                                  fp8_mha=False)
        return self._recipes[key]

    def make_linear(self, in_features, out_features, bias):
        # On the meta device: the layer's own weight and bias are put in for each call.
        return self.te.Linear(in_features, out_features, bias=bias, params_dtype=torch.bfloat16, device='meta')

    def run_linear(self, module, recipe, x, weight, bias, cache):
        placeholders = dict(module._parameters)
        module._parameters['weight'] = weight
        if bias is not None:
            module._parameters['bias'] = bias
        try:
            with self.te.autocast(enabled=True, recipe=recipe):
                # is_first_microbatch=False keeps the fp8 copy of the weight in the module's
                # workspace and reuses it from then on (frozen weights only).
                return module(x, is_first_microbatch=False if cache else None)
        finally:
            # Do not keep this call's (possibly cast) weight alive until the next call.
            module._parameters.update(placeholders)

    def fp8_weight(self, data, scale_inv, dtype):
        """A TE Float8Tensor over stored e4m3 data (dequantized value = data * scale_inv)."""
        from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer, Float8Tensor
        quantizer = Float8CurrentScalingQuantizer(self.tex.DType.kFloat8E4M3, device=data.device)
        return Float8Tensor(shape=data.shape, dtype=dtype, data=data.view(torch.uint8), fp8_scale_inv=scale_inv,
                            fp8_dtype=self.tex.DType.kFloat8E4M3, quantizer=quantizer, requires_grad=False,
                            device=data.device)

    def thd_ok(self, tensor):
        """Whether TE trains fp8 attention on packed (thd) inputs here: Blackwell (sm100) and newer
        with cuDNN 9.23 or newer (TE's fused_attn.cpp); before it they run padded."""
        major, _ = torch.cuda.get_device_capability(tensor.device)
        return major >= 10 and (torch.backends.cudnn.version() or 0) >= 92300

    def make_attention(self, heads, kv_heads, head_dim, layout, mask_type, scale):
        return self.te.DotProductAttention(heads, head_dim, num_gqa_groups=kv_heads, attention_dropout=0.0,
                                           qkv_format=layout, attn_mask_type=mask_type, softmax_scale=scale)

    def prime_attention(self, module, recipe, amax_fwd, amax_bwd):
        """Sets a new DotProductAttention's scales from measured amaxes before its first fp8 call."""
        from transformer_engine.pytorch.cpp_extensions.fused_attn import (META_DO, META_DP, META_DQKV, META_O,
                                                                           META_QKV, META_S)
        # no_grad: leaving an outermost TE autocast with gradients on would update every scale.
        with torch.no_grad(), self.te.autocast(enabled=True, recipe=recipe):
            module.init_fp8_metadata(num_gemms=3)
        meta = module.fp8_meta
        margin = 2.0 ** recipe.margin

        def put(state, index, amax, fp8_max):
            scale, index = state.scale, int(index)
            amax = amax.to(scale.device, torch.float32)
            scale[index] = torch.where(amax > 0, fp8_max / amax / margin, scale[index])

        fwd, bwd = meta['scaling_fwd'], meta['scaling_bwd']
        max_fwd, max_bwd = meta['fp8_max_fwd'], meta['fp8_max_bwd']
        put(fwd, META_QKV, amax_fwd[0], max_fwd)
        put(fwd, META_S, amax_fwd[1], max_fwd)
        put(fwd, META_O, amax_fwd[2], max_fwd)
        if amax_bwd is not None:
            put(bwd, META_DO, amax_bwd[0], max_bwd)
            put(bwd, META_DP, amax_bwd[1], max_bwd)
            put(bwd, META_DQKV, amax_bwd[2], max_bwd)

    def run_attention(self, module, recipe, q, k, v, mask_type, attention_mask=None, cu_seqlens_q=None,
                      cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None):
        kwargs = {}
        if cu_seqlens_q is not None:
            kwargs.update(cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=int(max_seqlen_q),
                          max_seqlen_kv=int(max_seqlen_kv))
        with self.te.autocast(enabled=True, recipe=recipe):
            return module(q, k, v, attention_mask=attention_mask, attn_mask_type=mask_type, **kwargs)

    def begin_scope(self):
        context = self.te.autocast(enabled=False)
        context.__enter__()
        # Keeps TE's modules from updating scales in their backward: step_scope updates them once.
        self.state.quantization_state.is_first_fp8_module = False
        self._scope_contexts.append(context)

    def end_scope(self, update):
        self._scope_contexts.pop().__exit__(None, None, None)
        if update:
            self.state.reduce_and_update_fp8_tensors(forward=True)
            self.state.reduce_and_update_fp8_tensors(forward=False)

    def discard_amax(self, module):
        """Forgets the amaxes an attention module recorded since the last update (an eval pass):
        TE starts each update's history row at zero and fills it with the maxima it sees."""
        for key in ('scaling_fwd', 'scaling_bwd'):
            state = module.fp8_meta.get(key)
            if state is not None:
                state.amax_history[0].zero_()


@contextlib.contextmanager
def disabled():
    """bf16 for everything run inside (sampling previews)."""
    global _disabled
    _disabled += 1
    try:
        yield
    finally:
        _disabled -= 1


@contextlib.contextmanager
def step_scope(training):
    """Wraps one training step (training=True) or one eval batch: fp8 attention runs only inside,
    with the same scales throughout; a training step's end updates them and starts the call sites
    whose bf16 calibration step is complete."""
    global _scope
    if not attention_enabled() or _scope is not None:
        yield
        return
    _scope = training
    _backend.begin_scope()
    completed = False
    try:
        yield
        completed = True
    finally:
        _scope = None
        _backend.end_scope(update=training and completed)
        if not training:
            # Eval does not move the scales: what it measured is not kept for the next update.
            for site in _attention.sites.values():
                if site.state == 'active':
                    _backend.discard_amax(site.module)
    if training:
        _attention.after_training_step()


def _note(counter, reason, line):
    counter[reason] += 1
    if (id(counter), reason) not in _noted:
        _noted.add((id(counter), reason))
        _print(line)


class Fp8Linear(nn.Linear):
    """An nn.Linear whose GEMMs run in fp8 through a hidden te.Linear.

    Holds the original layer's weight and bias Parameters; the original module (hidden, so not in
    the state dict) runs the bf16 path for calls fp8 cannot take, with its own forward (ComfyUI's
    weight casting, quantized-weight dispatch).
    """

    def __init__(self, source, name, recipe, mode, weight=None, scale_inv=None, cache=False):
        nn.Module.__init__(self)
        self.in_features, self.out_features = source.in_features, source.out_features
        # Through _parameters: a quantized weight is not always an nn.Parameter.
        self._parameters['weight'] = source.weight if weight is None else weight
        self._parameters['bias'] = source.bias
        self._source = (source,)
        self._dequantized = weight is not None
        self._te = None
        self._scale_inv = scale_inv
        self.fp8_name = name
        self.fp8_recipe = recipe
        self.fp8_mode = mode
        self.fp8_cache = cache
        # None until the first fp8 call works; False once it failed (bf16 from then on).
        self.fp8_ok = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            source = self.__dict__.get('_source')
            if source is None or name.startswith('__'):
                raise
            return getattr(source[0], name)

    def extra_repr(self):
        return (f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, '
                f'fp8={self.fp8_recipe}, weights={self.fp8_mode}')

    def _bf16(self, x):
        if self._dequantized:
            weight, bias = self.weight, self.bias
            if not torch.is_autocast_enabled(x.device.type):
                weight = weight.to(x.dtype)
                bias = bias.to(x.dtype) if bias is not None else None
            return F.linear(x, weight, bias)
        source = self._source[0]
        # Block swapping and PEFT keep these Parameter objects, but a reassignment would not reach
        # the hidden module.
        source._parameters['weight'] = self.weight
        if 'bias' in source._parameters:
            source._parameters['bias'] = self.bias
        return source(x)

    def _skip_reason(self, x):
        if _disabled:
            return 'sampling'
        if self.fp8_ok is False:
            return 'layers where TE failed'
        if not _backend.device_ok(x) or x.dtype not in HALF + (torch.float32,):
            return 'inputs that are not float CUDA tensors'
        if x.shape[-1] != self.in_features or x.numel() == 0 or (x.numel() // x.shape[-1]) % 16:
            # the weight-gradient GEMM reduces over the tokens
            return 'calls whose token count is not a multiple of 16'
        return None

    def forward(self, x):
        reason = self._skip_reason(x)
        if reason:
            _note(_linear_counts, reason, f'Linear calls that run bf16: {reason} (first: {self.fp8_name}, input '
                                          f'{tuple(x.shape)}); further ones are counted')
            return self._bf16(x)
        try:
            out = self._fp8(x)
        except Exception as error:
            if self.fp8_ok or isinstance(error, torch.OutOfMemoryError):
                raise
            self.fp8_ok = False
            _print(f'WARNING: {self.fp8_name} cannot run in fp8 ({type(error).__name__}: {error}); it runs bf16 '
                   'from here on')
            _linear_counts['layers where TE failed'] += 1
            return self._bf16(x)
        self.fp8_ok = True
        _linear_counts['fp8'] += 1
        return out

    def _fp8(self, x):
        if self._te is None:
            self._te = (_backend.make_linear(self.in_features, self.out_features, self.bias is not None),)
        weight, bias = self.weight, self.bias
        autocast = torch.is_autocast_enabled(x.device.type)
        if autocast:
            dtype = torch.get_autocast_dtype(x.device.type)
        elif x.dtype in HALF:
            dtype = x.dtype
        elif weight.dtype in HALF:
            dtype = weight.dtype
        else:
            dtype = torch.bfloat16
        inp = x if autocast else x.to(dtype)
        if self.fp8_mode == 'direct':
            data = weight._qdata if _is_quantized_tensor(weight) else weight.data
            if self._scale_inv.device != data.device:
                # Loaded on the CPU, moved by DeepSpeed or block swapping with the weight.
                self._scale_inv = self._scale_inv.to(data.device)
            weight = _backend.fp8_weight(data, self._scale_inv, dtype)
        elif not autocast:
            weight = weight.to(dtype)
        if bias is not None and not autocast:
            bias = bias.to(dtype)
        recipe = _backend.linear_recipe(self.fp8_recipe)
        cache = self.fp8_cache and self.fp8_mode != 'direct' and not self.weight.requires_grad
        out = _backend.run_linear(self._te[0], recipe, inp, weight, bias, cache)
        return out if out.dtype == x.dtype or autocast else out.to(x.dtype)


def _is_quantized_tensor(tensor):
    return hasattr(tensor, '_qdata') and hasattr(tensor, '_params')


def _pre_quant_scale(module, value):
    """ComfyUI's quantized Linear multiplies its input by pre_quant_scale (ModelOpt checkpoints):
    as a (1, in_features) factor on the weight, or 1."""
    scale = getattr(module, 'pre_quant_scale', None)
    if scale is None:
        return 1
    return scale.detach().to(value.device, torch.float32).reshape(1, -1)


def _direct_scale(module, weight):
    """(e4m3 data, scale_inv as a 1-element float32 tensor) for a weight 'direct' can use, else None."""
    if getattr(module, 'pre_quant_scale', None) is not None:
        return None
    if _is_quantized_tensor(weight):
        layout = getattr(weight, '_layout_cls', '')
        scale = getattr(weight._params, 'scale', None)
        if (layout in ('TensorCoreFP8Layout', 'TensorCoreFP8E4M3Layout') and weight._qdata.dtype == torch.float8_e4m3fn
                and scale is not None and scale.numel() == 1
                and tuple(weight._qdata.shape) == tuple(getattr(weight._params, 'orig_shape', weight._qdata.shape))):
            return weight._qdata, scale.detach().reshape(1).float()
        return None
    if weight.dtype == torch.float8_e4m3fn:
        scale = getattr(module, 'scale_weight', None)
        if scale is None:
            scale = torch.ones(1, dtype=torch.float32, device=weight.device)
        elif scale.numel() != 1:
            return None
        return weight.data, scale.detach().reshape(1).float()
    return None


def _describe(weight):
    if _is_quantized_tensor(weight):
        return str(getattr(weight, '_layout_cls', 'quantized'))
    return str(weight.dtype).replace('torch.', '')


def _dequantize(module, weight, dtype):
    if _is_quantized_tensor(weight):
        value = weight.dequantize().float()
    else:
        value = weight.data.to(torch.float32)
        scale = getattr(module, 'scale_weight', None)
        if scale is not None:
            value = value * scale.to(value.device, torch.float32)
    value = value * _pre_quant_scale(module, value)
    p = nn.Parameter(value.to(dtype), requires_grad=weight.requires_grad)
    if hasattr(weight, 'original_name'):
        p.original_name = weight.original_name
    return p


def target_classes(model):
    """The block classes whose Linear layers (and attention calls) fp8 covers for a model."""
    classes = getattr(model, 'fp8_target_modules', None) or getattr(model, 'adapter_target_modules', None)
    if not classes:
        raise ValueError(f'[{TABLE}] {type(model).__name__} names no transformer block classes '
                         '(adapter_target_modules / fp8_target_modules)')
    return list(classes)


def convert(target_model, classes, dtype, keep=()):
    """Replaces the Linear layers inside the blocks of `target_model` whose class name is in
    `classes`; returns a summary Counter. Layers whose name matches a regex of `keep` (the model's
    fp8_keep_modules) stay as they are whatever `exclude` says."""
    settings = _settings
    summary = collections.Counter()
    formats = collections.Counter()
    unfit = []
    names = []
    for block_name, block in target_model.named_modules():
        if type(block).__name__ not in classes:
            continue
        for name, module in block.named_modules(prefix=block_name):
            if isinstance(module, nn.Linear) and not isinstance(module, Fp8Linear) and name not in names:
                names.append(name)
    modules = dict(target_model.named_modules())
    for name in names:
        module = modules[name]
        if any(p.search(name) for p in keep):
            summary['kept'] += 1
            continue
        if any(p.search(name) for p in settings.exclude):
            summary['excluded'] += 1
            continue
        if module.in_features % 16 or module.out_features % 16:
            summary['unfit'] += 1
            unfit.append(f'{name} ({module.in_features}->{module.out_features})')
            continue
        weight = module.weight
        recipe, mode, replacement, scale_inv = settings.recipe, 'quantize', None, None
        stored = _is_quantized_tensor(weight) or weight.dtype in FP8_DTYPES
        if stored:
            direct = _direct_scale(module, weight) if settings.weights == 'direct' else None
            if direct is not None:
                mode, recipe, scale_inv = 'direct', 'current', direct[1]
                summary['direct'] += 1
            else:
                replacement = _dequantize(module, weight, dtype)
                formats[_describe(weight)] += 1
                summary['dequantized'] += 1
        if replacement is not None:
            # The hidden original then holds no reference to the stored weight, which is freed.
            module._parameters['weight'] = replacement
        layer = Fp8Linear(module, name, recipe, mode, weight=replacement, scale_inv=scale_inv,
                          cache=settings.cache_frozen_weights)
        parent_name, _, child = name.rpartition('.')
        parent = modules[parent_name] if parent_name else target_model
        parent._modules[child] = layer
        summary['fp8'] += 1
    line = f"{summary['fp8']} Linear layers run their GEMMs in fp8 (recipe '{settings.recipe}'"
    if summary['direct']:
        line += f", {summary['direct']} of them from their stored e4m3 weights with recipe 'current'"
    line += ')'
    if formats:
        line += '; dequantized at load to ' + str(dtype).replace('torch.', '') + ': ' + ', '.join(
            f'{n} {fmt}' for fmt, n in formats.most_common())
    if summary['kept']:
        line += f"; {summary['kept']} kept in their dtype by the model (fp8_keep_modules)"
    if summary['excluded']:
        line += f"; {summary['excluded']} excluded by exclude"
    if unfit:
        line += (f"; {len(unfit)} stay bf16, their sizes do not fit TE's fp8 GEMM (in_features and out_features "
                 f"multiples of 16): {', '.join(unfit[:4])}{', ...' if len(unfit) > 4 else ''}")
    _print(line)
    if summary['fp8'] == 0:
        raise ValueError(f'[{TABLE}] found no Linear layer to run in fp8 in the {", ".join(classes)} blocks')
    return summary


def prepare_model(model):
    """Converts a loaded model's Linear layers (before its adapter is built) and, with attention
    on, hooks its blocks. Called by train.py after load_diffusion_model()."""
    if _settings is None:
        return None
    target = model.fp8_target_model()
    classes = target_classes(model)
    keep = [re.compile(p) for p in getattr(model, 'fp8_keep_modules', ())]
    summary = convert(target, classes, model.model_config['dtype'], keep)
    if _settings.attention:
        _attention.hook_blocks(target, classes)
    return summary


class _Block:
    __slots__ = ('name', 'calls')

    def __init__(self, name):
        self.name, self.calls = name, 0


class _Site:
    """One attention call site: a block, the call's place in it, and the call's kind."""

    def __init__(self, key, signature, device):
        self.key, self.signature, self.device = key, signature, device
        self.state = 'calibrating'
        self.module = None
        self.amax_fwd = None
        self.amax_bwd = None
        self.forward_seen = False
        self.grad_seen = False
        self.ok = False

    def record_forward(self, q, k, v, out):
        with torch.no_grad():
            qkv = torch.stack([q.detach().abs().amax(), k.detach().abs().amax(), v.detach().abs().amax()]).amax()
            amax = torch.stack([qkv.float(), torch.ones((), device=qkv.device),
                                out.detach().abs().amax().float()])
            self.amax_fwd = amax if self.amax_fwd is None else torch.maximum(self.amax_fwd, amax)
        self.forward_seen = True

    def record_backward(self, index, value):
        value = value.detach().float()
        if self.amax_bwd is None:
            self.amax_bwd = torch.zeros(3, device=value.device)
        self.amax_bwd[index] = torch.maximum(self.amax_bwd[index], value)

    def hook(self, q, k, v, out):
        """Gradient hooks that measure what the fp8 backward will see (calibration only)."""
        if not out.requires_grad:
            return
        self.grad_seen = True
        with torch.no_grad():
            v_norm = v.detach().float().norm(dim=-1).amax()

        def on_output_grad(grad, v_norm=v_norm):
            with torch.no_grad():
                self.record_backward(0, grad.abs().amax())
                # The softmax gradient P * (dO V^T - D) is at most 2 max|dO V^T|, which Cauchy-Schwarz
                # bounds by 2 max|dO_i| max|V_j| (row norms): a bound, which wastes a little of e5m2's range.
                self.record_backward(1, 2 * grad.float().norm(dim=-1).amax() * v_norm)

        def on_input_grad(grad):
            with torch.no_grad():
                self.record_backward(2, grad.abs().amax())

        out.register_hook(on_output_grad)
        for t in (q, k, v):
            if t.requires_grad:
                t.register_hook(on_input_grad)

    def ready(self):
        return self.forward_seen and (self.amax_bwd is not None or not self.grad_seen)


class AttentionSites:
    def __init__(self):
        self.sites = {}
        self.stack = []
        self.hooked = 0
        # Set while a call runs: attention that TE or a fallback runs inside it (through the SDPA
        # replacement) is that call's own, not another site.
        self.running = False

    def hook_blocks(self, target_model, classes):
        def enter(module, args, name):
            self.stack.append(_Block(name))

        def leave(module, args, output):
            if self.stack:
                self.stack.pop()

        for name, module in target_model.named_modules():
            if type(module).__name__ in classes:
                module.register_forward_pre_hook(lambda m, a, name=name: enter(m, a, name))
                module.register_forward_hook(leave, always_call=True)
                self.hooked += 1

    def after_training_step(self):
        started = 0
        for site in self.sites.values():
            if site.state == 'calibrating' and site.ready():
                heads, kv_heads, head_dim, layout, mask_type, scale = site.signature[:6]
                recipe = _backend.attention_recipe(_settings.attention_amax_history)
                with torch.cuda.device(site.device) if site.device.type == 'cuda' else contextlib.nullcontext():
                    site.module = _backend.make_attention(heads, kv_heads, head_dim, layout, mask_type, scale)
                    _backend.prime_attention(site.module, recipe, site.amax_fwd, site.amax_bwd)
                site.state = 'active'
                started += 1
        if started:
            _print(f'{started} attention call sites finished their bf16 calibration step and run fp8 from the next '
                   'step on')

    def _skip(self, reason, q):
        _note(_attention_counts, reason, f'attention calls that run without fp8: {reason} (q {tuple(q.shape)}, '
                                         f'{q.dtype}); further ones are counted')

    def skip(self, reason, q, fallback):
        """fallback(), counting the call when it would otherwise have been a candidate."""
        if not (_disabled or _scope is None or self.running):
            self._skip(reason, q)
        return fallback()

    def run(self, layout, q, k, v, fallback, mask_type='no_mask', scale=None, kv_pack=None, cu_seqlens_q=None,
            cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, post=None, observe=None):
        """The attention output: fp8 through TE where this call can take it, else fallback().

        post() turns TE's output into the layout fallback() returns; observe is the (q, k, v)
        fallback() computes from, when not these (the calibration hooks their gradients)."""
        if _disabled or _scope is None or self.running:
            return fallback()
        self.running = True
        try:
            out = self._run(layout, q, k, v, fallback, mask_type, scale, kv_pack, cu_seqlens_q, cu_seqlens_kv,
                            max_seqlen_q, max_seqlen_kv, observe)
        finally:
            self.running = False
        if isinstance(out, _Fp8Out):
            return post(out.value) if post is not None else out.value
        return out

    def _run(self, layout, q, k, v, fallback, mask_type, scale, kv_pack, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q,
             max_seqlen_kv, observe):
        if not self.stack:
            self._skip('calls outside the transformer blocks', q)
            return fallback()
        block = self.stack[-1]
        index = block.calls
        block.calls += 1
        reason = self._reason(layout, q, k, v)
        if reason:
            self._skip(reason, q)
            return fallback()
        heads, head_dim = (q.shape[1], q.shape[3]) if layout == 'bhsd' else (q.shape[-2], q.shape[-1])
        kv_heads = k.shape[1] if layout == 'bhsd' else k.shape[-2]
        signature = (heads, kv_heads, head_dim, self._te_layout(layout, q), mask_type, scale, q.dtype)
        key = (block.name, index, signature)
        site = self.sites.get(key)
        if site is None:
            site = self.sites[key] = _Site(key, signature, q.device)
        if site.state == 'calibrating':
            out = fallback()
            seen = observe if observe is not None else (q, k, v)
            site.record_forward(*seen, out)
            if _scope:
                site.hook(*seen, out)
            _attention_counts['bf16 calibration'] += 1
            return out
        if site.state == 'off':
            _attention_counts['sites where TE failed'] += 1
            return fallback()
        try:
            out = self._fp8(site, layout, q, k, v, kv_pack, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
        except Exception as error:
            if site.ok or isinstance(error, torch.OutOfMemoryError):
                raise
            site.state = 'off'
            _print(f'WARNING: attention in {block.name} (call {index}, {mask_type}, q {tuple(q.shape)}) cannot run '
                   f'in fp8 ({type(error).__name__}: {error}); it runs without fp8 from here on')
            _attention_counts['sites where TE failed'] += 1
            return fallback()
        site.ok = True
        _attention_counts['fp8'] += 1
        return _Fp8Out(out)

    def _reason(self, layout, q, k, v):
        if any(type(t) is not torch.Tensor for t in (q, k, v)):
            return 'tensor subclasses'
        if not all(_backend.device_ok(t) for t in (q, k, v)):
            return 'inputs not on the GPU'
        if not (q.dtype == k.dtype == v.dtype and q.dtype in HALF):
            return 'inputs that are not all bfloat16 / float16'
        d = q.shape[-1]
        if k.shape[-1] != d or v.shape[-1] != d:
            return 'different query and value head sizes'
        if d % 16 or d > 256:
            return f'head_dim {d}'
        heads = q.shape[1] if layout == 'bhsd' else q.shape[-2]
        kv_heads = k.shape[1] if layout == 'bhsd' else k.shape[-2]
        if heads % kv_heads:
            return 'query heads that are not a multiple of key heads'
        return None

    @staticmethod
    def _te_layout(layout, q):
        """The qkv_format TE runs a call in. Packed (thd) calls run padded (bshd) where TE has no
        fp8 backward for thd (before Blackwell)."""
        if layout == 'bhsd' or (layout == 'thd' and not _backend.thd_ok(q)):
            return 'bshd'
        return layout

    def _fp8(self, site, layout, q, k, v, kv_pack, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv):
        recipe = _backend.attention_recipe(_settings.attention_amax_history)
        mask_type = site.signature[4]
        if layout == 'thd' and site.signature[3] == 'bshd':
            return self._fp8_padded(site, recipe, mask_type, q, k, v, cu_seqlens_q, cu_seqlens_kv, int(max_seqlen_q),
                                    int(max_seqlen_kv))
        if layout == 'bhsd':
            qt, kt, vt = (t.transpose(1, 2).contiguous() for t in (q, k, v))
        else:
            qt, kt, vt = q.contiguous(), k.contiguous(), v.contiguous()
        mask = None
        if kv_pack is not None:
            # TE reads only how many keys each item keeps from a padding mask (the first ones), so
            # the kept keys are gathered to the front, as FlashAttention's varlen path does.
            batch, keys, cu_seqlens_kv, max_seqlen_kv = kv_pack
            b, lq = qt.shape[:2]
            rows, mask_kv = _padding(cu_seqlens_kv, max_seqlen_kv, batch.shape[0])

            def gather(t):
                packed = t[batch, keys]
                full = packed.new_zeros((b * max_seqlen_kv, *packed.shape[1:])).index_copy(0, rows, packed)
                return full.view(b, max_seqlen_kv, *packed.shape[1:])

            kt, vt = gather(kt), gather(vt)
            mask = (torch.zeros((b, 1, 1, lq), dtype=torch.bool, device=q.device), mask_kv)
        out = _backend.run_attention(site.module, recipe, qt, kt, vt, mask_type, attention_mask=mask,
                                     cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv,
                                     max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv)
        out = out.view(*qt.shape[:-2], qt.shape[-2], vt.shape[-1])
        return out.transpose(1, 2) if layout == 'bhsd' else out

    def _fp8_padded(self, site, recipe, mask_type, q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv):
        """A packed (total, H, D) call run as padded (B, S, H, D) with a query and a key padding
        mask; returns the packed output."""
        rows_q, mask_q = _padding(cu_seqlens_q, max_seqlen_q, q.shape[0])
        rows_kv, mask_kv = _padding(cu_seqlens_kv, max_seqlen_kv, k.shape[0])
        b = cu_seqlens_q.shape[0] - 1

        def pad(t, rows, length):
            full = t.new_zeros((b * length, *t.shape[1:])).index_copy(0, rows, t)
            return full.view(b, length, *t.shape[1:])

        qp = pad(q, rows_q, max_seqlen_q)
        kp, vp = pad(k, rows_kv, max_seqlen_kv), pad(v, rows_kv, max_seqlen_kv)
        out = _backend.run_attention(site.module, recipe, qp, kp, vp, mask_type, attention_mask=(mask_q, mask_kv))
        return out.reshape(b * max_seqlen_q, q.shape[-2], v.shape[-1]).index_select(0, rows_q)


def _padding(cu_seqlens, max_seqlen, total):
    """For packed sequences: the row of each packed token in the padded (B * max_seqlen) layout,
    and the padding mask (B, 1, 1, max_seqlen), True where padded, as TE takes it."""
    cu = cu_seqlens.long()
    lengths = cu[1:] - cu[:-1]
    b = lengths.shape[0]
    batch = torch.repeat_interleave(torch.arange(b, device=cu.device), lengths, output_size=total)
    rows = batch * max_seqlen + torch.arange(total, device=cu.device) - cu[:-1][batch]
    positions = torch.arange(max_seqlen, device=cu.device)
    mask = (positions[None, :] >= lengths[:, None]).view(b, 1, 1, max_seqlen)
    return rows, mask


class _Fp8Out:
    # TE's output, which post() turns into the fallback's layout once the call is done.
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value


_attention = AttentionSites()


def sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa, extra, fallback, mask_info):
    """For utils/attention.py's SDPA replacement: TE fp8 where the call fits. mask_info(attn_mask)
    gives the key padding of a mask (an object with .dense / .batch, .key) or the reason it is none."""
    if extra:
        return fallback()
    if dropout_p:
        return _attention.skip('attention dropout', query, fallback)
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return _attention.skip('inputs that are not 4-D', query, fallback)
    lq, lk = query.shape[2], key.shape[2]
    if key.shape[1] != query.shape[1] and not enable_gqa:
        return _attention.skip('different numbers of query and key heads', query, fallback)
    mask_type, keep = 'no_mask', None
    if is_causal:
        if attn_mask is not None or lq != lk:
            return _attention.skip('is_causal with a mask or different query and key lengths', query,
                                          fallback)
        mask_type = 'causal'
    elif attn_mask is not None:
        info = mask_info(attn_mask, query.shape[0], query.shape[1], lq, lk)
        if isinstance(info, str):
            return _attention.skip(info, query, fallback)
        if not info.dense:
            keep = (info.batch, info.key, info.cu_seqlens_k, int(info.max_seqlen_k))
            mask_type = 'padding'
    return _attention.run('bhsd', query, key, value, fallback, mask_type=mask_type, scale=scale, kv_pack=keep)


def dense(q, k, v, softmax_scale, causal, fallback):
    """(B, L, H, D) attention: TE fp8 where the call fits, else fallback()."""
    if not attention_enabled():
        return fallback()
    if causal and q.shape[1] != k.shape[1]:
        return _attention.skip('causal with different query and key lengths', q, fallback)
    return _attention.run('bshd', q, k, v, fallback, mask_type='causal' if causal else 'no_mask', scale=softmax_scale)


def varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, fallback,
           post=None, observe=None):
    """Packed (total, H, D) attention with flash-attn's varlen arguments: TE fp8 where it fits."""
    if not attention_enabled():
        return fallback()
    if causal and not torch.equal(cu_seqlens_q, cu_seqlens_k):
        return _attention.skip('causal with different query and key lengths', q, fallback)
    return _attention.run('thd', q, k, v, fallback, mask_type='padding_causal' if causal else 'padding',
                          scale=softmax_scale, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_k,
                          max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_k, post=post, observe=observe)


def counts():
    return dict(_linear_counts), dict(_attention_counts)


def reset(backend=None):
    """Back to the state of a fresh process (fp8 off), with `backend` as the TE wrapper."""
    global _settings, _backend, _disabled, _scope, _attention
    _settings, _backend, _disabled, _scope = None, backend, 0, None
    _attention = AttentionSites()
    reset_counts()


def reset_counts():
    _linear_counts.clear()
    _attention_counts.clear()
    _noted.clear()


def _gather(counter):
    total = collections.Counter(counter)
    if torch.distributed.is_initialized():
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, dict(counter))
        total = collections.Counter()
        for c in gathered:
            total.update(c)
    return total


def report():
    """Prints, for all ranks together, how many Linear and attention calls ran fp8 and why the
    others did not."""
    if _settings is None:
        return
    for what, counter in (('Linear', _linear_counts), ('attention', _attention_counts)):
        if what == 'attention' and not _settings.attention:
            continue
        totals = _gather(counter)
        if not _is_main():
            continue
        fp8 = totals.pop('fp8', 0)
        line = f'so far, all ranks: {fp8} {what} calls ran fp8'
        if totals:
            line += '; without fp8: ' + ', '.join(f'{n} {reason}' for reason, n in totals.most_common())
        _print(line)
        if what == 'attention' and fp8 == 0 and not totals.get('bf16 calibration'):
            _print('WARNING: no attention call reached fp8 attention: this model\'s attention does not go through '
                   'PyTorch SDPA or the flash-attn entry points it covers')
