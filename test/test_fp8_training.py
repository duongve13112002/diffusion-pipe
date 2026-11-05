"""[fp8_training] (utils/fp8_training.py) on the CPU, with a stand-in for Transformer Engine.

The stand-in emulates fp8 by rounding through torch's float8 dtypes with one scale per tensor, so
results are close to bf16 but not equal, and records what the code under test asked TE to do.
tools/check_fp8_training.py runs the real TE on a GPU.
"""

import copy
import re
import json
import math
import pathlib
import sys

import peft
import pytest
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F

from utils import attention as attention_util
from utils import fp8_training

ROOT = pathlib.Path(__file__).resolve().parent.parent
REAL_SDPA = F.scaled_dot_product_attention


def emulate_fp8(t, dtype=torch.float8_e4m3fn):
    amax = t.detach().abs().amax().float().clamp_min(1e-12)
    scale = torch.finfo(dtype).max / amax
    rounded = ((t.detach().float() * scale).to(dtype).float() / scale).to(t.dtype)
    # Straight-through: the gradient must not pass through a float8 cast, which would round it too.
    return t + (rounded - t).detach()


class FakeWeight:
    """What TransformerEngineBackend.fp8_weight returns: stored e4m3 data and its scale_inv."""

    def __init__(self, data, scale_inv, dtype):
        self.data, self.scale_inv, self.dtype = data, scale_inv, dtype

    def dequantize(self):
        return (self.data.float() * self.scale_inv).to(self.dtype)


class FakeLinear:
    def __init__(self, in_features, out_features, bias):
        self.shape = (in_features, out_features, bias)
        self.cached = None


class FakeAttention:
    def __init__(self, signature):
        self.signature = signature
        self.primed = None
        self.calls = 0


class FakeTE:
    def __init__(self):
        self.linear_calls = []
        self.attention_calls = []
        self.updates = 0
        self.scopes = 0
        self.fail_linear = False
        self.fail_attention = False
        # Whether fp8 attention trains on packed (thd) inputs: Blackwell, not Hopper.
        self.thd = True

    def device_ok(self, tensor):
        return True

    def check(self, settings):
        pass

    def linear_recipe(self, name):
        return name

    def attention_recipe(self, history):
        return ('delayed', history)

    def make_linear(self, in_features, out_features, bias):
        return FakeLinear(in_features, out_features, bias)

    def run_linear(self, module, recipe, x, weight, bias, cache):
        if self.fail_linear:
            raise RuntimeError('no fp8 GEMM for this shape')
        self.linear_calls.append(dict(module=module, recipe=recipe, cache=cache, direct=isinstance(weight, FakeWeight),
                                      x_dtype=x.dtype, requires_grad=getattr(weight, 'requires_grad', False),
                                      x_shape=tuple(x.shape)))
        if isinstance(weight, FakeWeight):
            w = weight.dequantize()
        elif cache:
            if module.cached is None:
                module.cached = emulate_fp8(weight.detach())
            w = module.cached
        else:
            w = emulate_fp8(weight)
        return F.linear(emulate_fp8(x), w, bias)

    def fp8_weight(self, data, scale_inv, dtype):
        return FakeWeight(data, scale_inv, dtype)

    def thd_ok(self, tensor):
        return self.thd

    def make_attention(self, heads, kv_heads, head_dim, layout, mask_type, scale):
        return FakeAttention((heads, kv_heads, head_dim, layout, mask_type, scale))

    def prime_attention(self, module, recipe, amax_fwd, amax_bwd):
        module.primed = (recipe, amax_fwd.clone(), None if amax_bwd is None else amax_bwd.clone())

    def run_attention(self, module, recipe, q, k, v, mask_type, attention_mask=None, cu_seqlens_q=None,
                      cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None):
        if self.fail_attention:
            raise RuntimeError('no fused attention backend')
        module.calls += 1
        self.attention_calls.append(dict(module=module, mask_type=mask_type, mask=attention_mask,
                                         cu=cu_seqlens_q, grad=torch.is_grad_enabled()))
        heads, kv_heads, _, layout, _, scale = module.signature
        q, k, v = emulate_fp8(q), emulate_fp8(k), emulate_fp8(v)
        if layout == 'thd':
            outs = []
            for i in range(len(cu_seqlens_q) - 1):
                a, b = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
                c, d = int(cu_seqlens_kv[i]), int(cu_seqlens_kv[i + 1])
                o = F.scaled_dot_product_attention(q[a:b].transpose(0, 1)[None], k[c:d].transpose(0, 1)[None],
                                                   v[c:d].transpose(0, 1)[None], scale=scale,
                                                   is_causal=mask_type == 'padding_causal', enable_gqa=True)
                outs.append(o[0].transpose(0, 1))
            out = torch.cat(outs)
            return out.reshape(out.shape[0], -1)
        mask = None
        if attention_mask is not None:
            # As TE: a padding mask only says how many keys each item keeps, the first ones.
            lengths = (~attention_mask[1]).sum(-1, keepdim=True)
            mask = torch.arange(k.shape[1]) < lengths
        if mask_type == 'padding_causal':
            mask = mask & torch.ones(q.shape[1], k.shape[1], dtype=torch.bool).tril()
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask,
                                             is_causal=mask_type == 'causal', scale=scale, enable_gqa=True)
        out = out.transpose(1, 2)
        return out.reshape(*out.shape[:2], -1)

    def begin_scope(self):
        self.scopes += 1

    def end_scope(self, update):
        self.updates += bool(update)

    def discard_amax(self, module):
        module.discarded = getattr(module, 'discarded', 0) + 1

    def pin_softmax_scale(self, module, recipe):
        module.pinned = getattr(module, 'pinned', 0) + 1


@pytest.fixture
def te():
    backend = FakeTE()
    fp8_training.reset(backend)
    yield backend
    attention_util.uninstall()
    fp8_training.reset()


def settings(**table):
    config = {'model': {'type': 'flux'}, 'adapter': {'type': 'lora'}, 'fp8_training': dict(table)}
    fp8_training.apply_fp8_training_config_defaults(config)
    fp8_training.configure_settings(config)
    return config


class Attention(nn.Module):
    def __init__(self, dim, heads, mask=None):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.mask = mask

    def forward(self, x):
        b, s, _ = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask)
        return self.out(o.transpose(1, 2).reshape(b, s, -1))


class Block(nn.Module):
    def __init__(self, dim=32, heads=2, odd=False):
        super().__init__()
        self.attn = Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.odd = nn.Linear(dim, 5) if odd else None

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)


class Net(nn.Module):
    def __init__(self, blocks=2, dim=32, odd=False):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([Block(dim, odd=odd) for _ in range(blocks)])
        self.final = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        return self.final(x)


class Pipeline:
    adapter_target_modules = ['Block']

    def __init__(self, net):
        self.diffusion_model = net
        self.model_config = {'dtype': torch.bfloat16}

    def fp8_target_model(self):
        return self.diffusion_model


def test_config_defaults_and_refusals():
    config = {'model': {'type': 'flux'}, 'fp8_training': {}}
    table = fp8_training.apply_fp8_training_config_defaults(config)
    assert table == {'recipe': 'block', 'weights': 'dequantize', 'exclude': [], 'cache_frozen_weights': False,
                     'attention': False, 'attention_amax_history': 16, 'attention_min_tokens': 8192}
    assert fp8_training.apply_fp8_training_config_defaults({'model': {'type': 'flux'}}) is None

    def refused(table, match, **top):
        config = {'model': {'type': top.pop('model_type', 'flux')}, 'fp8_training': table, **top}
        with pytest.raises(ValueError, match=match):
            fp8_training.apply_fp8_training_config_defaults(config)

    refused({'recipes': 'block'}, r"unknown keys \['recipes'\]")
    refused({'recipe': 'delayed'}, 'recipe must be one of')
    refused({'weights': 'fp8'}, 'weights must be one of')
    refused({'exclude': ['(']}, 'is not a regular expression')
    refused({'exclude': [1]}, 'list of regular expressions')
    refused({'attention': 'yes'}, "attention must be true, false or 'auto'")
    refused({'attention': 1}, "attention must be true, false or 'auto'")
    refused({'attention': True, 'attention_min_tokens': 4096}, "applies only with attention = 'auto'")
    refused({'attention_min_tokens': 4096}, "applies only with attention = 'auto'")
    refused({'attention': 'auto', 'attention_min_tokens': 0}, 'whole number of tokens')
    refused({'attention': 'auto', 'attention_min_tokens': True}, 'whole number of tokens')
    refused({'attention': 'auto'}, 'compile = true', compile=True)
    refused({'attention_amax_history': 0}, 'from 1 to 1024')
    refused({'attention_amax_history': 8}, 'applies only with attention = true')
    refused({'weights': 'direct'}, "needs an \\[adapter\\]")
    refused({'weights': 'direct'}, 'use_dora', adapter={'use_dora': True})
    refused({'cache_frozen_weights': True}, 'applies only to frozen base weights')
    refused({'cache_frozen_weights': True}, 'blocks_to_swap', adapter={}, blocks_to_swap=4)
    refused({'attention': True}, 'compile = true', compile=True)
    refused({'attention': True}, 'does not apply to cosmos', model_type='cosmos')
    config = {'model': {'type': 'flux'}, 'adapter': {}, 'fp8_training': {'exclude': 'adaLN', 'weights': 'direct'}}
    assert fp8_training.apply_fp8_training_config_defaults(config)['exclude'] == ['adaLN']


def test_train_py_wires_fp8_training():
    source = (ROOT / 'train.py').read_text()
    for call in ('fp8_training.apply_fp8_training_config_defaults(config)', 'fp8_training.check_environment(config)',
                 'fp8_training.prepare_model(model)', 'fp8_training.step_scope(training=True)',
                 'fp8_training.step_scope(training=False)', 'fp8_training.disabled()', 'fp8_training.report()'):
        assert call in source, call
    # Converted before the adapter wraps the Linear layers.
    assert source.index('fp8_training.prepare_model(model)') < source.index('model.configure_adapter(adapter_config)')


def test_convert_replaces_block_linears_and_keeps_parameters(te):
    settings(exclude=['mlp\\.2'])
    torch.manual_seed(0)
    net = Net(odd=True)
    before = {name: p for name, p in net.named_parameters()}
    keys = set(net.state_dict())
    reference = copy.deepcopy(net)
    summary = fp8_training.prepare_model(Pipeline(net))
    # qkv, out, mlp.0 in each of the two blocks; mlp.2 excluded, odd (out_features 5) does not fit.
    assert summary['fp8'] == 6 and summary['excluded'] == 2 and summary['unfit'] == 2
    assert isinstance(net.blocks[0].attn.qkv, fp8_training.Fp8Linear)
    assert not isinstance(net.blocks[0].mlp[2], fp8_training.Fp8Linear)
    assert not isinstance(net.proj_in, fp8_training.Fp8Linear)
    assert set(net.state_dict()) == keys
    for name, p in net.named_parameters():
        assert p is before[name]
    x = torch.randn(2, 8, 32)
    out, expected = net(x), reference(x)
    assert len(te.linear_calls) == 6
    assert not torch.equal(out, expected)
    torch.testing.assert_close(out, expected, rtol=0.1, atol=0.1)


def test_a_model_keeps_its_fp8_keep_modules_whatever_exclude_says(te):
    settings()
    net = Net(blocks=2)

    class KeepingPipeline(Pipeline):
        fp8_keep_modules = ['mlp\\.0']

    summary = fp8_training.prepare_model(KeepingPipeline(net))
    # qkv, out and mlp.2 in each block run in fp8; mlp.0 is the model's own to keep.
    assert summary['fp8'] == 6 and summary['kept'] == 2 and summary['excluded'] == 0
    assert not isinstance(net.blocks[0].mlp[0], fp8_training.Fp8Linear)
    assert isinstance(net.blocks[0].mlp[2], fp8_training.Fp8Linear)


def test_minimax_h3_keeps_its_adaln_projections_out_of_fp8():
    from models.minimax_h3 import MinimaxH3Pipeline
    from models.p_h3_image import PH3ImagePipeline
    for cls in (MinimaxH3Pipeline, PH3ImagePipeline):
        keep = [re.compile(p) for p in cls.fp8_keep_modules]
        assert any(p.search('blocks.3.adaln_proj.linear') for p in keep)
        assert any(p.search('token_refiner.blocks.0.adaln_proj.linear') for p in keep)
        assert not any(p.search('blocks.3.attn.qkv') or p.search('blocks.3.mlp.fc1') for p in keep)


def test_full_finetune_gradients_reach_the_weights(te):
    settings()
    net = Net(blocks=1)
    fp8_training.prepare_model(Pipeline(net))
    net(torch.randn(2, 8, 32)).square().mean().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert all(call['requires_grad'] for call in te.linear_calls)


@pytest.mark.parametrize('tokens', [16, 7], ids=['16-tokens', 'padded-7-tokens'])
@pytest.mark.parametrize('adapter', ['lora', 'lokr', 'dora', 'oplora'])
def test_adapters_wrap_fp8_layers(te, adapter, tokens):
    settings()
    torch.manual_seed(0)
    net = Net(blocks=1)
    fp8_training.prepare_model(Pipeline(net))
    targets = [n for n, m in net.named_modules() if isinstance(m, fp8_training.Fp8Linear)]
    if adapter == 'lokr':
        config = peft.LoKrConfig(r=4, alpha=4, target_modules=targets)
    else:
        config = peft.LoraConfig(r=4, lora_alpha=4, target_modules=targets, use_dora=adapter == 'dora')
    model = peft.get_peft_model(net, config)
    for name, p in model.named_parameters():
        if p.requires_grad:
            # Adapters start as a zero update, where some factors get no gradient.
            nn.init.normal_(p, std=0.1)
    if adapter == 'oplora':
        from utils.oplora import OPLoRAProjector
        projector = OPLoRAProjector.build(model, rank=2, full_svd=True)
        assert len(projector) == len(targets)
        projector.project()
    model(torch.randn(1, tokens, 32)).square().mean().backward()
    trained = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trained and all(p.grad is not None and p.grad.abs().sum() > 0 for _, p in trained)
    assert te.linear_calls and not any(call['requires_grad'] for call in te.linear_calls)
    # every base layer ran fp8, a 7-token call on rows padded to 16
    assert all(math.prod(call['x_shape'][:-1]) % 16 == 0 for call in te.linear_calls)
    assert fp8_training.counts()[0]['fp8'] == len(te.linear_calls) == len(targets)


def test_bf16_for_calls_and_layers_fp8_cannot_take(te):
    settings()
    net = Net(blocks=1)
    fp8_training.prepare_model(Pipeline(net))
    layer = net.blocks[0].mlp[0]
    with fp8_training.disabled():
        layer(torch.ones(16, 32))
    assert not te.linear_calls
    te.fail_linear = True
    torch.testing.assert_close(layer(torch.ones(16, 32)), F.linear(torch.ones(16, 32), layer.weight, layer.bias))
    assert layer.fp8_ok is False
    te.fail_linear = False
    layer(torch.ones(16, 32))
    assert not te.linear_calls
    other = net.blocks[0].mlp[2]
    other(torch.ones(16, 64))
    te.fail_linear = True
    with pytest.raises(RuntimeError, match='no fp8 GEMM'):
        other(torch.ones(16, 64))


def test_out_features_must_be_a_multiple_of_16(te):
    # the input-gradient GEMM reduces over out_features
    settings()

    class Wide(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(32, 24)
            self.b = nn.Linear(32, 48)

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Wide()])

    class WidePipeline(Pipeline):
        adapter_target_modules = ['Wide']

    net = Holder()
    summary = fp8_training.prepare_model(WidePipeline(net))
    assert summary['fp8'] == 1 and summary['unfit'] == 1
    assert isinstance(net.blocks[0].b, fp8_training.Fp8Linear) and not isinstance(net.blocks[0].a, fp8_training.Fp8Linear)


def test_cache_frozen_weights_only_for_frozen_layers(te):
    settings(cache_frozen_weights=True)
    net = Net(blocks=1)
    fp8_training.prepare_model(Pipeline(net))
    net.blocks[0].mlp[0].weight.requires_grad_(False)
    net(torch.randn(1, 16, 32))
    cached = {id(c['module']): c['cache'] for c in te.linear_calls}
    assert cached[id(net.blocks[0].mlp[0]._te[0])] is True
    assert cached[id(net.blocks[0].mlp[2]._te[0])] is False


class StoredQuantized:
    """ComfyUI's QuantizedTensor as far as fp8_training looks at it."""

    class Params:
        def __init__(self, scale, shape):
            self.scale, self.orig_shape, self.orig_dtype = scale, shape, torch.bfloat16

    def __init__(self, value, layout='TensorCoreFP8E4M3Layout'):
        scale = value.abs().amax().float() / 448
        self._qdata = (value / scale).to(torch.float8_e4m3fn)
        self._params = self.Params(scale.reshape(()), tuple(value.shape))
        self._layout_cls = layout
        self.requires_grad = False
        self.shape = value.shape
        self.dtype = torch.bfloat16

    def dequantize(self):
        return self._qdata.float() * self._params.scale


def quantized_net(layout='TensorCoreFP8E4M3Layout'):
    torch.manual_seed(0)
    net = Net(blocks=1)
    stored = {}
    for name, module in net.named_modules():
        if isinstance(module, nn.Linear) and name.startswith('blocks'):
            stored[name] = StoredQuantized(module.weight.detach(), layout)
            module._parameters['weight'] = stored[name]
    return net, stored


def test_direct_mode_uses_stored_e4m3_weights_with_current_scaling(te):
    settings(weights='direct', recipe='block')
    net, stored = quantized_net()
    summary = fp8_training.prepare_model(Pipeline(net))
    assert summary['direct'] == 4 and summary['dequantized'] == 0
    layer = net.blocks[0].attn.qkv
    assert layer.fp8_recipe == 'current' and layer.weight is stored['blocks.0.attn.qkv']
    layer(torch.randn(16, 32))
    call = te.linear_calls[-1]
    assert call['direct'] and call['recipe'] == 'current'
    plain = Net(blocks=1)
    plain.blocks[0].attn.qkv.weight.data = torch.ones(96, 32, dtype=torch.float8_e4m3fn)
    fp8_training.prepare_model(Pipeline(plain))
    assert plain.blocks[0].attn.qkv.fp8_mode == 'direct'
    plain.blocks[0].attn.qkv(torch.randn(16, 32))
    assert te.linear_calls[-1]['direct']


def test_pre_quant_scale_is_folded_into_the_weight(te):
    # ModelOpt checkpoints in ComfyUI: the layer computes (x * pre_quant_scale) @ W^T.
    settings(weights='direct')
    net, stored = quantized_net()
    layer = net.blocks[0].attn.qkv
    pre = torch.linspace(0.5, 2.0, 32)
    layer.register_buffer('pre_quant_scale', pre)
    x = torch.randn(16, 32, dtype=torch.bfloat16)
    expected = F.linear(x.float() * pre, stored['blocks.0.attn.qkv'].dequantize(), layer.bias.float())
    summary = fp8_training.prepare_model(Pipeline(net))
    # Not 'direct': the stored weight alone is not what the layer multiplies by.
    assert summary['direct'] == 3 and net.blocks[0].attn.qkv.fp8_mode == 'quantize'
    torch.testing.assert_close(net.blocks[0].attn.qkv(x).float(), expected, rtol=0.1, atol=0.15)


def test_dequantize_mode_and_formats_direct_cannot_use(te):
    settings(weights='direct')
    net, stored = quantized_net(layout='TensorCoreMXFP8Layout')
    summary = fp8_training.prepare_model(Pipeline(net))
    assert summary['direct'] == 0 and summary['dequantized'] == 4
    layer = net.blocks[0].attn.qkv
    assert layer.weight.dtype == torch.bfloat16 and layer.fp8_mode == 'quantize'
    torch.testing.assert_close(layer.weight.float(), stored['blocks.0.attn.qkv'].dequantize(), rtol=0.01, atol=0.01)
    # The hidden original no longer holds the stored weight.
    assert layer._source[0].weight is layer.weight
    fp8_training.reset(te)
    settings()
    net, stored = quantized_net()
    assert fp8_training.prepare_model(Pipeline(net))['dequantized'] == 4


def attention_net(blocks=2):
    torch.manual_seed(0)
    return Net(blocks=blocks).to(torch.bfloat16)


def inputs(*shape):
    return torch.randn(*shape, dtype=torch.bfloat16)


def train_step(net, x, checkpoint=None):
    with fp8_training.step_scope(training=True):
        if checkpoint is None:
            out = net(x)
        else:
            h = net.proj_in(x)
            for block in net.blocks:
                h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=checkpoint == 'reentrant')
            out = net.final(h)
        out.square().mean().backward()
    return out


def setup_attention(net, **table):
    settings(attention=True, **table)
    fp8_training.prepare_model(Pipeline(net))
    attention_util.install()


def test_attention_calibrates_then_runs_fp8_with_measured_scales(te):
    net = attention_net()
    setup_attention(net, attention_amax_history=4)
    reference = copy.deepcopy(net)
    x = inputs(2, 16, 32)
    out = train_step(net, x)
    # The calibration step is the bf16 path itself.
    torch.testing.assert_close(out, reference(x))
    assert not te.attention_calls
    sites = fp8_training._attention.sites
    assert len(sites) == 2 and all(s.state == 'active' for s in sites.values())
    site = sites[('blocks.0', 0, (2, 2, 16, 'bshd', 'no_mask', None, torch.bfloat16))]
    recipe, amax_fwd, amax_bwd = site.module.primed
    assert recipe == ('delayed', 4)
    assert amax_fwd[1] == 1 and (amax_fwd > 0).all() and (amax_bwd > 0).all()
    assert te.updates == 1
    # primed with S at 1; pinned there after every update from here on
    assert not hasattr(site.module, 'pinned')
    train_step(net, x)
    assert len(te.attention_calls) == 2 and fp8_training.counts()[1]['fp8'] == 2
    assert te.updates == 2 and all(s.module.pinned == 1 for s in sites.values())
    with fp8_training.step_scope(training=False):
        net(x)
    assert all(s.module.pinned == 1 for s in sites.values())
    # TE's own use of the (patched) SDPA inside a call is part of that call, not another site.
    assert len(sites) == 2


@pytest.mark.parametrize('checkpoint', ['reentrant', 'non_reentrant'])
def test_attention_sites_are_found_again_by_recomputation(te, checkpoint):
    net = attention_net()
    setup_attention(net)
    x = inputs(2, 16, 32)
    train_step(net, x, checkpoint)
    sites = fp8_training._attention.sites
    # Gradient hooks fired in the backward that follows each recomputation.
    assert len(sites) == 2 and all(s.amax_bwd is not None and (s.amax_bwd > 0).all() for s in sites.values())
    train_step(net, x, checkpoint)
    assert len(fp8_training._attention.sites) == 2
    # Each site ran in the forward and again in its recomputation.
    assert all(s.module.calls == 2 for s in sites.values())


def test_eval_and_calls_outside_a_step_do_not_start_fp8(te):
    net = attention_net()
    setup_attention(net)
    x = inputs(2, 16, 32)
    net(x)
    assert not fp8_training._attention.sites
    with torch.no_grad(), fp8_training.step_scope(training=False):
        net(x)
    sites = fp8_training._attention.sites
    assert len(sites) == 2 and all(s.state == 'calibrating' for s in sites.values())
    assert te.updates == 0
    train_step(net, x)
    assert all(s.state == 'active' for s in sites.values())
    with fp8_training.disabled(), fp8_training.step_scope(training=True):
        net(x)
    assert not te.attention_calls
    with torch.no_grad(), fp8_training.step_scope(training=False):
        net(x)
    # Eval runs fp8, and what it measured is dropped instead of joining the next scale update.
    assert len(te.attention_calls) == 2 and te.updates == 2
    assert all(s.module.discarded == 1 for s in sites.values())


def test_attention_masks(te):
    net = attention_net(blocks=1)
    setup_attention(net)
    x = inputs(2, 16, 32)
    keep = torch.ones(2, 1, 1, 16, dtype=torch.bool)
    keep[1, ..., 10:] = False
    net.blocks[0].attn.mask = keep
    train_step(net, x)
    train_step(net, x)
    call = te.attention_calls[-1]
    assert call['mask_type'] == 'padding'
    assert call['mask'][1][1, 0, 0].tolist() == [False] * 10 + [True] * 6
    reference = copy.deepcopy(net)
    with attention_util.bypass():
        expected = reference(x)
    with fp8_training.step_scope(training=False):
        torch.testing.assert_close(net(x), expected, rtol=0.1, atol=0.1)
    # Kept keys that are not the first ones (padded text before image tokens, as Qwen-Image,
    # Z-Image and Krea 2 build it): they are gathered to the front for TE.
    keep = torch.ones(2, 1, 1, 16, dtype=torch.bool)
    keep[0, ..., 3:7] = False
    net.blocks[0].attn.mask = keep
    train_step(net, x)
    reference = copy.deepcopy(net)
    with attention_util.bypass():
        expected = reference(x)
    with fp8_training.step_scope(training=False):
        torch.testing.assert_close(net(x), expected, rtol=0.1, atol=0.1)
    assert te.attention_calls[-1]['mask'][1][:, 0, 0].tolist() == [[False] * 12 + [True] * 4, [False] * 16]
    net.blocks[0].attn.mask = inputs(1, 1, 16, 16)
    fp8_training.reset_counts()
    train_step(net, x)
    assert fp8_training.counts()[1] == {'a float mask with bias values': 1}


def test_attention_outside_blocks_and_te_failures(te):
    net = attention_net(blocks=1)
    setup_attention(net)
    q = inputs(1, 2, 16, 16)
    with fp8_training.step_scope(training=True):
        F.scaled_dot_product_attention(q, q, q)
    assert fp8_training.counts()[1] == {'calls outside the transformer blocks': 1}
    x = inputs(2, 16, 32)
    train_step(net, x)
    te.fail_attention = True
    out = train_step(net, x)
    assert torch.isfinite(out).all()
    site = next(iter(fp8_training._attention.sites.values()))
    assert site.state == 'off'
    # tools/check_fp8_training.py reports TE's reason
    assert site.error == 'RuntimeError: no fused attention backend'


@pytest.mark.parametrize('thd', [True, False], ids=['thd', 'padded'])
@pytest.mark.parametrize('causal', [False, True], ids=['full', 'causal'])
def test_dense_and_varlen_entry_points(te, thd, causal):
    te.thd = thd
    setup_attention(attention_net(blocks=1))
    q = inputs(1, 16, 2, 16).requires_grad_()
    cu = torch.tensor([0, 6, 16], dtype=torch.int32)
    packed = inputs(16, 2, 16).requires_grad_()

    def dense_fallback():
        return F.scaled_dot_product_attention(q.transpose(1, 2), q.transpose(1, 2), q.transpose(1, 2)).transpose(1, 2)

    def varlen_fallback():
        parts = [REAL_SDPA(*(packed[a:b].transpose(0, 1)[None],) * 3, is_causal=causal)[0].transpose(0, 1)
                 for a, b in ((0, 6), (6, 16))]
        return torch.cat(parts)

    def step():
        with fp8_training.step_scope(training=True):
            fp8_training._attention.stack.append(fp8_training._Block('blocks.0'))
            try:
                a = attention_util.fp8_dense(q, q, q, None, False, dense_fallback)
                b = attention_util.fp8_varlen(packed, packed, packed, cu, cu, 10, 10, None, causal, varlen_fallback)
            finally:
                fp8_training._attention.stack.pop()
            (a.sum() + b.sum()).backward()
        return a, b

    a0, b0 = step()
    grad0 = packed.grad.clone()
    packed.grad = None
    a1, b1 = step()
    torch.testing.assert_close(a1, a0, rtol=0.1, atol=0.1)
    # The fp8 step against the bf16 calibration step: the same attention, packed or padded.
    torch.testing.assert_close(b1, b0, rtol=0.1, atol=0.1)
    torch.testing.assert_close(packed.grad, grad0, rtol=0.1, atol=0.1)
    kinds = sorted(c['mask_type'] for c in te.attention_calls)
    assert kinds == ['no_mask', 'padding_causal' if causal else 'padding']
    call = next(c for c in te.attention_calls if c['mask_type'] != 'no_mask')
    if thd:
        assert call['module'].signature[3] == 'thd' and call['cu'] is cu
    else:
        # Hopper: run padded to the longest sequence (10), with query and key padding masks.
        assert call['module'].signature[3] == 'bshd' and call['cu'] is None
        assert call['mask'][0][:, 0, 0].tolist() == [[False] * 6 + [True] * 4, [False] * 10]


def test_unpadded_calibration_observes_the_tensors_the_fallback_uses(te):
    net = attention_net(blocks=1)
    setup_attention(net)
    q = inputs(2, 8, 2, 16).requires_grad_()
    keep = torch.ones(2, 8, dtype=torch.bool)
    keep[0, 5:] = False

    def fallback():
        return F.scaled_dot_product_attention(q.transpose(1, 2), q.transpose(1, 2), q.transpose(1, 2)).transpose(1, 2)

    with fp8_training.step_scope(training=True):
        fp8_training._attention.stack.append(fp8_training._Block('blocks.0'))
        out = attention_util.flash_unpadded(q, q, q, keep, fallback=fallback)
        fp8_training._attention.stack.pop()
        out.sum().backward()
    site = next(iter(fp8_training._attention.sites.values()))
    assert site.amax_bwd[2] > 0 and site.state == 'active'


def test_install_offers_sdpa_to_fp8_with_the_sdpa_backend(te):
    original = F.scaled_dot_product_attention
    settings(attention=True)
    attention_util.install()
    assert F.scaled_dot_product_attention is not original
    attention_util.uninstall()
    assert F.scaled_dot_product_attention is original
    fp8_training.reset(te)
    settings()
    attention_util.install()
    assert F.scaled_dot_product_attention is original


def test_flash_stays_off_when_only_fp8_replaces_sdpa(te):
    # Lumina 2, Wan, HunyuanVideo... take their FlashAttention branch when flash_enabled(): with
    # [model] attention = 'sdpa' there is no FlashAttention to run.
    settings(attention=True)
    attention_util.install()
    assert F.scaled_dot_product_attention is not attention_util._original_sdpa
    assert not attention_util.flash_enabled()


def test_iris_flash_backends_are_offered_to_fp8_attention(te, monkeypatch):
    sys.path.insert(0, str(ROOT / 'submodules' / 'iris-3b' / 'src'))
    from models import iris
    import iris3b.models.blocks.single_stream as single_stream
    import iris3b.nn.attention as iris_attention
    assert iris_attention.scaled_dot_product is iris._scaled_dot_product
    assert single_stream.scaled_dot_product is iris._scaled_dot_product
    ran = []

    def kernel(q, k, v, backend='sdpa', attn_mask=None, enable_gqa=False):
        # Stands for Iris's fa3 / fa4 call (B, H, N, D in and out).
        ran.append(backend)
        return REAL_SDPA(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

    monkeypatch.setattr(iris, '_iris_scaled_dot_product', kernel)
    settings(attention=True)

    class SingleStreamBlock(nn.Module):
        def forward(self, q, k, v, backend):
            return iris_attention.scaled_dot_product(q, k, v, backend=backend, enable_gqa=True)

    block = SingleStreamBlock()
    fp8_training._attention.hook_blocks(block, ['SingleStreamBlock'])
    q, kv = inputs(1, 4, 16, 16).requires_grad_(), inputs(1, 2, 16, 16).requires_grad_()
    outs = []
    for _ in range(2):
        with fp8_training.step_scope(training=True):
            out = block(q, kv, kv, 'fa3')
            out.sum().backward()
        outs.append(out)
    # The calibration step ran Iris's kernel, the next one fp8 (GQA kept: 4 query, 2 key heads).
    assert ran == ['fa3'] and len(te.attention_calls) == 1
    assert te.attention_calls[0]['module'].signature[:3] == (4, 2, 16)
    assert outs[1].shape == outs[0].shape == (1, 4, 16, 16)
    torch.testing.assert_close(outs[1], outs[0], rtol=0.1, atol=0.1)
    with fp8_training.step_scope(training=True):
        block(q, kv, kv, 'torch_flash')
    assert ran == ['fa3', 'torch_flash']


def test_sdxl_names_its_transformer_blocks():
    source = (ROOT / 'models' / 'sdxl.py').read_text()
    assert "fp8_target_modules = ['Transformer2DModel', 'BasicTransformerBlock']" in source


def test_report_lists_reasons(te, capsys):
    net = attention_net(blocks=1)
    setup_attention(net)
    net.blocks[0].mlp[0](torch.ones(3, 32, dtype=torch.bfloat16))
    with fp8_training.disabled():
        net.blocks[0].mlp[0](torch.ones(3, 32, dtype=torch.bfloat16))
    train_step(net, inputs(2, 16, 32))
    capsys.readouterr()
    fp8_training.report()
    out = capsys.readouterr().out
    assert '(1 of them padded to a multiple of 16 tokens); without fp8: 1 sampling' in out
    assert '2 bf16 calibration' not in out and 'attention calls ran fp8' in out


def _distributed_worker(rank, world_size, init_file, out_dir):
    torch.distributed.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank, world_size=world_size)
    try:
        te = FakeTE()
        fp8_training.reset(te)
        net = attention_net()
        setup_attention(net)
        torch.manual_seed(rank)
        # A different batch on each rank, gradients averaged as data parallelism does.
        for _ in range(3):
            net.zero_grad()
            train_step(net, inputs(2, 16, 32), checkpoint='non_reentrant')
            for p in net.parameters():
                torch.distributed.all_reduce(p.grad)
        fp8_training.report()
        sites = {repr(key): site.state for key, site in fp8_training._attention.sites.items()}
        with open(f'{out_dir}/{rank}.json', 'w') as f:
            json.dump({'sites': sites, 'counts': [dict(c) for c in fp8_training.counts()], 'updates': te.updates}, f)
    finally:
        attention_util.uninstall()
        fp8_training.reset()
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize('world_size', [2, 3])
def test_every_rank_promotes_the_same_sites_and_report_sums_all_ranks(world_size, tmp_path, capfd):
    mp.spawn(_distributed_worker, args=(world_size, str(tmp_path / 'init'), str(tmp_path)), nprocs=world_size)
    results = [json.loads((tmp_path / f'{rank}.json').read_text()) for rank in range(world_size)]
    for result in results:
        assert result['sites'] == results[0]['sites'] and set(result['sites'].values()) == {'active'}
        assert result['updates'] == 3
    linear, attention = results[0]['counts']
    out = capfd.readouterr().out
    # Printed once (rank 0), with every rank's calls.
    assert out.count('Linear calls ran fp8') == 1
    assert f"{linear['fp8'] * world_size} Linear calls ran fp8" in out
    assert f"{attention['fp8'] * world_size} attention calls ran fp8" in out


def test_te_backend_pins_the_softmax_scale_and_takes_padding_masks_as_cross_attention(monkeypatch):
    """The real backend against stand-ins for the TE pieces it calls (no GPU here)."""
    import types
    backend = object.__new__(fp8_training.TransformerEngineBackend)
    made = []
    backend.te = types.SimpleNamespace(DotProductAttention=lambda *a, **k: made.append(k) or k)
    for mask_type, expected in (('no_mask', 'self'), ('causal', 'self'), ('padding', 'cross'), ('padding_causal', 'cross')):
        backend.make_attention(16, 4, 128, 'bshd', mask_type, None)
        assert made[-1]['attention_type'] == expected and made[-1]['attn_mask_type'] == mask_type

    fused = types.ModuleType('transformer_engine.pytorch.cpp_extensions.fused_attn')
    fused.META_S = 1
    monkeypatch.setitem(sys.modules, 'transformer_engine.pytorch.cpp_extensions.fused_attn', fused)
    state = types.SimpleNamespace(scale=torch.tensor([83.8, 3389.0, 959.4]))
    module = types.SimpleNamespace(fp8_meta={'scaling_fwd': state, 'fp8_max_fwd': 448.0})
    backend.pin_softmax_scale(module, types.SimpleNamespace(margin=0))
    assert state.scale.tolist() == pytest.approx([83.8, 448.0, 959.4])
    backend.pin_softmax_scale(module, types.SimpleNamespace(margin=1))
    assert state.scale[1].item() == 224.0


def test_auto_attention_runs_fp8_only_on_long_sequences(te):
    net = attention_net()
    settings(attention='auto', attention_min_tokens=16)
    fp8_training.prepare_model(Pipeline(net))
    attention_util.install()
    assert fp8_training.attention_enabled()
    short, long = inputs(2, 8, 32), inputs(2, 16, 32)
    train_step(net, short)
    train_step(net, short)
    assert not fp8_training._attention.sites and not te.attention_calls
    assert fp8_training.counts()[1]['sequences shorter than attention_min_tokens'] == 4
    train_step(net, long)  # bf16 calibration
    train_step(net, long)
    assert len(te.attention_calls) == 2 and fp8_training.counts()[1]['fp8'] == 2
    train_step(net, short)
    assert len(te.attention_calls) == 2
    # true keeps every length
    fp8_training.reset(te)
    settings(attention=True)
    assert fp8_training._settings.attention_min_tokens == 0


def test_token_counts_are_padded_to_a_multiple_of_16(te):
    """TE's weight-gradient GEMM reduces over the tokens and needs a multiple of 16: other counts
    run fp8 on zero-padded rows, with the same output and gradients as unpadded."""
    settings()
    torch.manual_seed(0)
    net = Net(blocks=1)
    fp8_training.prepare_model(Pipeline(net))
    layer = net.blocks[0].mlp[0]
    for shape in ((3, 32), (2, 5, 32), (16, 32)):
        x = torch.randn(*shape)
        tokens = x.numel() // 32
        xs = x.detach().clone().requires_grad_()
        layer.weight.grad = None
        out = layer(xs)
        assert math.prod(te.linear_calls[-1]['x_shape'][:-1]) % 16 == 0 and out.shape == (*shape[:-1], 64)
        out.square().sum().backward()
        grads = xs.grad.clone(), layer.weight.grad.clone()
        # the same rows padded by hand, a call of a multiple of 16 tokens
        rows = torch.cat([x.reshape(tokens, 32), torch.zeros(-tokens % 16, 32)]).requires_grad_()
        layer.weight.grad = None
        full = layer(rows)
        full[:tokens].square().sum().backward()
        torch.testing.assert_close(out.reshape(tokens, 64), full[:tokens].detach())
        torch.testing.assert_close(grads[0].reshape(tokens, 32), rows.grad[:tokens])
        torch.testing.assert_close(grads[1], layer.weight.grad)
    counts = fp8_training.counts()[0]
    assert counts['fp8'] == 6 and counts['fp8 calls padded to a multiple of 16 tokens'] == 2


def test_a_site_seen_only_in_eval_waits_for_a_training_step(te):
    """With 'auto', a long eval batch before short training steps creates sites whose backward
    was never measured: they stay in calibration until a training step reaches them."""
    net = attention_net()
    settings(attention='auto', attention_min_tokens=16)
    fp8_training.prepare_model(Pipeline(net))
    attention_util.install()
    short, long = inputs(2, 8, 32), inputs(2, 16, 32)
    with torch.no_grad(), fp8_training.step_scope(training=False):
        net(long)
    train_step(net, short)
    sites = list(fp8_training._attention.sites.values())
    assert sites and all(site.state == 'calibrating' for site in sites)
    train_step(net, long)
    assert all(site.state == 'active' and site.module.primed[2] is not None for site in sites)
