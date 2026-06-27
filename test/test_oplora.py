"""Tests for OPLoRA (utils/oplora.py).

These run on CPU without DeepSpeed. The math and determinism tests use a small
fake module that mirrors peft's LoRA layout (base_layer / lora_A / lora_B); the
smoke test uses a real peft LoRA model when peft is importable.
"""

import pytest
import torch
from torch import nn

from utils.oplora import OPLoRAProjector, apply_oplora_config_defaults


class FakeLoraLayer(nn.Module):
    """Minimal stand-in for a peft LoRA layer, with the same attribute layout."""

    def __init__(self, out_features, in_features, r, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        self.lora_A = nn.ModuleDict({'default': nn.Linear(in_features, r, bias=False)})
        self.lora_B = nn.ModuleDict({'default': nn.Linear(r, out_features, bias=False)})
        with torch.no_grad():
            self.base_layer.weight.copy_(torch.randn(out_features, in_features, generator=gen))
            self.lora_A['default'].weight.copy_(torch.randn(r, in_features, generator=gen))
            self.lora_B['default'].weight.copy_(torch.randn(out_features, r, generator=gen))


def _bases_from_full_svd(weight, k):
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    return u[:, :k], s[:k], vh[:k, :].transpose(0, 1)


class TestConfigDefaults:
    def test_disabled_by_default(self):
        cfg = {'type': 'lora', 'rank': 16}
        apply_oplora_config_defaults(cfg)
        assert cfg['oplora'] is False

    def test_fills_defaults_when_enabled(self):
        cfg = {'type': 'lora', 'rank': 16, 'oplora': True, 'oplora_rank': 8}
        apply_oplora_config_defaults(cfg)
        assert cfg['oplora_full_svd'] is False
        assert cfg['oplora_seed'] == 0

    def test_rejects_non_lora(self):
        cfg = {'type': 'lokr', 'oplora': True, 'oplora_rank': 8}
        with pytest.raises(ValueError, match='only supported for adapter type'):
            apply_oplora_config_defaults(cfg)

    def test_requires_rank(self):
        cfg = {'type': 'lora', 'oplora': True}
        with pytest.raises(ValueError, match='oplora_rank is not set'):
            apply_oplora_config_defaults(cfg)

    def test_rejects_non_positive_rank(self):
        cfg = {'type': 'lora', 'oplora': True, 'oplora_rank': 0}
        with pytest.raises(ValueError, match='positive integer'):
            apply_oplora_config_defaults(cfg)

    def test_rejects_bool_rank(self):
        cfg = {'type': 'lora', 'oplora': True, 'oplora_rank': True}
        with pytest.raises(ValueError, match='positive integer'):
            apply_oplora_config_defaults(cfg)


class TestProjectionMath:
    def test_projection_removes_top_k_components(self):
        layer = FakeLoraLayer(out_features=32, in_features=24, r=6, seed=1)
        k = 4
        projector = OPLoRAProjector.build(layer, rank=k, full_svd=True, compute_device=torch.device('cpu'))
        projector.project()
        assert projector.max_residual() < 1e-4

    def test_top_k_singular_triples_are_preserved(self):
        layer = FakeLoraLayer(out_features=32, in_features=24, r=6, seed=2)
        k = 4
        base = layer.base_layer.weight.detach().clone()
        u_k, s_k, v_k = _bases_from_full_svd(base, k)

        projector = OPLoRAProjector.build(layer, rank=k, full_svd=True, compute_device=torch.device('cpu'))
        projector.project()

        up = layer.lora_B['default'].weight.detach()
        down = layer.lora_A['default'].weight.detach()
        effective = base + up @ down

        torch.testing.assert_close(effective @ v_k, u_k * s_k, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(effective.transpose(0, 1) @ u_k, v_k * s_k, atol=1e-4, rtol=1e-4)

    def test_rank_too_large_raises(self):
        layer = FakeLoraLayer(out_features=8, in_features=6, r=4, seed=3)
        with pytest.raises(ValueError, match='larger than the smallest dimension'):
            OPLoRAProjector.build(layer, rank=10, compute_device=torch.device('cpu'))


class TestDataParallelDeterminism:
    def test_randomized_bases_match_across_replicas(self):
        # Two replicas hold identical base weights; with the same seed the randomized
        # SVD must produce identical bases, or data-parallel replicas would diverge.
        layer_a = FakeLoraLayer(out_features=40, in_features=28, r=6, seed=7)
        layer_b = FakeLoraLayer(out_features=40, in_features=28, r=6, seed=7)

        proj_a = OPLoRAProjector.build(layer_a, rank=5, full_svd=False, base_seed=123, compute_device=torch.device('cpu'))
        proj_b = OPLoRAProjector.build(layer_b, rank=5, full_svd=False, base_seed=123, compute_device=torch.device('cpu'))

        proj_a.project()
        proj_b.project()

        torch.testing.assert_close(
            layer_a.lora_B['default'].weight, layer_b.lora_B['default'].weight, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            layer_a.lora_A['default'].weight, layer_b.lora_A['default'].weight, atol=0.0, rtol=0.0
        )

    def test_randomized_bases_differ_for_different_seed(self):
        layer_a = FakeLoraLayer(out_features=40, in_features=28, r=6, seed=7)
        layer_b = FakeLoraLayer(out_features=40, in_features=28, r=6, seed=7)

        proj_a = OPLoRAProjector.build(layer_a, rank=5, full_svd=False, base_seed=1, compute_device=torch.device('cpu'))
        proj_b = OPLoRAProjector.build(layer_b, rank=5, full_svd=False, base_seed=2, compute_device=torch.device('cpu'))
        proj_a.project()
        proj_b.project()

        assert not torch.allclose(layer_a.lora_B['default'].weight, layer_b.lora_B['default'].weight)


class TestPeftSmoke:
    def test_train_steps_keep_updates_orthogonal(self):
        peft = pytest.importorskip('peft')

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 24, bias=False)
                self.fc2 = nn.Linear(24, 16, bias=False)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        torch.manual_seed(0)
        cfg = peft.LoraConfig(r=4, lora_alpha=4, target_modules=['fc1', 'fc2'], lora_dropout=0.0, bias='none')
        model = peft.get_peft_model(Net(), cfg)

        projector = OPLoRAProjector.build(model, rank=4, full_svd=True, compute_device=torch.device('cpu'))
        assert len(projector) == 2

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=1e-2)

        for _ in range(5):
            x = torch.randn(8, 16)
            loss = model(x).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            projector.project()
            assert torch.isfinite(loss)

        assert projector.max_residual() < 1e-3
