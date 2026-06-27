"""Orthogonal Projection LoRA (OPLoRA) for diffusion-pipe.

After every optimizer step, each trainable LoRA factor is projected into the
orthogonal complement of the top-k singular subspace of its frozen base weight.
For a base weight ``W = U S V^T`` we keep the top-k left and right singular
vectors ``U_k`` and ``V_k`` and, with peft storing the adapter as
``lora_A`` (shape ``(r, in)``) and ``lora_B`` (shape ``(out, r)``), apply::

    up   = lora_B - U_k @ (U_k^T @ lora_B)
    down = lora_A - (lora_A @ V_k) @ V_k^T

so that ``U_k^T @ up == 0`` and ``down @ V_k == 0``. The updated effective
weight then keeps ``W``'s top-k singular triples unchanged.

The projection is rank-local: it only touches the LoRA parameters and the
precomputed bases, so under DeepSpeed data parallelism every replica applies the
same projection (the bases come from the identical frozen base weight). The only
requirement is that the bases are computed deterministically, which matters for
the randomized path and is handled by seeding from a stable per-layer key.

Reference: "OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting
during Parameter-Efficient Fine-Tuning" (arXiv:2510.13003).
"""

import hashlib

import torch


def apply_oplora_config_defaults(adapter_config):
    """Validate and fill defaults for the OPLoRA keys in an ``[adapter]`` config table.

    Kept here, free of training-stack imports, so the rules can be unit-tested on CPU.
    """
    adapter_config.setdefault('oplora', False)
    if not adapter_config['oplora']:
        return
    if adapter_config.get('type') != 'lora':
        raise ValueError(f"oplora is only supported for adapter type 'lora', not '{adapter_config.get('type')}'")
    if 'oplora_rank' not in adapter_config:
        raise ValueError('oplora is enabled but oplora_rank is not set. Set oplora_rank to the size of the protected top-k singular subspace.')
    rank = adapter_config['oplora_rank']
    # bool is a subclass of int, so reject it explicitly.
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError(f'oplora_rank must be a positive integer, got {rank!r}')
    adapter_config.setdefault('oplora_full_svd', False)
    adapter_config.setdefault('oplora_seed', 0)


def _stable_seed(base_seed, key):
    # Python's builtin hash() is randomized per process, which would give each
    # data-parallel rank different bases. Use a stable digest so every rank that
    # owns the same layer derives the same seed.
    digest = hashlib.sha256(key.encode('utf-8')).digest()
    return (base_seed ^ int.from_bytes(digest[:8], 'big')) & 0xFFFFFFFFFFFFFFFF


def _iter_lora_layers(root):
    for name, module in root.named_modules():
        if hasattr(module, 'base_layer') and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            yield name, module


def _is_zero_partitioned(param):
    # DeepSpeed ZeRO-3 replaces the real storage with a 0-length placeholder and
    # tags the parameter with ds_* attributes. The repo runs ZeRO stage 0, so this
    # should never be true, but projecting a partitioned shard would be wrong.
    return getattr(param, 'ds_status', None) is not None or getattr(param, 'ds_numel', None) is not None


def _full_bases(weight, rank):
    u, _, vh = torch.linalg.svd(weight, full_matrices=False)
    u_k = u[:, :rank].contiguous()
    v_k = vh[:rank, :].transpose(0, 1).contiguous()
    return u_k, v_k


def _randomized_bases(weight, rank, generator, niter=2, oversample=8):
    out_features, in_features = weight.shape
    sketch_width = min(rank + oversample, out_features, in_features)
    omega = torch.randn(in_features, sketch_width, generator=generator, device=weight.device, dtype=weight.dtype)
    y = weight @ omega
    for _ in range(niter):
        y = weight @ (weight.transpose(0, 1) @ y)
    q, _ = torch.linalg.qr(y)
    b = q.transpose(0, 1) @ weight
    u_small, _, vh = torch.linalg.svd(b, full_matrices=False)
    u = q @ u_small
    u_k = u[:, :rank].contiguous()
    v_k = vh[:rank, :].transpose(0, 1).contiguous()
    return u_k, v_k


def _compute_bases(weight, rank, full_svd, seed, compute_device):
    # SVD always runs in float32 for numerical stability, regardless of the
    # (possibly bf16/fp16) training dtype.
    work = weight.detach().to(device=compute_device, dtype=torch.float32)
    if full_svd:
        u_k, v_k = _full_bases(work, rank)
    else:
        generator = torch.Generator(device=work.device)
        generator.manual_seed(seed)
        u_k, v_k = _randomized_bases(work, rank, generator)
    return u_k, v_k


class _Entry:
    __slots__ = ('name', 'up_param', 'down_param', 'u_k', 'v_k')

    def __init__(self, name, up_param, down_param, u_k, v_k):
        self.name = name
        self.up_param = up_param
        self.down_param = down_param
        self.u_k = u_k
        self.v_k = v_k


class OPLoRAProjector:
    """Builds and applies the OPLoRA projection for every LoRA layer under a module."""

    def __init__(self, entries, rank, full_svd, num_skipped):
        self._entries = entries
        self.rank = rank
        self.full_svd = full_svd
        self.num_skipped = num_skipped

    def __len__(self):
        return len(self._entries)

    def describe(self):
        mode = 'full' if self.full_svd else 'randomized'
        extra = f', skipped {self.num_skipped} non-2D modules' if self.num_skipped else ''
        return f'OPLoRA enabled: projecting {len(self._entries)} LoRA modules with rank={self.rank} ({mode} SVD){extra}'

    @classmethod
    def build(cls, root, rank, full_svd=False, base_seed=0, compute_device=None):
        if rank < 1:
            raise ValueError(f'oplora_rank must be >= 1, got {rank}')
        if compute_device is None:
            compute_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        entries = []
        skipped = 0
        for name, layer in _iter_lora_layers(root):
            base_weight = layer.base_layer.weight
            if base_weight.ndim != 2:
                # configure_adapter only targets nn.Linear, so this is a guard
                # against an unexpected module type rather than a normal case.
                skipped += 1
                continue
            out_features, in_features = base_weight.shape
            if rank > min(out_features, in_features):
                raise ValueError(
                    f'oplora_rank={rank} is larger than the smallest dimension '
                    f'({min(out_features, in_features)}) of layer {name} with weight shape '
                    f'{tuple(base_weight.shape)}'
                )
            adapter_names = [n for n in layer.lora_A.keys() if n in layer.lora_B]
            for adapter_name in adapter_names:
                up_param = layer.lora_B[adapter_name].weight
                down_param = layer.lora_A[adapter_name].weight
                if _is_zero_partitioned(up_param) or _is_zero_partitioned(down_param):
                    raise NotImplementedError(
                        'OPLoRA does not support ZeRO-partitioned parameters. Run with ZeRO stage 0 '
                        '(the default for pipeline-parallel training in this repo).'
                    )
                seed = _stable_seed(base_seed, f'{name}.{adapter_name}')
                u_k, v_k = _compute_bases(base_weight, rank, full_svd, seed, compute_device)
                entries.append(_Entry(
                    f'{name}.{adapter_name}',
                    up_param,
                    down_param,
                    u_k.to(up_param.device),
                    v_k.to(down_param.device),
                ))

        return cls(entries, rank, full_svd, skipped)

    @torch.no_grad()
    def project(self):
        for entry in self._entries:
            up = entry.up_param
            down = entry.down_param
            u_k = entry.u_k
            v_k = entry.v_k

            up32 = up.data.to(torch.float32)
            up32 -= u_k @ (u_k.transpose(0, 1) @ up32)
            up.data.copy_(up32.to(up.dtype))

            down32 = down.data.to(torch.float32)
            down32 -= (down32 @ v_k) @ v_k.transpose(0, 1)
            down.data.copy_(down32.to(down.dtype))

    @torch.no_grad()
    def max_residual(self):
        """Largest leftover overlap of the LoRA factors with the protected subspace.

        Returns the max Frobenius norm of ``U_k^T @ up`` and ``down @ V_k`` across all
        projected modules. After ``project()`` this should be at floating-point noise
        level; it is a cheap health check for tests and GPU verification.
        """
        worst = 0.0
        for entry in self._entries:
            up = entry.up_param.data.to(torch.float32)
            down = entry.down_param.data.to(torch.float32)
            residual_up = (entry.u_k.transpose(0, 1) @ up).norm().item()
            residual_down = (down @ entry.v_k).norm().item()
            worst = max(worst, residual_up, residual_down)
        return worst
