"""End-to-end training tests for the anima_refiner architecture.

These build a small MiniTrainDIT with a ContextRefiner and run real optimisation steps through
the same layer stack CosmosPredict2Pipeline.to_layers() produces, on CPU. They cover:

  * a mock training run, asserting the loss falls and that only the intended parameters move,
  * the Stage 1 freeze contract (DiT frozen, refiner training),
  * the shape invariants pipeline parallelism depends on,
  * the layer-boundary contract that lets the stack be split across GPUs.

Real multi-GPU execution needs hardware this suite does not assume. The shapes and layer
boundaries it checks are the parts that actually break under pipeline parallelism; to exercise
DeepSpeed itself, run test/debug_deepspeed_init.py with `deepspeed --num_gpus=2`.
"""

import pytest
import torch

pytest.importorskip('models.cosmos_predict2')

from models.cosmos_predict2 import (  # noqa: E402
    ContextRefinerLayer,
    FinalLayer,
    InitialLayer,
    TransformerLayer,
)
from models.cosmos_predict2_modeling import MiniTrainDIT  # noqa: E402

CAP_FEAT_DIM = 96
LATENT_CHANNELS = 16
CROSSATTN_DIM = 1024


class _NullOffloader:
    """TransformerLayer talks to a ModelOffloader; block swapping is irrelevant on CPU."""

    def wait_for_block(self, idx):
        pass

    def submit_move_blocks_forward(self, idx):
        pass


def build_dit(num_blocks=2, n_refiner_layers=2, seed=0):
    torch.manual_seed(seed)
    dit = MiniTrainDIT(
        max_img_h=32, max_img_w=32, max_frames=4,
        in_channels=LATENT_CHANNELS, out_channels=LATENT_CHANNELS,
        patch_spatial=2, patch_temporal=1, model_channels=64, concat_padding_mask=True,
        crossattn_emb_channels=CROSSATTN_DIM, pos_emb_cls='rope3d', pos_emb_learnable=True,
        num_blocks=num_blocks, num_heads=4, use_adaln_lora=True, adaln_lora_dim=16,
        cap_feat_dim=CAP_FEAT_DIM, n_refiner_layers=n_refiner_layers,
    )
    dit.context_refiner.init_weights()
    dit.train()
    for name, p in dit.named_parameters():
        p.original_name = name
    return dit


def build_layers(dit):
    """Mirrors CosmosPredict2Pipeline.to_layers() for the refiner architecture."""
    layers = [
        InitialLayer(dit, None, True, True, None),
        ContextRefinerLayer(dit.context_refiner),
    ]
    for i, block in enumerate(dit.blocks):
        layers.append(TransformerLayer(block, i, _NullOffloader()))
    layers.append(FinalLayer(dit))
    return layers


def make_inputs(batch=2, height=16, width=16, text_len=12, valid=8, seed=1):
    torch.manual_seed(seed)
    latents = torch.randn(batch, LATENT_CHANNELS, 1, height, width)
    timesteps = torch.rand(batch, 1)
    text = torch.randn(batch, text_len, CAP_FEAT_DIM)
    mask = torch.ones(batch, text_len, dtype=torch.long)
    if valid < text_len:
        mask[-1, valid:] = 0
    return (latents, timesteps, text, mask)


def run_stack(layers, inputs):
    for layer in layers:
        inputs = layer(inputs)
    return inputs


class TestForwardPass:
    def test_full_stack_runs_and_shapes_match(self):
        dit = build_dit()
        layers = build_layers(dit)
        inputs = make_inputs()
        out = run_stack(layers, inputs)
        assert out.shape == inputs[0].shape

    def test_layer_stack_composition(self):
        dit = build_dit(num_blocks=3)
        layers = build_layers(dit)
        assert [type(layer).__name__ for layer in layers] == [
            'InitialLayer', 'ContextRefinerLayer',
            'TransformerLayer', 'TransformerLayer', 'TransformerLayer',
            'FinalLayer',
        ]

    def test_refiner_actually_participates(self):
        """Perturbing refiner weights must change the output, or it is not in the graph."""
        dit = build_dit()
        layers = build_layers(dit)
        inputs = make_inputs()
        before = run_stack(layers, inputs)
        with torch.no_grad():
            dit.context_refiner.cap_embedder[1].weight.add_(1.0)
        after = run_stack(layers, inputs)
        assert not torch.allclose(before, after)


class TestMockTraining:
    """A real optimisation loop over the real layer stack."""

    @staticmethod
    def train(dit, layers, steps=30, lr=1e-3, params=None):
        inputs = make_inputs()
        torch.manual_seed(7)
        target = torch.randn_like(inputs[0])
        params = list(params if params is not None else dit.parameters())
        optimizer = torch.optim.AdamW([p for p in params if p.requires_grad], lr=lr)
        losses = []
        for _ in range(steps):
            out = run_stack(layers, inputs)
            loss = torch.nn.functional.mse_loss(out.float(), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    def test_loss_decreases_training_refiner_only(self):
        dit = build_dit()
        layers = build_layers(dit)
        refiner_params = [p for n, p in dit.named_parameters() if 'context_refiner' in n]
        for name, p in dit.named_parameters():
            p.requires_grad_('context_refiner' in name)

        losses = self.train(dit, layers, params=refiner_params)
        assert losses[-1] < losses[0], f'loss did not fall: {losses[0]:.5f} -> {losses[-1]:.5f}'

    def test_stage1_freeze_leaves_dit_untouched(self):
        """The Stage 1 contract: only context_refiner weights may move."""
        dit = build_dit()
        layers = build_layers(dit)
        for name, p in dit.named_parameters():
            p.requires_grad_('context_refiner' in name)

        snapshot = {n: p.detach().clone() for n, p in dit.named_parameters()}
        self.train(dit, layers, steps=10,
                   params=[p for n, p in dit.named_parameters() if 'context_refiner' in n])

        moved = {n for n, p in dit.named_parameters() if not torch.equal(p.detach(), snapshot[n])}
        assert moved, 'nothing trained at all'
        assert all('context_refiner' in n for n in moved), (
            f'frozen DiT parameters changed: {sorted(n for n in moved if "context_refiner" not in n)[:5]}'
        )

    def test_gradients_reach_the_refiner_through_the_dit(self):
        dit = build_dit()
        layers = build_layers(dit)
        inputs = make_inputs()
        out = run_stack(layers, inputs)
        out.float().square().mean().backward()

        got_grad = [n for n, p in dit.named_parameters()
                    if 'context_refiner' in n and p.grad is not None and p.grad.abs().max() > 0]
        assert got_grad, 'no gradient reached the refiner'
        assert any('cap_embedder' in n for n in got_grad)
        assert any('blocks.' in n for n in got_grad)

    def test_stage2_also_moves_cross_attention(self):
        """Stage 2 opens cross-attention k/v; self-attention and mlp stay put."""
        dit = build_dit()
        layers = build_layers(dit)
        trainable = []
        for name, p in dit.named_parameters():
            train_it = 'context_refiner' in name or '.cross_attn.k_proj' in name or '.cross_attn.v_proj' in name
            p.requires_grad_(train_it)
            if train_it:
                trainable.append(p)

        snapshot = {n: p.detach().clone() for n, p in dit.named_parameters()}
        self.train(dit, layers, steps=10, params=trainable)
        moved = {n for n, p in dit.named_parameters() if not torch.equal(p.detach(), snapshot[n])}

        assert any('cross_attn.k_proj' in n for n in moved)
        assert any('context_refiner' in n for n in moved)
        assert not any('.self_attn.' in n for n in moved)
        assert not any('.mlp.' in n and 'context_refiner' not in n for n in moved)


class TestPipelineParallelInvariants:
    """Shapes must not depend on batch content, or pipeline parallelism breaks.

    lumina_2.py and z_image.py both carry comments about this: DeepSpeed's pipeline engine
    sends fixed-size tensors between stages, so two micro batches that produce differently
    shaped activations deadlock or corrupt the pipe.
    """

    def test_shapes_are_independent_of_caption_length(self):
        dit = build_dit()
        layers = build_layers(dit)
        # Same padded length, different numbers of real tokens -- exactly what varying caption
        # lengths look like after padding='max_length'.
        short = make_inputs(text_len=12, valid=2)
        long = make_inputs(text_len=12, valid=11)

        shapes_short = self.collect_shapes(layers, short)
        shapes_long = self.collect_shapes(layers, long)
        assert shapes_short == shapes_long

    @staticmethod
    def collect_shapes(layers, inputs):
        shapes = []
        for layer in layers:
            inputs = layer(inputs)
            if isinstance(inputs, tuple):
                shapes.append(tuple(tuple(t.shape) for t in inputs))
            else:
                shapes.append(tuple(inputs.shape))
        return shapes

    def test_shapes_are_stable_across_micro_batches(self):
        dit = build_dit()
        layers = build_layers(dit)
        first = self.collect_shapes(layers, make_inputs(seed=1))
        second = self.collect_shapes(layers, make_inputs(seed=2))
        assert first == second

    def test_every_layer_boundary_is_a_valid_split_point(self):
        """A pipeline_stages=N run cuts the layer list; each cut must still compose.

        Feeding one half's output into the other half is exactly what DeepSpeed does across a
        stage boundary, so this covers single-GPU (no split) and multi-GPU (any split) alike.
        """
        dit = build_dit(num_blocks=3)
        layers = build_layers(dit)
        reference = run_stack(layers, make_inputs())

        for split in range(1, len(layers)):
            inputs = make_inputs()
            for layer in layers[:split]:
                inputs = layer(inputs)
            assert isinstance(inputs, tuple), f'split at {split} does not hand over a tuple'
            for layer in layers[split:]:
                inputs = layer(inputs)
            torch.testing.assert_close(inputs, reference, msg=f'split at layer {split} changed the result')

    def test_intermediate_tensors_are_contiguous(self):
        """make_contiguous() exists because the pipeline engine requires it."""
        dit = build_dit()
        layers = build_layers(dit)
        inputs = make_inputs()
        for layer in layers[:-1]:
            inputs = layer(inputs)
            for tensor in inputs:
                assert tensor.is_contiguous(), f'{type(layer).__name__} emitted a non-contiguous tensor'


class TestDistillObjective:
    """The core of tools/distill_refiner.py: matching at the cross-attention output."""

    def test_distillation_loss_decreases(self):
        dit = build_dit()
        cross_attn = dit.blocks[0].cross_attn
        cross_attn.requires_grad_(False)
        refiner = dit.context_refiner

        torch.manual_seed(3)
        # Stand-in for the teacher's LLMAdapter output: different sequence length from the
        # student's, exactly as real T5 vs LLM tokenization would give.
        teacher_feats = torch.randn(2, 17, CROSSATTN_DIM)
        student_hidden = torch.randn(2, 12, CAP_FEAT_DIM)
        student_mask = torch.ones(2, 12, dtype=torch.long)
        probe = torch.randn(1, 8, 64).expand(2, -1, -1)

        with torch.no_grad():
            target = cross_attn(probe, context=teacher_feats)

        optimizer = torch.optim.AdamW(refiner.parameters(), lr=1e-3)
        losses = []
        for _ in range(40):
            pred = cross_attn(probe, context=refiner(student_hidden, student_mask))
            loss = torch.nn.functional.mse_loss(pred.float(), target.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] * 0.9, (
            f'distillation loss barely moved: {losses[0]:.5f} -> {losses[-1]:.5f}. The objective '
            'must be learnable despite teacher and student having different sequence lengths.'
        )


class TestWeightRoundTrip:
    """The refiner has to survive being saved and reloaded, in both save formats."""

    def test_full_finetune_save_format_keeps_refiner(self):
        """save_model() prefixes 'net.'; load_diffusion_model() strips it and detects the
        refiner by looking for context_refiner.* keys."""
        dit = build_dit()
        saved = {'net.' + k: v for k, v in dit.state_dict().items()}
        stripped = {(k[len('net.'):] if k.startswith('net.') else k): v for k, v in saved.items()}

        assert any(k.startswith('context_refiner.') for k in stripped), (
            'a full fine tune checkpoint must carry the refiner, or reloading it silently '
            'rebuilds an untrained one'
        )
        # And the DiT would then load them through the ordinary named_parameters loop.
        names = {n for n, _ in dit.named_parameters()}
        assert {k for k in stripped if k.startswith('context_refiner.')} <= names

    def test_refiner_state_dict_loads_into_a_fresh_module(self):
        """context_refiner_path strips the prefix and loads into ContextRefiner directly."""
        dit = build_dit(seed=0)
        other = build_dit(seed=5)
        extracted = {
            k[len('context_refiner.'):]: v
            for k, v in dit.state_dict().items() if k.startswith('context_refiner.')
        }
        other.context_refiner.load_state_dict(extracted)
        for (n, a), (_, b) in zip(dit.context_refiner.named_parameters(),
                                  other.context_refiner.named_parameters()):
            torch.testing.assert_close(a, b, msg=f'{n} did not round trip')


class TestAdapterConfiguration:
    """LoRA/LoKr interaction with the refiner, which peft would otherwise freeze."""

    @staticmethod
    def configure(dit, train_context_refiner):
        import models.base as base
        from models.cosmos_predict2 import CosmosPredict2Pipeline

        class Stub(CosmosPredict2Pipeline):
            def __init__(self, transformer):
                self.transformer = transformer
                self.use_context_refiner = True
                self.model_config = {'dtype': torch.float32}
                self.adapter_target_modules = ['Block', 'ContextRefiner']

        stub = Stub(dit)
        adapter_config = {'type': 'lora', 'rank': 4, 'alpha': 4, 'dropout': 0.0, 'dtype': torch.float32}
        if train_context_refiner:
            adapter_config['train_context_refiner'] = True

        original = base.is_main_process
        base.is_main_process = lambda: False
        try:
            stub.configure_adapter(adapter_config)
        finally:
            base.is_main_process = original
        return stub

    @staticmethod
    def counts(dit):
        trainable = [n for n, p in dit.named_parameters() if p.requires_grad]
        return {
            'lora_on_refiner': sum('context_refiner' in n and 'lora_' in n for n in trainable),
            'dense_refiner': sum('context_refiner' in n and 'lora_' not in n for n in trainable),
            'lora_on_dit': sum('context_refiner' not in n and 'lora_' in n for n in trainable),
        }

    def test_default_adapts_the_refiner(self):
        dit = build_dit()
        stub = self.configure(dit, train_context_refiner=False)
        counts = self.counts(dit)
        assert 'ContextRefiner' in stub.adapter_target_modules
        assert counts['lora_on_refiner'] > 0
        # cap_embedder is the 2048->1024 projection absorbing the whole distribution gap and
        # the largest tensor in the refiner. Targeting RefinerBlock instead would silently
        # leave it frozen while this assertion's weaker sibling above still passed.
        adapted = [n for n, p in dit.named_parameters() if p.requires_grad]
        assert any('cap_embedder' in n and 'lora_' in n for n in adapted), 'cap_embedder must be adapted'
        assert counts['dense_refiner'] == 0, 'base weights must stay frozen under peft'
        assert counts['lora_on_dit'] > 0

    def test_train_context_refiner_trains_it_densely_instead(self):
        """A freshly initialised refiner has nothing for a low-rank update to build on."""
        dit = build_dit()
        stub = self.configure(dit, train_context_refiner=True)
        counts = self.counts(dit)
        assert 'ContextRefiner' not in stub.adapter_target_modules
        assert counts['lora_on_refiner'] == 0
        assert counts['dense_refiner'] > 0
        assert counts['lora_on_dit'] > 0, 'the DiT must still get its adapter'
