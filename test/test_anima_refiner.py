"""Tests for the anima_refiner text frontend (models/text_refiner.py and its wiring).

These run on CPU without DeepSpeed or GPU, and never download model weights. The heavier
integration tests skip themselves when their imports are unavailable.
"""

import pytest
import torch
from torch import nn

from models.text_refiner import ContextRefiner, RefinerBlock


CAP_FEAT_DIM = 128
MODEL_DIM = 64


def make_refiner(num_layers=2, cap_feat_dim=CAP_FEAT_DIM, model_dim=MODEL_DIM, seed=0):
    torch.manual_seed(seed)
    refiner = ContextRefiner(
        cap_feat_dim=cap_feat_dim, model_dim=model_dim, num_layers=num_layers, num_heads=4
    )
    refiner.init_weights()
    refiner.eval()
    return refiner


def make_batch(batch=2, length=12, cap_feat_dim=CAP_FEAT_DIM, valid=8, seed=1):
    torch.manual_seed(seed)
    hidden = torch.randn(batch, length, cap_feat_dim)
    mask = torch.ones(batch, length, dtype=torch.long)
    mask[-1, valid:] = 0
    return hidden, mask


class TestContextRefiner:
    def test_output_shape(self):
        refiner = make_refiner()
        hidden, mask = make_batch()
        out = refiner(hidden, mask)
        assert out.shape == (hidden.shape[0], hidden.shape[1], MODEL_DIM)

    def test_runs_without_mask(self):
        refiner = make_refiner()
        hidden, _ = make_batch()
        assert refiner(hidden).shape == (hidden.shape[0], hidden.shape[1], MODEL_DIM)

    def test_identity_at_init(self):
        """Zero-init residual branches mean the refiner starts as a plain linear projection.

        This is what makes a fresh refiner a stable starting point in front of a frozen DiT.
        """
        refiner = make_refiner(num_layers=4)
        hidden, mask = make_batch()
        out = refiner(hidden, mask)
        expected = refiner.norm_out(refiner.cap_embedder(hidden)) * mask.unsqueeze(-1)
        torch.testing.assert_close(out, expected)

    def test_padded_positions_are_zeroed(self):
        refiner = make_refiner()
        hidden, mask = make_batch(valid=8)
        out = refiner(hidden, mask)
        assert out[-1, 8:].abs().max() == 0

    def test_padding_does_not_leak_into_real_tokens(self):
        """Changing masked-out input must not move the output at unmasked positions."""
        refiner = make_refiner()
        hidden, mask = make_batch(valid=8)
        other = hidden.clone()
        torch.manual_seed(99)
        other[-1, 8:] = torch.randn_like(other[-1, 8:])
        out_a = refiner(hidden, mask)
        out_b = refiner(other, mask)
        torch.testing.assert_close(out_a[-1, :8], out_b[-1, :8])

    def test_attention_is_bidirectional(self):
        """A later token must be able to influence an earlier one.

        This is the property the refiner exists to restore: causal LLMs (and hybrid
        linear-attention models like Qwen3.5) only give each position left context, while the
        DiT cross-attention was trained on bidirectional T5 encoder output.
        """
        refiner = make_refiner(num_layers=2)
        # Non-zero residual branches, otherwise the identity init hides all mixing.
        for block in refiner.blocks:
            nn.init.normal_(block.attn.o_proj.weight, std=0.05)
            nn.init.normal_(block.mlp[2].weight, std=0.05)

        hidden, mask = make_batch(batch=1, length=8, valid=8)
        out_a = refiner(hidden, mask)
        perturbed = hidden.clone()
        perturbed[0, 7] += 10.0  # change the LAST token only
        out_b = refiner(perturbed, mask)

        first_token_delta = (out_a[0, 0] - out_b[0, 0]).abs().max()
        assert first_token_delta > 1e-4, 'later tokens must influence earlier ones'

    def test_state_dict_round_trip(self):
        refiner = make_refiner(seed=0)
        reloaded = make_refiner(seed=7)
        hidden, mask = make_batch()

        # Perturb so the two are genuinely different before loading.
        with torch.no_grad():
            for p in refiner.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        assert not torch.allclose(refiner(hidden, mask), reloaded(hidden, mask))
        reloaded.load_state_dict(refiner.state_dict())
        torch.testing.assert_close(refiner(hidden, mask), reloaded(hidden, mask))

    def test_rotary_buffer_is_not_persistent(self):
        """inv_freq must stay out of the state dict so saved refiners load cleanly."""
        refiner = make_refiner()
        assert not any('inv_freq' in k for k in refiner.state_dict())

    def test_gradients_flow_through_zero_init(self):
        """Zero-initialised weights must still receive gradient, or the blocks never train."""
        refiner = make_refiner(num_layers=2)
        refiner.train()
        hidden, mask = make_batch()
        refiner(hidden, mask).square().mean().backward()
        for i, block in enumerate(refiner.blocks):
            assert block.attn.o_proj.weight.grad is not None
            assert block.attn.o_proj.weight.grad.abs().max() > 0, f'block {i} o_proj got no gradient'
            assert block.mlp[2].weight.grad.abs().max() > 0, f'block {i} mlp got no gradient'

    def test_param_names_are_stable(self):
        """get_param_groups and KEEP_IN_HIGH_PRECISION match on these substrings."""
        refiner = make_refiner()
        names = set(refiner.state_dict())
        assert 'cap_embedder.1.weight' in names
        assert 'cap_embedder.1.bias' in names
        assert 'norm_out.weight' in names
        assert any(n.startswith('blocks.0.attn.') for n in names)


class TestRefinerBlock:
    def test_zero_init_is_identity(self):
        torch.manual_seed(0)
        block = RefinerBlock(MODEL_DIM, num_heads=4)
        block.init_weights()
        block.eval()
        x = torch.randn(2, 6, MODEL_DIM)
        torch.testing.assert_close(block(x), x)


class TestCrossAttentionInvariance:
    """The distillation objective in tools/distill_refiner.py rests on this property.

    Teacher features are indexed by T5 tokens and student features by the source LLM's
    tokens, so a position-wise loss compares unrelated slots. Cross-attention output is a
    weighted sum over text positions and does not depend on their order, which is why the
    distillation loss is measured there instead.
    """

    def test_cross_attention_output_is_permutation_invariant(self):
        modeling = pytest.importorskip('models.cosmos_predict2_modeling')
        torch.manual_seed(0)
        attn = modeling.Attention(
            query_dim=32, context_dim=MODEL_DIM, n_heads=4, head_dim=8, backend='torch'
        )
        attn.eval()
        query = torch.randn(1, 5, 32)
        context = torch.randn(1, 10, MODEL_DIM)
        permutation = torch.randperm(10)

        with torch.no_grad():
            out = attn(query, context=context)
            out_permuted = attn(query, context=context[:, permutation])

        torch.testing.assert_close(out, out_permuted, atol=1e-5, rtol=1e-5)

    def test_output_magnitude_depends_on_padded_length(self):
        """The property permutation invariance does NOT give you, and the reason the auxiliary
        distillation term normalises by the padded length rather than the real token count.

        Padded rows project to k = 0 (k_proj has no bias, RMSNorm(0) = 0), so each contributes
        exp(0) = 1 to the softmax denominator and nothing to the numerator. The same real
        content therefore yields a smaller output as padding grows -- and teacher and student
        tokenizers give different real-token counts for the same caption.
        """
        modeling = pytest.importorskip('models.cosmos_predict2_modeling')
        torch.manual_seed(0)
        attn = modeling.Attention(query_dim=32, context_dim=MODEL_DIM, n_heads=4, head_dim=8, backend='torch')
        attn.eval()
        query = torch.randn(1, 5, 32)
        real = torch.randn(1, 8, MODEL_DIM)

        norms = []
        for total in (16, 64, 512):
            context = torch.zeros(1, total, MODEL_DIM)
            context[:, :8] = real
            with torch.no_grad():
                norms.append(attn(query, context=context).norm().item())

        assert norms[0] > norms[1] > norms[2], f'expected dilution with padding, got {norms}'
        assert norms[0] / norms[2] > 5, 'the effect is large, not a rounding detail'


class TestMiniTrainDitWiring:
    @staticmethod
    def build(**overrides):
        modeling = pytest.importorskip('models.cosmos_predict2_modeling')
        kwargs = dict(
            max_img_h=64, max_img_w=64, max_frames=8, in_channels=17, out_channels=16,
            patch_spatial=2, patch_temporal=1, model_channels=128, concat_padding_mask=True,
            crossattn_emb_channels=1024, pos_emb_cls='rope3d', pos_emb_learnable=True,
            num_blocks=2, num_heads=4, use_adaln_lora=True, adaln_lora_dim=32,
        )
        kwargs.update(overrides)
        return modeling.MiniTrainDIT(**kwargs)

    def test_refiner_built_when_cap_feat_dim_given(self):
        model = self.build(cap_feat_dim=2048, n_refiner_layers=2)
        assert model.use_context_refiner
        assert not model.use_llm_adapter
        assert model.context_refiner.cap_embedder[1].in_features == 2048
        assert model.context_refiner.cap_embedder[1].out_features == 1024
        assert len(model.context_refiner.blocks) == 2

    def test_llm_adapter_path_unchanged(self):
        """Existing anima / cosmos_predict2 training must not be affected."""
        model = self.build(use_llm_adapter=True)
        assert model.use_llm_adapter
        assert not model.use_context_refiner
        assert not hasattr(model, 'context_refiner')

    def test_neither_by_default(self):
        model = self.build()
        assert not model.use_llm_adapter
        assert not model.use_context_refiner

    def test_refiner_and_llm_adapter_are_mutually_exclusive(self):
        with pytest.raises(AssertionError):
            self.build(cap_feat_dim=2048, use_llm_adapter=True)

    def test_refiner_params_are_named_for_matching(self):
        model = self.build(cap_feat_dim=2048, n_refiner_layers=1)
        names = [n for n, _ in model.named_parameters() if 'context_refiner' in n]
        assert names, 'refiner parameters must contain "context_refiner" for param grouping'
        assert all(n.startswith('context_refiner.') for n in names)


class TestPipelineIntegration:
    """Wiring inside models/cosmos_predict2.py. Skipped when its heavy imports are missing."""

    @staticmethod
    def pipeline_module():
        return pytest.importorskip('models.cosmos_predict2')

    def test_context_refiner_layer_tuple_arity(self):
        """ContextRefinerLayer must consume InitialLayer's 7-tuple and emit TransformerLayer's 6."""
        module = self.pipeline_module()
        refiner = make_refiner(model_dim=MODEL_DIM)
        layer = module.ContextRefinerLayer(refiner)

        hidden, mask = make_batch()
        inputs = (
            torch.randn(2, 1, 4, 4, 8),   # x_B_T_H_W_D
            torch.randn(2, 1, 8),         # t_embedding
            hidden,                       # crossattn_emb (pre-refiner)
            mask,                         # attn_mask
            torch.randn(4, 1, 1, 8),      # rope
            torch.randn(2, 1, 24),        # adaln_lora
            torch.randn(2, 1),            # timesteps
        )
        outputs = layer(inputs)
        assert len(outputs) == 6, 'must drop attn_mask before the transformer blocks'
        assert outputs[2].shape == (2, hidden.shape[1], MODEL_DIM)

    def test_keep_in_high_precision_covers_refiner(self):
        module = self.pipeline_module()
        assert 'context_refiner' in module.KEEP_IN_HIGH_PRECISION

    def test_checkpointable_layers_covers_refiner(self):
        module = self.pipeline_module()
        assert 'ContextRefinerLayer' in module.CosmosPredict2Pipeline.checkpointable_layers

    def _param_groups(self, model_config, params, use_context_refiner=True):
        module = self.pipeline_module()

        class Stub:
            pass

        stub = Stub()
        stub.model_config = model_config
        stub.config = {'optimizer': {'lr': 1e-4}}
        stub.use_context_refiner = use_context_refiner
        # is_main_process() calls into deepspeed.comm, which is not initialised under pytest.
        original = module.is_main_process
        module.is_main_process = lambda: False
        try:
            return module.CosmosPredict2Pipeline.get_param_groups(stub, params)
        finally:
            module.is_main_process = original

    @staticmethod
    def named_param(name, requires_grad=True):
        p = nn.Parameter(torch.zeros(2), requires_grad=requires_grad)
        p.original_name = name
        return p

    def test_refiner_params_get_their_own_group(self):
        refiner_mlp = self.named_param('context_refiner.blocks.0.mlp.0.weight')
        refiner_attn = self.named_param('context_refiner.blocks.0.attn.q_proj.weight')
        dit_cross = self.named_param('blocks.0.cross_attn.k_proj.weight')
        groups = self._param_groups({'refiner_lr': 5e-4, 'cross_attn_lr': 1e-6}, [refiner_mlp, refiner_attn, dit_cross])

        by_lr = {g['lr']: g['params'] for g in groups}
        assert set(by_lr[5e-4]) == {refiner_mlp, refiner_attn}, (
            'refiner params must not fall through into the DiT mlp/attn groups'
        )
        assert by_lr[1e-6] == [dit_cross]

    def test_refiner_lr_zero_freezes_refiner(self):
        refiner_param = self.named_param('context_refiner.norm_out.weight')
        groups = self._param_groups({'refiner_lr': 0}, [refiner_param])
        assert refiner_param.requires_grad is False
        assert groups == []

    def test_base_lr_override_freezes_everything_else(self):
        """Stage 1: base_lr = 0 freezes the DiT while the refiner still trains."""
        base = self.named_param('x_embedder.weight')
        self_attn = self.named_param('blocks.0.self_attn.q_proj.weight')
        refiner = self.named_param('context_refiner.cap_embedder.1.weight')
        groups = self._param_groups({'base_lr': 0, 'refiner_lr': 1e-4}, [base, self_attn, refiner])

        assert base.requires_grad is False
        assert self_attn.requires_grad is False
        assert refiner.requires_grad is True
        assert len(groups) == 1
        assert groups[0]['params'] == [refiner]
        assert groups[0]['lr'] == 1e-4


class TestAdapterSaving:
    """save_adapter must not confuse LoRA tensors with densely trained refiner weights."""

    @staticmethod
    def call_save_adapter(train_context_refiner, peft_state_dict, tmp_path):
        module = pytest.importorskip('models.cosmos_predict2')

        class Stub:
            train_context_refiner = None
            saved = {}

            def __init__(self):
                self.peft_config = type('C', (), {'save_pretrained': lambda s, d: None})()

        stub = Stub()
        stub.train_context_refiner = train_context_refiner
        written = {}

        import safetensors.torch
        original = safetensors.torch.save_file
        safetensors.torch.save_file = lambda sd, path, metadata=None: written.__setitem__(
            str(path).rsplit('/', 1)[-1], dict(sd)
        )
        try:
            module.CosmosPredict2Pipeline.save_adapter(stub, tmp_path, peft_state_dict)
        finally:
            safetensors.torch.save_file = original
        return written

    def test_lora_tensors_on_the_refiner_stay_in_the_adapter(self, tmp_path):
        """When the refiner is a LoRA target its names contain 'context_refiner' too."""
        state = {
            'context_refiner.blocks.0.attn.q_proj.lora_A.weight': torch.zeros(2),
            'blocks.0.cross_attn.k_proj.lora_A.weight': torch.zeros(2),
        }
        written = self.call_save_adapter(False, dict(state), tmp_path)
        assert 'context_refiner.safetensors' not in written
        assert len(written['adapter_model.safetensors']) == 2

    def test_dense_refiner_weights_are_split_out(self, tmp_path):
        state = {
            'context_refiner.cap_embedder.1.weight': torch.zeros(2),
            'blocks.0.cross_attn.k_proj.lora_A.weight': torch.zeros(2),
        }
        written = self.call_save_adapter(True, dict(state), tmp_path)
        # Saved under names context_refiner_path can load directly.
        assert set(written['context_refiner.safetensors']) == {'cap_embedder.1.weight'}
        assert set(written['adapter_model.safetensors']) == {
            'diffusion_model.blocks.0.cross_attn.k_proj.lora_A.weight'
        }


class TestCaptionEnumeration:
    """tools/distill_refiner.py reads captions through the ordinary dataset.toml flow.

    It must resolve captions the same way DirectoryDataset does, so the distillation stage
    sees the caption distribution training will see -- while never opening an image.
    """

    @staticmethod
    def enumerate_captions(*args, **kwargs):
        dataset = pytest.importorskip('utils.dataset')
        return dataset.enumerate_captions(*args, **kwargs)

    @staticmethod
    def make_dir(tmp_path, files):
        for name, content in files.items():
            (tmp_path / name).write_text(content)
        return {'directory': [{'path': str(tmp_path)}]}

    def test_reads_txt_captions_next_to_images(self, tmp_path):
        config = self.make_dir(tmp_path, {
            'a.jpg': 'x', 'a.txt': '1girl, solo',
            'b.jpg': 'x', 'b.txt': 'landscape',
        })
        assert sorted(self.enumerate_captions(config)) == ['1girl, solo', 'landscape']

    def test_captions_json_takes_priority(self, tmp_path):
        config = self.make_dir(tmp_path, {
            'a.jpg': 'x', 'a.txt': 'from txt',
            'captions.json': '{"a.jpg": ["from json"]}',
        })
        assert self.enumerate_captions(config) == ['from json']

    def test_caption_prefix_is_applied(self, tmp_path):
        config = self.make_dir(tmp_path, {'a.jpg': 'x', 'a.txt': '1girl'})
        config['caption_prefix'] = 'anime, '
        assert self.enumerate_captions(config) == ['anime, 1girl']

    def test_skip_empty_caption(self, tmp_path):
        config = self.make_dir(tmp_path, {'a.jpg': 'x', 'b.jpg': 'x', 'b.txt': 'has one'})
        assert self.enumerate_captions(config) == ['has one']

        config['skip_empty_caption'] = False
        assert sorted(self.enumerate_captions(config)) == ['', 'has one']

    def test_tag_shuffling_multiplies_captions(self, tmp_path):
        config = self.make_dir(tmp_path, {'a.jpg': 'x', 'a.txt': 'one, two, three'})
        config['cache_shuffle_num'] = 4
        captions = self.enumerate_captions(config)
        assert len(captions) == 4
        for caption in captions:
            assert sorted(caption.split(', ')) == ['one', 'three', 'two']

    def test_num_repeats_is_opt_in(self, tmp_path):
        config = self.make_dir(tmp_path, {'a.jpg': 'x', 'a.txt': 'c'})
        config['directory'][0]['num_repeats'] = 3
        assert self.enumerate_captions(config) == ['c']
        assert self.enumerate_captions(config, apply_num_repeats=True) == ['c', 'c', 'c']

    def test_text_only_directory_is_accepted(self, tmp_path):
        """Distillation needs no images, so a folder of bare .txt files still works."""
        config = self.make_dir(tmp_path, {'a.txt': 'first', 'b.txt': 'second'})
        assert sorted(self.enumerate_captions(config)) == ['first', 'second']

    def test_multiple_directories_are_concatenated(self, tmp_path):
        d1, d2 = tmp_path / 'one', tmp_path / 'two'
        d1.mkdir(); d2.mkdir()
        (d1 / 'a.jpg').write_text('x'); (d1 / 'a.txt').write_text('from one')
        (d2 / 'b.jpg').write_text('x'); (d2 / 'b.txt').write_text('from two')
        config = {'directory': [{'path': str(d1)}, {'path': str(d2)}]}
        assert sorted(self.enumerate_captions(config)) == ['from one', 'from two']

    def test_non_media_files_are_not_treated_as_images(self, tmp_path):
        config = self.make_dir(tmp_path, {
            'a.jpg': 'x', 'a.txt': 'real',
            'notes.bak': 'x', 'cache.db': 'x', 'meta.parquet': 'x', 'z.npz': 'x',
        })
        assert self.enumerate_captions(config) == ['real']


class TestCheckpointResolution:
    """_resolve_context_refiner: shape comes from the weights, never from the config.

    The loading loop skips names absent from the state dict, so a checkpoint holding more
    refiner layers than the config asked for would quietly lose the surplus. Deriving the
    shape from the weights removes that failure mode.
    """

    @staticmethod
    def resolve(model_config, state_dict, cap_feat_dim=CAP_FEAT_DIM, dtype=torch.float32):
        module = pytest.importorskip('models.cosmos_predict2')

        class Stub:
            pass

        stub = Stub()
        stub.model_config = model_config
        stub.cap_feat_dim = cap_feat_dim
        original = module.is_main_process
        module.is_main_process = lambda: False
        dit_config = {}
        try:
            result = module.CosmosPredict2Pipeline._resolve_context_refiner(
                stub, state_dict, dit_config, dtype
            )
        finally:
            module.is_main_process = original
        return result, dit_config

    @staticmethod
    def refiner_weights(num_layers, cap_feat_dim=CAP_FEAT_DIM, prefix=''):
        refiner = make_refiner(num_layers=num_layers, cap_feat_dim=cap_feat_dim)
        return {prefix + k: v for k, v in refiner.state_dict().items()}

    def test_fresh_init_when_checkpoint_has_no_refiner(self):
        result, dit_config = self.resolve({'n_refiner_layers': 4}, {'blocks.0.mlp.weight': torch.zeros(2)})
        assert result == 'init'
        assert dit_config['n_refiner_layers'] == 4
        assert dit_config['cap_feat_dim'] == CAP_FEAT_DIM

    def test_layer_count_is_derived_from_the_checkpoint(self):
        state_dict = self.refiner_weights(3, prefix='context_refiner.')
        result, dit_config = self.resolve({}, state_dict)
        assert result is None, 'weights already in the checkpoint are loaded by the main loop'
        assert dit_config['n_refiner_layers'] == 3
        assert dit_config['cap_feat_dim'] == CAP_FEAT_DIM

    def test_config_contradicting_the_checkpoint_raises(self):
        """The silent-layer-drop bug: config said 6, checkpoint had 3."""
        state_dict = self.refiner_weights(3, prefix='context_refiner.')
        with pytest.raises(RuntimeError, match='n_refiner_layers'):
            self.resolve({'n_refiner_layers': 6}, state_dict)

    def test_config_agreeing_with_the_checkpoint_is_fine(self):
        state_dict = self.refiner_weights(3, prefix='context_refiner.')
        _, dit_config = self.resolve({'n_refiner_layers': 3}, state_dict)
        assert dit_config['n_refiner_layers'] == 3

    def test_text_encoder_size_mismatch_raises(self):
        state_dict = self.refiner_weights(2, cap_feat_dim=64, prefix='context_refiner.')
        with pytest.raises(RuntimeError, match='hidden size'):
            self.resolve({}, state_dict, cap_feat_dim=2048)

    def test_context_refiner_path_wins_over_the_checkpoint(self, tmp_path):
        import safetensors.torch
        path = tmp_path / 'refiner.safetensors'
        external = self.refiner_weights(2)
        safetensors.torch.save_file({k: v.contiguous() for k, v in external.items()}, str(path))

        # Checkpoint has a DIFFERENT layer count, so the winner is unambiguous.
        state_dict = self.refiner_weights(4, prefix='context_refiner.')
        result, dit_config = self.resolve({'context_refiner_path': str(path),
                                           'transformer_path': 'ckpt.safetensors'}, state_dict)
        assert result is not None, 'the external file must be loaded explicitly'
        assert dit_config['n_refiner_layers'] == 2, 'shape must follow the file that won'

    def test_context_refiner_path_alone_still_works(self, tmp_path):
        import safetensors.torch
        path = tmp_path / 'refiner.safetensors'
        external = self.refiner_weights(2)
        safetensors.torch.save_file({k: v.contiguous() for k, v in external.items()}, str(path))
        result, dit_config = self.resolve({'context_refiner_path': str(path)}, {})
        assert set(result) == set(external)
        assert dit_config['n_refiner_layers'] == 2


class TestTextEncoderCacheKey:
    """Swapping the text encoder must re-cache text embeddings but never the latents."""

    @staticmethod
    def key_for(model_config, use_refiner=True, cap_feat_dim=2048):
        module = pytest.importorskip('models.cosmos_predict2')

        class Stub:
            pass

        stub = Stub()
        stub.model_config = model_config
        stub.use_context_refiner = use_refiner
        stub.cap_feat_dim = cap_feat_dim
        stub.llm_hidden_layer = model_config.get('llm_hidden_layer', None)
        stub.max_text_length = model_config.get('max_text_length', 512)
        return module.CosmosPredict2Pipeline.text_encoder_cache_key(stub, 0)

    def test_other_models_supply_no_key(self):
        """Existing models must keep their exact cache fingerprint."""
        assert self.key_for({}, use_refiner=False) == ''

    def test_key_changes_with_the_llm(self):
        a = self.key_for({'llm_path': '/models/qwen3_5_2b_base'})
        b = self.key_for({'llm_path': '/models/something_else'})
        assert a != b

    def test_key_changes_with_hidden_layer_and_length(self):
        base = {'llm_path': '/models/x'}
        assert self.key_for(base) != self.key_for({**base, 'llm_hidden_layer': -2})
        assert self.key_for(base) != self.key_for({**base, 'max_text_length': 256})

    def test_key_is_stable_for_identical_config(self):
        cfg = {'llm_path': '/models/x', 'llm_hidden_layer': -1, 'max_text_length': 512}
        assert self.key_for(cfg) == self.key_for(dict(cfg))


class TestExistingModelFingerprintsUnchanged:
    """The cache change must not invalidate caches for wan / anima / cosmos_predict2.

    Asserted against the arguments _cache_text_embeddings actually passes, not against the
    hashing library's determinism -- the claim is about this repo's conditional, so that is
    what gets exercised.
    """

    @staticmethod
    def captured_fingerprint_args(text_encoder_key):
        dataset = pytest.importorskip('utils.dataset')
        captured = {}

        def fake_map_and_cache(ds, map_fn, cache_dir, cache_file_prefix='', new_fingerprint_args=None, **kwargs):
            captured['args'] = new_fingerprint_args
            captured['prefix'] = cache_file_prefix
            return []

        original = dataset._map_and_cache
        dataset._map_and_cache = fake_map_and_cache
        try:
            fake_metadata = type('D', (), {
                'map': lambda self, *a, **k: self,
                'column_names': [],
            })()
            try:
                dataset._cache_text_embeddings(
                    fake_metadata, map_fn=None, i=0, cache_dir='/tmp',
                    regenerate_cache=False, caching_batch_size=1,
                    text_encoder_key=text_encoder_key,
                )
            except Exception:
                pass  # only the captured arguments matter
        finally:
            dataset._map_and_cache = original
        return captured.get('args')

    def test_no_key_passes_the_original_arguments(self):
        """Models supplying no key must produce byte-identical fingerprint input to before."""
        assert self.captured_fingerprint_args('') == [0]

    def test_a_key_is_appended_when_present(self):
        assert self.captured_fingerprint_args('qwen35-base|-1|512') == [0, 'qwen35-base|-1|512']


class TestBaseLrIsRefinerOnly:
    """base_lr must stay an anima_refiner option; other models keep the optimizer lr."""

    def test_other_models_ignore_base_lr(self):
        integration = TestPipelineIntegration()
        param = integration.named_param('x_embedder.weight')
        # base_lr = 0 would freeze this if the option applied.
        groups = integration._param_groups({'base_lr': 0}, [param], use_context_refiner=False)
        assert param.requires_grad is True, 'base_lr must not affect cosmos_predict2 / anima'
        assert len(groups) == 1
        assert groups[0]['lr'] == 1e-4, 'lr must come from [optimizer] as it always did'

    def test_anima_refiner_honours_base_lr(self):
        integration = TestPipelineIntegration()
        param = integration.named_param('x_embedder.weight')
        groups = integration._param_groups({'base_lr': 0}, [param], use_context_refiner=True)
        assert param.requires_grad is False
        assert groups == []


class TestFullyMaskedRow:
    """An empty caption tokenizes to an all-padding row -- the uncond embedding CFG needs.

    A row with no unmasked key makes the attention softmax degenerate. The CPU math backend
    returns zeros, but the fused CUDA backends can return NaN, and NaN * 0 is still NaN, so
    zeroing the padded output afterwards would not contain it. Such rows are allowed to attend
    freely instead; their output is discarded either way.
    """

    @staticmethod
    def refiner_with_live_residuals(num_layers=2):
        refiner = make_refiner(num_layers=num_layers)
        # Zero-init residuals would hide any attention misbehaviour entirely.
        for block in refiner.blocks:
            nn.init.normal_(block.attn.o_proj.weight, std=0.05)
            nn.init.normal_(block.mlp[2].weight, std=0.05)
        return refiner

    def test_empty_caption_row_is_finite(self):
        refiner = self.refiner_with_live_residuals()
        hidden = torch.randn(2, 16, CAP_FEAT_DIM)
        mask = torch.ones(2, 16, dtype=torch.long)
        mask[0] = 0  # fully padded, exactly what an empty caption produces
        out = refiner(hidden, mask)
        assert torch.isfinite(out).all(), 'a fully masked row must not produce NaN or inf'
        assert out[0].abs().max() == 0, 'its output is still zeroed'

    def test_unmasked_rows_are_unaffected_by_the_guard(self):
        """Letting an empty row attend freely must not alter any other row in the batch."""
        refiner = self.refiner_with_live_residuals()
        hidden = torch.randn(2, 16, CAP_FEAT_DIM)
        with_empty = torch.ones(2, 16, dtype=torch.long)
        with_empty[0] = 0
        all_valid = torch.ones(2, 16, dtype=torch.long)
        torch.testing.assert_close(refiner(hidden, with_empty)[1], refiner(hidden, all_valid)[1])

    def test_all_rows_empty_is_finite(self):
        refiner = self.refiner_with_live_residuals()
        out = refiner(torch.randn(2, 8, CAP_FEAT_DIM), torch.zeros(2, 8, dtype=torch.long))
        assert torch.isfinite(out).all()
        assert out.abs().max() == 0


class TestExtractRefinerStateDict:
    """One helper, used by both the training pipeline and the distillation tool.

    Refiner weights arrive in three shapes and every one has to work wherever a refiner is
    accepted; having each call site reimplement the stripping is what let context_refiner_path
    crash on a full checkpoint.
    """

    @staticmethod
    def bare():
        refiner = make_refiner(num_layers=2)
        return refiner.state_dict()

    def test_bare_refiner_passes_through(self):
        from models.text_refiner import extract_refiner_state_dict
        bare = self.bare()
        assert set(extract_refiner_state_dict(dict(bare))) == set(bare)

    def test_full_checkpoint_with_net_prefix(self):
        from models.text_refiner import extract_refiner_state_dict
        bare = self.bare()
        full = {'net.context_refiner.' + k: v for k, v in bare.items()}
        full['net.blocks.0.self_attn.q_proj.weight'] = torch.zeros(2, 2)
        full['net.x_embedder.proj.1.weight'] = torch.zeros(2, 2)
        assert set(extract_refiner_state_dict(full)) == set(bare)

    def test_checkpoint_already_stripped_of_net(self):
        from models.text_refiner import extract_refiner_state_dict
        bare = self.bare()
        assert set(extract_refiner_state_dict({'context_refiner.' + k: v for k, v in bare.items()})) == set(bare)

    def test_no_refiner_raises_instead_of_returning_junk(self):
        from models.text_refiner import extract_refiner_state_dict
        with pytest.raises(RuntimeError, match='No context_refiner weights found'):
            extract_refiner_state_dict({'net.blocks.0.mlp.weight': torch.zeros(2, 2)})


class TestContextRefinerPathAcceptsFullCheckpoint:
    """context_refiner_path may point at a full model.safetensors, not just a bare refiner."""

    def test_full_checkpoint_resolves(self, tmp_path):
        import safetensors.torch
        module = pytest.importorskip('models.cosmos_predict2')

        refiner = make_refiner(num_layers=2)
        full = {'net.context_refiner.' + k: v.contiguous() for k, v in refiner.state_dict().items()}
        full['net.blocks.0.self_attn.q_proj.weight'] = torch.zeros(4, 4)
        path = tmp_path / 'model.safetensors'
        safetensors.torch.save_file(full, str(path))

        resolution = TestCheckpointResolution()
        result, dit_config = resolution.resolve(
            {'context_refiner_path': str(path), 'transformer_path': 'ckpt.safetensors'}, {}
        )
        assert set(result) == set(refiner.state_dict()), 'net. prefix must be stripped'
        assert dit_config['n_refiner_layers'] == 2
        assert dit_config['cap_feat_dim'] == CAP_FEAT_DIM


class TestCaptionEnumerationMatchesTraining:
    """The cases the first version of enumerate_captions got wrong.

    A mismatch here means distillation trains on captions the diffusion stages never see, which
    is the exact drift this helper exists to prevent.
    """

    @staticmethod
    def enumerate_captions(*args, **kwargs):
        dataset = pytest.importorskip('utils.dataset')
        return dataset.enumerate_captions(*args, **kwargs)

    @staticmethod
    def write(tmp_path, files):
        import json
        for name, content in files.items():
            path = tmp_path / name
            path.write_text(json.dumps(content) if name.endswith('.json') else content)
        return {'directory': [{'path': str(tmp_path)}]}

    def test_captions_json_disables_the_txt_fallback(self, tmp_path):
        """DirectoryDataset kills the .txt fallback for the WHOLE directory once a
        captions.json exists (`if has_captions_json or not os.path.exists(...)`), so an image
        missing from the json is dropped rather than falling back to its .txt."""
        config = self.write(tmp_path, {
            'a.jpg': 'x', 'b.jpg': 'x',
            'b.txt': 'this must NOT be picked up',
            'captions.json': {'a.jpg': ['from json']},
        })
        assert self.enumerate_captions(config) == ['from json']

    def test_tar_members_are_keyed_by_full_path(self, tmp_path):
        """add_captions() only takes the basename when tar_file is None; a tar member is
        looked up by its full path inside the archive."""
        import tarfile
        member_dir = tmp_path / 'src'
        member_dir.mkdir()
        for name in ('a.jpg', 'b.jpg'):
            (member_dir / name).write_text('x')
        with tarfile.TarFile(tmp_path / 'shard.tar', 'w') as tar:
            for name in ('a.jpg', 'b.jpg'):
                tar.add(member_dir / name, arcname=f'sub/{name}')
        for name in ('a.jpg', 'b.jpg'):
            (member_dir / name).unlink()
        member_dir.rmdir()

        config = self.write(tmp_path, {'captions.json': {'sub/a.jpg': ['tar A'], 'sub/b.jpg': ['tar B']}})
        assert sorted(self.enumerate_captions(config)) == ['tar A', 'tar B']

    def test_missing_json_entry_is_skipped_not_guessed(self, tmp_path):
        config = self.write(tmp_path, {
            'a.jpg': 'x', 'b.jpg': 'x',
            'captions.json': {'a.jpg': ['only this one']},
        })
        assert self.enumerate_captions(config) == ['only this one']

    def test_multiple_captions_per_image(self, tmp_path):
        config = self.write(tmp_path, {
            'a.jpg': 'x',
            'captions.json': {'a.jpg': ['first', 'second', 'third']},
        })
        assert self.enumerate_captions(config) == ['first', 'second', 'third']

    def test_fractional_num_repeats(self, tmp_path):
        """SizeBucketDataset accepts any num_repeats > 0 and takes int(len * num_repeats)."""
        config = self.write(tmp_path, {f'{i}.jpg': 'x' for i in range(4)})
        for i in range(4):
            (tmp_path / f'{i}.txt').write_text(f'caption {i}')
        config['directory'][0]['num_repeats'] = 0.5
        assert len(self.enumerate_captions(config, apply_num_repeats=True)) == 2
        config['directory'][0]['num_repeats'] = 2
        assert len(self.enumerate_captions(config, apply_num_repeats=True)) == 8

    def test_custom_shuffle_delimiter(self, tmp_path):
        config = self.write(tmp_path, {'a.jpg': 'x', 'a.txt': 'one;two;three'})
        config['cache_shuffle_num'] = 3
        config['cache_shuffle_delimiter'] = ';'
        captions = self.enumerate_captions(config)
        assert len(captions) == 3
        for caption in captions:
            assert sorted(caption.split(';')) == ['one', 'three', 'two']
