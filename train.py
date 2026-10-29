import argparse
import os
import wandb
from datetime import datetime, timezone
import shutil
import time
import random
import json
import inspect
from pathlib import Path

import toml
import deepspeed
from deepspeed import comm as dist
from deepspeed.runtime.pipe import module as ds_pipe_module
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import multiprocess as mp
import numpy as np

from utils import dataset as dataset_util
from utils import common
from utils.common import is_main_process, get_rank, DTYPE_MAP, empty_cuda_cache, all_ok
import utils.saver
from utils.isolate_rng import isolate_rng
from utils.text_encoding import TextEncodingOnTheFly, encodes_text_in_loader
from utils.patches import apply_patches
from utils.unsloth_utils import unsloth_checkpoint
from utils.pipeline import ManualPipelineModule
from utils.dora import apply_dora_config_defaults, check_model_dtypes as check_dora_model_dtypes
from utils.oplora import OPLoRAProjector, apply_oplora_config_defaults
from utils.lr_schedule import create_lr_scheduler
from utils.checkpoint_layout import check_checkpoint_layers, shift_checkpoint_layers
from utils.param_groups import check_resumed_param_groups, group_names, split_2d_param_groups
from utils.ema import ParameterEMA, apply_ema_config_defaults, load_ema_checkpoint, load_ema_weights
from utils.optimizer_factory import resolve_optimizer_class
from utils.step_stats import LossAverage, StepStats
from utils.pipeline_optim import (DummyOptimizer, check_pipeline_optimizer, check_prodigy_learning_rates,
                                  pipeline_collective, sync_prodigy)

# needed for broadcasting Queue in dataset.py
mp.current_process().authkey = b'afsaskgfdjh4'

wandb_enable = False

TIMESTEP_QUANTILES_FOR_EVAL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--resume_from_checkpoint', nargs='?', const=True, default=None,
                    help='resume training from checkpoint. If no value is provided, resume from the most recent checkpoint. If a folder name is provided, resume from that specific folder.')
parser.add_argument('--reset_dataloader', action='store_true', help='Start dataloader from scratch when resuming from checkpoint, i.e. only load the optimizer states.')
parser.add_argument('--reset_optimizer', action='store_true')
parser.add_argument('--reset_optimizer_params', action='store_true')
parser.add_argument('--regenerate_cache', action='store_true', help='Force regenerate cache.')
parser.add_argument('--cache_only', action='store_true', help='Cache model inputs then exit.')
parser.add_argument('--metadata_only', action='store_true', help='Scan the datasets, group them into buckets and cache that metadata, print the files per bucket, then exit. No latents or text embeddings: neither the VAE nor a text encoder runs.')
parser.add_argument('--cache_status', action='store_true', help='Print how many files of each dataset directory (and bucket) have their latents and text embeddings cached, then exit. Reads what is cached, also from a run that was interrupted or is still caching (rows it has not committed yet are not counted); encodes and writes no latents or text embeddings. Combine with --trust_cache to skip the metadata scan, e.g. while another run is caching.')
parser.add_argument('--trust_cache', action='store_true', help='Trust source metadata in the legacy cache workflow. Snapshot mode reuses sources automatically and always applies current caption configuration profiles.')
parser.add_argument('--nan_check', action='store_true', help='Stop at the first NaN/inf in a layer output, gradient, loss or weight, and report the layer, step and images. Slow; for debugging.')
parser.add_argument('--i_know_what_i_am_doing', action='store_true', help="Skip certain checks and overrides. You may end up using settings that won't work.")
parser.add_argument('--master_port', type=int, default=None,
                    help="Master port for distributed training (default: the launcher's MASTER_PORT, else 29500)")
parser.add_argument('--dump_dataset', type=Path, default=None, help='Decode cached latents and dump the dataset to this directory.')
parser.add_argument('--test_sample', action='store_true', help='Generate and write an image to example.png and then quit.')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


# Monkeypatch this so it counts all layer parameters, not just trainable parameters.
# This helps it divide the layers between GPUs more evenly when training a LoRA.
def _count_all_layer_params(self):
    param_counts = [0] * len(self._layer_specs)
    for idx, layer in enumerate(self._layer_specs):
        if isinstance(layer, ds_pipe_module.LayerSpec):
            l = layer.build()
            param_counts[idx] = sum(p.numel() for p in l.parameters())
        elif isinstance(layer, nn.Module):
            param_counts[idx] = sum(p.numel() for p in layer.parameters())
    return param_counts
ds_pipe_module.PipelineModule._count_layer_params = _count_all_layer_params


def set_config_defaults(config):
    # Force the user to set this. If we made it a default of 1, it might use a lot of disk space.
    assert 'save_every_n_epochs' in config or 'save_every_n_steps' in config or 'save_every_n_examples' in config

    config.setdefault('pipeline_stages', 1)
    if 'optimizer' in config:
        check_pipeline_optimizer(config['optimizer']['type'], config['pipeline_stages'])
    config.setdefault('activation_checkpointing', False)
    config.setdefault('reentrant_activation_checkpointing', False)
    if config['activation_checkpointing'] == 'unsloth':
        config['reentrant_activation_checkpointing'] = True
    config.setdefault('warmup_steps', 0)
    if 'keep_last_n_checkpoints' in config:
        # Validated here for the same reason tools/distill_refiner.py validates it: unset means
        # keep everything, but 0 or a negative number reads as "keep none" and is silently
        # treated as unset by prune_all_checkpoint_kinds. The two trainers share the option, so
        # they should agree on what a typo does.
        keep = config['keep_last_n_checkpoints']
        if not isinstance(keep, int) or isinstance(keep, bool) or keep < 2:
            raise ValueError(
                f'keep_last_n_checkpoints must be an integer >= 2, got {keep!r}: with 1, a torn newest '
                'checkpoint leaves nothing to resume from. Remove the key to keep every checkpoint.'
            )
    if 'save_dtype' in config:
        config['save_dtype'] = DTYPE_MAP[config['save_dtype']]

    model_config = config['model']
    model_dtype_str = model_config['dtype']
    model_config['dtype'] = DTYPE_MAP[model_dtype_str]
    if transformer_dtype := model_config.get('transformer_dtype', None):
        model_config['transformer_dtype'] = DTYPE_MAP[transformer_dtype]
    if diffusion_model_dtype := model_config.get('diffusion_model_dtype', None):
        model_config['diffusion_model_dtype'] = DTYPE_MAP[diffusion_model_dtype]
    model_config.setdefault('guidance', 1.0)

    if 'adapter' in config:
        adapter_config = config['adapter']
        adapter_type = adapter_config['type']
        if 'alpha' in adapter_config:
            raise NotImplementedError(
                'This script forces alpha=rank to make the saved adapter format simpler and more predictable with downstream inference programs. Please remove alpha from the config.'
            )
        adapter_config['alpha'] = adapter_config['rank']
        adapter_config.setdefault('dtype', model_dtype_str)
        adapter_config['dtype'] = DTYPE_MAP[adapter_config['dtype']]

        apply_dora_config_defaults(adapter_config)
        check_dora_model_dtypes(adapter_config, model_config)
        # per-adapter defaults
        if adapter_config['type'] == 'lora':
            adapter_config.setdefault('dropout', 0.0)
        elif adapter_config['type'] == 'lokr':
            adapter_config.setdefault('decompose_factor', -1)
            adapter_config.setdefault('rank_dropout', 0.0)
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')

        # OPLoRA projects each LoRA update onto the orthogonal complement of the base
        # weight's top-k singular subspace, so the base model's dominant directions are
        # preserved. Validation and defaults live in utils.oplora so they can be
        # unit-tested without importing the training stack.
        apply_oplora_config_defaults(adapter_config)

    # EMA of the trained weights (utils/ema.py): validated here so a typo fails before any
    # model is loaded.
    apply_ema_config_defaults(config)

    config.setdefault('logging_steps', 1)
    # The size of each step's change to the weights (utils/step_stats.py): on by default for an
    # adapter. It keeps a copy of the trained weights through the step, which for a full fine-tune
    # is as large as they are.
    config.setdefault('log_update_norm', 'adapter' in config)
    if not isinstance(config['log_update_norm'], bool):
        raise ValueError(f"log_update_norm must be true or false, got {config['log_update_norm']!r}")
    # train/loss_avg: the loss averaged over this many steps, which shows the trend one step's
    # loss hides.
    config.setdefault('loss_average_steps', 100)
    if type(config['loss_average_steps']) is not int or config['loss_average_steps'] < 1:
        raise ValueError(f"loss_average_steps must be a whole number of steps, at least 1, got {config['loss_average_steps']!r}")
    # Several nodes share dataset caching out by bucket, each node on its own GPUs (utils/dataset.py
    # DatasetManager.cache). Off, global rank 0's node caches everything while the others wait.
    config.setdefault('multi_node_caching', True)
    if not isinstance(config['multi_node_caching'], bool):
        raise ValueError(f"multi_node_caching must be true or false, got {config['multi_node_caching']!r}")
    config.setdefault('eval_datasets', [])
    config.setdefault('eval_gradient_accumulation_steps', 1)
    config.setdefault('eval_every_n_steps', None)
    config.setdefault('eval_every_n_epochs', None)
    config.setdefault('eval_every_n_examples', None)
    config.setdefault('eval_before_first_step', True)
    config.setdefault('compile', False)
    config.setdefault('x_axis_examples', False)


def print_model_info(model):
    if not is_main_process():
        return
    print(model)
    for name, module in model.named_modules():
        print(f'{type(module)}: {name}')
        for pname, p in module.named_parameters(recurse=False):
            print(pname)
            print(p.dtype)
            print(p.device)
            print(p.requires_grad)
            print()


# Need to preload all micro batches since pulling from the dataloader does IPC between the
# first and last stage. Can't do that during the train or inference pipeline schedule execution
# because it conflicts with the send / recv steps.
def get_data_iterator_for_step(dataloader, engine, num_micro_batches=None):
    num_micro_batches = num_micro_batches or engine.micro_batches
    if not (engine.is_first_stage() or engine.is_last_stage()):
        return None
    dataloader_iter = iter(dataloader)
    items = [next(dataloader_iter) for _ in range(num_micro_batches)]
    return iter(items)


def evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=None):
    eval_dataloader.set_eval_quantile(quantile)
    total_loss = 0
    count = 0
    while True:
        model_engine.reset_activation_shape()
        iterator = get_data_iterator_for_step(eval_dataloader, model_engine, num_micro_batches=eval_gradient_accumulation_steps)
        loss = model_engine.eval_batch(iterator, num_micro_batches=eval_gradient_accumulation_steps).item()
        eval_dataloader.sync_epoch()
        if pbar:
            pbar.update(1)
        total_loss += loss
        count += 1
        if eval_dataloader.epoch == 2:
            break

    eval_dataloader.reset()
    return total_loss / count


def _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps):
    pbar_total = 0
    for eval_dataloader in eval_dataloaders.values():
        pbar_total += len(eval_dataloader) * len(TIMESTEP_QUANTILES_FOR_EVAL) // eval_gradient_accumulation_steps
    if is_main_process():
        print('Running eval')
        pbar = tqdm(total=pbar_total)
    else:
        pbar = None

    start = time.time()
    for name, eval_dataloader in eval_dataloaders.items():
        losses = []
        for quantile in TIMESTEP_QUANTILES_FOR_EVAL:
            loss = evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=pbar)
            losses.append(loss)
            if is_main_process():
                tb_writer.add_scalar(f'{name}/loss_quantile_{quantile:.2f}', loss, step)
                if wandb_enable:
                    wandb.log({f'{name}/loss_quantile_{quantile:.2f}': loss, 'step': step})
        avg_loss = sum(losses) / len(losses)
        if is_main_process():
            tb_writer.add_scalar(f'{name}/loss', avg_loss, step)
            if wandb_enable:
                wandb.log({f'{name}/loss': avg_loss, 'step': step})

    duration = time.time() - start
    if is_main_process():
        tb_writer.add_scalar('eval/eval_time_sec', duration, step)
        if wandb_enable:
            wandb.log({'eval/eval_time_sec': duration, 'step': step})
        pbar.close()


def evaluate(model, model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps, disable_block_swap):
    if len(eval_dataloaders) == 0:
        return
    empty_cuda_cache()
    model.prepare_block_swap_inference(disable_block_swap=disable_block_swap)
    with torch.no_grad(), isolate_rng():
        seed = get_rank()
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps)
    empty_cuda_cache()
    model.prepare_block_swap_training()


def get_prodigy_d(optimizer):
    d = 0
    for group in optimizer.param_groups:
        d += group['d']
    return d / len(optimizer.param_groups)


def _get_automagic_lrs(optimizer):
    lrs = []
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            lr = optimizer._get_lr(group, state)
            lrs.append(lr)
    lrs = torch.stack(lrs)
    return lrs, lrs.mean()


if __name__ == '__main__':
    # With multiple GPUs / large batch sizes, the dataloader can trigger "too many open files" errors unless we do this.
    torch.multiprocessing.set_sharing_strategy('file_system')
    deepspeed.utils.set_log_level_from_string('info')
    apply_patches()

    with open(args.config) as f:
        # Inline TOML tables are not pickleable, which messes up the multiprocessing dataset stuff. This is a workaround.
        config = json.loads(json.dumps(toml.load(f)))

    if args.cache_status:
        conflicting = [flag for flag, given in (
            ('--metadata_only', args.metadata_only), ('--cache_only', args.cache_only),
            ('--dump_dataset', args.dump_dataset is not None), ('--test_sample', args.test_sample),
            ('--regenerate_cache', args.regenerate_cache),
            ('--resume_from_checkpoint', args.resume_from_checkpoint is not None),
            ('a [turbo] config', 'turbo' in config)) if given]
        if conflicting:
            raise SystemExit(f'--cache_status only reads the caches; it does not go with {", ".join(conflicting)}')

    if args.metadata_only:
        conflicting = [flag for flag, given in (
            ('--cache_only', args.cache_only), ('--dump_dataset', args.dump_dataset is not None),
            ('--test_sample', args.test_sample), ('--resume_from_checkpoint', args.resume_from_checkpoint is not None),
            ('a [turbo] config', 'turbo' in config)) if given]
        if conflicting:
            raise SystemExit(f'--metadata_only only caches dataset metadata; it does not go with {", ".join(conflicting)}')

    if 'turbo' in config:
        # Iris-3B Turbo: few-step distillation with its own trainers, not this training loop.
        from utils import iris_turbo
        try:
            iris_turbo.run(config)
        except iris_turbo.TurboConfigError as error:
            raise SystemExit(f'[iris_turbo] {error}')
        raise SystemExit(0)

    set_config_defaults(config)
    # Fail now rather than after the dataset and model have loaded.
    common.ensure_writable_dir(config['output_dir'], 'output_dir')
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE') or os.environ.get('LOCAL_SIZE')
                           or max(1, torch.cuda.device_count()))
    if (cpu_threads := common.limit_cpu_threads(local_world_size)) is not None and \
            int(os.environ.get('LOCAL_RANK', max(args.local_rank, 0))) == 0:
        print(f'CPU threads per GPU process: {cpu_threads} (set OMP_NUM_THREADS to override)')
    common.AUTOCAST_DTYPE = config['model']['dtype']
    dataset_util.UNCOND_FRACTION = config.get('uncond_fraction', 0.0)
    if map_num_proc := config.get('map_num_proc', None):
        dataset_util.NUM_PROC = map_num_proc
    if 'metadata_read_threads' in config:
        threads = config['metadata_read_threads']
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError(f'metadata_read_threads must be a whole number of at least 1, got {threads!r}')
        dataset_util.METADATA_READ_THREADS = threads

    common.set_rendezvous_env(args.master_port)
    deepspeed.init_distributed()
    # After init_distributed, which sets LOCAL_RANK itself when it discovers an MPI launch.
    torch.cuda.set_device(common.resolve_local_rank(args.local_rank))

    resume_from_checkpoint = (
        args.resume_from_checkpoint if args.resume_from_checkpoint is not None
        else config.get('resume_from_checkpoint', False)
    )
    regenerate_cache = (
        args.regenerate_cache if args.regenerate_cache is not None
        else config.get('regenerate_cache', False)
    )

    model_type = config['model']['type']

    if model_type == 'flux':
        from models import flux
        model = flux.FluxPipeline(config)
    elif model_type == 'ltx-video':
        from models import ltx_video
        model = ltx_video.LTXVideoPipeline(config)
    elif model_type == 'hunyuan-video':
        from models import hunyuan_video
        model = hunyuan_video.HunyuanVideoPipeline(config)
    elif model_type == 'sdxl':
        from models import sdxl
        model = sdxl.SDXLPipeline(config)
    elif model_type == 'cosmos':
        from models import cosmos
        model = cosmos.CosmosPipeline(config)
    elif model_type == 'lumina_2':
        from models import lumina_2
        model = lumina_2.Lumina2Pipeline(config)
    elif model_type == 'wan':
        from models.wan import wan
        model = wan.WanPipeline(config)
    elif model_type == 'chroma':
        from models import chroma
        model = chroma.ChromaPipeline(config)
    elif model_type == 'hidream':
        from models import hidream
        model = hidream.HiDreamPipeline(config)
    elif model_type == 'sd3':
        from models import sd3
        model = sd3.SD3Pipeline(config)
    elif model_type == 'cosmos_predict2' or model_type == 'anima' or model_type == 'anima_refiner':
        from models import cosmos_predict2
        model = cosmos_predict2.CosmosPredict2Pipeline(config)
    elif model_type == 'omnigen2':
        from models import omnigen2
        model = omnigen2.OmniGen2Pipeline(config)
    elif model_type == 'qwen_image':
        from models import qwen_image
        model = qwen_image.QwenImagePipeline(config)
    elif model_type == 'hunyuan_image':
        from models import hunyuan_image
        model = hunyuan_image.HunyuanImagePipeline(config)
    elif model_type == 'auraflow':
        from models import auraflow
        model = auraflow.AuraFlowPipeline(config)
    elif model_type == 'z_image':
        from models import z_image
        model = z_image.ZImagePipeline(config)
    elif model_type == 'hunyuan_video_15':
        from models import hunyuan_video_15
        model = hunyuan_video_15.HunyuanVideo15Pipeline(config)
    elif model_type == 'flux2':
        from models import flux2
        model = flux2.Flux2Pipeline(config)
    elif model_type == 'ernie_image':
        from models import ernie_image
        model = ernie_image.ErnieImagePipeline(config)
    elif model_type == 'ltx2':
        from models import ltx2
        model = ltx2.LTX2Pipeline(config)
    elif model_type == 'ideogram4':
        from models import ideogram4
        model = ideogram4.Ideogram4Pipeline(config)
    elif model_type == 'krea2':
        from models import krea2
        model = krea2.Krea2Pipeline(config)
    elif model_type == 'minimax_h3':
        from models import minimax_h3
        model = minimax_h3.MinimaxH3Pipeline(config)
    elif model_type == 'qwen_image21':
        from models import qwen_image21
        model = qwen_image21.QwenImage21Pipeline(config)
    elif model_type == 'iris':
        from models import iris
        model = iris.IrisPipeline(config)
    elif model_type == 'moss_image':
        from models import moss_image
        model = moss_image.MossImagePipeline(config)
    else:
        raise NotImplementedError(f'Model type {model_type} is not implemented')

    # import sys, PIL
    # test_image = sys.argv[1]
    # with torch.no_grad():
    #     vae = model.get_vae().to('cuda')
    #     latents = dataset.encode_pil_to_latents(PIL.Image.open(test_image), vae)
    #     pil_image = dataset.decode_latents_to_pil(latents, vae)
    #     pil_image.save('test.jpg')
    # quit()

    with open(config['dataset']) as f:
        dataset_config = toml.load(f)

    micro_batch_size_per_gpu = config.get('micro_batch_size_per_gpu', 1)
    if isinstance(micro_batch_size_per_gpu, int):
        micro_batch_size_per_gpu = {None: micro_batch_size_per_gpu}
    elif isinstance(micro_batch_size_per_gpu, list):
        micro_batch_size_per_gpu = {x[0]: x[1] for x in micro_batch_size_per_gpu}

    eval_micro_batch_size_per_gpu = config.get('eval_micro_batch_size_per_gpu', micro_batch_size_per_gpu)
    if isinstance(eval_micro_batch_size_per_gpu, int):
        eval_micro_batch_size_per_gpu = {None: eval_micro_batch_size_per_gpu}
    elif isinstance(eval_micro_batch_size_per_gpu, list):
        eval_micro_batch_size_per_gpu = {x[0]: x[1] for x in eval_micro_batch_size_per_gpu}

    image_micro_batch_size_per_gpu = config.get('image_micro_batch_size_per_gpu', micro_batch_size_per_gpu)
    if isinstance(image_micro_batch_size_per_gpu, int):
        image_micro_batch_size_per_gpu = {None: image_micro_batch_size_per_gpu}
    elif isinstance(image_micro_batch_size_per_gpu, list):
        image_micro_batch_size_per_gpu = {x[0]: x[1] for x in image_micro_batch_size_per_gpu}

    eval_image_micro_batch_size_per_gpu = config.get('eval_image_micro_batch_size_per_gpu', eval_micro_batch_size_per_gpu)
    if isinstance(eval_image_micro_batch_size_per_gpu, int):
        eval_image_micro_batch_size_per_gpu = {None: eval_image_micro_batch_size_per_gpu}
    elif isinstance(eval_image_micro_batch_size_per_gpu, list):
        eval_image_micro_batch_size_per_gpu = {x[0]: x[1] for x in eval_image_micro_batch_size_per_gpu}

    default_micro_batch_size_per_gpu = list(micro_batch_size_per_gpu.values())[0]

    gradient_release = config['optimizer'].get('gradient_release', False)
    ds_config = {
        'train_micro_batch_size_per_gpu': default_micro_batch_size_per_gpu,
        'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 1),
        # Can't do gradient clipping with gradient release, since there are no grads at the end of the step anymore.
        'gradient_clipping': 0. if gradient_release else config.get('gradient_clipping', 1.0),
        'steps_per_print': config.get('steps_per_print', 1),
    }
    caching_batch_size = config.get('caching_batch_size', 1)
    dataset_manager = dataset_util.DatasetManager(model, regenerate_cache=regenerate_cache, trust_cache=args.trust_cache, caching_batch_size=caching_batch_size, keep_models_loaded=args.test_sample,
                                                  multi_node=config['multi_node_caching'])
    # cache_text_embeddings = false on a model that does not encode its captions itself: its text
    # encoders encode each batch's captions in the training process instead of filling a cache.
    text_encoding = (TextEncodingOnTheFly(model, caching_batch_size=caching_batch_size)
                     if encodes_text_in_loader(model) else None)

    # The batch fill keys belong in the dataset config, but several training configs share one
    # dataset TOML, so the same keys are accepted here and win. Applied to the eval datasets
    # too: a run that fills its training batches and drops its eval ones would report a metric
    # over a different subset than the one it trains on.
    batch_fill_overrides = {key: config[key] for key in dataset_util.BATCH_FILL_DEFAULTS
                            if key in config}
    train_data = dataset_util.Dataset(dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing,
                                      batch_fill_overrides=batch_fill_overrides)
    dataset_manager.register(train_data)

    eval_data_map = {}
    for i, eval_dataset in enumerate(config['eval_datasets']):
        if type(eval_dataset) == str:
            name = f'eval{i}'
            config_path = eval_dataset
        else:
            name = eval_dataset['name']
            config_path = eval_dataset['config']
        with open(config_path) as f:
            eval_dataset_config = toml.load(f)
        eval_data_map[name] = dataset_util.Dataset(
            eval_dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing,
            # Eval should be the same set every time it runs, so a figure from epoch 7 is
            # comparable with one from epoch 3. Rotating the tail would move part of the eval
            # set between runs for reasons that have nothing to do with the model.
            batch_fill_defaults={'fill_rotate_per_epoch': False},
            batch_fill_overrides=batch_fill_overrides)
        dataset_manager.register(eval_data_map[name])

    # For testing

    # import imageio
    # from pathlib import Path
    # import torch.nn.functional as F
    # dataset_manager.cache(unload_models=False)
    # output_dir = Path('/home/anon/tmp')
    # train_data.post_init(
    #     0,
    #     1,
    #     1,
    #     1,
    # )
    # vae = model.vae
    # vae.model.to('cuda')
    # count = 1
    # for item in train_data:
    #     latents = item['latents'].to('cuda')
    #     h, w = latents.shape[-2:]
    #     mask = item['mask'].to('cuda')
    #     caption = item['caption'][0]
    #     mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
    #     mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
    #     mask = mask.unsqueeze(2)  # make mask same number of dims as target
    #     latents = latents * mask.to(latents.device)
    #     video = vae.model.decode(latents, vae.scale).float().clamp_(-1, 1).squeeze(0)
    #     video = torch.permute(video, (1, 2, 3, 0))
    #     video = (video + 1) / 2
    #     video = (video * 255).type(torch.uint8).cpu()
    #     imageio.v3.imwrite(output_dir / f'{count}.mp4', video, fps=16)
    #     with open(output_dir / f'{count}.txt', 'w') as f:
    #         f.write(caption)
    #     if count >= 10:
    #         break
    #     count += 1
    # quit()

    if args.cache_status:
        # The metadata, as every run loads it (--trust_cache skips the scan); then the caches are read.
        dataset_manager.cache(metadata_only=True)
        error = None
        if is_main_process():
            try:
                for name, data in [('train', train_data)] + list(eval_data_map.items()):
                    print(f'Cache status of the {name} dataset:')
                    for line in dataset_util.describe_cache_status(
                            dataset_util.cache_status(data, len(dataset_manager.text_encoders)),
                            len(dataset_manager.text_encoders)):
                        print(line)
            except Exception as e:
                error = e
        all_ok(error)
        dist.barrier()
        quit()

    if args.metadata_only:
        dataset_manager.cache(metadata_only=True)
        if is_main_process():
            for name, data in [('train', train_data)] + list(eval_data_map.items()):
                print(f'Metadata of the {name} dataset:')
                for line in dataset_util.describe_buckets(data):
                    print(line)
        dist.barrier()
        quit()

    if args.dump_dataset:
        # only works for flux
        import torchvision
        dataset_manager.cache(unload_models=False)
        if is_main_process():
            with torch.no_grad():
                os.makedirs(args.dump_dataset, exist_ok=True)
                vae = model.vae.to('cuda')
                train_data.post_init(
                    0,
                    1,
                    1,
                    1,
                    1,
                )
                for i, item in enumerate(train_data):
                    latents = item['latents']
                    latents = latents / vae.config.scaling_factor
                    if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                        latents = latents + vae.config.shift_factor
                    img = vae.decode(latents.to(vae.device, vae.dtype)).sample.to(torch.float32)
                    img = img.squeeze(0)
                    img = ((img + 1) / 2).clamp(0, 1)
                    pil_img = torchvision.transforms.functional.to_pil_image(img)
                    pil_img.save(args.dump_dataset / f'{i}.png')
                    if i >= 100:
                        break
        dist.barrier()
        quit()

    dataset_manager.cache()
    if args.cache_only:
        quit()

    # Free up as much RAM as we can.
    del dataset_manager
    if (config['model'].get('cache_text_embeddings', True) or text_encoding is not None) and not args.test_sample:
        # Only ComfyUI-based models, and only if we are caching text embeddings (which most models require).
        # text_encoding holds its own references to the text encoders, so only the VAE goes then.
        model.free_vae_and_te()

    if args.test_sample:
        model.prepare_sample_test('a golden retriever running through a grassy field', cfg=1)

    model.load_diffusion_model()

    if adapter_config := config.get('adapter', None):
        model.configure_adapter(adapter_config)
        is_adapter = True
        if init_from_existing := adapter_config.get('init_from_existing', None):
            model.load_adapter_weights(init_from_existing)
    else:
        is_adapter = False

    # Determine run_dir (and the checkpoint to resume) on rank 0 and broadcast it, or its error
    run_dir_container = [None]
    if is_main_process():
        try:
            if resume_from_checkpoint:
                run_dir_container[0] = (*common.resume_run_dir(config['output_dir'], resume_from_checkpoint), None)
            else:
                run_dir = os.path.join(config['output_dir'], datetime.now(timezone.utc).strftime('%Y%m%d_%H-%M-%S'))
                run_dir_container[0] = (run_dir, None, None)
        except Exception as e:
            run_dir_container[0] = (None, None, f'{type(e).__name__}: {e}')

    torch.distributed.broadcast_object_list(run_dir_container, src=0, group=dist.get_world_group())
    run_dir, resume_tag, error = run_dir_container[0]
    if error is not None:
        raise RuntimeError(error)

    error = None
    if not resume_from_checkpoint and is_main_process():
        try:
            os.makedirs(run_dir, exist_ok=True)
            shutil.copy(args.config, run_dir)
            shutil.copy(config['dataset'], run_dir)
            for eval_dataset in config['eval_datasets']:
                # Same two forms the loader accepts: a path, or {name, config}.
                shutil.copy(eval_dataset if isinstance(eval_dataset, str) else eval_dataset['config'], run_dir)
        except Exception as e:
            error = e
    all_ok(error)
    # Every rank writes checkpoints into run_dir and the pipeline stages pass saved weights through
    # it, so every node must see what rank 0 wrote there. Found now rather than at the first save.
    error = None
    try:
        common.wait_until_visible(
            Path(run_dir) / ('latest' if resume_from_checkpoint else os.path.basename(args.config)),
            'output_dir must be on a volume that every node shares')
    except Exception as e:
        error = e
    all_ok(error)

    # WandB logging
    wandb_enable = config.get('monitoring', {}).get('enable_wandb', False)
    if wandb_enable and is_main_process():
        wandb_api_key     = config['monitoring']['wandb_api_key']
        wandb_tracker     = config['monitoring']['wandb_tracker_name']
        wandb_run_name    = config['monitoring']['wandb_run_name']
        logging_dir       = run_dir
        wandb.login(key=wandb_api_key)
        wandb.init(
            project=wandb_tracker,
            name=wandb_run_name,
            config=config,
            dir=logging_dir
        )

    # Block swapping
    if blocks_to_swap := config.get('blocks_to_swap', 0):
        assert config['pipeline_stages'] == 1, 'Block swapping only works with pipeline_stages=1'
        assert 'adapter' in config, 'Block swapping only works when training LoRA'
        assert (config['model'].get('cache_text_embeddings', True) or text_encoding is not None
                or getattr(model, 'block_swap_supports_uncached_text_embeddings', False)), (
            'blocks_to_swap cannot be combined with cache_text_embeddings = false for this model. Block '
            'swapping replaces PipelineModule.to with a no-op and moves only the transformer to '
            'CUDA, so a resident text encoder -- which lives on the pipeline layer, not the '
            'transformer -- silently stays on CPU and returns CPU embeddings into a CUDA batch. '
            'Use one or the other.'
        )
        # Don't automatically move to GPU, we'll do that ourselves.
        def to(self, *args, **kwargs):
            pass
        deepspeed.pipe.PipelineModule.to = to
        model.enable_block_swap(blocks_to_swap)

    layers = model.to_layers()
    additional_pipeline_module_kwargs = {}
    activation_checkpointing = config['activation_checkpointing']
    if activation_checkpointing:
        if activation_checkpointing == True:
            # TODO: block swapping doesn't work with Deepspeed non-reentrant checkpoint, but PyTorch native one is fine. Some
            # weights end up on CPU where they shouldn't. Why? Are we giving anything up by not using the Deepspeed implementation?
            #checkpoint_func = deepspeed.checkpointing.non_reentrant_checkpoint
            from functools import partial
            checkpoint_func = partial(torch.utils.checkpoint.checkpoint, use_reentrant=config['reentrant_activation_checkpointing'])
        elif activation_checkpointing == 'unsloth':
            checkpoint_func = unsloth_checkpoint
        else:
            raise NotImplementedError(f'activation_checkpointing={activation_checkpointing} is not implemented')
        additional_pipeline_module_kwargs.update({
            'activation_checkpoint_interval': 1,
            'checkpointable_layers': model.checkpointable_layers,
            'activation_checkpoint_func': checkpoint_func,
        })

    num_stages = config.get('pipeline_stages', 1)
    partition_method=config.get('partition_method', 'parameters')
    partition_split = config.get('partition_split',[len(layers) / num_stages])
    pipeline_model = ManualPipelineModule(
        layers=layers,
        num_stages=num_stages,
        partition_method=partition_method,
        manual_partition_split=partition_split,
        loss_fn=model.get_loss_fn(),
        dynamic_shape=True,
        **additional_pipeline_module_kwargs
    )
    model.pipeline_model = pipeline_model
    nan_check = None
    if args.nan_check:
        from utils.nan_check import NanCheck
        nan_check = NanCheck(pipeline_model, model, get_rank())
        if is_main_process():
            print('--nan_check: every layer output, gradient and weight is checked (slow)')
    parameters_to_train = [p for p in pipeline_model.parameters() if p.requires_grad]

    if config['compile']:
        pipeline_model.compile(dynamic=True)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=pipeline_model,
        config=ds_config,
    )
    # Newer Deepspeed versions fail when pipeline_stages>1 because of a check on this field which defaults to False. But, pipeline
    # parallelism has always relied on "Torch-style" backward(), so I think this is an oversight by Deepspeed devs and it's safe
    # to force this to True to get it to work.
    model_engine._support_torch_style_backward = True
    global_batch_size = model_engine.train_micro_batch_size_per_gpu() * model_engine.gradient_accumulation_steps() * model_engine.grid.get_data_parallel_world_size()
    print(f'Global batch size = {global_batch_size}')

    if args.test_sample:
        import torchvision
        img = model.sample(w=512, h=512)
        img = img.squeeze(0).movedim(-1, 0)
        print(img.shape, img.min().item(), img.max().item())
        torchvision.utils.save_image(img, 'example.png')
        quit()

    if save_every_n_examples := config.pop('save_every_n_examples', None):
        config['save_every_n_steps'] = save_every_n_examples // global_batch_size
        print(f"Computed save_every_n_steps = {config['save_every_n_steps']}")
    if eval_every_n_examples := config.pop('eval_every_n_examples', None):
        config['eval_every_n_steps'] = eval_every_n_examples // global_batch_size
        print(f"Computed eval_every_n_steps = {config['eval_every_n_steps']}")

    def get_optimizer(model_parameters):
        optim_config = config['optimizer']
        optim_type = optim_config['type']
        optim_type_lower = optim_type.lower()
        collective = pipeline_collective(optim_type_lower, optim_config.get('gradient_release', False), pipeline_model.num_stages)
        pipeline_group = pipeline_model.mpu().get_model_parallel_group()

        if len(model_parameters) == 0:
            return DummyOptimizer(collective, pipeline_group)

        if beta2_half_life := optim_config.pop('beta2_half_life', None):
            betas = optim_config['betas']
            assert len(betas) == 2
            betas[1] = 0.5 ** (global_batch_size / beta2_half_life)
            print(f'Computed beta2 = {betas[1]}')
            optim_config['betas'] = betas

        # The name-to-class mapping lives in utils/optimizer_factory so tools/distill_refiner.py
        # can accept the same optimizer names. Everything below here is pipeline-specific and
        # stays put.
        klass, args, kwargs = resolve_optimizer_class(optim_config)

        if optim_config.get('gradient_release', False):
            # Prevent deepspeed from logging every single param group lr
            def _report_progress(self, step):
                lr = self.get_lr()
                mom = self.get_mom()
                deepspeed.utils.logging.log_dist(f"step={step}, skipped={self.skipped_steps}, lr={lr[0]}, mom={mom[0]}", ranks=[0])
            deepspeed.runtime.engine.DeepSpeedEngine._report_progress = _report_progress

            # Deepspeed executes all the code to reduce grads across data parallel ranks even if the DP world size is 1.
            # As part of this, any grads that are None are set to zeros. We're doing gradient release to save memory,
            # so we have to avoid this.
            def _exec_reduce_grads(self):
                assert self.mpu.get_data_parallel_world_size() == 1, 'When using gradient release, data parallel world size must be 1. Make sure pipeline_stages = num_gpus.'
                return
            deepspeed.runtime.pipe.engine.PipelineEngine._INSTRUCTION_MAP[deepspeed.runtime.pipe.schedule.ReduceGrads] = _exec_reduce_grads

            # When pipelining multiple forward and backward passes, normally updating the parameter in-place causes an error when calling
            # backward() on future micro-batches. But we can modify .data directly so the autograd engine doesn't detect in-place modifications.
            # TODO: this is unbelievably hacky and not mathematically sound, I'm just seeing if it works at all.
            def add_(self, *args, **kwargs):
                self.data.add_(*args, **kwargs)
            for p in model_parameters:
                p.add_ = add_.__get__(p)

            if 'foreach' in inspect.signature(klass).parameters:
                kwargs['foreach'] = False

            # We're doing an optimizer step for each micro-batch. Scale momentum and EMA betas so that the contribution
            # decays at the same rate it would if we were doing one step per batch like normal.
            # Reference: https://alexeytochin.github.io/posts/batch_size_vs_momentum/batch_size_vs_momentum.html
            gas = ds_config['gradient_accumulation_steps']
            if 'betas' in kwargs:
                for i in range(len(kwargs['betas'])):
                    kwargs['betas'][i] = kwargs['betas'][i] ** (1/gas)
            if 'momentum' in kwargs:
                kwargs['momentum'] = kwargs['momentum'] ** (1/gas)

            optimizer_dict = {}
            for pg in model.get_param_groups(model_parameters):
                param_kwargs = kwargs.copy()
                if isinstance(pg, dict):
                    # param group
                    for p in pg['params']:
                        param_kwargs['lr'] = pg['lr']
                        optimizer_dict[p] = klass([p], **param_kwargs)
                else:
                    # param
                    optimizer_dict[pg] = klass([pg], **param_kwargs)

            def optimizer_hook(p):
                optimizer_dict[p].step()
                optimizer_dict[p].zero_grad()

            for p in model_parameters:
                p.register_post_accumulate_grad_hook(optimizer_hook)

            from optimizers import gradient_release
            return gradient_release.GradientReleaseOptimizerWrapper(list(optimizer_dict.values()))
        elif optim_type_lower == 'genericoptim':
            kwargs['compile'] = config['compile']
            kwargs['mpu'] = pipeline_model.mpu()
            param_groups = split_2d_param_groups(model.get_param_groups(model_parameters), kwargs)
        else:
            param_groups = model.get_param_groups(model_parameters)

        if not param_groups:
            # get_param_groups legitimately returns [] when every learning rate that applies to
            # this rank is 0. The DummyOptimizer escape above cannot catch it, because
            # parameters_to_train is snapshotted from requires_grad BEFORE get_param_groups
            # freezes the zero-lr groups. Left alone, the optimizer constructor raises
            # 'optimizer got an empty parameter list', which names neither the learning rate nor
            # the pipeline stage.
            raise RuntimeError(
                'No trainable parameters on this pipeline stage: every learning rate that '
                'applies here is 0.\n'
                '  A refiner-only configuration (base_lr = self_attn_lr = cross_attn_lr = '
                'mlp_lr = mod_lr = 0) has trainable parameters only on the stage holding the '
                'refiner, so it needs pipeline_stages = 1. Use data parallelism across GPUs '
                'instead, or give another parameter group a non-zero learning rate.'
            )

        # split weight decay and no weight decay params
        new_param_groups = []
        for pg in param_groups:
            params_no_wd = []
            params_wd = []
            params = pg.pop('params')
            for p in params:
                if p.ndim == 1 or p.original_name.startswith('llm_adapter.embed'):
                    params_no_wd.append(p)
                else:
                    params_wd.append(p)
            pg_no_wd = pg.copy()
            pg['params'] = params_wd
            pg_no_wd['params'] = params_no_wd
            pg_no_wd['weight_decay'] = 0
            if optim_type_lower == 'genericoptim':
                # If we aren't using weight decay, don't use Muon either (handles LLM adapter embed properly)
                pg_no_wd['muon'] = False
                pg_no_wd['adamuon'] = False
                pg_no_wd['normuon'] = False
            if len(params_wd) > 0:
                new_param_groups.append(pg)
            if len(params_no_wd) > 0:
                new_param_groups.append(pg_no_wd)
        param_groups = new_param_groups

        optimizer = klass(param_groups, *args, **kwargs)
        if collective == 'prodigy':
            sync_prodigy(optimizer, pipeline_group)
        return optimizer

    # A refusal raised on one pipeline stage only (no trainable parameter there, a model's
    # get_param_groups check) must stop the other stages too, not leave them waiting.
    error = None
    try:
        model_engine._configure_optimizer(get_optimizer, parameters_to_train)
    except Exception as e:
        error = e
    all_ok(error)
    optimizer = model_engine.optimizer
    check_prodigy_learning_rates(optimizer, config['optimizer']['type'].lower(),
                                 config['optimizer'].get('gradient_release', False), config['pipeline_stages'])

    model.model_engine = model_engine

    train_data.post_init(
        model_engine.grid.get_data_parallel_rank(),
        model_engine.grid.get_data_parallel_world_size(),
        micro_batch_size_per_gpu,
        model_engine.gradient_accumulation_steps(),
        image_micro_batch_size_per_gpu,
    )
    for eval_data in eval_data_map.values():
        eval_data.post_init(
            model_engine.grid.get_data_parallel_rank(),
            model_engine.grid.get_data_parallel_world_size(),
            eval_micro_batch_size_per_gpu,
            config['eval_gradient_accumulation_steps'],
            eval_image_micro_batch_size_per_gpu,
        )

    # Might be useful because we set things in fp16 / bf16 without explicitly enabling Deepspeed fp16 mode.
    # Unsure if really needed. The trained weights are the adapter's when there is one: an fp32
    # adapter keeps its gradients in fp32 through the data-parallel all-reduce.
    communication_data_type = config['adapter']['dtype'] if 'adapter' in config else config['model']['dtype']
    model_engine.communication_data_type = communication_data_type

    train_dataloader = dataset_util.PipelineDataLoader(
        train_data, model_engine, model_engine.gradient_accumulation_steps(), model,
        # More than one worker pays off when the dataset decodes media at __getitem__ time
        # (models with load_media_on_the_fly, e.g. iris); cached latents need only the default.
        num_dataloader_workers=config.get('dataloader_workers', 1),
        text_encoding=text_encoding,
    )
    steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()

    scheduler_type = config.get('lr_scheduler', 'constant')
    lr_scheduler = create_lr_scheduler(
        optimizer,
        scheduler_type,
        total_steps=config['epochs'] * steps_per_epoch,
        warmup_steps=config['warmup_steps'],
        num_cycles=config.get('lr_scheduler_num_cycles', 1),
    )
    model_engine.lr_scheduler = lr_scheduler

    step = 1
    examples = global_batch_size
    loss_average = LossAverage(config['loss_average_steps'])
    # make sure to do this before calling model_engine.set_dataloader(), as that method creates an iterator
    # which starts creating dataloader internal state
    if resume_from_checkpoint:
        # Every rank reads the checkpoint's file list; one failing (a volume error) stops all.
        error = layer_shift = None
        try:
            layer_shift = check_checkpoint_layers(
                run_dir, layers, getattr(model, 'checkpoint_layers_added_in_front', 0), tag=resume_tag)
        except Exception as e:
            error = e
        all_ok(error)
        restore = None
        if layer_shift:
            if is_main_process():
                print(f'The checkpoint has the layout from before {layer_shift} layer(s) without parameters '
                      'were put in front; loading it shifted. Its next save has the new layout.')
            restore = shift_checkpoint_layers(model_engine.module, layer_shift)
        param_groups = optimizer.param_groups.copy()
        names_before = group_names(optimizer.param_groups)
        try:
            load_path, client_state = model_engine.load_checkpoint(
                run_dir,
                tag=resume_tag,
                load_module_strict=False,
                load_lr_scheduler_states='force_constant_lr' not in config and not args.reset_optimizer and not args.reset_optimizer_params,
                load_optimizer_states=not args.reset_optimizer,
            )
        finally:
            if restore is not None:
                restore()
        if not args.reset_optimizer:
            check_resumed_param_groups(names_before, group_names(optimizer.param_groups))
        if args.reset_optimizer_params:
            optimizer.param_groups = param_groups
        dist.barrier()  # just so the print below doesn't get swamped
        assert load_path is not None
        if args.reset_dataloader:
            train_dataloader.epoch = client_state['custom_loader']['epoch']
        else:
            train_dataloader.load_state_dict(client_state['custom_loader'])
        step = client_state['step'] + 1
        if 'examples' in client_state:
            examples = client_state['examples'] + global_batch_size
        else:
            examples = step * global_batch_size
        # Checkpoints from before it start the averages afresh.
        if 'loss_average' in client_state:
            loss_average.load_state_dict(client_state['loss_average'])
        del client_state
        if is_main_process():
            print(f'Resuming training from checkpoint. Resuming at epoch: {train_dataloader.epoch}, step: {step}')

    if 'force_constant_lr' in config:
        model_engine.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        for pg in optimizer.param_groups:
            pg['lr'] = config['force_constant_lr']

    eval_dataloaders = {
        name: dataset_util.PipelineDataLoader(eval_data, model_engine, config['eval_gradient_accumulation_steps'], model,
                                              num_dataloader_workers=0, text_encoding=text_encoding)
        for name, eval_data in eval_data_map.items()
    }

    oplora_projector = None
    if is_adapter and config['adapter']['oplora']:
        oplora_projector = OPLoRAProjector.build(
            model_engine.module,
            rank=config['adapter']['oplora_rank'],
            full_svd=config['adapter']['oplora_full_svd'],
            base_seed=config['adapter']['oplora_seed'],
            exclude_names=model.oplora_exclude_names,
        )
        if is_main_process():
            print(oplora_projector.describe())

    ema = None
    if 'ema_decay' in config:
        # After the optimizer (get_param_groups may freeze zero-lr groups) and after a resume
        # restored the weights. Only data-parallel rank 0 of each stage keeps the average: the
        # replicas hold identical weights and only those ranks save.
        holds_ema = model_engine.grid.get_data_parallel_rank() == 0
        ema = ParameterEMA(
            pipeline_model.named_parameters() if holds_ema else [],
            config['ema_decay'], config['ema_update_every'], config['ema_start_step'],
            config['ema_device'], is_adapter=is_adapter, dtype=config['ema_dtype'],
        )
        if resume_from_checkpoint:
            tag = resume_tag
            found, missing = load_ema_checkpoint(ema, run_dir, tag)
            if holds_ema and missing:
                print(f'WARNING: stage {model_engine.grid.get_pipe_parallel_rank()}: checkpoint {tag} has no EMA for '
                      f'{len(missing)} of {len(ema)} parameters (EMA was off or the run changed); their average '
                      'starts from the current weights.')
            if config.get('ema_init_path') and holds_ema:
                print(f'Resuming: the EMA comes from checkpoint {tag}; ema_init_path is not used.')
        elif config.get('ema_init_path'):
            loaded = load_ema_weights(ema, config['ema_init_path'])
            if holds_ema:
                print(f'EMA (stage {model_engine.grid.get_pipe_parallel_rank()}): all {loaded} tensors start '
                      f'from {config["ema_init_path"]}')
        if holds_ema:
            print(f'EMA (stage {model_engine.grid.get_pipe_parallel_rank()}): {len(ema)} tensors, '
                  f'{ema.num_bytes() / 1024**3:.2f} GB {config["ema_dtype"]} on {ema.device}, decay {config["ema_decay"]}, '
                  f'every {config["ema_update_every"]} step(s), from step {config["ema_start_step"]}')

    epoch = train_dataloader.epoch
    tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None
    saver = utils.saver.Saver(args, config, is_adapter, run_dir, model, train_dataloader, model_engine, pipeline_model,
                              ema=ema, loss_average=loss_average)

    step_stats = StepStats(model_engine, config['log_update_norm'], config['optimizer'].get('gradient_release', False))
    steps_per_print = config.get('steps_per_print', 1)

    disable_block_swap_for_eval = config.get('disable_block_swap_for_eval', False)
    if config['eval_before_first_step'] and not resume_from_checkpoint:
        evaluate(model, model_engine, eval_dataloaders, tb_writer, 0, config['eval_gradient_accumulation_steps'], disable_block_swap_for_eval)

    empty_cuda_cache()
    while True:
        model_engine.reset_activation_shape()
        iterator = get_data_iterator_for_step(train_dataloader, model_engine)
        if nan_check is not None:
            nan_check.before_step(step)
        logged_step = step % config['logging_steps'] == 0 or step % steps_per_print == 0
        step_stats.before_step(logged_step)
        loss = model_engine.train_batch(iterator).item()
        if nan_check is not None:
            nan_check.after_step(step, loss)
        loss_average.add(loss)
        if oplora_projector is not None:
            oplora_projector.project()
        # After the projection, which is part of what the step changes.
        stats = step_stats.after_step() if logged_step else {}
        if stats and optimizer.param_groups:
            stats['train/lr'] = optimizer.param_groups[0]['lr']
        if logged_step:
            stats = {'train/loss_avg': loss_average.average(), **stats}
        if ema is not None:
            ema.update(step)
        train_dataloader.sync_epoch()

        # Before process_epoch, whose checkpoint must not carry the finished epoch's losses into
        # the next one.
        epoch_loss = loss_average.end_epoch() if train_dataloader.epoch != epoch else None
        new_epoch, checkpointed, saved = saver.process_epoch(epoch, step, examples)
        finished_epoch = True if new_epoch != epoch else False

        x_axis = examples if config['x_axis_examples'] else step

        # Models may expose extra per-step scalars. Reached through getattr so a model that
        # defines nothing logs nothing and is untouched by this.
        extra_scalars = getattr(model, 'get_extra_log_scalars', lambda: {})()
        # .get, because steps_per_print is read straight into the DeepSpeed config dict and is
        # never setdefault-ed onto config itself.
        if is_main_process() and (extra_scalars or stats) and step % steps_per_print == 0:
            # Echoed to the console as well as Tensorboard. A run on a cluster usually has no
            # Tensorboard to hand, and a number nobody can see is not much better than no number:
            # teacher guidance in particular looks identical whether lambda is 0.9 or has decayed
            # to 0. On the steps_per_print cadence, because logging_steps defaults to every step.
            summary = '  '.join(f'{name.rsplit("/", 1)[-1]}={value:.4g}'
                                for name, value in {'train/loss': loss, **stats, **extra_scalars}.items())
            print(f'step {step}: {summary}')

        if is_main_process() and step % config['logging_steps'] == 0:
            if 'train/grad_norm' not in stats and hasattr(optimizer, '_grad_norm'):
                # Gradient release: GenericOptim's own norm, the only one there is.
                stats['train/grad_norm'] = optimizer._grad_norm
            tb_writer.add_scalar(f'train/loss', loss, x_axis)
            for name, value in {**stats, **extra_scalars}.items():
                tb_writer.add_scalar(name, value, x_axis)
            if wandb_enable:
                wandb.log({'train/loss': loss, **stats, **extra_scalars, 'step': x_axis})
            if optimizer.__class__.__name__ == 'Prodigy':
                prodigy_d = get_prodigy_d(optimizer)
                tb_writer.add_scalar(f'train/prodigy_d', prodigy_d, x_axis)
            if optimizer.__class__.__name__ in ('Automagic', 'GenericOptim'):
                lrs, avg_lr = _get_automagic_lrs(optimizer)
                if avg_lr > 0:
                    tb_writer.add_histogram(f'train/automagic_lrs', lrs, x_axis)
                    tb_writer.add_scalar(f'train/automagic_avg_lr', avg_lr, x_axis)

        if (config['eval_every_n_steps'] and step % config['eval_every_n_steps'] == 0) or (finished_epoch and config['eval_every_n_epochs'] and epoch % config['eval_every_n_epochs'] == 0):
            evaluate(model, model_engine, eval_dataloaders, tb_writer, x_axis, config['eval_gradient_accumulation_steps'], disable_block_swap_for_eval)

        if finished_epoch:
            if is_main_process():
                tb_writer.add_scalar(f'train/epoch_loss', epoch_loss, epoch)
                if wandb_enable:
                    wandb.log({'train/epoch_loss': epoch_loss, 'epoch': epoch})
            if new_epoch is None:
                final_model_name = f'epoch{epoch}'
                break
            epoch = new_epoch

        checkpointed, saved = saver.process_step(step, examples)
        if 'max_steps' in config and step >= config['max_steps']:
            final_model_name = f'step{step}'
            break
        step += 1
        examples += global_batch_size

    # Save final model and training state checkpoint, unless we just saved them. The model
    # first, as in the saver: resuming from the checkpoint never skips a save.
    if not saved:
        saver.save_model(final_model_name)
    if not checkpointed:
        saver.save_checkpoint(step, examples)

    if is_main_process():
        print('TRAINING COMPLETE!')
