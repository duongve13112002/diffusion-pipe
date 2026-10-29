# diffusion-pipe
A pipeline parallel training script for diffusion models.

Models supported: SDXL, Flux, LTX-Video, HunyuanVideo (t2v), Cosmos, Lumina Image 2.0, Wan2.1 (t2v and i2v), Chroma, HiDream, Stable Diffusion 3, Cosmos-Predict2, OmniGen2, Flux Kontext, Wan2.2, Qwen-Image, Qwen-Image-Edit, HunyuanImage-2.1, AuraFlow, Z-Image, HunyuanVideo-1.5, Flux 2 (Dev and Klein), Anima, Ernie-Image, LTX 2.3, Ideogram4, Krea 2, MiniMax H3, Qwen-Image-2.1, Iris-3B, MOSS-Image.

## Features
- Pipeline parallelism, for training models larger than can fit on a single GPU
- Useful metrics logged to Tensorboard
- Compute metrics on a held-out eval set, for measuring generalization
- Training state checkpointing and resuming from checkpoint
- Efficient multi-process, multi-GPU pre-caching of latents and text embeddings
- Seemlessly supports both image and video models in a unified way
- Easily add new models by implementing a single subclass
- Optional OPLoRA (orthogonal projection LoRA) to reduce catastrophic forgetting when training LoRAs

## Recent changes
- 2026-10-29
  - `--cache_status`: prints how many files of each dataset directory (and of each bucket) have their latents and their text embeddings cached, counted in files, not captions, then exits. It reads what is cached, also from a run that was interrupted or is still caching, and encodes, clears and writes nothing. See [pre-caching](#pre-caching).
  - The progress bar of caching now shows the files done (`latents: 41%|... 4120/10000 [..., 4120/10000 files]`, `text_embeddings_1: ... 880/3000 files`), also for text embeddings, whose bar counts a file's captions one by one.
  - Caching shows where its time goes. The bar ends with `saving 12%, waiting 88%`: the share of the time the process that collects the results spent writing them to the cache, and waiting for the encoders and workers to send more. Each GPU rank prints a line every minute, `[CACHE] rank 0, last 60 s, 18 tasks: waiting for work 4%, encoding 71%, remembering rows 9%, moving results 16%`, which says what the GPU waited for. The results are now written by a thread of their own (at most 512 MB queued, then the process waits for the disk), and the workers run at most `2 * map_num_proc` batches ahead of what has been collected, so a slow disk can no longer pile finished batches up in memory.
- 2026-10-28
  - `use_cached_latents_only` and `use_cached_embeddings_only` (top level or per `[[directory]]`, default `false`) train on what an unfinished caching run left, without encoding more. With the latents option, only images whose latents are cached are trained; with the embeddings option, only captions whose embeddings (for every text encoder) are cached; with both, only samples that have both. The option left off still caches as usual, and with `cache_text_embeddings = false` the captions are encoded during training as ever, so the latents option alone is what limits the data. The startup log says how many samples are used (`training on the cached latents only: 41200 of 90000 samples`). The empty caption's embedding is still encoded when it is missing (one caption). A cache made for other inputs (images, captions or a setting changed since) or by another VAE or text encoder is refused, never cleared; `--regenerate_cache` is refused with these options. Data parallel ranks train on the samples every one of them can read. The cache now commits what it has written every 5 minutes, so an interrupted run keeps (and a resumed run skips) nearly everything it encoded; before, nothing was kept until a 10 GB shard was full.
  - A cache records the VAE or text encoder that fills it from its first row, not only once complete: a run resumed with another one rebuilds the cache instead of appending its rows to the other encoder's.
  - Refused: `max_cached_captions` with `online_captions` when text embeddings are not cached (the live caption was read by the wrong number), and `text_embedding_dtype` for a model that encodes its captions itself (MOSS-Image, SDXL, Wan, Cosmos-Predict2, Iris) with `cache_text_embeddings = false`, where it was ignored.
  - float8 text embeddings: a token whose largest value is below about 6e-43 no longer decodes to NaN.
  - ComfyUI submodule updated to master 79cf64c (v0.43.0).
- 2026-10-27
  - MOSS-Image caches its latents in `vae_dtype` instead of float32 (ComfyUI's VAE always returned float32). Existing caches are kept and still work.
  - `text_embedding_dtype` in `[model]` chooses the dtype cached text embeddings are stored in, for every model: `'float8'` (e4m3 with a scale per token, about half the size of bfloat16), `'bfloat16'`, `'float16'` or `'float32'`. They are read back in the text encoder's own dtype. See [text embeddings: cached or on the fly](#text-embeddings-cached-or-on-the-fly).
  - `max_cached_captions` (top level or per `[[directory]]`) keeps at most that many captions of each image, picked at random once per image; unset keeps them all. The pick is one seeded shuffle per image, so a smaller limit keeps a subset of what a larger one kept: with `reuse_metadata_cache = true`, lowering the limit re-encodes nothing and raising it encodes only the added captions.
- 2026-10-26
  - DoRA: set `use_dora = true` in a `type = "lora"` `[adapter]` table. It costs extra VRAM and step time. It cannot be combined with `oplora` (refused since 2026-10-28: the trained magnitude rescales the base weight and moves the subspace OPLoRA protects). The magnitude vector is saved as `<module>.dora_scale` (shape `(out, 1)`, the kohya name for SDXL), which ComfyUI and kohya read; ComfyUI measures the norm on the original weight, a small approximation (a few percent for a small adapter) of what PEFT trains with. Flux and SDXL resume a DoRA adapter through their own loader (diffusers would drop the magnitude); Flux's `fuse_adapters`, `tools/iris_merge_adapter.py --strength` and `tools/sample_anima_refiner.py --lora-strength` other than 1 refuse a DoRA adapter; the other models' `fuse_adapters` merge it through PEFT. Refused since 2026-10-28: `use_dora` with a float8 `transformer_dtype` or `diffusion_model_dtype`, and an `init_from_existing` of the other kind (a plain LoRA with `use_dora = true`, or a DoRA without it); EMA `ema_init_path` now reads a DoRA save.
- 2026-10-25
  - `--metadata_only`: scans the datasets, groups them into buckets and caches that metadata, prints the files per bucket, then exits, with no latents or text embeddings. See [pre-caching](#pre-caching).
  - `resolutions`, `min_ar`, `max_ar` and `num_ar_buckets` set only on a `[[directory]]` no longer fail with a `KeyError` when the top level leaves them out; a setting missing from both is named in the error.
  - Aspect ratio buckets from the images: with `enable_ar_bucket = true` and no `ar_buckets`, a `min_ar` or `max_ar` set neither on the `[[directory]]` nor at the top level is the narrowest or widest image of that directory, and a missing `num_ar_buckets` is the fewest buckets that crop no image by more than `max_ar_crop` (new, default 0.1: 10% of its width or height). Before, all three were required. See [aspect ratio buckets from the images](#aspect-ratio-buckets-from-the-images).
- 2026-10-24
  - `cache_text_embeddings = false` works for every model: the latents stay cached and each batch's captions are encoded during training with the model's own text encoders, giving the same embeddings the cache would (at the default `caching_batch_size = 1`). Before, models other than Wan, Cosmos-Predict2/Anima, Iris and MOSS-Image ignored the setting and cached anyway. See [text embeddings: cached or on the fly](#text-embeddings-cached-or-on-the-fly).
  - Fixed AuraFlow's UMT5 and HunyuanImage-2.1's ByT5 text encoders running in train mode: their dropout (0.1) noised every cached embedding. Their text embedding caches are rebuilt on the next run (latents are kept). AuraFlow also no longer encodes an unused negative prompt per caption.
  - Fixed `size_buckets`: media were matched to a bucket by an index into the sorted, deduplicated aspect ratios, so unless the buckets were listed in increasing aspect ratio (each ratio once), images went to a bucket of another aspect ratio, and with image and video buckets together every image was dropped. Datasets whose buckets were affected rebuild their metadata and latent caches on the next run, `--trust_cache` included, and `tools/inherit_metadata_cache.py` refuses to copy metadata grouped before the fix.
  - Dataset caching on several nodes: the nodes take directories (metadata), then buckets (latents and text embeddings), from pools they share until none are left. See [multi-node training](#multi-node-training).
  - With `reuse_metadata_cache = false`, `--trust_cache` no longer reuses metadata grouped for other buckets: the buckets (aspect ratios, frame buckets or `size_buckets`) are recorded with the grouping, and a change rescans the dataset, as a run without `--trust_cache` does. Metadata grouped before this is checked against the current buckets, and an interrupted scan is never trusted. A train and an eval dataset reading one directory with the same caption settings must use the same buckets here (they share the cache), or set `reuse_metadata_cache = true`. `tools/inherit_metadata_cache.py` refuses a donor grouped by other buckets. Also, each caption setting of a directory keeps its own per-resolution copy of an AR bucket's metadata (rebuilt once, a quick pass); under `--trust_cache` a second setting used to read the first one's captions.
  - Faster metadata scan, on one node too: a directory is listed once without a stat per file, and image headers and caption files are read on `metadata_read_threads` threads per map process (default 16). On network storage the scan waits on each file open: measured with 3 ms per open, 16 threads scanned 5000 images 7x faster than one. On a local disk threads are slower (about 2.7x), so set `metadata_read_threads = 1` there. Cached metadata is unaffected.
- 2026-10-16
  - Support MOSS-Image (Base and SFT): LoRA, LoKr, OPLoRA and full fine-tuning, from a diffusers directory or ComfyUI single files (fp8 and comfy-quant included). See [supported models](docs/supported_models.md#moss-image) and `examples/moss_image/`. The transformer is ComfyUI's native MOSS-Image, and saved adapters and fine-tunes load in ComfyUI's own loaders.
  - Layout (JSON) captions for MOSS-Image and Ideogram4: a caption file holding the model's JSON prompt has its boxes rewritten for each bucket's crop, and tag operations skip it. Detection is per caption. The caption cache key of these two models changes, so their caption caches are rebuilt.
- 2026-10-15
  - `diffusion_model_dtype` now keeps every weight under the model's high-precision modules (e.g. `time_in`, `final_layer`, Krea2's `txtfusion`, MiniMax H3's `token_refiner`) in the base dtype, as intended; before, only weights directly in those modules were kept. This uses a little more VRAM (up to ~5% of the weights for Krea2).
  - Saved models, adapters and training checkpoints are written under a temporary name and renamed into place, so an interrupted save never leaves a torn one; a checkpoint's `latest` is written only once all its files are in place, so `--resume_from_checkpoint` always loads a complete one. A failed save now stops every rank instead of leaving them waiting. `keep_last_n_checkpoints` must be at least 2, and never removes the checkpoint `latest` names.
- 2026-09-28
  - Support Qwen-Image-2.1 (T2I training only).
- 2026-08-08
  - Add CFG-augmented training for MiniMax H3. You need to enable it, see the example TOML file. You probably always want to use either this, or a training adapter.
- 2026-08-06
  - Support MiniMax H3. Only T2I and T2VA training.
  - Modify dataset code to support audio from videos.
  - Allow training LoRAs directly on quantized models. This should theoretically work with any ComfyUI-based model (Z-Image and later, excluding Anima).
  - Simplified some latent caching code. If training Z-Image, Flux2, or Ernie-Image, make sure to `--regenerate_cache` because latent scaling is now done in the caching phase and not in the model.
- 2026-06-27
  - Add OPLoRA (orthogonal projection LoRA) to reduce catastrophic forgetting during LoRA training. Enable it with `oplora = true` and `oplora_rank` in the `[adapter]` table. See [docs/oplora.md](./docs/oplora.md).
  - Add the `cosine_with_restarts` LR scheduler (set `lr_scheduler = 'cosine_with_restarts'`, optionally `lr_scheduler_num_cycles`).
- 2026-06-24
  - Support Krea 2.
- 2026-06-07
  - Remove attention masking from Flux2. The model is supposed to take the full 512 length text embeds even with padding.
    - This gives slightly lower starting loss, but seems to make almost no difference to the final trained result. Still, it should theoretically be better now.
    - Delete cache folder or use `--regenerate_cache` or else you might get Tensor shape errors from the old cached files.
  - For ComfyUI-based models (Z-Image and later), support training directly from Comfy quantized weights, e.g. fp8_scaled.
  - Support Ideogram4.
- 2026-05-15
  - Initial LTX 2.3 support. Only T2I and T2V training for now, and no audio.
- 2026-04-23
  - Support Ernie-Image.
- 2026-02-04
  - Support Anima.

## Windows support
It will be difficult or impossible to make training work on native Windows. This is because Deepspeed only has [partial Windows support](https://github.com/microsoft/DeepSpeed/blob/master/blogs/windows/08-2024/README.md). Deepspeed is a hard requirement because the entire training script is built around Deepspeed pipeline parallelism. However, it will work on Windows Subsystem for Linux, specifically WSL 2. If you must use Windows I recommend trying WSL 2.

## Installing
Clone the repository:
```
git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe
```

If you alread cloned it and forgot to do --recurse-submodules:
```
git submodule init
git submodule update
```

Install Miniconda: https://docs.anaconda.com/miniconda/

Create the environment:
```
conda create -n diffusion-pipe python=3.12
conda activate diffusion-pipe
```

Install PyTorch first. It is not listed in the requirements file, because certain GPUs sometimes need different versions of PyTorch or CUDA, and you might have to find a combination that works for your hardware. As of this writing (October 26, 2025), PyTorch 2.9.0 with CUDA 12.8 works on my 4090, and is compatible with the current latest flash-attn 2.8.3:
```
pip install torch torchvision torchaudio
```

Install all three from the same official PyTorch CUDA index, so their versions match. All three
are required: `models/base.py` uses `torchaudio` for video-audio resampling, and the vendored
ComfyUI imports it at module scope as well, so omitting it prevents every model module from
loading. Leaving it out fails with a message naming the fix, rather than a bare import error from
inside a vendored file.

Install nvcc: https://anaconda.org/nvidia/cuda-nvcc. Probably try to make it match the CUDA version of PyTorch.

Install the rest of the dependencies:
```
pip install -r requirements.txt
```

(Optional) Install Flash Attention. It's not in the requirements file. Some models need it.
```
pip install flash-attn
```

### Cosmos requirements
NVIDIA Cosmos (the original Cosmos video model, not Cosmos-Predict2) additionally requires TransformerEngine.

This dependency isn't in the requirements file. You probably need to set some environment variables for it to install correctly. The following command worked for me:
```
C_INCLUDE_PATH=/home/anon/miniconda3/envs/diffusion-pipe/lib/python3.12/site-packages/nvidia/cudnn/include:$C_INCLUDE_PATH CPLUS_INCLUDE_PATH=/home/anon/miniconda3/envs/diffusion-pipe/lib/python3.12/site-packages/nvidia/cudnn/include:$CPLUS_INCLUDE_PATH pip install --no-build-isolation transformer_engine[pytorch]
```
Edit the paths above for your conda environment.

## Updating
```
git pull
git submodule update
```
Make sure to run `git submodule update` in case any submodule has been updated to a new commit.

The current ComfyUI pin needs `comfy-aimdo>=0.6.0` and `comfy-kitchen>=0.2.40` (the Flux 2 VAE
runs on comfy-kitchen kernels on CUDA). After `git submodule update`, run
`pip install -r requirements.txt` inside your project environment, and run the
[upstream API audit](docs/note/upstream-api-drift-audit.md) after submodule changes.

### Updating dependencies
Most dependencies are intentionally left unpinned in the requirements.txt file. If you want to update them to the latest version, you can run ```pip install -r requirements.txt -U```.

## Dataset preparation
A dataset consists of one or more directories containing image or video files, and corresponding captions. You can mix images and videos in the same directory, but it's probably a good idea to separate them in case you need to specify certain settings on a per-directory basis. Caption files should be .txt files with the same base name as the corresponding media file, e.g. image1.png should have caption file image1.txt in the same directory. If a media file doesn't have a matching caption file, a warning is printed, but training will proceed with an empty caption.

For images, any image format that can be loaded by Pillow should work. For videos, any format that can be loaded by ImageIO should work. Note that this means **WebP videos are not supported**, because ImageIO can't load multi-frame WebPs.

### Multi-caption and tag augmentation

An image can have more than one caption. `captions.json` holds a list per image, and every
caption in that list becomes its own training sample — nothing is picked at random and
discarded. A `.txt` sidecar is one caption by default, newlines included; set
`multiline_captions = true` to treat each non-empty line as a separate caption instead. That
one is opt-in because turning it on changes how many samples an existing dataset produces, and
like `cache_shuffle_num` it needs `--regenerate_cache` to take effect.

Two caption augmentations, both off by default and both settable per directory or dataset-wide:

| Setting | What it does |
| --- | --- |
| `cache_shuffle_num` / `shuffle_tags` | Shuffle the tag order. `cache_shuffle_num` caches that many shuffled variants per caption. |
| `tag_dropout_rate` | Drop each tag independently with this probability. At least one tag always survives — an all-dropped caption is the *unconditional* embedding, which the trainer already produces deliberately, so minting more of them would quietly change the conditioning ratio. |

`prefix_tag_caption` says which captions those apply to. A dataset that mixes tag lists with
natural language should not have its prose shuffled on commas; mark the tag captions with a
prefix and only they are augmented:

```toml
prefix_tag_caption = "Special: "
```

The marker is **stripped before training**, so `"Special: a, b, c"` is trained as `"a, b, c"`.
Matching **ignores case**, and both the marker and what follows it are stripped of surrounding
whitespace — `"Special:"`, `"special: "` and `"SPECIAL:"` all match, and `"SPECIAL:   a, b"`
still trains as `"a, b"`. Unset (the default) means the dataset is not annotated and every
caption is treated as tags, which is how this repo behaved before the setting existed.

Both augmentations are applied when the text embeddings are **cached**, so changing either needs
`--regenerate_cache`, and each cached variant is a separate training sample. That means
`tag_dropout_rate` with `cache_shuffle_num = 0` gives exactly **one** frozen draw — permanent tag
deletion, not augmentation. Raise `cache_shuffle_num` to cache several draws; the code warns if
you don't.

They are applied fresh per sample in two cases: when the model caches no text embeddings at all
(so the caption string is what gets encoded every step — SDXL always, any model with
`cache_text_embeddings = false`), and on the `online_captions` path below. A model that caches
*some* encoders and reads the caption for others, like HiDream with cached embeddings, is
deliberately excluded: re-augmenting text whose embedding is already frozen would make the two
disagree.

### Choosing which caption an image trains on

By default **every** caption is its own training sample: an image with three captions appears
three times per epoch, and one epoch is `n_images × n_captions`. Nothing is picked at random and
discarded.

```toml
caption_sampling = "random_per_epoch"
```

switches to one sample per image, with a caption drawn at random on every access — so an epoch is
`n_images` and an image gets a different caption on each pass. Every caption's embedding is still
cached; the draw is just an index into that cache, so it costs nothing extra to cache. The same
draw selects the caption text and its embedding together, so the two can never disagree.

This also restores tag augmentation with cached embeddings: with `cache_shuffle_num = 10` the
cache holds ten shuffled/dropped variants per caption, and each epoch draws a different one.

### Whole-caption dropout and line filtering

Dataset TOML supports `caption_dropout_rate` (a probability in `[0, 1]`) and
`enable_remove_non_latin` (a boolean, default `false`), globally or per `[[directory]]`.
Whole-caption dropout is re-drawn on every sample access, including with cached embeddings:
the caption and every cached text encoder switch together to the conditioning for `""`.
An explicit rate overrides the training TOML's `uncond_fraction`; an absent rate preserves it.
Changing the rate does not invalidate caches.

Line filtering removes whole lines containing the requested non-Latin Unicode ranges in
memory before augmentation/encoding. Source caption files stay unchanged. Tag markers are
excluded from the check, and fully filtered captions remain empty without a training prefix.
Changing the filter flag separates caption metadata and refreshes text embeddings as needed,
while unchanged image latents are reused. See [caption processing](./docs/caption-processing.md)
for exact ranges, precedence, examples, cache behavior and validation commands.

To select whole images/videos, set dataset `require_non_latin_caption = true` to keep
only media with at least one non-Latin source-caption line, or `false` to exclude any
such media. All alternative captions follow the selected image. Omit the setting to keep
normal behavior. This selection runs before caption processing and also applies to corpus
export/dataset-driven distillation. Configuration profiles reference compatible existing
latent/text tensors instead of duplicating each selected subset. New caption inputs still
need text encoding. See [cache profiles](./docs/caption-processing.md#source-snapshots-and-cache-profiles).

### `online_captions`

```toml
online_captions = true
```

With `reuse_metadata_cache = false`, this reads captions from `captions.json` at access time instead of from the cached metadata, so you can
edit captions without rebuilding the metadata cache. It does **not** change how many samples an
image produces. And note the caption text only reaches the model when the model does not have a
cached embedding for it — with text embeddings cached, editing `captions.json` changes nothing
until you regenerate the cache.

The default `reuse_metadata_cache = true` freezes source captions, including online captions,
in a metadata snapshot until `--regenerate_cache`. On-the-fly text encoding still encodes
the processed snapshot caption every step; it does not require a text embedding cache.

## Supported models
See the [supported models doc](./docs/supported_models.md) for more information on how to configure each model, the options it supports, and the format of the saved LoRAs.

## Training
**Start by reading through the config files in the examples directory.** Almost everything is commented, explaining what each setting does. [This config file](./examples/main_example.toml) is the main example with all of the comments. [This dataset config file](./examples/dataset.toml) has the documentation for the dataset options.

Once you've familiarized yourself with the config file format, go ahead and make a copy and edit to your liking. At minimum, change all the paths to conform to your setup, including the paths in the dataset config file.

Launch training like this:
```
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config examples/hunyuan_video.toml
```
RTX 4000 series needs those 2 environment variables set. Other GPUs may not need them. You can try without them, Deepspeed will complain if it's wrong.

If you enabled checkpointing, you can resume training from the latest checkpoint by simply re-running the exact same command but with the `--resume_from_checkpoint` flag. You can also specify a specific checkpoint folder name after the flag to resume from that particular checkpoint (e.g. `--resume_from_checkpoint "20250212_07-06-40"`). This option is particularly useful if you have run multiple training sessions with different datasets and want to resume from a specific training folder.

Please note that resuming from checkpoint uses the **config file on the command line**, not the config file saved into the output directory. You are responsible for making sure that the config file you pass in matches what was previously used.

## Output files
A new directory will be created in ```output_dir``` for each training run. This contains the checkpoints, saved models, and Tensorboard metrics. Saved models/LoRAs will be in directories named like epoch1, epoch2, etc. Deepspeed checkpoints are in directories named like global_step1234. These checkpoints contain all training state, including weights, optimizer, and dataloader state, but can't be used directly for inference. The saved model directory will have the safetensors weights, PEFT adapter config JSON, as well as the diffusion-pipe config file for easier tracking of training run settings.

### Step metrics
Every `logging_steps` steps Tensorboard (and wandb) get, besides `train/loss`, the size of the step; every
`steps_per_print` steps the console gets the same numbers on one line:
- `train/loss_avg`: the loss averaged over the last `loss_average_steps` steps (100 by default). One step's
  loss depends more on the timesteps and noise it drew than on the weights, so it jumps around; the average is
  the one to read for whether the loss is still going down. `train/epoch_loss`, logged at the end of each epoch,
  is the average over the epoch. The checkpoint carries both, so a resumed run goes on with them.
- `train/grad_norm`: the norm of the whole model's gradient before clipping, over every pipeline stage. This is
  the number `gradient_clipping` is compared with.
- `train/grad_clipped`: 1 on the steps that were clipped, 0 otherwise (with `gradient_clipping` above 0), so
  its average over a window is the share of clipped steps. Clipping most steps means `gradient_clipping` is
  below the usual norm and no longer only catches spikes.
- `train/update_norm` and `train/update_ratio`: the norm of the change the step made to the trained weights,
  and that over the norm of the weights before it. The ratio is how much of the weights one step moves, what
  the learning rate (and Prodigy's `d`, or a warmup) controls in the end. A ratio that climbs and stays well
  above its usual level often comes before a loss spike; one at 0 means the updates round away in the weights'
  dtype. Measuring it keeps a copy of the trained weights through the step: `log_update_norm` is on by default
  when training an adapter and off for a full fine-tune, where the copy is as large as the model.
- `train/lr`: the first parameter group's learning rate.

## Reducing VRAM requirements
The [wan_14b_min_vram.toml](./examples/wan_14b_min_vram.toml) example file has all of these settings enabled.
- Use AdamW8BitKahan optimizer:
  ```
  [optimizer]
  type = 'AdamW8bitKahan'
  lr = 5e-5
  betas = [0.9, 0.99]
  weight_decay = 0.01
  stabilize = false
  ```
- Use block swapping if the model supports it: ```blocks_to_swap = 32```
- Try the expandable_segments feature in the CUDA memory allocator:
  - ```PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config /home/you/path/to/config.toml```
  - I've seen this help a lot when training on video with multiple aspect ratio buckets.
  - On my system, sometimes this causes random CUDA failures. If training gets through a few steps though, it will train indefinitely without failures. Very weird.
- Use unsloth activation checkpointing: ```activation_checkpointing = 'unsloth'```

## Parallelism
This code uses hybrid data- and pipeline-parallelism. Set the ```--num_gpus``` flag appropriately for your setup. Set ```pipeline_stages``` in the config file to control the degree of pipeline parallelism. Then the data parallelism degree will automatically be set to use all GPUs (number of GPUs must be divisible by pipeline_stages). For example, with 4 GPUs and pipeline_stages=2, you will run two instances of the model, each divided across two GPUs.

Each pipeline stage runs its own optimizer over its own layers. Two optimizers that take one
statistic over the whole model share it across stages: GenericOptim's gradient norm, and Prodigy's
step size `d`. Prodigy steps every parameter group with one learning rate, so with
`pipeline_stages > 1` it is refused when a model gives its groups different ones (with one stage,
a warning says they are not applied). The DAdapt* optimizers estimate their step size from all the
parameters they update and cannot share it, so they are refused with `pipeline_stages > 1`. Other
optimizers with an option computed over all their parameters (a global gradient norm clip in
pytorch_optimizer's Adan, Lamb, AdaGC and similar) compute it per stage.

### Multi-node training
The command is the usual one on every node, through any launcher that sets `RANK`, `WORLD_SIZE`,
`LOCAL_RANK`, `MASTER_ADDR` and `MASTER_PORT`. That covers the deepspeed launcher with a hostfile (or
`--no_ssh --num_nodes N --node_rank R --master_addr ...` on each node), torchrun
(`torchrun --nnodes N --node_rank R --nproc_per_node 8 --master_addr ... --master_port ... train.py
--deepspeed --config ...`), and a Kubernetes PyTorchJob. `--master_port` is only needed to override
the launcher's port.

- Ranks are numbered stage by stage: with 2 nodes of 8 GPUs and `pipeline_stages = 2`, stage 0 is the
  first node and stage 1 the second. The data-parallel gradient all-reduce then stays inside each
  node, and only the activations between the two stages cross the network. With
  `pipeline_stages = 1`, the all-reduce crosses nodes on every step.
- `output_dir` and the dataset directories (their `cache` folders) must be on a volume that every
  node shares, such as NFS or a ReadWriteMany PVC. Training checks `output_dir` at start; pipeline
  stages pass saved weights to stage 0 through it.
- Dataset caching runs on every node, from pools of work the nodes share: a node that finishes a
  unit takes the next one, so faster nodes and lighter units even out by themselves. First the
  metadata, one `[[directory]]` at a time (entries reading the same `path` count as one), then,
  once every directory's metadata is cached, the latents, one bucket at a time, largest first (a
  size bucket, or a whole aspect ratio bucket with all its resolutions), and after them the text
  embeddings, bucket by bucket again, one text encoder at a time: each round ends on every node
  before the next starts, so a GPU swaps models once per round. Every node then reads the whole cache from the
  shared folders. A directory or a bucket is the
  smallest unit, so a dataset with fewer of them than nodes leaves some nodes idle, and one very
  large directory has its metadata cached by one node. A caching failure on any node stops every
  node, and so does a rank that gives no sign of life for 15 minutes (a killed or hung process),
  rather than leaving the others waiting. `multi_node_caching = false` caches everything on global rank 0's node instead, as before.
- Network: over Ethernet, set `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` to the interface the
  nodes reach each other on. Over InfiniBand, give the pods the RDMA device (and `IPC_LOCK`), and set
  `NCCL_IB_HCA` if needed. With `NCCL_DEBUG=INFO`, the log shows `NET/IB` rather than `NET/Socket`
  when NCCL uses it, since NCCL silently falls back to TCP.
- Collectives time out after 30 minutes by default. Raise `DEEPSPEED_TIMEOUT` (minutes) if the ranks
  can drift further apart than that, for example while every rank loads a large dataset's metadata
  from a slow volume at start.
- Resuming on another number of GPUs or nodes keeps the epoch and continues it after about the same
  number of examples, with a warning: a batch then holds a different number of examples.

## Pre-caching
Latents and text embeddings are cached to disk before training happens. This way, the VAE and text encoders don't need to be kept loaded during training. The Huggingface Datasets library is used for all the caching. Cache files are reused between training runs if they exist. All cache files are written into a directory named "cache" inside each dataset directory.

This caching also means that training LoRAs for text encoders is not currently supported.

`--cache_only` performs caching then exits. Default dataset `reuse_metadata_cache = true`
reuses source metadata without rescanning files, with or without `--trust_cache`, while
always selecting the profile for the current caption settings. Source edits require
`--regenerate_cache`, which creates a fresh snapshot/tensor generation and preserves the
old one. With `reuse_metadata_cache = false`, the legacy workflow remains available:
omitting `--trust_cache` rescans sources, and supplying it trusts existing metadata.
First migration from older caches may read sidecars and build input indexes once;
new profiles still process metadata, so startup is not constant-time on very large datasets.

`--metadata_only` does only the metadata part: it scans the datasets, groups them into
buckets, caches that, prints how many files each bucket of each directory holds (train and
eval datasets), then exits. Neither the VAE nor a text encoder runs and nothing is encoded,
so it is a quick way to check the buckets before spending GPU time. Some models still read
their weights into RAM when the pipeline is built, as for any run; nothing goes to the GPU.
It is launched like training (`deepspeed train.py --deepspeed --config ... --metadata_only`)
and splits the directories across nodes as caching does. The next run reuses this metadata
as it reuses any metadata cache (with `reuse_metadata_cache = false`, pass `--trust_cache` to
skip checking the sources again) and only caches the latents and text embeddings.
`--regenerate_cache` and `--trust_cache` work with it as with `--cache_only`; it refuses
`--cache_only`, `--dump_dataset`, `--test_sample`, `--resume_from_checkpoint` and Turbo configs.

`--cache_status` tells how much is cached. It loads the metadata like `--metadata_only`, then
reads the latent and text embedding caches of every directory (train and eval datasets) and prints,
per directory and per bucket, how many files have them:

```
/data/photos: 12000 files in 9 buckets
  latents: 8123 of 12000 files (67.7%)
  text embeddings, text encoder 1: 5000 of 12000 files (41.7%); 40 more have the embeddings of only some of their captions
  text embeddings, text encoder 2: 4980 of 12000 files (41.5%)
    1024x1024: 3000 files, latents 2400, text encoder 1 1500, text encoder 2 1490
```

It counts files, not captions: a file has its text embeddings when every one of its captions has
them, for each text encoder; a file with only some is counted apart. An aspect ratio bucket's file
has its latents when all its resolutions do. It encodes nothing, puts no model on the GPU (some models still read their weights into RAM when the
pipeline is built, as for `--metadata_only`), and does not clear or rewrite any cache: a cache made for other inputs (the images, captions or a
setting changed) or by another VAE or text encoder is reported on its bucket's line, not rebuilt.
A directory whose model reads its media while training, or with `cache_text_embeddings = false`,
says so instead of counting. A run that is caching commits what it wrote every 5 minutes, so the
count can be up to that far behind; when another run is caching the same dataset, add
`--trust_cache` so that this one does not rescan the sources (with `reuse_metadata_cache = true`
nothing is rescanned anyway). Launch it like training (`deepspeed train.py --deepspeed --config ... --cache_status`);
it refuses `--metadata_only`, `--cache_only`, `--regenerate_cache`, `--dump_dataset`,
`--test_sample` and Turbo configs. `--resume_from_checkpoint` may stay in the command: it exits
before any checkpoint is loaded, so it does nothing here.

While caching, the progress bar shows the same count for the cache it is filling:
`latents: 41%|████ | 4120/10000 [...]` ends with `4120/10000 files`, and
`text_embeddings_1` shows the files whose captions are all encoded, although its bar moves one
caption at a time (a file counts once its rows are handed to the thread that writes them, at most
512 MB behind the disk). After the files, `saving N%, waiting M%` shows what limits the speed:
a high `saving` means the disk (or network volume) is the limit; a high `waiting` means the
encoders, or the workers that feed them, are. For the GPU side, every GPU rank prints
`[CACHE] rank 0, last 60 s, 18 tasks: waiting for work 4%, encoding 71%, remembering rows 9%,
moving results 16%` once a minute: `waiting for work` high means too few CPU workers (or slow
image decoding) are feeding the GPU, `encoding` is the GPU itself, `remembering rows` is the copy
that lets repeated captions be encoded once, and `moving results` is copying results to the CPU and
sending them to the caching process. Raising `caching_batch_size` only helps the `encoding` share.

### Text embeddings: cached or on the fly

`cache_text_embeddings` in `[model]` chooses, for every model, whether the text embeddings are
cached or the captions are encoded during training. Latents are cached either way.

**`true`** (the default for most models): the text embeddings are cached with the latents, and
the text encoders are freed before training. `caching_batch_size` is the batch size of caching
(VAE and text encoders): each caption is padded to the longest one of its group of
`caching_batch_size` captions, and its embedding is stored on its own. Training reads
`micro_batch_size_per_gpu` rows from the cache per step, so the two settings are unrelated here
(say `caching_batch_size = 32` with a micro batch of 8: caching encodes 32 at a time, training
reads 8 per step).

`text_embedding_dtype` in `[model]` sets the dtype the cached embeddings are stored in; unset
stores what the text encoder returns. The embeddings are put back in the encoder's own dtype when
read, so the model is unchanged, and masks and token ids are stored as they are. A cache in another
dtype is rebuilt, and removing the setting finds the old cache again. It needs
`cache_text_embeddings = true`.

| value | size (vs bfloat16) | precision |
|---|---|---|
| `'float8'` | about half | each value off by about 2% (up to about 6%) |
| `'bfloat16'` | same | exact for an encoder that returns bfloat16 |
| `'float16'` | same | finer than bfloat16; refused if a value is above 65504 |
| `'float32'` | double | exact |

`'float8'` stores float8_e4m3fn with each token's vector divided by a power-of-two scale (its
largest value over 448, rounded up), so the outliers of text encoder hidden states are not clipped.

**`false`**: only the latents are cached. The text encoders stay loaded on the GPU and encode
each batch's captions during training, so tag shuffling and `tag_dropout_rate` draw anew for
every sample. It costs the encoders' VRAM and an encoder pass per step. The caption text still
comes from the metadata cache, so an edited caption file is picked up the same way as with
`true` (see [Pre-caching](#pre-caching)); only re-encoding is saved. How the captions are encoded
depends on the model:

| Model | How captions are encoded with `false` | What `caching_batch_size` does |
|---|---|---|
| Wan, Cosmos-Predict2 / Anima, Iris, MOSS-Image | Inside the model's own pipeline: the whole batch (see below) is tokenized and encoded at once on the GPU. Iris and MOSS-Image default to `false`. | Latent caching only. |
| SDXL | Always encoded inside its pipeline (its text encoders can be trained); text embeddings are never cached. | Latent caching only. |
| Every other model (Flux, Flux2, Chroma, HiDream, SD3, Lumina 2, OmniGen2, Cosmos, LTX-Video, LTX-2, HunyuanVideo, HunyuanVideo-1.5, HunyuanImage, Qwen-Image, Qwen-Image-2.1, AuraFlow, Z-Image, ERNIE-Image, Ideogram4, Krea2, MiniMax H3) | Each batch is encoded before the model sees it, with the same functions that fill the cache (`utils/text_encoding.py`). | Latent caching, and how many captions are encoded together: a batch is split into groups of `caching_batch_size` (a value above the batch size encodes the whole batch at once). |

The "batch" encoded per step is `micro_batch_size_per_gpu` x `gradient_accumulation_steps`
captions on each GPU. For example, with a micro batch of 16 (and no gradient accumulation) and
`caching_batch_size = 4`, each text encoder runs 4 times per step on 4 captions, and the results
are put back together as 16.

Padding, for the models of the last row: each caption is padded to the longest one of its group,
as caching does with the same `caching_batch_size`, except that the group is taken from the
training batch instead of the caption files' order. So:

- `caching_batch_size = 1` (the default): every caption is encoded alone, and a batch holds
  exactly the embeddings the cache would hold at its default.
- A larger value is faster but pads captions as caching with that value would. Chroma, HiDream,
  Lumina 2, OmniGen2 and Cosmos always pad to a fixed length, so for them it only changes the
  speed (apart from bf16 rounding). For the other models, whether the padding affects training
  depends on whether the model masks it out; that has not been checked model by model, so keep
  `1` if you need the result to match the cache exactly.

With `pipeline_stages` above 1, the first and last stages both load the text encoders.
Switching between `true` and `false` keeps the latent caches; the text embedding cache is only
written and read with `true`.

### Aspect ratio buckets from the images

With `enable_ar_bucket = true` and no `ar_buckets`, the buckets are spaced evenly (in log space)
from `min_ar` to `max_ar`. Each of the three settings may be left out, on the `[[directory]]`
and at the top level:

- `min_ar` / `max_ar` left out: the narrowest / widest aspect ratio among that directory's
  images, found by the metadata scan. Each directory gets its own range.
- `num_ar_buckets` left out: the fewest buckets that crop no image by more than `max_ar_crop`
  of its width or height (default `0.1`, i.e. 10%). An image halfway between two neighbouring
  buckets, whose aspect ratios differ by a factor `step`, loses `1 - 1/sqrt(step)` of one side,
  so the count follows the range: from 0.5 to 2.0 at 10% that is 8 buckets, at 5% 15.

| Range | `max_ar_crop` | Buckets |
|---|---|---|
| 0.5 to 2.0 | 0.1 | 8 |
| 0.5 to 2.0 | 0.05 | 15 |
| 0.43 to 2.4 | 0.1 | 10 |

A smaller `max_ar_crop` crops less but splits the images over more buckets, each holding fewer;
with few images per bucket, batches are filled by `batch_fill_strategy`. `max_ar_crop` is refused
beside `num_ar_buckets`, `ar_buckets` or `size_buckets` on the same level, on a directory
without `enable_ar_bucket`, and at the top level when no directory uses it. A directory that sets `num_ar_buckets` or `max_ar_crop` overrides
both top-level values. The caching log prints the buckets each directory gets; `--metadata_only`
shows how many images went into each. A change of `max_ar_crop` regroups the dataset like any
other bucket change (below). With a range from the images, each caption setting of a directory
that keeps a different set of images (`require_non_latin_caption`, `skip_empty_caption`) may get
a slightly different range.

### Changing the buckets

Each image's bucket is fixed when its metadata is scanned, so changing `num_ar_buckets`,
`min_ar`/`max_ar`, `ar_buckets`, `frame_buckets` or `size_buckets` needs a new grouping. What it
costs depends on the mode, per model (each model has its own `cache/<model>`):

- `reuse_metadata_cache = true`: each bucket setting keeps its own snapshot and tensor caches
  (`profiles/<hash>`). The first run with a new setting scans the dataset once (only the media
  that no longer fit when this model has legacy metadata to start from); switching back to a
  setting used before (16 to 64, then back to 16) loads its metadata, latents and text
  embeddings as they were, with no scan and nothing re-encoded.
  Prefer this when you switch between bucket settings.
- `reuse_metadata_cache = false`: one grouping per caption setting. Every switch rescans the
  dataset, `--trust_cache` included (the buckets are recorded with the grouping, so an old one
  is never loaded for new buckets). Latents and text embeddings are kept per bucket and checked by
  content: switching back reuses those of buckets the other setting did not also use, and
  re-encodes the rest (for 16 and 64 aspect ratios from 0.5 to 2.0, the 0.5 and 2.0 buckets).
  To scan once for several models, scan with one and copy its metadata to the others with
  `tools/inherit_metadata_cache.py`, which requires the same buckets and caption settings.

## Extra
You can check out my [qlora-pipe](https://github.com/tdrussell/qlora-pipe) project, which is basically the same thing as this but for LLMs.
