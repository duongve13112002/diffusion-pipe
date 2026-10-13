# Iris-3B Turbo: distilling your own Iris model

[Iris-3B Turbo](https://github.com/speridlabs/iris-3b#iris-3b-turbo) samples in 4 steps with no
CFG, each step a single forward pass. Iris made it from Iris-3B with
[TDM](https://arxiv.org/abs/2503.06674) distillation, and the trainer ships in the
`submodules/iris-3b` submodule. This page turns a model you trained with diffusion-pipe into a
Turbo version of itself, with the same recipe.

The distillation is not diffusion-pipe's training loop. It holds three copies of the network
(the frozen teacher, the student and an online "fake" score network) and reads only prompts, no
images. You still write it as a diffusion-pipe config and start it with `train.py`
([Configs for train.py](#configs-for-trainpy)); a `[turbo]` table makes `train.py` hand the run
to one of the trainers below. diffusion-pipe adds these pieces:

| File | What it is |
|---|---|
| `examples/iris/turbo/iris_3b_turbo_lora.toml`, `iris_3b_turbo_stage1.toml`, `iris_3b_turbo_stage2.toml` | diffusion-pipe configs for `train.py`: Turbo as a LoRA, and the two full stages. |
| `utils/iris_turbo.py` | Translates those configs for the trainers and writes the prompts from the dataset. |
| `tools/iris_turbo_prompts.py` | Builds the prompts file from the captions of your training datasets. |
| `tools/iris_turbo_distill.py` | Launches Iris's trainer, with checks before training, safe restarts, checkpoint pruning and a longer process group timeout. |
| `examples/iris/turbo/stage1_6step.yaml`, `stage2_4step.yaml` | Iris's two stage configs laid out for 8 GPUs and a persistent volume. |
| `examples/iris/turbo/run_turbo.sh` | Prompts, both stages and the export in one restartable command. |
| `examples/iris/turbo/k8s_job.yaml` | A Kubernetes Job running that script on one 8-GPU node with a PVC. |
| `tools/iris_turbo_lora.py` | The trainer for Turbo as a LoRA, on one GPU or several: see [Turbo as a LoRA](#turbo-as-a-lora). |

**Two ways.** The full distillation reproduces Iris's release and needs a node of 8 large GPUs
(H200-class). [Turbo as a LoRA](#turbo-as-a-lora) uses the same objective, but trains a rank-64
adapter on one frozen copy of the model. It runs on a single GPU, uses more GPUs if you have
them, and gives a ~230 MB adapter instead of a 6 GB model. Expect it to be somewhat behind the
full distillation in quality: an adapter has less room to change the model.

## What you need

- **The teacher.** An Iris-3B model at Iris's exported layout (`model.safetensors` +
  `config.yaml`):
  - **Full fine-tune:** a save directory of the run. If you trained with `ema_decay`, use its
    `ema/` directory: Iris's release was distilled from EMA weights, and an EMA teacher gives
    smoother targets. `examples/iris/iris_3b_full.toml` has the EMA options commented out
    (`ema_decay = 0.9999`); turn them on for a run you mean to distill.
  - **LoRA / LoKr:** merge it first,
    `python -m tools.iris_merge_adapter --model speridlabs/iris-3b --adapter <save dir> --output <dir>`
    (with EMA, pass the adapter's `ema/` directory as `--adapter`). If training changed `shift`,
    `shift_law` or `max_text_length`, pass `--shift`, `--shift-law` and `--max-text-length` too:
    they change what the teacher computes.
  - **The base Iris-3B** works too, if you only want Iris's own recipe with your prompts. Download
    it first (`hf download speridlabs/iris-3b --local-dir iris-3b --exclude "depth/*" "upscaler/*" "turbo/*"`):
    the teacher must be a local directory, not a repo id.
  - **Not** a Turbo export or a distillation checkpoint. The launcher refuses those. Fine-tuning
    `iris-3b/turbo` itself with diffusion-pipe works but erodes its few-step sampling, and
    diffusion-pipe warns when it loads one: fine-tune the base model and distill the result.
- **GPUs.** The recipe is sized for 8 × H200 (141 GB) on one node, with `strategy: fsdp2`. With
  `ddp` it needs about 160 GB per GPU at batch 8, more than an H200 has.
- **Host memory.** About 100 GB free for rank 0 when it saves (see [Memory](#memory)), so a
  pod limit of 400 GiB for 8 GPUs is comfortable.
- **Disk.** About 84 GB per checkpoint. With the default `--keep-last 2`, a stage holds at most
  three while it saves: about 250 GB during stage 1, and at the end of stage 2 about 420 GB
  (stage 1's last two and stage 2's three). The text encoder (`Qwen/Qwen3-VL-4B-Instruct`), the
  export and the prompts add about 25 GB.
- **Time.** Stage 1 is 1,600 student updates at a global batch of 64, stage 2 500 at 16. Each
  update runs the student over its few-step trajectory and the teacher and fake network
  on the result, and the fake network takes its own update: count on several times
  the cost of a plain 1024px fine-tuning step with the same batch.

## Configs for train.py

A Turbo run is a diffusion-pipe config with a `[turbo]` table, started like any other run:

```bash
# Turbo as a LoRA, one GPU or several
deepspeed --num_gpus=1 train.py --deepspeed --config examples/iris/turbo/iris_3b_turbo_lora.toml
deepspeed --num_gpus=8 train.py --deepspeed --config examples/iris/turbo/iris_3b_turbo_lora.toml
# full Turbo, Iris's two stages, 8 x H200
deepspeed --num_gpus=8 train.py --deepspeed --config examples/iris/turbo/iris_3b_turbo_stage1.toml
deepspeed --num_gpus=8 train.py --deepspeed --config examples/iris/turbo/iris_3b_turbo_stage2.toml
```

`torchrun --nproc_per_node=N train.py --config ...` works too. With an `[adapter]` table the run
is [Turbo as a LoRA](#turbo-as-a-lora) (`tools/iris_turbo_lora.py`). Without one it is full Turbo:
Iris's own trainer through `tools/iris_turbo_distill.py`, with its checks, `--auto-resume` and
checkpoint pruning. DeepSpeed itself is not used: these trainers do their own data parallelism
(gradient averaging for the LoRA, FSDP2 or DDP for full Turbo).

The keys keep their diffusion-pipe names where one exists:

| diffusion-pipe key | Turbo meaning |
|---|---|
| `output_dir` | The run directory: adapters, checkpoints, previews, `prompts.txt`. Full Turbo: Iris's `work_dir`. |
| `dataset` | The prompts: the captions of this `dataset.toml`, written once to `output_dir/prompts.txt` (see [Prompts](#prompts)). A list of files works too. |
| `micro_batch_size_per_gpu` | Prompts per GPU per micro-batch (`batch_size`). |
| `gradient_accumulation_steps` | LoRA only. Iris's full trainer has none: every update reads a fresh global batch. |
| `gradient_clipping` | `max_grad_norm`, for the student and the fake network. |
| `activation_checkpointing` | `true` = `'full'`, `false` = `'none'`, or `'selective_op'` / `'selective_layer'`. |
| `max_steps` | Student updates. Turbo has no epochs. |
| `save_every_n_steps` | How often to export an adapter and save a training checkpoint. |
| `steps_per_print` | Log every this many updates. |
| `save_dtype` | LoRA only: the dtype of the exported adapter files. |
| `[model] type`, `transformer_path` | `'iris'`, and the model to distill (the teacher). |
| `[model] dtype` | LoRA only: the frozen base the adapters sit on. Full Turbo keeps Iris's FP32 weights with bf16 autocast. |
| `[adapter] type`, `rank`, `alpha`, `dropout`, `target_modules` | The student's LoRA (`type = 'lora'` only). `fake_rank` and `fake_alpha` size the fake network's. The adapters train in float32 whatever `dtype` says. |
| `[optimizer] type`, `lr`, `betas`, `weight_decay` | AdamW (`type = 'adamw'`) for the student. `fake_lr` is the fake network's rate. |
| `[turbo]` | The distillation: `sigmas`, `cfg_scale`, `sigma_min`, `resolutions`, `fake_updates`, `ema_decay`, `seed`, `init_from`, `keep_last` (training checkpoints kept), `nccl_timeout_minutes`, and the prompt sources `prompts_file` and `extra_prompts`. LoRA only: `text_encoder`, `sample_every`, `sample_prompts`, `sample_size`, `sample_seed`. Full only: `strategy`, `init_weights`, `height`, `width`. |

Keys that only make sense for diffusion-pipe's own loop (`epochs`, `warmup_steps`, `eval_*`,
`blocks_to_swap`, other optimizers, other adapter types, wandb) are refused with a message
naming them, rather than ignored. So is a setting that belongs to the other kind of run.

**Prompts.** With `dataset`, rank 0 writes `output_dir/prompts.txt` from its captions before
training, with `tools/iris_turbo_prompts.py`'s defaults: caption augmentation off, `num_repeats`
weighting, shuffled with seed 0. `[turbo] extra_prompts` adds plain text files of prompts (one
per line), e.g. general prompts that keep the student close to the base model's range. The file
is written once: a restart reuses it, because a resume refuses a changed prompts file. To build
the prompts differently (`--dedupe`, `--ignore-num-repeats`), run the tool yourself and set
`[turbo] prompts_file` instead of `dataset`. Stage 2 points `prompts_file` at stage 1's
`prompts.txt` so both stages read the same prompts.

**Restarts.** Run the same command again. Both trainers continue from the newest readable
checkpoint in `output_dir` and exit at once when the run is finished; `--resume_from_checkpoint`
is not needed. A full Turbo stage writes the Iris YAML it ran with to
`output_dir/turbo_distill.yaml`; a LoRA run writes its settings to `output_dir/config.toml`.

The rest of this page describes the trainers in Iris's own terms (`batch_size`, `work_dir`,
`learning_rate`, `dmd.sigmas`): the table above gives the diffusion-pipe name of each. The YAML
configs, `run_turbo.sh` and the Kubernetes Job below run the same full Turbo recipe directly
with `tools/iris_turbo_distill.py`.

## Quick start

`run_turbo.sh` runs the whole full Turbo recipe in one restartable command: prompts, both stages
and the export. From the diffusion-pipe repository root, on one node with 8 GPUs:

```bash
TEACHER=/pvc/runs/iris_full/<run>/epoch6/ema \
DATASET=/pvc/configs/dataset.toml \
OUT=/pvc/turbo \
    bash examples/iris/turbo/run_turbo.sh
```

The result is `/pvc/turbo/export`: `model.safetensors` (bf16), `config.yaml` and
`distill.yaml`. Sample it with Iris's sampler, which reads the 4-step schedule from
`distill.yaml`:

```bash
python submodules/iris-3b/scripts/sample.py --checkpoint /pvc/turbo/export \
    --prompt "a red fox sleeping in fresh snow, golden hour"
```

The schedule is part of the model: `sample.py` refuses `--steps`, `--order`, `--cfg-scale`,
`--shift` and `--negative-prompt` for it. The prompt, `--height`, `--width` and `--seed` apply. It was distilled at about one megapixel, so stay
near the 7 training sizes (1024×1024, 752×1344, 896×1344, 896×1200 and their transposes).

`run_turbo.sh` runs these steps. Once `$OUT/export` exists it does nothing; before that it skips
each step whose output already exists:

1. Builds `$OUT/prompts.txt` from `DATASET` (see [Prompts](#prompts)).
2. Downloads the teacher's text encoder into `HF_HOME` (default `$OUT/hf`), so the eight ranks
   do not all fetch it at the same moment. If that fails it only warns. The encoder is the one
   in the teacher's `config.yaml` (`text_encoder.pretrained`); diffusion-pipe's
   `text_encoder_path` is not saved there. To use a local copy, set `pretrained` in the teacher's
   `config.yaml` to its directory before stage 1 (a resume refuses a changed teacher config).
3. Stage 1 into `$OUT/stage1`, then stage 2 into `$OUT/stage2` from stage 1's final checkpoint.
   Both use `--auto-resume`, so a stage that already finished returns at once and an interrupted
   one continues from its newest readable checkpoint. Once stage 2 has a checkpoint, stage 1 is
   not run again, even if its checkpoints were deleted.
4. Exports stage 2's final EMA into `$OUT/export`, through a scratch directory, so a pod killed
   mid-export never leaves a half-written export behind.

Its settings are environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `TEACHER` | required | The model to distill. Keep it the same for both stages and every restart. |
| `DATASET` | | Space-separated `dataset.toml` files of the teacher's training run. Needed only while `$OUT/prompts.txt` does not exist. |
| `EXTRA_PROMPTS` | | Text files with one prompt per line, mixed in as they are. |
| `OUT` | `/pvc/turbo` | Everything the run writes. |
| `NGPU` | all visible GPUs | GPUs on this node. |
| `BATCH1`, `BATCH2` | `64/NGPU`, `16/NGPU` | Per-GPU batch of each stage; they override `batch_size` in the YAMLs. The script stops if the recipe's global batch does not split over `NGPU`. |
| `KEEP_LAST` | `2` | Checkpoints kept per stage, `0` for all. `1` is refused (see [Restarts](#restarts-and-crashes)). |
| `HF_HOME` | `$OUT/hf` | Hugging Face cache, on the volume. Set `HF_TOKEN` too if your cache needs it. |
| `EXPORT_DTYPE` | `bf16` | As Iris's release. `fp32` keeps the full EMA. |

The script runs from the repository root wherever it is called from, and relative paths in these
variables are taken from there. It passes SIGTERM on to the trainer and stops, so an evicted pod
does not keep training. To start over, delete `$OUT/stage1`, `$OUT/stage2` and `$OUT/export`. To change the prompts,
delete `$OUT/prompts.txt` as well: a stage that resumes checks that the file has not changed.

### On Kubernetes

`examples/iris/turbo/k8s_job.yaml` runs `run_turbo.sh` as a Job on one 8-GPU node. Set the
image (diffusion-pipe with its submodules), the PVC name, `TEACHER` and `DATASET`, then

```bash
kubectl apply -f examples/iris/turbo/k8s_job.yaml
kubectl logs -f job/iris-turbo
```

It sets up what the run needs from the pod:

- **The volume** mounted at `/pvc`, the same path in every pod. Iris repoints
  `checkpoints/latest.pth` with an absolute symlink, which dangles under another mount path.
  `--auto-resume` does not use it, but anything you pass `latest.pth` to by hand does.
  The volume must accept symlinks and a rename over an existing file. The launcher tests both
  before training; mountpoint-s3 and SMB without `mfsymlinks` fail it.
- **`/dev/shm`** as a memory-backed `emptyDir`. Kubernetes' 64 MB default is too small for NCCL.
  Its contents count against the pod's memory limit.
- **Memory** 400 GiB. A limit too low for rank 0's save gets the pod OOM-killed at step 100, which
  looks like a random rank dying.
- **`restartPolicy: OnFailure`** and `backoffLimit: 6`. Each restart resumes where the last
  checkpoint left off, so at most 100 student updates are lost. A configuration error also
  restarts six times before the Job fails; check the first pod's log.
- **`podReplacementPolicy: Failed`** (Kubernetes 1.29+), so a replacement pod starts only after
  the old one has stopped, and two pods never write the same checkpoints.
- **The teacher and the dataset** must be reachable from the pod as well: on the same volume, or
  another mount.

## The configs

`examples/iris/turbo/stage1_6step.yaml` and `stage2_4step.yaml` are Iris's
`configs/iris3b/turbo_stage1_6step.yaml` and `turbo_stage2_4step.yaml` with only these changes:

| | Iris | Here |
|---|---|---|
| `work_dir` | `output/turbo/stage1_6step`, `output/turbo/stage2_4step` | `/pvc/turbo/stage1`, `/pvc/turbo/stage2` |
| `strategy` | `ddp` | `fsdp2` |
| stage 2 `batch_size` | 8 (on 2 GPUs) | 2 (on 8 GPUs); the global batch stays 16 |
| stage 2 `init_from` | `output/turbo/stage1_6step/...` | `/pvc/turbo/stage1/checkpoints/step_00001600.pth` |

The recipe itself:

| | Stage 1 | Stage 2 (released) |
|---|---|---|
| Sampling steps (σ) | 6 (1, 0.926, 0.833, 0.714, 0.556, 0.333) | 4 (1, 0.938, 0.833, 0.625) |
| Starts from | the teacher | student: stage 1's EMA; fake network: stage 1's |
| Student updates (`max_steps`) | 1,600 | 500 |
| Global batch (`batch_size` × GPUs) | 64 | 16 |
| Student / fake-network learning rate | 1e-6 / 5e-6 | 2.5e-7 / 2.5e-6 |
| EMA decay | 0.999 | 0.9995 |
| Prompts read | 204,800 | 16,000 |

Both stages: AdamW with betas (0, 0.999) and no weight decay, gradient clipping at 1.0, one
fake-network update per student update, FP32 weights under BF16 autocast, full activation
checkpointing, and CFG 3.5 (empty negative prompt) baked into the teacher's target. The released
Turbo is stage 2's EMA at step 500.

Every key can be overridden on the command line as `key=value`, as in Iris. Things to keep in
mind when you do:

- **Global batch.** The learning rates were set for 64 and 16. On another GPU count, change
  `batch_size` so that `batch_size` × GPUs stays the same: on 4 GPUs 16 and 4. A larger per-GPU
  batch needs more GPU memory for activations, so check the launcher's GPU memory warning.
- **`cfg_scale`.** The guidance the student learns to reproduce in one pass. 3.5 is Iris's usual
  CFG. If your fine-tune samples best at another scale, set `dmd.cfg_scale` to it in both stages.
- **`resolutions`.** Each global batch draws one bucket. Turbo samples best at the sizes it was
  distilled at.
- **`save_every`.** Each save writes about 84 GB and every rank waits for it. 100 loses at most
  100 updates on a crash.
- **Resuming** needs the same GPU count, the same prompts file content and the same teacher
  weights and `config.yaml` (Iris hashes them; the paths may move). Of the config, only paths (`teacher`,
  `prompts_file`, `work_dir`, `init_from`, `init_weights`), `strategy`, `max_steps`,
  `save_every`, `log_every`, the two learning rates and `max_grad_norm` may change; anything else,
  `batch_size` included, is refused. To continue a finished stage for longer, resume it with a
  higher `max_steps`.
- **Paths in overrides** are parsed as YAML values. A path with ` #` is cut there, and one with
  `: ` or starting with `*`, `&` or `@` fails. Plain spaces are fine.

## Prompts

Distillation reads only prompts: one per line, UTF-8, read in order and cycled. Each student
update and each fake-network update takes a fresh global batch. Build them from the captions the
teacher was trained on, so the student learns the distribution the teacher knows:

```bash
python tools/iris_turbo_prompts.py --dataset dataset.toml --output /pvc/turbo/prompts.txt
```

- Each prompt is the caption as training fed it to the text encoder: `caption_prefix` added, the
  `prefix_tag_caption` marker stripped, `enable_remove_non_latin` applied, `captions.json` and
  `multiline_captions` resolved.
- There is no tag shuffle, tag dropout or caption dropout. Turbo learns guidance from the
  teacher's CFG, not from empty prompts.
- A multi-line caption becomes one line. Empty captions and captions that are only their
  `caption_prefix` are dropped and counted.
- Each directory is weighted by its `num_repeats` (`--ignore-num-repeats` turns that off), not
  by `cache_shuffle_num`, resolutions or `caption_sampling`.
- The result is shuffled with a fixed seed (`--seed`), because the trainer reads the file in
  order and dataset order would give each batch prompts from one directory.
- `--extra-prompts general.txt` mixes in other prompts, one per line. General prompts keep the
  student closer to the base model's range; without them, it is distilled only on your domain.
- `--dedupe` keeps one copy of identical prompts.
- The file is written to a temporary name and renamed into place, so an interrupted run never
  leaves a truncated file.

Iris's prompt sets were large (about 1M per stage). A fine-tune's captions are usually far
fewer and get cycled many times; the tool prints how many passes each stage makes over the
file. Repetition is less harmful here than in ordinary fine-tuning, since every pass draws new
noise and the teacher provides the target, but a few thousand varied prompts serve better than a
few hundred.

Do not rebuild the file during a run: a resume checks its hash and stops if it changed.

## Running the stages by hand

`run_turbo.sh` is a thin wrapper. The same run step by step, from the repository root:

```bash
torchrun --standalone --nproc_per_node=8 tools/iris_turbo_distill.py --auto-resume \
    --config examples/iris/turbo/stage1_6step.yaml \
    teacher=/pvc/runs/iris_full/<run>/epoch6/ema prompts_file=/pvc/turbo/prompts.txt
torchrun --standalone --nproc_per_node=8 tools/iris_turbo_distill.py --auto-resume \
    --config examples/iris/turbo/stage2_4step.yaml \
    teacher=/pvc/runs/iris_full/<run>/epoch6/ema prompts_file=/pvc/turbo/prompts.txt \
    init_from=/pvc/turbo/stage1/checkpoints/step_00001600.pth
python submodules/iris-3b/scripts/export_distilled.py \
    /pvc/turbo/stage2/checkpoints/step_00000500.pth /pvc/turbo/export --dtype bf16
```

- Both stages use the same `teacher`.
- `init_from` is only read when stage 2 starts. On a resume it is ignored.
- `tools/iris_turbo_distill.py` also accepts Iris's own configs
  (`submodules/iris-3b/configs/iris3b/turbo_*.yaml`) with the same overrides.

Launcher options:

| Option | Default | Meaning |
|---|---|---|
| `--auto-resume` | off | Continue from the newest readable `step_*.pth` in `work_dir`. Without it, a `work_dir` with checkpoints is refused rather than restarted from step 0. |
| `--keep-last N` | 2 | Delete older `step_*.pth` after each save, never the one `latest.pth` points to. `0` keeps all. |
| `--nccl-timeout-minutes` | 120 | Process group timeout. The other ranks wait while rank 0 writes each checkpoint. |
| `--allow-cudnn-attention` | off | Keep PyTorch's cuDNN attention kernel, whose backward returned NaN on H200 / torch 2.12 / cuDNN 9.20. |
| `--skip-write-test` | off | Skip the timed 1 GiB write into `work_dir`. |
| `--skip-preflight` | off | Run the trainer without any of the checks below. Not allowed with `--auto-resume`. |

## What the launcher checks

On rank 0, before any weights load. A failed check stops every rank with its reason:

- The teacher is a text-to-image Iris model: not a Turbo export, a distillation checkpoint or a
  student, and its files are all there.
- The teacher's `attn_backend` is `sdpa`. Iris applies a forced kernel to the masked text adapter
  too, and it fails there.
- `prompts_file` is non-empty UTF-8.
- `init_from` and `resume_from` are readable distillation checkpoints. A resume matches the GPU
  count and the prompts file, and no newer checkpoint sits beside the one resumed.
- A fresh start does not land on a `work_dir` that already has checkpoints.
- The checkpoint directory accepts a symlink and a rename over an existing file.

These only warn:

- how long a checkpoint write will take, from the timed 1 GiB write;
- free disk space;
- the memory limit against rank 0's save;
- the size of `/dev/shm`;
- GPU memory against the strategy and batch size.

It also removes scratch files left by an interrupted save and a torn empty-prompt cache
(`output/pretrained_models/null_embed_*.pth` under the working directory).

## Monitoring

Rank 0 prints one JSON line per `log_every` (10) student updates:

- `generator_loss` and `generator_grad_norm`: the student.
- `fake_loss` and `fake_grad_norm`: the fake network.
- `step`, `fake_updates` and `prompt_cursor`.

Iris's trainer stops with an error on a non-finite loss. The first line, `distill_start`, shows
the world size, strategy, global batch, sigmas and the step it started or resumed from. Check
there that `global_batch` is 64 (stage 1) or 16 (stage 2).

The losses of a distribution-matching run do not fall like a fine-tuning loss. The student and
the fake network chase each other, so watch for spikes and NaN rather than a downward trend. To
judge quality, sample a checkpoint along the way. `scripts/sample.py` takes a training `.pth`
directly, but treats it as an ordinary model; export it first to sample with the few-step
schedule. Use the newest `step_*.pth` (with `--keep-last 2` older ones are deleted as training
goes) and a GPU the training is not using:

```bash
python submodules/iris-3b/scripts/export_distilled.py /pvc/turbo/stage1/checkpoints/step_00000800.pth \
    /tmp/turbo_s1_800 --dtype bf16
python submodules/iris-3b/scripts/sample.py --checkpoint /tmp/turbo_s1_800 --txt-file my_eval_prompts.txt
```

A stage-1 export samples with stage 1's 6 steps.

## Restarts and crashes

The launcher never restarts a stage from step 0 over existing checkpoints, which is what Iris
does when a pod restarts with the same command. With `--auto-resume`:

- It continues from the newest `step_*.pth` that loads. It does not rely on `latest.pth`, which
  lags one save behind if a pod dies at the wrong moment.
- A checkpoint left incomplete by a node crash is renamed to `*.torn` and skipped. Iris writes
  checkpoints without fsync, so after a node crash the newest one can be incomplete. This is why
  `--keep-last` should stay at 2 or more on a network volume. The check catches a truncated or
  unreadable file, not one of full length whose data was lost; if a run goes wrong right after
  a node crash, resume from the previous checkpoint by hand.
- A stage that already reached `max_steps` exits at once with status 0.

So the same command is right for the first start and every restart. Without `--auto-resume`,
pass `resume_from=<work_dir>/checkpoints/step_XXXXXXXX.pth` yourself.

## Memory

- **GPU.** `strategy: fsdp2` shards the student, the fake network, the EMA and both optimizer
  states across the GPUs; the frozen teacher stays whole on every GPU. That fits 8 × H200 at the recipe's
  batch. With `ddp` every GPU holds everything (about 160 GB at batch 8). Shard rather than lower
  the global batch.
- **Host.** With `fsdp2`, each save gathers the whole checkpoint (about 84 GB: the FP32 student,
  EMA, fake network and both AdamW states) into rank 0's CPU memory before writing it. Together
  with each process's own baseline that is around 100 GB on rank 0.
- **Disk.** Iris keeps every checkpoint: 16 saves, about 1.3 TB, for stage 1. `--keep-last 2`
  keeps two plus the one being written. Stage 2 needs stage 1's final checkpoint, which is always
  kept. Once stage 2 has written its first checkpoint, stage 1's checkpoints can go.

## Turbo as a LoRA

`tools/iris_turbo_lora.py` distills with Iris's TDM objective and few-step rollout
(`iris3b.distill.objective` and `sampler`, the code behind the released Turbo), but the student
and the fake score network are LoRA adapters on **one** frozen copy of the teacher:

| Adapter | Acts as |
|---|---|
| none (adapters off) | the teacher: the model you distill, frozen |
| `student` | the few-step student, trained against the teacher's guided target |
| `fake` | the fake score network, trained on the student's samples |
| `ema` | the moving average of the student, which is what gets exported and previewed |

The base weights are held once, in bf16 by default. Teacher and fake share them, so the lower
precision cannot bias the teacher-minus-fake difference the student learns from. The adapters
and their optimizer states stay in float32.

### Running it

Copy `examples/iris/turbo/iris_3b_turbo_lora.toml`, set `output_dir`, `dataset` and
`[model] transformer_path`, then

```bash
deepspeed --num_gpus=1 train.py --deepspeed --config my_turbo_lora.toml    # one GPU
deepspeed --num_gpus=4 train.py --deepspeed --config my_turbo_lora.toml    # 4 GPUs
```

The tool takes the same config directly, with `--set` overrides:
`torchrun --standalone --nproc_per_node=4 tools/iris_turbo_lora.py --config my_turbo_lora.toml --set max_steps=3000`
(a dotted key reaches a table: `--set adapter.rank=32`, `--set turbo.cfg_scale=4.0`).

- **Restarts.** Run the same command again after any interruption. It continues from the newest
  readable checkpoint in `output_dir/checkpoints` and exits at once when the run is finished. To
  train longer, raise `max_steps` and run it again.
  - A checkpoint that does not load (cut off by a crash) is renamed `*.torn` and the one before is
    used. If none loads, the run stops rather than starting over at step 0; delete the `*.torn`
    files if starting over is what you want. A read error from the volume stops the run without
    renaming anything.
  - A start at step 0 in an `output_dir` that already holds adapters moves them to
    `adapters.before-<time>`, so an earlier run's adapters are not mixed with this one's.
  - An unreadable empty-prompt cache (`output_dir/cache`) is computed again.
- **What a resume may change.** Paths, `max_steps`, the learning rates (`lr`, `fake_lr`),
  `gradient_clipping`, saving, logging and preview settings, `text_encoder`,
  `activation_checkpointing` and `nccl_timeout_minutes`. Anything else,
  the GPU count, the prompts file content or the teacher's weights is refused: use a new
  `output_dir`.
- **Warm start.** `[turbo] init_from` takes a checkpoint of another LoRA run: its EMA student and
  its fake network, with fresh optimizers. Use it to continue at a lower learning rate after a run
  that started to degrade, or for a 6-step stage followed by a 4-step one, as Iris did. It warns
  when that run distilled a different teacher.

### Several GPUs

Every GPU holds the whole model. After each update the adapters' gradients are averaged across
the GPUs, so the replicas stay identical and the global batch is
`micro_batch_size_per_gpu` × `gradient_accumulation_steps` × GPUs. The learning rates are not scaled for you.
More GPUs give a less noisy update, and the same number of `max_steps` finishes sooner. A
resume needs the same GPU count.

`gradient_accumulation_steps` gives a single GPU the same larger batch at the cost of time.

### Memory

These are estimates; the tool has not been profiled on every card.

| | |
|---|---|
| Base model, bf16 (the embedders and pixel head stay float32, ~575M parameters) | ~7 GB |
| Text encoder (Qwen3-VL-4B, bf16), `text_encoder = 'gpu'` | ~8 GB |
| Student, fake and EMA adapters at rank 64, with gradients and AdamW states | ~4 GB |
| Several GPUs: the flat buffer the gradients are averaged in | ~0.5 GB |
| Activations at 1024², batch 1, `activation_checkpointing = 'full'` | several GB |

That is roughly 25-29 GB with the text encoder on the GPU (a 32 GB card or larger) and roughly
17-21 GB with `[turbo] text_encoder = 'offload'`, which keeps the encoder in CPU memory and moves it to
the GPU once per update. A community DMD2 LoRA of the same size measured ~18 GB on an RTX 5090
with its text encoder on another GPU. A lower `[adapter] rank` saves a little; `[model] dtype =
'float32'` adds ~5 GB. Saves copy the adapters to host memory, so they need no extra GPU memory.

### Output and using the adapter

```
output_dir/
  adapters/step_00000100/        adapter_model.safetensors, adapter_config.json, distill.yaml
  checkpoints/step_00000100.pt   training state (adapters, EMA, optimizers, prompt cursor, RNG)
  samples/step_00000100/         previews of sample_prompts with the EMA student
  config.toml                    the settings of the run
```

- The adapter is the EMA student in diffusion-pipe's LoRA format, bf16 by default (~230 MB at
  rank 64). Each one is written to a scratch name and renamed, before the training checkpoint
  of the same step, so neither is ever half-written.
- `distill.yaml` holds the few-step schedule and records the teacher it was distilled from.
- Training checkpoints are about 3 GB at rank 64; `keep_last` (default 2) keeps the newest.
  Every exported adapter is kept.

Merge an adapter into **the same model it was distilled from** and sample with Iris's sampler,
which reads `distill.yaml` from the merged directory:

```bash
python -m tools.iris_merge_adapter --model /data/iris_full/my-run/epoch6/ema \
    --adapter /data/turbo_lora/run1/adapters/step_00002000 --output /data/my-iris-turbo-lora
python submodules/iris-3b/scripts/sample.py --checkpoint /data/my-iris-turbo-lora --prompt "a red fox in snow"
```

The merge tool copies the adapter's `distill.yaml`, and warns if `--model` is not the teacher
recorded there. An adapter distilled from your fine-tune does not transfer to another model.

### Settings

`examples/iris/turbo/iris_3b_turbo_lora.toml` explains every key and sets the defaults. They are
a starting point, not a tuned recipe:

| Key | Default | Note |
|---|---|---|
| `[turbo] sigmas` | `[1.0, 0.938, 0.833, 0.625]` | Iris's 4-step Turbo schedule. Six steps (`[1.0, 0.926, 0.833, 0.714, 0.556, 0.333]`) are easier to learn. |
| `[turbo] cfg_scale` | 3.5 | The guidance baked into the student. Use the scale your model samples best at. |
| `[optimizer] lr` / `fake_lr` | 2e-5 / 1e-4 | The fake network learns 5x faster; Iris's recipe uses 5x and 10x. A community DMD2 LoRA collapsed at 5e-5 and was stable at 2.5e-5. |
| `[turbo] fake_updates` | 2 | Fake updates per student update. The fake network must keep up with the student. |
| `[turbo] ema_decay` | 0.99 | Per student update. The exported adapter is the EMA. |
| `max_steps` | 2000 | Student updates. Pick the adapter by its previews, not the last one by default. |
| `[adapter] rank` | 64 | For both the student and (`fake_rank`) the fake network. |
| `[adapter] target_modules` | every linear layer but the text adapter | Class names or top-level modules of `IrisDiT`. |

**Watch the previews.** Distillation losses do not show quality: the student and the fake
network chase each other. `samples/step_00000000` is the teacher sampled in the student's 4
steps, the starting point. If later previews lose contrast or dissolve into texture, take an
earlier adapter, and continue from its checkpoint with `[turbo] init_from` at a lower `lr`.

The log is one JSON line per `steps_per_print` updates: `generator_loss`, `fake_loss`, both gradient
norms, `dm_rms` (the size of the teacher-minus-fake signal), `cfg_rms`, seconds per step and an
ETA. A non-finite loss or gradient stops the run before any update is applied; lower the learning
rates before running it again.

## What Turbo trades away

From Iris's evaluation of its own Turbo: slightly better prompt following (GenEval, DPG-Bench),
weaker long rendered text, and noticeably less variety across seeds. Expect the same shape from a
distilled fine-tune, and keep the original model for work that needs variety or many steps.
