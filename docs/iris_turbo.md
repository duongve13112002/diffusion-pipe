# Iris-3B Turbo: distilling your own Iris model

[Iris-3B Turbo](https://github.com/speridlabs/iris-3b#iris-3b-turbo) samples in 4 steps with no
CFG, each step a single forward pass. Iris made it from Iris-3B with
[TDM](https://arxiv.org/abs/2503.06674) distillation, and the trainer ships in the
`submodules/iris-3b` submodule. This page turns a model you trained with diffusion-pipe into a
Turbo version of itself, with the same recipe.

The distillation trainer is not a diffusion-pipe model type. It holds three copies of the network
(the frozen teacher, the student and an online "fake" score network) and reads only prompts, no
images. diffusion-pipe adds the pieces around it:

| File | What it is |
|---|---|
| `tools/iris_turbo_prompts.py` | Builds the prompts file from the captions of your training datasets. |
| `tools/iris_turbo_distill.py` | Launches Iris's trainer, with checks before training, safe restarts, checkpoint pruning and a longer process group timeout. |
| `examples/iris/turbo/stage1_6step.yaml`, `stage2_4step.yaml` | Iris's two stage configs laid out for 8 GPUs and a persistent volume. |
| `examples/iris/turbo/run_turbo.sh` | Prompts, both stages and the export in one restartable command. |
| `examples/iris/turbo/k8s_job.yaml` | A Kubernetes Job running that script on one 8-GPU node with a PVC. |

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

## Quick start

From the diffusion-pipe repository root, on one node with 8 GPUs:

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

## What Turbo trades away

From Iris's evaluation of its own Turbo: slightly better prompt following (GenEval, DPG-Bench),
weaker long rendered text, and noticeably less variety across seeds. Expect the same shape from a
distilled fine-tune, and keep the original model for work that needs variety or many steps.
