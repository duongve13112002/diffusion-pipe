#!/usr/bin/env bash
# Iris-3B Turbo from your own Iris model, end to end on one node: prompts, stage 1 (6 steps),
# stage 2 (4 steps) and the export. Run it from the diffusion-pipe repository root:
#
#   TEACHER=/pvc/runs/iris_full/<run>/epoch6/ema DATASET=/pvc/configs/dataset.toml \
#       bash examples/iris/turbo/run_turbo.sh
#
# Every step is skipped once its output exists and both stages run with --auto-resume, so the
# same command is right for the first start and for every restart of a pod (see k8s_job.yaml).
#
#   TEACHER   required. The fine-tune to distill: a diffusion-pipe full fine-tune save directory
#             (its ema/ directory if you trained with ema_decay) or a LoRA merged with
#             tools.iris_merge_adapter. Keep it the same for both stages and every restart.
#   DATASET   dataset.toml(s) the teacher was trained on, space separated. Needed only while
#             $OUT/prompts.txt does not exist yet; or put your own prompts file there.
#   EXTRA_PROMPTS  optional text file(s), one prompt per line, mixed into the prompts.
#   OUT       default /pvc/turbo. Holds prompts.txt, stage1/, stage2/ and export/.
#   NGPU      default: every visible GPU. The configs keep the recipe's global batch on 8 GPUs;
#             on another count set BATCH1/BATCH2 so that BATCH x NGPU is 64 and 16.
#   BATCH1, BATCH2  default 64/NGPU and 16/NGPU.
#   KEEP_LAST default 2 (checkpoints kept per stage, ~84 GB each). Never 1 on a network volume.
#   HF_HOME   default $OUT/hf, so the text encoder is downloaded once onto the volume.
#   EXPORT_DTYPE  default bf16, as Iris's release.
set -euo pipefail

: "${TEACHER:?set TEACHER to the model to distill}"
OUT=${OUT:-/pvc/turbo}
NGPU=${NGPU:-$(python -c 'import torch; print(torch.cuda.device_count())')}
KEEP_LAST=${KEEP_LAST:-2}
EXPORT_DTYPE=${EXPORT_DTYPE:-bf16}
export HF_HOME=${HF_HOME:-$OUT/hf}

if [ "$NGPU" -lt 1 ]; then
    echo "no GPU visible" >&2
    exit 1
fi
for stage_batch in "64 BATCH1" "16 BATCH2"; do
    set -- $stage_batch
    if [ -z "${!2:-}" ]; then
        if [ $(( $1 % NGPU )) -ne 0 ]; then
            echo "a global batch of $1 does not split over $NGPU GPUs; set $2 (per GPU) yourself" >&2
            exit 1
        fi
        printf -v "$2" '%d' $(( $1 / NGPU ))
    fi
done

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROMPTS=$OUT/prompts.txt
mkdir -p "$OUT"

newest_step() {  # newest step_*.pth in a checkpoints directory, or nothing
    ls -1 "$1"/step_*.pth 2>/dev/null | sort | tail -n 1 || true
}

# 1. Prompts, once. The trainer checks their hash on every resume: never rebuild them mid-run.
if [ ! -s "$PROMPTS" ]; then
    args=()
    for d in ${DATASET:-}; do args+=(--dataset "$d"); done
    for e in ${EXTRA_PROMPTS:-}; do args+=(--extra-prompts "$e"); done
    if [ ${#args[@]} -eq 0 ]; then
        echo "$PROMPTS does not exist: set DATASET (and/or EXTRA_PROMPTS) to build it" >&2
        exit 1
    fi
    python tools/iris_turbo_prompts.py "${args[@]}" --output "$PROMPTS" --no-progress
fi

# 2. The teacher's text encoder, once, before eight ranks would each fetch it.
python - "$TEACHER" <<'PY'
import sys
from pathlib import Path
import yaml
from huggingface_hub import snapshot_download

teacher = Path(sys.argv[1])
config = (teacher if teacher.is_dir() else teacher.parent) / 'config.yaml'
name = 'Qwen/Qwen3-VL-4B-Instruct'
if config.is_file():
    name = (yaml.safe_load(config.read_text()).get('text_encoder') or {}).get('pretrained', name)
if Path(name).exists():
    print(f'text encoder: local {name}')
else:
    print(f'text encoder: {name} -> {snapshot_download(name)}')
PY

run_stage() {  # run_stage <config> <work_dir> <batch_size> [overrides...]
    local config=$1 work_dir=$2 batch=$3
    shift 3
    torchrun --standalone --nproc_per_node="$NGPU" tools/iris_turbo_distill.py \
        --auto-resume --keep-last "$KEEP_LAST" --config "$config" \
        teacher="$TEACHER" prompts_file="$PROMPTS" work_dir="$work_dir" batch_size="$batch" "$@"
}

# 3. Stage 1: 6 steps, 1600 student updates. Returns at once when it already finished.
run_stage "$HERE/stage1_6step.yaml" "$OUT/stage1" "$BATCH1"

# 4. Stage 2: 4 steps from stage 1's final checkpoint. init_from only matters on the first
# start; on a restart the stage resumes from its own checkpoints.
stage2=()
stage1_final=$(newest_step "$OUT/stage1/checkpoints")
if [ -n "$stage1_final" ]; then
    stage2+=(init_from="$stage1_final")
elif [ -z "$(newest_step "$OUT/stage2/checkpoints")" ]; then
    echo "no stage 1 checkpoint in $OUT/stage1/checkpoints to start stage 2 from" >&2
    exit 1
fi
run_stage "$HERE/stage2_4step.yaml" "$OUT/stage2" "$BATCH2" ${stage2[@]+"${stage2[@]}"}

# 5. Export the stage-2 EMA. Written to a scratch directory and renamed, so a pod killed
# mid-export never leaves a half-written export that this step would then skip.
if [ ! -s "$OUT/export/model.safetensors" ]; then
    final=$(newest_step "$OUT/stage2/checkpoints")
    rm -rf "$OUT/export.partial"
    python submodules/iris-3b/scripts/export_distilled.py "$final" "$OUT/export.partial" --dtype "$EXPORT_DTYPE"
    rm -rf "$OUT/export"
    mv "$OUT/export.partial" "$OUT/export"
fi
echo "Turbo export: $OUT/export (sample with: python submodules/iris-3b/scripts/sample.py --checkpoint $OUT/export ...)"
