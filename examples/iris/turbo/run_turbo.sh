#!/usr/bin/env bash
# Iris-3B Turbo from your own Iris model, end to end on one node: prompts, stage 1 (6 steps),
# stage 2 (4 steps) and the export:
#
#   TEACHER=/pvc/runs/iris_full/<run>/epoch6/ema DATASET=/pvc/configs/dataset.toml \
#       bash examples/iris/turbo/run_turbo.sh
#
# Relative paths in the variables below are taken from the repository root. Every step is
# skipped once its output exists and both stages run with --auto-resume, so the
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
#   BATCH1, BATCH2  default 64/NGPU and 16/NGPU. They override batch_size in the stage YAMLs.
#   KEEP_LAST default 2 (checkpoints kept per stage, ~84 GB each; 0 keeps all). 1 is refused:
#             after a node crash the newest checkpoint can be torn, and the one before it is needed.
#   HF_HOME   default $OUT/hf, so the text encoder is downloaded once onto the volume.
#   EXPORT_DTYPE  default bf16, as Iris's release.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/../../.."   # the repository root: tools/ and submodules/ are relative to it

: "${TEACHER:?set TEACHER to the model to distill}"
OUT=${OUT:-/pvc/turbo}
NGPU=${NGPU:-$(python -c 'import torch; print(torch.cuda.device_count())')}
KEEP_LAST=${KEEP_LAST:-2}
EXPORT_DTYPE=${EXPORT_DTYPE:-bf16}
export HF_HOME=${HF_HOME:-$OUT/hf}

if [ "$KEEP_LAST" = 1 ]; then
    echo "KEEP_LAST=1 leaves nothing to resume from if a crash tears the newest checkpoint; use 2 or more (0 keeps all)" >&2
    exit 1
fi
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

PROMPTS=$OUT/prompts.txt
STAGE1=$OUT/stage1
STAGE2=$OUT/stage2
EXPORT=$OUT/export
mkdir -p "$OUT"

newest_step() {  # newest step_*.pth in a checkpoints directory, or nothing
    ls -1 "$1"/step_*.pth 2>/dev/null | sort | tail -n 1 || true
}

# Run one command in the background and wait for it, passing SIGTERM/SIGINT on. As a pod's main
# process bash would otherwise ignore the TERM of an eviction and leave training running until
# the SIGKILL, possibly beside the replacement pod on the same volume.
child=
stopping=
forward() {
    stopping=1
    if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; fi
}
trap forward TERM INT
run() {
    local status=0
    "$@" &
    child=$!
    wait "$child" || status=$?
    while kill -0 "$child" 2>/dev/null; do  # interrupted by the trap: wait for the real exit
        status=0
        wait "$child" || status=$?
    done
    child=
    if [ -n "$stopping" ]; then
        echo "stopped by a signal" >&2
        exit 143
    fi
    return "$status"
}

if [ -s "$EXPORT/model.safetensors" ]; then
    echo "Turbo export already in $EXPORT; delete it (and $STAGE1, $STAGE2) to distill again."
    exit 0
fi

# 1. Prompts, once. The trainer checks their hash on every resume: never rebuild them mid-run.
if [ ! -s "$PROMPTS" ]; then
    args=()
    for d in ${DATASET:-}; do args+=(--dataset "$d"); done
    for e in ${EXTRA_PROMPTS:-}; do args+=(--extra-prompts "$e"); done
    if [ ${#args[@]} -eq 0 ]; then
        echo "$PROMPTS does not exist: set DATASET (and/or EXTRA_PROMPTS) to build it" >&2
        exit 1
    fi
    run python tools/iris_turbo_prompts.py "${args[@]}" --output "$PROMPTS" --no-progress
fi

# 2. The teacher's text encoder, before eight ranks would each fetch it. Only a convenience: if it
# fails (no network but a filled cache, say), the trainer still loads the encoder itself.
python - "$TEACHER" <<'PY' || echo "warning: could not pre-download the text encoder; the trainer will try itself" >&2
import sys
from pathlib import Path
import yaml
from huggingface_hub import snapshot_download

teacher = Path(sys.argv[1])
config = (teacher if teacher.is_dir() else teacher.parent) / 'config.yaml'
name = 'Qwen/Qwen3-VL-4B-Instruct'
if config.is_file():
    name = (yaml.safe_load(config.read_text()).get('text_encoder') or {}).get('pretrained', name)
if Path(name).exists() or name.startswith(('/', '.')):
    print(f'text encoder: local {name}')
else:
    print(f'text encoder: {name} -> {snapshot_download(name)}')
PY

run_stage() {  # run_stage <config> <work_dir> <batch_size> [overrides...]
    local config=$1 work_dir=$2 batch=$3
    shift 3
    run torchrun --standalone --nproc_per_node="$NGPU" tools/iris_turbo_distill.py \
        --auto-resume --keep-last "$KEEP_LAST" --config "$config" \
        teacher="$TEACHER" prompts_file="$PROMPTS" work_dir="$work_dir" batch_size="$batch" "$@"
}

# 3. Stage 1: 6 steps, 1600 student updates. It returns at once when it already finished, and is
# skipped once stage 2 has checkpoints, so deleting stage 1's checkpoints then never retrains it.
if [ -z "$(newest_step "$STAGE2/checkpoints")" ]; then
    run_stage "$HERE/stage1_6step.yaml" "$STAGE1" "$BATCH1"
fi

# 4. Stage 2: 4 steps from stage 1's final checkpoint. init_from is only read on stage 2's first
# start; afterwards the stage resumes from its own checkpoints.
stage1_final=$(newest_step "$STAGE1/checkpoints")
if [ -z "$stage1_final" ] && [ -z "$(newest_step "$STAGE2/checkpoints")" ]; then
    echo "no stage 1 checkpoint in $STAGE1/checkpoints to start stage 2 from" >&2
    exit 1
fi
# never the YAML's default path, which may belong to another run
run_stage "$HERE/stage2_4step.yaml" "$STAGE2" "$BATCH2" init_from="${stage1_final:-$STAGE1/checkpoints/step_00001600.pth}"

# 5. Export the stage-2 EMA. Written to a scratch directory and renamed, so a pod killed
# mid-export never leaves a half-written export that a restart would take as finished.
final=$(newest_step "$STAGE2/checkpoints")
rm -rf "$EXPORT.partial"
run python submodules/iris-3b/scripts/export_distilled.py "$final" "$EXPORT.partial" --dtype "$EXPORT_DTYPE"
rm -rf "$EXPORT"
mv "$EXPORT.partial" "$EXPORT"
echo "Turbo export: $EXPORT (sample with: python submodules/iris-3b/scripts/sample.py --checkpoint $EXPORT ...)"
