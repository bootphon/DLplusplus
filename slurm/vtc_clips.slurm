#!/bin/bash
#SBATCH --job-name=vtc_clips
#SBATCH --output=/home/%u/logs/dlplusplus/vtc/vtc_clips_%A_%a.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=erc-dupoux,gpu-p1
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH --export=ALL

DATASET="$1"
MODEL_ROOT="${MODEL_ROOT:-$HOME/.cache/dlplusplus}"
shift 1  # remaining args passed through to vtc_on_clips.py

if [ -z "$DATASET" ]; then
    echo "Usage: sbatch slurm/vtc_clips.slurm <DATASET> [ADDITIONAL_ARGS...]"
    echo "Example: sbatch slurm/vtc_clips.slurm seedlings_10"
    exit 1
fi

echo ""
echo "VTC on Clips"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  Node: $(hostname)"
echo "Shard: ${SLURM_ARRAY_TASK_ID}/$((SLURM_ARRAY_TASK_COUNT-1))"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"

export POLARS_SKIP_CPU_CHECK=1

echo ""

PYTHONUNBUFFERED=1 \
uv run python -m audio_pipeline.analysis.vtc_on_clips "$DATASET" \
    --array_id "$SLURM_ARRAY_TASK_ID" \
    --array_count "$SLURM_ARRAY_TASK_COUNT" \
    --thresholds-path "${MODEL_ROOT}/vtc/thresholds/hp.toml" \
    "$@"

EXIT_CODE=$?

echo ""
echo "Completed: $(date '+%H:%M:%S')  exit=$EXIT_CODE"
exit $EXIT_CODE
