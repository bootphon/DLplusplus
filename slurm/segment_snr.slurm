#!/bin/bash
#SBATCH --job-name=segment_snr
#SBATCH --output=/home/%u/logs/dlplusplus/snr/segment_snr_%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --export=ALL
#SBATCH --time=04:00:00
#SBATCH --partition=erc-dupoux,gpu-p1

DATASET="$1"
shift 1  # remaining args passed through

if [ -z "$DATASET" ]; then
    echo "Usage: sbatch segment_snr.slurm <DATASET> [ADDITIONAL_ARGS...]"
    echo "Example: sbatch segment_snr.slurm seedlings_10"
    exit 1
fi

echo ""
echo "Per-Segment SNR & C50 (Brouhaha)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  Node: $(hostname)"
echo "Shard: ${SLURM_ARRAY_TASK_ID}/$((SLURM_ARRAY_TASK_COUNT-1))"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"

export POLARS_SKIP_CPU_CHECK=1

echo ""

mkdir -p "${HOME}/logs/dlplusplus/snr"

PYTHONUNBUFFERED=1 \
uv run python -m audio_pipeline.pipeline.segment_snr "$DATASET" \
    --array_id "$SLURM_ARRAY_TASK_ID" \
    --array_count "$SLURM_ARRAY_TASK_COUNT" \
    "$@"

EXIT_CODE=$?

echo ""
echo "Completed: $(date '+%H:%M:%S')  exit=$EXIT_CODE"
exit $EXIT_CODE
