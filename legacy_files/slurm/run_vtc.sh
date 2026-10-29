#!/bin/bash
# Run VTC inference only — mirrors how pipeline.sh submits it.
# Usage:
#   bash slurm/run_vtc.sh seedlings_small

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

DATASET="${1:?Usage: bash slurm/run_vtc.sh DATASET [--batch-size N] [--array-count N] [--sample N]}"
shift

BATCH_SIZE=128
ARRAY_COUNT=3
EXTRA_ARGS=""

mkdir -p "${HOME}/logs/dlplusplus/vtc"

VTC_ARRAY="0-$((ARRAY_COUNT - 1))"

VTC_JOB=$(sbatch --parsable \
    --array="${VTC_ARRAY}" \
    slurm/vtc.slurm "$DATASET" \
        --batch-size "$BATCH_SIZE" \
        $EXTRA_ARGS)

echo "VTC job : $VTC_JOB  (array ${VTC_ARRAY}, batch=${BATCH_SIZE})"
echo "Logs    : ${HOME}/logs/dlplusplus/vtc/vtc_${VTC_JOB}_*.out"
echo "Monitor : squeue -u \$USER"
echo "Cancel  : scancel $VTC_JOB"
