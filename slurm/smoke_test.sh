#!/usr/bin/env bash
# smoke_test.sh — local end-to-end pipeline smoke test using test fixtures.
#
# Usage (from repo root):
#   bash slurm/smoke_test.sh           # VAD only — no GPU, no model download needed
#   bash slurm/smoke_test.sh --full    # all 5 steps — GPU + models required
#
# DLPP_WORKSPACE is set to a temp dir; cleaned up on exit regardless of outcome.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
FIXTURES="$REPO_ROOT/tests/fixtures"

FULL=0
for arg in "$@"; do [[ "$arg" == "--full" ]] && FULL=1; done

# ── workspace ────────────────────────────────────────────────────────────────
WORKSPACE="$(mktemp -d -t dlpp_smoke_XXXXXX)"
export DLPP_WORKSPACE="$WORKSPACE"

KEEP_WORKSPACE=0
trap '[[ $KEEP_WORKSPACE -eq 0 ]] && rm -rf "$WORKSPACE" || true' EXIT

DATASET="smoke"
echo "=== DL++ Smoke Test ==="
echo "Workspace : $WORKSPACE"
echo "Fixtures  : $FIXTURES"
[[ $FULL -eq 1 ]] && echo "Mode      : full (VAD + VTC + SNR + ESC + Package)" \
                  || echo "Mode      : CPU only (VAD); re-run with --full for all steps"
echo ""

# ── manifest ─────────────────────────────────────────────────────────────────
mkdir -p "$WORKSPACE/manifests"
MANIFEST="$WORKSPACE/manifests/${DATASET}.csv"
printf "path,uid,ext\n" > "$MANIFEST"
for f in "$FIXTURES"/*.wav; do
    uid="$(basename "$f" .wav)"
    printf "%s,%s,wav\n" "$f" "$uid" >> "$MANIFEST"
done
n_files="$(tail -n +2 "$MANIFEST" | wc -l | tr -d ' ')"
echo "Manifest  : $MANIFEST ($n_files files)"
echo ""

# ── step runner ───────────────────────────────────────────────────────────────
declare -A RESULTS

run_step() {
    local label="$1"; shift
    local log_file="$WORKSPACE/logs/${label}.log"
    mkdir -p "$WORKSPACE/logs"
    echo "--- ${label} ---"
    if (cd "$REPO_ROOT" && uv run python -m "$@") 2>&1 | tee "$log_file"; then
        RESULTS[$label]="PASS"
    else
        RESULTS[$label]="FAIL"
        KEEP_WORKSPACE=1
        echo ""
        echo "!!! ${label} FAILED — last 50 lines:"
        echo "------------------------------------------------------------"
        tail -n 50 "$log_file"
        echo "------------------------------------------------------------"
    fi
    echo ""
}

# ── pipeline steps ────────────────────────────────────────────────────────────
run_step VAD   audio_pipeline.pipeline.vad     "$DATASET" --workers 2

if [[ $FULL -eq 1 ]]; then
    run_step VTC     audio_pipeline.pipeline.vtc     "$DATASET"
    run_step SNR     audio_pipeline.pipeline.snr     "$DATASET"
    run_step ESC     audio_pipeline.pipeline.esc     "$DATASET"
    run_step Package audio_pipeline.pipeline.package "$DATASET"
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo "=== Summary ================================================================"
all_pass=1
for step in VAD VTC SNR ESC Package; do
    if [[ -v RESULTS[$step] ]]; then
        if [[ "${RESULTS[$step]}" == "PASS" ]]; then
            printf "  %-10s ✅  PASS\n" "$step"
        else
            printf "  %-10s ❌  %s\n" "$step" "${RESULTS[$step]}"
            all_pass=0
        fi
    else
        printf "  %-10s ⬜  SKIPPED  (re-run with --full)\n" "$step"
    fi
done
echo "============================================================================"

if [[ $KEEP_WORKSPACE -eq 1 ]]; then
    echo ""
    echo "Workspace kept for inspection: $WORKSPACE"
    echo "  Full logs : $WORKSPACE/logs/"
    echo "  Outputs   : $WORKSPACE/outputs/$DATASET/"
fi
echo "============================================================================"

[[ $all_pass -eq 1 ]] && exit 0 || exit 1
