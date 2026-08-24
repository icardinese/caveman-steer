#!/usr/bin/env bash
# GPU-bound steps, run in order. Assumes Drive is already mounted from a notebook cell:
#     from google.colab import drive; drive.mount('/content/drive')
# and that infra/setup_colab.sh has already run once this session.
#
# Unlike the RunPod version, this backs up to Drive CONTINUOUSLY (every 5 min) via a background
# rsync loop, not just once at the end via a manual zip-and-drag step -- Colab disconnects
# unpredictably, and losing an overnight run to that is exactly the failure mode we're avoiding.
#
# Set FORCE_RERUN=1 to bypass all skip checks and redo everything.
# Set RUN_ALPHA_SWEEP=0 to skip the alpha sweep (it's what actually fills a 6-7hr overnight window --
# the default single PSR-Conceptor run is only ~1.5hrs on its own).
set -euo pipefail

cd "$(dirname "$0")/../src"
RESULTS_DIR="../results"
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_ALPHA_SWEEP="${RUN_ALPHA_SWEEP:-1}"
ALPHA_GRID=(2 4 8 16 32)

DRIVE_BACKUP_DIR="/content/drive/Shareddrives/Eric/LLM_STEER_BACKUP"
DRIVE_RESULTS="$DRIVE_BACKUP_DIR/results"
if [ ! -d "/content/drive/Shareddrives" ]; then
    echo "ERROR: Drive not mounted (or no Shared Drive access). Run from a notebook cell first:"
    echo "  from google.colab import drive; drive.mount('/content/drive')"
    exit 1
fi
mkdir -p "$DRIVE_RESULTS"

# Background sync loop: mirrors results/ into Drive every 5 min, for the entire lifetime of this
# script. Killed automatically on exit (normal or Ctrl-C) via the trap -- doesn't linger as an
# orphaned background process after the pipeline finishes. Pooled activations are excluded from the
# sync since they're a local speed cache (see train_psr_conceptor.py), not something worth Drive
# space or upload time -- they're cheap to regenerate from the responses cache, which IS synced.
sync_to_drive() {
    while true; do
        rsync -a --exclude 'pooled_activations_*' "$RESULTS_DIR/" "$DRIVE_RESULTS/" 2>/dev/null || true
        sleep 300
    done
}
sync_to_drive &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null || true' EXIT
echo "== background Drive sync started (pid $SYNC_PID, every 5 min) =="

if [ "$FORCE_RERUN" != "1" ] && [ -f "$RESULTS_DIR/const_steer_directions.pt" ] && [ -f "$RESULTS_DIR/const_steer_config.json" ]; then
    echo "== const_steer_directions.pt + config already exist, skipping steering_const.py =="
else
    echo "== calibrating constant activation-steering direction/layer/coefficient on dev =="
    python3 -u steering_const.py
fi

if [ "$FORCE_RERUN" != "1" ] && [ -f "$RESULTS_DIR/psr_probe.pt" ]; then
    echo "== psr_probe.pt already exists, skipping steering_psr.py =="
else
    echo "== training S-PSR probe (single layer, baseline) =="
    python3 -u steering_psr.py
fi

if [ "$FORCE_RERUN" != "1" ] && [ -f "$RESULTS_DIR/a_psr_probe.pt" ]; then
    echo "== a_psr_probe.pt already exists, skipping steering_a_psr.py =="
else
    echo "== training A-PSR probe (all candidate layers, jointly, baseline) =="
    python3 -u steering_a_psr.py
fi

echo "== training S-PSR-Conceptor gate (default alpha=4) =="
FORCE_RERUN="$FORCE_RERUN" python3 -u train_psr_conceptor.py

echo "== training paper-faithful PSR (same fidelity fixes, jointly-trained direction, no conceptor) =="
echo "   (isolates the direction choice against the conceptor run above -- same masking/MSE/reg either way)"
FORCE_RERUN="$FORCE_RERUN" python3 -u train_psr_proper.py

if [ "$RUN_ALPHA_SWEEP" = "1" ]; then
    echo "== sweeping conceptor aperture across alphas: ${ALPHA_GRID[*]} =="
    echo "   (responses + pooled activations are cached from the run above -- only alpha=4 pays the"
    echo "    full generation cost; each additional sweep point just retrains the gate, ~15-20 min each)"
    for alpha in "${ALPHA_GRID[@]}"; do
        echo "-- alpha=$alpha --"
        PSR_CONCEPTOR_ALPHA="$alpha" PSR_CONCEPTOR_OUT_TAG="_alpha${alpha}" FORCE_RERUN="$FORCE_RERUN" \
            python3 -u train_psr_conceptor.py
    done
    echo "== alpha sweep done. Compare results/psr_conceptor_train_log_alpha*.json for best dev MSE =="
fi

echo "== generating all 8 conditions over the test split =="
python3 -u generate.py --split test

echo "== zipping results and copying straight into Drive (no manual drag-and-drop needed) =="
python3 -u drive_utils.py
cp "$RESULTS_DIR/backup_results.zip" "$DRIVE_RESULTS/"

echo "== final sync =="
rsync -a --exclude 'pooled_activations_*' "$RESULTS_DIR/" "$DRIVE_RESULTS/"
echo "== done. Everything is in $DRIVE_RESULTS =="