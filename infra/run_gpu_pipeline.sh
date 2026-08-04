#!/usr/bin/env bash
# The GPU-bound steps, run in order on the pod. data_prep.py, judge.py, and analyze.py
# do not need a GPU and are meant to run locally instead.
#
# Each stage is skipped if its expected output already exists locally -- filenames here are fixed
# (no Hydra-style parameterized paths to get wrong, unlike the Stolfo pipeline), so this is just a
# plain file-existence check per stage. generate.py additionally resumes at the ROW level internally
# (see generate.py's own docstring) since it's the single longest stage and losing partial progress
# there is the most expensive place to get this wrong.
#
# Set FORCE_RERUN=1 to bypass all skip checks below and redo everything regardless of what already exists.
set -euo pipefail

cd "$(dirname "$0")/../src"
RESULTS_DIR="../results"
FORCE_RERUN="${FORCE_RERUN:-0}"

if [ "$FORCE_RERUN" != "1" ] && [ -f "$RESULTS_DIR/const_steer_directions.pt" ] && [ -f "$RESULTS_DIR/const_steer_config.json" ]; then
    echo "== const_steer_directions.pt + config already exist, skipping steering_const.py (FORCE_RERUN=1 to redo) =="
else
    echo "== calibrating constant activation-steering direction/layer/coefficient on dev (steering on top of the caveman-style instruction) =="
    python3 -u steering_const.py
fi

if [ "$FORCE_RERUN" != "1" ] && [ -f "$RESULTS_DIR/psr_probe.pt" ]; then
    echo "== psr_probe.pt already exists, skipping steering_psr.py (FORCE_RERUN=1 to redo) =="
else
    echo "== training S-PSR probe (single layer) =="
    python3 -u steering_psr.py
fi

if [ "$FORCE_RERUN" != "1" ] && [ -f "$RESULTS_DIR/a_psr_probe.pt" ]; then
    echo "== a_psr_probe.pt already exists, skipping steering_a_psr.py (FORCE_RERUN=1 to redo) =="
else
    echo "== training A-PSR probe (all candidate layers, jointly) =="
    python3 -u steering_a_psr.py
fi

# Not stage-skipped the same way -- generate.py is internally resumable at the row level, and will
# just quickly confirm "already done" and exit fast if a previous run already completed everything.
echo "== generating all 8 conditions over the test split =="
python3 -u generate.py --split test

echo "== zipping results for manual Drive backup =="
python3 -u drive_utils.py
