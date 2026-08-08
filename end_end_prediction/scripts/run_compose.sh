#!/usr/bin/env bash
# Compose e2e predictions vs shipped ground truth. Run from end_end_prediction/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
export PYTHONPATH=.
CKPT="${CKPT:-artifacts/e2e_all_h128.pt}"
GRAPHS=data/e2e_coverage_graphs
MODE="${MODE:-amortize_launch}"

for gt in data/e2e_groundtruth/rtx_pro_6000.json data/e2e_groundtruth/h200.json; do
  tag=$(basename "$gt" .json)
  out="artifacts/compose_${MODE}_${tag}.json"
  echo "===== compose $MODE on $tag ====="
  $PY compose.py \
    --checkpoint "$CKPT" \
    --groundtruth "$gt" \
    --coverage-graphs "$GRAPHS" \
    --compose-mode "$MODE" \
    --output "$out"
done
echo "COMPOSE_DONE"
