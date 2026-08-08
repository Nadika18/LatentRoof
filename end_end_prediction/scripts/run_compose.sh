#!/usr/bin/env bash
# Compose e2e predictions vs shipped ground truth. Run from end_end_prediction/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
export PYTHONPATH=.
CKPT="${CKPT:-artifacts/e2e_all_h128.pt}"
GRAPHS=data/e2e_coverage_graphs
MODE="${MODE:-amortize_launch}"

# Output names must match plot_e2e_poster.py: compose_<mode>_{rtx,h200}.json
declare -A GT_MAP=(
  [rtx]=data/e2e_groundtruth/rtx_pro_6000.json
  [h200]=data/e2e_groundtruth/h200.json
)

for tag in rtx h200; do
  out="artifacts/compose_${MODE}_${tag}.json"
  echo "===== compose $MODE on $tag ====="
  $PY compose.py \
    --checkpoint "$CKPT" \
    --groundtruth "${GT_MAP[$tag]}" \
    --coverage-graphs "$GRAPHS" \
    --compose-mode "$MODE" \
    --output "$out"
done
echo "COMPOSE_DONE"
