#!/usr/bin/env bash
# End-to-end coverage training. Run from end_end_prediction/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
export PYTHONPATH=.
COMMON=(--hardware-dir hardware --mode latent_physics --hidden-dim 128 --message-steps 2
  --epochs 20 --batch-size 64 --learning-rate 1e-3 --auxiliary-weight 0.2
  --patience 5 --seed 29 --device "${DEVICE:-cpu}")
MANIFEST=data/measurements.jsonl

echo "===== [1/3] e2e_all_h128 (grouped split, no held-out) ====="
$PY -m schedule_free_perf.cli train "$MANIFEST" "${COMMON[@]}" \
  --output artifacts/e2e_all_h128.pt --history artifacts/e2e_all_h128_history.json

echo "===== [2/3] e2e_rtx_out_h128 ====="
$PY -m schedule_free_perf.cli train "$MANIFEST" "${COMMON[@]}" \
  --held-out-hardware nvidia_rtx_pro_6000 \
  --output artifacts/e2e_rtx_out_h128.pt --history artifacts/e2e_rtx_out_h128_history.json

echo "===== [3/3] e2e_h200_out_h128 ====="
$PY -m schedule_free_perf.cli train "$MANIFEST" "${COMMON[@]}" \
  --held-out-hardware nvidia_h200 \
  --output artifacts/e2e_h200_out_h128.pt --history artifacts/e2e_h200_out_h128_history.json

echo "E2E_TRAIN_DONE"
