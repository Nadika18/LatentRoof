#!/usr/bin/env bash
# Leave-one-GPU-out training (operator / block level). Run from operator_workload_prediction/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
export PYTHONPATH=.
COMMON=(--hardware-dir hardware --mode latent_physics --hidden-dim 128 --message-steps 2
  --epochs 20 --batch-size 64 --learning-rate 1e-3 --auxiliary-weight 0.2
  --patience 5 --seed 29 --device "${DEVICE:-cpu}")
MANIFEST=data/measurements.jsonl

echo "===== hold out GB10 ====="
$PY -m schedule_free_perf.cli train "$MANIFEST" "${COMMON[@]}" \
  --held-out-hardware nvidia_gb10 \
  --output artifacts/gb10_h128.pt --history artifacts/gb10_h128_history.json

echo "===== hold out H200 ====="
$PY -m schedule_free_perf.cli train "$MANIFEST" "${COMMON[@]}" \
  --held-out-hardware nvidia_h200 \
  --output artifacts/h200_h128.pt --history artifacts/h200_h128_history.json

echo "===== hold out RTX PRO 6000 ====="
$PY -m schedule_free_perf.cli train "$MANIFEST" "${COMMON[@]}" \
  --held-out-hardware nvidia_rtx_pro_6000 \
  --output artifacts/rtx_h128.pt --history artifacts/rtx_h128_history.json

echo "LOGO_TRAIN_DONE"
