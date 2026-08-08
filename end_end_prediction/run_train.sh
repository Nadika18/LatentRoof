#!/bin/bash
set -e
cd /home/nadika/Nadika/stable_hlo_tinkering/cursor_ablation_dual_peak_E
PY=../.venv_train/bin/python3
COMMON="--hardware-dir hardware --mode latent_physics --hidden-dim 128 --message-steps 2 \
  --epochs 20 --batch-size 64 --learning-rate 1e-3 --auxiliary-weight 0.2 \
  --patience 5 --seed 29 --device cpu"
echo "===== [1/3] e2e_all_h128 (no held-out, grouped split) ====="
$PY -m schedule_free_perf.cli train artifacts/manifest_E.json $COMMON \
  --output artifacts/e2e_all_h128.pt --history artifacts/e2e_all_h128_history.json
echo "===== [2/3] e2e_rtx_out_h128 (held out RTX) ====="
$PY -m schedule_free_perf.cli train artifacts/manifest_E.json $COMMON \
  --held-out-hardware nvidia_rtx_pro_6000 \
  --output artifacts/e2e_rtx_out_h128.pt --history artifacts/e2e_rtx_out_h128_history.json
echo "===== [3/3] e2e_h200_out_h128 (held out H200) ====="
$PY -m schedule_free_perf.cli train artifacts/manifest_E.json $COMMON \
  --held-out-hardware nvidia_h200 \
  --output artifacts/e2e_h200_out_h128.pt --history artifacts/e2e_h200_out_h128_history.json
echo "ALL_E_TRAIN_DONE"
