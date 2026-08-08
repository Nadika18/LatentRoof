# Operator / block-level workload prediction

Predict latency of StableHLO operator and Transformer-block graphs on an unseen
GPU using only the graph and public hardware specifications.

## Layout

- `schedule_free_perf/` — GNN + hardware MLP + dual-peak roofline model
- `hardware/` — H200 / RTX PRO 6000 / GB10 resource JSONs
- `artifacts/` — mainline `hidden_dim=128` leave-one-GPU-out checkpoints + LOGO summary
- `DATA.md`, `DESIGN.md`, `LIMITATIONS.md`, `LITERATURE.md`

## Quick results (LOGO, $d=128$, mean OOD MAPE)

| Hold out | OOD MAPE |
|---|---:|
| GB10 | 12.2% |
| H200 | 14.3% |
| RTX | 17.6% |

## Train / eval

Requires a measurements JSONL + hardware dir (not shipped). Example:

```bash
PYTHONPATH=. python -m schedule_free_perf.cli train measurements.jsonl \
  --hardware-dir hardware --held-out-hardware nvidia_gb10 \
  --mode latent_physics --hidden-dim 128 --epochs 20 --batch-size 64 \
  --patience 5 --seed 29 --output artifacts/gb10_h128.pt
```
