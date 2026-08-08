# Operator / block-level workload prediction

Predict latency of StableHLO operator and Transformer-block graphs on an unseen
GPU using only the graph and public hardware specifications.

## Layout

- `schedule_free_perf/` — GNN + hardware MLP + dual-peak roofline model
- `hardware/` — H200 / RTX PRO 6000 / GB10 resource JSONs
- `data/measurements.jsonl` + `data/graphs/` — shipped corpus (no GPU needed)
- `artifacts/` — mainline `hidden_dim=128` LOGO checkpoints + summary
- `scripts/run_train_logo.sh` — reproduce leave-one-GPU-out training
- `DATA.md`, `DESIGN.md`, `LIMITATIONS.md`, `LITERATURE.md`

Full steps: repo-root [`REPRODUCE.md`](../REPRODUCE.md).  
Raw **XLA dump trees are not shipped** (labels are already inside `measurements.jsonl`).  
Optional remasure: [`collection/`](../collection/).

## Quick results (LOGO, $d=128$, mean OOD MAPE)

| Hold out | OOD MAPE |
|---|---:|
| GB10 | 12.2% |
| H200 | 14.3% |
| RTX | 17.6% |

## Train

```bash
pip install -e .
bash scripts/run_train_logo.sh
```
