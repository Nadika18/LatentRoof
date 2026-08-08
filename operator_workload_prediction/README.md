# Operator / block-level workload prediction

Leave-one-GPU-out latency prediction for StableHLO operator and Transformer-block
graphs. See the repo-root [`README.md`](../README.md) for overview, setup, and datasets.

## Build

```bash
# from LatentRoof/
python3 -m venv ../.venv && source ../.venv/bin/activate
pip install -U pip torch numpy
cd operator_workload_prediction
pip install -e .
```

## Train / evaluate

```bash
bash scripts/run_train_logo.sh

PYTHONPATH=. python -m schedule_free_perf.cli evaluate \
  data/measurements.jsonl \
  --hardware-dir hardware \
  --held-out-hardware nvidia_gb10 \
  artifacts/gb10_h128.pt
```

## Layout

| Path | Contents |
|---|---|
| `schedule_free_perf/` | GNN + hardware MLP + dual-peak roofline |
| `hardware/` | H200 / RTX PRO 6000 / GB10 JSON specs |
| `data/` | `measurements.jsonl` (6,778 rows) + StableHLO graphs |
| `artifacts/` | `hidden_dim=128` LOGO checkpoints |
| `scripts/run_train_logo.sh` | Three test-GPU folds |
| `DATA.md`, `DESIGN.md` | Protocol and model notes |

## Results (`d=128`)

| Test GPU | OOD MAPE |
|---|---:|
| GB10 | 12.2% |
| H200 | 14.3% |
| RTX PRO 6000 | 17.6% |

More detail: [`../REPRODUCE.md`](../REPRODUCE.md) · optional remasure [`../collection/`](../collection/).
