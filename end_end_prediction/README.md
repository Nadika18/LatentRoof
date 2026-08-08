# End-to-end prediction

Compose transformer-block predictions into GPT-2 / BERT backbone latency.
Same model family as `operator_workload_prediction`, trained with GPT/BERT-dimension
coverage. Repo overview and setup: [`../README.md`](../README.md).

## Build

```bash
# from LatentRoof/
python3 -m venv ../.venv && source ../.venv/bin/activate
pip install -U pip torch numpy matplotlib
cd end_end_prediction
pip install -e .
```

## Train / compose

```bash
bash scripts/run_train.sh
bash scripts/run_compose.sh          # amortize_launch
python plot_e2e_poster.py
```

## Layout

| Path | Contents |
|---|---|
| `schedule_free_perf/` | Model, train, eval |
| `hardware/` | Public GPU JSONs |
| `data/measurements.jsonl` | 7,864 training rows (+ graphs) |
| `data/e2e_groundtruth/` | Backbone labels (RTX + H200) |
| `data/e2e_coverage_graphs/` | Block + LN StableHLO for compose |
| `compose.py` | `N×block (+ LN)` modes |
| `scripts/` | `run_train.sh`, `run_compose.sh` |

## Compose modes

| Mode | Formula |
|---|---|
| `naive` | `N×block + LN` |
| `amortize_launch` (recommended) | `N×(block−launch) + launch + LN` |

| Test GPU | MAPE (`amortize_launch`) |
|---|---:|
| RTX PRO 6000 | 17.1% |
| H200 NVL | 9.5% |

More detail: [`../REPRODUCE.md`](../REPRODUCE.md) · optional remasure [`../collection/`](../collection/).
