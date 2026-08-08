# End-to-end prediction

Compose StableHLO transformer-block latency predictions into full GPT-2 / BERT
backbone latency, then grade against measured ground truth.

Architecture matches `operator_workload_prediction` (GNN + hardware MLP +
dual-peak roofline, `hidden_dim=128`). Training adds GPT/BERT-dimension coverage
blocks so those shapes are in-distribution.

## Layout

- `schedule_free_perf/` — model, training, eval
- `hardware/` — public GPU resource JSONs
- `data/measurements.jsonl` + `data/graphs/` — training corpus (no GPU needed)
- `data/e2e_groundtruth/` — backbone latency labels (RTX + H200)
- `data/e2e_coverage_graphs/` — StableHLO used by `compose.py`
- `compose.py`, `plot_e2e_poster.py`
- `scripts/run_train.sh`, `scripts/run_compose.sh`
- `artifacts/` — checkpoints, compose JSON, poster PNGs

See also the repo-root [`REPRODUCE.md`](../REPRODUCE.md) and [`collection/`](../collection/).

## Compose modes

| Mode | Formula |
|---|---|
| `naive` | `N×block + LN` |
| `amortize_launch` (recommended) | `N×(block−launch) + launch + LN` |

Example results (`amortize_launch`): RTX **17.1%** MAPE, H200 **9.5%** MAPE.

## Train + compose

```bash
pip install -e .
bash scripts/run_train.sh
bash scripts/run_compose.sh
```
