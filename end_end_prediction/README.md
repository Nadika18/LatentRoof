# End-to-end prediction

Compose StableHLO transformer-block latency predictions into full GPT-2 / BERT
backbone latency, then grade against measured ground truth.

Architecture matches `operator_workload_prediction` (GNN + hardware MLP +
dual-peak roofline, `hidden_dim=128`). Training adds GPT/BERT-dimension coverage
blocks so those shapes are in-distribution.

## Layout

- `schedule_free_perf/` — model, training, eval
- `hardware/` — public GPU resource JSONs
- `compose.py` — `N × block + LN` with optional launch amortization
- `plot_e2e_poster.py` — poster figures
- `artifacts/` — checkpoints, compose JSON, poster PNGs

## Compose modes

| Mode | Formula |
|---|---|
| `naive` | `N×block + LN` |
| `amortize_launch` (recommended) | `N×(block−launch) + launch + LN` |

Example results (`amortize_launch`): RTX **17.1%** MAPE, H200 **9.5%** MAPE.

## Note on data

Raw coverage / ground-truth measurement trees are **not** included (large).
Point `--coverage-graphs` and `--groundtruth` at your local `data/e2e_*` dirs.
