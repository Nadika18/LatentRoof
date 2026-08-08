# End-to-end prediction

Compose StableHLO transformer-block latency predictions into full GPT-2 / BERT
backbone latency, then grade against measured ground truth.

Architecture matches `operator_workload_prediction` (GNN + hardware MLP +
dual-peak roofline, `hidden_dim=128`). Training adds GPT/BERT-dimension coverage
blocks so those shapes are in-distribution.

## Dataset (included under `data/`)

| | |
|---|---|
| Training rows | **7,864** (operator/block suite + coverage blocks) |
| Coverage dims | GPT-2 small/medium/large and BERT base/large `(hidden, ff=4×hidden)` · seq ∈ {512,1024,2048} · batch ∈ {1,8,16,32} · bf16/f16 |
| Coverage families | `transformer`, `mha`, `feedforward`, `mlp3`, `layernorm`, `softmax`, `gelu`, `residual` |
| Backbone GT | **40** configs per GPU on **RTX** and **H200**: 5 models × {B1,B8} × {S512,S1024} × {bf16,f16} |
| Compose graphs | `data/e2e_coverage_graphs/` — one transformer block + final LN per shape |

## Layout

- `schedule_free_perf/` — model, training, eval
- `hardware/` — public GPU resource JSONs
- `data/measurements.jsonl` + `data/graphs/` — training corpus (no GPU needed)
- `data/e2e_groundtruth/` — backbone latency labels (RTX + H200)
- `data/e2e_coverage_graphs/` — StableHLO used by `compose.py`
- `compose.py`, `plot_e2e_poster.py`
- `scripts/run_train.sh`, `scripts/run_compose.sh`
- `artifacts/` — checkpoints, compose JSON, poster PNGs

Full steps: repo-root [`REPRODUCE.md`](../REPRODUCE.md).  
Raw **XLA dump trees are not included** (labels are already inside `measurements.jsonl`).  
Optional remasure: [`collection/`](../collection/).

## Compose modes

| Mode | Formula |
|---|---|
| `naive` | `N×block + LN` |
| `amortize_launch` (recommended) | `N×(block−launch) + launch + LN` |

Example results (`amortize_launch`): test GPU RTX **17.1%** MAPE, test GPU H200 **9.5%** MAPE.

## Train + compose

```bash
pip install -e .
bash scripts/run_train.sh
bash scripts/run_compose.sh
```
