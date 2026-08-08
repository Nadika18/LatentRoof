# Operator / block-level workload prediction

Predict latency of StableHLO operator and Transformer-block graphs on an unseen
(**test**) GPU using only the graph and public hardware specifications.

## Dataset (included under `data/`)

| | |
|---|---|
| Rows | **6,778** latency measurements |
| Configs / graphs | **2,593** unique configs · **2,292** unique StableHLO graphs |
| GPUs | H200 NVL · RTX PRO 6000 · GB10 |
| Dtypes | FP32, FP16, BF16 |
| Families (11) | `gemm`, `batchmatmul`, `gelu`, `softmax`, `layernorm`, `residual`, `feedforward`, `attention`, `mha`, `mlp3`, `transformer` |

Rows include measured latency plus optional privileged XLA labels used only as
training targets. See [`DATA.md`](DATA.md) for conversion / split rules.

## Layout

- `schedule_free_perf/` — GNN + hardware MLP + dual-peak roofline model
- `hardware/` — H200 / RTX PRO 6000 / GB10 resource JSONs
- `data/measurements.jsonl` + `data/graphs/` — included corpus (no GPU needed)
- `artifacts/` — mainline `hidden_dim=128` LOGO checkpoints + summary
- `scripts/run_train_logo.sh` — reproduce leave-one-GPU-out training
- `DATA.md`, `DESIGN.md`, `LIMITATIONS.md`, `LITERATURE.md`

Full steps: repo-root [`REPRODUCE.md`](../REPRODUCE.md).  
Raw **XLA dump trees are not included** (labels are already inside `measurements.jsonl`).  
Optional remasure: [`collection/`](../collection/).

## Quick results (LOGO, $d=128$, mean OOD MAPE)

| Test GPU | OOD MAPE |
|---|---:|
| GB10 | 12.2% |
| H200 | 14.3% |
| RTX PRO 6000 | 17.6% |

## Train

```bash
pip install -e .
bash scripts/run_train_logo.sh
```
