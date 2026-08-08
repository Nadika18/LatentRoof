# LatentRoof

Schedule-free cross-GPU latency prediction from **StableHLO + public hardware specs**
(no target compiler schedule or profiling at inference).

A graph neural network encodes the StableHLO workload; an MLP encodes public hardware
resources; fused latents drive a **dual-peak analytical roofline** plus a non-negative
residual. Privileged XLA/compiler labels can supervise some latents in training and are
**never required at inference**.

| Folder | What it is |
|---|---|
| [`operator_workload_prediction/`](operator_workload_prediction/) | Operator / block-level predictor (leave-one-GPU-out) |
| [`end_end_prediction/`](end_end_prediction/) | End-to-end GPT-2 / BERT composition + grading |
| [`collection/`](collection/) | Optional live-GPU measurement scripts |
| [`REPRODUCE.md`](REPRODUCE.md) | **Full step-by-step reproduction** |

---

## Hardware

Measurements and evaluation use three NVIDIA platforms (public peak specs in each
folder’s `hardware/`):

| GPU | Class | Role in this repo |
|---|---|---|
| **H200 NVL** | Hopper datacenter | Train / test GPU |
| **RTX PRO 6000** | Blackwell workstation / server | Train / test GPU |
| **GB10** | Grace–Blackwell client | Train / test GPU (operator LOGO); e2e GT not yet shipped |

---

## Dataset

### Operator / block corpus (`operator_workload_prediction/data/`)

| | |
|---|---|
| Latency rows | **6,778** |
| Unique configs | **2,593** |
| Unique StableHLO graphs | **2,292** |
| Per-GPU rows (approx.) | H200 2,593 · RTX 2,576 · GB10 1,609 |
| Stack | JAX / XLA → StableHLO + isolated latency |

Each row pairs a StableHLO graph with measured latency (µs), hardware id, workload
family / config, and optional **privileged labels** parsed from XLA at collection time
(DRAM bytes, fusion stats, kernel count, precision mix, …). Those labels train auxiliary
heads only; inference uses graph + public hardware JSON.

### Workload families (11)

| Group | Families |
|---|---|
| Linear algebra | `gemm`, `batchmatmul` |
| Element-wise / norm | `gelu`, `softmax`, `layernorm`, `residual` |
| Transformer building blocks | `feedforward`, `attention`, `mha`, `mlp3`, `transformer` |

Configs sweep **FP32 / FP16 / BF16** and varied batch, sequence, hidden, FFN, and matrix
shapes (balanced grids; see [`collection/collect_data.py`](collection/collect_data.py)).

### End-to-end corpus (`end_end_prediction/data/`)

| | |
|---|---|
| Training rows | **7,864** (operator suite + GPT/BERT-dimension **coverage** blocks) |
| Coverage intent | Hidden/FFN dims of GPT-2 small/medium/large and BERT base/large so composition shapes are in-distribution |
| Backbone ground truth | **40 configs × {RTX, H200}** — 5 models × {B1, B8} × {S512, S1024} × {bf16, f16} |
| Compose inputs | StableHLO for one transformer block + final LayerNorm (`data/e2e_coverage_graphs/`) |

Composition (recommended):  
`N × (block − launch) + launch + LN` (`amortize_launch`).

---

## Evaluation protocol

**Operator / block:** leave-one-GPU-out (LOGO). Train on two GPUs, **test GPU** = the
held-out device. In-distribution validation on seen GPUs groups by workload ID (no
leakage of the same config into train and val).

Mainline: `latent_physics`, `hidden_dim=128`, seed 29, 20 epochs, patience 5.

| Test GPU | Mean OOD MAPE |
|---|---:|
| GB10 | 12.2% |
| H200 | 14.3% |
| RTX PRO 6000 | 17.6% |

**End-to-end:** train on coverage-augmented rows; predict block (+ LN) on the test GPU;
compose to full GPT-2 / BERT depth; grade vs measured backbone latency.

| Test GPU | MAPE (`amortize_launch`) |
|---|---:|
| RTX PRO 6000 | 17.1% |
| H200 NVL | 9.5% |

---

## What is included vs not included

### Included in the repo (enough to train / evaluate / compose **without a GPU**)

- Model code, hardware JSONs, train/compose scripts
- Mainline `hidden_dim=128` checkpoints + compose / poster artifacts
- `data/measurements.jsonl` — latencies + **already-extracted** compiler privilege labels
- `data/graphs/*.stablehlo.txt` — StableHLO sources used as model inputs
- End-to-end: backbone ground truth + coverage StableHLO graphs for composition

### Not included

- **Raw XLA compiler dump trees** (`xla_dumps/`, multi‑GB per GPU run)
- Local venvs, scratch TMPDIR trees, and collection re-run outputs under `collection/data/`

During measurement, XLA dumps are generated temporarily, parsed into per-row labels
(`label_n_kernels`, DRAM bytes, fusion stats, …), and those labels are stored inside
`measurements.jsonl` / `dataset.json`. **Training and compose do not read the dump
trees** — only the StableHLO graphs, latencies, and stored labels. If you re-collect
on your own GPU, dumps are written locally under the collection output directory and
can be deleted after a successful run to save disk.

Full commands: **[`REPRODUCE.md`](REPRODUCE.md)**.
