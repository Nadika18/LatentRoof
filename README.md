# LatentRoof - An Analytical-ML framework for Cross-architecture Performance prediction from StableHLO graph


**Cross-architecture latency prediction from StableHLO + public hardware specs.**

LatentRoof predicts operator / block level and end-to-end latency on an unseen NVIDIA GPU without running
the target compiler schedule or profiling the target device at inference. A GNN encodes
the StableHLO graph; an MLP encodes public hardware resources; fused latents drive a
**dual-peak analytical roofline** plus a non-negative residual. Privileged XLA labels
may supervise some latents during training and are **never required at inference**.

This repository has two experiment packages:

| Package | Task |
|---|---|
| [`operator_workload_prediction/`](operator_workload_prediction/) | Leave-one-GPU-out operator / block latency |
| [`end_end_prediction/`](end_end_prediction/) | Compose blocks into GPT-2 / BERT backbone latency |
| [`collection/`](collection/) | Optional live-GPU data collection (JAX/XLA) |

---

## How it works

1. Parse the StableHLO program into a semantic graph (ops, shapes, dtypes, edges).
2. Encode the graph with a GNN and the target GPU with a hardware MLP.
3. Predict latents (effective DRAM, utilization, kernels, …) that parameterize a
   dual-peak roofline; add a non-negative residual for unmodeled overhead.
4. For end-to-end models, compose `N` transformer-block predictions (+ final LN),
   optionally amortizing launch overhead across layers.

---

## Repository structure

```
LatentRoof/
├── README.md                          # This file
├── REPRODUCE.md                       # Detailed train / compose / collect steps
├── collection/                        # Live-GPU measurement scripts (optional)
│   ├── collect_data.py                # Operator / block sweep
│   ├── collect_e2e_coverage.py        # GPT/BERT-dimension coverage blocks
│   ├── measure_e2e_groundtruth.py     # Full backbone latency labels
│   └── setup_bench_env.sh
│
├── operator_workload_prediction/      # Block-level LOGO predictor
│   ├── schedule_free_perf/            # Model, training, eval, CLI
│   ├── hardware/                      # GPU JSON specs
│   ├── data/                          # measurements.jsonl + StableHLO graphs
│   ├── artifacts/                     # Checkpoints + LOGO summary
│   └── scripts/run_train_logo.sh
│
└── end_end_prediction/                # End-to-end composition
    ├── schedule_free_perf/            # Same model family (coverage-augmented train)
    ├── hardware/
    ├── data/                          # Train corpus + GT + coverage graphs
    ├── compose.py                     # N×block (+ LN) composition
    ├── plot_e2e_poster.py
    ├── artifacts/
    └── scripts/run_train.sh, run_compose.sh
```

---

## Setup

Python **≥ 3.10**. CPU is enough to retrain and compose with the included data.

```bash
git clone git@github.com:Nadika18/LatentRoof.git
cd LatentRoof

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch numpy matplotlib
```

Install either experiment package (editable):

```bash
# Operator / block LOGO
cd operator_workload_prediction
pip install -e .
cd ..

# and/or end-to-end
cd end_end_prediction
pip install -e .
cd ..
```

Optional GPU training: install a CUDA build of PyTorch and set `DEVICE=cuda` when
running the train scripts.

Optional **data collection** (live NVIDIA GPU + JAX/XLA):

```bash
cd collection
python3 -m venv .venv_collect
source .venv_collect/bin/activate
pip install -U pip
pip install "jax[cuda12]" numpy   # match your CUDA / JAX docs
```

See [`collection/README.md`](collection/README.md).

---

## Quick start

### Operator / block leave-one-GPU-out

```bash
cd operator_workload_prediction
bash scripts/run_train_logo.sh
```

Or evaluate a shipped checkpoint on a test GPU:

```bash
PYTHONPATH=. python -m schedule_free_perf.cli evaluate \
  data/measurements.jsonl \
  --hardware-dir hardware \
  --held-out-hardware nvidia_gb10 \
  artifacts/gb10_h128.pt
```

### End-to-end train + compose

```bash
cd end_end_prediction
bash scripts/run_train.sh
bash scripts/run_compose.sh          # MODE=amortize_launch by default
python plot_e2e_poster.py
```

Full command reference: [`REPRODUCE.md`](REPRODUCE.md).

---

## Workflow

### A. Reproduce from included data (no GPU)

1. **Setup** — create the venv and `pip install -e .` in the package you need.
2. **Train** — `scripts/run_train_logo.sh` or `scripts/run_train.sh`.
3. **Compose (e2e)** — `scripts/run_compose.sh` grades vs included backbone GT.
4. **Figures** — `plot_e2e_poster.py` regenerates poster PNGs.

### B. Collect your own measurements (GPU)

1. **Bench hygiene** — `sudo bash collection/setup_bench_env.sh <gpu_index>`.
2. **Operator sweep** — `collect_data.py` with `XLA_FLAGS=--xla_dump_to=…`
   (dumps are parsed into labels, then optional to delete).
3. **Convert** — `python -m schedule_free_perf.cli convert-data …/dataset.json …`.
4. **Coverage + GT (e2e)** — `collect_e2e_coverage.py`, `measure_e2e_groundtruth.py`.
5. **Train / compose** as in path A, pointing at the new paths.

---

## Supported hardware

Public resource JSONs live under each package’s `hardware/`.

| GPU | Architecture | In this artifact |
|-----|--------------|------------------|
| H200 NVL | Hopper | Operator LOGO + e2e GT / compose |
| RTX PRO 6000 | Blackwell | Operator LOGO + e2e GT / compose |
| GB10 | Grace–Blackwell | Operator LOGO (e2e GT not shipped yet) |

---

## Datasets

### Operator / block corpus

Path: `operator_workload_prediction/data/`

| | |
|---|---|
| Latency rows | **6,778** |
| Unique configs | **2,593** |
| Unique StableHLO graphs | **2,292** |
| Approx. per GPU | H200 2,593 · RTX 2,576 · GB10 1,609 |
| Stack | JAX / XLA → StableHLO + isolated latency |
| Dtypes | FP32, FP16, BF16 |

**Workload families (11)**

| Group | Families |
|---|---|
| Linear algebra | `gemm`, `batchmatmul` |
| Element-wise / norm | `gelu`, `softmax`, `layernorm`, `residual` |
| Transformer blocks | `feedforward`, `attention`, `mha`, `mlp3`, `transformer` |

Each row stores latency, hardware id, config, and optional privileged labels
(DRAM, fusion, kernels, …) used only as training targets. Conversion and split
rules: [`operator_workload_prediction/DATA.md`](operator_workload_prediction/DATA.md).

### End-to-end corpus

Path: `end_end_prediction/data/`

| | |
|---|---|
| Training rows | **7,864** (operator suite + GPT/BERT-dimension coverage) |
| Coverage shapes | GPT-2 small/medium/large, BERT base/large · seq ∈ {512,1024,2048} · batch ∈ {1,8,16,32} · bf16/f16 |
| Backbone GT | **40** configs × {RTX, H200}: 5 models × {B1, B8} × {S512, S1024} × {bf16, f16} |
| Compose graphs | `e2e_coverage_graphs/` — transformer block + final LN per shape |

Recommended composition: `N × (block − launch) + launch + LN` (`amortize_launch`).

---

## Results (mainline, `hidden_dim=128`)

**Operator LOGO** (mean OOD MAPE on the test GPU)

| Test GPU | OOD MAPE |
|---|---:|
| GB10 | 12.2% |
| H200 | 14.3% |
| RTX PRO 6000 | 17.6% |

**End-to-end** (`amortize_launch`)

| Test GPU | MAPE |
|---|---:|
| RTX PRO 6000 | 17.1% |
| H200 NVL | 9.5% |

Hyper-parameters: `latent_physics`, `hidden_dim=128`, `message_steps=2`, 20 epochs,
batch 64, lr `1e-3`, auxiliary weight `0.2`, patience 5, seed 29.

---

## What is included vs not included

**Included** (enough to train / evaluate / compose without a GPU)

- Model code, hardware JSONs, scripts, mainline checkpoints
- `measurements.jsonl` + StableHLO graphs (labels already extracted)
- E2E backbone GT + coverage graphs for compose

**Not included**

- Raw XLA dump trees (`xla_dumps/`, multi‑GB) — generated locally during collection,
  parsed into row labels, then safe to delete
- Local venvs and collection scratch under `collection/data/`

---

## Citation

If you use this artifact, please cite the accompanying paper / poster (to be filled in).
