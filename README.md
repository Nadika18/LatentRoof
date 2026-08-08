# LatentRoof

Schedule-free cross-GPU latency prediction from **StableHLO + public hardware specs**
(no target compiler schedule or profiling at inference).

| Folder | What it is |
|---|---|
| [`operator_workload_prediction/`](operator_workload_prediction/) | Operator / block-level predictor (leave-one-GPU-out) |
| [`end_end_prediction/`](end_end_prediction/) | End-to-end GPT-2 / BERT composition + grading |
| [`collection/`](collection/) | Optional live-GPU measurement scripts |
| [`REPRODUCE.md`](REPRODUCE.md) | **Full step-by-step reproduction** |

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
