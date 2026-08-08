# LatentRoof

Schedule-free cross-GPU latency prediction from **StableHLO + public hardware specs**
(no target compiler schedule or profiling at inference).

| Folder | What it is |
|---|---|
| [`operator_workload_prediction/`](operator_workload_prediction/) | Operator / block-level predictor (leave-one-GPU-out) |
| [`end_end_prediction/`](end_end_prediction/) | End-to-end GPT-2 / BERT composition + grading |
| [`collection/`](collection/) | Optional live-GPU measurement scripts |
| [`REPRODUCE.md`](REPRODUCE.md) | Train / compose / collect instructions |

## Shipped for reproduction (no GPU required)

Each experiment folder includes:
- Model code + public hardware JSONs
- Mainline `hidden_dim=128` checkpoints
- Portable `data/measurements.jsonl` + `data/graphs/` (StableHLO + latencies)

End-to-end also ships backbone ground truth and coverage graphs for compose.

## Optional: collect on your GPU

See [`collection/README.md`](collection/README.md). XLA dumps from collection are large and are **not** shipped.
