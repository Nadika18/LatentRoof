# StableHLO latency prediction

Schedule-free cross-GPU latency prediction from **StableHLO + public hardware specs**
(no target compiler schedule or profiling at inference).

| Folder | What it is |
|---|---|
| [`operator_workload_prediction/`](operator_workload_prediction/) | Operator / block-level predictor (leave-one-GPU-out) |
| [`end_end_prediction/`](end_end_prediction/) | End-to-end GPT-2 / BERT composition + grading |

## Not included

- Raw measurement corpora (`measurements.jsonl`, `data/e2e_*` trees)
- Large evaluation dumps / width-ablation checkpoints
- Training venvs

Point the CLIs at your local data paths when retraining or composing.
