# Reproduce results

Two experiment packages ship **code + hardware specs + checkpoints + measurement data**,
so you can retrain and compose **without a GPU**. Optional GPU collection lives in
[`collection/`](collection/).

| Folder | Role |
|---|---|
| [`operator_workload_prediction/`](operator_workload_prediction/) | Leave-one-GPU-out operator / block latency |
| [`end_end_prediction/`](end_end_prediction/) | Train on coverage + compose GPT-2 / BERT |
| [`collection/`](collection/) | Re-measure on a live GPU (optional) |

## Shared setup (CPU train / eval)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch  # CPU wheel is fine for the reported runs
```

Mainline hyper-parameters (both folders): `mode=latent_physics`, `hidden_dim=128`,
`epochs=20`, `batch_size=64`, `lr=1e-3`, `auxiliary_weight=0.2`, `patience=5`, `seed=29`.

---

## 1. Operator / block LOGO

```bash
cd operator_workload_prediction
pip install -e .
bash scripts/run_train_logo.sh
```

This trains three leave-one-GPU-out checkpoints (GB10 / H200 / RTX) using
`data/measurements.jsonl` + `data/graphs/`.

Or evaluate a shipped checkpoint:

```bash
PYTHONPATH=. python -m schedule_free_perf.cli evaluate data/measurements.jsonl \
  --hardware-dir hardware \
  --checkpoint artifacts/gb10_h128.pt \
  --held-out-hardware nvidia_gb10
```

Expected ballpark (mean OOD MAPE, $d=128$): GB10 **12.2%**, H200 **14.3%**, RTX **17.6%**.

---

## 2. End-to-end train + compose

```bash
cd end_end_prediction
pip install -e .
bash scripts/run_train.sh
bash scripts/run_compose.sh
```

Compose uses shipped ground truth (`data/e2e_groundtruth/`) and coverage StableHLO
graphs (`data/e2e_coverage_graphs/`). Recommended mode: `amortize_launch`
(~**17.1%** MAPE RTX, ~**9.5%** MAPE H200).

Poster figures:

```bash
python plot_e2e_poster.py
```

---

## 3. Collect your own GPU data (optional)

See [`collection/README.md`](collection/README.md). After collecting, convert with
`schedule_free_perf.cli convert-data` and point train / compose at the new paths.
