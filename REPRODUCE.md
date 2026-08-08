# Reproduce results

This guide covers:

1. **Path A — no GPU:** retrain / evaluate / compose from shipped data  
2. **Path B — with GPU:** re-collect measurements (including local XLA dumps), convert, then train  

Hyper-parameters used for reported mainline numbers (both experiment folders):

| Setting | Value |
|---|---|
| `mode` | `latent_physics` |
| `hidden_dim` | `128` |
| `message_steps` | `2` |
| `epochs` | `20` |
| `batch_size` | `64` |
| `learning_rate` | `1e-3` |
| `auxiliary_weight` | `0.2` |
| `patience` | `5` |
| `seed` | `29` |

---

## Important: XLA dumps are not in this repo

Collection uses JAX/XLA with `--xla_dump_to=…` so compiler decisions can be parsed.
Those **raw dump directories are intentionally not published** (large; not needed at train time).

What *is* published already includes the extracted fields in each measurement row
(`privileged_labels` / `label_*`). Path A therefore does **not** need dumps.

If you follow Path B, dumps appear under your local
`collection/data/<run>/xla_dumps/` (and the temporary `--xla_dump_to` folder). Keep them
only if you want to re-audit labels; otherwise delete after `dataset.json` is written.

---

## 0. Shared Python env (CPU is enough for Path A)

```bash
git clone git@github.com:Nadika18/LatentRoof.git
cd LatentRoof

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch numpy matplotlib   # CPU torch wheel is fine
```

Optional GPU train: install a CUDA build of PyTorch and set `DEVICE=cuda` when running scripts.

---

## Path A — reproduce from shipped data (no GPU)

### A1. Operator / block leave-one-GPU-out (LOGO)

```bash
cd operator_workload_prediction
pip install -e .

# Train all three folds (writes artifacts/*_h128.pt)
bash scripts/run_train_logo.sh

# Or evaluate a shipped checkpoint on the held-out GPU
PYTHONPATH=. python -m schedule_free_perf.cli evaluate \
  data/measurements.jsonl \
  --hardware-dir hardware \
  --held-out-hardware nvidia_gb10 \
  artifacts/gb10_h128.pt
```

Held-out hardware ids: `nvidia_gb10`, `nvidia_h200`, `nvidia_rtx_pro_6000`.

**Expected ballpark** (mean OOD MAPE, $d=128$): GB10 **12.2%**, H200 **14.3%**, RTX **17.6%**.  
Summary JSON: `artifacts/logo_three_fold_h128.json`.

Data used: `data/measurements.jsonl` + `data/graphs/`.

### A2. End-to-end train

```bash
cd ../end_end_prediction
pip install -e .

bash scripts/run_train.sh
# -> artifacts/e2e_all_h128.pt
# -> artifacts/e2e_rtx_out_h128.pt
# -> artifacts/e2e_h200_out_h128.pt
```

### A3. End-to-end compose (grade vs ground truth)

```bash
# Uses shipped GT + coverage StableHLO (not XLA dumps)
bash scripts/run_compose.sh
# default MODE=amortize_launch
# -> artifacts/compose_amortize_launch_rtx.json
# -> artifacts/compose_amortize_launch_h200.json
```

Manual one-shot:

```bash
PYTHONPATH=. python compose.py \
  --checkpoint artifacts/e2e_all_h128.pt \
  --groundtruth data/e2e_groundtruth/rtx_pro_6000.json \
  --coverage-graphs data/e2e_coverage_graphs \
  --compose-mode amortize_launch \
  --output artifacts/compose_amortize_launch_rtx.json
```

Compose modes: `naive`, `amortize_residual`, `amortize_launch` (recommended), `amortize_both`.

**Expected ballpark** (`amortize_launch`): RTX **~17.1%** MAPE, H200 **~9.5%** MAPE.

### A4. Poster figures

```bash
python plot_e2e_poster.py
# -> artifacts/poster_figs/e2e_mape_by_model_batch_amortize_launch.png
```

Requires the `compose_amortize_launch_{rtx,h200}.json` files from A3 (or the shipped copies).

---

## Path B — collect on a live GPU, then train

Needs NVIDIA driver + a JAX CUDA build. See also [`collection/README.md`](collection/README.md).

### B0. Collect env

```bash
cd collection
python3 -m venv .venv_collect
source .venv_collect/bin/activate
pip install -U pip
pip install "jax[cuda12]" numpy   # match your CUDA / JAX docs if needed

mkdir -p .xla_tmp
sudo bash setup_bench_env.sh 0    # optional: clocks / persistence / busy check
```

### B1. Operator / block sweep

Produces `dataset.json`, `graphs/*.stablehlo.txt`, and local `xla_dumps/` (not required later).

```bash
cd collection
source .venv_collect/bin/activate

CUDA_VISIBLE_DEVICES=0 \
XLA_FLAGS="--xla_dump_to=./xla_dump_temp --xla_dump_hlo_as_text" \
TMPDIR=$PWD/.xla_tmp \
python collect_data.py --output-dir data/my_gpu_balanced
```

Useful flags: `--max-configs N`, `--workload gemm`, `--n-trials 2`, `--n-runs 100`, `--per-workload N`.

Repeat on each GPU you care about (H200 / RTX / GB10), with a distinct `--output-dir`.

**After a good run you may delete** `data/my_gpu_balanced/xla_dumps/` and `./xla_dump_temp` to reclaim disk. Keep `dataset.json` + `graphs/`.

### B2. Convert collected datasets → training manifest

```bash
cd ../operator_workload_prediction
source ../.venv/bin/activate   # torch env from section 0
pip install -e .

PYTHONPATH=. python -m schedule_free_perf.cli convert-data \
  ../collection/data/my_gpu_balanced/dataset.json \
  ../collection/data/other_gpu_balanced/dataset.json \
  --manifest data/measurements_new.jsonl \
  --audit-output artifacts/audit_new.json
```

Train like Path A, but point at the new manifest (graph paths inside the manifest still
refer to the collection `graphs/` directories — leave those dirs in place):

```bash
PYTHONPATH=. python -m schedule_free_perf.cli train data/measurements_new.jsonl \
  --hardware-dir hardware \
  --held-out-hardware nvidia_gb10 \
  --mode latent_physics --hidden-dim 128 --message-steps 2 \
  --epochs 20 --batch-size 64 --learning-rate 1e-3 --auxiliary-weight 0.2 \
  --patience 5 --seed 29 --device cpu \
  --output artifacts/gb10_h128_new.pt \
  --history artifacts/gb10_h128_new_history.json
```

Or copy/merge into `data/measurements.jsonl` + `data/graphs/` if you want the helper scripts unchanged.

### B3. End-to-end coverage blocks (for composition training)

```bash
cd ../collection
source .venv_collect/bin/activate

CUDA_VISIBLE_DEVICES=0 \
XLA_FLAGS="--xla_dump_to=./xla_dump_e2e --xla_dump_hlo_as_text" \
TMPDIR=$PWD/.xla_tmp \
python collect_e2e_coverage.py
# -> data/e2e_coverage_<timestamp>/  (dataset.json, graphs/, xla_dumps/)
```

Convert coverage (+ your operator datasets) into one e2e manifest the same way as B2,
then train from `end_end_prediction/` (or use `scripts/run_train.sh` after replacing
`data/measurements.jsonl`).

For compose graph lookup you need the StableHLO files under `graphs/` (names like
`transformer_batch…stablehlo.txt`). Point `--coverage-graphs` at that `graphs/` folder.
You do **not** need `xla_dumps/` for compose.

### B4. End-to-end backbone ground truth

```bash
cd collection
source .venv_collect/bin/activate

CUDA_VISIBLE_DEVICES=0 TMPDIR=$PWD/.xla_tmp \
  python measure_e2e_groundtruth.py
# -> data/e2e_groundtruth_<timestamp>/results.json
```

Rows include `gpu_name`. Concatenate RTX / H200 / GB10 JSON lists as needed, then:

```bash
cd ../end_end_prediction
PYTHONPATH=. python compose.py \
  --checkpoint artifacts/e2e_all_h128.pt \
  --groundtruth ../collection/data/e2e_groundtruth_<timestamp>/results.json \
  --coverage-graphs ../collection/data/e2e_coverage_<timestamp>/graphs \
  --compose-mode amortize_launch \
  --output artifacts/compose_amortize_launch_custom.json
```

---

## Quick map of shipped data paths

| Path | Role |
|---|---|
| `operator_workload_prediction/data/measurements.jsonl` | Operator LOGO training rows |
| `operator_workload_prediction/data/graphs/` | StableHLO inputs for those rows |
| `end_end_prediction/data/measurements.jsonl` | E2E training rows (operators + coverage) |
| `end_end_prediction/data/graphs/` | StableHLO for e2e training |
| `end_end_prediction/data/e2e_groundtruth/` | Backbone latency labels (RTX, H200, combined) |
| `end_end_prediction/data/e2e_coverage_graphs/` | StableHLO for `compose.py` |
| `*/artifacts/*.pt` | Shipped checkpoints |
| *(not present)* `**/xla_dumps/` | Raw XLA dumps — collect locally only |
