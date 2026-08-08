# GPU data collection

Scripts to **re-measure** latencies on a live NVIDIA GPU (JAX / XLA).
Shipped measurements under each experiment folder already let you train without a GPU.

## Dependencies

```bash
python3 -m venv .venv_collect
source .venv_collect/bin/activate
pip install -U pip
pip install "jax[cuda12]" numpy  # or the JAX wheel matching your CUDA
```

Needs a working NVIDIA driver + CUDA matching your JAX build.

## Bench hygiene (recommended)

```bash
sudo bash setup_bench_env.sh 0   # GPU index
```

Locks clocks / persistence mode when supported; warns if the GPU is busy.

## Operator / block sweep (`collect_data.py`)

Full balanced grid (~2k configs/GPU). Writes `data/<name>/dataset.json` + `graphs/`.

```bash
cd collection
mkdir -p .xla_tmp
CUDA_VISIBLE_DEVICES=0 \
XLA_FLAGS="--xla_dump_to=./xla_dump_temp --xla_dump_hlo_as_text" \
TMPDIR=$PWD/.xla_tmp \
python collect_data.py --output-dir data/my_gpu_balanced
```

Useful flags: `--max-configs N`, `--workload gemm`, `--n-trials 2`, `--n-runs 100`.

Convert a collected dataset into a training manifest (from an experiment folder):

```bash
cd ../operator_workload_prediction
PYTHONPATH=. python -m schedule_free_perf.cli convert-data \
  ../collection/data/my_gpu_balanced/dataset.json \
  --manifest data/measurements_new.jsonl \
  --audit-output artifacts/audit_new.json
```

Then either replace `data/measurements.jsonl` + copy graphs, or merge manifests.

## End-to-end coverage blocks (`collect_e2e_coverage.py`)

GPT-2 / BERT dimension grids for composition training. Same XLA flags as above.

```bash
cd collection
CUDA_VISIBLE_DEVICES=0 \
XLA_FLAGS="--xla_dump_to=./xla_dump_e2e --xla_dump_hlo_as_text" \
TMPDIR=$PWD/.xla_tmp \
python collect_e2e_coverage.py
# -> data/e2e_coverage_<timestamp>/
```

## End-to-end backbone ground truth (`measure_e2e_groundtruth.py`)

Full N-layer GPT-2 / BERT forward latency (for compose grading).

```bash
cd collection
CUDA_VISIBLE_DEVICES=0 TMPDIR=$PWD/.xla_tmp \
  python measure_e2e_groundtruth.py
# -> data/e2e_groundtruth_<timestamp>/results.json
```

Rows include `gpu_name` so H200 / RTX / GB10 runs can be concatenated.
