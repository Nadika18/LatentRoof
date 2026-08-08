#!/usr/bin/env python3
"""Separate collection of GPT-2 / BERT-dimension configs for end-to-end composition.

Reuses the ENTIRE collect_data.py pipeline (workload builders, StableHLO extraction,
XLA-dump parsing, benchmarking, row schema) by overriding ONLY the config grid — via a
monkeypatch of make_workload_registry. It writes to its own data/e2e_coverage_<ts>/
directory. collect_data.py is never modified and the existing balanced datasets are
never opened for writing.

Run exactly like collect_data.py (needs XLA_FLAGS for compiler labels):
  CUDA_VISIBLE_DEVICES=0 \
  XLA_FLAGS="--xla_dump_to=./xla_dump_e2e --xla_dump_hlo_as_text" \
  TMPDIR=$PWD/.xla_tmp \
  python collect_e2e_coverage.py
"""
import os
import sys
from datetime import datetime

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # match collect_data

import jax.numpy as jnp          # noqa: E402  (after env, before collect_data)
import collect_data              # noqa: E402

# Standard model dimensions we lack in the balanced grid (coverage check gaps).
# (hidden, ff_dim) coupled as in real transformers (ff = 4 x hidden).
HF_PAIRS = [
    (768, 3072),    # GPT-2 small / BERT-base
    (1024, 4096),   # GPT-2 medium / BERT-large
    (1280, 5120),   # GPT-2 large
]
SEQS = [512, 1024, 2048]
BATCHES = [1, 8, 16, 32]
DTYPES = [jnp.bfloat16, jnp.float16]   # real GPT-2/BERT inference dtypes (f32 omitted)

# Transformer-block families used for composition. gemm/batchmatmul/attention are
# excluded: the block families already contain the matmuls, and attention's "hidden"
# is a per-head dim (64) that is already covered.
FAMILIES = ("transformer", "mha", "feedforward", "mlp3",
            "layernorm", "softmax", "gelu", "residual")
_HAS_FF = ("transformer", "feedforward", "mlp3")


def coverage_configs(name):
    cfgs = []
    for hidden, ff in HF_PAIRS:
        for seq in SEQS:
            for batch in BATCHES:
                for dt in DTYPES:
                    c = {"batch": batch, "seq": seq, "hidden": hidden, "dtype": dt}
                    if name in _HAS_FF:
                        c["ff_dim"] = ff
                    cfgs.append(c)
    return cfgs


_orig_registry = collect_data.make_workload_registry


def coverage_registry():
    """Same builders as collect_data, but only the block families with model dims."""
    reg = _orig_registry()
    return {
        name: (fn, input_builder, flops_fn, coverage_configs(name))
        for name, (fn, input_builder, flops_fn, _cfgs) in reg.items()
        if name in FAMILIES
    }


# Override the grid; collect_data.main() will call this instead of the balanced one.
collect_data.make_workload_registry = coverage_registry


if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"data/e2e_coverage_{ts}"
    # Same measurement protocol as the balanced runs; no --per-workload so every
    # coverage config is collected.
    sys.argv = [
        "collect_e2e_coverage",
        "--output-dir", out,
        "--workload", "all",
        "--n-trials", "2", "--n-runs", "100", "--cooldown", "1.0",
    ]
    collect_data.main()
