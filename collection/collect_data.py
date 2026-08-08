"""
Data Collection Pipeline for Cross-Architecture Performance Prediction

For each workload config, collects:
  1. INPUT FEATURES  — from StableHLO graph (before compilation)
  2. LABELS          — from XLA dump files (after compilation)
  3. LATENCY         — from isolated benchmark measurement

Usage:
  # Setup isolation first
  sudo bash setup_bench_env.sh 0

  # Collect all workloads (full sweep, ~2000 configs)
  CUDA_VISIBLE_DEVICES=0 XLA_FLAGS="--xla_dump_to=./xla_dump_temp --xla_dump_hlo_as_text" \
    python collect_data.py

  # Limit to N configs (for testing or time-constrained runs)
  CUDA_VISIBLE_DEVICES=0 XLA_FLAGS="--xla_dump_to=./xla_dump_temp --xla_dump_hlo_as_text" \
    python collect_data.py --max-configs 100

  # Single workload type
  CUDA_VISIBLE_DEVICES=0 XLA_FLAGS="--xla_dump_to=./xla_dump_temp --xla_dump_hlo_as_text" \
    python collect_data.py --workload gemm
"""

import os
import sys
import re
import time
import json
import shutil
import argparse
import subprocess
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np

# CUDA defaults to FASTEST_FIRST, which orders devices by compute capability rather
# than PCI slot, so CUDA_VISIBLE_DEVICES=N can select a different card than the one
# nvidia-smi calls N. Pin PCI order so the two agree. Must precede the jax import.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import jax
import jax.numpy as jnp
from jax import random


# ═══════════════════════════════════════════════════════════════
# Dtype helpers
# ═══════════════════════════════════════════════════════════════

DTYPE_MAP = {
    "f32": jnp.float32,
    "f16": jnp.float16,
    "bf16": jnp.bfloat16,
}

def dtype_str(dt):
    for k, v in DTYPE_MAP.items():
        if v == dt:
            return k
    return "f32"


# ═══════════════════════════════════════════════════════════════
# Memory estimation (bytes) — skip configs that would OOM
# ═══════════════════════════════════════════════════════════════

MAX_GPU_BYTES = 100 * 1024**3  # 100 GB limit (GB10 has 128GB unified, leave headroom)

def _dbytes(dt):
    """Bytes per element for a dtype."""
    if dt in (jnp.float16, jnp.bfloat16):
        return 2
    return 4  # f32


def estimate_memory(workload_name, cfg):
    """Rough memory estimate for a config (inputs + outputs + buffers × 4 copies)."""
    dt = _dbytes(cfg.get("dtype", jnp.float32))
    b = cfg.get("batch", 1)

    if workload_name == "gemm":
        M, N, K = cfg["M"], cfg["N"], cfg["K"]
        mem = (M * K + K * N + M * N) * dt
    elif workload_name == "batchmatmul":
        M, N, K = cfg["M"], cfg["N"], cfg["K"]
        mem = b * (M * K + K * N + M * N) * dt
    elif workload_name in ("softmax", "gelu"):
        mem = b * cfg["seq"] * cfg["hidden"] * dt * 2
    elif workload_name == "layernorm":
        mem = b * cfg["seq"] * cfg["hidden"] * dt * 2 + cfg["hidden"] * dt * 2
    elif workload_name == "feedforward":
        D, FF = cfg["hidden"], cfg.get("ff_dim", cfg["hidden"] * 4)
        mem = (b * cfg["seq"] * D + D * FF + FF + FF * D + D) * dt * 2
    elif workload_name == "residual":
        D = cfg["hidden"]
        mem = (b * cfg["seq"] * D * 2 + D * D + D) * dt
    elif workload_name == "attention":
        S, D = cfg["seq"], cfg["hidden"]
        mem = b * (3 * S * D + S * S) * dt
    elif workload_name == "mha":
        S, D = cfg["seq"], cfg["hidden"]
        mem = (b * S * D + 4 * D * D + b * S * S + b * S * D) * dt
    elif workload_name == "transformer":
        S, D, FF = cfg["seq"], cfg["hidden"], cfg.get("ff_dim", cfg["hidden"] * 4)
        mem = (b * S * D + 4 * D * D + D * FF * 2 + b * S * FF + b * S * S) * dt
    elif workload_name == "mlp3":
        D, FF = cfg["hidden"], cfg.get("ff_dim", cfg["hidden"] * 4)
        mem = (b * cfg["seq"] * D + D * FF + FF * FF + FF * D + b * cfg["seq"] * FF) * dt
    else:
        mem = 0

    return mem * 5  # ×4 buffer copies + intermediate outputs


# ═══════════════════════════════════════════════════════════════
# Workload definitions — multi-dimensional configs
# ═══════════════════════════════════════════════════════════════

def make_workload_registry():
    """Return dict of workload_name -> (fn, input_builder, flops_fn, configs).

    Each config is a dict with shape params + dtype.
    input_builder(config, key) -> tuple of JAX arrays.
    flops_fn(config) -> estimated FLOPs.
    """
    registry = {}
    dtypes = [jnp.float32, jnp.float16, jnp.bfloat16]

    # ─── 1. GEMM: C = A @ B, shapes (M,K) x (K,N) ───
    def gemm(a, b):
        return jnp.dot(a, b)

    def gemm_inputs(cfg, key):
        k1, k2 = random.split(key)
        dt = cfg.get("dtype", jnp.float32)
        return (random.normal(k1, (cfg["M"], cfg["K"]), dtype=dt),
                random.normal(k2, (cfg["K"], cfg["N"]), dtype=dt))

    # Sizes start where measured latency clears ~500us, so dispatch overhead
    # (~75us on this host) stays a small fraction of every measurement.
    gemm_configs = []
    # Square
    for s in [2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192]:
        for dt in dtypes:
            gemm_configs.append({"M": s, "N": s, "K": s, "dtype": dt})
    # Rectangular (common in transformers: hidden × 4*hidden)
    for D in [1024, 1536, 2048, 2560, 3072, 4096]:
        for dt in dtypes:
            gemm_configs.append({"M": D, "N": 4 * D, "K": D, "dtype": dt})
            gemm_configs.append({"M": D, "N": D, "K": 4 * D, "dtype": dt})
            gemm_configs.append({"M": 4 * D, "N": D, "K": D, "dtype": dt})
    # Tall-skinny (batch × seq_len matmuls)
    for B in [1024, 2048, 4096, 8192, 16384, 32768]:
        for D in [1024, 2048, 3072, 4096]:
            for dt in dtypes:
                gemm_configs.append({"M": B, "N": D, "K": D, "dtype": dt})
    # General rectangular — independent M/N/K so aspect ratio varies
    for M in [2048, 4096, 8192]:
        for N in [1024, 2048, 4096, 8192]:
            for K in [1024, 2048, 4096]:
                if M == N == K:
                    continue
                for dt in dtypes:
                    gemm_configs.append({"M": M, "N": N, "K": K, "dtype": dt})

    registry["gemm"] = (gemm, gemm_inputs,
                        lambda c: 2 * c["M"] * c["N"] * c["K"],
                        gemm_configs)

    # ─── 2. BatchMatmul ───
    def batchmatmul(a, b):
        return jnp.matmul(a, b)

    def bmm_inputs(cfg, key):
        k1, k2 = random.split(key)
        dt = cfg.get("dtype", jnp.float32)
        B = cfg["batch"]
        return (random.normal(k1, (B, cfg["M"], cfg["K"]), dtype=dt),
                random.normal(k2, (B, cfg["K"], cfg["N"]), dtype=dt))

    bmm_configs = []
    for B in [2, 4, 8, 12, 16, 24, 32, 48]:
        for s in [768, 1024, 1536, 2048, 2560, 3072, 4096]:
            for dt in dtypes:
                bmm_configs.append({"batch": B, "M": s, "N": s, "K": s, "dtype": dt})
        for D in [768, 1024, 1536, 2048]:
            for dt in dtypes:
                bmm_configs.append({"batch": B, "M": D, "N": 4 * D, "K": D, "dtype": dt})

    registry["batchmatmul"] = (batchmatmul, bmm_inputs,
                               lambda c: c["batch"] * 2 * c["M"] * c["N"] * c["K"],
                               bmm_configs)

    # ─── 3. Softmax ───
    def softmax(x):
        return jax.nn.softmax(x, axis=-1)

    def softmax_inputs(cfg, key):
        dt = cfg.get("dtype", jnp.float32)
        return (random.normal(key, (cfg["batch"], cfg["seq"], cfg["hidden"]), dtype=dt),)

    # Bandwidth-bound: needs ~1e8 elements before it runs long enough to measure
    # cleanly, so tiny combinations are dropped rather than swept.
    softmax_configs = []
    for B in [8, 16, 32, 64]:
        for S in [2048, 3072, 4096, 6144, 8192]:
            for D in [1024, 1536, 2048, 3072, 4096]:
                if B * S * D < 6e7:
                    continue
                for dt in dtypes:
                    softmax_configs.append({"batch": B, "seq": S, "hidden": D, "dtype": dt})

    registry["softmax"] = (softmax, softmax_inputs,
                           lambda c: 5 * c["batch"] * c["seq"] * c["hidden"],
                           softmax_configs)

    # ─── 4. GELU ───
    def gelu_op(x):
        return jax.nn.gelu(x)

    def gelu_inputs(cfg, key):
        dt = cfg.get("dtype", jnp.float32)
        return (random.normal(key, (cfg["batch"], cfg["seq"], cfg["hidden"]), dtype=dt),)

    gelu_configs = softmax_configs.copy()  # same shape space

    registry["gelu"] = (gelu_op, gelu_inputs,
                        lambda c: 8 * c["batch"] * c["seq"] * c["hidden"],
                        gelu_configs)

    # ─── 5. LayerNorm ───
    def layernorm(x, gamma, beta):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / jnp.sqrt(var + 1e-5) + beta

    def ln_inputs(cfg, key):
        dt = cfg.get("dtype", jnp.float32)
        D = cfg["hidden"]
        return (random.normal(key, (cfg["batch"], cfg["seq"], D), dtype=dt),
                jnp.ones(D, dtype=dt), jnp.zeros(D, dtype=dt))

    ln_configs = []
    for B in [8, 16, 32, 64]:
        for S in [2048, 3072, 4096, 6144, 8192]:
            for D in [1024, 1536, 2048, 3072, 4096, 6144]:
                if B * S * D < 8e7:
                    continue
                for dt in dtypes:
                    ln_configs.append({"batch": B, "seq": S, "hidden": D, "dtype": dt})

    registry["layernorm"] = (layernorm, ln_inputs,
                             lambda c: 8 * c["batch"] * c["seq"] * c["hidden"],
                             ln_configs)

    # ─── 6. Feedforward: dot -> add -> relu -> dot -> add ───
    def feedforward(x, w1, b1, w2, b2):
        h = jax.nn.relu(jnp.dot(x, w1) + b1)
        return jnp.dot(h, w2) + b2

    def ff_inputs(cfg, key):
        keys = random.split(key, 5)
        dt = cfg.get("dtype", jnp.float32)
        B, S, D = cfg["batch"], cfg["seq"], cfg["hidden"]
        FF = cfg.get("ff_dim", 4 * D)
        return (random.normal(keys[0], (B * S, D), dtype=dt),
                random.normal(keys[1], (D, FF), dtype=dt),
                random.normal(keys[2], (FF,), dtype=dt),
                random.normal(keys[3], (FF, D), dtype=dt),
                random.normal(keys[4], (D,), dtype=dt))

    ff_configs = []
    for B in [4, 8, 12, 16, 24, 32]:
        for S in [512, 1024, 1536, 2048]:
            for D in [1536, 2048, 3072, 4096]:
                for ff_mult in [4]:
                    for dt in dtypes:
                        ff_configs.append({"batch": B, "seq": S, "hidden": D,
                                           "ff_dim": ff_mult * D, "dtype": dt})

    registry["feedforward"] = (feedforward, ff_inputs,
                               lambda c: 2 * 2 * c["batch"] * c["seq"] * c["hidden"] * c.get("ff_dim", 4 * c["hidden"]),
                               ff_configs)

    # ─── 7. Residual: dot -> add -> relu -> add(skip) ───
    def residual(x, w, b):
        return x + jax.nn.relu(jnp.dot(x, w) + b)

    def res_inputs(cfg, key):
        k1, k2, k3 = random.split(key, 3)
        dt = cfg.get("dtype", jnp.float32)
        B, S, D = cfg["batch"], cfg["seq"], cfg["hidden"]
        return (random.normal(k1, (B * S, D), dtype=dt),
                random.normal(k2, (D, D), dtype=dt),
                random.normal(k3, (D,), dtype=dt))

    res_configs = []
    for B in [4, 8, 12, 16, 24, 32]:
        for S in [512, 1024, 2048, 4096]:
            for D in [2048, 3072, 4096, 8192]:
                if B * S * D < 8e6:
                    continue
                for dt in dtypes:
                    res_configs.append({"batch": B, "seq": S, "hidden": D, "dtype": dt})

    registry["residual"] = (residual, res_inputs,
                            lambda c: 2 * c["batch"] * c["seq"] * c["hidden"]**2 + 2 * c["batch"] * c["seq"] * c["hidden"],
                            res_configs)

    # ─── 8. Attention: Q@K^T -> scale -> softmax -> @V ───
    def attention(q, k, v):
        d_k = q.shape[-1]
        scores = jnp.matmul(q, k.swapaxes(-2, -1)) / jnp.sqrt(d_k).astype(q.dtype)
        weights = jax.nn.softmax(scores, axis=-1)
        return jnp.matmul(weights, v)

    def attn_inputs(cfg, key):
        keys = random.split(key, 3)
        dt = cfg.get("dtype", jnp.float32)
        B, S, D = cfg["batch"], cfg["seq"], cfg["hidden"]
        return (random.normal(keys[0], (B, S, D), dtype=dt),
                random.normal(keys[1], (B, S, D), dtype=dt),
                random.normal(keys[2], (B, S, D), dtype=dt))

    # Cost grows with seq^2, so the sequence length does the heavy lifting here.
    attn_configs = []
    for B in [4, 8, 12, 16, 24, 32, 48]:
        for S in [2048, 3072, 4096, 6144, 8192]:
            for D in [64, 128, 256, 512]:  # head_dim for attention
                if B * S * S * D < 5e9:
                    continue
                for dt in dtypes:
                    attn_configs.append({"batch": B, "seq": S, "hidden": D, "dtype": dt})

    registry["attention"] = (attention, attn_inputs,
                             lambda c: c["batch"] * (2 * c["seq"]**2 * c["hidden"] + 5 * c["seq"] * c["hidden"]),
                             attn_configs)

    # ─── 9. Multi-Head Attention ───
    def mha(x, wq, wk, wv, wo):
        B, S, D = x.shape
        H = x.shape[-1] // 64  # head_dim = 64
        Dh = D // H
        Q = jnp.dot(x.reshape(B * S, D), wq).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
        K = jnp.dot(x.reshape(B * S, D), wk).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
        V = jnp.dot(x.reshape(B * S, D), wv).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(Dh).astype(x.dtype)
        weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.matmul(weights, V).transpose(0, 2, 1, 3).reshape(B * S, D)
        return jnp.dot(out, wo).reshape(B, S, D)

    def mha_inputs(cfg, key):
        keys = random.split(key, 5)
        dt = cfg.get("dtype", jnp.float32)
        B, S, D = cfg["batch"], cfg["seq"], cfg["hidden"]
        return (random.normal(keys[0], (B, S, D), dtype=dt),
                random.normal(keys[1], (D, D), dtype=dt),
                random.normal(keys[2], (D, D), dtype=dt),
                random.normal(keys[3], (D, D), dtype=dt),
                random.normal(keys[4], (D, D), dtype=dt))

    mha_configs = []
    # Model-like configs: hidden must be divisible by 64 (head_dim)
    for B in [2, 4, 8, 16, 24, 32]:
        for S in [1024, 2048, 3072, 4096]:
            for D in [1024, 1536, 2048, 4096]:
                for dt in dtypes:
                    mha_configs.append({"batch": B, "seq": S, "hidden": D, "dtype": dt})

    registry["mha"] = (mha, mha_inputs,
                       lambda c: c["batch"] * c["seq"] * (4 * 2 * c["hidden"]**2 + 2 * c["seq"] * c["hidden"]),
                       mha_configs)

    # ─── 10. Transformer Block: LN -> MHA -> res -> LN -> FFN -> res ───
    def transformer_block(x, gamma1, beta1, wq, wk, wv, wo,
                          gamma2, beta2, w1, b1, w2, b2):
        B, S, D = x.shape
        H = D // 64
        Dh = D // H

        # Pre-norm MHA
        mean1 = jnp.mean(x, axis=-1, keepdims=True)
        var1 = jnp.var(x, axis=-1, keepdims=True)
        x_ln1 = gamma1 * (x - mean1) / jnp.sqrt(var1 + 1e-5) + beta1

        flat = x_ln1.reshape(B * S, D)
        Q = jnp.dot(flat, wq).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
        K = jnp.dot(flat, wk).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
        V = jnp.dot(flat, wv).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(Dh).astype(x.dtype)
        weights = jax.nn.softmax(scores, axis=-1)
        attn_out = jnp.matmul(weights, V).transpose(0, 2, 1, 3).reshape(B * S, D)
        attn_out = jnp.dot(attn_out, wo).reshape(B, S, D)
        x = x + attn_out

        # Pre-norm FFN
        mean2 = jnp.mean(x, axis=-1, keepdims=True)
        var2 = jnp.var(x, axis=-1, keepdims=True)
        x_ln2 = gamma2 * (x - mean2) / jnp.sqrt(var2 + 1e-5) + beta2
        h = jax.nn.gelu(jnp.dot(x_ln2.reshape(B * S, D), w1) + b1)
        ffn_out = jnp.dot(h, w2) + b2
        return x + ffn_out.reshape(B, S, D)

    def transformer_inputs(cfg, key):
        keys = random.split(key, 9)
        dt = cfg.get("dtype", jnp.float32)
        B, S, D = cfg["batch"], cfg["seq"], cfg["hidden"]
        FF = cfg.get("ff_dim", 4 * D)
        return (random.normal(keys[0], (B, S, D), dtype=dt),
                jnp.ones(D, dtype=dt), jnp.zeros(D, dtype=dt),
                random.normal(keys[1], (D, D), dtype=dt),
                random.normal(keys[2], (D, D), dtype=dt),
                random.normal(keys[3], (D, D), dtype=dt),
                random.normal(keys[4], (D, D), dtype=dt),
                jnp.ones(D, dtype=dt), jnp.zeros(D, dtype=dt),
                random.normal(keys[5], (D, FF), dtype=dt),
                random.normal(keys[6], (FF,), dtype=dt),
                random.normal(keys[7], (FF, D), dtype=dt),
                random.normal(keys[8], (D,), dtype=dt))

    transformer_configs = []
    for B in [2, 4, 8, 16, 24]:
        for S in [1024, 1536, 2048, 2560, 3072]:
            for D in [1024, 1536, 2048, 4096]:
                for dt in dtypes:
                    transformer_configs.append({"batch": B, "seq": S, "hidden": D,
                                                "ff_dim": 4 * D, "dtype": dt})

    registry["transformer"] = (transformer_block, transformer_inputs,
                               lambda c: c["batch"] * c["seq"] * (8 * c["hidden"]**2 + 2 * c["seq"] * c["hidden"] + 4 * c["hidden"] * c.get("ff_dim", 4 * c["hidden"])),
                               transformer_configs)

    # ─── 11. MLP3: dot->gelu->dot->gelu->dot ───
    def mlp3(x, w1, b1, w2, b2, w3, b3):
        h1 = jax.nn.gelu(jnp.dot(x, w1) + b1)
        h2 = jax.nn.gelu(jnp.dot(h1, w2) + b2)
        return jnp.dot(h2, w3) + b3

    def mlp3_inputs(cfg, key):
        keys = random.split(key, 7)
        dt = cfg.get("dtype", jnp.float32)
        B, S, D = cfg["batch"], cfg["seq"], cfg["hidden"]
        FF = cfg.get("ff_dim", 4 * D)
        return (random.normal(keys[0], (B * S, D), dtype=dt),
                random.normal(keys[1], (D, FF), dtype=dt),
                random.normal(keys[2], (FF,), dtype=dt),
                random.normal(keys[3], (FF, FF), dtype=dt),
                random.normal(keys[4], (FF,), dtype=dt),
                random.normal(keys[5], (FF, D), dtype=dt),
                random.normal(keys[6], (D,), dtype=dt))

    mlp3_configs = []
    for B in [2, 4, 8, 16, 24, 32]:
        for S in [512, 1024, 1536, 2048]:
            for D in [1024, 1536, 2048, 3072]:
                for dt in dtypes:
                    mlp3_configs.append({"batch": B, "seq": S, "hidden": D,
                                         "ff_dim": 4 * D, "dtype": dt})

    registry["mlp3"] = (mlp3, mlp3_inputs,
                        lambda c: c["batch"] * c["seq"] * (2 * c["hidden"] * c.get("ff_dim", 4*c["hidden"]) + 2 * c.get("ff_dim", 4*c["hidden"])**2 + 2 * c.get("ff_dim", 4*c["hidden"]) * c["hidden"]),
                        mlp3_configs)

    return registry


def config_to_id(workload_name, cfg):
    """Generate a unique experiment ID from config."""
    parts = [workload_name]
    for k in sorted(cfg.keys()):
        v = cfg[k]
        if k == "dtype":
            parts.append(dtype_str(v))
        else:
            parts.append(f"{k}{v}")
    return "_".join(parts)


# ═══════════════════════════════════════════════════════════════
# Feature extraction from StableHLO
# ═══════════════════════════════════════════════════════════════

ELEMENTWISE_OPS = {"add", "subtract", "multiply", "divide", "maximum", "minimum",
                   "exp", "log", "tanh", "sine", "cosine", "abs", "negate",
                   "sqrt", "rsqrt", "clamp", "compare", "select", "convert",
                   "bitcast_convert", "not", "and", "or", "xor",
                   # StableHLO spells several of these out in full — without these
                   # names the op falls into no category at all.
                   "exponential", "exponential_minus_one", "log_plus_one",
                   "logistic", "erf", "power", "remainder", "sign", "is_finite",
                   "floor", "ceiling", "round_nearest_afz", "round_nearest_even",
                   "cbrt", "atan2", "shift_left", "shift_right_logical",
                   "shift_right_arithmetic"}
CONTRACTION_OPS = {"dot_general", "convolution", "dot"}
REDUCTION_OPS = {"reduce", "reduce_window"}
DATA_MOVEMENT_OPS = {"reshape", "transpose", "broadcast_in_dim", "slice",
                      "dynamic_slice", "concatenate", "gather", "scatter",
                      "pad", "reverse", "iota", "broadcast"}


def categorize_op(opcode):
    op = opcode.lower().strip()
    return (
        1 if op in ELEMENTWISE_OPS else 0,
        1 if op in CONTRACTION_OPS else 0,
        1 if op in REDUCTION_OPS else 0,
        1 if op in DATA_MOVEMENT_OPS else 0,
    )


def parse_tensor_type(type_str):
    """Parse 'tensor<256x256xf32>' -> ('f32', [256, 256])."""
    m = re.search(r'tensor<([\dx]+)x(\w+)>', type_str)
    if m:
        dims_str = m.group(1)
        dtype = m.group(2)
        dims = [int(d) for d in dims_str.split("x")]
        return dtype, dims
    m2 = re.search(r'tensor<(\w+)>', type_str)
    if m2:
        return m2.group(1), []
    return "f32", []


def dtype_bytes(dtype_str):
    mapping = {"f32": 4, "f16": 2, "bf16": 2, "f64": 8, "i32": 4, "i64": 8,
               "i8": 1, "i16": 2, "u8": 1, "u16": 2, "u32": 4, "u64": 8,
               "pred": 1, "c64": 8, "c128": 16}
    return mapping.get(dtype_str, 4)


def extract_node_features(stablehlo_text):
    """Parse StableHLO text and extract per-node features + edges."""
    nodes = []
    edges = []
    name_to_idx = {}

    # Trailing [\s(] so paren-form ops match too, e.g. `stablehlo.reduce(%arg0 init: %cst)`.
    op_pattern = re.compile(
        r'%(\w+)\s*=\s*stablehlo\.(\w+)[\s(]'
    )

    for line in stablehlo_text.split("\n"):
        line_stripped = line.strip()
        m = op_pattern.search(line_stripped)
        if not m:
            continue

        name = m.group(1)
        opcode = m.group(2)
        idx = len(nodes)
        name_to_idx[name] = idx

        tensor_matches = re.findall(r'tensor<[^>]+>', line_stripped)
        if tensor_matches:
            result_type = tensor_matches[-1]
            dtype, dims = parse_tensor_type(result_type)
        else:
            dtype, dims = "f32", []

        elem_size = dtype_bytes(dtype)
        output_bytes = float(max(1, int(np.prod(dims))) * elem_size) if dims else float(elem_size)
        rank = len(dims)
        padded_dims = (dims + [0] * 6)[:6]

        is_elem, is_contract, is_reduce, is_datamov = categorize_op(opcode)

        if is_contract and dims:
            contract_match = re.search(r'contracting_dims\s*=\s*\[(\d+)\]\s*x\s*\[(\d+)\]', line_stripped)
            if contract_match:
                operand_types = re.findall(r'tensor<([^>]+)>', line_stripped)
                if len(operand_types) >= 2:
                    first_dims_str = operand_types[0]
                    first_parts = first_dims_str.replace('x', ' ').split()
                    first_dims = [int(p) for p in first_parts if p.isdigit()]
                    contract_idx = int(contract_match.group(1))
                    if contract_idx < len(first_dims):
                        k_dim = first_dims[contract_idx]
                        op_flops = float(2 * int(np.prod(dims)) * k_dim)
                    else:
                        op_flops = float(2 * int(np.prod(dims)))
                else:
                    op_flops = float(2 * int(np.prod(dims)))
            else:
                op_flops = float(2 * int(np.prod(dims)))
        elif is_elem and dims:
            op_flops = float(int(np.prod(dims)))
        elif is_reduce and dims:
            op_flops = float(int(np.prod(dims)))
        else:
            op_flops = 0.0

        op_bytes_in = 0.0
        operand_section = line_stripped[m.end():]
        operand_names = re.findall(r'%(\w+)', operand_section)
        for op_name in operand_names:
            if op_name in name_to_idx:
                src_idx = name_to_idx[op_name]
                edges.append((src_idx, idx))
                if src_idx < len(nodes):
                    op_bytes_in += nodes[src_idx]["output_bytes"]

        op_intensity = float(op_flops / max(op_bytes_in + output_bytes, 1))

        reduction_dim_size = 0
        if "dot_general" in opcode or "dot" == opcode:
            contract_match = re.search(r'contracting_dims\s*=\s*\[(\d+)\]\s*x\s*\[(\d+)\]', line_stripped)
            if contract_match:
                operand_types = re.findall(r'tensor<([^>]+)>', line_stripped)
                if operand_types:
                    first_parts = operand_types[0].replace('x', ' ').split()
                    first_dims = [int(p) for p in first_parts if p.isdigit()]
                    contract_idx = int(contract_match.group(1))
                    if contract_idx < len(first_dims):
                        reduction_dim_size = first_dims[contract_idx]
        elif is_reduce:
            # Size of the axes being collapsed, matching the depth meaning the
            # dot_general branch above stores. The output size is a different
            # quantity (how many reductions run in parallel), so using it here
            # would give this one field two incompatible meanings.
            red_dim_match = re.search(r'dimensions\s*=\s*\[([\d,\s]*)\]', line_stripped)
            if red_dim_match:
                operand_types = re.findall(r'tensor<([^>]+)>', line_stripped)
                if operand_types:
                    first_parts = operand_types[0].replace('x', ' ').split()
                    first_dims = [int(p) for p in first_parts if p.isdigit()]
                    red_idxs = [int(d) for d in red_dim_match.group(1).replace(',', " ").split()]
                    reduced = [first_dims[i] for i in red_idxs if i < len(first_dims)]
                    if reduced:
                        reduction_dim_size = int(np.prod(reduced))

        nodes.append({
            "name": name,
            "opcode": opcode,
            "is_elementwise": is_elem,
            "is_contraction": is_contract,
            "is_reduction": is_reduce,
            "is_data_movement": is_datamov,
            "output_bytes": output_bytes,
            "output_rank": rank,
            "output_dims": padded_dims,
            "op_flops": op_flops,
            "op_bytes_in": op_bytes_in,
            "op_intensity": op_intensity,
            "reduction_dim_size": reduction_dim_size,
        })

    # Structural features
    n_nodes = len(nodes)
    if n_nodes > 0:
        consumer_count = [0] * n_nodes
        for src, dst in edges:
            if src < n_nodes:
                consumer_count[src] += 1
        for i, node in enumerate(nodes):
            node["n_consumers"] = consumer_count[i]

        for i, node in enumerate(nodes):
            node["graph_depth_position"] = float(i / max(n_nodes - 1, 1))

        for i, node in enumerate(nodes):
            node["is_root"] = 1 if consumer_count[i] == 0 else 0

        adjacency = {i: set() for i in range(n_nodes)}
        for src, dst in edges:
            adjacency[src].add(dst)
            adjacency[dst].add(src)
        for i, node in enumerate(nodes):
            node["n_elementwise_neighbors"] = sum(
                1 for j in adjacency[i] if j < n_nodes and nodes[j]["is_elementwise"]
            )

        contraction_indices = {i for i, n in enumerate(nodes) if n["is_contraction"]}
        for i, node in enumerate(nodes):
            if node["is_contraction"]:
                node["dist_to_nearest_contraction"] = 0
            elif contraction_indices:
                visited = {i}
                queue = [(i, 0)]
                found = False
                while queue and not found:
                    curr, dist = queue.pop(0)
                    for neighbor in adjacency.get(curr, []):
                        if neighbor in contraction_indices:
                            node["dist_to_nearest_contraction"] = dist + 1
                            found = True
                            break
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))
                if not found:
                    node["dist_to_nearest_contraction"] = n_nodes
            else:
                node["dist_to_nearest_contraction"] = n_nodes

    total_flops = sum(n["op_flops"] for n in nodes)
    total_bytes = sum(n["output_bytes"] for n in nodes)
    n_dots = sum(1 for n in nodes if n["is_contraction"])
    dot_flops = sum(n["op_flops"] for n in nodes if n["is_contraction"])

    if edges and nodes:
        depth_cache = {}
        def get_depth(idx):
            if idx in depth_cache:
                return depth_cache[idx]
            children = [dst for src, dst in edges if src == idx]
            if not children:
                depth_cache[idx] = 0
                return 0
            d = 1 + max(get_depth(c) for c in children)
            depth_cache[idx] = d
            return d
        graph_depth = max(get_depth(i) for i in range(n_nodes)) if n_nodes else 0
    else:
        graph_depth = 0

    if n_nodes > 0:
        node_depths = [0] * n_nodes
        for src, dst in edges:
            node_depths[dst] = max(node_depths[dst], node_depths[src] + 1)
        from collections import Counter
        depth_counts = Counter(node_depths)
        graph_width = max(depth_counts.values()) if depth_counts else 1
    else:
        graph_width = 0

    graph_features = {
        "total_flops": float(total_flops),
        "total_bytes": float(total_bytes),
        "arithmetic_intensity": float(total_flops / max(total_bytes, 1)),
        "graph_depth": graph_depth,
        "graph_width": graph_width,
        "n_ops": n_nodes,
        "n_dots": n_dots,
        "dot_flops_fraction": float(dot_flops / max(total_flops, 1)),
    }

    return nodes, edges, graph_features


# ═══════════════════════════════════════════════════════════════
# Label extraction from XLA dumps
# ═══════════════════════════════════════════════════════════════

# HLO instruction names carry dots and dashes (%reduce_max.3), and a computation's
# output line is prefixed with ROOT — both must be allowed or the count comes out 0.
_HLO_INSTR_RE = r'^\s+(?:ROOT\s+)?%[\w.\-]+\s*='


def _count_hlo_ops(text):
    return len(re.findall(_HLO_INSTR_RE, text, re.MULTILINE))


def _hlo_instruction_lines(body_lines):
    """Instruction lines within a computation body ('  %name = ...')."""
    return [l for l in body_lines if re.match(_HLO_INSTR_RE, l)]


def _split_hlo_computations(text):
    """Split an HLO module into its computations.

    Returns (entry_name, {name: body_lines}). Headers sit at column 0 and end
    with '{'; the matching '}' is also at column 0. Splitting structurally
    matters because fusion bodies contain nested braces (backend_config), which
    a brace-counting regex mis-handles.
    """
    comps, entry, cur, body = {}, None, None, []
    for line in text.split("\n"):
        if cur is None:
            m = re.match(r'^(ENTRY\s+)?%([\w.\-]+)\s*\(.*\{\s*$', line)
            if m:
                cur, body = m.group(2), []
                if m.group(1):
                    entry = m.group(2)
        elif line.startswith("}"):
            comps[cur] = body
            cur = None
        else:
            body.append(line)
    return entry, comps


_DRAM_ELEM = {'pred':1,'s8':1,'u8':1,'s16':2,'u16':2,'bf16':2,'f16':2,'f8e4m3':1,'f8e5m2':1,
              's32':4,'u32':4,'f32':4,'s64':8,'u64':8,'f64':8,'c64':8,'c128':16}


def _dram_shape_bytes(dtype, dims_str):
    dims = [int(x) for x in re.findall(r'\d+', dims_str)]
    n = 1
    for d in dims:
        n *= d
    return n * _DRAM_ELEM.get(dtype, 4)


def _dram_all_shapes(type_str, skip_scratch=False):
    """Sum bytes of every dtype[...] token in a (possibly tuple) result type."""
    total = 0
    comps = []
    for dt, dims in re.findall(r'([a-z][a-z0-9]*)\[([\d,\s]*)\]', type_str):
        if skip_scratch and dt == 's8':   # cuBLAS scratch workspace, not real traffic
            continue
        b = _dram_shape_bytes(dt, dims)
        comps.append(b)
        total += b
    return total, comps


def _dram_entry_block(text):
    out, f = [], False
    for l in text.split("\n"):
        if l.startswith("ENTRY"):
            f = True
        if f:
            out.append(l)
            if l.startswith("}"):
                break
    return out


def dram_traffic_bytes(after_text):
    """Fusion-aware post-optimization DRAM traffic (bytes moved to/from HBM).

    Sums, over the top-level ENTRY kernels (fusion / custom-call), the operand bytes
    read plus result bytes written. A value produced by one kernel and consumed by
    another is counted once as a write and once per read — real traffic. Intermediates
    inside a fusion never appear at ENTRY level, so fusion's memory savings (and the
    re-reads of a large input across kernels) are captured automatically. Verified
    against hand calculations: GEMM = A+B+C, LayerNorm = input re-read Nx + output.
    """
    entry = _dram_entry_block(after_text)
    instr = re.compile(r'\s*(?:ROOT\s+)?%(\S+)\s*=\s*(.*?)\s+([a-z][a-z0-9-]*)\(')
    comp_map = {}
    for l in entry:
        m = instr.match(l)
        if m:
            _, comps = _dram_all_shapes(m.group(2))
            comp_map[m.group(1)] = comps if comps else [0]

    total = 0
    for l in entry:
        m = instr.match(l)
        if not m:
            continue
        _, type_str, opcode = m.group(1), m.group(2), m.group(3)
        if opcode not in ("fusion", "custom-call"):
            continue
        out_bytes, _ = _dram_all_shapes(type_str, skip_scratch=(opcode == "custom-call"))
        call = re.search(re.escape(opcode) + r'\(([^)]*)\)', l)
        in_bytes = 0
        if call:
            for op in call.group(1).split(","):
                om = re.match(r'%(\S+?)(?:#(\d+))?$', op.strip())
                if not om:
                    continue
                comps = comp_map.get(om.group(1), [0])
                idx = om.group(2)
                in_bytes += comps[int(idx)] if idx is not None and int(idx) < len(comps) else sum(comps)
        total += in_bytes + out_bytes
    return total


def parse_xla_dump(dump_dir):
    """Extract compiler decision labels from an XLA dump directory.

    Files are selected by exact name suffix and read in a fixed order. Substring
    matching is unsafe here: 'after_optimizations' also appears in the
    -buffer-assignment, -live-range and -memory-usage-report companion files, so
    a scan over rglob() picks up four extra files and the last one processed wins.
    """
    labels = {}
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        return labels

    files = [f for f in dump_path.rglob("*") if f.is_file()]
    if not files:
        return labels

    def pick(suffix):
        # sorted() keeps the choice deterministic when a directory holds more
        # than one module (duplicate experiment ids merge two compilations).
        hits = sorted((f for f in files if f.name.endswith(suffix)), key=lambda f: f.name)
        return hits[0] if hits else None

    f_before = pick("before_optimizations.txt")
    f_after = pick("after_optimizations.txt")
    f_buffer = pick("after_optimizations-buffer-assignment.txt")
    f_memrep = pick("after_optimizations-memory-usage-report.txt")
    f_auto = pick("autotune_results.pbtxt")
    f_thunk = pick("thunk_sequence.txt")

    # ─── Pre-optimization op count (read first; op_count_reduction needs it) ───
    ops_before = None
    if f_before:
        ops_before = _count_hlo_ops(f_before.read_text(errors="ignore"))

    # ─── Post-optimization HLO: fusion structure, op mix, precision ───
    if f_after:
        text = f_after.read_text(errors="ignore")
        entry, comps = _split_hlo_computations(text)

        # Fusion-aware DRAM traffic — the physically correct memory term for the
        # roofline. This is a training TARGET the model learns to predict; at test
        # time the model predicts it from the StableHLO graph (no compiled graph).
        labels["dram_bytes"] = dram_traffic_bytes(text)

        entry_ops = _hlo_instruction_lines(comps.get(entry, [])) if entry else []
        fusion_instrs = [l for l in entry_ops if re.search(r'=\s*\S+\s+fusion\(', l)]
        labels["n_fusions"] = len(fusion_instrs)

        called = []
        for l in fusion_instrs:
            m = re.search(r'calls=%([\w.\-]+)', l)
            if m:
                called.append(m.group(1))
        ops_per_fusion = [len(_hlo_instruction_lines(comps.get(c, []))) for c in called]
        labels["max_ops_per_fusion"] = max(ops_per_fusion) if ops_per_fusion else 0
        labels["mean_ops_per_fusion"] = float(np.mean(ops_per_fusion)) if ops_per_fusion else 0.0

        ops_after = sum(len(_hlo_instruction_lines(b)) for b in comps.values())
        labels["ops_after"] = ops_after
        labels["fused_op_ratio"] = float(sum(ops_per_fusion) / max(ops_after, 1))

        # The opcode follows the result type, as in '%c = f32[8]{0} copy(%a)'.
        labels["n_copy_ops"] = len(re.findall(r'=\s*\S+\s+copy\(', text))
        labels["n_async_ops"] = len(re.findall(r'async-start', text, re.IGNORECASE))
        labels["n_rematerialized_ops"] = len(re.findall(r'remat', text, re.IGNORECASE))

        bf16_ops = len(re.findall(r'bf16', text))
        f32_ops = len(re.findall(r'(?<!\w)f32(?!\w)', text))
        f16_ops = len(re.findall(r'(?<!b)(?<!\w)f16(?!\w)', text))
        total_typed = bf16_ops + f32_ops + f16_ops
        if total_typed > 0:
            labels["precision_bf16_fraction"] = float(bf16_ops / total_typed)
            labels["precision_f16_fraction"] = float(f16_ops / total_typed)
            labels["precision_f32_fraction"] = float(f32_ops / total_typed)
        if bf16_ops > f32_ops:
            labels["compute_precision"] = "bf16"
        elif f16_ops > f32_ops:
            labels["compute_precision"] = "f16"
        else:
            labels["compute_precision"] = "f32"

        if ops_before:
            labels["ops_before"] = ops_before
            labels["op_count_reduction"] = float(1.0 - ops_after / ops_before)

    # ─── Memory: authoritative total from the memory-usage report ───
    if f_memrep:
        m = re.search(r'Total bytes:\s*(\d+)', f_memrep.read_text(errors="ignore"))
        if m:
            labels["peak_memory_bytes"] = int(m.group(1))

    # ─── Buffer assignment: logical values packed into physical allocations ───
    if f_buffer:
        text = f_buffer.read_text(errors="ignore")
        alloc_sizes = [int(x) for x in re.findall(r'^allocation \d+: size (\d+)', text, re.MULTILINE)]
        logical = len(re.findall(r'^\s*value: <', text, re.MULTILINE))
        # The allocation sizes sum to exactly the memory report's 'Total bytes',
        # so only the buffer counts are recorded here to avoid a duplicate column.
        if alloc_sizes:
            labels["n_physical_buffers"] = len(alloc_sizes)
        if logical:
            labels["n_logical_buffers"] = logical
        if alloc_sizes and logical:
            labels["buffer_reuse_ratio"] = float(logical / max(len(alloc_sizes), 1))

    # ─── Autotuning: tile shape and codegen backend ───
    if f_auto:
        text = f_auto.read_text(errors="ignore")
        triton_match = re.search(
            r'block_m:\s*(\d+).*?block_n:\s*(\d+).*?block_k:\s*(\d+)', text, re.DOTALL)
        if triton_match:
            labels["tile_m"] = int(triton_match.group(1))
            labels["tile_n"] = int(triton_match.group(2))
            labels["tile_k"] = int(triton_match.group(3))
            labels["codegen_type"] = "triton"
        low = text.lower()
        if "cublas" in low and "codegen_type" not in labels:
            labels["codegen_type"] = "cublas"
        if "rocblas" in low:
            labels["codegen_type"] = "rocblas"

    # ─── Thunks: one launch record per line ('001: kCustomKernel [...]') ───
    if f_thunk:
        text = f_thunk.read_text(errors="ignore")
        thunk_lines = re.findall(r'^\s*\d+:\s*k\w+', text, re.MULTILINE)
        if thunk_lines:
            labels["n_kernels"] = len(thunk_lines)
        else:
            kernels = re.findall(
                r'(?:GemmThunk|KernelThunk|CustomKernelThunk|ConvolutionThunk|TritonKernel|kCustomKernel)',
                text, re.IGNORECASE)
            labels["n_kernels"] = len(kernels)

    return labels

# ═══════════════════════════════════════════════════════════════
# Buffer rotation + benchmark
# ═══════════════════════════════════════════════════════════════

def _tensor_bytes(x):
    return x.size * x.dtype.itemsize

def make_rotated_buffers(args, n_copies=4):
    buffers = [args]
    for _ in range(n_copies - 1):
        copy = tuple(jnp.array(a) for a in args)
        for a in copy:
            a.block_until_ready()
        buffers.append(copy)
    total_bytes = sum(_tensor_bytes(a) for a in args) * n_copies
    return buffers, total_bytes


def benchmark_fn(fn, args, n_warmup=10, n_runs=100):
    """Benchmark with buffer rotation, IQR outlier removal, wall-clock timing."""
    buffers, total_bytes = make_rotated_buffers(args, n_copies=4)

    for i in range(n_warmup):
        result = fn(*buffers[i % len(buffers)])
        result.block_until_ready()

    times_us = []
    for i in range(n_runs):
        cur_args = buffers[i % len(buffers)]
        t0 = time.perf_counter()
        result = fn(*cur_args)
        result.block_until_ready()
        t1 = time.perf_counter()
        times_us.append((t1 - t0) * 1e6)

    times = np.array(times_us)
    q1, q3 = np.percentile(times, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        clean = times[(times >= q1 - 1.5 * iqr) & (times <= q3 + 1.5 * iqr)]
    else:
        clean = times

    return {
        "mean_us": float(np.mean(clean)),
        "median_us": float(np.median(clean)),
        "std_us": float(np.std(clean)),
        "cv_percent": float(np.std(clean) / np.mean(clean) * 100) if np.mean(clean) > 0 else 0,
        "n_outliers": int(len(times) - len(clean)),
        "buffer_total_bytes": total_bytes,
    }


# ═══════════════════════════════════════════════════════════════
# GPU info
# ═══════════════════════════════════════════════════════════════

def get_gpu_info():
    """Identify the device JAX is actually running on.

    The name comes from JAX rather than from CUDA_VISIBLE_DEVICES + nvidia-smi:
    device ordering can differ between the two, so an env-var lookup can name a
    card we are not using. nvidia-smi supplies temperature/clock only, and only
    once its name is confirmed to match.
    """
    try:
        gpu_name = jax.local_devices()[0].device_kind
    except Exception:
        return {"gpu_name": "unknown"}

    info = {"gpu_name": gpu_name}
    try:
        # nvidia-smi ignores CUDA_VISIBLE_DEVICES and lists every GPU, so ask for
        # the visible one explicitly and read only its line.
        cmd = ["nvidia-smi", "--query-gpu=name,temperature.gpu,clocks.sm",
               "--format=csv,noheader"]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        if visible:
            cmd += ["-i", visible]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        parts = [p.strip() for p in result.stdout.strip().split("\n")[0].split(", ")]
        if parts[0] == gpu_name:
            info["temperature_c"] = int(parts[1])
            info["sm_clock_mhz"] = parts[2]
        else:
            print(f"WARNING: nvidia-smi reports '{parts[0]}' but JAX is on "
                  f"'{gpu_name}' — skipping temperature/clock.")
    except Exception:
        pass
    return info


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Collect training data (multi-dim sweep)")
    parser.add_argument("--workload", type=str, default="all",
                        help="Workload type or 'all'")
    parser.add_argument("--max-configs", type=int, default=0,
                        help="Max configs to run (0 = all). Use for time-limited runs.")
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--n-trials", type=int, default=2)
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for config shuffling")
    parser.add_argument("--per-workload", type=int, default=0,
                        help="Keep at most N configs per workload so every workload is "
                             "equally represented (0 = keep all).")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or f"data/{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graphs").mkdir(exist_ok=True)

    xla_flags = os.environ.get("XLA_FLAGS", "")
    has_dump = "--xla_dump_to" in xla_flags
    if has_dump:
        dump_base = re.search(r'--xla_dump_to=(\S+)', xla_flags).group(1)
        print(f"XLA dump enabled: {dump_base}")
    else:
        print("WARNING: XLA_FLAGS not set — no compiler labels will be collected.")
        dump_base = None

    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info['gpu_name']} ({gpu_info.get('sm_clock_mhz', '?')} MHz, {gpu_info.get('temperature_c', '?')}C)")
    print(f"Output: {out_dir}")

    # Build config list
    registry = make_workload_registry()
    all_experiments = []  # list of (workload_name, fn, input_builder, flops_fn, config)

    for wl_name, (fn, input_builder, flops_fn, configs) in registry.items():
        if args.workload != "all" and args.workload != wl_name:
            continue
        for cfg in configs:
            mem_est = estimate_memory(wl_name, cfg)
            if mem_est > MAX_GPU_BYTES:
                continue
            all_experiments.append((wl_name, fn, input_builder, flops_fn, cfg))

    # Shuffle for diversity (so partial runs cover all workload types)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(all_experiments)

    # Balance: the list is already shuffled, so taking the first N of each workload
    # is a random sample of that workload's config space.
    if args.per_workload > 0:
        seen, balanced = {}, []
        for e in all_experiments:
            n = seen.get(e[0], 0)
            if n < args.per_workload:
                seen[e[0]] = n + 1
                balanced.append(e)
        short = {w: n for w, n in seen.items() if n < args.per_workload}
        if short:
            print(f"WARNING: fewer than {args.per_workload} configs available for: {short}")
        all_experiments = balanced

    if args.max_configs > 0:
        all_experiments = all_experiments[:args.max_configs]

    n_total = len(all_experiments)
    print(f"Configs: {n_total} experiments to run")
    if args.workload == "all":
        # Print per-workload counts
        from collections import Counter
        wl_counts = Counter(e[0] for e in all_experiments)
        for wl, cnt in sorted(wl_counts.items()):
            print(f"  {wl}: {cnt}")
    print()

    dataset = []
    key = random.PRNGKey(args.seed)
    n_skipped = 0
    t_start = time.time()

    for exp_idx, (wl_name, fn, input_builder, flops_fn, cfg) in enumerate(all_experiments):
        exp_id = config_to_id(wl_name, cfg)
        elapsed = time.time() - t_start
        rate = (exp_idx + 1) / max(elapsed, 1) * 3600  # experiments/hour
        eta_h = (n_total - exp_idx - 1) / max(rate, 1)

        print(f"\n[{exp_idx + 1}/{n_total}] {exp_id}  (ETA: {eta_h:.1f}h)")

        try:
            # Build inputs
            fn_args = input_builder(cfg, key)

            # Step 1: StableHLO features
            jax.clear_caches()
            if dump_base and os.path.exists(dump_base):
                shutil.rmtree(dump_base)

            jit_fn = jax.jit(fn)
            lowered = jit_fn.lower(*fn_args)
            stablehlo_text = lowered.as_text(dialect="stablehlo")
            (out_dir / "graphs" / f"{exp_id}.stablehlo.txt").write_text(stablehlo_text)
            nodes, edges, graph_features = extract_node_features(stablehlo_text)

            # Step 2: Compile + XLA dump. Calling jit_fn compiles on its own; an extra
            # lowered.compile() here would compile a second time and emit a duplicate
            # set of dump files, making label parsing depend on rglob ordering.
            result = jit_fn(*fn_args)
            result.block_until_ready()

            xla_labels = {}
            if dump_base:
                xla_labels = parse_xla_dump(dump_base)
                exp_dump_dir = out_dir / "xla_dumps" / exp_id
                if os.path.exists(dump_base):
                    shutil.copytree(dump_base, exp_dump_dir, dirs_exist_ok=True)

            # Step 3: Benchmark
            trial_medians = []
            trial_results = []
            for t in range(args.n_trials):
                timing = benchmark_fn(jit_fn, fn_args,
                                      n_warmup=args.n_warmup,
                                      n_runs=args.n_runs)
                trial_medians.append(timing["median_us"])
                trial_results.append(timing)

            trial_medians_arr = np.array(trial_medians)
            trial_spread = (np.max(trial_medians_arr) - np.min(trial_medians_arr)) / np.median(trial_medians_arr) * 100
            best_trial_idx = np.argmin([r["cv_percent"] for r in trial_results])
            timing = trial_results[best_trial_idx]

            est_flops = flops_fn(cfg)
            final_latency = float(np.median(trial_medians_arr))
            achieved_tflops = est_flops / (final_latency * 1e-6) / 1e12 if final_latency > 0 else 0

            cv_flag = " !!!" if timing["cv_percent"] > 5 else ""
            print(f"  {len(nodes)} nodes | {final_latency:.1f} us | CV={timing['cv_percent']:.1f}%{cv_flag} | {achieved_tflops:.2f} TFLOP/s | {len(xla_labels)} labels")

            # Serialize config (convert dtype to string)
            cfg_serializable = {}
            for k, v in cfg.items():
                if k == "dtype":
                    cfg_serializable[k] = dtype_str(v)
                else:
                    cfg_serializable[k] = v

            row = {
                "experiment_id": exp_id,
                "workload": wl_name,
                "config": cfg_serializable,
                "gpu_name": gpu_info["gpu_name"],

                **{f"graph_{k}": v for k, v in graph_features.items()},

                "n_nodes": len(nodes),
                "n_edges": len(edges),
                "mean_output_bytes": float(np.mean([n["output_bytes"] for n in nodes])) if nodes else 0,
                "max_output_bytes": float(np.max([n["output_bytes"] for n in nodes])) if nodes else 0,
                "mean_op_flops": float(np.mean([n["op_flops"] for n in nodes])) if nodes else 0,
                "mean_op_intensity": float(np.mean([n["op_intensity"] for n in nodes])) if nodes else 0,
                "n_elementwise": sum(1 for n in nodes if n["is_elementwise"]),
                "n_contractions": sum(1 for n in nodes if n["is_contraction"]),
                "n_reductions": sum(1 for n in nodes if n["is_reduction"]),
                "n_data_movement": sum(1 for n in nodes if n["is_data_movement"]),

                **{f"label_{k}": v for k, v in xla_labels.items()},

                "est_flops": est_flops,
                "latency_us": final_latency,
                "latency_mean_us": timing["mean_us"],
                "latency_std_us": timing["std_us"],
                "latency_cv_percent": timing["cv_percent"],
                "trial_spread_percent": trial_spread,
                "trial_medians_us": trial_medians,
                "achieved_tflops": achieved_tflops,
            }
            dataset.append(row)

            with open(out_dir / "graphs" / f"{exp_id}.graph.json", "w") as f:
                json.dump({"nodes": nodes, "edges": edges,
                           "graph_features": graph_features}, f, indent=2)

        except Exception as e:
            print(f"  SKIPPED: {e}")
            n_skipped += 1

        if args.cooldown > 0:
            time.sleep(args.cooldown)

        # Save incrementally every 50 experiments
        if len(dataset) % 50 == 0 and dataset:
            with open(out_dir / "dataset.json", "w") as f:
                json.dump(dataset, f, indent=2)

    # Final save
    with open(out_dir / "dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    # Summary
    elapsed_total = time.time() - t_start
    print(f"\n{'=' * 90}")
    print(f"  Collected {len(dataset)} experiments in {elapsed_total / 3600:.1f} hours ({n_skipped} skipped)")
    print(f"  Dataset: {out_dir / 'dataset.json'}")
    print(f"  Graphs:  {out_dir / 'graphs/'}")
    if dump_base:
        print(f"  XLA:     {out_dir / 'xla_dumps/'}")

    if dataset:
        from collections import Counter
        wl_counts = Counter(r["workload"] for r in dataset)
        print(f"\n  Per-workload:")
        for wl, cnt in sorted(wl_counts.items()):
            print(f"    {wl}: {cnt}")


if __name__ == "__main__":
    main()
