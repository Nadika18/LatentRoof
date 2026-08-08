#!/usr/bin/env python3
"""End-to-end ground-truth latency for GPT-2 / BERT transformer backbones.

Measures the *actual* full forward latency of a stack of N transformer blocks
(the same pre-norm LN -> MHA -> residual -> LN -> GELU-FFN -> residual block used
in collect_data.py's `transformer` family), so it composes exactly from our
per-block predictions. Reports wall-clock latency under the same protocol as the
operator collection (input-buffer rotation for cache eviction, warm-up, IQR
filtering, two trials of 100 timed runs).

Writes data/e2e_groundtruth_<timestamp>/results.json — every row keyed by
gpu_name (from JAX), so H200/RTX/GB10 runs stay distinguishable.

Run (needs a live GPU; matches collect_data conventions):
  CUDA_VISIBLE_DEVICES=1 TMPDIR=$PWD/.xla_tmp python measure_e2e_groundtruth.py
"""
import os, sys, json, time
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
from datetime import datetime
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax

import collect_data as cd  # reuse get_gpu_info only

DTYPES = {"bf16": jnp.bfloat16, "f16": jnp.float16, "f32": jnp.float32}

# name -> (hidden, n_layers, ff, n_heads)   [head_dim = 64 throughout]
MODELS = {
    "gpt2_small":  (768, 12, 3072, 12),
    "gpt2_medium": (1024, 24, 4096, 16),
    "gpt2_large":  (1280, 36, 5120, 20),
    "bert_base":   (768, 12, 3072, 12),
    "bert_large":  (1024, 24, 4096, 16),
}
SEQS = [512, 1024]
BATCHES = [1, 8]
PRECS = ["bf16", "f16"]


def _block(x, g1, b1, wq, wk, wv, wo, g2, b2, w1, bb1, w2, bb2):
    """One transformer block — identical to collect_data.py's transformer family."""
    B, S, D = x.shape
    H = D // 64
    Dh = D // H
    m1 = jnp.mean(x, -1, keepdims=True); v1 = jnp.var(x, -1, keepdims=True)
    xln = g1 * (x - m1) / jnp.sqrt(v1 + 1e-5) + b1
    flat = xln.reshape(B * S, D)
    Q = jnp.dot(flat, wq).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
    K = jnp.dot(flat, wk).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
    V = jnp.dot(flat, wv).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
    sc = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(Dh).astype(x.dtype)
    w = jax.nn.softmax(sc, -1)
    ao = jnp.matmul(w, V).transpose(0, 2, 1, 3).reshape(B * S, D)
    ao = jnp.dot(ao, wo).reshape(B, S, D)
    x = x + ao
    m2 = jnp.mean(x, -1, keepdims=True); v2 = jnp.var(x, -1, keepdims=True)
    xln2 = g2 * (x - m2) / jnp.sqrt(v2 + 1e-5) + b2
    h = jax.nn.gelu(jnp.dot(xln2.reshape(B * S, D), w1) + bb1)
    ffn = jnp.dot(h, w2) + bb2
    return x + ffn.reshape(B, S, D)


def backbone(x, layers, fg, fb):
    """Stack of N identical blocks (via scan over stacked weights) + final LN."""
    def step(h, lp):
        return _block(h, *lp), None
    x, _ = lax.scan(step, x, layers)
    m = jnp.mean(x, -1, keepdims=True); v = jnp.var(x, -1, keepdims=True)
    return fg * (x - m) / jnp.sqrt(v + 1e-5) + fb


def init_model(cfg, key):
    D, N, FF, _H = cfg
    ks = random.split(key, 14)
    def stk(k, shape, i):  # N stacked copies of a weight
        return random.normal(k, (N, *shape), dtype=jnp.float32) * 0.02
    layers = (
        jnp.ones((N, D)), jnp.zeros((N, D)),                 # g1, b1
        stk(ks[0], (D, D), 0), stk(ks[1], (D, D), 1),
        stk(ks[2], (D, D), 2), stk(ks[3], (D, D), 3),        # wq,wk,wv,wo
        jnp.ones((N, D)), jnp.zeros((N, D)),                 # g2, b2
        stk(ks[4], (D, FF), 4), jnp.zeros((N, FF)),          # w1, b1
        stk(ks[5], (FF, D), 5), jnp.zeros((N, D)),           # w2, b2
    )
    return layers, jnp.ones((D,)), jnp.zeros((D,))


def bench_input_rotated(fn, x, n_warmup=10, n_runs=100, n_copies=4):
    """Same protocol as collect_data.benchmark_fn but rotates ONLY the input x
    (weights stay resident, as in real inference)."""
    bufs = [x] + [jnp.array(x) for _ in range(n_copies - 1)]
    for b in bufs: b.block_until_ready()
    for i in range(n_warmup):
        fn(bufs[i % n_copies]).block_until_ready()
    t = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        fn(bufs[i % n_copies]).block_until_ready()
        t.append((time.perf_counter() - t0) * 1e6)
    a = np.array(t)
    q1, q3 = np.percentile(a, [25, 75]); iqr = q3 - q1
    clean = a[(a >= q1 - 1.5 * iqr) & (a <= q3 + 1.5 * iqr)] if iqr > 0 else a
    return float(np.median(clean)), float(np.std(clean) / max(np.mean(clean), 1e-9) * 100)


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(f"data/e2e_groundtruth_{ts}"); out.mkdir(parents=True, exist_ok=True)
    gpu = cd.get_gpu_info()["gpu_name"]
    print(f"GPU: {gpu}\nOutput: {out}")
    key = random.PRNGKey(0)
    rows = []
    jobs = [(m, cfg, S, B, p) for m, cfg in MODELS.items()
            for S in SEQS for B in BATCHES for p in PRECS]
    for idx, (mname, cfg, S, B, prec) in enumerate(jobs):
        D, N, FF, H = cfg
        dt = DTYPES[prec]
        try:
            layers, fg, fb = init_model(cfg, key)
            layers = tuple(w.astype(dt) for w in layers)
            fg, fb = fg.astype(dt), fb.astype(dt)
            x = random.normal(key, (B, S, D), dtype=dt)
            fwd = jax.jit(lambda xx: backbone(xx, layers, fg, fb))
            fwd(x).block_until_ready()  # compile
            med_us, cv = bench_input_rotated(fwd, x)
            trial2, _ = bench_input_rotated(fwd, x)
            med_us = float(np.median([med_us, trial2]))
            row = dict(model=mname, hidden=D, n_layers=N, ff=FF, n_heads=H,
                       seq=S, batch=B, dtype=prec, gpu_name=gpu,
                       latency_us=med_us, cv_percent=cv)
            rows.append(row)
            print(f"[{idx+1}/{len(jobs)}] {mname} B{B} S{S} {prec}: "
                  f"{med_us/1000:.2f} ms  CV={cv:.1f}%")
            json.dump(rows, open(out / "results.json", "w"), indent=2)
        except Exception as e:
            print(f"[{idx+1}/{len(jobs)}] {mname} B{B} S{S} {prec}: SKIP ({type(e).__name__}: {e})")
    print(f"\nDone. {len(rows)} measurements -> {out/'results.json'}")


if __name__ == "__main__":
    main()
