#!/usr/bin/env python3
"""Compose per-block predictions into a full-model latency and grade vs ground truth.

Default (naive):
  predict(backbone) = N * predict(transformer_block) + predict(final LayerNorm)

Overhead-amortized modes:
  amortize_residual: N * physics + (block - physics) + LN
  amortize_launch:   N * (block - launch) + launch + LN
  amortize_both:     N * (physics - launch) + launch + residual + LN

Usage (from end_end_prediction/):
  python3 compose.py \\
      --checkpoint artifacts/e2e_all_h128.pt \\
      --groundtruth data/e2e_groundtruth/rtx_pro_6000.json \\
      --coverage-graphs data/e2e_coverage_graphs \\
      --compose-mode amortize_launch \\
      --output artifacts/compose_amortize_launch_rtx.json

Raw XLA dump trees are not required — only StableHLO graphs + ground-truth JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from schedule_free_perf.contracts import load_hardware
from schedule_free_perf.model import batch_graphs, hardware_tensor, physics_tensor
from schedule_free_perf.stablehlo import parse_stablehlo_file
from schedule_free_perf.training import load_checkpoint

HW_JSON = {
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": "rtx_pro_6000.json",
    "NVIDIA H200 NVL": "h200.json",
    "NVIDIA H200": "h200.json",
    "NVIDIA GB10": "gb10.json",
}

COMPOSE_MODES = ("naive", "amortize_residual", "amortize_launch", "amortize_both")


def block_id(b, dt, ff, d, s):
    return f"transformer_batch{b}_{dt}_ff_dim{ff}_hidden{d}_seq{s}"


def ln_id(b, dt, d, s):
    return f"layernorm_batch{b}_{dt}_hidden{d}_seq{s}"


def predict_detail(model, graph_path, gid, hw):
    g = parse_stablehlo_file(str(graph_path), gid)
    with torch.no_grad():
        p = model(batch_graphs([g]), hardware_tensor([hw]), physics_tensor([g], [hw]))
    latency = float(p.latency_us[0].cpu())
    physics = float(p.physics_us[0].cpu())
    if "log_n_kernels" in p.auxiliary:
        kernel_count = float(torch.exp(p.auxiliary["log_n_kernels"][0]).cpu())
    else:
        kernel_count = 1.0
    launch_time = kernel_count * float(hw.launch_overhead_us)
    residual = max(latency - physics, 0.0)
    return {
        "latency_us": latency,
        "physics_us": physics,
        "launch_time_us": launch_time,
        "residual_us": residual,
        "kernel_count": kernel_count,
    }


def compose_pred(N: int, block: dict, ln_us: float, mode: str) -> float:
    L = block["latency_us"]
    P = block["physics_us"]
    launch = block["launch_time_us"]
    residual = block["residual_us"]
    if mode == "naive":
        return N * L + ln_us
    if mode == "amortize_residual":
        return N * P + residual + ln_us
    if mode == "amortize_launch":
        return N * max(L - launch, 0.0) + launch + ln_us
    if mode == "amortize_both":
        return N * max(P - launch, 0.0) + launch + residual + ln_us
    raise ValueError(f"unknown compose mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--groundtruth", required=True)
    ap.add_argument("--coverage-graphs", required=True)
    ap.add_argument("--hardware-dir", default="hardware")
    ap.add_argument("--output")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compose-mode", default="naive", choices=COMPOSE_MODES)
    a = ap.parse_args()

    model, _ = load_checkpoint(a.checkpoint, a.device)
    model.eval()
    graphs = Path(a.coverage_graphs)
    hwdir = Path(a.hardware_dir)
    gt = json.load(open(a.groundtruth))

    rows, cache = [], {}
    for r in gt:
        gpu = r["gpu_name"]
        hw = load_hardware(str(hwdir / HW_JSON[gpu]))
        B, S, D, FF, N, dt = r["batch"], r["seq"], r["hidden"], r["ff"], r["n_layers"], r["dtype"]
        bid, lid = block_id(B, dt, FF, D, S), ln_id(B, dt, D, S)
        bpath, lpath = graphs / f"{bid}.stablehlo.txt", graphs / f"{lid}.stablehlo.txt"
        if not bpath.is_file():
            print(f"SKIP {r['model']} B{B} S{S} {dt}: no block graph {bid}")
            continue
        key = (gpu, bid)
        if key not in cache:
            cache[key] = predict_detail(model, bpath, bid, hw)
        block = cache[key]
        ln_us = predict_detail(model, lpath, lid, hw)["latency_us"] if lpath.is_file() else 0.0
        pred = compose_pred(N, block, ln_us, a.compose_mode)
        meas = r["latency_us"]
        ape = abs(pred - meas) / meas * 100.0
        rows.append(
            dict(
                model=r["model"], gpu=gpu, batch=B, seq=S, dtype=dt, n_layers=N,
                compose_mode=a.compose_mode,
                block_us=block["latency_us"], pred_us=pred, meas_us=meas, ape=ape,
            )
        )
        print(
            f"{r['model']:12s} B{B} S{S} {dt}: pred {pred/1000:7.2f} ms | "
            f"meas {meas/1000:7.2f} ms | APE {ape:5.1f}%"
        )

    apes = np.array([x["ape"] for x in rows])
    summary = dict(
        checkpoint=a.checkpoint, groundtruth=a.groundtruth, compose_mode=a.compose_mode,
        n=len(rows), mape=float(apes.mean()), median_ape=float(np.median(apes)),
        p90_ape=float(np.percentile(apes, 90)),
        within_15pct=float((apes <= 15).mean() * 100),
        within_25pct=float((apes <= 25).mean() * 100),
    )
    per_model = {}
    for m in sorted(set(x["model"] for x in rows)):
        ma = np.array([x["ape"] for x in rows if x["model"] == m])
        per_model[m] = dict(n=len(ma), mape=float(ma.mean()), median_ape=float(np.median(ma)))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(json.dumps(per_model, indent=2))
    if a.output:
        json.dump(dict(summary=summary, per_model=per_model, rows=rows), open(a.output, "w"), indent=2)
        print(f"-> {a.output}")


if __name__ == "__main__":
    main()
