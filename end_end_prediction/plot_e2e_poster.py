#!/usr/bin/env python3
"""Regenerate poster figures for end-to-end MAPE.

All MAPE bar charts use a shared 0–40 y-axis (same convention as operator LOGO plots).
Figures are sized for poster readability.

Run from end_end_prediction/ (or cursor_ablation_dual_peak_E/):
  python3 plot_e2e_poster.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
OUT = ART / "poster_figs"

MODELS = ["gpt2_small", "gpt2_medium", "gpt2_large", "bert_base", "bert_large"]
LABELS = {
    "gpt2_small": "GPT-2\nSmall",
    "gpt2_medium": "GPT-2\nMedium",
    "gpt2_large": "GPT-2\nLarge",
    "bert_base": "BERT\nBase",
    "bert_large": "BERT\nLarge",
}
YMAX = 40


def load(mode: str, gpu: str) -> dict:
    return json.loads((ART / f"compose_{mode}_{gpu}.json").read_text())


def mean_ape(rows, model, batch=None) -> float:
    sel = [
        r["ape"]
        for r in rows
        if r["model"] == model and (batch is None or r["batch"] == batch)
    ]
    return float(np.mean(sel)) if sel else float("nan")


def style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def fig_batch_split():
    """RTX | H200 panels: Batch 1 vs Batch 8, amort 0–100."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
    colors = {"B1": "#2C5F8A", "B8": "#C46B2D"}
    x = np.arange(len(MODELS))
    width = 0.36
    handles = None
    for ax, gpu, title in zip(axes, ["rtx", "h200"], ["RTX PRO 6000", "H200 NVL"]):
        data = load("amortize_launch", gpu)
        rows = data["rows"]
        overall = data["summary"]["mape"]
        b1 = [mean_ape(rows, m, 1) for m in MODELS]
        b8 = [mean_ape(rows, m, 8) for m in MODELS]
        bars1 = ax.bar(
            x - width / 2, b1, width, label="Batch 1",
            color=colors["B1"], edgecolor="white", lw=0.5,
        )
        bars8 = ax.bar(
            x + width / 2, b8, width, label="Batch 8",
            color=colors["B8"], edgecolor="white", lw=0.5,
        )
        if handles is None:
            handles = [bars1, bars8]
        ax.axhline(overall, color="#444444", ls="--", lw=1.4, alpha=0.85)
        ax.text(
            0.98, overall + 3.0, f"Overall {overall:.1f}%",
            transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=11, color="#333333",
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in MODELS])
        if ax is axes[0]:
            ax.set_ylabel("Prediction error (%)")
        ax.set_ylim(0, YMAX)
        ax.set_yticks(np.arange(0, YMAX + 1, 10))
        ax.yaxis.grid(True, ls=":", alpha=0.5)
        ax.set_axisbelow(True)
    fig.legend(
        handles, ["Batch 1", "Batch 8"],
        loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05),
    )
    fig.suptitle(
        "End-to-end latency prediction error",
        fontsize=16, fontweight="bold", y=1.10,
    )
    fig.tight_layout()
    path = OUT / "e2e_mape_by_model_batch_amortize_launch.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_by_model_gpus():
    """Per-model MAPE: RTX vs H200 (amortize_launch), ylim 0–40."""
    fig, ax = plt.subplots(figsize=(14, 6.5))
    colors = {"rtx": "#2C5F8A", "h200": "#C46B2D"}
    x = np.arange(len(MODELS))
    width = 0.36
    series = {}
    for gpu in ("rtx", "h200"):
        rows = load("amortize_launch", gpu)["rows"]
        series[gpu] = [mean_ape(rows, m) for m in MODELS]
    bars_r = ax.bar(
        x - width / 2, series["rtx"], width,
        label="RTX PRO 6000", color=colors["rtx"], edgecolor="white", lw=0.5,
    )
    bars_h = ax.bar(
        x + width / 2, series["h200"], width,
        label="H200 NVL", color=colors["h200"], edgecolor="white", lw=0.5,
    )
    ax.bar_label(bars_r, fmt="%.0f", fontsize=11, padding=3, color=colors["rtx"])
    ax.bar_label(bars_h, fmt="%.0f", fontsize=11, padding=3, color=colors["h200"])
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in MODELS])
    ax.set_ylabel("Prediction error (%)")
    ax.set_ylim(0, YMAX)
    ax.set_yticks(np.arange(0, YMAX + 1, 10))
    ax.set_title("End-to-end latency prediction error by model", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.yaxis.grid(True, ls=":", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = OUT / "e2e_mape_by_model_gpus.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_naive_vs_amortize():
    """Per-model MAPE: naive vs amortize_launch, one panel per GPU, ylim 0–60."""
    ymax = 60
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
    colors = {"naive": "#6B7280", "amortize_launch": "#2C5F8A"}
    x = np.arange(len(MODELS))
    width = 0.36
    handles = None
    for ax, gpu, title in zip(axes, ["rtx", "h200"], ["RTX PRO 6000", "H200 NVL"]):
        vals = {}
        for mode in ("naive", "amortize_launch"):
            rows = load(mode, gpu)["rows"]
            vals[mode] = [mean_ape(rows, m) for m in MODELS]
        b0 = ax.bar(
            x - width / 2, vals["naive"], width,
            label="Naive", color=colors["naive"], edgecolor="white", lw=0.5,
        )
        b1 = ax.bar(
            x + width / 2, vals["amortize_launch"], width,
            label="Amortize launch", color=colors["amortize_launch"],
            edgecolor="white", lw=0.5,
        )
        if handles is None:
            handles = [b0, b1]
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in MODELS])
        if ax is axes[0]:
            ax.set_ylabel("Prediction error (%)")
        ax.set_ylim(0, ymax)
        ax.set_yticks(np.arange(0, ymax + 1, 10))
        ax.yaxis.grid(True, ls=":", alpha=0.5)
        ax.set_axisbelow(True)
    fig.legend(
        handles, ["Naive", "Amortize launch"],
        loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05),
    )
    fig.suptitle(
        "Naive vs launch-amortized composition",
        fontsize=16, fontweight="bold", y=1.10,
    )
    fig.tight_layout()
    path = OUT / "e2e_mape_naive_vs_amortize_by_model.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (fig_batch_split(), fig_by_model_gpus(), fig_naive_vs_amortize()):
        print(path)


if __name__ == "__main__":
    main()
