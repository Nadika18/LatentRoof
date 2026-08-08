#!/usr/bin/env python3
"""Regenerate poster figures for end-to-end MAPE.

Run from end_end_prediction/:
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


def load(mode: str, gpu: str) -> dict:
    return json.loads((ART / f"compose_{mode}_{gpu}.json").read_text())


def mean_ape(rows, model, batch=None) -> float:
    sel = [r["ape"] for r in rows if r["model"] == model and (batch is None or r["batch"] == batch)]
    return float(np.mean(sel)) if sel else float("nan")


def style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def fig_batch_split():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
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
        bars1 = ax.bar(x - width / 2, b1, width, label="Batch 1", color=colors["B1"], edgecolor="white", lw=0.5)
        bars8 = ax.bar(x + width / 2, b8, width, label="Batch 8", color=colors["B8"], edgecolor="white", lw=0.5)
        if handles is None:
            handles = [bars1, bars8]
        ax.axhline(overall, color="#444444", ls="--", lw=1.2, alpha=0.85)
        ax.text(0.98, overall + 2.5, f"Overall {overall:.1f}%",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=9, color="#333333")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in MODELS])
        if ax is axes[0]:
            ax.set_ylabel("Prediction error (%)")
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, ls=":", alpha=0.5)
        ax.set_axisbelow(True)
    fig.legend(handles, ["Batch 1", "Batch 8"], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("End-to-end latency prediction error", fontsize=13, fontweight="bold", y=1.16)
    fig.tight_layout()
    path = OUT / "e2e_mape_by_model_batch_amortize_launch.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    print(fig_batch_split())


if __name__ == "__main__":
    main()
