#!/usr/bin/env python3
"""Figure 8: Training throughput comparison bar chart."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "axes.linewidth": 0.9,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

KERAS_C   = "#4C72B0"
PYTORCH_C = "#DD8452"

# Data: (label, keras_mean, keras_std, pytorch_mean, pytorch_std)
DATA = [
    ("ResNet-50\n(PV)",      97.9, 31.9,  124.9, 0.9),
    ("VGG-16\n(PV)",        72.8, 0.1,   76.3, 0.6),
    ("MobileNetV2\n(PV)",   210.3, 23.7,  179.3, 17.9),
    ("ResNet-50\n(C10)",    213.9, 0.6,   113.7, 0.2),
    ("VGG-16\n(C10)",       137.4, 0.3,   65.7, 0.1),
    ("MobileNetV2\n(C10)",  369.4, 1.0,   160.8, 0.3),
]


def main():
    labels = [d[0] for d in DATA]
    k_mean = [d[1] for d in DATA]
    k_std  = [d[2] for d in DATA]
    p_mean = [d[3] for d in DATA]
    p_std  = [d[4] for d in DATA]

    n = len(labels)
    x = np.arange(n)
    w = 0.34

    fig, ax = plt.subplots(figsize=(11, 6.4), dpi=300)
    fig.patch.set_facecolor("white")

    bars_k = ax.bar(x - w/2, k_mean, w, yerr=k_std,
                    color=KERAS_C, edgecolor="black", linewidth=0.8,
                    capsize=4, error_kw={"lw": 1.0, "capthick": 1.0},
                    label="Keras", zorder=3)
    bars_p = ax.bar(x + w/2, p_mean, w, yerr=p_std,
                    color=PYTORCH_C, edgecolor="black", linewidth=0.8,
                    capsize=4, error_kw={"lw": 1.0, "capthick": 1.0},
                    label="PyTorch", zorder=3)

    # Value labels on top
    for bar, val in zip(bars_k, k_mean):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 12,
                f"{val:.0f}", ha="center", va="bottom",
                fontsize=8, color=KERAS_C, fontweight="bold")
    for bar, val in zip(bars_p, p_mean):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 12,
                f"{val:.0f}", ha="center", va="bottom",
                fontsize=8, color=PYTORCH_C, fontweight="bold")

    # Vertical separator
    ax.axvline(2.5, color="#CCCCCC", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(1.0, 420, "PlantVillage (PV)", ha="center", fontsize=11,
            color="#666", fontstyle="italic", fontweight="bold")
    ax.text(4.0, 420, "CIFAR-10 (C10)", ha="center", fontsize=11,
            color="#666", fontstyle="italic", fontweight="bold")

    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylabel("Training Throughput (images/sec)", fontsize=13,
                  fontweight="bold")
    ax.set_ylim(0, 450)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="y", linestyle="-", alpha=0.15, linewidth=0.5,
            color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.95,
              edgecolor="#999", fancybox=False)

    plt.tight_layout(pad=0.8)
    for fmt in ("png", "pdf"):
        out = f"plots/fig8_throughput_comparison.{fmt}"
        fig.savefig(out, dpi=300, facecolor="white")
        print(f"[Saved] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
