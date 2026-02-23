#!/usr/bin/env python3
"""Figure 7: Test accuracy comparison bar chart."""

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
    ("ResNet-50\n(PV)",      99.68, 0.06, 99.58, 0.17),
    ("VGG-16\n(PV)",         99.42, 0.14, 99.50, 0.03),
    ("MobileNetV2\n(PV)",    39.76, 17.37, 99.74, 0.06),
    ("ResNet-50\n(C10)",     95.85, 0.15, 93.55, 2.70),
    ("VGG-16\n(C10)",        93.99, 0.19, 93.65, 0.07),
    ("MobileNetV2\n(C10)",   56.64, 14.97, 95.39, 0.12),
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

    # 90% dashed reference line
    ax.axhline(90, color="#888", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)
    ax.text(n - 0.3, 91.5, "90%", fontsize=9, color="#888", va="bottom",
            fontweight="bold")

    # Vertical separator between datasets
    ax.axvline(2.5, color="#CCCCCC", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(1.0, 3, "PlantVillage (PV)", ha="center", fontsize=11,
            color="#666", fontstyle="italic", fontweight="bold")
    ax.text(4.0, 3, "CIFAR-10 (C10)", ha="center", fontsize=11,
            color="#666", fontstyle="italic", fontweight="bold")

    # Annotation arrows for catastrophic failure (Keras MobileNetV2 bars)
    # PlantVillage MobileNetV2 Keras (index 2)
    ax.annotate("Catastrophic\nfailure",
                xy=(2 - w/2, k_mean[2] + k_std[2] + 2),
                xytext=(2 - w/2 - 0.6, 70),
                fontsize=9, color="#CC3333", fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#CC3333", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CC3333",
                          lw=0.8, alpha=0.95),
                zorder=6)

    # CIFAR-10 MobileNetV2 Keras (index 5)
    ax.annotate("Catastrophic\nfailure",
                xy=(5 - w/2, k_mean[5] + k_std[5] + 2),
                xytext=(5 - w/2 - 0.6, 80),
                fontsize=9, color="#CC3333", fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#CC3333", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CC3333",
                          lw=0.8, alpha=0.95),
                zorder=6)

    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="y", linestyle="-", alpha=0.15, linewidth=0.5,
            color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.95,
              edgecolor="#999", fancybox=False)

    plt.tight_layout(pad=0.8)
    for fmt in ("png", "pdf"):
        out = f"plots/fig7_accuracy_comparison.{fmt}"
        fig.savefig(out, dpi=300, facecolor="white")
        print(f"[Saved] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
