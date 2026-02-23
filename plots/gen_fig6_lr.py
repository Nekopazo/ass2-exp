#!/usr/bin/env python3
"""Figure 6: Learning rate schedule — Linear Warmup + Cosine Annealing."""

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

BASE_LR      = 0.04
WARMUP_EP    = 5
TOTAL_EP     = 30
STEPS_PER_EP = 250   # approximate for illustration
ETA_MIN      = 0.0


def compute_lr_schedule():
    """Reproduce per-step LR: linear warmup then cosine decay."""
    total_steps = TOTAL_EP * STEPS_PER_EP
    warmup_steps = WARMUP_EP * STEPS_PER_EP
    cosine_steps = total_steps - warmup_steps

    lrs = []
    for step in range(total_steps):
        if step < warmup_steps:
            lr = BASE_LR * step / warmup_steps
        else:
            progress = (step - warmup_steps) / cosine_steps
            lr = ETA_MIN + 0.5 * (BASE_LR - ETA_MIN) * (1 + np.cos(np.pi * progress))
        lrs.append(lr)
    return np.array(lrs)


def main():
    lrs = compute_lr_schedule()
    total_steps = len(lrs)
    # Convert to epoch-based x-axis
    epochs = np.arange(total_steps) / STEPS_PER_EP

    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=300)
    fig.patch.set_facecolor("white")

    # Warmup shading
    ax.axvspan(0, WARMUP_EP, color="#E3F2FD", alpha=0.5, zorder=0)

    # LR curve
    ax.plot(epochs, lrs, color="#2E6DA4", linewidth=2.0, zorder=4)

    # Phase labels
    ax.text(WARMUP_EP / 2, BASE_LR * 0.55, "Linear\nWarmup",
            ha="center", va="center", fontsize=9, color="#1565C0",
            fontweight="bold", fontstyle="italic", zorder=5)
    ax.text((WARMUP_EP + TOTAL_EP) / 2, BASE_LR * 0.55, "Cosine Annealing",
            ha="center", va="center", fontsize=9, color="#555",
            fontweight="bold", fontstyle="italic", zorder=5)

    # Key point annotations
    ax.annotate(f"Peak LR = {BASE_LR}",
                xy=(WARMUP_EP, BASE_LR),
                xytext=(WARMUP_EP + 4, BASE_LR * 0.88),
                fontsize=8, color="#333",
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CCC",
                          lw=0.5, alpha=0.9),
                zorder=6)

    ax.annotate("LR ≈ 0",
                xy=(TOTAL_EP, 0),
                xytext=(TOTAL_EP - 4, BASE_LR * 0.15),
                fontsize=8, color="#333",
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CCC",
                          lw=0.5, alpha=0.9),
                zorder=6)

    # Vertical dashed line at warmup boundary
    ax.axvline(WARMUP_EP, color="#90CAF9", linestyle="--", linewidth=1.0,
               alpha=0.7, zorder=2)
    ax.text(WARMUP_EP, -0.0025, "Epoch 5", ha="center", va="top",
            fontsize=7, color="#1565C0")

    # Styling
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Learning Rate", fontsize=10)
    ax.set_xlim(0, TOTAL_EP)
    ax.set_ylim(-0.001, BASE_LR * 1.08)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.01))
    ax.tick_params(labelsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=0.5)
    for fmt in ("png", "pdf"):
        out = f"plots/fig6_lr_schedule.{fmt}"
        fig.savefig(out, dpi=300, facecolor="white")
        print(f"[Saved] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
