#!/usr/bin/env python3
"""Figure 9: BatchNorm momentum divergence — PyTorch vs Keras.
Two-panel line plot showing how different BN momentum values cause
running-statistics lag during warmup and beyond.
"""

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


def simulate_bn(momentum, n_steps=200, seed=42):
    """Simulate batch statistics and running mean with given momentum.
    PyTorch convention: running = (1-momentum)*running + momentum*batch
    """
    rng = np.random.RandomState(seed)

    # Underlying mean: stable initially, then shifts during warmup (LR ramp),
    # then stabilises with mild oscillation
    true_mean = np.zeros(n_steps)
    warmup_end = 50  # ~step 50 represents end of warmup (maps to ~epoch 5)
    for i in range(n_steps):
        if i < 10:
            true_mean[i] = 0.0
        elif i < warmup_end:
            # LR ramping causes feature drift
            true_mean[i] = 2.5 * (i - 10) / (warmup_end - 10)
        else:
            true_mean[i] = 2.5 + 0.4 * np.sin(2 * np.pi * (i - warmup_end) / 100)

    # Batch statistics (noisy observations)
    batch_stats = true_mean + rng.randn(n_steps) * 0.5

    # Running mean with exponential moving average
    running = np.zeros(n_steps)
    running[0] = batch_stats[0]
    for i in range(1, n_steps):
        running[i] = (1 - momentum) * running[i - 1] + momentum * batch_stats[i]

    return batch_stats, running, true_mean


def draw_panel(ax, momentum, panel_title, color_ema, shading_color):
    batch, running, true = simulate_bn(momentum)
    steps = np.arange(len(batch))
    warmup_end = 50

    # Batch statistics (noisy gray)
    ax.plot(steps, batch, color="#AAAAAA", linewidth=0.7, alpha=0.55,
            label="Batch statistics", zorder=2)

    # Running mean (EMA) — smooth colored line
    ax.plot(steps, running, color=color_ema, linewidth=2.5,
            label="Running mean (EMA)", zorder=4)

    # Shaded divergence area between running and batch/true
    ax.fill_between(steps, running, true,
                    color=shading_color, alpha=0.20, zorder=1,
                    label="Train\u2013eval gap" if momentum < 0.05 else "Train\u2013eval gap")

    # Vertical dashed line at warmup end
    ax.axvline(warmup_end, color="#666666", linestyle="--", linewidth=1.0,
               alpha=0.7, zorder=3)
    ax.annotate("warmup ends\nLR 0.008\u21920.04",
                xy=(warmup_end, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 3.0),
                xytext=(warmup_end + 15, 3.8),
                fontsize=8, color="#666666",
                arrowprops=dict(arrowstyle="-|>", color="#999", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CCC",
                          lw=0.5, alpha=0.9),
                ha="left", zorder=6)

    # Panel title
    ax.set_title(panel_title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Training Step", fontsize=12, fontweight="bold")
    ax.set_ylabel("Statistic Value", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 200)
    ax.set_ylim(-1.5, 5.0)
    ax.tick_params(labelsize=11)
    ax.grid(True, linestyle="-", alpha=0.15, linewidth=0.5, color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95,
              edgecolor="#CCC", fancybox=False)


def main():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8, 5), dpi=300,
        gridspec_kw={"wspace": 0.32, "left": 0.09, "right": 0.97,
                     "top": 0.88, "bottom": 0.14})
    fig.patch.set_facecolor("white")

    # Left: PyTorch momentum=0.1 (fast tracking, small gap)
    draw_panel(ax1, momentum=0.1,
               panel_title="PyTorch (momentum = 0.1)",
               color_ema="#DD8452", shading_color="#DD8452")

    # Right: Keras momentum=0.99 -> update weight = 0.01 (slow, large gap)
    draw_panel(ax2, momentum=0.01,
               panel_title="Keras (momentum = 0.99)",
               color_ema="#4C72B0", shading_color="#CC3333")

    for fmt in ("png", "pdf"):
        out = f"plots/fig9_bn_momentum.{fmt}"
        fig.savefig(out, dpi=300, facecolor="white")
        print(f"[Saved] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
