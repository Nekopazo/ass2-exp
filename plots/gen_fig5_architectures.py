#!/usr/bin/env python3
"""Figure 5: Three model architecture comparison diagrams.
(a) ResNet-50  (b) VGG-16-BN  (c) MobileNetV2
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
})

# ── Colors ──
C_CONV  = "#D6E8F0"
C_BN    = "#E8F5E9"
C_FC    = "#FFF3E0"
C_POOL  = "#F3E5F5"
C_SKIP  = "#2E6DA4"
C_EDGE  = "#555555"
C_INPUT = "#E0E0E0"


def _box(ax, cx, cy, w, h, label, fc, ec=C_EDGE, fs=7, fw="normal", lw=1.0):
    b = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(b)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs,
            fontweight=fw, color="#222", zorder=4)
    return cy - h/2, cy + h/2  # bottom, top


def _arrow_down(ax, x, y1, y2, color=C_EDGE, lw=1.2):
    a = FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>",
                        color=color, lw=lw, mutation_scale=10, zorder=2)
    ax.add_patch(a)


def _skip_arrow(ax, x_from, y_from, x_to, y_to, color=C_SKIP):
    a = FancyArrowPatch((x_from, y_from), (x_to, y_to),
                        arrowstyle="-|>", color=color, lw=1.2,
                        mutation_scale=10, connectionstyle="arc3,rad=-0.5",
                        linestyle="--", zorder=2)
    ax.add_patch(a)


def draw_resnet(ax):
    """(a) ResNet-50"""
    ax.set_xlim(-2, 2); ax.set_ylim(-0.5, 11)
    ax.axis("off")
    ax.set_title("(a)  ResNet-50\n23.6M params", fontsize=10,
                 fontweight="bold", pad=8)

    bw, bh = 2.2, 0.55
    x = 0
    elements = [
        ("Input\n224×224×3", C_INPUT),
        ("Conv 7×7, stride 2\n+ BN + ReLU", C_CONV),
        ("MaxPool 3×3", C_POOL),
        ("Res Block ×3\n(64 filters)", C_CONV),
        ("Res Block ×4\n(128 filters)", C_CONV),
        ("Res Block ×6\n(256 filters)", C_CONV),
        ("Res Block ×3\n(512 filters)", C_CONV),
        ("Global Avg Pool", C_POOL),
        ("FC → num_classes", C_FC),
    ]

    y_positions = []
    for i, (label, color) in enumerate(elements):
        y = 10 - i * 1.15
        _box(ax, x, y, bw, bh, label, color, fs=6.5)
        y_positions.append(y)
        if i > 0:
            _arrow_down(ax, x, y_positions[i-1] - bh/2 - 0.02,
                        y + bh/2 + 0.02)

    # Skip connections on residual blocks
    for i in [3, 4, 5, 6]:
        y_top = y_positions[i] + bh/2
        y_bot = y_positions[i] - bh/2
        _skip_arrow(ax, -bw/2 - 0.1, y_top, -bw/2 - 0.1, y_bot, C_SKIP)

    ax.text(1.5, y_positions[4], "skip\nconnection", fontsize=5.5,
            color=C_SKIP, ha="center", fontstyle="italic")


def draw_vgg(ax):
    """(b) VGG-16-BN"""
    ax.set_xlim(-2, 2); ax.set_ylim(-0.5, 11)
    ax.axis("off")
    ax.set_title("(b)  VGG-16-BN\n134.4M params", fontsize=10,
                 fontweight="bold", pad=8)

    bw, bh = 2.2, 0.55
    x = 0
    elements = [
        ("Input\n224×224×3", C_INPUT),
        ("Conv3-64 ×2\n+ BN + ReLU → Pool", C_CONV),
        ("Conv3-128 ×2\n+ BN + ReLU → Pool", C_CONV),
        ("Conv3-256 ×3\n+ BN + ReLU → Pool", C_CONV),
        ("Conv3-512 ×3\n+ BN + ReLU → Pool", C_CONV),
        ("Conv3-512 ×3\n+ BN + ReLU → Pool", C_CONV),
        ("FC-4096 + ReLU", C_FC),
        ("FC-4096 + ReLU", C_FC),
        ("FC → num_classes", C_FC),
    ]

    y_positions = []
    for i, (label, color) in enumerate(elements):
        y = 10 - i * 1.15
        _box(ax, x, y, bw, bh, label, color, fs=6.5)
        y_positions.append(y)
        if i > 0:
            _arrow_down(ax, x, y_positions[i-1] - bh/2 - 0.02,
                        y + bh/2 + 0.02)

    # Bracket for "13 Conv layers"
    ax.annotate("", xy=(-bw/2 - 0.25, y_positions[1] + bh/2),
                xytext=(-bw/2 - 0.25, y_positions[5] - bh/2),
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.8))
    ax.text(-bw/2 - 0.35, (y_positions[1] + y_positions[5]) / 2,
            "13 Conv\nlayers", fontsize=5.5, color="#666", ha="right",
            va="center", fontstyle="italic")


def draw_mobilenet(ax):
    """(c) MobileNetV2"""
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-0.5, 11)
    ax.axis("off")
    ax.set_title("(c)  MobileNetV2\n2.3M params", fontsize=10,
                 fontweight="bold", pad=8)

    bw, bh = 2.4, 0.55
    x = 0
    elements = [
        ("Input\n224×224×3", C_INPUT),
        ("Conv 3×3, stride 2\n+ BN + ReLU6", C_CONV),
        ("Inverted Residual ×1\n(t=1, c=16, s=1)", C_CONV),
        ("Inverted Residual ×2\n(t=6, c=24, s=2)", C_CONV),
        ("Inverted Residual ×3\n(t=6, c=32, s=2)", C_CONV),
        ("Inverted Residual ×7\n(t=6, c=64/96, s=1-2)", C_CONV),
        ("Inverted Residual ×3\n(t=6, c=160/320, s=2-1)", C_CONV),
        ("Conv 1×1 + GAP", C_POOL),
        ("FC → num_classes", C_FC),
    ]

    y_positions = []
    for i, (label, color) in enumerate(elements):
        y = 10 - i * 1.15
        _box(ax, x, y, bw, bh, label, color, fs=6.5)
        y_positions.append(y)
        if i > 0:
            _arrow_down(ax, x, y_positions[i-1] - bh/2 - 0.02,
                        y + bh/2 + 0.02)

    # Skip connections on inverted residuals (where stride=1)
    for i in [2, 4, 5]:
        y_top = y_positions[i] + bh/2
        y_bot = y_positions[i] - bh/2
        _skip_arrow(ax, -bw/2 - 0.15, y_top, -bw/2 - 0.15, y_bot, C_SKIP)

    # Expanded detail box for inverted residual
    detail_x = 1.85
    detail_y = y_positions[3]
    detail_items = ["1×1 Expand", "3×3 Depthwise", "1×1 Project"]
    detail_colors = ["#FFF9C4", "#E1F5FE", "#FFF9C4"]
    dh = 0.32
    for k, (dlabel, dc) in enumerate(zip(detail_items, detail_colors)):
        dy = detail_y + (1 - k) * (dh + 0.06)
        b = FancyBboxPatch((detail_x - 0.6, dy - dh/2), 1.2, dh,
                           boxstyle="round,pad=0.04", fc=dc, ec="#AAA",
                           lw=0.6, zorder=3)
        ax.add_patch(b)
        ax.text(detail_x, dy, dlabel, ha="center", va="center",
                fontsize=5.5, color="#333", zorder=4)
        if k > 0:
            _arrow_down(ax, detail_x,
                        detail_y + (1 - (k-1)) * (dh + 0.06) - dh/2 - 0.01,
                        dy + dh/2 + 0.01, color="#AAA", lw=0.7)

    # Label
    ax.text(detail_x, detail_y + 1.5 * (dh + 0.06) + 0.15,
            "Inverted\nResidual Block", fontsize=5.5, ha="center",
            color="#666", fontstyle="italic")


def main():
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(14, 9), dpi=300,
        gridspec_kw={"wspace": 0.15, "left": 0.02, "right": 0.98,
                     "top": 0.92, "bottom": 0.01})
    fig.patch.set_facecolor("white")

    draw_resnet(ax1)
    draw_vgg(ax2)
    draw_mobilenet(ax3)

    for fmt in ("png", "pdf"):
        out = f"plots/fig5_architectures.{fmt}"
        fig.savefig(out, dpi=300, facecolor="white")
        print(f"[Saved] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
