#!/usr/bin/env python3
"""Generate Figure 1: End-to-end benchmark pipeline diagram (publication-ready)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.unicode_minus": False,
})

# ── Palette ───────────────────────────────────────────────────
SHARED_EC    = "#2B5C7A"
SHARED_FC    = "#DAEAF3"
KERAS_EC     = "#C9513E"
KERAS_FC     = "#FAE8E3"
PYTORCH_EC   = "#2D6AA6"
PYTORCH_FC   = "#DEEAF6"
ARROW_C      = "#3A3A3A"
ZONE_SHARED  = "#7FAEC8"
ZONE_FW      = "#D4A574"
SUB_COLOR    = "#606060"


def _box(ax, cx, cy, w, h, title, sub, fc, ec,
         title_fs=11, title_color="#111", sub_fs=7.5, lw=2.0):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.18", facecolor=fc,
                       edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(p)
    if sub:
        ax.text(cx, cy + 0.18, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=title_color, zorder=4)
        ax.text(cx, cy - 0.22, sub, ha="center", va="center",
                fontsize=sub_fs, color=SUB_COLOR, zorder=4, linespacing=1.35)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=title_color, zorder=4)


def _arrow(ax, x1, y1, x2, y2, rad=0):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", color=ARROW_C, linewidth=1.8,
                        mutation_scale=16, connectionstyle=f"arc3,rad={rad}",
                        zorder=2)
    ax.add_patch(a)


def _zone(ax, x, y, w, h, label, ec):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                       facecolor="none", edgecolor=ec, linewidth=1.4,
                       linestyle=(0, (6, 3)), zorder=1, alpha=0.85)
    ax.add_patch(r)
    ax.text(x + w/2, y + h + 0.12, label,
            ha="center", va="bottom", fontsize=8, fontstyle="italic",
            fontweight="bold", color=ec, zorder=5)


def main():
    fig, ax = plt.subplots(figsize=(14, 5.2), dpi=300)
    ax.set_xlim(-0.3, 14.3)
    ax.set_ylim(-2.2, 2.8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    bw, bh = 1.85, 0.95          # box width / height
    xs = [1.1, 3.55, 6.35, 9.15, 11.2, 13.2]  # 6 stage x-centres
    y0 = 1.0                      # shared row y
    yk = 1.0                      # keras y
    yp = -0.45                    # pytorch y

    # ════════════════ ZONES ════════════════════════════════════
    pad = 0.28
    # Left shared
    _zone(ax,
          xs[0] - bw/2 - pad, y0 - bh/2 - pad,
          (xs[1] - xs[0]) + bw + 2*pad, bh + 2*pad,
          "Shared Components", ZONE_SHARED)
    # Framework-specific
    _zone(ax,
          xs[2] - bw/2 - pad - 0.1, yp - bh/2 - pad - 0.05,
          bw + 2*pad + 0.2, (yk + bh/2 + pad) - (yp - bh/2 - pad) + 0.15,
          "Framework-Specific", ZONE_FW)
    # Right shared
    _zone(ax,
          xs[3] - bw/2 - pad, y0 - bh/2 - pad,
          (xs[5] - xs[3]) + bw + 2*pad, bh + 2*pad,
          "Shared Components", ZONE_SHARED)

    # ════════════════ BOXES ════════════════════════════════════
    _box(ax, xs[0], y0, bw, bh,
         "Data Preparation", "PlantVillage (38 cls)\nCIFAR-10 (10 cls)",
         SHARED_FC, SHARED_EC)

    _box(ax, xs[1], y0, bw, bh,
         "3-Fold CV Split", "Stratified\n80 / 20 test hold-out",
         SHARED_FC, SHARED_EC)

    _box(ax, xs[2], yk, bw, bh,
         "Keras Training", "tf.GradientTape\nfloat32",
         KERAS_FC, KERAS_EC, title_color=KERAS_EC)

    _box(ax, xs[2], yp, bw, bh,
         "PyTorch Training", "Standard loop\nAMP float16",
         PYTORCH_FC, PYTORCH_EC, title_color=PYTORCH_EC)

    _box(ax, xs[3], y0, bw, bh,
         "Evaluation", "Test Acc · Macro-F1\nConfusion Matrix",
         SHARED_FC, SHARED_EC)

    _box(ax, xs[4], y0, bw, bh,
         "Aggregation", "Mean ± Std\nacross 3 folds",
         SHARED_FC, SHARED_EC)

    _box(ax, xs[5], y0, bw, bh,
         "Statistical Testing", "Paired t-test\nCohen's d · Cliff's δ",
         SHARED_FC, SHARED_EC, title_fs=10)

    # ════════════════ ARROWS ══════════════════════════════════
    g = 0.06  # gap from box edge
    _arrow(ax, xs[0]+bw/2+g, y0,       xs[1]-bw/2-g, y0)
    _arrow(ax, xs[1]+bw/2+g, y0,       xs[2]-bw/2-g, yk)          # → Keras
    _arrow(ax, xs[1]+bw/2+g, y0-0.18,  xs[2]-bw/2-g, yp+0.18,
           rad=0.4)                                                 # → PyTorch
    _arrow(ax, xs[2]+bw/2+g, yk,       xs[3]-bw/2-g, y0)          # Keras →
    _arrow(ax, xs[2]+bw/2+g, yp+0.18,  xs[3]-bw/2-g, y0-0.18,
           rad=-0.4)                                                # PyTorch →
    _arrow(ax, xs[3]+bw/2+g, y0,       xs[4]-bw/2-g, y0)
    _arrow(ax, xs[4]+bw/2+g, y0,       xs[5]-bw/2-g, y0)

    # ════════════════ CONFIG BAR ══════════════════════════════
    cfg_y  = -1.65
    cfg_x0 = xs[0] - bw/2 - 0.15
    cfg_w  = xs[5] + bw/2 + 0.15 - cfg_x0
    cfg_h  = 0.48
    cfg = FancyBboxPatch((cfg_x0, cfg_y - cfg_h/2), cfg_w, cfg_h,
                         boxstyle="round,pad=0.10", facecolor="#F4F4F4",
                         edgecolor="#BBBBBB", linewidth=1.0, zorder=1)
    ax.add_patch(cfg)

    parts = [
        ("Shared Hyperparameters:", True),
        ("  SGD (momentum = 0.9, Nesterov)    ·    ", False),
        ("LR = 0.04 + Cosine + 5-ep Warmup    ·    ", False),
        ("Batch 128    ·    Early Stopping (patience = 7)", False),
    ]
    # Build as single string; bold prefix via a separate text call
    ax.text(cfg_x0 + 0.25, cfg_y, "Shared Hyperparameters:",
            ha="left", va="center", fontsize=7.5, fontweight="bold",
            color="#333", zorder=4)
    rest = ("SGD (momentum = 0.9, Nesterov)    ·    "
            "LR = 0.04 + Cosine Decay + 5-epoch Warmup    ·    "
            "Batch Size = 128    ·    "
            "Early Stopping (patience = 7)")
    ax.text(cfg_x0 + 3.65, cfg_y, rest,
            ha="left", va="center", fontsize=7.5, color="#555", zorder=4)

    # ── Save ──
    for fmt in ("png", "pdf"):
        out = f"plots/fig1_pipeline.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight",
                    facecolor="white", pad_inches=0.15)
        print(f"[Saved] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
