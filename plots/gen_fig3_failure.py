#!/usr/bin/env python3
"""Figure 3: Keras MobileNetV2 failure analysis — val_loss explosion
vs. training accuracy, annotated with LR at each epoch.

Two panels: (a) CIFAR-10  (b) PlantVillage,  fold 0 each.
PyTorch MobileNetV2 (same fold) shown as reference.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.unicode_minus": False,
    "axes.linewidth": 0.9,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

KERAS_C   = "#D15B3B"
PYTORCH_C = "#2E6DA4"
LOSS_C    = "#8B2500"      # dark red for val_loss explosion
LOGS      = Path("logs")


def _load(fw, ds, model, fold):
    p = LOGS / f"{fw}_{ds}_{model}_fold{fold}.csv"
    if p.exists():
        return pd.read_csv(p).sort_values("epoch").reset_index(drop=True)
    return None


def draw_panel(ax, ds, fold, label, nice_ds):
    """Draw one dataset panel."""
    kdf = _load("keras", ds, "mobilenetv2", fold)
    pdf = _load("pytorch", ds, "mobilenetv2", fold)

    if kdf is None:
        return

    ep_k = kdf["epoch"].values
    ax2 = ax.twinx()

    # ── Keras Training Accuracy (solid, rising) ──
    ax.plot(ep_k, kdf["train_accuracy"], color=KERAS_C, linewidth=2.2,
            marker="o", markersize=5, markerfacecolor="white",
            markeredgewidth=1.4, markeredgecolor=KERAS_C,
            label="Keras Train Acc", zorder=5)

    # ── Keras Validation Accuracy (solid, collapsing) ──
    ax.plot(ep_k, kdf["val_accuracy"], color=KERAS_C, linewidth=2.2,
            marker="s", markersize=5, markerfacecolor=KERAS_C,
            markeredgewidth=0, linestyle="-",
            label="Keras Val Acc", zorder=5, alpha=0.85)

    # ── Keras Validation Loss (right axis, exploding) ──
    ax2.plot(ep_k, kdf["val_loss"], color=LOSS_C, linewidth=2.4,
             marker="D", markersize=5, markerfacecolor="white",
             markeredgewidth=1.4, markeredgecolor=LOSS_C,
             label="Keras Val Loss", zorder=4)

    # ── Red shading for explosion zone ──
    ax2.fill_between(ep_k, 0, kdf["val_loss"],
                     color=LOSS_C, alpha=0.05, zorder=1)

    # ── PyTorch reference (val acc, grey-blue, dashed) ──
    if pdf is not None:
        # Only show first 8 epochs for alignment, or all
        ep_show = min(len(pdf), 30)
        pdf_s = pdf.iloc[:ep_show]
        ax.plot(pdf_s["epoch"], pdf_s["val_accuracy"],
                color=PYTORCH_C, linewidth=1.8, linestyle="--",
                marker="^", markersize=4, markerfacecolor=PYTORCH_C,
                markeredgewidth=0, alpha=0.7,
                label="PyTorch Val Acc (ref)", zorder=3)

    # ── LR annotations — only at key epochs to avoid clutter ──
    vl_max = kdf["val_loss"].max()
    key_epochs = {1, 3, 5, max(ep_k)}  # warmup start, mid, peak, last
    for _, row in kdf.iterrows():
        ep = int(row["epoch"])
        if ep not in key_epochs:
            continue
        lr = row["learning_rate"]
        vl = row["val_loss"]
        lr_text = f"LR={lr:.3f}"
        # stagger y offset to avoid overlap
        y_off = vl_max * 0.12
        x_off = 0.25
        if ep == 1:
            x_off, y_off = 0.3, vl_max * 0.15
        elif ep == max(ep_k):
            x_off, y_off = -0.5, vl_max * 0.10
        ax2.annotate(
            lr_text, xy=(ep, vl),
            xytext=(ep + x_off, vl + y_off),
            fontsize=6.5, color="#444",
            fontweight="bold" if ep == 5 else "normal",
            ha="left", va="bottom",
            arrowprops=dict(arrowstyle="-|>", color="#999", lw=0.7,
                            shrinkA=2, shrinkB=2),
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec="#CCC", alpha=0.85, lw=0.5),
            zorder=6)

    # ── Warmup phase bracket ──
    ax.axvspan(0.5, 5.5, color="#FFF3E0", alpha=0.35, zorder=0)
    ax.text(3, 0.02, "Warmup Phase", ha="center", va="bottom",
            fontsize=7, color="#B8860B", fontstyle="italic", zorder=6)

    # ── Styling ──
    ax.set_title(f"({label})  {nice_ds} — MobileNetV2 (Fold {fold})",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_xlim(0.4, max(ep_k) + 0.6)
    ax.set_ylim(-0.05, 1.08)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.tick_params(axis="both", labelsize=8.5)
    ax.grid(True, which="major", linestyle="--", alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)

    # Right axis
    ax2.set_ylabel("Validation Loss", fontsize=10, color=LOSS_C,
                   rotation=270, labelpad=16)
    ax2.tick_params(axis="y", labelsize=8.5, colors=LOSS_C)
    ax2.set_ylim(0, vl_max * 1.45)

    # ── Combined legend ──
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center left",
              framealpha=0.92, edgecolor="#CCC",
              handlelength=2.0, borderpad=0.5, labelspacing=0.4)


def main():
    fig, (ax_c, ax_p) = plt.subplots(
        1, 2, figsize=(14.4, 5.5), dpi=300,
        gridspec_kw={"wspace": 0.40, "left": 0.065, "right": 0.935,
                     "top": 0.88, "bottom": 0.11})

    draw_panel(ax_c, "cifar10", 0, "a", "CIFAR-10")
    draw_panel(ax_p, "plantvillage", 0, "b", "PlantVillage")

    # ── Save ──
    for fmt in ("png", "pdf"):
        out = f"plots/fig3_mobilenetv2_failure.{fmt}"
        fig.savefig(out, dpi=300, facecolor="white")
        print(f"[Saved] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
