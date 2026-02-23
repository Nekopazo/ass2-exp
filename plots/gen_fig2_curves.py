#!/usr/bin/env python3
"""Generate Figure 2: Training curves — 2×3 grid (dataset × model)."""

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
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

KERAS_C   = "#D15B3B"
PYTORCH_C = "#2E6DA4"
FOLD_ALPHA = 0.18
FILL_ALPHA = 0.12

LOGS = Path("logs")
DATASETS = ["plantvillage", "cifar10"]
MODELS   = ["resnet50", "vgg16", "mobilenetv2"]
FOLDS    = [0, 1, 2]
FW_META  = {
    "keras":   {"color": KERAS_C,   "label": "Keras"},
    "pytorch": {"color": PYTORCH_C, "label": "PyTorch"},
}
NICE_DS    = {"plantvillage": "PlantVillage", "cifar10": "CIFAR-10"}
NICE_MODEL = {"resnet50": "ResNet-50", "vgg16": "VGG-16-BN",
              "mobilenetv2": "MobileNetV2"}


def _load(fw, ds, model):
    dfs = []
    for f in FOLDS:
        p = LOGS / f"{fw}_{ds}_{model}_fold{f}.csv"
        if p.exists():
            df = pd.read_csv(p).sort_values("epoch").reset_index(drop=True)
            df["fold"] = f
            dfs.append(df)
    return dfs


def _mean_std_series(dfs, col):
    """Return (epochs, mean, std) arrays aligned by epoch."""
    if not dfs:
        return None, None, None
    max_ep = max(int(df["epoch"].max()) for df in dfs)
    epochs = np.arange(1, max_ep + 1)
    mat = np.full((len(dfs), len(epochs)), np.nan)
    for i, df in enumerate(dfs):
        for _, row in df.iterrows():
            idx = int(row["epoch"]) - 1
            if 0 <= idx < len(epochs):
                mat[i, idx] = row[col]
    counts = np.sum(~np.isnan(mat), axis=0)
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat, axis=0)
    valid = counts >= 1
    return epochs[valid], mean[valid], std[valid]


def main():
    fig, axes = plt.subplots(
        2, 3, figsize=(14.4, 8.0), dpi=300,
        gridspec_kw={"hspace": 0.42, "wspace": 0.50,
                     "left": 0.055, "right": 0.945, "top": 0.94, "bottom": 0.07})

    labels_iter = iter("abcdef")

    for row_i, ds in enumerate(DATASETS):
        for col_j, model in enumerate(MODELS):
            ax = axes[row_i, col_j]
            ax2 = ax.twinx()
            lbl = next(labels_iter)

            all_loss_vals = []  # track to set y-lim

            for fw in ["keras", "pytorch"]:
                c = FW_META[fw]["color"]
                fl = FW_META[fw]["label"]
                dfs = _load(fw, ds, model)
                if not dfs:
                    continue

                # ── Per-fold thin ──
                for df in dfs:
                    ax.plot(df["epoch"], df["val_accuracy"],
                            color=c, alpha=FOLD_ALPHA, linewidth=0.7, zorder=2)
                    ax2.plot(df["epoch"], df["train_loss"],
                             color=c, alpha=FOLD_ALPHA * 0.7,
                             linewidth=0.5, linestyle=":", zorder=1)
                    all_loss_vals.extend(df["train_loss"].tolist())

                # ── Mean + std band (val accuracy) ──
                ep, mn, sd = _mean_std_series(dfs, "val_accuracy")
                if ep is not None:
                    ax.plot(ep, mn, color=c, linewidth=2.2,
                            label=f"{fl} Val Acc", zorder=5)
                    ax.fill_between(ep, mn - sd, mn + sd,
                                    color=c, alpha=FILL_ALPHA, zorder=1)

                # ── Mean (train loss — dashed) ──
                ep2, mn2, sd2 = _mean_std_series(dfs, "train_loss")
                if ep2 is not None:
                    ax2.plot(ep2, mn2, color=c, linewidth=1.5,
                             linestyle="--", alpha=0.6,
                             label=f"{fl} Loss", zorder=3)

            # ── Axis styling ──
            ax.set_title(f"({lbl})  {NICE_DS[ds]} — {NICE_MODEL[model]}",
                         fontsize=10.5, fontweight="bold", pad=7)

            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_xlim(0.5, 31)

            # Val accuracy axis
            ax.set_ylabel("Validation Accuracy", fontsize=9)
            ax.set_ylim(-0.02, 1.05)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
            ax.tick_params(axis="both", labelsize=7.5)

            # Loss axis — clip to reasonable range
            if all_loss_vals:
                p95 = np.percentile(all_loss_vals, 95)
                loss_max = min(max(all_loss_vals), p95 * 1.4, 2.5)
                ax2.set_ylim(0, max(loss_max, 0.5))
            ax2.set_ylabel("Training Loss", fontsize=9, color="#555",
                           rotation=270, labelpad=14)
            ax2.tick_params(axis="y", labelsize=7.5, colors="#555")

            # Grid
            ax.grid(True, which="major", linestyle="--",
                    alpha=0.22, linewidth=0.5)
            ax.set_axisbelow(True)

            # ── Collapse annotation for MobileNetV2 ──
            if model == "mobilenetv2":
                k_dfs = _load("keras", ds, model)
                if k_dfs:
                    max_ep_k = max(df["epoch"].max() for df in k_dfs)
                    if max_ep_k <= 12:
                        _, mn_k, _ = _mean_std_series(k_dfs, "val_accuracy")
                        last_acc = mn_k[-1] if mn_k is not None and len(mn_k) > 0 else 0.2
                        ax.annotate(
                            "Keras collapsed",
                            xy=(max_ep_k, last_acc),
                            xytext=(18, 0.55),
                            fontsize=7, color=KERAS_C, fontweight="bold",
                            ha="center",
                            arrowprops=dict(arrowstyle="-|>",
                                            color=KERAS_C, lw=1.0,
                                            connectionstyle="arc3,rad=-0.15"),
                            bbox=dict(boxstyle="round,pad=0.25",
                                      fc="#FFF3EE", ec=KERAS_C,
                                      alpha=0.92, lw=0.8),
                            zorder=7)

            # ── Legend — all subplots use lower right for consistency ──
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            if h1 or h2:
                ax.legend(h1 + h2, l1 + l2, fontsize=6,
                          loc="lower right",
                          framealpha=0.92, edgecolor="#CCC",
                          handlelength=1.5, borderpad=0.3,
                          labelspacing=0.3)

    # ── Save ──
    for fmt in ("png", "pdf"):
        out = f"plots/fig2_training_curves.{fmt}"
        fig.savefig(out, dpi=300, facecolor="white")
        print(f"[Saved] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
