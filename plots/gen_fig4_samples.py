#!/usr/bin/env python3
"""Figure 4: Dataset sample visualization.
Top row: PlantVillage 6 representative classes (224×224).
Bottom row: CIFAR-10 6 classes, each showing 32×32 original + 224×224 upsampled.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from PIL import Image

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
})

DATA = Path("data")

# PlantVillage classes to show (indices into class_names)
PV_SHOW = {
    38: "Tomato — Healthy",        # Tomato___healthy
    29: "Tomato — Early Blight",   # Tomato___Early_blight
    21: "Potato — Late Blight",    # Potato___Late_blight
    0:  "Apple — Scab",            # Apple___Apple_scab
    11: "Grape — Black Rot",       # Grape___Black_rot
    8:  "Corn — Common Rust",      # Corn_(maize)___Common_rust_
}
PV_INDICES = list(PV_SHOW.keys())
PV_LABELS  = list(PV_SHOW.values())

# CIFAR-10 classes to show
C10_SHOW = {
    0: "Airplane", 1: "Automobile", 2: "Bird",
    3: "Cat", 4: "Deer", 5: "Dog",
}
C10_INDICES = list(C10_SHOW.keys())
C10_LABELS  = list(C10_SHOW.values())


def _find_sample(labels, target_class):
    """Find first image index with the given class label."""
    for i, l in enumerate(labels):
        if int(l) == target_class:
            return i
    return 0


def main():
    # Load data
    pv_images = np.load(DATA / "plantvillage_images.npy", mmap_mode="r")
    pv_labels = np.load(DATA / "plantvillage_labels.npy")
    c10_images = np.load(DATA / "cifar10_images.npy", mmap_mode="r")
    c10_labels = np.load(DATA / "cifar10_labels.npy")

    n_pv = len(PV_INDICES)
    n_c10 = len(C10_INDICES)

    # Layout: 2 rows
    # Row 1: 6 PlantVillage images
    # Row 2: 6 CIFAR-10 pairs (original 32×32 + upsampled 224×224) = 12 sub-images
    #         but we'll show them as 6 columns, each with a small inset

    fig = plt.figure(figsize=(14, 6.5), dpi=300)
    fig.patch.set_facecolor("white")

    # ── Row 1: PlantVillage ──
    gs_top = fig.add_gridspec(1, n_pv, left=0.03, right=0.97,
                              top=0.95, bottom=0.52, wspace=0.08)

    fig.text(0.01, 0.97, "(a)  PlantVillage (224 × 224, 38 classes)",
             fontsize=11, fontweight="bold", va="top")

    for j, (cls_idx, label) in enumerate(zip(PV_INDICES, PV_LABELS)):
        ax = fig.add_subplot(gs_top[0, j])
        idx = _find_sample(pv_labels, cls_idx)
        img = pv_images[idx]
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#AAAAAA")
            spine.set_linewidth(0.8)
        # wrap long labels
        wrapped = label.replace(" — ", "\n")
        ax.set_xlabel(wrapped, fontsize=7.5, labelpad=3, color="#333")

    # ── Row 2: CIFAR-10 ──
    # Each column: left half = 32×32 original, right half = 224×224 upsampled
    gs_bot = fig.add_gridspec(1, n_c10 * 2, left=0.03, right=0.97,
                              top=0.46, bottom=0.03, wspace=0.06,
                              width_ratios=[1, 3] * n_c10)

    fig.text(0.01, 0.48, "(b)  CIFAR-10 (32 × 32 → 224 × 224 bilinear upsample, 10 classes)",
             fontsize=11, fontweight="bold", va="top")

    for j, (cls_idx, label) in enumerate(zip(C10_INDICES, C10_LABELS)):
        idx = _find_sample(c10_labels, cls_idx)
        img_small = c10_images[idx]  # 32×32

        # Upsample to 224×224 using bilinear (PIL)
        pil_img = Image.fromarray(img_small)
        img_big = np.array(pil_img.resize((224, 224), Image.BILINEAR))

        # Left: original 32×32
        ax_s = fig.add_subplot(gs_bot[0, j * 2])
        ax_s.imshow(img_small, interpolation="nearest")
        ax_s.set_xticks([]); ax_s.set_yticks([])
        for spine in ax_s.spines.values():
            spine.set_edgecolor("#AAAAAA"); spine.set_linewidth(0.8)
        if j == 0:
            ax_s.set_ylabel("32×32", fontsize=7, color="#666")

        # Right: upsampled 224×224
        ax_b = fig.add_subplot(gs_bot[0, j * 2 + 1])
        ax_b.imshow(img_big)
        ax_b.set_xticks([]); ax_b.set_yticks([])
        for spine in ax_b.spines.values():
            spine.set_edgecolor("#AAAAAA"); spine.set_linewidth(0.8)

        # Label below the pair
        mid_x = (ax_s.get_position().x0 + ax_b.get_position().x1) / 2
        fig.text(mid_x, 0.025, label, ha="center", va="top",
                 fontsize=8, color="#333")

        # Arrow between small and big
        ax_s_pos = ax_s.get_position()
        ax_b_pos = ax_b.get_position()
        fig.text((ax_s_pos.x1 + ax_b_pos.x0) / 2,
                 (ax_s_pos.y0 + ax_s_pos.y1) / 2,
                 "→", ha="center", va="center", fontsize=10, color="#999")

    # ── Save ──
    for fmt in ("png", "pdf"):
        out = f"plots/fig4_dataset_samples.{fmt}"
        fig.savefig(out, dpi=300, facecolor="white")
        print(f"[Saved] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
