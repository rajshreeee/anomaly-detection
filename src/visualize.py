# src/visualize.py

import logging
import random
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve

from src.dataset import TestDataset, denormalize

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _get_image(test_ds: TestDataset, idx: int) -> np.ndarray:
    """Return denormalized [H, W, 3] numpy image for display."""
    img_t, _, _ = test_ds[idx]
    return denormalize(img_t).permute(1, 2, 0).numpy()


# ─────────────────────────────────────────────
#  Per-class anomaly map comparison
# ─────────────────────────────────────────────

def plot_anomaly_maps(
    test_ds:      TestDataset,
    y_true:       List[int],
    classes_list: List[str],
    masks_gt_np:  List[np.ndarray],
    model_outputs: dict,             # {'PatchCore': {'scores': [...], 'maps': [...]}, ...}
    n_samples:    int   = 3,
    alpha:        float = 0.5,
) -> None:
    """
    For each defect class, plot n_samples random defective images.

    Columns:
      1. Original image (with per-model anomaly scores on y-axis)
      2. Ground truth mask (green overlay)
      3+. One column per model — heatmap overlay + colorbar

    Args:
        model_outputs: dict keyed by model name, each containing
                       'scores' (List[float]) and 'maps' (List[np.ndarray])
    """
    model_names = list(model_outputs.keys())
    n_cols      = 2 + len(model_names)
    col_titles  = ['Original', 'Ground Truth'] + model_names

    # Build class → defective image indices
    class_to_indices = {}
    for i, cls in enumerate(classes_list):
        if cls != 'good':
            class_to_indices.setdefault(cls, []).append(i)

    for cls in sorted(class_to_indices.keys()):
        indices = class_to_indices[cls]
        chosen  = random.sample(indices, min(n_samples, len(indices)))

        fig, axes = plt.subplots(
            len(chosen), n_cols,
            figsize=(3.2 * n_cols, 3.2 * len(chosen)),
            squeeze=False
        )
        fig.suptitle(f'Defect class: {cls}', fontsize=13,
                     fontweight='bold', y=1.01)

        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=10)

        for row, idx in enumerate(chosen):
            orig = _get_image(test_ds, idx)
            col  = 0

            # ── Original + score labels ──
            score_label = '\n'.join(
                f"{name}: {model_outputs[name]['scores'][idx]:.3f}"
                for name in model_names
            )
            axes[row, col].imshow(orig)
            axes[row, col].set_ylabel(score_label, fontsize=7, labelpad=4)
            col += 1

            # ── Ground truth mask ──
            axes[row, col].imshow(orig)
            axes[row, col].imshow(
                masks_gt_np[idx], cmap='Greens',
                alpha=0.6, vmin=0, vmax=1
            )
            col += 1

            # ── One column per model ──
            for name in model_names:
                amap = model_outputs[name]['maps'][idx]
                axes[row, col].imshow(orig)
                im = axes[row, col].imshow(
                    amap, cmap='hot', alpha=alpha,
                    vmin=amap.min(), vmax=amap.max()
                )
                plt.colorbar(im, ax=axes[row, col],
                             fraction=0.046, pad=0.04)
                col += 1

        for ax in axes.flatten():
            ax.axis('off')

        plt.tight_layout()
        plt.show()
        plt.close(fig)


# ─────────────────────────────────────────────
#  Score distributions
# ─────────────────────────────────────────────

def plot_score_distributions(
    y_true:        List[int],
    model_outputs: dict,
    results:       dict,             # {'PatchCore': {'Image AUROC': ...}, ...}
    bins:          int  = 30,
) -> None:
    """
    Histogram of anomaly scores split by good vs defective, one subplot per model.
    Shows how well each model separates the two distributions.
    """
    model_names = list(model_outputs.keys())
    fig, axes   = plt.subplots(1, len(model_names),
                                figsize=(6 * len(model_names), 4),
                                squeeze=False)

    good_mask = np.array(y_true) == 0
    anom_mask = ~good_mask

    for col, name in enumerate(model_names):
        scores    = np.array(model_outputs[name]['scores'])
        auc_label = results.get(name, {}).get('Image AUROC', float('nan'))

        axes[0, col].hist(scores[good_mask], bins=bins, alpha=0.6,
                          label='Good',    color='steelblue')
        axes[0, col].hist(scores[anom_mask], bins=bins, alpha=0.6,
                          label='Anomaly', color='tomato')
        axes[0, col].set_title(
            f'{name} score distribution\n'
            f'Image AUROC = {auc_label*100:.1f}%'
        )
        axes[0, col].set_xlabel('Anomaly score')
        axes[0, col].set_ylabel('Count')
        axes[0, col].legend()

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────
#  ROC curves
# ─────────────────────────────────────────────

def plot_roc_curves(
    y_true:        List[int],
    model_outputs: dict,
    results:       dict,
) -> None:
    """
    Overlaid ROC curves for all models on the same axes.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    for name in model_outputs:
        scores    = model_outputs[name]['scores']
        auc_label = results.get(name, {}).get('Image AUROC', float('nan'))
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, label=f'{name}  {auc_label*100:.1f}%')

    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — Carpet Anomaly Detection')
    ax.legend()

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────
#  Per-class bar chart
# ─────────────────────────────────────────────

def plot_per_class_auroc(
    model_per_class: dict,           # {'PatchCore': {'cut': 0.9, ...}, 'EfficientAD': {...}}
) -> None:
    """
    Grouped bar chart of per-class image AUROC for all models.
    Makes it easy to spot which defect types each model struggles with.
    """
    model_names  = list(model_per_class.keys())
    all_classes  = sorted({
        cls for m in model_per_class.values() for cls in m.keys()
    })
    x      = np.arange(len(all_classes))
    width  = 0.8 / len(model_names)
    colors = plt.cm.Set2.colors

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(all_classes)), 5))

    for i, name in enumerate(model_names):
        vals   = [model_per_class[name].get(cls, float('nan')) for cls in all_classes]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, [v * 100 for v in vals],
                        width=width * 0.9, label=name, color=colors[i])
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f'{val*100:.0f}', ha='center', va='bottom',
                        fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=20, ha='right')
    ax.set_ylabel('Image AUROC (%)')
    ax.set_ylim(0, 110)
    ax.set_title('Per-class Image AUROC')
    ax.legend()
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.7)

    plt.tight_layout()
    plt.show()
    plt.close(fig)
