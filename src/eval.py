import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import config
from efficient_ad.inference import run_inference as run_ead
from patchcore.inference   import run_inference as run_pc

logger     = logging.getLogger(__name__)
METRIC_KEYS = ["Image AUROC", "Full Pixel AUROC", "Anomaly Pixel AUROC"]


def _save_summary_table(ead: dict, pc: dict, out_dir: Path) -> pd.DataFrame:
    rows = [
        {
            "Metric":      key,
            "EfficientAD": round(ead[key], 4),
            "PatchCore":   round(pc[key],  4),
        }
        for key in METRIC_KEYS
    ]
    df = pd.DataFrame(rows).set_index("Metric")
    df.to_csv(out_dir / "summary_metrics.csv")
    (out_dir / "summary_metrics.txt").write_text(df.to_string())
    logger.info("Saved summary_metrics.csv / .txt")
    return df


def _save_per_class_table(ead: dict, pc: dict, out_dir: Path) -> pd.DataFrame:
    all_classes = sorted(
        set(ead["per_class_auroc"]) | set(pc["per_class_auroc"])
    )
    rows = [
        {
            "Class":       cls,
            "EfficientAD": round(ead["per_class_auroc"].get(cls, float("nan")), 4),
            "PatchCore":   round(pc["per_class_auroc"].get(cls,  float("nan")), 4),
        }
        for cls in all_classes
    ]
    df = pd.DataFrame(rows).set_index("Class")
    df.to_csv(out_dir / "per_class_image_auroc.csv")
    (out_dir / "per_class_image_auroc.txt").write_text(df.to_string())
    logger.info("Saved per_class_image_auroc.csv / .txt")
    return df

_COLORS = {"EfficientAD": "#4C72B0", "PatchCore": "#DD8452"}


def _grouped_bar(
    df: pd.DataFrame,
    title: str,
    ylabel: str,
    path: Path,
    rot: int = 15,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.3), 5))
    x, w   = np.arange(len(df)), 0.35
    ax.bar(x - w / 2, df["EfficientAD"], w, label="EfficientAD", color=_COLORS["EfficientAD"])
    ax.bar(x + w / 2, df["PatchCore"],   w, label="PatchCore",   color=_COLORS["PatchCore"])
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=rot, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path.name}")

def _plot_anomaly_maps(ead: dict, pc: dict, out_dir: Path, n: int = 6) -> None:
    anomaly_idx = [i for i, y in enumerate(ead["y_true"]) if y == 1][:n]
    if not anomaly_idx:
        logger.warning("No anomalous samples found — skipping anomaly_maps.png")
        return

    num_samples = len(anomaly_idx)
    fig, axes = plt.subplots(3, num_samples + 1, 
                             figsize=(3 * (num_samples + 1), 9),
                             gridspec_kw={'width_ratios': [0.5] + [1] * num_samples})

    row_labels = ["GT Mask", "EfficientAD", "PatchCore"]
    data_rows  = [
        [ead["masks"][i] for i in anomaly_idx],
        [ead["maps"][i]  for i in anomaly_idx],
        [pc["maps"][i]   for i in anomaly_idx],
    ]

    for r, (label, row_data) in enumerate(zip(row_labels, data_rows)):
        ax_label = axes[r, 0]
        ax_label.text(0.5, 0.5, label, fontsize=14, fontweight='bold', 
                      ha='center', va='center', rotation=0)
        ax_label.axis("off")

        for c, img in enumerate(row_data):
            ax = axes[r, c + 1]
            cmap = "gray" if r == 0 else "jet"
            im = ax.imshow(img, cmap=cmap, interpolation="nearest")
            ax.axis("off")
            
            if r == 0:
                class_name = ead["classes"][anomaly_idx[c]]
                ax.set_title(f"Sample {anomaly_idx[c]}\n({class_name})", fontsize=10)

    fig.suptitle("Anomaly Localization Comparison", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_dir / "anomaly_maps.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved anomaly_maps.png with clear row labels")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")

    parser = argparse.ArgumentParser(
        description="Run EfficientAD + PatchCore inference and save results."
    )
    parser.add_argument("--test_path",  default=config.TEST_DATASET_PATH,
                        help=f"Path to test images (default: config.TEST_DATASET_PATH)")
    parser.add_argument("--mask_path",  default=config.MASK_DATASET_PATH,
                        help=f"Path to ground-truth masks (default: config.MASK_DATASET_PATH)")
    parser.add_argument("--index_path", default=config.PC_MEMORY_BANK_FAISS,
                        help=f"PatchCore .faiss index (default: config.PC_MEMORY_BANK_FAISS)")
    parser.add_argument("--output_dir", default=config.RESULTS_PATH,
                        help=f"Directory for all output files (default: config.RESULTS_PATH)")
    parser.add_argument("--device",     default="cuda",
                        help="Compute device: cuda | cpu  (default: cuda)")
    parser.add_argument("--n_maps",     type=int, default=6,
                        help="Number of anomaly map samples to visualise (default: 6)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running EfficientAD inference …")
    ead_res = run_ead(
        test_dataset_path=args.test_path,
        mask_dataset_path=args.mask_path,
        device=args.device,
    )

    logger.info("Running PatchCore inference …")
    pc_res = run_pc(
        test_dataset_path=args.test_path,
        mask_dataset_path=args.mask_path,
        index_path=args.index_path,
        device=args.device,
    )

    df_summary = _save_summary_table(ead_res, pc_res, out_dir)
    df_cls     = _save_per_class_table(ead_res, pc_res, out_dir)

    _grouped_bar(
        df_summary,
        title="EfficientAD vs PatchCore — Overall Metrics",
        ylabel="AUROC",
        path=out_dir / "metric_comparison.png",
        rot=15,
    )
    _grouped_bar(
        df_cls,
        title="Per-class Image AUROC — EfficientAD vs PatchCore",
        ylabel="Image AUROC",
        path=out_dir / "per_class_auroc.png",
        rot=30,
    )
    _plot_anomaly_maps(ead_res, pc_res, out_dir, n=args.n_maps)

    logger.info(f"All outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
