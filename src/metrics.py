import logging
from typing import List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

def image_auroc(y_true: List[int], scores: List[float]) -> float:
    auc = roc_auc_score(y_true, scores)
    logger.info(f"Image AUROC: {auc*100:.2f}%")
    return auc

def pixel_auroc(
    masks_gt:  List[np.ndarray],
    maps_pred: List[np.ndarray],
) -> float:
    flat_gt   = np.concatenate([m.flatten() for m in masks_gt])
    flat_pred = np.concatenate([m.flatten() for m in maps_pred])
    auc = roc_auc_score(flat_gt, flat_pred)
    logger.info(f"Full pixel AUROC: {auc*100:.2f}%")
    return auc


def pixel_auroc_anomalous_only(
    masks_gt:  List[np.ndarray],
    maps_pred: List[np.ndarray],
) -> float:
    sel_masks = [m for m in masks_gt  if np.sum(m) > 0]
    sel_maps  = [m for m in maps_pred if True]  # filter by same indices

    # Filter both by defective mask presence
    pairs = [(gt, pred) for gt, pred in zip(masks_gt, maps_pred) if np.sum(gt) > 0]
    if not pairs:
        logger.warning("No anomalous images found for pixel AUROC (anomalous only)")
        return float('nan')

    sel_masks, sel_maps = zip(*pairs)
    flat_gt   = np.concatenate([m.flatten() for m in sel_masks])
    flat_pred = np.concatenate([m.flatten() for m in sel_maps])
    auc = roc_auc_score(flat_gt, flat_pred)
    logger.info(f"Anomaly-only pixel AUROC: {auc*100:.2f}%")
    return auc


def per_class_image_auroc(
    y_true:   List[int],
    scores:   List[float],
    classes:  List[str],
) -> dict:
    results = {}
    defect_classes = [c for c in set(classes) if c != 'good']
    good_indices   = [i for i, c in enumerate(classes) if c == 'good']

    for cls in sorted(defect_classes):
        cls_indices = [i for i, c in enumerate(classes) if c == cls]
        idx = good_indices + cls_indices

        sub_true   = [y_true[i]  for i in idx]
        sub_scores = [scores[i]  for i in idx]

        try:
            auc = roc_auc_score(sub_true, sub_scores)
        except ValueError:
            auc = float('nan')

        results[cls] = auc
        logger.info(f"  {cls:25s} AUROC: {auc*100:.2f}%")

    return results


def paired_bootstrap(
    y_true:   List[int],
    scores_a: List[float],
    scores_b: List[float],
    n_iter:   int = 10000,
    seed:     int = 42,
) -> Tuple[float, float]:
    rng  = np.random.default_rng(seed)
    y    = np.array(y_true)
    sa   = np.array(scores_a)
    sb   = np.array(scores_b)
    n    = len(y)

    observed_diff = abs(roc_auc_score(y, sa) - roc_auc_score(y, sb))

    count = 0
    for _ in range(n_iter):
        idx  = rng.integers(0, n, size=n)
        diff = abs(roc_auc_score(y[idx], sa[idx]) - roc_auc_score(y[idx], sb[idx]))
        if diff >= observed_diff:
            count += 1

    p_value = count / n_iter
    logger.info(f"Bootstrap p-value: {p_value:.4f}  |  Observed diff: {observed_diff*100:.2f}%")
    return p_value, observed_diff


def summarize(results: dict) -> str:
    if not results:
        return ""

    models   = list(results.keys())
    metrics  = list(next(iter(results.values())).keys())
    col_w    = max(len(m) for m in metrics) + 2
    name_w   = max(len(m) for m in models) + 2

    header = f"{'Model':<{name_w}}" + "".join(f"{m:>{col_w}}" for m in metrics)
    sep    = "-" * len(header)
    rows   = [header, sep]

    for model, vals in results.items():
        row = f"{model:<{name_w}}"
        for metric in metrics:
            v = vals.get(metric, float('nan'))
            row += f"{v*100:>{col_w}.2f}%"[:-1] + "%"  # format as XX.XX%
        rows.append(row)

    table = "\n".join(rows)
    print(table)
    return table

