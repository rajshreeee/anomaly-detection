# src/efficientad/inference.py

import logging
from pathlib import Path

import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import TestDataset, MaskDataset
from metrics import (
    image_auroc, pixel_auroc, pixel_auroc_anomalous_only,
    per_class_image_auroc, summarize
)
from .model import get_pdn_small, get_autoencoder, predict_map, load_teacher

logger = logging.getLogger(__name__)

EAD_IMAGE_SIZE   = config.EAD_IMAGE_SIZE
EAD_OUT_CHANNELS = config.EAD_OUT_CHANNELS
EAD_TRAIN_STEPS  = config.EAD_TRAIN_STEPS
EAD_BATCH_SIZE   = config.EAD_BATCH_SIZE
EAD_OUTPUT_DIR = config.EAD_OUTPUT_DIR
EAD_MODEL_PATH = config.EAD_MODEL_PATH
EAD_LR           = config.EAD_LR
EAD_WEIGHT_DECAY = config.EAD_WEIGHT_DECAY
EAD_VAL_RATIO    = config.EAD_VAL_RATIO
EAD_SEED         = config.EAD_SEED
EAD_TEACHER_PATH = config.EAD_TEACHER_PATH
EAD_STUDENT_PATH = config.EAD_STUDENT_PATH
EAD_AUTOENCODER_PATH = config.EAD_AUTOENCODER_PATH
EAD_STATS_PATH = config.EAD_STATS_PATH

def _load_models(device: str):
    """
    Load teacher, student, autoencoder and normalization stats from output dir.
    Teacher weights always come from the pretrained path in config.
    """
    teacher     = load_teacher(EAD_TEACHER_PATH, EAD_OUT_CHANNELS, device)
    student     = get_pdn_small(2 * EAD_OUT_CHANNELS).to(device)
    autoencoder = get_autoencoder(EAD_OUT_CHANNELS).to(device)

    student.load_state_dict(
        torch.load(EAD_STUDENT_PATH, map_location=device)
    )
    autoencoder.load_state_dict(
        torch.load(EAD_AUTOENCODER_PATH, map_location=device)
    )

    stats       = torch.load(EAD_STATS_PATH, map_location=device)
    t_mean      = stats['mean']
    t_std       = stats['std']
    q_st_start  = stats['q_st_start']
    q_st_end    = stats['q_st_end']
    q_ae_start  = stats['q_ae_start']
    q_ae_end    = stats['q_ae_end']

    teacher.eval(); student.eval(); autoencoder.eval()

    return teacher, student, autoencoder, t_mean, t_std, \
           q_st_start, q_st_end, q_ae_start, q_ae_end


def _get_anomaly_map(
    img:         torch.Tensor,
    teacher:     torch.nn.Module,
    student:     torch.nn.Module,
    autoencoder: torch.nn.Module,
    t_mean:      torch.Tensor,
    t_std:       torch.Tensor,
    q_st_start:  torch.Tensor,
    q_st_end:    torch.Tensor,
    q_ae_start:  torch.Tensor,
    q_ae_end:    torch.Tensor,
    device:      str,
) -> np.ndarray:
    """
    Run one image through EfficientAD and return [EVAL_SIZE, EVAL_SIZE] anomaly map.

    Steps:
      1. Resize image to EAD_IMAGE_SIZE (256) — model's expected input
      2. Forward pass → combined ST + AE anomaly map
      3. Pad edges (teacher PDN loses border pixels due to no-padding conv)
      4. Upsample to EVAL_SIZE (224) for fair comparison with masks
    """
    img_ead = F.interpolate(
        img.to(device),
        size=(EAD_IMAGE_SIZE, EAD_IMAGE_SIZE),
        mode='bilinear', align_corners=False
    )                                                    # [1, 3, 256, 256]

    amap = predict_map(
        img_ead, teacher, student, autoencoder,
        t_mean, t_std, EAD_OUT_CHANNELS,
        q_st_start, q_st_end, q_ae_start, q_ae_end
    )                                                    # [1, 1, ~56, ~56]

    amap = F.pad(amap, (4, 4, 4, 4))                    # restore border lost by PDN convs
    amap = F.interpolate(amap, size=(EAD_IMAGE_SIZE, EAD_IMAGE_SIZE),
                         mode='bilinear', align_corners=False)
    return amap.squeeze().cpu().numpy()                  # [EVAL_SIZE, EVAL_SIZE]


def run_inference(
    test_dataset_path: str,
    mask_dataset_path: str,
    device:            str = 'cuda',
) -> dict:
    """
    Full EfficientAD inference + evaluation pipeline.

    Args:
        test_dataset_path : path to carpet/test/
        mask_dataset_path : path to carpet/ground_truth/
        ckpt_dir          : directory containing student_final.pth,
                            autoencoder_final.pth, stats.pth
        device            : 'cuda' or 'cpu'

    Returns:
        dict with keys: scores, maps, masks, y_true, classes,
                        image_auroc, full_pixel_auroc, anomaly_pixel_auroc,
                        per_class_auroc
    """
    # ── Data ──
    # Load test images at EVAL_SIZE — resized to EAD_IMAGE_SIZE inside _get_anomaly_map
    test_ds = TestDataset(test_dataset_path, image_size=EAD_IMAGE_SIZE)
    mask_ds = MaskDataset(test_ds, mask_dataset_path, image_size=EAD_IMAGE_SIZE)
    loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    logger.info(f"Test set: {len(test_ds)} images | Classes: {test_ds.classes}")

    # ── Models ──
    (teacher, student, autoencoder,
     t_mean, t_std,
     q_st_start, q_st_end,
     q_ae_start, q_ae_end) = _load_models(device)

    # ── Inference ──
    scores, maps, masks_np, y_true, classes = [], [], [], [], []

    with torch.no_grad():
        for i, (img, label, path) in enumerate(tqdm(loader, desc='EfficientAD inference')):
            y_true.append(label.item())
            classes.append(Path(path[0]).parent.name)
            masks_np.append(mask_ds[i].squeeze().numpy())    # [H, W]

            amap = _get_anomaly_map(
                img, teacher, student, autoencoder,
                t_mean, t_std,
                q_st_start, q_st_end,
                q_ae_start, q_ae_end,
                device
            )
            maps.append(amap)
            scores.append(float(amap.max()))

    logger.info("Inference complete")

    # ── Metrics ──
    img_auc        = image_auroc(y_true, scores)
    full_px_auc    = pixel_auroc(masks_np, maps)
    anomaly_px_auc = pixel_auroc_anomalous_only(masks_np, maps)

    logger.info("Per-class image AUROC:")
    per_class = per_class_image_auroc(y_true, scores, classes)

    results = {
        'Image AUROC'         : img_auc,
        'Full Pixel AUROC'    : full_px_auc,
        'Anomaly Pixel AUROC' : anomaly_px_auc,
    }
    summarize({'EfficientAD': results})

    return {
        'scores'              : scores,
        'maps'                : maps,
        'masks'               : masks_np,
        'y_true'              : y_true,
        'classes'             : classes,
        'per_class_auroc'     : per_class,
        **results,
    }


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(name)s — %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--mask_path', required=True)
    parser.add_argument('--device',    default='cuda')
    args = parser.parse_args()

    run_inference(
        test_dataset_path = args.test_path,
        mask_dataset_path = args.mask_path,
        device            = args.device,
    )
