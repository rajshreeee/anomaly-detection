# src/patchcore/inference.py

import logging
from pathlib import Path

import config
import faiss
import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TestDataset, MaskDataset
from metrics import (
    image_auroc, pixel_auroc, pixel_auroc_anomalous_only,
    per_class_image_auroc, summarize
)
from .model import FeatureExtractor, patchify


logger = logging.getLogger(__name__)

PC_IMAGE_SIZE = config.PC_IMAGE_SIZE
PC_BATCH_SIZE = config.PC_BATCH_SIZE
PC_LAYERS = config.PC_LAYERS
PC_PRETRAIN_DIM = config.PC_PRETRAIN_DIM
PC_TARGET_DIM = config.PC_TARGET_DIM
PC_PATCH_SIZE = config.PC_PATCH_SIZE
PC_CORESET_RATIO = config.PC_CORESET_RATIO
PC_CORESET_PROJ_DIM = config.PC_CORESET_PROJ_DIM
PC_N_NEAREST = config.PC_N_NEAREST
PC_OUTPUT_DIR = config.PC_OUTPUT_DIR
PC_MEMORY_BANK_FAISS = config.PC_MEMORY_BANK_FAISS
PC_MEMORY_BANK_NPY = config.PC_MEMORY_BANK_NPY

def _extract_features(
    img:       torch.Tensor,
    extractor: FeatureExtractor,
    device:    str,
) -> np.ndarray:
    """
    Extract, aggregate and project patch features from one image.
    Returns [N_patches, PC_TARGET_DIM] float32 numpy array.
    """
    B         = img.shape[0]
    feat_map  = extractor(img.to(device))
    layer_patches, ref_shape = [], None

    for layer_name in PC_LAYERS:
        f = feat_map[layer_name].cpu()
        patches, shape = patchify(f, PC_PATCH_SIZE)

        if ref_shape is None:
            ref_shape = shape
        elif shape != ref_shape:
            patches = patches.reshape(B, shape[0], shape[1], -1).permute(0, 3, 1, 2)
            patches = F.interpolate(patches.float(), size=ref_shape,
                                    mode='bilinear', align_corners=False)
            patches = patches.permute(0, 2, 3, 1).reshape(B, -1, patches.shape[1])

        _, N, D = patches.shape
        flat    = patches.reshape(-1, D)
        proj    = F.adaptive_avg_pool1d(
            flat.unsqueeze(1).expand(-1, 1, -1), PC_PRETRAIN_DIM
        ).squeeze(1)
        layer_patches.append(proj.reshape(B, N, PC_PRETRAIN_DIM))

    combined = torch.cat(layer_patches, dim=-1)          # [B, N, PRETRAIN_DIM * n_layers]
    B, N, D  = combined.shape
    final    = F.adaptive_avg_pool1d(
        combined.reshape(-1, D).unsqueeze(1).expand(-1, 1, -1), PC_TARGET_DIM
    ).squeeze(1)                                         # [B*N, TARGET_DIM]

    return final.numpy().astype(np.float32), ref_shape


def _score_map(
    distances:  np.ndarray,
    ref_shape:  tuple,
    eval_size:  int,
    gaussian_sigma: float = 4.0,
) -> np.ndarray:
    """
    Reshape patch distances into a spatial map, upsample and smooth.
    Returns [eval_size, eval_size] float32 numpy array.
    """
    score_map = torch.from_numpy(
        distances[:, 0].reshape(ref_shape[0], ref_shape[1])
    ).unsqueeze(0).unsqueeze(0)                          # [1, 1, H, W]

    score_map = F.interpolate(score_map, size=(eval_size, eval_size),
                               mode='bilinear', align_corners=False)
    score_map = scipy.ndimage.gaussian_filter(
        score_map.squeeze().numpy(), sigma=gaussian_sigma
    )
    return score_map


def run_inference(
    test_dataset_path: str,
    mask_dataset_path: str,
    index_path:        str,
    device:            str = 'cuda',
) -> dict:
    """
    Full PatchCore inference + evaluation pipeline.

    Args:
        test_dataset_path : path to carpet/test/
        mask_dataset_path : path to carpet/ground_truth/
        index_path        : path to saved .faiss index
        device            : 'cuda' or 'cpu'

    Returns:
        dict with keys: scores, maps, masks, y_true, classes,
                        image_auroc, full_pixel_auroc, anomaly_pixel_auroc,
                        per_class_auroc
    """
    # ── Data ──
    test_ds = TestDataset(test_dataset_path, image_size=PC_IMAGE_SIZE)
    mask_ds = MaskDataset(test_ds, mask_dataset_path, image_size=PC_IMAGE_SIZE)
    loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    logger.info(f"Test set: {len(test_ds)} images | Classes: {test_ds.classes}")

    # ── Load FAISS index ──
    index = faiss.read_index(index_path)
    if device == 'cuda' and faiss.get_num_gpus() > 0:
        res   = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    logger.info(f"Memory bank: {index.ntotal} vectors")

    # ── Model ──
    extractor = FeatureExtractor(layers=PC_LAYERS, device=device)

    # ── Inference ──
    scores, maps, masks_np, y_true, classes = [], [], [], [], []

    with torch.no_grad():
        for i, (img, label, path) in enumerate(tqdm(loader, desc='PatchCore inference')):
            y_true.append(label.item())
            classes.append(Path(path[0]).parent.name)
            masks_np.append(mask_ds[i].squeeze().numpy())   # [H, W]

            features, ref_shape = _extract_features(img, extractor, device)
            dists, _            = index.search(features, 1)  # [N_patches, 1]

            amap = _score_map(dists, ref_shape, PC_IMAGE_SIZE)
            maps.append(amap)
            scores.append(float(amap.max()))

    logger.info("Inference complete")

    # ── Metrics ──
    img_auc         = image_auroc(y_true, scores)
    full_px_auc     = pixel_auroc(masks_np, maps)
    anomaly_px_auc  = pixel_auroc_anomalous_only(masks_np, maps)

    logger.info("Per-class image AUROC:")
    per_class = per_class_image_auroc(y_true, scores, classes)

    results = {
        'Image AUROC'         : img_auc,
        'Full Pixel AUROC'    : full_px_auc,
        'Anomaly Pixel AUROC' : anomaly_px_auc,
    }
    summarize({'PatchCore': results})

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
    logging.basicConfig(level=logging.INFO, format='%(name)s — %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path',  required=True)
    parser.add_argument('--mask_path',  required=True)
    parser.add_argument('--index_path', required=True)
    parser.add_argument('--device',     default='cuda')
    args = parser.parse_args()

    run_inference(
        test_dataset_path = args.test_path,
        mask_dataset_path = args.mask_path,
        index_path        = args.index_path,
        device            = args.device,
    )
