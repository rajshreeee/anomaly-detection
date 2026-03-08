# src/patchcore/model.py

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Feature Extractor
# ─────────────────────────────────────────────

class FeatureExtractor(torch.nn.Module):
    """
    Extracts intermediate feature maps from WideResNet50 using forward hooks.
    Hooks into named layers (e.g. 'layer2', 'layer3') and collects their outputs.
    The backbone runs in eval mode and is never updated.
    """

    def __init__(self, layers: List[str], device: str):
        super().__init__()
        self.device = device
        self.layers = layers
        self.outputs: Dict[str, torch.Tensor] = {}
        self._hooks = []

        backbone = models.wide_resnet50_2(
            weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        )
        backbone.eval()

        for name in layers:
            layer = dict(backbone.named_modules())[name]
            self._hooks.append(
                layer.register_forward_hook(
                    lambda m, i, o, n=name: self.outputs.update({n: o})
                )
            )
            logger.debug(f"Hook registered on layer: {name}")

        self.backbone = backbone.to(device)
        logger.info(f"FeatureExtractor ready — layers: {layers}")

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.outputs.clear()
        self.backbone(x)
        return {k: v.detach() for k, v in self.outputs.items()}

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        logger.debug("All hooks removed")


# ─────────────────────────────────────────────
#  Patch & Aggregate
# ─────────────────────────────────────────────

def patchify(features: torch.Tensor, patch_size: int = 3) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Extract overlapping local patches around each spatial position.

    Args:
        features:   [B, C, H, W] feature map
        patch_size: neighbourhood size (3 = 3×3 local context per position)

    Returns:
        patches:    [B, H*W, C * patch_size^2]
        shape:      (H, W) spatial dimensions
    """
    padding = (patch_size - 1) // 2
    B, C, H, W = features.shape
    unfold  = torch.nn.Unfold(kernel_size=patch_size, padding=padding, stride=1)
    patches = unfold(features)          # [B, C*p*p, H*W]
    patches = patches.permute(0, 2, 1) # [B, H*W, C*p*p]
    return patches, (H, W)


def _mean_pool(features: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Pool a flat feature vector to target_dim via adaptive average pooling."""
    f = features.reshape(features.shape[0], 1, -1)
    return F.adaptive_avg_pool1d(f, target_dim).squeeze(1)


def extract_patch_features(
    dataloader,
    extractor: FeatureExtractor,
    layers: List[str],
    patch_size: int,
    pretrain_dim: int,
    target_dim: int,
    device: str,
    desc: str = 'Extracting features',
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Extract multi-scale patch features for all images in a dataloader.

    Pipeline per image:
      1. Extract feature maps from each layer via hooks
      2. Patchify each feature map (overlapping 3×3 neighbourhoods)
      3. Upsample smaller maps to match layer2 spatial resolution
      4. Project each layer's patches to pretrain_dim
      5. Concatenate across layers, pool to target_dim

    Returns:
        all_features: [N_total_patches, target_dim] numpy array
        all_shapes:   spatial shape (H, W) per batch (for score map reconstruction)
    """
    all_features = []
    all_shapes   = []

    for batch in tqdm(dataloader, desc=desc):
        imgs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)

        feat_map     = extractor(imgs)
        layer_patches = []
        ref_shape     = None
        B             = imgs.shape[0]

        for layer_name in layers:
            f = feat_map[layer_name].cpu()          # move off GPU immediately
            patches, shape = patchify(f, patch_size)

            if ref_shape is None:
                ref_shape = shape
            elif shape != ref_shape:
                # Upsample smaller feature maps (layer3) to match layer2 spatial size
                patches = patches.reshape(B, shape[0], shape[1], -1).permute(0, 3, 1, 2)
                patches = F.interpolate(
                    patches.float(), size=ref_shape, mode='bilinear', align_corners=False
                )
                patches = patches.permute(0, 2, 3, 1).reshape(B, -1, patches.shape[1])

            # Project to pretrain_dim
            _, N, D    = patches.shape
            flat       = patches.reshape(-1, D)
            projected  = _mean_pool(flat.unsqueeze(1).expand(-1, 1, -1), pretrain_dim)
            layer_patches.append(projected.reshape(B, N, pretrain_dim))

        # Concat layers → pool to target_dim
        combined = torch.cat(layer_patches, dim=-1)     # [B, N, pretrain_dim * n_layers]
        B, N, D  = combined.shape
        final    = _mean_pool(
            combined.reshape(-1, D).unsqueeze(1).expand(-1, 1, -1), target_dim
        )                                                # [B*N, target_dim]

        all_features.append(final.numpy())
        all_shapes.append(ref_shape)

    logger.info(f"Extracted {sum(f.shape[0] for f in all_features)} patch vectors")
    return np.concatenate(all_features, axis=0), all_shapes


# ─────────────────────────────────────────────
#  Greedy Coreset Sampler
# ─────────────────────────────────────────────

def greedy_coreset(
    features: np.ndarray,
    ratio: float,
    proj_dim: int = 128,
    n_starting_points: int = 10,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Approximate Greedy Coreset subsampling.

    Selects a maximally spread subset of features without building the full
    N×N distance matrix. Instead, distances are approximated relative to
    n_starting_points random anchors — O(N × n_starts) memory instead of O(N²).

    Args:
        features:         [N, D] memory bank candidates
        ratio:            fraction to keep, e.g. 0.10 for 10%
        proj_dim:         project to this dim before distance computation
        n_starting_points: number of random anchors for approximation
        device:           'cuda' or 'cpu'

    Returns:
        Subsampled features [n_select, D]
    """
    features_t = torch.from_numpy(features).float().to(device)
    N          = len(features_t)
    n_select   = max(1, int(N * ratio))

    # Random linear projection for cheaper distance computation
    with torch.no_grad():
        mapper  = torch.nn.Linear(features_t.shape[1], proj_dim, bias=False).to(device)
        reduced = mapper(features_t)                           # [N, proj_dim]

    # Compute distances to random starting anchors only
    start_idx = np.random.choice(N, min(n_starting_points, N), replace=False).tolist()
    with torch.no_grad():
        anchors = reduced[start_idx]                           # [n_starts, proj_dim]
        a2      = (reduced ** 2).sum(dim=1, keepdim=True)     # [N, 1]
        b2      = (anchors ** 2).sum(dim=1, keepdim=True).T   # [1, n_starts]
        ab      = reduced @ anchors.T                          # [N, n_starts]
        dist_to_anchors = (a2 + b2 - 2 * ab).clamp(min=0).sqrt()

    approx_dists = dist_to_anchors.mean(dim=1, keepdim=True)  # [N, 1]
    selected     = []

    for _ in tqdm(range(n_select), desc='Greedy coreset'):
        idx = torch.argmax(approx_dists).item()
        selected.append(idx)

        with torch.no_grad():
            sel       = reduced[idx: idx + 1]
            b2_sel    = (sel ** 2).sum()
            new_dists = (a2 + b2_sel - 2 * (reduced @ sel.T)).clamp(min=0).sqrt()

        approx_dists = torch.minimum(approx_dists, new_dists)

    selected = np.array(selected)
    logger.info(f"Coreset: {N} → {len(selected)} vectors ({ratio*100:.0f}%)")
    return features[selected]
