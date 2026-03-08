import json
import os
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────
#  Transform Factories
# ─────────────────────────────────────────────

def image_transform(image_size: int) -> transforms.Compose:
    """
    Standard ImageNet-normalized transform for model input.
    Used by both PatchCore (224) and EfficientAD (256) — just pass the right size.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def mask_transform(image_size: int) -> transforms.Compose:
    """
    Transform for ground-truth binary masks.
    - NEAREST interpolation: preserves binary 0/1 values, no blurry edges
    - No normalization: masks are labels, not pixel intensities
    """
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.NEAREST
        ),
        transforms.ToTensor(),   # [0, 255] -> [0.0, 1.0], shape [1, H, W]
    ])


def efficientad_train_transform(image_size: int) -> Callable:
    """
    EfficientAD needs two versions of each training image:
      - image_st : standard transform  (for student-teacher loss)
      - image_ae : color-jittered      (for autoencoder loss)
    Returns a callable that takes a PIL image and returns (image_st, image_ae).
    PatchCore does NOT use this — it has no augmentation during training.
    """
    base = image_transform(image_size)
    jitter = transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.2),
        transforms.ColorJitter(contrast=0.2),
        transforms.ColorJitter(saturation=0.2),
    ])
    def _transform(image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return base(image), base(jitter(image))

    return _transform


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Undo ImageNet normalization for visualization.
    Input:  [C, H, W] or [B, C, H, W] normalized tensor
    Output: same shape, values clamped to [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


# ─────────────────────────────────────────────
#  Dataset Classes
# ─────────────────────────────────────────────

class TrainDataset(ImageFolder):
    """
    Loads train/good/ — drops the class label since everything is normal.
    Accepts any transform callable, including EfficientAD's dual-image transform.
    """
    def __init__(self, train_dir: str, transform: Callable):
        super().__init__(train_dir, transform=None)
        self._img_transform = transform

    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        image = self.loader(path)
        return self._img_transform(image)


class TestDataset(ImageFolder):
    """
    Loads test/ — returns (image, label, path).
    Label is 0 for 'good', 1 for any defect class.
    Path is needed to locate the corresponding ground-truth mask.
    """
    def __init__(self, test_dir: str, image_size: int):
        super().__init__(test_dir, transform=None)
        self._transform = image_transform(image_size)
        self._good_idx  = self.class_to_idx.get('good', -1)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        path, target = self.samples[index]
        image = self._transform(self.loader(path))
        label = 0 if target == self._good_idx else 1
        return image, label, path

    @property
    def defect_classes(self) -> List[str]:
        return [c for c in self.classes if c != 'good']


class MaskDataset(torch.utils.data.Dataset):
    """
    Loads ground-truth masks from ground_truth/ to match test images.
    For 'good' test images (no mask file), returns a zero mask.

    Expected MVTec structure:
        ground_truth/<defect_class>/<image_name>_mask.png
    """
    def __init__(self, test_dataset: TestDataset, gt_dir: str, image_size: int):
        self.test_dataset  = test_dataset
        self.gt_dir        = Path(gt_dir)
        self._mask_tfm     = mask_transform(image_size)

    def __len__(self) -> int:
        return len(self.test_dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        path, target = self.test_dataset.samples[index]
        good_idx = self.test_dataset._good_idx

        if target == good_idx:
            # Good images have no defect — mask is all zeros
            return torch.zeros(1, self._mask_tfm.transforms[0].size[0],
                               self._mask_tfm.transforms[0].size[0])

        defect_class = Path(path).parent.name
        img_name     = Path(path).stem + '_mask.png'
        mask_path    = self.gt_dir / defect_class / img_name

        mask = Image.open(mask_path).convert('L')   # grayscale
        return self._mask_tfm(mask)


# ─────────────────────────────────────────────
#  Train / Val Split
# ─────────────────────────────────────────────

def make_split(
    dataset: torch.utils.data.Dataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """
    Splits a dataset into train/val subsets.
    Call this ONCE and save with save_split() so both models use identical splits.
    """
    n_total = len(dataset)
    n_val   = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    return torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )


def save_split(train_subset, val_subset, path: str) -> None:
    """Persist split indices to JSON so both models load identical splits."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(
        {'train': train_subset.indices, 'val': val_subset.indices},
        open(path, 'w'), indent=2
    )
    print(f"Split saved → {path}  (train={len(train_subset)}, val={len(val_subset)})")


def load_split(
    dataset: torch.utils.data.Dataset,
    path: str,
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """Load previously saved split indices and return Subset objects."""
    data = json.load(open(path))
    train_subset = torch.utils.data.Subset(dataset, data['train'])
    val_subset   = torch.utils.data.Subset(dataset, data['val'])
    print(f"Split loaded ← {path}  (train={len(train_subset)}, val={len(val_subset)})")
    return train_subset, val_subset
