"""Albumentations pipelines for DeepGlobe road segmentation (Phase 1.1).

Occlusion-focused augmentation is the key differentiator: CoarseDropout,
GridDropout, and a custom shadow overlay simulate the tree/building occlusion
that the model must be robust to on Sentinel-2.

Written against Albumentations 2.x (num_holes_range/fill API).
"""
from __future__ import annotations

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomShadow(A.ImageOnlyTransform):
    """Overlay a random dark polygon to simulate building/tree shadow.

    Picks a convex-ish polygon of random vertices and multiplies that region
    by a random factor in [0.4, 0.6], darkening it like a cast shadow.
    """

    def __init__(self, num_vertices_range=(3, 6), darkness_range=(0.4, 0.6), p=0.3):
        super().__init__(p=p)
        self.num_vertices_range = num_vertices_range
        self.darkness_range = darkness_range

    def apply(self, img, **params):
        h, w = img.shape[:2]
        n = np.random.randint(self.num_vertices_range[0], self.num_vertices_range[1] + 1)
        pts = np.column_stack([
            np.random.randint(0, w, size=n),
            np.random.randint(0, h, size=n),
        ]).astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        factor = np.random.uniform(*self.darkness_range)
        out = img.copy()
        region = mask.astype(bool)
        out[region] = (out[region].astype(np.float32) * factor).astype(img.dtype)
        return out

    def get_transform_init_args_names(self):
        return ("num_vertices_range", "darkness_range")


def get_train_transforms(size: int = 512) -> A.Compose:
    return A.Compose([
        A.RandomCrop(height=size, width=size, pad_if_needed=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        # --- occlusion-focused (key differentiator) ---
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 64),
            hole_width_range=(8, 64),
            fill=0,
            p=0.5,
        ),
        A.GridDropout(ratio=0.3, p=0.3),
        RandomShadow(p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(height=size, width=size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
