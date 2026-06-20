"""DeepGlobe Road Extraction dataset (Phase 1.1).

Pairs `*_sat.jpg` (RGB) with `*_mask.png` (white-on-black road mask). Masks are
binarized to {0, 1}. A seeded 90/10 split keeps train/val reproducible.
"""
from __future__ import annotations

import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _list_pairs(root_dir: str) -> list[tuple[str, str]]:
    """Return sorted (sat, mask) path pairs that both exist."""
    sat_paths = sorted(glob.glob(os.path.join(root_dir, "*_sat.jpg")))
    pairs = []
    for sat in sat_paths:
        mask = sat.replace("_sat.jpg", "_mask.png")
        if os.path.exists(mask):
            pairs.append((sat, mask))
    return pairs


class DeepGlobeRoadDataset(Dataset):
    def __init__(self, root_dir, split="train", transforms=None, val_frac=0.1, seed=42):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.transforms = transforms

        pairs = _list_pairs(root_dir)
        if not pairs:
            raise FileNotFoundError(
                f"No *_sat.jpg / *_mask.png pairs found in {root_dir!r}. "
                "Download DeepGlobe into data/deepglobe/train/."
            )

        # Seeded shuffle, then deterministic split.
        rng = random.Random(seed)
        rng.shuffle(pairs)
        n_val = max(1, int(len(pairs) * val_frac))
        self.pairs = pairs[n_val:] if split == "train" else pairs[:n_val]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        sat_path, mask_path = self.pairs[idx]

        img = cv2.imread(sat_path, cv2.IMREAD_COLOR)  # BGR uint8
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # uint8, white roads
        mask = (mask > 127).astype(np.float32)  # binarize -> {0., 1.}

        if self.transforms is not None:
            out = self.transforms(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
            # ToTensorV2 leaves mask as HxW; ensure float32.
            mask = mask.float()
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask)

        return img, mask
