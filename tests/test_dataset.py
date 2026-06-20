"""Phase 1.1 smoke test.

Generates synthetic DeepGlobe-style pairs in a tmp dir so it runs without the
real download, while exercising the actual transform pipeline. Also runs
against real data/deepglobe/train if present.
"""
import os

import cv2
import numpy as np
import pytest
import torch

from src.data.dataset import DeepGlobeRoadDataset
from src.data.augmentations import get_train_transforms, get_val_transforms


def _make_fake_deepglobe(tmp_dir, n=4, size=1024):
    for i in range(n):
        img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.line(mask, (0, i * 50), (size, size - i * 50), 255, 12)  # a "road"
        cv2.imwrite(os.path.join(tmp_dir, f"{i}_sat.jpg"), img)
        cv2.imwrite(os.path.join(tmp_dir, f"{i}_mask.png"), mask)


@pytest.fixture
def fake_root(tmp_path):
    _make_fake_deepglobe(str(tmp_path))
    return str(tmp_path)


def test_train_sample_shapes_and_values(fake_root):
    ds = DeepGlobeRoadDataset(fake_root, split="train",
                              transforms=get_train_transforms(512))
    img, mask = ds[0]
    assert img.shape == (3, 512, 512), img.shape
    assert mask.shape == (512, 512), mask.shape
    assert img.dtype == torch.float32
    uniq = set(torch.unique(mask).tolist())
    assert uniq.issubset({0.0, 1.0}), uniq


def test_val_transforms(fake_root):
    ds = DeepGlobeRoadDataset(fake_root, split="val",
                              transforms=get_val_transforms(512))
    img, mask = ds[0]
    assert img.shape == (3, 512, 512)
    assert mask.shape == (512, 512)


def test_split_disjoint_and_seeded(fake_root):
    tr = DeepGlobeRoadDataset(fake_root, split="train")
    va = DeepGlobeRoadDataset(fake_root, split="val")
    assert len(tr) + len(va) == 4
    assert set(p[0] for p in tr.pairs).isdisjoint(p[0] for p in va.pairs)
    # re-instantiation gives identical split (seeded)
    assert DeepGlobeRoadDataset(fake_root, split="val").pairs == va.pairs


@pytest.mark.skipif(
    not os.path.isdir("data/deepglobe/train")
    or not any(f.endswith("_sat.jpg") for f in os.listdir("data/deepglobe/train"))
    if os.path.isdir("data/deepglobe/train") else True,
    reason="real DeepGlobe not present",
)
def test_real_deepglobe_loads():
    ds = DeepGlobeRoadDataset("data/deepglobe/train", split="train",
                              transforms=get_train_transforms(512))
    img, mask = ds[0]
    assert img.shape == (3, 512, 512)
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})
