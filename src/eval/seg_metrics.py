"""Segmentation evaluation metrics (Phase 1.6).

All functions take binary numpy arrays (or tensors convertible via np.asarray)
of shape (H, W) with values in {0, 1}: `pred` and `target`. They return plain
floats so they're easy to tabulate for the pitch.

- iou_score / dice_score: standard overlap metrics
- relaxed_iou: dilates the ground truth by N px before scoring, rewarding
  near-misses (fair for thin, hand-drawn road labels)
- occlusion_recall: recall computed ONLY on pixels flagged as occluded — our
  signature "robustness under trees/shadows" number
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


def _binarize(x) -> np.ndarray:
    return (np.asarray(x) > 0.5).astype(np.uint8)


def iou_score(pred, target, eps: float = 1e-6) -> float:
    p, t = _binarize(pred), _binarize(target)
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return float((inter + eps) / (union + eps))


def dice_score(pred, target, eps: float = 1e-6) -> float:
    p, t = _binarize(pred), _binarize(target)
    inter = np.logical_and(p, t).sum()
    return float((2 * inter + eps) / (p.sum() + t.sum() + eps))


def relaxed_iou(pred, target, buffer_px: int = 3, eps: float = 1e-6) -> float:
    """IoU after dilating the ground truth by `buffer_px`.

    A prediction within `buffer_px` of a true road counts as a hit. Rewards
    near-misses, which matters because road labels are only a few pixels wide.
    """
    p, t = _binarize(pred), _binarize(target)
    struct = np.ones((2 * buffer_px + 1, 2 * buffer_px + 1), dtype=bool)
    t_dil = binary_dilation(t, structure=struct)
    inter = np.logical_and(p, t_dil).sum()
    union = np.logical_or(p, t_dil).sum()
    return float((inter + eps) / (union + eps))


def occlusion_recall(pred, target, occlusion_mask, eps: float = 1e-6) -> float:
    """Recall computed only on pixels marked occluded.

    occlusion_mask: (H, W) {0,1}, 1 where a road is hidden by tree/shadow.
    Of the true-road pixels inside occluded regions, what fraction did we find?
    This is the occlusion-robustness benchmark for the pitch.
    """
    p, t, o = _binarize(pred), _binarize(target), _binarize(occlusion_mask)
    true_occ = np.logical_and(t, o)
    hit = np.logical_and(p, true_occ).sum()
    total = true_occ.sum()
    return float((hit + eps) / (total + eps))
