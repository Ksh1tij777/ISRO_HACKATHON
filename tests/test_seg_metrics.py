"""Phase 1.6 smoke tests for segmentation metrics."""
import numpy as np

from src.eval.seg_metrics import iou_score, dice_score, relaxed_iou, occlusion_recall


def test_perfect_and_disjoint():
    t = np.zeros((32, 32)); t[10:14, :] = 1
    assert iou_score(t, t) > 0.999
    assert dice_score(t, t) > 0.999
    empty = np.zeros((32, 32))
    assert iou_score(empty, t) < 0.01


def test_iou_le_dice():
    rng = np.random.default_rng(0)
    p = (rng.random((64, 64)) > 0.5).astype(np.uint8)
    t = (rng.random((64, 64)) > 0.5).astype(np.uint8)
    # Dice >= IoU always
    assert dice_score(p, t) + 1e-6 >= iou_score(p, t)


def test_relaxed_rewards_near_miss():
    t = np.zeros((32, 32)); t[16, :] = 1          # true road on row 16
    p = np.zeros((32, 32)); p[18, :] = 1          # predicted 2px off
    assert relaxed_iou(p, t, buffer_px=3) > iou_score(p, t)
    assert iou_score(p, t) < 0.01                 # strict IoU misses entirely


def test_occlusion_recall():
    t = np.zeros((32, 32)); t[16, :] = 1          # road across the row
    occ = np.zeros((32, 32)); occ[:, 8:16] = 1    # occluded patch
    # prediction recovers the road only outside the occluded patch
    p = t.copy(); p[16, 8:16] = 0
    assert occlusion_recall(p, t, occ) < 0.01     # found none under occlusion
    # prediction that recovers under occlusion scores high
    assert occlusion_recall(t, t, occ) > 0.999
