"""Dice loss for binary segmentation (Phase 1.4)."""
import torch


def dice_loss(pred, target, eps=1e-6):
    """pred: (B, H, W) raw logits. target: (B, H, W) {0,1} float."""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()
