"""Combined segmentation loss (Phase 1.4): dice + bce + clDice."""
import torch.nn.functional as F

from .dice import dice_loss
from .cldice import cl_dice_loss

__all__ = ["dice_loss", "cl_dice_loss", "combined_loss"]


def combined_loss(pred, target, w_dice=0.5, w_bce=0.3, w_cldice=0.2,
                  cldice_iters=5):
    """pred: (B, H, W) single-channel logits. target: (B, H, W) {0,1} float.

    For SegFormer's 2-class output, collapse with
    `models.segformer.to_binary_logits` before calling this.
    """
    return (
        w_dice * dice_loss(pred, target)
        + w_bce * F.binary_cross_entropy_with_logits(pred, target)
        + w_cldice * cl_dice_loss(pred, target, iters=cldice_iters)
    )
