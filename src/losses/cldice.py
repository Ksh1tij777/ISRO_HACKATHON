"""clDice — topology-preserving loss (Phase 1.4).

Differentiable soft-skeleton via iterative soft erosion/dilation, per
Shit et al., CVPR 2021. The skeleton-overlap terms reward connectivity, which
is exactly what road-network extraction needs (a broken road costs IoU little
but topology a lot).

Operates on 4D (B, 1, H, W) internally; accepts (B, H, W) and unsqueezes.
"""
import torch
import torch.nn.functional as F


def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), 1, (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), 1, (0, 1))
    return torch.min(p1, p2)


def soft_dilate(img):
    return F.max_pool2d(img, (3, 3), 1, 1)


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iters=5):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cl_dice_loss(pred, target, iters=5, eps=1e-6):
    """pred: (B, H, W) raw logits. target: (B, H, W) {0,1} float."""
    pred = torch.sigmoid(pred)
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
    skel_pred = soft_skel(pred, iters)
    skel_true = soft_skel(target, iters)
    tprec = (skel_pred * target).sum() / (skel_pred.sum() + eps)
    tsens = (skel_true * pred).sum() / (skel_true.sum() + eps)
    cl_dice = 2.0 * tprec * tsens / (tprec + tsens + eps)
    return 1 - cl_dice
