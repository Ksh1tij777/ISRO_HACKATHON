"""SegFormer wrapper (Phase 1.3).

HuggingFace SegFormer emits logits at 1/4 resolution; this wrapper upsamples to
full input resolution so the forward call returns (B, num_labels, H, W).

For binary road segmentation the combined loss wants single-channel logits.
`to_binary_logits` collapses the 2-class output to (B, H, W) via the
class-1-minus-class-0 trick.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class SegformerRoad(nn.Module):
    def __init__(self, model_name="nvidia/segformer-b2-finetuned-ade-512-512",
                 num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,  # re-heads the classifier
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        logits = self.net(pixel_values=x).logits  # (B, num_labels, h/4, w/4)
        logits = F.interpolate(logits, size=(h, w), mode="bilinear",
                               align_corners=False)
        return logits  # (B, num_labels, H, W)


def to_binary_logits(logits):
    """(B, num_labels, H, W) -> (B, H, W) single-channel logit for binary loss."""
    if logits.dim() == 4 and logits.shape[1] >= 2:
        return logits[:, 1] - logits[:, 0]
    return logits.squeeze(1)


def build_segformer(model_name="nvidia/segformer-b2-finetuned-ade-512-512",
                    num_labels=2):
    return SegformerRoad(model_name, num_labels)
