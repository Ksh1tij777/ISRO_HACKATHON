"""Baseline U-Net (Phase 1.2).

Exists purely as a reference number to beat. Returns single-channel logits
(B, 1, H, W); squeeze to (B, H, W) for the binary losses.
"""
import segmentation_models_pytorch as smp


def build_unet(encoder="resnet34", in_channels=3, classes=1):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes,
    )
