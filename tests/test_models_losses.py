"""Phase 1.2-1.4 smoke tests: model forward shapes + loss sanity."""
import pytest
import torch

from src.losses import dice_loss, cl_dice_loss, combined_loss


def test_unet_forward_shape():
    from src.models.unet_baseline import build_unet
    model = build_unet().eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 256, 256), out.shape


def test_losses_perfect_vs_wrong():
    # logits: large +ve where road, large -ve where bg
    target = torch.zeros(2, 64, 64)
    target[:, 20:24, :] = 1.0  # a horizontal road band

    perfect = (target * 2 - 1) * 20.0  # +20 on road, -20 off
    wrong = -perfect

    for loss_fn in (dice_loss, cl_dice_loss, combined_loss):
        lp = loss_fn(perfect, target).item()
        lw = loss_fn(wrong, target).item()
        assert lp < lw, f"{loss_fn.__name__}: perfect {lp} !< wrong {lw}"
        assert lp < 0.2, f"{loss_fn.__name__}: perfect loss too high {lp}"


def test_combined_loss_is_differentiable():
    pred = torch.randn(2, 64, 64, requires_grad=True)
    target = (torch.rand(2, 64, 64) > 0.7).float()
    loss = combined_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(loss)


def test_cldice_rewards_connectivity():
    # A continuous line vs the same line with a gap: clDice should prefer continuous.
    target = torch.zeros(1, 64, 64)
    target[0, 32, :] = 1.0

    continuous = (target * 2 - 1) * 20.0
    broken = continuous.clone()
    broken[0, 32, 28:36] = -20.0  # punch a gap in the predicted road

    assert cl_dice_loss(continuous, target) < cl_dice_loss(broken, target)


@pytest.mark.network
def test_segformer_forward_fullres():
    try:
        from src.models.segformer import build_segformer, to_binary_logits
        model = build_segformer(num_labels=2).eval()
    except Exception as e:  # offline / download failure
        pytest.skip(f"SegFormer weights unavailable: {e}")
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 2, 512, 512), logits.shape
    assert to_binary_logits(logits).shape == (1, 512, 512)
