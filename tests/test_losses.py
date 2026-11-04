import pytest
import torch

from btseg.losses import build_loss


def target(shape=(2, 3, 16, 16, 16), seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(shape, generator=g) > 0.7).float()


@pytest.mark.parametrize("name", ["dice", "dicefocal"])
def test_loss_ranks_good_predictions_lower_and_backprops(name):
    loss_fn = build_loss(name)
    y = target()
    good = (y * 2 - 1) * 10  # confident, correct logits
    bad = -good
    assert loss_fn(good, y).item() < 0.05
    assert loss_fn(bad, y).item() > 0.5

    logits = torch.zeros_like(y, requires_grad=True)
    loss = loss_fn(logits, y)
    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


@pytest.mark.parametrize("name", ["dice", "dicefocal"])
def test_empty_region_with_empty_prediction_is_not_penalized(name):
    y = torch.zeros(1, 3, 128, 128, 128)  # full training crop, region absent
    assert build_loss(name)(torch.full_like(y, -10.0), y).item() < 0.01
    # a false-positive blob on an empty region is penalized
    fp = torch.full_like(y, -10.0)
    fp[:, :, :8, :8, :8] = 10.0
    assert build_loss(name)(fp, y).item() > 0.9


def test_kwargs_pass_through_and_softmax_is_rejected():
    loss = build_loss("dicefocal", lambda_focal=2.0)
    assert loss.lambda_focal == 2.0
    with pytest.raises(ValueError, match="sigmoid"):
        build_loss("dice", sigmoid=False, softmax=True)
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss("ce")
