import warnings

import pytest
import torch

from spectramr.models.losses.noise2noise_loss import Noise2NoiseLoss


def test_l2_and_l1_match_the_plain_losses():
    a, b = torch.randn(2, 1, 4, 4), torch.randn(2, 1, 4, 4)
    assert torch.allclose(Noise2NoiseLoss()(a, b), torch.nn.functional.mse_loss(a, b))
    assert torch.allclose(Noise2NoiseLoss("l1")(a, b), torch.nn.functional.l1_loss(a, b))


def test_unknown_norm_raises():
    with pytest.raises(ValueError, match="norm must be"):
        Noise2NoiseLoss("l3")


def test_identical_views_warn_once_not_every_step():
    x = torch.randn(2, 1, 4, 4)
    loss = Noise2NoiseLoss()
    with pytest.warns(RuntimeWarning, match="bit-identical"):
        loss(x, x.clone())
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        loss(x, x.clone())
    assert not seen
