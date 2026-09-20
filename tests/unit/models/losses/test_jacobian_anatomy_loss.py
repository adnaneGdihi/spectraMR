"""The Jacobian guard: analytic determinants, then the ordering it has to produce.

The determinant is checked against fields whose ``det J`` is known in closed
form, because that is the half that must be exactly right. The loss itself is
checked for ORDERING only -- it sits on a data-dependent noise floor and the
module says so; asserting an absolute threshold here would pin a number that
means nothing on other data.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from spectramr.models.losses.jacobian_anatomy_loss import (
    JacobianAnatomyLoss,
    displacement_by_local_correlation,
    jacobian_determinant,
)
from spectramr.models.losses.registry import LossRegistry

SIDE = 48


def _phantom() -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, SIDE), torch.linspace(-1, 1, SIDE), indexing="ij")
    img = ((yy**2 + xx**2) < 0.5).float()
    img = img + 0.5 * (((yy - 0.3) ** 2 + (xx + 0.2) ** 2) < 0.05).float()
    img = img.unsqueeze(0).unsqueeze(0)
    return F.avg_pool2d(F.pad(img, (2, 2, 2, 2), mode="replicate"), 5, stride=1)


def _zoom(img: torch.Tensor, crop: int) -> torch.Tensor:
    inner = img[:, :, crop : SIDE - crop, crop : SIDE - crop]
    return F.interpolate(inner, size=(SIDE, SIDE), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# jacobian_determinant against closed-form fields
# ---------------------------------------------------------------------------


def test_a_zero_field_has_unit_determinant() -> None:
    det = jacobian_determinant(torch.zeros(1, 2, 16, 16))
    assert torch.allclose(det, torch.ones_like(det), atol=1e-6)


def test_a_constant_field_has_unit_determinant() -> None:
    """Translation is not anatomy distortion, and the guard must be blind to it."""
    field = torch.full((1, 2, 16, 16), 3.0)
    det = jacobian_determinant(field)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-6)


def test_a_uniform_dilation_matches_the_closed_form() -> None:
    """``u = a * r`` gives ``det J = (1 + a)^2``; at a = 0.1 that is 1.21."""
    yy, xx = torch.meshgrid(torch.arange(16.0), torch.arange(16.0), indexing="ij")
    field = torch.stack([0.1 * yy, 0.1 * xx]).unsqueeze(0)
    interior = jacobian_determinant(field)[..., 2:-2, 2:-2]
    assert float(interior.mean()) == pytest.approx(1.21, abs=1e-4)


def test_a_reversing_field_folds() -> None:
    """``det J <= 0`` is a topology change: two places became one."""
    yy, xx = torch.meshgrid(torch.arange(16.0), torch.arange(16.0), indexing="ij")
    field = torch.stack([-2.0 * yy, torch.zeros_like(xx)]).unsqueeze(0)
    det = jacobian_determinant(field)
    assert float((det <= 0).float().mean()) == pytest.approx(1.0, abs=1e-6)


def test_a_malformed_field_raises() -> None:
    with pytest.raises(ValueError, match=r"must be \[B, 2, H, W\]"):
        jacobian_determinant(torch.zeros(1, 3, 8, 8))


# ---------------------------------------------------------------------------
# The displacement estimator
# ---------------------------------------------------------------------------


def test_the_estimator_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        displacement_by_local_correlation(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 9))


def test_the_estimator_bounds_the_field_it_can_express() -> None:
    """A guard that can explain away an arbitrarily large warp reports no warp."""
    img = _phantom()
    field = displacement_by_local_correlation(img, torch.roll(img, (1, 0), (-2, -1)), search=2)
    assert field.abs().max() <= 2.0 + 1e-5


def test_a_search_below_one_raises() -> None:
    with pytest.raises(ValueError, match="search must be >= 1"):
        displacement_by_local_correlation(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8), search=0)


# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------


def test_the_loss_is_registered_and_resolvable() -> None:
    """Decorating is the easy half; the arm reaches it through the registry."""
    assert "jacobian_anatomy" in LossRegistry.list_available()
    assert isinstance(LossRegistry.create("jacobian_anatomy"), JacobianAnatomyLoss)


def test_a_worse_warp_scores_higher() -> None:
    """Ordering, not absolute value -- the module documents the noise floor."""
    img = _phantom()
    loss = JacobianAnatomyLoss()
    mild = _zoom(img, 6)
    severe = _zoom(img, 9)
    assert float(loss(img, img)) < float(loss(mild, img)) < float(loss(severe, img))


def test_a_rigid_shift_is_not_charged_as_distortion() -> None:
    """Translation has unit Jacobian, so it must land on the same floor as an
    identical pair -- a shifted image has lost no anatomy."""
    img = _phantom()
    loss = JacobianAnatomyLoss()
    identical = float(loss(img, img))
    shifted = float(loss(torch.roll(img, (1, 0), (-2, -1)), img))
    assert abs(shifted - identical) < 0.25 * identical


def test_folding_fraction_is_the_absolute_companion() -> None:
    """Zero for a clean pair, positive for a real warp -- unlike the loss, this
    one is comparable against a threshold."""
    img = _phantom()
    loss = JacobianAnatomyLoss()
    assert float(loss.folding_fraction(img, img)) == pytest.approx(0.0, abs=1e-6)
    assert float(loss.folding_fraction(_zoom(img, 6), img)) > 0.0


def test_a_complex_input_is_reduced_to_geometry() -> None:
    """A warp is about where tissue is, not about its phase."""
    img = _phantom()
    loss = JacobianAnatomyLoss()
    as_complex = (img * torch.exp(1j * torch.rand_like(img))).to(torch.complex64)
    assert float(loss(as_complex, img)) == pytest.approx(float(loss(img, img)), abs=1e-4)


def test_multichannel_input_is_combined_before_estimation() -> None:
    """Displacement estimated on a coil profile is not displacement of anatomy."""
    img = _phantom()
    loss = JacobianAnatomyLoss()
    stacked = img.repeat(1, 4, 1, 1)
    assert torch.isfinite(loss(stacked, stacked)).all()


def test_the_structure_weight_ignores_flat_background() -> None:
    """A correlation peak in flat background is arbitrary; averaging over it
    charged a constant 0.165 to two identical images."""
    img = _phantom()
    weight = JacobianAnatomyLoss.structure_weight(img)
    assert float(weight.max()) == pytest.approx(1.0, abs=1e-5)
    assert float(weight[..., 0, 0]) < 1e-3


@pytest.mark.parametrize("kw", [{"fold_weight": -1.0}, {"volume_weight": -1.0}])
def test_a_negative_weight_raises(kw) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        JacobianAnatomyLoss(**kw)


def test_disabling_both_terms_raises() -> None:
    """A guard that is identically zero is a declared mechanism that cannot fire."""
    with pytest.raises(ValueError, match="cannot fire"):
        JacobianAnatomyLoss(fold_weight=0.0, volume_weight=0.0)
