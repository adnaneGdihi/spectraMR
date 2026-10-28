"""C5 null-space content loss: supervision confined to the unobserved bins."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.null_space_loss import NullSpaceContentLoss
from spectramr.models.losses.registry import create_loss, is_registered


def _mask() -> torch.Tensor:
    mask = torch.zeros(1, 1, 16, 16)
    mask[..., ::2] = 1.0
    return mask


class TestNullSpaceContentLoss:
    def test_zero_at_equality(self):
        mask = _mask()
        target = torch.randn(2, 2, 16, 16)
        loss = NullSpaceContentLoss()(target.clone(), target, mask=mask)
        assert loss.item() == pytest.approx(0.0)

    def test_insensitive_to_observed_lines(self):
        """Changing the prediction ONLY on the observed support changes nothing."""
        mask = _mask()
        target = torch.randn(2, 2, 16, 16)
        pred = target + torch.randn_like(target) * mask
        loss = NullSpaceContentLoss()(pred, target, mask=mask)
        assert loss.item() == pytest.approx(0.0)

    def test_planted_value_is_exact(self):
        """diff = c on every bin ⇒ per-null-bin mean = c² (mask-independent)."""
        mask = _mask()
        target = torch.zeros(1, 2, 16, 16)
        pred = torch.full((1, 2, 16, 16), 0.5)
        loss = NullSpaceContentLoss()(pred, target, mask=mask)
        assert loss.item() == pytest.approx(0.25, rel=1e-6)
        # Same per-bin error under a denser mask -> same loss (C5 scale rule).
        denser = torch.zeros(1, 1, 16, 16)
        denser[..., ::4] = 1.0
        loss_denser = NullSpaceContentLoss()(pred, target, mask=denser)
        assert loss_denser.item() == pytest.approx(0.25, rel=1e-6)

    def test_gradient_flows_only_through_unobserved_bins(self):
        mask = _mask()
        target = torch.randn(1, 2, 16, 16)
        pred = torch.randn(1, 2, 16, 16, requires_grad=True)
        NullSpaceContentLoss()(pred, target, mask=mask).backward()
        grad = pred.grad
        assert grad is not None
        observed = mask.expand_as(grad) > 0
        assert torch.all(grad[observed] == 0.0)
        assert torch.any(grad[~observed] != 0.0)

    def test_missing_mask_raises(self):
        target = torch.randn(1, 2, 8, 8)
        with pytest.raises(ValueError, match="mask"):
            NullSpaceContentLoss()(target.clone(), target)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="matching shapes"):
            NullSpaceContentLoss()(
                torch.randn(1, 2, 8, 8), torch.randn(1, 2, 4, 4), mask=_mask()
            )

    def test_constructor_validation(self):
        with pytest.raises(ValueError, match="input_domain"):
            NullSpaceContentLoss(input_domain="wavelet")
        with pytest.raises(ValueError, match="loss_type"):
            NullSpaceContentLoss(loss_type="huber")

    def test_image_domain_path(self):
        """Image inputs are mapped through fft2c; equality still scores zero
        and an off-support k-space difference scores positive."""
        mask = _mask()
        target = torch.randn(1, 2, 16, 16)
        loss_fn = NullSpaceContentLoss(input_domain="image")
        assert loss_fn(target.clone(), target, mask=mask).item() == pytest.approx(0.0)
        perturbed = target + 0.3 * torch.randn_like(target)
        assert loss_fn(perturbed, target, mask=mask).item() > 0.0

    def test_l1_variant(self):
        mask = _mask()
        target = torch.zeros(1, 2, 16, 16)
        pred = torch.full((1, 2, 16, 16), 0.5)
        loss = NullSpaceContentLoss(loss_type="l1")(pred, target, mask=mask)
        assert loss.item() == pytest.approx(0.5, rel=1e-4)


class TestMaskDtypeTolerance:
    """#2253: the loader delivers a BOOL mask, and `1.0 - m` raises on one.

    Every test above passes a float mask, which is why 38 kspace_filling arms could
    die at iteration 1 with the suite green. Each case here plants a dtype the
    production path actually hands over.
    """

    @pytest.mark.parametrize("dtype", [torch.bool, torch.float32, torch.float64, torch.uint8])
    def test_forward_accepts_mask_dtype(self, dtype):
        mask = _mask().to(dtype)
        target = torch.zeros(1, 2, 16, 16)
        pred = torch.full((1, 2, 16, 16), 0.5)
        loss = NullSpaceContentLoss()(pred, target, mask=mask)
        assert loss.item() == pytest.approx(0.25, rel=1e-6)

    def test_bool_mask_scores_identically_to_float(self):
        """The dtype is a container, not a meaning: same mask, same number."""
        float_mask = _mask()
        target = torch.randn(2, 2, 16, 16)
        pred = target + torch.randn_like(target)
        as_float = NullSpaceContentLoss()(pred, target, mask=float_mask)
        as_bool = NullSpaceContentLoss()(pred, target, mask=float_mask.bool())
        assert as_bool.item() == pytest.approx(as_float.item(), rel=1e-6)

    def test_bool_mask_still_confines_gradient_to_the_null_space(self):
        """The complement must stay a 0/1 WEIGHT under bool, not become `~m` semantics."""
        mask = _mask().bool()
        target = torch.randn(1, 2, 16, 16)
        pred = torch.randn(1, 2, 16, 16, requires_grad=True)
        NullSpaceContentLoss()(pred, target, mask=mask).backward()
        grad = pred.grad
        assert grad is not None
        observed = mask.expand_as(grad)
        assert torch.all(grad[observed] == 0.0)
        assert torch.any(grad[~observed] != 0.0)

    def test_soft_mask_keeps_its_fractional_values(self):
        """Guards the fix against a `~m` regression, which would binarise this."""
        mask = torch.full((1, 1, 16, 16), 0.25)
        target = torch.zeros(1, 2, 16, 16)
        pred = torch.full((1, 2, 16, 16), 1.0)
        # per-bin diff² = 1, weight 0.75 everywhere -> Σ(1·0.75) / Σ0.75 = 1.0
        loss = NullSpaceContentLoss()(pred, target, mask=mask)
        assert loss.item() == pytest.approx(1.0, rel=1e-6)

    def test_complex_mask_takes_the_real_part(self):
        mask = torch.complex(_mask(), torch.zeros_like(_mask()))
        target = torch.zeros(1, 2, 16, 16)
        pred = torch.full((1, 2, 16, 16), 0.5)
        loss = NullSpaceContentLoss()(pred, target, mask=mask)
        assert loss.item() == pytest.approx(0.25, rel=1e-6)


class TestRegistration:
    def test_registered_under_canonical_name_and_alias(self):
        assert is_registered("null_space_content")
        assert is_registered("unobserved_line_loss")
        loss_fn = create_loss("null_space_content")
        assert isinstance(loss_fn, NullSpaceContentLoss)

    def test_composes_with_other_kspace_losses(self):
        from spectramr.models.losses.registry import create_composite_loss

        composite = create_composite_loss(
            {"l1": {"weight": 1.0}, "null_space_content": {"weight": 0.25}}
        )
        # Construction is the contract here (the computer stack calls each
        # component with filtered kwargs at compute time).
        names = [w.name for w in composite.losses]
        assert "null_space_content" in names


class TestNullSpaceAcquiredPlane:
    """Padded (never-acquired) lines are no one's null space (#2160)."""

    @staticmethod
    def _case(n_pad: int):
        torch.manual_seed(0)
        n = 16 + n_pad
        pred = torch.zeros(1, 2, 8, n)
        target = torch.zeros(1, 2, 8, n)
        pred[..., :16] = torch.randn(1, 2, 8, 16)
        target[..., :16] = torch.randn(1, 2, 8, 16)
        mask = torch.zeros(1, 1, 8, n)
        mask[..., 0:16:2] = 1.0
        acquired = torch.zeros(1, 1, 8, n)
        acquired[..., :16] = 1.0
        return pred, target, mask, acquired

    def test_padding_does_not_change_the_value(self):
        loss = NullSpaceContentLoss()
        base = loss(*self._case(0)[:3], acquired=self._case(0)[3])
        for n_pad in (4, 12):
            pred, target, mask, acq = self._case(n_pad)
            assert loss(pred, target, mask=mask, acquired=acq).item() == pytest.approx(
                base.item(), rel=1e-6
            )
            # Planted violation: without the plane the padded bins dilute the mean.
            assert loss(pred, target, mask=mask).item() != pytest.approx(base.item(), rel=1e-3)

    def test_acquired_none_is_the_legacy_value(self):
        pred, target, mask, _ = self._case(6)
        null = (1.0 - mask).expand_as(pred)
        expected = (((pred - target) ** 2) * null).sum() / null.sum()
        got = NullSpaceContentLoss()(pred, target, mask=mask)
        assert got.item() == pytest.approx(expected.item(), rel=1e-6)

    def test_denominator_counts_only_acquired_unkept_bins(self):
        """diff = 1 on every bin: mean over the weighted bins is 1 only if the
        denominator equals the numerator's weight sum, padded bins excluded."""
        pred, target, mask, acq = self._case(5)
        pred = torch.ones_like(pred)
        target = torch.zeros_like(target)
        loss = NullSpaceContentLoss()(pred, target, mask=mask, acquired=acq)
        assert loss.item() == pytest.approx(1.0, rel=1e-6)
        # Padded bins carry weight 0: perturbing them cannot move the value.
        pred[..., 16:] = 7.0
        again = NullSpaceContentLoss()(pred, target, mask=mask, acquired=acq)
        assert again.item() == pytest.approx(1.0, rel=1e-6)
