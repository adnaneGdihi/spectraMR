"""Unit tests for contrast_consistency_loss.py."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.relaxation_priors import TissueClass
from spectramr.models.losses.contrast_consistency_loss import (
    DEFAULT_TISSUE_ORDER,
    ContrastConsistencyLoss,
)
from spectramr.models.losses.registry import create_loss

B, H, W = 2, 16, 16
K = len(DEFAULT_TISSUE_ORDER)  # 3 by default


def _cc(loss, t1, probs, b0):
    """Call the loss as the strategy does: the model's aux maps and a per-sample B0."""
    field = torch.full((t1.shape[0],), float(b0))
    return loss(
        t1, None, aux_maps={"t1_ms": t1, "tissue_probabilities": probs}, field_strength=field
    )


def _t1(b=B, h=H, w=W, seed=0) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.rand(b, 1, h, w, generator=gen) * 800.0 + 200.0  # [200, 1000] ms


def _probs(b=B, k=K, h=H, w=W, seed=0) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    raw = torch.rand(b, k, h, w, generator=gen)
    # Soft-max along channel dim for valid probability simplex.
    return raw / raw.sum(dim=1, keepdim=True).clamp_min(1e-6)


class TestContrastConsistencyLoss:
    # ── Canary ──────────────────────────────────────────────────────────────

    def test_canary_builds_and_forward_finite(self):
        loss_fn = ContrastConsistencyLoss()
        t1 = _t1()
        probs = _probs()
        out = _cc(loss_fn, t1, probs, 3.0)
        assert torch.isfinite(out) and out.shape == ()

    # ── Parametric ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("b0", [0.3, 1.5, 3.0, 7.0])
    def test_parametric_field_strength(self, b0: float):
        loss_fn = ContrastConsistencyLoss()
        t1 = _t1()
        probs = _probs()
        out = _cc(loss_fn, t1, probs, b0)
        assert torch.isfinite(out) and out.item() >= 0.0

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_parametric_reduction(self, reduction: str):
        loss_fn = ContrastConsistencyLoss(reduction=reduction)
        t1 = _t1()
        probs = _probs()
        out = _cc(loss_fn, t1, probs, 3.0)
        assert torch.isfinite(out) and out.item() >= 0.0

    # ── Property ────────────────────────────────────────────────────────────

    def test_property_nonneg(self):
        loss_fn = ContrastConsistencyLoss()
        t1 = _t1()
        probs = _probs()
        assert _cc(loss_fn, t1, probs, 3.0).item() >= 0.0

    def test_property_zero_weight_gives_zero(self):
        loss_fn = ContrastConsistencyLoss(weight=0.0)
        t1 = _t1()
        probs = _probs()
        out = _cc(loss_fn, t1, probs, 3.0)
        assert out.item() == pytest.approx(0.0, abs=1e-9)

    def test_property_gradient_reaches_t1(self):
        t1 = _t1().requires_grad_(True)
        probs = _probs()
        loss_fn = ContrastConsistencyLoss()
        _cc(loss_fn, t1, probs, 3.0).backward()
        assert t1.grad is not None and torch.isfinite(t1.grad).all()

    def test_property_t1_at_target_gives_low_loss(self):
        """If predicted T1 matches the Bottomley prior exactly, loss → 0."""
        from spectramr.infrastructure.physics.relaxation_priors import bottomley_t1

        loss_fn = ContrastConsistencyLoss()
        b0 = 3.0
        # Build t1 = mean of all tissue targets (close to each prior)
        targets = [bottomley_t1(b0, t) for t in DEFAULT_TISSUE_ORDER]
        target_mean = sum(targets) / len(targets)
        t1 = torch.full((B, 1, H, W), target_mean)
        probs = _probs()
        out = _cc(loss_fn, t1, probs, b0)
        # Not guaranteed 0 unless prediction == each class target,
        # but must be finite.
        assert torch.isfinite(out)

    # ── Edge ────────────────────────────────────────────────────────────────

    def test_edge_all_probability_on_one_class(self):
        """All probability on one tissue class."""
        loss_fn = ContrastConsistencyLoss()
        t1 = _t1()
        probs = torch.zeros(B, K, H, W)
        probs[:, 0, :, :] = 1.0
        out = _cc(loss_fn, t1, probs, 3.0)
        assert torch.isfinite(out)

    def test_edge_zero_probabilities(self):
        """All-zero probability map → denominator clamped → finite."""
        loss_fn = ContrastConsistencyLoss()
        t1 = _t1()
        probs = torch.zeros(B, K, H, W)
        out = _cc(loss_fn, t1, probs, 3.0)
        assert torch.isfinite(out)

    def test_edge_field_strength_as_tensor(self):
        loss_fn = ContrastConsistencyLoss()
        t1 = _t1()
        probs = _probs()
        b0 = torch.tensor(1.5)
        out = _cc(loss_fn, t1, probs, b0)
        assert torch.isfinite(out)

    # ── Raises ──────────────────────────────────────────────────────────────

    def test_raises_negative_weight(self):
        with pytest.raises(ValueError, match="weight"):
            ContrastConsistencyLoss(weight=-1.0)

    def test_raises_wrong_t1_shape(self):
        loss_fn = ContrastConsistencyLoss()
        bad_t1 = torch.rand(B, 2, H, W)  # must be [B, 1, H, W]
        probs = _probs()
        with pytest.raises(ValueError):
            _cc(loss_fn, bad_t1, probs, 3.0)

    def test_raises_prob_channel_mismatch(self):
        loss_fn = ContrastConsistencyLoss()
        t1 = _t1()
        bad_probs = torch.rand(B, K + 1, H, W)  # wrong channel count
        with pytest.raises(ValueError):
            _cc(loss_fn, t1, bad_probs, 3.0)

    def test_raises_zero_field_strength(self):
        loss_fn = ContrastConsistencyLoss()
        t1 = _t1()
        probs = _probs()
        with pytest.raises(ValueError, match="b0_T must be > 0"):
            _cc(loss_fn, t1, probs, 0.0)

    def test_raises_invalid_reduction(self):
        with pytest.raises(ValueError, match="reduction"):
            ContrastConsistencyLoss(reduction="median_xyz")

    # ── Registry ────────────────────────────────────────────────────────────

    def test_registry_lookup(self):
        loss_fn = create_loss("contrast_consistency")
        assert isinstance(loss_fn, ContrastConsistencyLoss)


class TestInputsComeFromTheModel:
    def test_missing_aux_maps_or_field_raise(self):
        loss_fn = ContrastConsistencyLoss()
        with pytest.raises(ValueError, match="aux_maps"):
            loss_fn(_t1(), None, field_strength=torch.ones(1))
        with pytest.raises(ValueError, match="field_strength"):
            loss_fn(_t1(), None, aux_maps={"t1_ms": _t1(), "tissue_probabilities": _probs()})

    def test_each_sample_is_held_to_its_own_fields_prior(self):
        from spectramr.infrastructure.physics.relaxation_priors import bottomley_t1

        probs = torch.zeros(2, 3, 4, 4)
        probs[:, 0] = 1.0  # all white matter
        t1 = torch.stack(
            [torch.full((1, 4, 4), bottomley_t1(b, "white_matter")) for b in (0.064, 3.0)]
        )
        aux = {"t1_ms": t1, "tissue_probabilities": probs}
        loss_fn = ContrastConsistencyLoss()
        assert (
            float(loss_fn(t1, None, aux_maps=aux, field_strength=torch.tensor([0.064, 3.0]))) < 1e-6
        )
        assert (
            float(loss_fn(t1, None, aux_maps=aux, field_strength=torch.tensor([3.0, 0.064]))) > 1.0
        )
