"""Tests for ``ContrastiveLoss``.

Targets ``spectramr.models.losses.contrastive_losses``. InfoNCE-style
contrastive loss with temperature scaling and supervised-contrastive
masking. ``CAContrastiveLoss`` (anatomy-disentanglement variant) is
covered in the integration tier — it requires multi-view fixtures.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.contrastive_losses import ContrastiveLoss

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_temperature() -> None:
    """SimCLR-style default temperature is 0.07."""
    loss = ContrastiveLoss()
    assert loss.temperature == 0.07


def test_explicit_temperature() -> None:
    """Custom temperature persisted."""
    loss = ContrastiveLoss(temperature=0.5)
    assert loss.temperature == 0.5


# ---------------------------------------------------------------------------
# Fail-loud: missing labels and mask
# ---------------------------------------------------------------------------


def test_no_labels_no_mask_raises() -> None:
    """Without labels or mask, the loss has no supervisory signal — fails loud."""
    loss = ContrastiveLoss()
    features = torch.randn(8, 16)
    with pytest.raises(ValueError, match="Must provide labels or mask"):
        loss(features)


# ---------------------------------------------------------------------------
# Forward — labels-supplied path
# ---------------------------------------------------------------------------


def test_forward_with_labels_returns_scalar() -> None:
    """Loss returns a scalar tensor when labels are supplied."""
    loss = ContrastiveLoss()
    features = torch.randn(8, 16)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])  # 4 classes × 2 views
    out = loss(features, labels=labels)
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_forward_with_explicit_mask_returns_scalar() -> None:
    """Loss accepts a precomputed boolean mask."""
    loss = ContrastiveLoss()
    features = torch.randn(4, 8)
    mask = torch.tensor([
        [True, True, False, False],
        [True, True, False, False],
        [False, False, True, True],
        [False, False, True, True],
    ])
    out = loss(features, mask=mask)
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_loss_lower_for_correct_pairing() -> None:
    """Aligned features (same-class similar) yield lower loss than misaligned."""
    torch.manual_seed(0)
    loss = ContrastiveLoss(temperature=0.5)
    # Two clusters in 8-D feature space.
    cluster_a = torch.randn(4, 8) + 5.0  # large mean
    cluster_b = torch.randn(4, 8) - 5.0
    features = torch.cat([cluster_a, cluster_b], dim=0)
    labels_correct = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    labels_wrong = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    loss_correct = loss(features, labels=labels_correct).item()
    loss_wrong = loss(features, labels=labels_wrong).item()
    assert loss_correct < loss_wrong


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_features() -> None:
    """Backward through the loss reaches the input features."""
    loss = ContrastiveLoss()
    features = torch.randn(4, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    loss(features, labels=labels).backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


# ---------------------------------------------------------------------------
# Sanity-shape matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size, feature_dim",
    [(4, 8), (16, 32), (8, 128), (32, 64)],
    ids=lambda p: f"B{p}",
)
def test_loss_shape_matrix(batch_size: int, feature_dim: int) -> None:
    """Loss is a scalar regardless of feature dimensionality."""
    torch.manual_seed(0)
    loss = ContrastiveLoss()
    features = torch.randn(batch_size, feature_dim)
    labels = torch.arange(batch_size) // 2  # pairs of same-class
    out = loss(features, labels=labels)
    assert out.dim() == 0
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_contrastive_registered() -> None:
    """``contrastive`` is in the loss registry."""
    from spectramr.models.losses.registry import list_available

    assert "contrastive" in list_available()


# ---------------------------------------------------------------------------
# Numerical form: the denominator is a logsumexp over every non-self sample
# ---------------------------------------------------------------------------


def _supcon_reference(features: torch.Tensor, labels: torch.Tensor, t: float) -> torch.Tensor:
    """Supervised-contrastive loss written directly from its definition, in float64."""
    f = torch.nn.functional.normalize(features.double(), dim=1)
    sim = f @ f.T / t
    eye = torch.eye(len(labels), dtype=torch.bool)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim.masked_fill(eye, float("-inf")), dim=1, keepdim=True)
    per_anchor = (log_prob * pos).sum(1) / pos.sum(1).clamp_min(1)
    return -per_anchor.mean()


@pytest.mark.parametrize("temperature", [0.5, 0.07, 0.01, 0.005])
def test_value_matches_the_definition_at_any_temperature(temperature: float) -> None:
    """``exp(1 / tau)`` overflows float32 below tau ~ 0.011, where the old
    ``log(sum(exp(.)) + 1e-9)`` returned NaN."""
    torch.manual_seed(0)
    features = torch.randn(8, 16)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    got = ContrastiveLoss(temperature=temperature)(features, labels=labels)
    assert torch.isfinite(got)
    want = _supcon_reference(features, labels, temperature).float()
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_a_single_sample_batch_is_finite_in_value_and_gradient() -> None:
    features = torch.randn(1, 4, requires_grad=True)
    loss = ContrastiveLoss()(features, labels=torch.tensor([0]))
    loss.backward()
    assert torch.isfinite(loss)
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_the_jaxtyping_annotations_resolve() -> None:
    """``Int[Tensor, B]`` named an undefined ``B``: any runtime type-hint
    resolution (a jaxtyped/beartype decorator, ``get_type_hints``) raised NameError."""
    import typing

    from spectramr.models.losses.contrastive_losses import CAContrastiveLoss

    assert "labels" in typing.get_type_hints(ContrastiveLoss.forward)
    assert "subject_ids" in typing.get_type_hints(CAContrastiveLoss.forward)
