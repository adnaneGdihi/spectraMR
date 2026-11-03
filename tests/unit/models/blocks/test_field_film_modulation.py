"""Tests for ``FieldFiLMBlock`` and ``FieldFiLMHypernet``.

Targets ``spectramr.models.blocks.field_film_modulation`` — the FiLM
modulation primitive of XField-FM (idea 2 of
``TODO/integration_plan_ulf_cheap_fast_mri.md``).

Categories:

- Construction: invalid ``num_channels`` / ``sequence_dim`` rejected
- Identity initialisation: at zero step the block is the identity
  (γ = 1, β = 0)
- Output shape: ``(γ, β)`` are ``[B, num_channels]``
- ``apply_to``: matches ``γ ⊙ h + β`` exactly
- Differentiability: gradients flow back into the host activation
- Validation: shape mismatches, dim mismatches → ``ValueError``
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.field_film_modulation import FieldFiLMBlock


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_zero_num_channels_rejected() -> None:
    """``num_channels < 1`` raises."""
    with pytest.raises(ValueError, match="num_channels"):
        FieldFiLMBlock(num_channels=0, sequence_dim=4)


def test_negative_sequence_dim_rejected() -> None:
    """``sequence_dim < 0`` raises."""
    with pytest.raises(ValueError, match="sequence_dim"):
        FieldFiLMBlock(num_channels=4, sequence_dim=-1)


def test_field_only_conditioning_takes_an_empty_sequence_vector() -> None:
    """``sequence_dim=0``: B0 alone conditions; the caller passes ``[B, 0]``."""
    block = FieldFiLMBlock(num_channels=4, sequence_dim=0)
    gamma, beta = block(torch.tensor([0.064, 3.0]), torch.zeros(2, 0))
    assert gamma.shape == beta.shape == (2, 4)


# ---------------------------------------------------------------------------
# Identity initialisation
# ---------------------------------------------------------------------------


def test_starts_as_identity() -> None:
    """At init: ``γ = 1`` and ``β = 0`` regardless of input."""
    block = FieldFiLMBlock(num_channels=8, sequence_dim=4)
    field = torch.tensor([0.064, 3.0])
    seq = torch.randn(2, 4)
    gamma, beta = block(field, seq)
    assert torch.allclose(gamma, torch.ones_like(gamma), atol=1e-6)
    assert torch.allclose(beta, torch.zeros_like(beta), atol=1e-6)


def test_apply_to_starts_identity() -> None:
    """``apply_to`` returns the input unchanged at init."""
    block = FieldFiLMBlock(num_channels=4, sequence_dim=2)
    activation = torch.randn(2, 4, 8, 8)
    field = torch.tensor([0.064, 3.0])
    seq = torch.randn(2, 2)
    out = block.apply_to(activation, field, seq)
    assert torch.allclose(out, activation, atol=1e-6)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


def test_gamma_beta_shape() -> None:
    """Output ``(γ, β)`` have shape ``[B, num_channels]``."""
    block = FieldFiLMBlock(num_channels=8, sequence_dim=3)
    gamma, beta = block(
        torch.tensor([0.064, 0.3, 1.5, 3.0]), torch.randn(4, 3)
    )
    assert gamma.shape == (4, 8)
    assert beta.shape == (4, 8)


def test_field_strength_2d_input_accepted() -> None:
    """``[B, 1]``-shaped field strength input works as well as ``[B]``."""
    block = FieldFiLMBlock(num_channels=4, sequence_dim=2)
    field_2d = torch.tensor([[0.064], [3.0]])
    seq = torch.randn(2, 2)
    gamma, beta = block(field_2d, seq)
    assert gamma.shape == (2, 4)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_field_strength_3d_rejected() -> None:
    """3-D field strength rejected."""
    block = FieldFiLMBlock(num_channels=4, sequence_dim=2)
    with pytest.raises(ValueError, match="field_strength_T"):
        block(torch.zeros(2, 1, 1), torch.zeros(2, 2))


def test_sequence_dim_mismatch_rejected() -> None:
    """``sequence_features.shape[-1] != sequence_dim`` raises."""
    block = FieldFiLMBlock(num_channels=4, sequence_dim=2)
    with pytest.raises(ValueError, match="sequence_dim"):
        block(torch.tensor([0.064]), torch.randn(1, 7))


def test_batch_size_mismatch_rejected() -> None:
    """Mismatched batch sizes raise."""
    block = FieldFiLMBlock(num_channels=4, sequence_dim=2)
    with pytest.raises(ValueError, match="Batch-size"):
        block(torch.tensor([0.064, 3.0]), torch.randn(3, 2))


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


def test_gradient_flows_to_host_activation() -> None:
    """Gradient flows through ``apply_to`` back to the host activation."""
    block = FieldFiLMBlock(num_channels=4, sequence_dim=2)
    activation = torch.randn(1, 4, 8, 8, requires_grad=True)
    out = block.apply_to(activation, torch.tensor([0.064]), torch.randn(1, 2))
    out.sum().backward()
    assert activation.grad is not None
    assert torch.isfinite(activation.grad).all()


def test_gradient_flows_to_block_parameters() -> None:
    """Gradient flows back to MLP weights (so the block is trainable)."""
    block = FieldFiLMBlock(num_channels=4, sequence_dim=2)
    field = torch.tensor([0.064], requires_grad=False)
    seq = torch.randn(1, 2)
    gamma, beta = block(field, seq)
    (gamma.sum() + beta.sum()).backward()
    # Final layer was zero-init; gradient must be non-zero post-backward.
    last_layer_grad = block.mlp[-1].weight.grad
    assert last_layer_grad is not None
    # Some entries should be non-zero.
    assert last_layer_grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# FieldFiLMHypernet — Hypernet adapter (sanity-shape + canary)
# ---------------------------------------------------------------------------


from spectramr.models.blocks.field_film_modulation import FieldFiLMHypernet  # noqa: E402


class TestFieldFiLMHypernet:
    """Canary and shape tests for the Hypernet-API adapter."""

    def test_canary_construction(self) -> None:
        """Default construction completes without error."""
        h = FieldFiLMHypernet(context_dim=3, param_shapes={"gamma": (8,), "beta": (8,)})
        assert h is not None

    def test_canary_forward_returns_gamma_beta(self) -> None:
        """Forward returns dict with 'gamma' and 'beta' of shape (B, C)."""
        h = FieldFiLMHypernet(context_dim=3, param_shapes={"gamma": (8,), "beta": (8,)})
        h.eval()
        ctx = torch.randn(4, 3)  # (B, context_dim)
        out = h(ctx)
        assert set(out) == {"gamma", "beta"}
        assert out["gamma"].shape == (4, 8)
        assert out["beta"].shape == (4, 8)
        assert torch.isfinite(out["gamma"]).all()
        assert torch.isfinite(out["beta"]).all()

    def test_starts_as_identity_gamma_one_beta_zero(self) -> None:
        """At init: gamma ≈ 1, beta ≈ 0 (mirrors FieldFiLMBlock init)."""
        h = FieldFiLMHypernet(context_dim=3, param_shapes={"gamma": (4,), "beta": (4,)})
        h.eval()
        ctx = torch.randn(2, 3)
        out = h(ctx)
        assert torch.allclose(out["gamma"], torch.ones_like(out["gamma"]), atol=1e-6)
        assert torch.allclose(out["beta"], torch.zeros_like(out["beta"]), atol=1e-6)

    def test_registered_under_field_film(self) -> None:
        """``get_hypernet('field_film', ...)`` returns a FieldFiLMHypernet."""
        from spectramr.models.blocks.hypernet_api import get_hypernet

        h = get_hypernet(
            "field_film",
            context_dim=3,
            param_shapes={"gamma": (4,), "beta": (4,)},
        )
        assert isinstance(h, FieldFiLMHypernet)

    def test_wrong_context_dim_raises(self) -> None:
        """Context tensor with wrong width raises ``ValueError``."""
        h = FieldFiLMHypernet(context_dim=3, param_shapes={"gamma": (4,), "beta": (4,)})
        ctx = torch.randn(1, 5)  # wrong width
        with pytest.raises(ValueError):
            h(ctx)

    def test_context_dim_less_than_2_raises(self) -> None:
        """context_dim < 2 raises ``ValueError`` at construction."""
        with pytest.raises(ValueError, match="context_dim"):
            FieldFiLMHypernet(
                context_dim=1,
                param_shapes={"gamma": (4,), "beta": (4,)},
            )

    def test_non_gamma_beta_keys_raise(self) -> None:
        """param_shapes with wrong keys raises ``ValueError``."""
        with pytest.raises(ValueError):
            FieldFiLMHypernet(
                context_dim=3,
                param_shapes={"scale": (4,), "bias": (4,)},
            )
