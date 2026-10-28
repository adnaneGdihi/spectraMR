"""End-to-end tests for the CSMNOOperator generator.

Pins the contract that the integration plan signs on:

* The model registers under ``cs_mno_operator`` with mode
  ``reconstruction``.
* Forward and backward pass succeed at the configured spatial shape.
* The model raises (no silent fallback) when called on a different
  spatial shape than it was constructed for.
* The 2-D / 3-D scan-mode cross-check rejects mismatched ranks.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from spectramr.models import generators  # noqa: F401, E402  triggers @register_model
from spectramr.models.generators.cs_mno_operator import CSMNOOperator  # noqa: E402
from spectramr.models.registry import MODEL_REGISTRY  # noqa: E402


def test_model_is_registered_for_reconstruction_mode() -> None:
    assert "cs_mno_operator" in MODEL_REGISTRY
    assert MODEL_REGISTRY["cs_mno_operator"]["mode"] == "reconstruction"
    assert MODEL_REGISTRY["cs_mno_operator"]["class"] is CSMNOOperator


def test_forward_2d_shape_and_autograd() -> None:
    m = CSMNOOperator(
        in_channels=1,
        out_channels=1,
        spatial_shape=(16, 16),
        n_layers=2,
        modes=4,
        dim=8,
        d_state=4,
    )
    x = torch.randn(2, 1, 16, 16, requires_grad=True)
    y = m(x)
    assert y.shape == (2, 1, 16, 16)
    y.sum().backward()
    assert x.grad is not None


def test_forward_resolution_agnostic_same_rank() -> None:
    """F37 regression — the operator must accept a different same-rank
    spatial extent than it was built for (training patch 16×16 vs
    full-res validation 32×32). Previously it hard-raised, which crashed
    validation for every *_mno arm and produced image-less green runs
    (smoke 20260521: c_mno / cs_mno_v0 / hs_mno / me_mno / s_mno).
    """
    m = CSMNOOperator(
        in_channels=1, out_channels=1, spatial_shape=(16, 16),
        n_layers=1, modes=4, dim=8, d_state=4,
    )
    y = m(torch.randn(2, 1, 32, 32))
    assert y.shape == (2, 1, 32, 32)
    # The construction-shape buffers are untouched; the new shape is cached.
    assert (32, 32) in m._dynamic_linearizers
    assert tuple(m.spatial_shape) == (16, 16)
    # Original shape still works after a foreign shape was seen.
    y0 = m(torch.randn(1, 1, 16, 16))
    assert y0.shape == (1, 1, 16, 16)


def test_forward_non_pow2_validation_shape_hilbert() -> None:
    """F37 sequel - a Hilbert-scan operator built on a square power-of-2
    training patch (16x16) must also validate on a NON-power-of-2, NON-square
    full-resolution shape (22x30). The strict ``hilbert_2d`` recompute asserted
    square-pow-2 and re-raised the very ValueError F37 set out to kill, so
    validation produced zero metrics yet the run exited green. The operator now
    routes dynamic shapes through the resolution-agnostic ``hilbert_2d_rect``
    curve (same Hilbert family as the training scan).
    """
    m = CSMNOOperator(
        in_channels=1, out_channels=1, spatial_shape=(16, 16),
        n_layers=1, modes=4, dim=8, d_state=4, scan_mode="hilbert_2d",
    )
    y = m(torch.randn(1, 1, 22, 30))  # would previously raise AssertionError
    assert y.shape == (1, 1, 22, 30)
    assert (22, 30) in m._dynamic_linearizers
    # construction-shape scan is untouched and still strict-Hilbert
    assert m.scan_mode == "hilbert_2d"


def test_forward_rejects_rank_mismatch() -> None:
    """A genuine rank mismatch (2-D operator fed 3-D spatial dims) is a
    config error and must still fail loudly — only same-rank extent
    differences are auto-resolved.
    """
    m = CSMNOOperator(
        in_channels=1, out_channels=1, spatial_shape=(16, 16),
        n_layers=1, modes=4, dim=8, d_state=4,
    )
    with pytest.raises(ValueError, match="rank-2 spatial input"):
        m(torch.randn(1, 1, 8, 8, 8))


def test_scan_mode_dimension_mismatch_raises() -> None:
    """3-D scan mode on 2-D shape (or vice versa) raises immediately."""
    with pytest.raises(ValueError, match="3-D spatial_shape"):
        CSMNOOperator(
            in_channels=1, out_channels=1, spatial_shape=(16, 16),
            n_layers=1, modes=4, dim=8,
            scan_mode="hilbert_3d",
        )
    with pytest.raises(ValueError, match="2-D spatial_shape"):
        CSMNOOperator(
            in_channels=1, out_channels=1, spatial_shape=(8, 8, 8),
            n_layers=1, modes=4, dim=8,
            scan_mode="hilbert_2d",
        )


def test_physical_arc_off_keeps_only_scalar_delta() -> None:
    m = CSMNOOperator(
        in_channels=1, out_channels=1, spatial_shape=(16, 16),
        n_layers=1, modes=4, dim=8, d_state=4,
        scan_mode="hilbert_2d", use_physical_arc=False,
    )
    # The delta_phys buffer is empty (not used) when fallback is on.
    assert m.delta_phys.numel() == 0
    # Forward still runs.
    y = m(torch.randn(1, 1, 16, 16))
    assert y.shape == (1, 1, 16, 16)


def test_forward_squeezes_torchio_trailing_singleton_2d() -> None:
    """TorchIO yields ``[B, C, H, W, D=1]`` for 2-D data; the forward
    must squeeze the trailing singleton instead of failing the
    shape-equality check (regression for cs_mno_darcy smoke crash:
    ``built for (64, 64) but received (64, 64, 1)``).
    """
    m = CSMNOOperator(
        in_channels=1, out_channels=1, spatial_shape=(16, 16),
        n_layers=1, modes=4, dim=8, d_state=4, scan_mode="hilbert_2d",
    )
    y = m(torch.randn(2, 1, 16, 16, 1))
    assert y.shape == (2, 1, 16, 16, 1)


def test_forward_squeezes_pde_trailing_singletons_1d() -> None:
    """1-D PDE benchmarks pass ``[B, C, X, 1, 1]`` through TorchIO;
    both trailing singletons must be squeezed (regression for
    cs_mno_burgers smoke crash: ``built for (128,) but received
    (128, 1, 1)``).
    """
    m = CSMNOOperator(
        in_channels=1, out_channels=1, spatial_shape=(32,),
        n_layers=1, modes=4, dim=8, d_state=4, scan_mode="raster_1d",
    )
    y = m(torch.randn(2, 1, 32, 1, 1))
    assert y.shape == (2, 1, 32, 1, 1)


def test_output_keeps_the_input_rank_for_the_metric() -> None:
    """cs_mno_burgers: collation hands ``[B, 1, X, 1]`` for input and target, and
    PSNR refused a ``[B, 1, X]`` prediction against that target."""
    m = CSMNOOperator(
        in_channels=1, out_channels=1, spatial_shape=(32,),
        n_layers=1, modes=4, dim=8, d_state=4, scan_mode="raster_1d",
    )
    x = torch.randn(2, 1, 32, 1)
    assert m(x).shape == x.shape


def test_forward_does_not_squeeze_non_singleton_trailing_dim() -> None:
    """Trailing non-singleton dims must still raise — the squeeze only
    strips ``size==1`` axes, never collapses real spatial extent.
    """
    m = CSMNOOperator(
        in_channels=1, out_channels=1, spatial_shape=(16, 16),
        n_layers=1, modes=4, dim=8, d_state=4, scan_mode="hilbert_2d",
    )
    with pytest.raises(ValueError, match="rank-2 spatial input"):
        m(torch.randn(2, 1, 16, 16, 4))


_TINY = {"in_channels": 1, "out_channels": 1, "n_layers": 1, "modes": 2, "dim": 8, "d_state": 4}


def test_fixed_resolution_arc_step_is_one_scalar_at_n_to_minus_1_over_d() -> None:
    m = CSMNOOperator(
        spatial_shape=(8, 8), scan_mode="hilbert_2d", arc_step="fixed_resolution", **_TINY
    )
    delta = m.delta_phys
    assert delta.shape == (64,)
    assert torch.allclose(delta, torch.full((64,), 1.0 / 8.0))


def test_physical_arc_step_is_default_and_keeps_zero_first_token() -> None:
    m = CSMNOOperator(spatial_shape=(8, 8), scan_mode="hilbert_2d", **_TINY)
    assert m.arc_step == "physical"
    assert m.delta_phys[0].item() == 0.0


def test_arc_step_rejects_unknown_mode_and_inert_combination() -> None:
    with pytest.raises(ValueError, match="arc_step must be one of"):
        CSMNOOperator(spatial_shape=(8, 8), scan_mode="hilbert_2d", arc_step="learned", **_TINY)
    with pytest.raises(ValueError, match="only applies with use_physical_arc=True"):
        CSMNOOperator(
            spatial_shape=(8, 8),
            scan_mode="hilbert_2d",
            use_physical_arc=False,
            arc_step="fixed_resolution",
            **_TINY,
        )

def test_grad_checkpointing_matches_plain_forward_and_grads() -> None:
    """Checkpointing recomputes layer interiors; outputs and grads must not change."""
    m = CSMNOOperator(
        in_channels=1, out_channels=1, spatial_shape=(16, 16),
        n_layers=2, modes=4, dim=8, d_state=4,
    )
    m.train()
    x = torch.randn(2, 1, 16, 16)

    y_ref = m(x)
    y_ref.sum().backward()
    g_ref = [p.grad.clone() for p in m.parameters() if p.grad is not None]

    m.zero_grad()
    m.set_grad_checkpointing(True)
    y = m(x)
    y.sum().backward()
    g = [p.grad.clone() for p in m.parameters() if p.grad is not None]

    assert torch.allclose(y, y_ref, atol=1e-5)
    assert len(g) == len(g_ref)
    assert all(torch.allclose(a, b, atol=1e-4) for a, b in zip(g, g_ref, strict=True))
