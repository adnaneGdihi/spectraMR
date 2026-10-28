"""Tests for the four sibling MNO variants from spec §3.

The five MNO architectures decompose into 4 flag combinations on
``CSMNOOperator`` (HS, C, S, ME-as-wrapper) plus one geodesic-curve
variant that uses the ``MetricSFCLinearizer`` from the GeoMamba-ULF
infrastructure (G-MNO).

This file pins:

1. Each of the five model_type names registers correctly.
2. Each variant produces the correct output shape on a small input.
3. ME-MNO's group averaging is **exact** in the no-op case (identity
   group element only) and produces the symmetric mean otherwise.
4. G-MNO raises on non-3-D shapes (no silent fallback).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from spectramr.models import generators  # noqa: F401, E402
from spectramr.models.generators.mno_family import (  # noqa: E402
    CMNOOperator,
    GMNOOperator,
    HSMNOOperator,
    MEMNOOperator,
    SMNOOperator,
)
from spectramr.models.registry import MODEL_REGISTRY  # noqa: E402

# ── Registry ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    ["hs_mno_operator", "c_mno_operator", "s_mno_operator",
     "me_mno_operator", "g_mno_operator"],
)
def test_variant_registered_for_reconstruction(name: str) -> None:
    assert name in MODEL_REGISTRY
    assert MODEL_REGISTRY[name]["mode"] == "reconstruction"


# ── HS / C / S / ME forward shape ─────────────────────────────────


_KW2D = dict(
    in_channels=1, out_channels=1, spatial_shape=(16, 16),
    n_layers=1, modes=4, dim=8, d_state=4,
)


@pytest.mark.parametrize("cls", [HSMNOOperator, CMNOOperator, SMNOOperator])
def test_2d_variants_forward_shape(cls: type) -> None:
    m = cls(**_KW2D)
    y = m(torch.randn(2, 1, 16, 16))
    assert y.shape == (2, 1, 16, 16)


def test_hs_mno_has_no_spectral_branch() -> None:
    """HS-MNO disables spectral; layers' .spectral should be Identity."""
    m = HSMNOOperator(**_KW2D)
    for layer in m.layers:
        assert isinstance(layer.spectral, torch.nn.Identity)


def test_c_mno_has_no_spectral_branch_but_has_physical_arc() -> None:
    m = CMNOOperator(**_KW2D)
    for layer in m.layers:
        assert isinstance(layer.spectral, torch.nn.Identity)
    assert m.delta_phys.numel() > 0  # physical arc enabled


def test_s_mno_keeps_spectral_but_no_physical_arc() -> None:
    m = SMNOOperator(**_KW2D)
    for layer in m.layers:
        assert not isinstance(layer.spectral, torch.nn.Identity)
    # When use_physical_arc is False, the operator stores an empty buffer.
    assert m.delta_phys.numel() == 0


# ── ME-MNO group averaging ────────────────────────────────────────


def test_me_mno_identity_group_equals_backbone() -> None:
    """With symmetry_group='identity' the wrapper must be a no-op."""
    torch.manual_seed(0)
    m = MEMNOOperator(symmetry_group="identity", **_KW2D)
    x = torch.randn(1, 1, 16, 16)
    y_me = m(x)
    y_bb = m.backbone(x)
    assert torch.allclose(y_me, y_bb, atol=1e-6)


def test_me_mno_d4_changes_output() -> None:
    """D4 averaging is not the identity for non-equivariant inputs."""
    torch.manual_seed(0)
    m_d4 = MEMNOOperator(symmetry_group="d4", **_KW2D)
    m_id = MEMNOOperator(symmetry_group="identity", **_KW2D)
    # Same backbone weights so the only difference is the averaging.
    m_id.backbone.load_state_dict(m_d4.backbone.state_dict())
    x = torch.randn(1, 1, 16, 16)
    assert not torch.allclose(m_d4(x), m_id(x), atol=1e-4)


def test_me_mno_rejects_unknown_group() -> None:
    with pytest.raises(ValueError, match="symmetry_group"):
        MEMNOOperator(symmetry_group="o(3)", **_KW2D)


# ── G-MNO (3-D, uses MetricSFCLinearizer with scipy MST) ─────────


def test_g_mno_forward_shape() -> None:
    pytest.importorskip("scipy")
    m = GMNOOperator(
        in_channels=1, out_channels=1, spatial_shape=(8, 8, 8),
        n_layers=1, modes=2, dim=4, d_state=4,
    )
    y = m(torch.randn(1, 1, 8, 8, 8))
    assert y.shape == (1, 1, 8, 8, 8)


def test_g_mno_forward_resolution_agnostic_same_rank() -> None:
    """F37 regression — a 3-D GMNO must accept a different 3-D extent than
    its construction shape (training patch vs full-res validation), instead
    of hard-raising and crashing validation.
    """
    pytest.importorskip("scipy")
    m = GMNOOperator(
        in_channels=1, out_channels=1, spatial_shape=(8, 8, 8),
        n_layers=1, modes=2, dim=4, d_state=4,
    )
    y = m(torch.randn(1, 1, 6, 6, 6))
    assert y.shape == (1, 1, 6, 6, 6)
    assert (6, 6, 6) in m._dynamic_linearizers


def test_g_mno_forward_rejects_rank_mismatch() -> None:
    """F37 — feeding 2-D spatial dims to a 3-D operator is a data/config
    error (this was the actual g_mno smoke crash: built for (64,64,64),
    received (221,221)). It must fail loudly, not silently green-pass.
    """
    pytest.importorskip("scipy")
    m = GMNOOperator(
        in_channels=1, out_channels=1, spatial_shape=(8, 8, 8),
        n_layers=1, modes=2, dim=4, d_state=4,
    )
    with pytest.raises(ValueError, match="3-D only"):
        m(torch.randn(1, 1, 16, 16))


def test_g_mno_rejects_2d_shape() -> None:
    with pytest.raises(ValueError, match="3-D"):
        GMNOOperator(
            in_channels=1, out_channels=1, spatial_shape=(16, 16),
            n_layers=1,
        )


# ── Oh group: full 3-D 48-element cubic point group ────────────────


def test_oh_group_has_48_elements() -> None:
    from spectramr.models.generators.mno_family import _OH_ELEMENTS
    assert len(_OH_ELEMENTS) == 48


def test_apply_oh_invert_oh_are_inverses() -> None:
    """Every Oh element must round-trip exactly under apply/invert."""
    from spectramr.models.generators.mno_family import (
        _OH_ELEMENTS,
        _apply_oh,
        _invert_oh,
    )
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 5, 6)
    for perm, signs in _OH_ELEMENTS:
        round_trip = _invert_oh(_apply_oh(x, perm, signs), perm, signs)
        assert torch.allclose(round_trip, x, atol=1e-6), (
            f"round-trip failed for perm={perm}, signs={signs}"
        )


def test_me_mno_oh_3d_forward() -> None:
    """ME-MNO with Oh group runs end-to-end on a 3-D backbone."""
    m = MEMNOOperator(
        symmetry_group="oh",
        in_channels=1, out_channels=1, spatial_shape=(8, 8, 8),
        n_layers=1, modes=2, dim=4, d_state=4,
        scan_mode="hilbert_3d",
    )
    y = m(torch.randn(1, 1, 8, 8, 8))
    assert y.shape == (1, 1, 8, 8, 8)


def test_me_mno_oh_requires_3d_backbone() -> None:
    with pytest.raises(ValueError, match="oh"):
        MEMNOOperator(
            symmetry_group="oh",
            in_channels=1, out_channels=1, spatial_shape=(16, 16),
            n_layers=1, modes=4, dim=8, d_state=4,
        )


def test_me_mno_d4_requires_2d_backbone() -> None:
    with pytest.raises(ValueError, match="d4"):
        MEMNOOperator(
            symmetry_group="d4",
            in_channels=1, out_channels=1, spatial_shape=(8, 8, 8),
            n_layers=1, modes=2, dim=4, d_state=4,
            scan_mode="hilbert_3d",
        )


def test_me_mno_oh_is_exactly_equivariant() -> None:
    """The Oh-averaged operator must satisfy g · f̂(x) = f̂(g · x) for every
    g in the cubic group. This is the formal claim of Proposition 3
    from spec §3.3 — the test exercises a sample of Oh elements rather
    than the full 48 to keep runtime reasonable.
    """
    from spectramr.models.generators.mno_family import _OH_ELEMENTS, _apply_oh
    m = MEMNOOperator(
        symmetry_group="oh",
        in_channels=1, out_channels=1, spatial_shape=(4, 4, 4),
        n_layers=1, modes=2, dim=4, d_state=4,
        scan_mode="hilbert_3d",
    )
    m.eval()
    x = torch.randn(1, 1, 4, 4, 4)
    y = m(x)
    # Sample the first 5 non-identity elements.
    for perm, signs in _OH_ELEMENTS[1:6]:
        y_g = m(_apply_oh(x, perm, signs))
        g_y = _apply_oh(y, perm, signs)
        assert torch.allclose(y_g, g_y, atol=1e-4), (
            f"equivariance failed for perm={perm}, signs={signs}: "
            f"max diff {(y_g - g_y).abs().max().item():.4e}"
        )


def test_me_mno_forwards_grad_checkpointing_to_backbone() -> None:
    m = MEMNOOperator(
        symmetry_group="identity", in_channels=1, out_channels=1,
        spatial_shape=(16, 16), n_layers=1, modes=4, dim=8, d_state=4,
    )
    m.set_grad_checkpointing(True)
    assert m.backbone.grad_checkpointing is True
