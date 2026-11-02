"""Unit tests for ConformalGeoMambaULF (2026-06 Tier-1 #16 fix).

The model is the reconstructor for the ``adaptive_sfc_hssc`` strategy, which
computes a learnable, regularised Beltrami space-filling-curve ordering and
hands it to the model via ``sfc_indices=``. ``forward`` uses the supplied
``sfc_indices`` and falls back to a fixed Hilbert scan for standalone calls.
The generator owns NO μ-head of its own — an earlier version instantiated a
second, never-trained ``BeltramiSFCBlock`` whose parameters got no gradient
when the strategy drove the scan (a dead code path, pitfall #16).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.generators.conformal_geomamba_ulf import (  # noqa: E402
    ConformalGeoMambaULF,
)
from spectramr.models.registry import MODEL_REGISTRY  # noqa: E402


def _model() -> ConformalGeoMambaULF:
    return ConformalGeoMambaULF(
        in_channels=1, out_channels=1, base_width=8, scales=1, grid_size=16
    )


def test_registered() -> None:
    assert "conformal_geomamba_ulf" in MODEL_REGISTRY


def test_standalone_forward_falls_back_to_fixed_hilbert() -> None:
    """Called without ``sfc_indices`` the model still runs (fixed Hilbert scan);
    the generator owns no μ-head."""
    m = _model().eval()
    assert not hasattr(m, "sfc")  # no internal Beltrami head
    y = m(torch.randn(2, 1, 32, 32))
    assert y.shape == (2, 1, 32, 32)


def test_supplied_sfc_indices_drive_reconstruction() -> None:
    """Two DIFFERENT orderings must yield DIFFERENT outputs — proof the
    supplied curve is actually used, not swallowed by ``**kwargs``."""
    m = _model().eval()
    x = torch.randn(2, 1, 32, 32)
    n = m.grid_size**2
    idx_identity = torch.arange(n).unsqueeze(0).expand(2, n).contiguous()
    idx_reversed = torch.arange(n - 1, -1, -1).unsqueeze(0).expand(2, n).contiguous()
    with torch.no_grad():
        y_id = m(x, sfc_indices=idx_identity)
        y_rev = m(x, sfc_indices=idx_reversed)
    assert not torch.allclose(y_id, y_rev)


def test_sfc_indices_shape_mismatch_raises() -> None:
    """A wrong-shaped ordering is a config error and must raise (no silent
    fallback to the internal head)."""
    m = _model().eval()
    x = torch.randn(2, 1, 32, 32)
    with pytest.raises(ValueError, match="sfc_indices must have shape"):
        m(x, sfc_indices=torch.arange(10).unsqueeze(0).expand(2, 10))


def test_strategy_kwarg_detection_sees_named_param() -> None:
    """The strategy gates the kwarg on signature inspection; the explicit
    ``sfc_indices`` parameter must be detectable."""
    from spectramr.infrastructure.training.strategies.sfc_conformal_kspace_strategies import (
        _callable_accepts_kwarg,
    )

    assert _callable_accepts_kwarg(_model().forward, "sfc_indices")


def test_soft_perm_without_indices_raises() -> None:
    m = _model().eval()
    n = m.grid_size**2
    with pytest.raises(ValueError, match="requires the sfc_indices"):
        m(torch.randn(2, 1, 32, 32), sfc_soft_perm=torch.eye(n).expand(2, n, n))


def test_soft_perm_shape_mismatch_raises() -> None:
    m = _model().eval()
    n = m.grid_size**2
    idx = torch.arange(n).unsqueeze(0).expand(2, n)
    with pytest.raises(ValueError, match="sfc_soft_perm must have shape"):
        m(torch.randn(2, 1, 32, 32), sfc_indices=idx, sfc_soft_perm=torch.eye(4).expand(2, 4, 4))
