"""Regression tests for the ESPIRiT->RSS fallback contract in ``synthesize_pseudo_gt``.

``synthesize_pseudo_gt`` documents that the coil maps are the canonical 7-step
ESPIRiT (Uecker 2014). When ESPIRiT estimation raises, the function falls back
to RSS coil maps — a materially different ``x_gt``. Previously this method swap
was logged only at DEBUG ("ESPIRiT unavailable, using RSS"), making a silent
ground-truth corruption invisible to every downstream sim2rank/BT comparison
(CLAUDE.md pitfall #9/#10). The fix promotes the log to WARNING and names RSS.

These tests assert the fallback (a) still returns finite ``x_gt`` of the right
shape and (b) emits a WARNING-level record naming RSS — not a DEBUG no-signal.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import spectramr.data.datasets.kspace_volume_stack as kio
import spectramr.data.datasets.pseudo_gt_reference as pgt
import spectramr.infrastructure.physics.coil_sensitivity as csm_mod


@pytest.fixture()
def fake_kspace(monkeypatch):
    """Patch the H5 loader to return a deterministic (S, C, H, W) k-space."""
    torch.manual_seed(0)
    ks = torch.complex(torch.randn(3, 2, 8, 8), torch.randn(3, 2, 8, 8))

    def _load(_path: Path, slice_index: int | None = None) -> torch.Tensor:
        return ks if slice_index is None else ks[slice_index]

    monkeypatch.setattr(kio, "kspace_shape_from_h5", lambda _p: tuple(ks.shape))
    monkeypatch.setattr(kio, "load_kspace_from_h5", _load)
    return ks


def test_espirit_failure_warns_not_silent(fake_kspace, monkeypatch, caplog) -> None:
    """A non-OOM ESPIRiT failure must be LOUD (naming RSS), never a silent swap.

    Regression: an ESPIRiT->RSS method swap on a *ground-truth synthesizer*
    corrupts every downstream comparison, so it must be loud (CLAUDE.md #9/#10).
    Promoted DEBUG -> WARNING, then WARNING -> ERROR once #521 established that
    the consequence is a cohort carrying two incompatible ground truths.

    A CUDA OOM no longer reaches this path at all — it retries ESPIRiT on CPU.
    See ``tests/unit/infrastructure/physics/test_m4raw_pseudo_gt.py``.
    """

    def _boom(kspace, num_coils, **_):
        raise RuntimeError("linalg: cusolver unavailable")

    monkeypatch.setattr(csm_mod, "estimate_csm_espirit", _boom)

    with caplog.at_level("ERROR", logger="spectramr"):
        coil_images, smaps, x_gt_mag, _p99 = pgt.synthesize_pseudo_gt(
            [Path("r1.h5"), Path("r2.h5")]
        )

    # The fallback still produces a valid pseudo-GT.
    assert smaps.shape == (1, 2, 8, 8)
    assert coil_images.shape == (1, 2, 8, 8) and coil_images.is_complex()
    assert x_gt_mag.shape == (1, 1, 8, 8)
    assert torch.isfinite(x_gt_mag).all()

    # The method swap must be visible at ERROR level and name RSS.
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert any(
        "RSS" in r.getMessage() for r in errors
    ), "ESPIRiT->RSS fallback must be reported at ERROR, not swapped silently"


def test_espirit_success_emits_no_rss_warning(fake_kspace, monkeypatch, caplog) -> None:
    """When ESPIRiT succeeds there must be NO RSS-fallback warning."""

    def _ok(kspace, num_coils, **_):
        return torch.ones_like(kspace)

    monkeypatch.setattr(csm_mod, "estimate_csm_espirit", _ok)

    with caplog.at_level("WARNING", logger="spectramr"):
        _ci, smaps, x_gt_mag, _p99 = pgt.synthesize_pseudo_gt(
            [Path("r1.h5"), Path("r2.h5")]
        )

    assert smaps.shape == (1, 2, 8, 8)
    assert torch.isfinite(x_gt_mag).all()
    assert not any(
        "RSS" in r.getMessage() for r in caplog.records if r.levelname == "WARNING"
    ), "ESPIRiT success path must not emit the RSS-fallback warning"
