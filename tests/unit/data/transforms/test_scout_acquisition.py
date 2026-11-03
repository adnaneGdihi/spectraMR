"""Tests for ``ScoutAcquisitionTransform``.

Targets ``spectramr.data.transforms.scout_acquisition`` — SCAS / idea 5
§5.2 of ``TODO/integration_plan_ulf_cheap_fast_mri.md``.

Plan acceptance criterion (§5.6):

- *"Scout has the configured low-resolution truncation"* — covered
  by ``test_scout_zeroes_outside_window``.

Other contract tests:

- Construction: invalid domain / non-positive scout dims rejected
- ``scout_resolution`` larger than input → ``ValueError``
- ``domain="kspace"`` zeroes outside the centre, preserves shape
- ``domain="image"`` runs FFT/iFFT and produces a low-pass image
- Missing keys are silently skipped
- Output shape matches input shape
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torchio as tio

from spectramr.data.transforms.scout_acquisition import ScoutAcquisitionTransform


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_invalid_domain_rejected() -> None:
    """Only ``kspace`` and ``image`` accepted."""
    with pytest.raises(ValueError, match="domain"):
        ScoutAcquisitionTransform(domain="foo")


def test_zero_scout_dim_rejected() -> None:
    """Scout dims must be positive."""
    with pytest.raises(ValueError, match="scout_resolution"):
        ScoutAcquisitionTransform(scout_resolution=(0, 32))


# ---------------------------------------------------------------------------
# Subject helpers
# ---------------------------------------------------------------------------


def _make_subject(data: torch.Tensor) -> tio.Subject:
    """A ``(1, H, W)`` plane as TorchIO's ``(C, W, H, D)`` with one slice."""
    return tio.Subject(input=tio.ScalarImage(tensor=data.unsqueeze(-1), affine=np.eye(4)))


def _plane(image: tio.Image) -> torch.Tensor:
    return image.data[..., 0]


# ---------------------------------------------------------------------------
# k-space domain (zero outside centre)
# ---------------------------------------------------------------------------


def test_scout_zeroes_outside_window_kspace() -> None:
    """k-space scout zeros every sample outside the central rectangle."""
    H, W = 64, 64
    Sx, Sy = 16, 16
    # Synthetic k-space — non-zero everywhere.
    data = torch.ones(1, H, W)
    subj = _make_subject(data)
    transform = ScoutAcquisitionTransform(
        scout_resolution=(Sx, Sy), domain="kspace", keys=("input",)
    )
    result = transform.apply_transform(subj)
    # 2026-06: the scout is now a SEPARATE 'scout' image; the input is preserved
    # (no longer overwritten in place).
    assert torch.equal(_plane(result["input"]), data)
    out = _plane(result["scout"])
    # Outside the centre is zero.
    x0 = (H - Sx) // 2
    y0 = (W - Sy) // 2
    outside = out.clone()
    outside[..., x0 : x0 + Sx, y0 : y0 + Sy] = 0.0
    assert outside.abs().max().item() == 0.0
    # Inside the centre is preserved.
    assert torch.allclose(
        out[..., x0 : x0 + Sx, y0 : y0 + Sy],
        data[..., x0 : x0 + Sx, y0 : y0 + Sy],
    )


def test_kspace_output_shape_matches_input() -> None:
    """Output shape equals input shape (only values change)."""
    data = torch.randn(1, 32, 32)
    subj = _make_subject(data)
    transform = ScoutAcquisitionTransform(
        scout_resolution=(8, 8), domain="kspace", keys=("input",)
    )
    out = _plane(transform.apply_transform(subj)["scout"])
    assert out.shape == data.shape


# ---------------------------------------------------------------------------
# Image domain (FFT-crop-iFFT)
# ---------------------------------------------------------------------------


def test_image_domain_low_pass() -> None:
    """Image-domain mode reduces high-frequency content."""
    torch.manual_seed(0)
    H, W = 32, 32
    data = torch.randn(1, H, W)
    subj = _make_subject(data)
    transform = ScoutAcquisitionTransform(
        scout_resolution=(8, 8), domain="image", keys=("input",)
    )
    out = _plane(transform.apply_transform(subj)["scout"])
    # Output is complex (after FFT/iFFT round trip); take magnitude.
    if torch.is_complex(out):
        out_mag = out.abs()
    else:
        out_mag = out.abs()
    # Full-image high-frequency energy via per-axis differences.
    hf_input = data.diff(dim=-1).abs().sum() + data.diff(dim=-2).abs().sum()
    hf_output = (
        out_mag.diff(dim=-1).abs().sum()
        + out_mag.diff(dim=-2).abs().sum()
    )
    assert hf_output.item() < hf_input.item()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_scout_larger_than_input_raises() -> None:
    """Scout window larger than the spatial dims raises."""
    data = torch.randn(1, 16, 16)
    subj = _make_subject(data)
    transform = ScoutAcquisitionTransform(
        scout_resolution=(32, 32), domain="kspace", keys=("input",)
    )
    with pytest.raises(ValueError, match="scout_resolution"):
        transform.apply_transform(subj)


def test_missing_key_skipped() -> None:
    """Subjects lacking the configured key are silently skipped."""
    data = torch.randn(1, 16, 16)
    subj = _make_subject(data)
    transform = ScoutAcquisitionTransform(
        scout_resolution=(8, 8), domain="kspace", keys=("ghost",)
    )
    out = transform.apply_transform(subj)
    # No-op: input unchanged.
    assert torch.equal(_plane(out["input"]), data)


# ---------------------------------------------------------------------------
# Multiple keys
# ---------------------------------------------------------------------------


def test_multiple_keys_processed_independently() -> None:
    """Each configured key is cropped independently."""
    affine = np.eye(4)
    subj = tio.Subject(
        input=tio.ScalarImage(tensor=torch.ones(1, 32, 32, 1), affine=affine),
        target=tio.ScalarImage(tensor=torch.ones(1, 32, 32, 1), affine=affine),
    )
    transform = ScoutAcquisitionTransform(
        scout_resolution=(8, 8), domain="kspace", keys=("input", "target")
    )
    out = transform.apply_transform(subj)
    # 2026-06: with multiple keys each scout is emitted as ``scout_<key>``; the
    # originals are preserved.
    for k in ("input", "target"):
        assert torch.equal(out[k].data, torch.ones(1, 32, 32, 1))
        d = _plane(out[f"scout_{k}"])
        assert d[..., 0, 0].item() == 0.0
        assert d[..., 16, 16].item() == 1.0  # inside the 8×8 window


# --- registry wiring (D9) ---------------------------------------------------


def test_scout_acquisition_is_registered_under_its_config_name() -> None:
    """The transform must be reachable from ``data.processing.transforms``.

    Before the registry it existed on disk with no way to be constructed from
    any config, so the consumer chain that reads its output was dead at link 0.
    """
    from spectramr.data.transforms.registry import get_transform

    entry = get_transform("scout_acquisition")
    assert entry.cls is ScoutAcquisitionTransform
    assert "scout" in entry.produces


def test_a_single_slice_volume_is_cropped_in_plane() -> None:
    """TorchIO's ``(C, W, H, 1)``: the crop used to read ``(H, D)`` and raise on ``D=1``."""
    data = torch.ones(1, 64, 48, 1)
    subj = tio.Subject(input=tio.ScalarImage(tensor=data, affine=np.eye(4)))
    t = ScoutAcquisitionTransform(scout_resolution=(32, 16), domain="kspace", keys=("input",))
    scout = t.apply_transform(subj)["scout"].data
    assert scout.shape == data.shape
    assert scout[0, 16:48, 16:32, 0].eq(1).all()
    assert scout.sum().item() == 32 * 16


def test_each_slice_of_a_volume_gets_the_same_in_plane_window() -> None:
    data = torch.ones(1, 32, 32, 3)
    subj = tio.Subject(input=tio.ScalarImage(tensor=data, affine=np.eye(4)))
    t = ScoutAcquisitionTransform(scout_resolution=(8, 8), domain="kspace", keys=("input",))
    scout = t.apply_transform(subj)["scout"].data
    assert [scout[..., d].sum().item() for d in range(3)] == [64.0, 64.0, 64.0]
