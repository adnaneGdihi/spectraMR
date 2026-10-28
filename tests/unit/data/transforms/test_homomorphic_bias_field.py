"""Tests for ``HomomorphicBiasFieldDecomposition``.

Targets ``spectramr.data.transforms.homomorphic_bias_field``. Decomposes a
ULF magnitude image into ``log_anatomy`` (high-frequency) and
``log_bias`` (low-frequency) components via Retinex-style filtering.

Categories:

- Construction: bad ``smoothness`` or even ``kernel_size`` → ``ValueError``
- Forward: image_key missing → no-op; ``enabled=False`` → no-op
- Property: ``exp(log_anatomy + log_bias) ≈ |x|`` reconstructs input
- Property: ``log_bias`` is *smoother* than ``log_anatomy`` (sums to less
  high-frequency energy)
- Inverse helper: ``reconstruct_from_log_components`` is the inverse
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torchio as tio

from spectramr.data.transforms.homomorphic_bias_field import (
    HomomorphicBiasFieldDecomposition,
    reconstruct_from_log_components,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_negative_smoothness_raises() -> None:
    """``smoothness <= 0`` → ``ValueError``."""
    with pytest.raises(ValueError, match="smoothness must be positive"):
        HomomorphicBiasFieldDecomposition(smoothness=0.0)


def test_even_kernel_size_raises() -> None:
    """Even ``kernel_size`` → ``ValueError``."""
    with pytest.raises(ValueError, match="must be odd"):
        HomomorphicBiasFieldDecomposition(kernel_size=10)


# ---------------------------------------------------------------------------
# Subject helpers
# ---------------------------------------------------------------------------


def _make_subject(image_data: torch.Tensor) -> tio.Subject:
    affine = np.eye(4)
    return tio.Subject(image=tio.ScalarImage(tensor=image_data, affine=affine))


# ---------------------------------------------------------------------------
# Forward — disabled / missing key
# ---------------------------------------------------------------------------


def test_disabled_is_no_op() -> None:
    """``enabled=False`` returns the subject unchanged."""
    img = torch.rand(1, 8, 8, 8) + 0.1
    subj = _make_subject(img)
    transform = HomomorphicBiasFieldDecomposition(enabled=False)
    out = transform.apply_transform(subj)
    assert "log_anatomy" not in out
    assert "log_bias" not in out


def test_missing_key_is_no_op() -> None:
    """If the image_key is absent, the transform returns subject unchanged."""
    img = torch.rand(1, 8, 8, 8) + 0.1
    subj = tio.Subject(other=tio.ScalarImage(tensor=img, affine=np.eye(4)))
    transform = HomomorphicBiasFieldDecomposition(image_key="image")
    out = transform.apply_transform(subj)
    assert "log_anatomy" not in out


# ---------------------------------------------------------------------------
# Forward — decomposition
# ---------------------------------------------------------------------------


def test_decomposition_attaches_both_components() -> None:
    """A valid 3D subject gets ``log_anatomy`` and ``log_bias`` attached."""
    img = torch.rand(1, 16, 16, 16) + 0.5  # strictly positive
    subj = _make_subject(img)
    transform = HomomorphicBiasFieldDecomposition(smoothness=4.0, kernel_size=11)
    out = transform.apply_transform(subj)
    assert "log_anatomy" in out
    assert "log_bias" in out
    assert out["log_anatomy"].data.shape == img.shape
    assert out["log_bias"].data.shape == img.shape


def test_decomposition_round_trips() -> None:
    """``exp(log_anatomy + log_bias)`` recovers ``|image| + ε`` within tol."""
    torch.manual_seed(0)
    img = torch.rand(1, 16, 16, 16) + 0.5
    subj = _make_subject(img)
    transform = HomomorphicBiasFieldDecomposition(
        epsilon=1e-6, smoothness=4.0, kernel_size=11
    )
    out = transform.apply_transform(subj)
    recon = reconstruct_from_log_components(
        out["log_anatomy"].data, out["log_bias"].data
    )
    # Absolute tolerance loose: smoothing introduces small numerical drift.
    assert torch.allclose(recon, img + 1e-6, atol=1e-3)


def test_log_bias_is_smoother_than_anatomy() -> None:
    """``log_bias`` should have less spatial high-frequency energy."""
    torch.manual_seed(0)
    img = torch.rand(1, 16, 16, 16) + 0.5
    subj = _make_subject(img)
    transform = HomomorphicBiasFieldDecomposition(smoothness=8.0, kernel_size=21)
    out = transform.apply_transform(subj)
    bias = out["log_bias"].data
    anatomy = out["log_anatomy"].data

    def _hf_energy(t: torch.Tensor) -> float:
        diffs = t.diff(dim=-1).abs().sum() + t.diff(dim=-2).abs().sum()
        return diffs.item()

    assert _hf_energy(bias) < _hf_energy(anatomy)


# ---------------------------------------------------------------------------
# Inverse helper
# ---------------------------------------------------------------------------


def test_reconstruct_returns_positive() -> None:
    """``exp`` is strictly positive — output is positive."""
    log_anatomy = torch.zeros(1, 4, 4)
    log_bias = torch.zeros_like(log_anatomy)
    out = reconstruct_from_log_components(log_anatomy, log_bias)
    # exp(0) = 1
    assert torch.allclose(out, torch.ones_like(out))


def test_reconstruct_additive_in_log_domain() -> None:
    """``reconstruct(a, b) == exp(a) * exp(b)``."""
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([0.5, -0.5])
    out = reconstruct_from_log_components(a, b)
    expected = torch.exp(a + b)
    assert torch.allclose(out, expected)


# ---------------------------------------------------------------------------
# HomomorphicBiasFieldCorrection (in-place, registry-reachable)
# ---------------------------------------------------------------------------


def _biased_disk(size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """A uniform disk times a smooth multiplicative ramp, shape (1, H, W, 1)."""
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    disk = (((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (size / 2.5) ** 2).float()
    bias = torch.exp(0.6 * (xx.float() / size - 0.5))
    return (disk * bias)[None, ..., None], disk.bool()[None, ..., None]


def test_correction_flattens_a_multiplicative_ramp_on_the_input_only() -> None:
    from spectramr.data.transforms.homomorphic_bias_field import HomomorphicBiasFieldCorrection

    biased, disk = _biased_disk()
    target = torch.rand_like(biased)
    subject = tio.Subject(
        input=tio.ScalarImage(tensor=biased.clone(), affine=np.eye(4)),
        target=tio.ScalarImage(tensor=target.clone(), affine=np.eye(4)),
    )
    out = HomomorphicBiasFieldCorrection(smoothness=4.0, kernel_size=15)(subject)

    def cv(x: torch.Tensor) -> float:
        inside = x[disk]
        return float(inside.std() / inside.mean())

    assert cv(out["input"].data) < 0.5 * cv(biased)
    assert torch.equal(out["input"].data[~disk], biased[~disk])
    assert torch.equal(out["target"].data, target)
    applied = out["input_log_bias"].data
    assert torch.allclose(out["input"].data * torch.exp(applied), biased, atol=1e-5)


def test_correction_is_reachable_from_yaml_by_name() -> None:
    from spectramr.data.transforms.homomorphic_bias_field import HomomorphicBiasFieldCorrection
    from spectramr.data.transforms.registry import build_transform, get_transform

    assert get_transform("homomorphic_bias_correction").cls is HomomorphicBiasFieldCorrection
    built = build_transform("homomorphic_bias_correction", smoothness=6.0, kernel_size=21)
    assert isinstance(built, HomomorphicBiasFieldCorrection)


def test_correction_refuses_what_it_cannot_correct() -> None:
    from spectramr.data.transforms.homomorphic_bias_field import HomomorphicBiasFieldCorrection

    with pytest.raises(ValueError, match="foreground_fraction"):
        HomomorphicBiasFieldCorrection(foreground_fraction=0.0)
    with pytest.raises(KeyError, match="no 'input' image"):
        HomomorphicBiasFieldCorrection()(_make_subject(torch.rand(1, 8, 8, 1)))
