"""Tests for realistic-degradation transforms.

Targets ``spectramr.data.transforms.realistic_degradations``:

- ``RicianNoise``: Gaussian-real + Gaussian-imag → magnitude pipeline
- ``MotionBlur3D``: depth-wise grouped 3D conv
- ``KSpaceUndersampling``: 3 patterns, all preserve centre region
- Functional helpers: ``add_rician_noise``, ``undersample_kspace``,
  ``apply_motion_blur_3d``

We do not exercise ``ElasticDeformation3D`` or
``RealisticDegradationPipeline`` here — they compose the primitives
above and are covered in the integration tier.
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch
import torchio as tio

from spectramr.data.transforms.realistic_degradations import (
    KSpaceUndersampling,
    MotionBlur3D,
    RandomB0Distortion,
    RicianNoise,
    add_rician_noise,
    apply_motion_blur_3d,
    create_realistic_degradation_pipeline,
    undersample_kspace,
)


# ---------------------------------------------------------------------------
# RicianNoise
# ---------------------------------------------------------------------------


def test_rician_noise_zero_level_is_identity() -> None:
    """``noise_level=0`` returns input unchanged (no random call)."""
    noise = RicianNoise(noise_level=0.0)
    x = torch.rand(1, 1, 8, 8)
    out = noise(x)
    assert torch.equal(out, x)


def test_rician_noise_preserves_shape() -> None:
    """Output shape matches input shape."""
    torch.manual_seed(0)
    noise = RicianNoise(noise_level=0.05)
    x = torch.rand(2, 1, 16, 16)
    out = noise(x)
    assert out.shape == x.shape


def test_rician_noise_output_is_non_negative() -> None:
    """Magnitude form ``sqrt((x+n_re)² + n_im²)`` is always ≥ 0."""
    torch.manual_seed(0)
    noise = RicianNoise(noise_level=0.1)
    x = torch.rand(1, 1, 16, 16)  # already non-negative
    out = noise(x)
    assert (out >= 0).all()


def test_rician_noise_higher_level_increases_variance() -> None:
    """Larger ``noise_level`` → larger output variance."""
    torch.manual_seed(0)
    x = torch.full((1, 1, 32, 32), 1.0)  # constant, so all variance is from noise
    out_lo = RicianNoise(noise_level=0.01)(x)
    torch.manual_seed(0)
    out_hi = RicianNoise(noise_level=0.5)(x)
    assert out_hi.var().item() > out_lo.var().item()


# ---------------------------------------------------------------------------
# MotionBlur3D
# ---------------------------------------------------------------------------


def test_motion_blur_zero_intensity_is_identity() -> None:
    """``motion_intensity=0`` returns input unchanged."""
    blur = MotionBlur3D(motion_intensity=0.0)
    x = torch.rand(1, 1, 4, 8, 8)
    out = blur(x)
    assert torch.equal(out, x)


def test_motion_blur_preserves_shape() -> None:
    """Output shape matches input ``[B, C, D, H, W]``."""
    blur = MotionBlur3D(motion_intensity=0.5, motion_direction=(1, 0, 0))
    x = torch.rand(1, 1, 4, 8, 8)
    out = blur(x)
    assert out.shape == x.shape


def test_motion_blur_normalises_motion_direction() -> None:
    """Motion direction is unit-norm after construction."""
    blur = MotionBlur3D(motion_direction=(2, 0, 0))
    norm = blur.motion_direction.norm().item()
    assert pytest.approx(norm, abs=1e-6) == 1.0


def test_motion_blur_smooths_signal() -> None:
    """A motion blur should reduce sharp-edge variance (low-pass effect)."""
    blur = MotionBlur3D(motion_intensity=0.5, motion_direction=(0, 1, 1))
    x = torch.zeros(1, 1, 4, 8, 8)
    x[..., 4:, :] = 1.0  # sharp edge along H
    out = blur(x)
    # Edge magnitude after blur should be smaller than before.
    edge_before = (x[..., 4, :] - x[..., 3, :]).abs().mean()
    edge_after = (out[..., 4, :] - out[..., 3, :]).abs().mean()
    assert edge_after < edge_before


# ---------------------------------------------------------------------------
# KSpaceUndersampling
# ---------------------------------------------------------------------------


def test_undersampling_acceleration_one_is_identity() -> None:
    """``acceleration_factor=1`` returns input unchanged."""
    us = KSpaceUndersampling(acceleration_factor=1)
    x = torch.rand(1, 1, 4, 8, 8)
    out = us(x)
    assert torch.equal(out, x)


@pytest.mark.parametrize("pattern", ["uniform", "equispaced", "gaussian"])
def test_undersampling_patterns_preserve_shape(pattern: str) -> None:
    """All three patterns return the input shape."""
    torch.manual_seed(0)
    us = KSpaceUndersampling(acceleration_factor=2, pattern=pattern)
    x = torch.rand(1, 1, 4, 8, 8)
    out = us(x)
    assert out.shape == x.shape


def test_undersampling_unknown_pattern_raises() -> None:
    """Unknown pattern → ``ValueError`` (CLAUDE.md #9)."""
    us = KSpaceUndersampling(acceleration_factor=2, pattern="not_a_pattern")
    with pytest.raises(ValueError, match="Unknown undersampling pattern"):
        us(torch.rand(1, 1, 2, 4, 4))


def test_undersampling_preserves_centre_region() -> None:
    """The DC / low-freq centre is always retained."""
    torch.manual_seed(0)
    us = KSpaceUndersampling(
        acceleration_factor=4, pattern="uniform", center_fraction=0.5
    )
    x = torch.ones(1, 1, 4, 8, 8)
    out = us(x)
    centre = out[..., 1:3, 2:6, 2:6]  # ~0.5 of D, H, W
    # All centre voxels should equal x's value (preserved).
    assert (centre == 1.0).all()


def test_undersampling_equispaced_uses_step() -> None:
    """Equispaced sampling: only every ``step``-th row along H is kept."""
    us = KSpaceUndersampling(
        acceleration_factor=4, pattern="equispaced", center_fraction=0.0
    )
    mask = us._create_mask((4, 8, 8))
    # Step = 4 → rows 0, 4 are sampled.
    sampled_rows = mask[0, :, 0]
    assert sampled_rows[0] == 1.0
    assert sampled_rows[4] == 1.0
    assert sampled_rows[1] == 0.0


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------


def test_add_rician_noise_functional_form() -> None:
    """Functional helper produces a finite tensor of the same shape."""
    torch.manual_seed(0)
    x = torch.rand(1, 1, 8, 8)
    out = add_rician_noise(x, noise_level=0.05)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_undersample_kspace_functional_form() -> None:
    """Functional helper produces a tensor of the same shape."""
    torch.manual_seed(0)
    x = torch.rand(1, 1, 4, 8, 8)
    out = undersample_kspace(x, acceleration_factor=2, pattern="uniform")
    assert out.shape == x.shape


def test_apply_motion_blur_3d_functional_form() -> None:
    """Functional 3D motion blur preserves shape."""
    x = torch.rand(1, 1, 4, 8, 8)
    out = apply_motion_blur_3d(x, intensity=0.3)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def test_create_realistic_degradation_pipeline_returns_module() -> None:
    """The factory returns a configured pipeline module."""
    pipeline = create_realistic_degradation_pipeline()
    assert hasattr(pipeline, "forward")
    assert callable(pipeline)


# ---------------------------------------------------------------------------
# RandomB0Distortion — TDA-001 regression
# ---------------------------------------------------------------------------


def test_random_b0_distortion_single_forward_pass() -> None:
    """Regression (TDA-001): only ONE B0GeometricDistortion forward runs per image.

    The pre-fix code called ``self.distortion`` twice — once with the raw
    (strength=1.0) ``b0_map`` whose result was silently discarded, then again
    with the strength-weighted map. The discarded first pass is removed; each
    image must trigger exactly one distortion forward pass.
    """
    torch.manual_seed(0)
    # p=1.0 so the transform always applies; one image -> one forward call.
    transform = RandomB0Distortion(max_displacement=1.0, p=1.0)

    # A subject with a single image keyed 'input', shape [C, W, H, D].
    subject = tio.Subject(input=tio.ScalarImage(tensor=torch.rand(1, 8, 8, 1)))

    # Mock forward to count calls; it must return a same-shaped tensor so the
    # downstream permute-back in apply_transform succeeds.
    def _identity_forward(tensor_in, b0_map):
        return tensor_in

    with mock.patch.object(
        transform.distortion,
        "forward",
        side_effect=_identity_forward,
    ) as mock_forward:
        transform(subject)

    assert mock_forward.call_count == 1, (
        "RandomB0Distortion must perform exactly one B0GeometricDistortion "
        f"forward pass per image; got {mock_forward.call_count}"
    )


# ── Sampling-grid correctness (identity, axis order) ─────────────────────────


class TestWarpGridIdentity:
    """Zero field must be the exact identity on a non-cubic (D, H, W) volume."""

    SHAPE = (4, 6, 8)

    def _vol(self):
        return torch.rand(1, 2, *self.SHAPE)

    @pytest.mark.parametrize("shift_dim", [3, 4])
    def test_b0_zero_field_is_identity(self, shift_dim):
        from spectramr.data.transforms.realistic_degradations import B0GeometricDistortion

        x = self._vol()
        out = B0GeometricDistortion(strength=1.0, shift_dim=shift_dim)(x, torch.zeros(1, 1, *self.SHAPE))
        torch.testing.assert_close(out, x, atol=1e-6, rtol=0)

    @pytest.mark.parametrize("shift_dim", [3, 4])
    def test_b0_shift_moves_only_along_shift_dim(self, shift_dim):
        from spectramr.data.transforms.realistic_degradations import B0GeometricDistortion

        x = torch.zeros(1, 1, *self.SHAPE)
        idx = [slice(None)] * 5
        idx[shift_dim] = 2
        x[tuple(idx)] = 1.0  # plane at index 2 along shift_dim
        b0 = torch.ones(1, 1, *self.SHAPE)  # +1 pixel everywhere
        out = B0GeometricDistortion(strength=1.0, shift_dim=shift_dim)(x, b0)
        other = [d for d in (2, 3, 4) if d != shift_dim]
        # Profile along every other axis stays constant (no transposition).
        for d in other:
            ref = out.select(d, 0)
            for i in range(1, out.shape[d]):
                torch.testing.assert_close(out.select(d, i), ref, atol=1e-6, rtol=0)
        prof = out.sum(dim=tuple(d for d in (2, 3, 4) if d != shift_dim)).flatten()
        # Pull-back out(r) = x(r + 1): the whole plane lands one voxel lower, none above.
        full = float(x.sum())
        assert prof[1].item() == pytest.approx(full, abs=1e-5)
        assert prof[2].item() < 1e-6 and prof[3].item() < 1e-6

    @pytest.mark.parametrize("bad", [0, 1, 2, 5, -1])
    def test_b0_rejects_other_shift_dim(self, bad):
        from spectramr.data.transforms.realistic_degradations import B0GeometricDistortion

        with pytest.raises(ValueError, match=r"3 \(H\) or 4 \(W\)"):
            B0GeometricDistortion(strength=1.0, shift_dim=bad)

    def test_b0_requires_shift_dim(self):
        from spectramr.data.transforms.realistic_degradations import B0GeometricDistortion

        with pytest.raises(TypeError):
            B0GeometricDistortion(strength=1.0)

    def test_elastic_zero_flow_is_identity(self, monkeypatch):
        from spectramr.data.transforms.realistic_degradations import ElasticDeformation3D

        monkeypatch.setattr(torch, "randn", lambda *a, **k: torch.zeros(*a, **k))
        x = self._vol()
        out = ElasticDeformation3D(alpha=1.0)(x)
        torch.testing.assert_close(out, x, atol=1e-6, rtol=0)

    def test_random_b0_tio_round_trip_strength_zero(self):
        data = torch.rand(1, 8, 6, 4)  # TorchIO (C, W, H, D), non-cubic
        subject = tio.Subject(img=tio.ScalarImage(tensor=data.clone()))
        out = RandomB0Distortion(max_displacement=0.0, p=1.0)(subject)["img"].data
        torch.testing.assert_close(out, data, atol=1e-6, rtol=0)


def test_random_b0_distortion_leaves_excluded_target_untouched() -> None:
    """The warp is an artefact of the degraded scan; the clean target must stay put."""
    torch.manual_seed(0)
    transform = RandomB0Distortion(max_displacement=2.0, p=1.0, exclude=("target",))
    vol = torch.rand(1, 16, 16, 2)
    subject = tio.Subject(
        input=tio.ScalarImage(tensor=vol.clone()), target=tio.ScalarImage(tensor=vol.clone())
    )
    out = transform(subject)
    assert torch.equal(out["target"].data, vol)
    assert not torch.equal(out["input"].data, vol)


def test_random_b0_distortion_draws_one_field_per_subject() -> None:
    """Two copies of the same scan get the same warp, so they stay registered."""
    torch.manual_seed(0)
    transform = RandomB0Distortion(max_displacement=2.0, p=1.0)
    vol = torch.rand(1, 16, 16, 2)
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=vol.clone()), input=tio.ScalarImage(tensor=vol.clone())
    )
    out = transform(subject)
    assert torch.equal(out["image"].data, out["input"].data)
    assert not torch.equal(out["input"].data, vol)
