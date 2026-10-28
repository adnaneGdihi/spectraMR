"""Tests for the normalization SSOT.

Targets ``spectramr.data.transforms.normalization``. The module is the single
source of truth for normalization across datasets, training strategies,
and TorchIO transforms — every consumer must delegate here, so the
tests verify the documented contracts:

- ``compute_magnitude`` handles every documented input layout
- ``normalize_percentile`` round-trips via ``denormalize_percentile``
- ``normalize_zscore`` produces zero-mean, unit-std output
- ``normalize_minmax`` maps the input range to ``out_range`` exactly
- ``normalize_tensor`` dispatcher routes to the right strategy
- ``NormalizationStrategy.from_string`` accepts case-insensitive aliases
"""

from __future__ import annotations

import math
from typing import ClassVar

import pytest
import torch
import torchio as tio

from spectramr.data.transforms.normalization import (
    DECOMPRESS_MAGNITUDE_CEILING,
    NormalizationConfig,
    NormalizationStrategy,
    compress_kspace_log,
    compute_magnitude,
    decompress_kspace_log,
    denormalize_percentile,
    normalize_minmax,
    normalize_percentile,
    normalize_tensor,
    normalize_zscore,
)


class TestLogKspaceCompression:
    """Phase-preserving log1p magnitude compression for k-space dynamic range.

    Regression for the experiment_11 DC-blob: raw k-space has a ~200x dynamic
    range the CNN can't represent, so data-consistency injection of the
    measured DC dominates the model's squashed output -> centre blob. Log
    compression reduces the range; this verifies it round-trips, preserves
    phase, and actually compresses.
    """

    def _make_complex_kspace(self):
        torch.manual_seed(0)
        # DC-dominated complex k-space: huge centre, tiny periphery.
        z = torch.randn(2, 4, 16, 16, dtype=torch.complex64) * 0.2
        z[:, :, 8, 8] = 50.0 + 10.0j  # DC spike
        return z

    def test_round_trip_complex(self):
        z = self._make_complex_kspace()
        recovered = decompress_kspace_log(compress_kspace_log(z))
        assert torch.allclose(recovered, z, atol=1e-3, rtol=1e-3)

    def test_round_trip_real_stacked_interleaved(self):
        z = self._make_complex_kspace()  # [B, C, H, W] complex
        # interleave R/I along channel dim 1 -> [B, 2C, H, W]
        stacked = torch.stack([z.real, z.imag], dim=2).reshape(z.shape[0], -1, *z.shape[2:])
        recovered = decompress_kspace_log(
            compress_kspace_log(stacked, channel_dim=1), channel_dim=1
        )
        assert torch.allclose(recovered, stacked, atol=1e-3, rtol=1e-3)

    def test_compression_reduces_dynamic_range(self):
        z = self._make_complex_kspace()
        comp = compress_kspace_log(z)
        raw_range = z.abs().max() / z.abs().median().clamp_min(1e-8)
        comp_range = comp.abs().max() / comp.abs().median().clamp_min(1e-8)
        assert comp_range < raw_range / 5, (
            f"log compression should shrink the dynamic range markedly: "
            f"raw={raw_range:.1f} comp={comp_range:.1f}"
        )

    def test_phase_is_preserved(self):
        z = self._make_complex_kspace()
        comp = compress_kspace_log(z)
        # Where magnitude is non-trivial, the phase angle must be unchanged.
        mask = z.abs() > 1e-3
        assert torch.allclose(torch.angle(comp)[mask], torch.angle(z)[mask], atol=1e-4)

    def test_odd_channel_count_raises(self):
        with pytest.raises(ValueError, match="even channel count"):
            compress_kspace_log(torch.randn(1, 3, 8, 8), channel_dim=1)

    def test_decompress_clamps_expm1_overflow(self):
        # Regression (experiment_11 kernelized-attention arm, iter-1000 smoke):
        # an under-trained model emitted a *compressed* |k| ~1750. Un-clamped
        # expm1(1750) overflows float32 -> inf -> inf*scale -> NaN that poisoned
        # every validation metric and made EarlyStopping pick "best" on NaN.
        # The decompress must stay finite for both memory layouts.
        huge_stacked = torch.full((1, 2, 4, 4), 1750.0, dtype=torch.float32)
        out = decompress_kspace_log(huge_stacked, channel_dim=1)
        assert torch.isfinite(out).all(), "decompress emitted inf/NaN on large |k|"

        huge_complex = torch.full((1, 1, 4, 4), 1750.0 + 0.0j, dtype=torch.complex64)
        out_c = decompress_kspace_log(huge_complex)
        assert torch.isfinite(out_c.abs()).all()

    def test_decompress_clamp_boundary(self):
        # Below the ceiling decompresses exactly (expm1(m)); far above is capped
        # at expm1(ceiling) rather than inf -- the ceiling sits well above any
        # physical compressed magnitude (<= ~6) so legitimate data is untouched.
        ceiling = DECOMPRESS_MAGNITUDE_CEILING
        below = torch.full((1, 1, 2, 2), ceiling - 5.0, dtype=torch.complex64)
        above = torch.full((1, 1, 2, 2), 500.0, dtype=torch.complex64)
        out_below = decompress_kspace_log(below).abs()
        out_above = decompress_kspace_log(above).abs()
        assert torch.allclose(out_below, torch.tensor(math.expm1(ceiling - 5.0)), rtol=1e-3)
        assert torch.allclose(out_above, torch.tensor(math.expm1(ceiling)), rtol=1e-3)
        assert torch.isfinite(out_above).all()


class TestSSOTRobustKspaceNormalization:
    """The single normalize/denormalize pair every k-space normalizer must use.

    Round-trips input AND output the same way (the experiment_11 fix requires
    input, prediction, and target to all live in the same compressed domain).
    """

    def _ksp(self):
        torch.manual_seed(1)
        z = torch.randn(2, 4, 16, 16, dtype=torch.complex64) * 0.2
        z[:, :, 8, 8] = 60.0 + 5.0j
        return z

    @pytest.mark.parametrize("log_scaling", [False, True])
    def test_round_trip_complex(self, log_scaling):
        from spectramr.data.transforms.normalization import (
            denormalize_kspace_robust,
            normalize_kspace_robust,
        )

        z = self._ksp()
        norm, scale = normalize_kspace_robust(z, percentile=0.95, log_scaling=log_scaling)
        back = denormalize_kspace_robust(norm, scale, log_scaling=log_scaling)
        assert torch.allclose(back, z, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("log_scaling", [False, True])
    def test_round_trip_real_stacked(self, log_scaling):
        from spectramr.data.transforms.normalization import (
            denormalize_kspace_robust,
            normalize_kspace_robust,
        )

        z = self._ksp()
        stk = torch.stack([z.real, z.imag], dim=2).reshape(z.shape[0], -1, *z.shape[2:])
        norm, scale = normalize_kspace_robust(
            stk, percentile=0.95, log_scaling=log_scaling, channel_dim=1
        )
        back = denormalize_kspace_robust(norm, scale, log_scaling=log_scaling, channel_dim=1)
        assert torch.allclose(back, stk, atol=1e-2, rtol=1e-2)

    def test_log_scaling_compresses_dynamic_range(self):
        from spectramr.data.transforms.normalization import normalize_kspace_robust

        z = self._ksp()
        lin, _ = normalize_kspace_robust(z, percentile=0.95, log_scaling=False)
        log, _ = normalize_kspace_robust(z, percentile=0.95, log_scaling=True)
        assert log.abs().max() < lin.abs().max() / 3

    def test_precomputed_scale_is_used(self):
        from spectramr.data.transforms.normalization import normalize_kspace_robust

        z = self._ksp()
        fixed = torch.tensor(4.0)
        norm, scale = normalize_kspace_robust(z, scale=fixed, log_scaling=False)
        assert torch.allclose(scale, fixed)
        assert torch.allclose(norm, z / fixed)


# ---------------------------------------------------------------------------
# NormalizationStrategy enum
# ---------------------------------------------------------------------------


def test_strategy_from_string_case_insensitive() -> None:
    """``from_string`` accepts mixed case."""
    assert NormalizationStrategy.from_string("PERCENTILE") == NormalizationStrategy.PERCENTILE
    assert NormalizationStrategy.from_string("ZScore") == NormalizationStrategy.ZSCORE


def test_strategy_robust_percentile_alias() -> None:
    """The legacy ``robust_percentile`` alias maps to ``PERCENTILE``."""
    assert (
        NormalizationStrategy.from_string("robust_percentile") == NormalizationStrategy.PERCENTILE
    )


def test_strategy_unknown_raises_with_valid_options() -> None:
    """Unknown strategy fails loud, listing the valid options."""
    with pytest.raises(ValueError, match="Unknown normalization strategy"):
        NormalizationStrategy.from_string("magic")


# ---------------------------------------------------------------------------
# compute_magnitude — multi-layout support
# ---------------------------------------------------------------------------


def test_compute_magnitude_complex_input() -> None:
    """Complex tensor → ``torch.abs(data)``."""
    cplx = torch.complex(torch.ones(2, 1, 4, 4), torch.zeros(2, 1, 4, 4))
    out = compute_magnitude(cplx)
    assert torch.allclose(out, torch.ones_like(out))


def test_compute_magnitude_b2hw_layout() -> None:
    """``[B, 2, H, W]`` real-stacked → sqrt(re² + im²)."""
    re = torch.full((1, 1, 2, 2), 3.0)
    im = torch.full((1, 1, 2, 2), 4.0)
    stacked = torch.cat([re, im], dim=1)  # [1, 2, 2, 2]
    out = compute_magnitude(stacked)
    assert torch.allclose(out, torch.full_like(out, 5.0), atol=1e-3)


def test_compute_magnitude_real_input_is_abs() -> None:
    """Real-only fallback returns ``abs(data)``."""
    data = torch.tensor([-2.0, 3.0, -5.0])
    out = compute_magnitude(data)
    assert torch.allclose(out, torch.tensor([2.0, 3.0, 5.0]))


# ---------------------------------------------------------------------------
# normalize_percentile + denormalize_percentile (round-trip)
# ---------------------------------------------------------------------------


def test_percentile_normalize_round_trip() -> None:
    """``denormalize(normalize(x))`` recovers x within float32 tolerance.

    Out-of-default range disables clamping in this test by setting
    ``clamp=False``.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 1, 16, 16) * 10.0
    norm, scale = normalize_percentile(x, percentile=0.99, clamp=False, out_range=(0.0, 1.0))
    recon = denormalize_percentile(norm, scale)
    assert torch.allclose(recon, x, atol=1e-4)


def test_percentile_normalize_scale_is_positive() -> None:
    """The returned scale is always positive (clamped to ``min_scale``)."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 8, 8)
    _, scale = normalize_percentile(x, percentile=0.99)
    assert scale.item() > 0


def test_percentile_normalize_handles_all_zero_input() -> None:
    """All-zero input → finite output (no NaN), scale clamped to eps."""
    x = torch.zeros(1, 1, 8, 8)
    norm, _ = normalize_percentile(x)
    assert torch.isfinite(norm).all()


def test_percentile_normalize_clamps_to_out_range() -> None:
    """``clamp=True`` enforces output range."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 16, 16) * 100.0
    norm, _ = normalize_percentile(x, percentile=0.5, clamp=True, out_range=(0.0, 1.0))
    assert norm.min().item() >= 0.0
    assert norm.max().item() <= 1.0


# ---------------------------------------------------------------------------
# normalize_zscore
# ---------------------------------------------------------------------------


def test_zscore_produces_zero_mean_unit_std() -> None:
    """Output has mean ≈ 0, std ≈ 1."""
    torch.manual_seed(0)
    x = torch.randn(1000) * 5.0 + 3.0  # large mean and std
    norm, (mean, std) = normalize_zscore(x)
    assert abs(norm.mean().item()) < 1e-5
    assert abs(norm.std().item() - 1.0) < 1e-3
    assert torch.allclose(mean, x.mean())
    assert torch.allclose(std, x.std() + 1e-8, atol=1e-9)


def test_zscore_constant_input_does_not_explode() -> None:
    """Constant input → std = 0 + eps; output is finite."""
    x = torch.full((100,), 7.0)
    norm, _ = normalize_zscore(x)
    assert torch.isfinite(norm).all()


# ---------------------------------------------------------------------------
# normalize_minmax
# ---------------------------------------------------------------------------


def test_minmax_maps_to_out_range_exactly() -> None:
    """min(out) = out_min, max(out) = out_max."""
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
    norm, _ = normalize_minmax(x, out_range=(-2.0, 2.0))
    assert torch.allclose(norm.min(), torch.tensor(-2.0))
    assert torch.allclose(norm.max(), torch.tensor(2.0))


def test_minmax_constant_input_returned_unchanged() -> None:
    """Constant input (range < eps) → returned unchanged with dummy stats."""
    x = torch.full((10,), 5.0)
    norm, (min_val, max_val) = normalize_minmax(x, eps=1e-3)
    assert torch.equal(norm, x)
    assert min_val == max_val


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_none_returns_input_unchanged() -> None:
    """Strategy NONE is identity."""
    x = torch.randn(2, 1, 4, 4)
    out = normalize_tensor(x, NormalizationConfig(strategy=NormalizationStrategy.NONE))
    assert torch.equal(out, x)


def test_dispatcher_percentile_routes_correctly() -> None:
    """Strategy PERCENTILE produces a normalised output."""
    x = torch.randn(1, 1, 8, 8) * 10.0
    cfg = NormalizationConfig(strategy=NormalizationStrategy.PERCENTILE, percentile=0.99)
    out = normalize_tensor(x, cfg)
    assert out.shape == x.shape
    assert not torch.equal(out, x)


def test_dispatcher_zscore_routes_correctly() -> None:
    """Strategy ZSCORE produces zero-mean, unit-std."""
    x = torch.randn(2000)
    cfg = NormalizationConfig(strategy=NormalizationStrategy.ZSCORE)
    out = normalize_tensor(x, cfg)
    assert abs(out.mean().item()) < 1e-3
    assert abs(out.std().item() - 1.0) < 1e-2


def test_dispatcher_minmax_routes_correctly() -> None:
    """Strategy MINMAX maps to default ``[0, 1]`` range."""
    x = torch.tensor([-2.0, 0.0, 2.0, 4.0])
    cfg = NormalizationConfig(strategy=NormalizationStrategy.MINMAX)
    out = normalize_tensor(x, cfg)
    assert torch.allclose(out.min(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(out.max(), torch.tensor(1.0), atol=1e-6)


# ---------------------------------------------------------------------------
# Sanity-shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 8, 8), (2, 2, 16, 16), (1, 4, 32, 32)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_percentile_shape_matrix(shape: tuple[int, ...]) -> None:
    """Percentile normalize preserves shape across the matrix."""
    x = torch.randn(*shape)
    norm, scale = normalize_percentile(x)
    assert norm.shape == shape
    assert torch.isfinite(norm).all()


# ---------------------------------------------------------------------------
# kspace_image_domain_scale — Parseval-compliant scale
# ---------------------------------------------------------------------------
# Ported out of ``M4RawRepetitionDataset.__getitem__``, which computed it inline
# while the transform ALSO normalized, double-scaling every arm. The dataset now
# matches and serves; this is the transform-layer home for the same physics.


class TestKSpaceImageDomainScale:
    """The scale is measured on the image-domain coil RSS, not on |k|."""

    @staticmethod
    def _kspace_of(img: torch.Tensor) -> torch.Tensor:
        from spectramr.infrastructure.physics.fft_ops import fft2c

        return fft2c(img)

    def test_matches_the_image_rss_quantile(self) -> None:
        """It equals the quantile of |RSS(ifft2c(k))| — the definition."""
        from spectramr.data.transforms.normalization import kspace_image_domain_scale

        torch.manual_seed(0)
        img = torch.randn(3, 16, 16, dtype=torch.complex64)  # (coils, H, W)
        k = self._kspace_of(img)

        expected_rss = torch.sqrt((img.abs() ** 2).sum(dim=0) + 1e-8)
        expected = torch.quantile(expected_rss.flatten().float(), 0.95)

        got = kspace_image_domain_scale(k, percentile=0.95)
        assert torch.allclose(got, expected, rtol=1e-4)

    def test_differs_from_the_kspace_magnitude_quantile(self) -> None:
        """The two domains are NOT interchangeable — k-space is DC-heavy.

        Guards the migration: swapping the domains silently would change what
        the network is trained on.
        """
        from spectramr.data.transforms.normalization import (
            compute_magnitude,
            kspace_image_domain_scale,
        )

        img = torch.zeros(1, 32, 32, dtype=torch.complex64)
        img[:, 8:24, 8:24] = 1.0
        k = self._kspace_of(img)

        image_scale = kspace_image_domain_scale(k, percentile=0.95)
        kspace_scale = torch.quantile(compute_magnitude(k).flatten().float(), 0.95)
        assert not torch.isclose(image_scale, kspace_scale, rtol=0.05)

    def test_accepts_real_interleaved_layout(self) -> None:
        """Real/imag interleaved along the channel axis gives the same scale."""
        from spectramr.data.transforms.normalization import kspace_image_domain_scale

        torch.manual_seed(1)
        img = torch.randn(2, 16, 16, dtype=torch.complex64)
        k = self._kspace_of(img)

        stacked = torch.empty(4, 16, 16)
        stacked[0::2] = k.real
        stacked[1::2] = k.imag

        assert torch.allclose(
            kspace_image_domain_scale(stacked, percentile=0.9),
            kspace_image_domain_scale(k, percentile=0.9),
            rtol=1e-4,
        )

    def test_odd_channel_count_falls_back_to_kspace_magnitude(self) -> None:
        """An odd channel count is not a real/imag stack — do not invent phase."""
        from spectramr.data.transforms.normalization import (
            compute_magnitude,
            kspace_image_domain_scale,
        )

        x = torch.rand(3, 8, 8)  # cannot be real/imag interleaved
        expected = torch.quantile(compute_magnitude(x).flatten().float(), 0.9)
        assert torch.allclose(kspace_image_domain_scale(x, percentile=0.9), expected, rtol=1e-4)


class TestKSpaceNormalizationScaleDomain:
    """``scale_domain`` selects the measurement domain and rejects typos."""

    def test_unknown_scale_domain_raises(self) -> None:
        """No silent fallback to a default domain (pitfall #9)."""
        from spectramr.data.transforms.normalization import KSpaceNormalizationTransform

        with pytest.raises(ValueError, match="Unknown scale_domain"):
            KSpaceNormalizationTransform(scale_domain="parseval")

    def test_image_domain_publishes_the_image_scale(self) -> None:
        """The published kspace_scale is the one actually divided by."""
        import torchio as tio

        from spectramr.data.transforms.normalization import (
            KSpaceNormalizationTransform,
            kspace_image_domain_scale,
        )
        from spectramr.infrastructure.physics.fft_ops import fft2c

        torch.manual_seed(2)
        img = torch.randn(1, 16, 16, dtype=torch.complex64)
        k = fft2c(img)
        stacked = torch.empty(2, 16, 16, 1)
        stacked[0::2, ..., 0] = k.real
        stacked[1::2, ..., 0] = k.imag

        subject = tio.Subject(kspace=tio.ScalarImage(tensor=stacked))
        out = KSpaceNormalizationTransform(
            percentile=0.95, log_scaling=False, scale_domain="image"
        )(subject)

        expected = kspace_image_domain_scale(stacked, percentile=0.95)
        assert torch.allclose(out["kspace_scale"], expected, rtol=1e-4)
        assert torch.allclose(out["kspace"].data, stacked / expected, rtol=1e-4, atol=1e-6)

    def test_the_source_contrast_takes_its_own_scale(self) -> None:
        """A 10x brighter source leaves the target's scale and values untouched (#2452)."""
        import torchio as tio

        from spectramr.data.transforms.normalization import KSpaceNormalizationTransform

        torch.manual_seed(4)
        target = torch.randn(2, 8, 8, 1) * 30
        source = torch.randn(2, 8, 8, 1) * 30
        transform = KSpaceNormalizationTransform(percentile=0.95, log_scaling=False)

        def run(src: torch.Tensor) -> tio.Subject:
            return transform(
                tio.Subject(
                    kspace=tio.ScalarImage(tensor=target.clone()),
                    input=tio.ScalarImage(tensor=target.clone()),
                    kspace_source=tio.ScalarImage(tensor=src.clone()),
                )
            )

        dim, bright = run(source), run(source * 10)
        assert torch.equal(dim["kspace_scale"], bright["kspace_scale"])
        assert torch.equal(dim["kspace"].data, bright["kspace"].data)
        # Each contrast lands on its own unit range, so the source is scale-free too.
        assert torch.allclose(dim["kspace_source"].data, bright["kspace_source"].data, rtol=1e-5)
        own = transform.spec.compute_scale(source, channel_dim=0)
        assert torch.allclose(dim["kspace_source"].data, source / own, rtol=1e-5)

    def test_the_second_repetition_shares_the_input_scale(self) -> None:
        """``input_alt`` is the input's sibling repetition; one scale for both.

        Unlisted, it was served raw beside a normalized input -- ~22x larger on
        arm G's config -- so the model was graded on an input far off its range.
        """
        import torchio as tio

        from spectramr.data.transforms.normalization import KSpaceNormalizationTransform

        torch.manual_seed(3)
        raw = {k: torch.randn(2, 8, 8, 1) * 30 for k in ("kspace", "input", "input_alt")}
        subject = tio.Subject(**{k: tio.ScalarImage(tensor=v.clone()) for k, v in raw.items()})
        out = KSpaceNormalizationTransform(log_scaling=False)(subject)
        scale = out["kspace_scale"]
        assert torch.allclose(out["input_alt"].data * scale, raw["input_alt"], rtol=1e-4)
        assert torch.allclose(out["input"].data * scale, raw["input"], rtol=1e-4)


class TestImageDomainScaleAxes:
    """The image-domain scale must FFT over (H, W), whatever the layout.

    Regression: the first implementation took the trailing two axes, so a
    multi-slice TorchIO subject ``(2C, H, W, D)`` was transformed over ``(W, D)``
    — a ~2.6x wrong scale on real anatomy.

    Fixtures here are STRUCTURED on purpose. White-noise k-space is invariant
    under any unitary transform, so a noise fixture cannot detect an axis bug
    (and the round-trip test cannot either — a scalar scale cancels regardless
    of how wrongly it was computed).
    """

    @staticmethod
    def _phantom(slices, coils, h, w):
        from spectramr.infrastructure.physics.fft_ops import fft2c

        torch.manual_seed(0)
        img = torch.zeros(slices, coils, h, w, dtype=torch.complex64)
        img[:, :, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0
        img = img + 0.02 * torch.randn(slices, coils, h, w, dtype=torch.complex64)
        k = fft2c(img.reshape(-1, h, w)).reshape(slices, coils, h, w)
        return img, k

    @staticmethod
    def _expected(img, percentile=0.95):
        rss = torch.sqrt((img.abs() ** 2).sum(dim=1) + 1e-8)
        return torch.quantile(rss.flatten().float(), percentile)

    @pytest.mark.parametrize(
        "slices,coils,h,w",
        [(4, 2, 32, 32), (1, 2, 32, 24), (3, 3, 40, 24)],
        ids=["multislice", "single-slice", "non-square"],
    )
    def test_torchio_layout_c_h_w_d(self, slices, coils, h, w):
        """TorchIO ``(2C, H, W, D)`` with channel_dim=0."""
        from spectramr.data.transforms.normalization import kspace_image_domain_scale

        img, k = self._phantom(slices, coils, h, w)
        stacked = torch.empty(coils * 2, h, w, slices)
        stacked[0::2] = k.real.permute(1, 2, 3, 0)
        stacked[1::2] = k.imag.permute(1, 2, 3, 0)

        got = kspace_image_domain_scale(stacked, percentile=0.95, channel_dim=0)
        assert torch.allclose(got, self._expected(img), rtol=1e-2)

    def test_batched_layout_b_c_h_w(self):
        """Inference ``(B, 2C, H, W)`` with channel_dim=1."""
        from spectramr.data.transforms.normalization import kspace_image_domain_scale

        img, k = self._phantom(2, 3, 32, 32)  # slices axis reused as batch
        stacked = torch.empty(2, 6, 32, 32)
        stacked[:, 0::2] = k.real
        stacked[:, 1::2] = k.imag

        got = kspace_image_domain_scale(stacked, percentile=0.95, channel_dim=1)
        assert torch.allclose(got, self._expected(img), rtol=1e-2)

    def test_too_few_spatial_axes_raises(self):
        """A tensor with no (H, W) after the channel axis cannot form an image."""
        from spectramr.data.transforms.normalization import kspace_image_domain_scale

        with pytest.raises(ValueError, match="two spatial axes"):
            kspace_image_domain_scale(torch.randn(4, 8), channel_dim=0)


class TestKSpaceNormalizationSpec:
    """One resolver for the k-space scale, shared by training and inference."""

    CFG = {
        "normalize_kspace": True,
        "kspace_percentile": 0.95,
        "log_scaling": True,
        "kspace_scale_domain": "image",
        "normalization_kwargs": {"percentile": 0.99, "clamp": False},
    }

    def test_from_dict_and_from_object_agree(self):
        """Inference passes a dict, training a pydantic model."""
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        class _Obj:
            normalize_kspace = True
            kspace_percentile = 0.95
            log_scaling = True
            kspace_scale_domain = "image"

        assert KSpaceNormalizationSpec.from_data_config(
            self.CFG
        ) == KSpaceNormalizationSpec.from_data_config(_Obj())

    def test_disabled_spec_is_a_no_op_with_unit_scale(self):
        """Callers never branch on ``enabled`` themselves."""
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        spec = KSpaceNormalizationSpec.from_data_config(dict(self.CFG, normalize_kspace=False))
        x = torch.randn(1, 4, 8, 8)
        out, scale = spec.normalize(x)
        assert out is x and float(scale) == 1.0
        assert spec.denormalize(x, scale) is x

    @pytest.mark.parametrize("domain", ["kspace", "image"])
    @pytest.mark.parametrize("log_scaling", [False, True])
    def test_round_trip(self, domain, log_scaling):
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec
        from spectramr.infrastructure.physics.fft_ops import fft2c

        spec = KSpaceNormalizationSpec.from_data_config(
            dict(self.CFG, kspace_scale_domain=domain, log_scaling=log_scaling)
        )
        torch.manual_seed(0)
        img = torch.zeros(1, 2, 16, 16, dtype=torch.complex64)
        img[:, :, 4:12, 4:12] = 1.0
        k = fft2c(img.reshape(-1, 16, 16)).reshape(1, 2, 16, 16)
        x = torch.empty(1, 4, 16, 16)
        x[:, 0::2], x[:, 1::2] = k.real, k.imag

        normed, scale = spec.normalize(x, channel_dim=1)
        restored = spec.denormalize(normed, scale, channel_dim=1)
        assert torch.allclose(restored, x, rtol=1e-3, atol=1e-4 * x.abs().max())

    def test_unknown_scale_domain_raises(self):
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        with pytest.raises(ValueError, match="scale_domain"):
            KSpaceNormalizationSpec(scale_domain="parseval")

    @pytest.mark.parametrize(
        ("normalize", "log_scaling", "expected"),
        [(True, True, True), (False, True, False), (True, False, False), (False, False, False)],
    )
    def test_log_compressed_needs_both_knobs(self, normalize, log_scaling, expected):
        """#2353: ``normalize`` is a no-op when disabled, so ``log_scaling`` alone
        compresses nothing and a reader that trusted it would ``expm1`` raw k-space."""
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        spec = KSpaceNormalizationSpec.from_data_config(
            dict(self.CFG, normalize_kspace=normalize, log_scaling=log_scaling)
        )
        assert spec.log_compressed is expected

    def test_log_compressed_agrees_with_what_normalize_does(self):
        """The predicate is pinned to the behaviour, not to the flag values."""
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        x = torch.randn(1, 4, 8, 8) * 50
        for normalize in (True, False):
            spec = KSpaceNormalizationSpec.from_data_config(
                dict(self.CFG, normalize_kspace=normalize, log_scaling=True)
            )
            out, scale = spec.normalize(x, channel_dim=1)
            linear = x / scale
            assert (not torch.allclose(out, linear)) is spec.log_compressed


# ---------------------------------------------------------------------------
# Image-domain normalization SSOT (#760) — the image twin of the #571 k-space fix
# ---------------------------------------------------------------------------


class TestKSpaceNormalizationSpecReadsTheNestedBlock:
    """The resolver reads ``data.processing``, not the flat legacy names.

    The phase-9 block decomposition re-parented all five knobs under
    ``data.processing`` and renamed two of them, but ``from_data_config`` kept
    reading the flat pre-decomposition spellings off ``data`` itself with
    ``getattr(data, name, default)``. Post-decomposition those names exist on no
    schema object and there is no forwarding shim, so every read missed and
    every declared value was replaced by its default.

    ``normalize`` is a *silent* no-op when disabled, so the
    ``apply_kspace_normalization`` fallback — which exists precisely to normalize
    a batch the dataloader served raw — returned the raw batch and a unit scale
    while reporting success. ``experiment_11_attention_none`` trained on raw
    k-space at ``|k|max ~ 2400`` and its k-space renders were ``expm1``-ed on
    the strength of the declaration, clamping the whole contrast-carrying band
    to ``DECOMPRESS_MAGNITUDE_CEILING`` and drawing a phase-only edge map.

    Every test here feeds the REAL ``DataProcessingConfigSchema``: a fixture
    that hand-writes the knobs would keep passing through the next rename, which
    is exactly how the flat reader survived a whole schema migration. The
    pre-existing suites above feed flat dicts, which is why they stayed green.
    """

    #: The exp_11 kspace_filling attention-shootout declaration.
    ARM: ClassVar[dict[str, object]] = {
        "enable_kspace_normalization": True,
        "enable_log_scaling": True,
        "kspace_percentile": 0.95,
        "kspace_scale_domain": "image",
    }

    @staticmethod
    def _data(**overrides):
        """A ``data``-shaped holder carrying the real processing schema block."""
        from types import SimpleNamespace

        from spectramr.config.schemas.data import DataProcessingConfigSchema

        return SimpleNamespace(
            processing=DataProcessingConfigSchema(
                **{**TestKSpaceNormalizationSpecReadsTheNestedBlock.ARM, **overrides}
            )
        )

    def test_reads_the_declared_nested_values(self):
        """The regression: this resolved to every default before the fix."""
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        spec = KSpaceNormalizationSpec.from_data_config(self._data())

        assert spec.enabled is True  # was False -> normalize() was a no-op
        assert spec.log_scaling is True  # was False -> no log1p compression
        assert spec.percentile == 0.95  # was the 0.99 default
        assert spec.scale_domain == "image"  # was the 'kspace' default

    def test_center_fraction_comes_from_the_block(self):
        """Parity with the transform, which is handed the same 0.25 default."""
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        assert KSpaceNormalizationSpec.from_data_config(self._data()).center_fraction == 0.25
        spec = KSpaceNormalizationSpec.from_data_config(self._data(log_scaling_center_fraction=0.1))
        assert spec.center_fraction == 0.1

    def test_nested_block_wins_over_stale_flat_names(self):
        """A leftover flat key must not override the block that supersedes it."""
        from types import SimpleNamespace

        from spectramr.config.schemas.data import DataProcessingConfigSchema
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        data = SimpleNamespace(
            processing=DataProcessingConfigSchema(**self.ARM),
            normalize_kspace=False,
            log_scaling=False,
            kspace_percentile=0.5,
        )
        spec = KSpaceNormalizationSpec.from_data_config(data)

        assert (spec.enabled, spec.log_scaling, spec.percentile) == (True, True, 0.95)

    def test_absent_block_and_absent_knobs_resolve_to_defaults(self):
        """Nothing declared anywhere: a standalone strategy, a dataless paradigm."""
        from types import SimpleNamespace

        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        for empty in ({}, SimpleNamespace()):
            spec = KSpaceNormalizationSpec.from_data_config(empty)
            assert spec == KSpaceNormalizationSpec()
            assert spec.enabled is False

    def test_block_that_omits_a_knob_raises(self):
        """Absent-block and absent-field are different facts (CLAUDE.md #3b).

        Defaulting here is how a rename disables the mechanism in silence: the
        reader keeps the old spelling, every read misses, and nothing goes red
        because "absent" and "off" are the same boolean.
        """
        from types import SimpleNamespace

        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        partial = SimpleNamespace(
            enable_kspace_normalization=True,
            kspace_percentile=0.95,
            kspace_scale_domain="image",
            log_scaling_center_fraction=0.25,
        )  # `enable_log_scaling` renamed away
        with pytest.raises(AttributeError, match="enable_log_scaling"):
            KSpaceNormalizationSpec.from_data_config(SimpleNamespace(processing=partial))

        with pytest.raises(AttributeError, match="enable_log_scaling"):
            KSpaceNormalizationSpec.from_data_config(
                {
                    "processing": {k: v for k, v in self.ARM.items() if k != "enable_log_scaling"}
                    | {"log_scaling_center_fraction": 0.25}
                }
            )

    def test_explicit_none_raises_for_a_non_nullable_knob(self):
        """An explicit None must not read as the schema default."""
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        with pytest.raises(ValueError, match="kspace_percentile"):
            KSpaceNormalizationSpec.from_data_config(
                {
                    "processing": dict(
                        self.ARM, kspace_percentile=None, log_scaling_center_fraction=0.25
                    )
                }
            )

    def test_nullable_center_fraction_may_be_none(self):
        """The one knob with a real null meaning: 'use the full magnitude'."""
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        spec = KSpaceNormalizationSpec.from_data_config(
            {"processing": dict(self.ARM, log_scaling_center_fraction=None)}
        )
        assert spec.center_fraction is None

    def test_legacy_flat_dict_still_resolves_and_says_so(self, caplog):
        """A checkpoint's stored `data` dict predates the block; honour it, loudly."""
        import logging

        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        flat = {
            "normalize_kspace": True,
            "log_scaling": True,
            "kspace_percentile": 0.95,
            "kspace_scale_domain": "image",
        }
        with caplog.at_level(logging.WARNING):
            spec = KSpaceNormalizationSpec.from_data_config(flat)

        assert (spec.enabled, spec.log_scaling, spec.percentile) == (True, True, 0.95)
        assert spec.scale_domain == "image"
        assert "pre-decomposition FLAT names" in caplog.text

    def test_nested_resolution_matches_what_the_transform_is_handed(self):
        """The fallback's stated job: reproduce the transform's normalization.

        ``TorchIOTransformBuilder`` reads the same block and hands those values
        to ``KSpaceNormalizationTransform``. If the two resolvers disagree, the
        fallback silently trains on a different distribution than the chain.
        """
        from spectramr.data.transforms.normalization import (
            KSpaceNormalizationSpec,
            KSpaceNormalizationTransform,
        )

        block = self._data().processing
        transform = KSpaceNormalizationTransform(
            percentile=block.kspace_percentile,
            log_scaling=block.enable_log_scaling,
            center_fraction=block.log_scaling_center_fraction,
            scale_domain=block.kspace_scale_domain,
        )
        resolved = KSpaceNormalizationSpec.from_data_config(self._data())

        for field in ("percentile", "log_scaling", "center_fraction", "scale_domain"):
            assert getattr(resolved, field) == getattr(transform.spec, field), field


class TestImageNormalizationSpecResolution:
    """One resolver decides WHAT normalizes image intensity, and records WHY."""

    @staticmethod
    def _spec(nt, dt=None, **kw):
        from spectramr.data.transforms.normalization import ImageNormalizationSpec

        return ImageNormalizationSpec.from_declared(nt, dt, kw or None)

    def test_declared_type_wins(self) -> None:
        from spectramr.data.transforms.normalization import NormalizationStrategy

        spec = self._spec("percentile", "nifti_paired")
        assert spec.source == "declared"
        assert spec.config.strategy is NormalizationStrategy.PERCENTILE

    def test_contrast_aware_with_no_declaration_inherits_the_dataset_default(
        self,
    ) -> None:
        """66 corpus arms declare NO ``normalization_type`` and relied entirely
        on the dataset's internal pass. Resolving them to "none" would have
        trained those arms on unnormalized intensities — the single most
        dangerous way this port could have gone wrong."""
        from spectramr.data.transforms.normalization import NormalizationStrategy

        for dtype in ("contrast_aware_paired", "nifti_paired"):
            spec = self._spec("none", dtype)
            assert spec.source == "contrast_aware_legacy_default"
            assert spec.config.strategy is NormalizationStrategy.PERCENTILE
            assert spec.enabled

    def test_other_dataset_types_with_none_really_get_none(self) -> None:
        """The inheritance is scoped. A ``nifti`` arm never had a dataset pass,
        so "none" must keep meaning none rather than silently gaining one."""
        spec = self._spec("none", "nifti")
        assert spec.source == "disabled"
        assert not spec.enabled

    def test_config_vocabulary_aliases_resolve(self) -> None:
        """The schema Literal and the strategy enum were named independently:
        the config says ``robust_percentile`` / ``standard`` where the enum says
        PERCENTILE / ZSCORE. 270 corpus arms use ``robust_percentile``."""
        from spectramr.data.transforms.normalization import NormalizationStrategy

        assert (
            self._spec("robust_percentile", "nifti").config.strategy
            is NormalizationStrategy.PERCENTILE
        )
        assert self._spec("standard", "nifti").config.strategy is NormalizationStrategy.ZSCORE

    def test_unknown_type_raises_rather_than_degrading(self) -> None:
        # The resolver reports in CONFIG vocabulary ("normalization_type"), not
        # enum vocabulary -- `NormalizationStrategy.from_string` is the one that
        # says "Unknown normalization strategy" (tested separately above). This
        # regex tracked the enum wording and had been red on dev since the two
        # messages diverged.
        with pytest.raises(ValueError, match="Unknown normalization_type"):
            self._spec("robust_percentil", "nifti")  # a typo, not a member

    def test_percentile_accepts_the_0_100_spelling(self) -> None:
        spec = self._spec("percentile", "nifti", percentile=99.5)
        assert spec.config.percentile == pytest.approx(0.995)


class TestResolveImageNormalization:
    """One owner for the three-part decision the builder and predict used to copy.

    ``resolve_image_normalization`` decides (1) whether k-space normalization
    takes precedence, (2) the vocabulary fold, (3) the spec. Before it, the
    train chain, the val chain and ``pipelines/infer.py`` each carried a copy;
    a drifted copy surfaces as a differently scaled tensor, not an error.
    """

    @staticmethod
    def _resolve(**overrides):
        from spectramr.data.transforms.normalization import resolve_image_normalization

        kwargs = {
            "normalization_type": "percentile",
            "dataset_type": "nifti",
            "normalization_kwargs": None,
            "kspace_normalization_enabled": False,
        }
        kwargs.update(overrides)
        return resolve_image_normalization(**kwargs)

    def test_kspace_normalization_takes_precedence_and_says_so(self, caplog) -> None:
        """An arm that normalizes k-space gets NO image normalization."""
        with caplog.at_level("INFO", logger="spectramr.data.transforms.normalization"):
            spec = self._resolve(
                normalization_type="robust_percentile", kspace_normalization_enabled=True
            )
        assert spec is None
        assert "Image normalization 'robust_percentile' SKIPPED" in caplog.text
        assert "normalize_kspace=True" in caplog.text

    def test_precedence_over_none_is_silent(self, caplog) -> None:
        """Nothing was declared, so there is nothing to report as skipped."""
        with caplog.at_level("INFO", logger="spectramr.data.transforms.normalization"):
            spec = self._resolve(normalization_type="none", kspace_normalization_enabled=True)
        assert spec is None
        assert "SKIPPED" not in caplog.text

    def test_precedence_wins_even_over_an_unknown_type(self) -> None:
        """The value is never dispatched behind the gate, so it cannot raise."""
        assert self._resolve(normalization_type="scalar", kspace_normalization_enabled=True) is None

    def test_robust_percentile_folds_through_the_enum_alias_table(self) -> None:
        """The fold has one owner: ``NormalizationStrategy.from_string``.

        The resolver does not fold beforehand; it hands the declared spelling
        to ``from_declared`` and gets PERCENTILE back, identical to what the
        canonical spelling resolves to.
        """
        from spectramr.data.transforms.normalization import NormalizationStrategy

        folded = self._resolve(normalization_type="robust_percentile")
        canonical = self._resolve(normalization_type="percentile")
        assert folded is not None and canonical is not None
        assert folded.config.strategy is NormalizationStrategy.PERCENTILE
        assert folded == canonical

    def test_unknown_type_raises_when_the_gate_is_open(self) -> None:
        with pytest.raises(ValueError, match="Unknown normalization_type"):
            self._resolve(normalization_type="scalar")

    def test_none_resolves_to_a_disabled_spec_not_to_none(self) -> None:
        """``None`` means "k-space won"; a declared 'none' is a spec that is off.

        The two are different provenance and the callers treat them the same
        way (append nothing), so the distinction must survive in the return
        value or a log reader cannot tell why nothing was applied.
        """
        spec = self._resolve(normalization_type="none")
        assert spec is not None
        assert not spec.enabled
        assert spec.source == "disabled"

    def test_none_on_a_contrast_aware_dataset_inherits_the_legacy_default(self) -> None:
        spec = self._resolve(normalization_type="none", dataset_type="contrast_aware_paired")
        assert spec is not None and spec.enabled
        assert spec.source == "contrast_aware_legacy_default"

    def test_the_resolved_spec_is_logged_once_with_its_source(self, caplog) -> None:
        with caplog.at_level("INFO", logger="spectramr.data.transforms.normalization"):
            self._resolve(normalization_type="percentile", normalization_kwargs={"clamp": False})
        lines = [
            r.getMessage()
            for r in caplog.records
            if "[NORMALIZATION] Image percentile" in r.getMessage()
        ]
        assert len(lines) == 1
        assert "clamp=False" in lines[0] and "source=declared" in lines[0]

    def test_kwargs_reach_the_spec_unchanged(self) -> None:
        spec = self._resolve(
            normalization_type="percentile",
            normalization_kwargs={"percentile": 99.5, "clamp": False, "min_scale": 0.1},
        )
        assert spec is not None
        assert spec.config.percentile == pytest.approx(0.995)
        assert spec.config.clamp is False
        assert spec.config.min_scale == pytest.approx(0.1)


class TestTheLegacyDefaultIsTheDATASETsDEFAULT:
    """Port fidelity, asserted against the thing it was ported FROM.

    The k-space port (#571) established the trap: the dataset scale and the
    transform scale were NOT equivalent, so deleting either side silently
    changed the physics. Pinning the ported numbers as literals would only
    prove they match the literals. This derives the expected config from
    ``ContrastConfig``'s own defaults — the values the dataset actually fed the
    SSOT — so a drift in either place fails here.
    """

    def test_it_matches_what_the_subject_builder_used_to_build(self) -> None:
        from spectramr.data.datasets.contrast_aware import ContrastConfig
        from spectramr.data.transforms.normalization import (
            _CONTRAST_AWARE_LEGACY_DEFAULT,
            NormalizationConfig,
            NormalizationStrategy,
        )

        cc = ContrastConfig(name="T1")  # the defaults every arm silently got
        percentile = cc.percentile / 100.0 if cc.percentile > 1.0 else cc.percentile
        expected = NormalizationConfig(
            strategy=NormalizationStrategy.from_string(cc.normalization),
            percentile=percentile,
            out_range=cc.out_range,
            clamp=cc.clamp,
            min_scale=0.05,  # the noise floor the builder hardcoded
        )
        assert expected == _CONTRAST_AWARE_LEGACY_DEFAULT

    def test_the_noise_floor_survived_the_port(self) -> None:
        """``min_scale`` is the one parameter torchio's RescaleIntensity had no
        equivalent for. Losing it means amplifying background noise on a
        low-signal ULF volume."""
        from spectramr.data.transforms.normalization import (
            _CONTRAST_AWARE_LEGACY_DEFAULT,
        )

        assert _CONTRAST_AWARE_LEGACY_DEFAULT.min_scale == 0.05


class TestImageNormalizationTransformAppliesOnce:
    """The served tensor must equal ONE application of the resolved config."""

    @staticmethod
    def _subject(data):
        import torchio as tio

        return tio.Subject(
            input=tio.ScalarImage(tensor=data.clone()),
            target=tio.ScalarImage(tensor=data.clone()),
        )

    def test_output_equals_a_single_ssot_application(self) -> None:
        import torch

        from spectramr.data.transforms.normalization import (
            ImageNormalizationSpec,
            ImageNormalizationTransform,
            normalize_tensor,
        )

        torch.manual_seed(0)
        data = torch.rand(1, 8, 8, 1) * 100.0
        spec = ImageNormalizationSpec.from_declared("percentile", "nifti_paired")
        out = ImageNormalizationTransform(spec)(self._subject(data))

        expected = normalize_tensor(data, spec.config)
        assert torch.allclose(out["input"].data, expected, atol=1e-6)
        assert torch.allclose(out["target"].data, expected, atol=1e-6)

    def test_a_second_pass_is_detectable(self) -> None:
        """The guard behind the guard: if applying twice were indistinguishable
        from once, the assertion above could not catch a re-introduced second
        normalizer. Percentile+clamp is NOT idempotent here, so it can."""
        import torch

        from spectramr.data.transforms.normalization import (
            ImageNormalizationSpec,
            ImageNormalizationTransform,
        )

        torch.manual_seed(0)
        data = torch.rand(1, 8, 8, 1) * 100.0
        spec = ImageNormalizationSpec.from_declared("percentile", "nifti", {"percentile": 0.5})
        tf = ImageNormalizationTransform(spec)
        once = tf(self._subject(data))["input"].data
        twice = tf(self._subject(once))["input"].data
        assert not torch.allclose(once, twice, atol=1e-6)

    def test_kspace_keys_are_left_alone(self) -> None:
        """Image normalizers clamp and shift, which destroys complex k-space."""
        import torch
        import torchio as tio

        from spectramr.data.transforms.normalization import (
            ImageNormalizationSpec,
            ImageNormalizationTransform,
        )

        data = torch.randn(2, 8, 8, 1) * 50.0
        subject = tio.Subject(kspace=tio.ScalarImage(tensor=data.clone()))
        spec = ImageNormalizationSpec.from_declared("percentile", "nifti")
        out = ImageNormalizationTransform(spec)(subject)
        assert torch.equal(out["kspace"].data, data)

    def test_the_source_is_stamped_onto_the_subject(self) -> None:
        """Which rule chose the numbers is what a reader needs when intensities
        move between two arms that look identically configured."""
        import torch

        from spectramr.data.transforms.normalization import (
            ImageNormalizationSpec,
            ImageNormalizationTransform,
        )

        spec = ImageNormalizationSpec.from_declared("none", "nifti_paired")
        out = ImageNormalizationTransform(spec)(self._subject(torch.rand(1, 4, 4, 1)))
        assert out["image_normalization_source"] == "contrast_aware_legacy_default"


class TestPercentileDividesRatherThanClips:
    """Plan item B5, fixed as a side effect of routing through the SSOT.

    ``tio.RescaleIntensity(percentiles=(0, p))`` CLIPS at the percentile before
    rescaling, while the builder's own comment described the operation as
    "divide by the 99th". With ``clamp=False`` the SSOT divides, so values above
    the percentile stay above 1 instead of being flattened onto it.
    """

    def test_values_above_the_percentile_are_not_flattened(self) -> None:
        import torch

        from spectramr.data.transforms.normalization import (
            ImageNormalizationSpec,
            ImageNormalizationTransform,
        )

        data = torch.cat([torch.ones(1, 1, 99, 1), torch.full((1, 1, 1, 1), 50.0)], 2)
        spec = ImageNormalizationSpec.from_declared(
            "percentile", "nifti", {"percentile": 0.9, "clamp": False}
        )
        out = ImageNormalizationTransform(spec)(
            TestImageNormalizationTransformAppliesOnce._subject(data)
        )
        assert out["input"].data.max() > 1.0, (
            "the bright voxel was clipped to the percentile — that is the "
            "RescaleIntensity behaviour this port replaced"
        )


class TestUnknownTypeRaisesInTheConfigVocabulary:
    """The unknown-``normalization_type`` raise moved here, and had to be re-worded.

    Two PRs collided on this. One made an unimplemented ``normalization_type``
    raise instead of warning-and-appending-nothing (#9), with a message quoting
    the CONFIG spellings. The other replaced both hand-written dispatch chains
    in the builder with this one resolver, which routed the raise through
    ``NormalizationStrategy.from_string`` -- whose message quotes the ENUM
    members. That regression is invisible to either PR alone: ``standard`` is
    the config spelling and ``zscore`` is the enum spelling, so the error told
    the author to write a value the schema Literal rejects.
    """

    @staticmethod
    def _resolve(value):
        from spectramr.data.transforms.normalization import ImageNormalizationSpec

        return ImageNormalizationSpec.from_declared(value, None, None)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown normalization_type"):
            self._resolve("scalar")

    def test_message_names_the_offending_value(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            self._resolve("scalar")
        assert "'scalar'" in str(excinfo.value)

    def test_message_quotes_the_config_spellings_not_the_enum_members(self) -> None:
        """``standard`` must appear; ``zscore`` is not writable in YAML."""
        from spectramr.data.transforms.normalization import (
            IMPLEMENTED_NORMALIZATION_TYPES,
        )

        with pytest.raises(ValueError) as excinfo:
            self._resolve("scalar")
        message = str(excinfo.value)
        for valid in IMPLEMENTED_NORMALIZATION_TYPES:
            assert repr(valid) in message, f"{valid!r} missing from the error text"
        assert "standard" in message

    def test_the_original_enum_error_is_chained(self) -> None:
        """Re-worded, not swallowed -- the root cause stays reachable."""
        with pytest.raises(ValueError) as excinfo:
            self._resolve("scalar")
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert "normalization strategy" in str(excinfo.value.__cause__)

    def test_the_builder_re_exports_the_same_object(self) -> None:
        """One definition. A copy in the builder is how the two sets drift."""
        from spectramr.data.builders import torchio_transform_builder
        from spectramr.data.transforms import normalization

        assert (
            torchio_transform_builder.IMPLEMENTED_NORMALIZATION_TYPES
            is normalization.IMPLEMENTED_NORMALIZATION_TYPES
        )

    def test_robust_percentile_is_not_in_the_set_but_still_resolves(self) -> None:
        """The one licensed gap: folded upstream, accepted here, not advertised."""
        from spectramr.data.transforms.normalization import (
            IMPLEMENTED_NORMALIZATION_TYPES,
            NormalizationStrategy,
        )

        assert "robust_percentile" not in IMPLEMENTED_NORMALIZATION_TYPES
        spec = self._resolve("robust_percentile")
        assert spec.config.strategy is NormalizationStrategy.PERCENTILE


class TestKSpaceNormalizationSurvivesPatchExtraction:
    """The transform's output must reach the batch, not just the Subject (#1213).

    ``tio.Subject`` mirrors its entries into ``self.__dict__`` and
    ``Subject.__setitem__`` is not defined, so the ``kspace_scale`` /
    ``kspace_log_scaled`` / ``kspace_normalized`` keys this transform publishes
    never reached ``__dict__``. ``tio.Crop`` — the engine behind every
    ``PatchSampler``, hence every ``tio.Queue`` — builds its output *solely* from
    ``__dict__``, so those keys were **dropped** at patch extraction.

    The consequence was not cosmetic. With no ``kspace_scale`` on the batch,
    ``DiffusionTrainingStrategy._batch_is_already_normalized`` took its
    ``kspace_scale is None`` route (the silent one, #1211), concluded the batch
    had never been normalized, and re-normalized with a scale of its own —
    discarding the declared percentile and domain while reporting success.
    """

    @staticmethod
    def _subject() -> tio.Subject:
        """A DC-heavy k-space Subject shaped like the M4Raw loader's output."""
        torch.manual_seed(0)
        k = torch.randn(2, 16, 16, 4) * 0.1
        k[:, 8, 8, :] = 2500.0  # DC blob: the ~200x range the transform tames
        return tio.Subject(input=tio.ScalarImage(tensor=k))

    def _normalized(self) -> tio.Subject:
        from spectramr.data.transforms.normalization import KSpaceNormalizationTransform

        return KSpaceNormalizationTransform(percentile=0.95, log_scaling=True)(self._subject())

    def test_the_markers_reach_the_patch(self) -> None:
        """A cropped subject still declares that it was normalized, and by how much."""
        normalized = self._normalized()
        assert normalized["kspace_normalized"] is True  # sanity: it ran

        patch = tio.Crop((0, 0, 0, 0, 0, 1))(normalized)

        for key in ("kspace_scale", "kspace_log_scaled", "kspace_normalized"):
            assert key in patch, (
                f"{key!r} dropped at patch extraction — the batch cannot tell the "
                "strategy it was already normalized (#1211/#1213)"
            )
        assert bool(patch["kspace_normalized"]) is True
        assert float(patch["kspace_scale"]) == pytest.approx(float(normalized["kspace_scale"]))

    def test_the_patch_carries_compressed_magnitudes_after_a_replacing_transform(
        self,
    ) -> None:
        """The patch's |k| is the transform's output, not the raw DC blob.

        Composed in the PRODUCTION order — ``EnsureSpatialConsistency`` first.
        That ordering is what made the image half of the defect reachable: this
        transform edits in place via ``Image.set_data``, which is safe on its own,
        but the chain's first member *replaces* every image object, so the tensor
        being edited was no longer the one ``__dict__`` (and therefore the crop)
        would read. A single-transform test cannot see this.

        A ``log1p``-compressed float32 magnitude cannot exceed ~44, so a patch
        anywhere near the input's 2500 proves the compression was discarded.
        """
        from spectramr.data.transforms.geometric import EnsureSpatialConsistency
        from spectramr.data.transforms.normalization import KSpaceNormalizationTransform

        chain = tio.Compose(
            [
                EnsureSpatialConsistency(),
                KSpaceNormalizationTransform(percentile=0.95, log_scaling=True),
            ]
        )
        normalized = chain(self._subject())
        patch = tio.Crop((0, 0, 0, 0, 0, 1))(normalized)

        assert float(patch["input"].data.abs().max()) < 44.0
        assert float(patch["input"].data.abs().max()) == pytest.approx(
            float(normalized["input"].data.abs().max())
        )


class TestAcsScaleDomain:
    """``scale_domain='acs'`` reads only the band every rung acquires, so the
    scale the dataloader stamps from the fully sampled reference is the scale
    inference computes from the undersampled measurement."""

    #: The cohort's R=32 rung at 256x256: the 5-line core plus 3 random lines.
    TOP_RUNG_ROWS = (72, 110, 126, 127, 128, 129, 130, 148)

    @staticmethod
    def _kspace() -> torch.Tensor:
        from spectramr.infrastructure.physics.fft_ops import fft2c

        g = torch.Generator().manual_seed(0)
        axis = torch.linspace(-1, 1, 256)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        image = (xx**2 + yy**2 < 0.6).float() * (1 + 0.3 * torch.rand(256, 256, generator=g))
        k = fft2c(image.to(torch.complex64).unsqueeze(0))
        return torch.stack([k.real, k.imag], dim=1).reshape(2, 256, 256).unsqueeze(-1)

    def _undersampled(self, k: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(256, dtype=torch.bool)
        mask[list(self.TOP_RUNG_ROWS)] = True
        return k * mask.view(1, 256, 1, 1)

    def test_the_band_is_centred_on_the_dc_row(self) -> None:
        from spectramr.data.transforms.normalization import acs_band_rows

        assert acs_band_rows(256, 0.02) == slice(126, 131)
        assert acs_band_rows(256, 0.0001) == slice(128, 129)

    @pytest.mark.parametrize(
        ("width", "band"),
        # int(W*f) would give 120:140 and 125:146: a line short, and offset by one.
        [(260, slice(120, 141)), (270, slice(124, 146))],
    )
    def test_the_band_width_rounds_as_the_masks_do(self, width: int, band: slice) -> None:
        from spectramr.data.transforms.normalization import acs_band_rows

        assert acs_band_rows(width, 0.08) == band

    def test_a_zero_floor_lets_a_zero_fraction_mean_no_band(self) -> None:
        from spectramr.data.transforms.normalization import acs_band_rows

        rows = acs_band_rows(256, 0.0, min_lines=0)
        assert rows.stop - rows.start == 0

    def test_acs_is_in_the_one_advertised_domain_set(self) -> None:
        from spectramr.data.transforms.normalization import (
            KSPACE_SCALE_DOMAINS,
            KSpaceNormalizationTransform,
        )

        assert "acs" in KSPACE_SCALE_DOMAINS
        # A second, narrower class-level set would misstate what the transform accepts.
        assert not hasattr(KSpaceNormalizationTransform, "SCALE_DOMAINS")
        transform = KSpaceNormalizationTransform(
            scale_domain="acs", center_fraction=0.08, phase_encode_axis=-1
        )
        assert transform.spec.scale_domain == "acs"

    @pytest.mark.parametrize(
        ("domain", "fraction", "log_scaling"),
        [("image", None, False), ("kspace", 0.25, True)],
    )
    def test_only_the_acs_scale_survives_undersampling(
        self, domain: str, fraction: float | None, log_scaling: bool
    ) -> None:
        """On this phantom ``image`` moves 1.7 % and the k-space patch 3.7x."""
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        k, u = self._kspace(), self._undersampled(self._kspace())
        acs = KSpaceNormalizationSpec(
            enabled=True,
            percentile=0.95,
            scale_domain="acs",
            center_fraction=0.02,
            phase_encode_axis=-2,
        )
        other = KSpaceNormalizationSpec(
            enabled=True,
            percentile=0.95,
            scale_domain=domain,
            center_fraction=fraction,
            log_scaling=log_scaling,
        )
        assert torch.equal(acs.compute_scale(k), acs.compute_scale(u))
        assert not torch.allclose(other.compute_scale(k), other.compute_scale(u), rtol=1e-3)

    def test_a_band_wider_than_the_acquired_core_is_not_invariant(self) -> None:
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        k = self._kspace()
        wide = KSpaceNormalizationSpec(
            enabled=True,
            percentile=0.95,
            scale_domain="acs",
            center_fraction=0.25,
            phase_encode_axis=-2,
        )
        assert not torch.equal(wide.compute_scale(k), wide.compute_scale(self._undersampled(k)))

    def test_batched_layout_matches_the_torchio_layout(self) -> None:
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        k = self._kspace()
        spec = KSpaceNormalizationSpec(
            enabled=True, scale_domain="acs", center_fraction=0.02, phase_encode_axis=-2
        )
        batched = k.squeeze(-1).unsqueeze(0)
        assert torch.allclose(spec.compute_scale(k), spec.compute_scale(batched, channel_dim=1))

    @pytest.mark.parametrize("fraction", [None, 0.0, 1.5])
    def test_acs_without_a_band_raises(self, fraction: float | None) -> None:
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        with pytest.raises(ValueError, match="scale_domain='acs'"):
            KSpaceNormalizationSpec(
                scale_domain="acs", center_fraction=fraction, phase_encode_axis=-2
            )

    def test_acs_without_a_phase_encode_axis_raises(self) -> None:
        """The band runs along the phase-encode axis; there is no default (#2160)."""
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        with pytest.raises(ValueError, match="phase_encode_axis"):
            KSpaceNormalizationSpec(scale_domain="acs", center_fraction=0.02)

    @staticmethod
    def _column_case() -> tuple[torch.Tensor, torch.Tensor]:
        """40x64 ``(C, H, W, D)`` k-space and its column-undersampled copy: the
        6-column core around DC plus three outer columns."""
        from spectramr.infrastructure.physics.fft_ops import fft2c

        g = torch.Generator().manual_seed(1)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 40), torch.linspace(-1, 1, 64), indexing="ij")
        image = (xx**2 + yy**2 < 0.6).float() * (1 + 0.3 * torch.rand(40, 64, generator=g))
        k = fft2c(image.to(torch.complex64).unsqueeze(0))
        k = torch.stack([k.real, k.imag], dim=1).reshape(2, 40, 64).unsqueeze(-1)
        keep = torch.zeros(64, dtype=torch.bool)
        keep[[5, 29, 30, 31, 32, 33, 34, 50]] = True
        return k, k * keep.view(1, 1, 64, 1)

    def test_the_band_follows_a_column_phase_encode_axis(self) -> None:
        """M4Raw / fastMRI brain: the mask keeps columns, so the band must be columns."""
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        k, u = self._column_case()
        columns = KSpaceNormalizationSpec(
            enabled=True, scale_domain="acs", center_fraction=0.05, phase_encode_axis=-1
        )
        assert torch.equal(columns.compute_scale(k), columns.compute_scale(u))
        batched = k.squeeze(-1).unsqueeze(0)
        assert torch.allclose(columns.compute_scale(k), columns.compute_scale(batched, channel_dim=1))

    def test_a_row_band_on_column_undersampled_data_is_not_invariant(self) -> None:
        """PLANTED: the pre-#2160 band (always rows) under a column mask."""
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        k, u = self._column_case()
        rows = KSpaceNormalizationSpec(
            enabled=True, scale_domain="acs", center_fraction=0.05, phase_encode_axis=-2
        )
        assert not torch.equal(rows.compute_scale(k), rows.compute_scale(u))

    def test_the_axis_is_read_from_data_acquisition(self) -> None:
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        data = {
            "processing": {
                "enable_kspace_normalization": True,
                "kspace_percentile": 0.99,
                "enable_log_scaling": False,
                "kspace_scale_domain": "acs",
                "log_scaling_center_fraction": 0.02,
            },
            "acquisition": {"phase_encode_axis": -1, "acquired_pe_lines": "measured"},
        }
        assert KSpaceNormalizationSpec.from_data_config(data).phase_encode_axis == -1


class TestKSpaceScaleReference:
    """``scale_reference`` picks the subject image the scale is measured on.

    The fixture's input is 10x dimmer than its target, so a test that read the
    wrong image would see a 10x different scale rather than pass by coincidence.
    """

    @staticmethod
    def _subject(*keys: str) -> tuple[tio.Subject, dict[str, torch.Tensor]]:
        torch.manual_seed(7)
        target = torch.randn(2, 8, 8, 1) * 30
        raw = {
            "kspace": target,
            "target": target * 2.0,
            "input": target * 0.1 + torch.randn(2, 8, 8, 1),
            "input_alt": target * 0.1 + torch.randn(2, 8, 8, 1),
        }
        raw = {k: raw[k] for k in keys}
        subject = tio.Subject(**{k: tio.ScalarImage(tensor=v.clone()) for k, v in raw.items()})
        return subject, raw

    @staticmethod
    def _transform(reference: str | None = None):
        from spectramr.data.transforms.normalization import KSpaceNormalizationTransform

        kwargs = {} if reference is None else {"scale_reference": reference}
        return KSpaceNormalizationTransform(percentile=0.95, log_scaling=False, **kwargs)

    @pytest.mark.parametrize(
        ("keys", "expected_ref"),
        [
            (("kspace", "target", "input"), "kspace"),
            (("target", "input"), "target"),
            (("input",), "input"),
        ],
        ids=["kspace-first", "target-before-input", "input-only-still-falls-through"],
    )
    def test_target_reproduces_the_historical_order(
        self, keys: tuple[str, ...], expected_ref: str
    ) -> None:
        """The default and the explicit value both keep the pre-knob reference."""
        for reference in (None, "target"):
            subject, raw = self._subject(*keys)
            transform = self._transform(reference)
            out = transform(subject)
            expected = transform.spec.compute_scale(raw[expected_ref], channel_dim=0)
            assert torch.equal(out["kspace_scale"], expected)

    def test_input_scales_every_image_by_the_inputs_quantile(self) -> None:
        subject, raw = self._subject("kspace", "input", "input_alt")
        transform = self._transform("input")
        out = transform(subject)
        expected = transform.spec.compute_scale(raw["input"], channel_dim=0)
        assert torch.equal(out["kspace_scale"], expected)
        target_scale = transform.spec.compute_scale(raw["kspace"], channel_dim=0)
        assert not torch.allclose(expected, target_scale, rtol=0.5)
        for key in ("kspace", "input", "input_alt"):
            assert torch.allclose(out[key].data * expected, raw[key], rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("reference", ["target", "input"])
    def test_a_published_measurement_sets_the_input_scale_and_is_never_served(
        self, reference: str
    ) -> None:
        """``input`` recorrupted as ``y + a*z`` with ``a*z`` large; ``y`` published.

        Under "input" the scale is ``y``'s; under "target" the key is ignored.
        Either way it is removed, so it never reaches a batch.
        """
        from spectramr.data.transforms.normalization import MEASURED_INPUT_KEY

        subject, raw = self._subject("kspace", "input")
        measured = raw["input"].clone()
        recorrupted = measured + 500.0 * torch.randn_like(measured)
        subject["input"].set_data(recorrupted.clone())
        subject.add_image(tio.ScalarImage(tensor=measured.clone()), MEASURED_INPUT_KEY)
        transform = self._transform(reference)
        out = transform(subject)
        ref = measured if reference == "input" else raw["kspace"]
        assert torch.equal(out["kspace_scale"], transform.spec.compute_scale(ref, channel_dim=0))
        y_az = transform.spec.compute_scale(recorrupted, channel_dim=0)
        assert not torch.allclose(out["kspace_scale"], y_az, rtol=0.5)
        assert MEASURED_INPUT_KEY not in out
        assert torch.allclose(out["input"].data * out["kspace_scale"], recorrupted, rtol=1e-4)

    def test_only_an_input_referenced_normalizer_consumes_the_measurement(self) -> None:
        """The dataset publishes the key on this answer, so it must not over-claim."""
        from spectramr.data.transforms.normalization import consumes_measured_input

        assert consumes_measured_input(tio.Compose([tio.Lambda(lambda x: x), self._transform("input")]))
        assert consumes_measured_input(self._transform("input"))
        assert not consumes_measured_input(self._transform("target"))
        assert not consumes_measured_input(tio.Compose([tio.Lambda(lambda x: x)]))
        assert not consumes_measured_input(None)

    def test_input_without_an_input_image_raises_by_name(self) -> None:
        """PLANTED: no silent fall-through to the target (pitfall #9)."""
        subject, _ = self._subject("kspace", "target")
        with pytest.raises(KeyError, match=r"kspace_scale_reference.*'input'"):
            self._transform("input")(subject)

    def test_an_unknown_reference_is_rejected(self) -> None:
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        with pytest.raises(ValueError, match="Unknown scale_reference"):
            self._transform("kspace")
        with pytest.raises(ValueError, match="Unknown scale_reference"):
            KSpaceNormalizationSpec(scale_reference="measured")

    def test_the_resolver_reads_the_declared_reference(self) -> None:
        """A stored config predating the knob resolves to what it trained under."""
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        processing = {
            "enable_kspace_normalization": True,
            "kspace_percentile": 0.99,
            "enable_log_scaling": False,
            "kspace_scale_domain": "image",
            "log_scaling_center_fraction": 0.25,
        }
        stored = KSpaceNormalizationSpec.from_data_config({"processing": processing})
        assert stored.scale_reference == "target"
        declared = {**processing, "kspace_scale_reference": "input"}
        spec = KSpaceNormalizationSpec.from_data_config({"processing": declared})
        assert spec.scale_reference == "input"


class TestTheAcsWindowScaleSurvivesA2DMask:
    """The sweep's Poisson-disc and radial masks keep a 20 x 20 window, not whole lines."""

    @staticmethod
    def _scene() -> tuple[torch.Tensor, torch.Tensor]:
        from spectramr.data.transforms.rep_mask_patterns import calibration_region
        from spectramr.infrastructure.physics.fft_ops import fft2c

        gen = torch.Generator().manual_seed(3)
        image = torch.zeros(4, 64, 64, 1, dtype=torch.complex64)
        image[:, 16:48, 20:44] = torch.complex(torch.rand(4, 32, 24, 1, generator=gen), torch.rand(4, 32, 24, 1, generator=gen))
        kspace = fft2c(image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        mask = (torch.rand(64, 64, generator=gen) < 0.15) | calibration_region(64, 64, 0.125)
        return kspace, kspace * mask[None, :, :, None]

    def _spec(self, domain: str):
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        return KSpaceNormalizationSpec(
            enabled=True, percentile=0.95, scale_domain=domain, center_fraction=0.125,
            phase_encode_axis=-1,
        )

    def test_the_window_scale_is_the_same_on_the_reference_and_the_mask(self) -> None:
        full, masked = self._scene()
        spec = self._spec("acs_window")
        assert torch.equal(spec.compute_scale(full), spec.compute_scale(masked))

    def test_the_line_band_is_not(self) -> None:
        """Planted: whole phase-encode lines are not inside a 2-D mask."""
        full, masked = self._scene()
        band = self._spec("acs")
        assert not torch.allclose(band.compute_scale(full), band.compute_scale(masked), rtol=1e-3)

    def test_the_masks_and_the_scale_share_one_window(self) -> None:
        from spectramr.data.transforms.normalization import acs_window
        from spectramr.data.transforms.rep_mask_patterns import calibration_region

        for h, w, f in ((256, 256, 0.08), (255, 200, 0.1), (64, 64, 0.0)):
            n = round(w * f)
            old = torch.zeros(h, w, dtype=torch.bool)
            if n:
                old[h // 2 - n // 2 : h // 2 - n // 2 + n, w // 2 - n // 2 : w // 2 - n // 2 + n] = True
            assert torch.equal(calibration_region(h, w, f), old)
            rows, cols = acs_window(h, w, f)
            assert (rows.stop - rows.start, cols.stop - cols.start) == (n, n)

    def test_acs_window_without_a_fraction_raises(self) -> None:
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        with pytest.raises(ValueError, match="acs_window"):
            KSpaceNormalizationSpec(enabled=True, scale_domain="acs_window", center_fraction=None)
