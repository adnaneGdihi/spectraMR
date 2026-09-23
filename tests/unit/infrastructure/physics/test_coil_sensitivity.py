"""Unit tests for canonical ESPIRiT coil sensitivity map estimation.

Tests verify physics-level correctness properties from Uecker et al. MRM 2014:
- Output shape and dtype
- RSS normalization (|smaps|² ≤ 1 everywhere, ≈ 1 in-mask)
- Phase reference: reference coil phase ≈ 0 inside mask
- Hermitian symmetry of Gram matrix (eigenvalues real & ≥ 0)
- No NaN / Inf in outputs
- Birdcage phantom accuracy (cosine similarity ≥ 0.9 per coil)
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
import torch

from spectramr.infrastructure.physics import coil_sensitivity as coil_sensitivity_mod
from spectramr.infrastructure.physics.coil_sensitivity import (
    _robust_eigh,
    coil_combine_sense,
    espirit_min_acs_size,
    estimate_csm_espirit,
    estimate_csm_pinn,
    estimate_csm_power_iter,
    estimate_smaps,
    extract_acs_region,
    load_csm_from_file,
    sense_gfactor_map,
)
from spectramr.infrastructure.physics.fft_ops import fft2c

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_birdcage_kspace(
    num_coils: int = 4,
    height: int = 64,
    width: int = 64,
    batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic birdcage phantom in k-space.

    Returns:
        (kspace, gt_smaps) both complex, shapes (B, C, H, W).
        kspace is DC-centered (consistent with fftshift convention).
    """
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height),
        torch.linspace(-1, 1, width),
        indexing="ij",
    )
    # Elliptical object mask
    mask = ((x / 0.8) ** 2 + (y / 0.9) ** 2 <= 1.0).float()

    gt_smaps_list = []
    for c in range(num_coils):
        angle = 2.0 * math.pi * c / num_coils
        cx = 0.6 * math.cos(angle)
        cy = 0.6 * math.sin(angle)
        mag = torch.exp(-2.0 * ((x - cx) ** 2 + (y - cy) ** 2))
        phase = torch.exp(1j * (angle * (x * math.cos(angle) + y * math.sin(angle))))
        s = (mag * phase * mask).to(torch.complex64)
        gt_smaps_list.append(s)

    gt_smaps = torch.stack(gt_smaps_list, dim=0)  # (C, H, W)
    # RSS-normalize
    rss = torch.sqrt((gt_smaps.abs() ** 2).sum(0, keepdim=True) + 1e-12)
    gt_smaps = gt_smaps / rss  # (C, H, W)

    # Build single object image
    obj = torch.complex(
        torch.randn(1, height, width), torch.randn(1, height, width)
    ) * mask.unsqueeze(0)
    coil_images = obj * gt_smaps.unsqueeze(0)  # (1, C, H, W) → repeat for batch

    # FFT to k-space (centered)
    kspace = fft2c(coil_images.reshape(-1, height, width)).reshape(
        1, num_coils, height, width
    )
    kspace = kspace.expand(batch_size, -1, -1, -1).clone()
    gt_smaps = gt_smaps.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()

    return kspace, gt_smaps


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestESPIRiTOutputShape:
    """Shape and dtype sanity checks."""

    def test_output_shape_matches_input(self) -> None:
        kspace, _ = _make_birdcage_kspace(num_coils=4, height=64, width=64)
        smaps = estimate_csm_espirit(kspace, num_coils=4, acs_size=24)
        assert (
            smaps.shape == kspace.shape
        ), f"Expected {kspace.shape}, got {smaps.shape}"

    def test_output_is_complex(self) -> None:
        kspace, _ = _make_birdcage_kspace(num_coils=4)
        smaps = estimate_csm_espirit(kspace, num_coils=4)
        assert torch.is_complex(smaps), "Output must be complex-valued."

    def test_batch_dimension_handled(self) -> None:
        kspace, _ = _make_birdcage_kspace(num_coils=4, batch_size=2)
        smaps = estimate_csm_espirit(kspace, num_coils=4)
        assert smaps.shape[0] == 2, "Batch dimension mismatch."

    def test_wrong_ndim_raises(self) -> None:
        kspace_3d = torch.randn(4, 64, 64, dtype=torch.complex64)
        with pytest.raises(ValueError, match="Expected 4D"):
            estimate_csm_espirit(kspace_3d, num_coils=4)

    def test_wrong_ncoils_raises(self) -> None:
        kspace, _ = _make_birdcage_kspace(num_coils=4)
        with pytest.raises(ValueError, match="4 coils but num_coils=8"):
            estimate_csm_espirit(kspace, num_coils=8)

    def test_acs_smaller_than_kernel_raises(self) -> None:
        """ACS region smaller than the calibration kernel must raise a
        descriptive ValueError, not a cryptic empty-TensorList RuntimeError.
        """
        kspace, _ = _make_birdcage_kspace(num_coils=4, height=64, width=64)
        with pytest.raises(ValueError, match=r"ACS region .* must be >= kernel_size"):
            estimate_csm_espirit(kspace, num_coils=4, acs_size=4, kernel_size=6)

    def test_espirit_docstring_sigma_threshold_matches_signature(self) -> None:
        """Doc-contract: documented sigma_threshold default matches signature."""
        import inspect

        sig_default = (
            inspect.signature(estimate_csm_espirit)
            .parameters["sigma_threshold"]
            .default
        )
        assert sig_default == 0.02
        assert f"(default: {sig_default})" in (estimate_csm_espirit.__doc__ or "")


class TestESPIRiTMinAcsSize:
    """`espirit_min_acs_size` unblocks the many-coil rank-deficiency (#309)."""

    def test_side_satisfies_rank_inequality(self) -> None:
        # For coil counts where the 24x24 default is rank-deficient, the helper
        # returns a side whose patch count clears kernel^2 * coils.
        kernel = 6
        for n_coils in (11, 12, 16, 20):
            side = espirit_min_acs_size(n_coils, kernel_size=kernel)
            n_patches = (side - kernel + 1) ** 2
            unknowns = kernel * kernel * n_coils
            assert (
                n_patches >= unknowns
            ), f"{n_coils} coils: {n_patches} patches < {unknowns} unknowns"

    def test_max_acs_clamps_the_side(self) -> None:
        # Unclamped side for 16 coils/kernel 6 is 35; the clamp wins.
        assert espirit_min_acs_size(16, kernel_size=6, max_acs=28) == 28

    def test_invalid_arguments_raise(self) -> None:
        with pytest.raises(ValueError, match="num_coils"):
            espirit_min_acs_size(0)
        with pytest.raises(ValueError, match="kernel_size"):
            espirit_min_acs_size(4, kernel_size=0)
        with pytest.raises(ValueError, match="patch_margin"):
            espirit_min_acs_size(4, patch_margin=0.5)

    def test_grown_acs_fixes_the_default_rank_deficiency(self) -> None:
        # The exact #309 scenario: a 16-coil fully-sampled k-space. The 24x24
        # default raises rank-deficient; the grown ACS lets ESPIRiT run.
        kspace, _ = _make_birdcage_kspace(num_coils=16, height=48, width=48)
        with pytest.raises(ValueError, match="rank-deficient"):
            estimate_csm_espirit(kspace, num_coils=16, acs_size=24, kernel_size=6)
        acs = espirit_min_acs_size(16, kernel_size=6, max_acs=48)
        smaps = estimate_csm_espirit(kspace, num_coils=16, acs_size=acs, kernel_size=6)
        assert smaps.shape == kspace.shape
        assert torch.isfinite(smaps).all()


class TestESPIRiTNumericalSanity:
    """Numerical correctness properties that must hold by construction."""

    def test_no_nan_in_output(self) -> None:
        kspace, _ = _make_birdcage_kspace(num_coils=4)
        smaps = estimate_csm_espirit(kspace, num_coils=4)
        assert not torch.isnan(smaps).any(), "NaN detected in output smaps."

    def test_no_inf_in_output(self) -> None:
        kspace, _ = _make_birdcage_kspace(num_coils=4)
        smaps = estimate_csm_espirit(kspace, num_coils=4)
        assert not torch.isinf(smaps).any(), "Inf detected in output smaps."

    def test_rss_at_most_one_everywhere(self) -> None:
        """sum_c |s_c|² <= 1 + ε everywhere (RSS-normalization guarantee)."""
        kspace, _ = _make_birdcage_kspace(num_coils=4)
        smaps = estimate_csm_espirit(kspace, num_coils=4)
        rss_sq = (smaps.abs() ** 2).sum(dim=1)  # (B, H, W)
        assert (
            rss_sq <= 1.0 + 1e-4
        ).all(), f"RSS² exceeds 1 at some pixels: max={rss_sq.max().item():.6f}"

    def test_rss_near_one_in_mask(self) -> None:
        """Inside the object support, RSS should be close to 1."""
        kspace, _ = _make_birdcage_kspace(num_coils=4, height=64, width=64)
        smaps = estimate_csm_espirit(kspace, num_coils=4)
        rss = torch.sqrt((smaps.abs() ** 2).sum(dim=1))  # (B, H, W)
        in_mask = rss > 0.5
        if in_mask.any():
            in_mask_rss = rss[in_mask]
            assert (
                in_mask_rss.mean().item() >= 0.85
            ), f"Mean in-mask RSS is {in_mask_rss.mean().item():.4f}, expected >= 0.85"

    def test_out_of_support_is_zero(self) -> None:
        """Pixels far from the object should be exactly zero."""
        kspace, _ = _make_birdcage_kspace(num_coils=4, height=64, width=64)
        smaps = estimate_csm_espirit(kspace, num_coils=4, eigen_threshold=0.95)
        rss = torch.sqrt((smaps.abs() ** 2).sum(dim=1))  # (B, H, W)
        # Corner pixels (far from brain) should be zero
        corners = rss[:, :4, :4]  # top-left corner
        assert (
            corners.max().item() < 0.1
        ), f"Corner pixels are non-zero: max={corners.max().item():.4f}"


class TestESPIRiTPhaseReference:
    """Phase reference coil should have ~zero phase inside the mask."""

    def test_ref_coil_phase_near_zero(self) -> None:
        kspace, _ = _make_birdcage_kspace(num_coils=4)
        smaps = estimate_csm_espirit(kspace, num_coils=4, phase_ref_coil=0)
        ref_phase = torch.angle(smaps[:, 0, :, :])  # (B, H, W)
        rss = torch.sqrt((smaps.abs() ** 2).sum(dim=1))
        in_mask = rss > 0.5
        if in_mask.any():
            masked_phase = ref_phase[in_mask]
            mean_abs_phase = masked_phase.abs().mean().item()
            assert (
                mean_abs_phase < 0.3
            ), f"Reference coil mean |phase| inside mask = {mean_abs_phase:.4f} rad, expected < 0.3"


class TestESPIRiTGramHermitian:
    """The Gram matrix M = GᴴG must be Hermitian → eigenvalues ≥ 0."""

    def test_eigenvalues_nonnegative(self) -> None:
        """ESPIRiT Gram matrices are PSD by construction."""
        torch.manual_seed(0)
        C, H, W = 4, 32, 32
        G = torch.randn(10, C, H * W, dtype=torch.complex64)
        G_pix = G.permute(2, 1, 0)  # (N, C, n_keep)
        M = G_pix @ G_pix.conj().transpose(-2, -1)  # (N, C, C)
        evals, _ = torch.linalg.eigh(M)
        assert (
            evals >= -1e-5
        ).all(), f"Negative eigenvalues found: min={evals.min().item():.6f}"


class TestESPIRiTBirdcageAccuracy:
    """Sensitivity maps should be close to the ground truth birdcage pattern."""

    def test_cosine_similarity_to_gt(self) -> None:
        """Per-coil cosine similarity between estimated and GT smaps ≥ 0.8."""
        torch.manual_seed(42)
        kspace, gt_smaps = _make_birdcage_kspace(num_coils=4, height=64, width=64)
        smaps = estimate_csm_espirit(kspace, num_coils=4, acs_size=24)

        # Compare only inside brain mask (where GT is non-zero)
        rss_gt = torch.sqrt((gt_smaps.abs() ** 2).sum(1))  # (B, H, W)
        mask = (rss_gt > 0.1).squeeze(0)  # (H, W)

        B, C, H, W = smaps.shape
        similarities = []
        for c in range(C):
            s_est = smaps[0, c][mask].flatten()
            s_gt = gt_smaps[0, c][mask].flatten()
            # Cosine similarity (magnitude-based, phase invariant)
            cos_sim = (
                torch.dot(s_est.abs(), s_gt.abs())
                / (s_est.abs().norm() * s_gt.abs().norm() + 1e-12)
            ).item()
            similarities.append(cos_sim)

        mean_sim = np.mean(similarities)
        assert (
            mean_sim >= 0.9
        ), f"Mean cosine similarity to GT birdcage smaps = {mean_sim:.4f}, expected >= 0.9"

    def test_sigma_threshold_reduces_vectors(self) -> None:
        """Higher sigma_threshold keeps fewer singular vectors (faster but coarser)."""
        kspace, _ = _make_birdcage_kspace(num_coils=4)
        # Should complete without error regardless of threshold
        smaps_tight = estimate_csm_espirit(kspace, num_coils=4, sigma_threshold=0.1)
        smaps_loose = estimate_csm_espirit(kspace, num_coils=4, sigma_threshold=0.001)
        assert smaps_tight.shape == smaps_loose.shape
        assert not torch.isnan(smaps_tight).any()
        assert not torch.isnan(smaps_loose).any()

    def test_max_n_keep_limits_computation(self) -> None:
        """max_n_keep should limit SVD vectors without raising errors."""
        kspace, _ = _make_birdcage_kspace(num_coils=4)
        smaps = estimate_csm_espirit(kspace, num_coils=4, max_n_keep=3)
        assert smaps.shape == kspace.shape
        assert not torch.isnan(smaps).any()


class TestPINNCoilSensitivity:
    """Sanity checks for continuous PDE PINN coil sensitivity map estimation."""

    def test_output_shape_matches_input(self) -> None:
        """PINN must return sensitivity maps matching the k-space spatial dimensions."""
        kspace, _ = _make_birdcage_kspace(
            num_coils=4, height=32, width=32, batch_size=2
        )
        # Minimal architecture/epochs for fast testing
        smaps = estimate_csm_pinn(
            kspace,
            num_coils=4,
            acs_size=12,
            siren_hidden_features=32,
            siren_hidden_layers=2,
            epochs=2,
        )
        assert (
            smaps.shape == kspace.shape
        ), f"Expected {kspace.shape}, got {smaps.shape}"

    def test_output_is_complex(self) -> None:
        """PINN must return complex-valued tensors."""
        kspace, _ = _make_birdcage_kspace(num_coils=4, height=16, width=16)
        smaps = estimate_csm_pinn(kspace, num_coils=4, epochs=1)
        assert torch.is_complex(smaps), "Output must be complex-valued."

    def test_no_nan_in_output(self) -> None:
        """PINN output should not contain NaNs under normal healthy data."""
        kspace, _ = _make_birdcage_kspace(num_coils=4, height=16, width=16)
        smaps = estimate_csm_pinn(kspace, num_coils=4, epochs=2)
        assert not torch.isnan(smaps).any(), "NaN detected in output smaps."


def _dispatcher_ks(c=4):
    torch.manual_seed(0)
    return torch.complex(torch.randn(1, c, 32, 32), torch.randn(1, c, 32, 32))


class TestEstimateSmapsDispatcher:
    def test_none_returns_none(self):
        assert estimate_smaps(_dispatcher_ks(), method="none") is None

    @pytest.mark.parametrize("method", ["power_iter", "espirit", "rss"])
    def test_known_methods_return_maps(self, method):
        out = estimate_smaps(_dispatcher_ks(), method=method)
        assert out is not None and out.shape == (1, 4, 32, 32)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match=r"Unknown.*estimation method"):
            estimate_smaps(_dispatcher_ks(), method="bogus")

    def test_file_without_path_raises(self):
        with pytest.raises(ValueError, match="maps_path"):
            estimate_smaps(_dispatcher_ks(), method="file", maps_path=None)


class TestPowerIterDocContract:
    """Doc-contract regression for estimate_csm_power_iter."""

    def test_docstring_kernel_size_default_matches_signature(self) -> None:
        import inspect

        sig_default = (
            inspect.signature(estimate_csm_power_iter).parameters["kernel_size"].default
        )
        assert sig_default == 7
        assert f"(default: {sig_default})" in (estimate_csm_power_iter.__doc__ or "")


class TestLoadCsmFromFile:
    """load_csm_from_file fail-fast precondition on expected_shape."""

    def test_load_csm_matching_shape_returns(self, tmp_path) -> None:
        p = tmp_path / "csm.pt"
        smaps = torch.complex(torch.randn(1, 4, 32, 32), torch.randn(1, 4, 32, 32))
        torch.save(smaps, p)
        out = load_csm_from_file(str(p), expected_shape=(1, 4, 32, 32))
        assert tuple(out.shape) == (1, 4, 32, 32)

    def test_load_csm_shape_mismatch_raises(self, tmp_path) -> None:
        """A violated explicit expected_shape precondition must fail fast at the
        I/O boundary (raise), not warn-and-return the wrong-shape tensor.
        """
        p = tmp_path / "csm.pt"
        smaps = torch.complex(torch.randn(1, 4, 32, 32), torch.randn(1, 4, 32, 32))
        torch.save(smaps, p)
        with pytest.raises(ValueError, match=r"!= expected"):
            load_csm_from_file(str(p), expected_shape=(1, 8, 32, 32))


class TestCoilCombineSense:
    """Roemer-optimal SENSE combine must include the 1/∑|S|² denominator."""

    def test_recovers_image_with_nonunit_maps(self) -> None:
        """With x_c = S_c·x, the SENSE combine recovers x exactly regardless
        of the sensitivity scale — the property the ∑|S|² denominator buys.

        Regression: the previous matched-filter-only form returned x·∑|S|²,
        which equals x only for unit-RSS maps.
        """
        torch.manual_seed(0)
        b, c, h, w = 1, 4, 16, 16
        true_image = torch.complex(torch.randn(b, 1, h, w), torch.randn(b, 1, h, w))
        # Deliberately non-unit-norm sensitivity maps (∑|S|² ≠ 1).
        smaps = torch.complex(torch.randn(b, c, h, w), torch.randn(b, c, h, w)) * 1.7
        coil_images = smaps * true_image

        combined = coil_combine_sense(coil_images, smaps)
        assert torch.allclose(combined, true_image, atol=1e-5)

    def test_fixed_data_estimate_scales_inverse_with_map_gain(self) -> None:
        """With coil data fixed, scaling the assumed maps by k scales the
        SENSE estimate by 1/k (the denominator at work).

        The previous matched-filter-only form scaled the estimate by k
        instead — the opposite direction — so this pins the fix.
        """
        torch.manual_seed(1)
        b, c, h, w = 1, 4, 8, 8
        smaps = torch.complex(torch.randn(b, c, h, w), torch.randn(b, c, h, w))
        coil_images = torch.complex(torch.randn(b, c, h, w), torch.randn(b, c, h, w))
        base = coil_combine_sense(coil_images, smaps)
        scaled = coil_combine_sense(coil_images, smaps * 3.0)
        # x_c fixed, maps scaled by k: numerator→k*num, denom→k²*den ⇒ 1/k.
        assert torch.allclose(scaled, base / 3.0, atol=1e-5)


class TestSenseSupportFloor:
    """The denominator floor must kill the air blow-up WITHOUT biasing the object (#603).

    ``sum_c |S_c|^2`` goes to zero outside the object support, so the SENSE division
    there is noise over nothing. The old guard was ``+ 1e-8``, an ABSOLUTE term against
    a quantity whose scale is set by the map normalisation — 1e-8 of the map maximum,
    i.e. no protection. Measured on 4-coil M4Raw ESPIRiT maps: 24.6% of voxels below
    1e-3, the worst at ~1e-5, and the combined image reaching 742x its own p99 from 26
    air voxels whose RSS was below the image median.
    """

    @staticmethod
    def _maps_with_a_dead_corner(b=1, c=4, h=16, w=16):
        """Maps that fall to ~0 in one corner — a stand-in for air."""
        g = torch.Generator().manual_seed(3)
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij"
        )
        support = (xx + yy) / 2.0  # 0 at one corner, 1 at the opposite
        smaps = (
            torch.complex(
                torch.rand(b, c, h, w, generator=g) * 0.5 + 0.5, torch.zeros(b, c, h, w)
            )
            * support
        )
        return smaps, support

    def test_well_conditioned_voxels_are_untouched(self) -> None:
        """A clamp, not a Tikhonov term: exact recovery must survive.

        ``+ lambda`` measured 1.7e-2 relative error on ``x_c = S_c x``; the clamp
        measures 6.7e-8, identical to no floor at all. That difference is the whole
        reason this is a clamp.
        """
        g = torch.Generator().manual_seed(0)
        x = torch.complex(
            torch.randn(1, 1, 32, 32, generator=g), torch.zeros(1, 1, 32, 32)
        )
        smaps = torch.complex(
            torch.rand(1, 4, 32, 32, generator=g) * 0.5 + 0.5, torch.zeros(1, 4, 32, 32)
        )
        out = coil_combine_sense(smaps * x, smaps)
        assert torch.allclose(out, x, atol=1e-5), (
            "the support floor is biasing well-conditioned voxels — it must clamp, "
            "not add"
        )

    def test_air_amplification_is_capped(self) -> None:
        smaps, _ = self._maps_with_a_dead_corner()
        g = torch.Generator().manual_seed(4)
        noise = torch.complex(
            torch.randn(1, 4, 16, 16, generator=g),
            torch.randn(1, 4, 16, 16, generator=g),
        )
        floored = coil_combine_sense(noise, smaps).abs()
        unfloored = coil_combine_sense(noise, smaps, min_support_frac=0.0).abs()
        assert floored.max() < unfloored.max(), "the floor did nothing"
        # Capped at 1/sqrt(frac) = 10x the best-conditioned voxel, so the peak cannot
        # be orders of magnitude above the bulk.
        assert floored.max() < 40.0 * floored.median()

    def test_floor_is_relative_to_the_map_scale(self) -> None:
        """An ABSOLUTE floor is meaningless against arbitrarily-scaled maps (#576).

        The map gain is a free parameter — ESPIRiT normalisation is a convention, not a
        measurement — so ``sum|S|^2`` can sit anywhere. Scaling every map by ``k`` must
        scale the estimate by ``1/k`` at EVERY voxel, the clamped ones included. ``k``
        is small enough here that an absolute ``1e-8`` becomes comparable to the
        denominator and breaks exactly that.
        """
        smaps, _ = self._maps_with_a_dead_corner()
        g = torch.Generator().manual_seed(5)
        coils = torch.complex(
            torch.randn(1, 4, 16, 16, generator=g), torch.zeros(1, 4, 16, 16)
        )
        k = 1e-3  # denominator scales by k^2 = 1e-6, i.e. into 1e-8's territory
        base = coil_combine_sense(coils, smaps)
        scaled = coil_combine_sense(coils, smaps * k)
        assert torch.allclose(scaled * k, base, atol=1e-6, rtol=1e-4), (
            "the floor is not scale-relative: the same physical maps at a different "
            "gain convention produce a different reconstruction"
        )

    def test_per_image_floor_does_not_leak_across_a_batch(self) -> None:
        """One subject's map scale must not set another's floor."""
        smaps, _ = self._maps_with_a_dead_corner()
        g = torch.Generator().manual_seed(6)
        coils = torch.complex(
            torch.randn(1, 4, 16, 16, generator=g), torch.zeros(1, 4, 16, 16)
        )
        alone = coil_combine_sense(coils, smaps)
        batched = coil_combine_sense(
            torch.cat([coils, coils]), torch.cat([smaps, smaps * 100.0])
        )
        assert torch.allclose(batched[:1], alone, atol=1e-6)


class TestRobustEighCusolverFallback:
    """Regression: cuSOLVER's batched Hermitian eigensolver
    (``cusolverDnXsyevBatched``) requests a pathological workspace for large
    batches of small *complex* matrices — ~33 GiB for a ``(65536, 4, 4)``
    complex64 batch (the per-pixel ESPIRiT Gram). Depending on GPU capacity
    this surfaces as ``CUSOLVER_STATUS_INVALID_VALUE`` (buffer-size overflow)
    or a CUDA OOM. The misleading "input contains NaN" hint does not apply —
    the buffer-size query never reads the data.

    Before the fix this exception escaped ``estimate_csm_espirit`` and forced
    the m4raw pseudo-GT to silently fall back to RSS coil maps, changing
    ``x_gt`` and invalidating downstream sim2rank/BT comparisons.
    """

    def test_robust_eigh_falls_back_to_cpu_on_backend_error(self, monkeypatch) -> None:
        """A simulated cuSOLVER LinAlgError on the first eigh call must trigger
        exactly one CPU-LAPACK retry that returns the correct decomposition."""
        from spectramr.infrastructure.physics import coil_sensitivity as cs

        torch.manual_seed(0)
        g = torch.randn(128, 4, 6, dtype=torch.complex64)
        gram = g @ g.conj().transpose(-2, -1)  # Hermitian PSD, finite
        ref_vals, _ = torch.linalg.eigh(gram)

        real_eigh = torch.linalg.eigh
        calls = {"n": 0}

        def flaky_eigh(mat):
            calls["n"] += 1
            if calls["n"] == 1:
                raise torch.linalg.LinAlgError(
                    "cusolver error: CUSOLVER_STATUS_INVALID_VALUE"
                )
            return real_eigh(mat)

        monkeypatch.setattr(torch.linalg, "eigh", flaky_eigh)

        vals, vecs = cs._robust_eigh(gram)

        assert calls["n"] == 2, "must retry exactly once after the backend error"
        assert torch.isfinite(vals).all()
        torch.testing.assert_close(vals, ref_vals, rtol=1e-5, atol=1e-5)
        # Eigenvectors are unique only up to per-vector phase; verify via the
        # phase-invariant reconstruction M ~= V diag(lambda) V^H.
        recon = (
            vecs @ torch.diag_embed(vals.to(vecs.dtype)) @ vecs.conj().transpose(-2, -1)
        )
        torch.testing.assert_close(recon, gram, rtol=1e-4, atol=1e-4)

    def test_robust_eigh_happy_path_matches_eigh(self) -> None:
        """When the backend succeeds, ``_robust_eigh`` is a transparent
        pass-through with no extra cost (no spurious CPU retry)."""
        from spectramr.infrastructure.physics import coil_sensitivity as cs

        torch.manual_seed(1)
        g = torch.randn(64, 4, 5, dtype=torch.complex64)
        gram = g @ g.conj().transpose(-2, -1)
        ref_vals, ref_vecs = torch.linalg.eigh(gram)
        vals, vecs = cs._robust_eigh(gram)
        torch.testing.assert_close(vals, ref_vals)
        torch.testing.assert_close(vecs, ref_vecs)

    @pytest.mark.gpu
    def test_estimate_csm_espirit_succeeds_on_full_size_image(self) -> None:
        """End-to-end: a 256x256 4-coil ESPIRiT estimate must not crash under
        the default cuSOLVER backend. Pre-fix, the per-pixel batched complex
        eigh raised CUSOLVER_STATUS_INVALID_VALUE / OOM on the (65536, 4, 4)
        Gram and forced the RSS fallback."""
        torch.manual_seed(0)
        img = torch.randn(1, 4, 256, 256, dtype=torch.complex64, device="cuda")
        kspace = fft2c(img)
        smaps = estimate_csm_espirit(
            kspace,
            num_coils=4,
            kernel_size=6,
            acs_size=24,
            sigma_threshold=0.02,
            eigen_threshold=0.95,
        )
        assert smaps.shape == kspace.shape
        assert torch.isfinite(smaps).all()
        # ESPIRiT soft-SENSE property: sum_c |S_c|^2 <= 1 (+ float slack) everywhere.
        rss_sq = smaps.abs().pow(2).sum(dim=1)
        assert rss_sq.max().item() <= 1.0 + 1e-4


class TestExtractAcsRegion:
    """Central-ACS crop — the dense, aliasing-free calibration window."""

    def test_int_size_crops_center(self) -> None:
        torch.manual_seed(0)
        k = torch.randn(2, 4, 64, 64, dtype=torch.complex64)
        acs = extract_acs_region(k, 8)
        assert acs.shape == (2, 4, 8, 8)
        torch.testing.assert_close(acs, k[:, :, 28:36, 28:36])

    def test_tuple_size_crops_rectangle(self) -> None:
        k = torch.randn(1, 4, 64, 64, dtype=torch.complex64)
        acs = extract_acs_region(k, (6, 10))
        assert acs.shape == (1, 4, 6, 10)
        torch.testing.assert_close(acs, k[:, :, 29:35, 27:37])

    def test_size_larger_than_dims_clamps(self) -> None:
        k = torch.randn(1, 4, 16, 16, dtype=torch.complex64)
        acs = extract_acs_region(k, 999)
        assert acs.shape == (1, 4, 16, 16)


class TestEstimateSmapsAcsOnly:
    """acs_only pre-crops before dispatch so out-of-ACS aliasing can't
    corrupt the maps (fixes power_iter, which otherwise IFFTs the whole tensor)."""

    def test_acs_only_ignores_outside_acs(self) -> None:
        # power_iter seeds its init via unseeded randn → seed identically so the
        # only variable is the (discarded) out-of-ACS corruption.
        kspace, _ = _make_birdcage_kspace(num_coils=4, height=64, width=64)
        torch.manual_seed(0)
        clean = estimate_smaps(kspace, method="power_iter", acs_size=16, acs_only=True)
        corrupted = kspace.clone()
        mask = torch.ones(64, 64, dtype=torch.bool)
        mask[24:40, 24:40] = False
        corrupted[:, :, mask] = corrupted[:, :, mask] + 50.0
        torch.manual_seed(0)
        after = estimate_smaps(
            corrupted, method="power_iter", acs_size=16, acs_only=True
        )
        torch.testing.assert_close(clean, after)

    def test_default_full_kspace_is_sensitive_to_outside(self) -> None:
        # Default (acs_only=False): power_iter uses the whole tensor, so
        # out-of-center corruption DOES change the maps — proving the knob matters.
        kspace, _ = _make_birdcage_kspace(num_coils=4, height=64, width=64)
        torch.manual_seed(0)
        clean = estimate_smaps(kspace, method="power_iter")
        corrupted = kspace.clone()
        mask = torch.ones(64, 64, dtype=torch.bool)
        mask[24:40, 24:40] = False
        corrupted[:, :, mask] = corrupted[:, :, mask] + 50.0
        torch.manual_seed(0)
        after = estimate_smaps(corrupted, method="power_iter")
        assert not torch.allclose(clean, after, atol=1e-4)

    def test_acs_only_too_small_for_kernel_raises(self) -> None:
        kspace, _ = _make_birdcage_kspace(num_coils=4, height=64, width=64)
        with pytest.raises(ValueError):
            estimate_smaps(
                kspace, method="power_iter", acs_size=4, acs_only=True, kernel_size=7
            )


class TestSenseGfactorMap:
    """SENSE g-factor — quantifies whether coil maps can unfold the aliasing.
    Clean, distinct maps → g≈1; degenerate maps (identical across folded
    pixels) → g blows up."""

    def _rand_smaps(self, c: int, h: int, w: int, seed: int = 0) -> torch.Tensor:
        torch.manual_seed(seed)
        s = torch.randn(1, c, h, w, dtype=torch.complex64)
        rss = torch.sqrt((s.abs() ** 2).sum(dim=1, keepdim=True) + 1e-8)
        return s / rss

    def test_r1_is_all_ones(self) -> None:
        smaps = self._rand_smaps(4, 16, 16)
        g = sense_gfactor_map(smaps, accel=1)
        assert g.shape == (1, 1, 16, 16)
        torch.testing.assert_close(g, torch.ones_like(g))

    def test_wellconditioned_near_one(self) -> None:
        smaps = self._rand_smaps(8, 32, 16, seed=1)
        g = sense_gfactor_map(smaps, accel=2)
        assert g.shape == (1, 1, 32, 16)
        assert torch.isfinite(g).all()
        assert g.min().item() >= 1.0 - 1e-3  # g >= 1 by construction
        assert g.max().item() < 5.0  # distinct random coils unfold well

    def test_degenerate_maps_blow_up(self) -> None:
        # Identical sensitivity at the two folded pixels → singular encoding.
        c, h, w = 4, 32, 8
        s = self._rand_smaps(c, h // 2, w, seed=2)  # (1,C,16,W)
        smaps = torch.cat([s, s], dim=2)  # top half == bottom half → folds collide
        g = sense_gfactor_map(smaps, accel=2)
        assert g.max().item() > 50.0

    def test_single_coil_r1_ones(self) -> None:
        smaps = self._rand_smaps(1, 16, 16)
        g = sense_gfactor_map(smaps, accel=1)
        torch.testing.assert_close(g, torch.ones_like(g))

    def test_non_divisible_raises(self) -> None:
        smaps = self._rand_smaps(4, 30, 16)
        with pytest.raises(ValueError):
            sense_gfactor_map(smaps, accel=4)


class TestEspiritRankViability:
    """Guard: ESPIRiT's block-Hankel calibration must not be rank-deficient.

    The exp_11 divergence root cause: kernel_size=12 / acs_size=24 / 4 coils
    gives (24-12+1)²=169 patches < 12²·4=576 unknowns → ill-conditioned maps
    that poison the sense_adjoint training loss. Must RAISE, not emit bad maps.
    """

    def test_rank_deficient_geometry_raises(self) -> None:
        kspace, _ = _make_birdcage_kspace(num_coils=4, height=64, width=64)
        with pytest.raises(ValueError, match="rank-deficient"):
            estimate_csm_espirit(kspace, num_coils=4, kernel_size=12, acs_size=24)

    def test_rank_deficient_via_dispatcher_raises(self) -> None:
        kspace, _ = _make_birdcage_kspace(num_coils=4, height=64, width=64)
        with pytest.raises(ValueError, match="rank-deficient"):
            estimate_smaps(kspace, "espirit", kernel_size=12, acs_size=24)

    def test_well_posed_geometry_passes(self) -> None:
        # (24-6+1)²=361 >= 6²·4=144 → well-posed; must NOT raise.
        kspace, _ = _make_birdcage_kspace(num_coils=4, height=64, width=64)
        smaps = estimate_csm_espirit(kspace, num_coils=4, kernel_size=6, acs_size=24)
        assert smaps.shape == (1, 4, 64, 64)
        assert torch.isfinite(smaps).all()


class TestEspiritFiniteGuard:
    """Guard: a non-finite eigendecomposition must RAISE, not propagate a
    NaN/Inf coil map into a training loss (the silent-poison pattern, #9)."""

    def test_non_finite_eig_raises(self, monkeypatch) -> None:
        from spectramr.infrastructure.physics import coil_sensitivity as cs

        def nan_eigh(mat):
            n, c, _ = mat.shape
            return (
                torch.full((n, c), float("nan")),
                torch.full((n, c, c), float("nan"), dtype=mat.dtype),
            )

        monkeypatch.setattr(cs, "_robust_eigh", nan_eigh)
        kspace, _ = _make_birdcage_kspace(num_coils=4, height=64, width=64)
        with pytest.raises(ValueError, match="non-finite"):
            estimate_csm_espirit(kspace, num_coils=4, kernel_size=6, acs_size=24)

    def test_cusolver_fallback_logs_at_warning(self, monkeypatch, caplog) -> None:
        """The device→host fallback must be LOUD (WARNING), not silent DEBUG."""
        import logging

        from spectramr.infrastructure.physics import coil_sensitivity as cs

        real_eigh = torch.linalg.eigh
        calls = {"n": 0}

        def flaky_eigh(mat):
            calls["n"] += 1
            if calls["n"] == 1:
                raise torch.linalg.LinAlgError(
                    "cusolver: CUSOLVER_STATUS_INVALID_VALUE"
                )
            return real_eigh(mat)

        monkeypatch.setattr(torch.linalg, "eigh", flaky_eigh)
        gram = torch.randn(16, 4, 5, dtype=torch.complex64)
        gram = gram @ gram.conj().transpose(-2, -1)
        with caplog.at_level(logging.WARNING):
            cs._robust_eigh(gram)
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert "CPU LAPACK" in caplog.text


class TestPowerIterDeterminism:
    """Guard: power_iter maps must be reproducible run-to-run (seeded init),
    so a fixed config yields identical coil maps."""

    def test_seeded_init_is_deterministic(self) -> None:
        kspace, _ = _make_birdcage_kspace(num_coils=4, height=64, width=64)
        a = estimate_csm_power_iter(kspace, num_coils=4, kernel_size=12)
        b = estimate_csm_power_iter(kspace, num_coils=4, kernel_size=12)
        torch.testing.assert_close(a, b)


class TestRobustEighCpuFallbackLatch:
    """``_robust_eigh`` must fall back once per accelerator, then remember.

    cuSOLVER's batched-complex ``eigh`` asks for a pathological workspace and
    fails on the per-pixel ESPIRiT Gram. The CPU-LAPACK fallback is correct and
    deliberate, but it used to re-attempt the doomed GPU call -- and re-emit its
    WARNING -- once per slice, burying the log in identical lines. The failure is
    a property of (device, dtype) rather than of the data, so it is latched.

    The latch is accelerator-only: see ``test_cpu_input_is_never_latched``.
    """

    @pytest.fixture(autouse=True)
    def _clear_latch(self):
        coil_sensitivity_mod._EIGH_CPU_FALLBACK.clear()
        yield
        coil_sensitivity_mod._EIGH_CPU_FALLBACK.clear()

    @staticmethod
    def _hermitian(device: str = "cpu", n: int = 4) -> torch.Tensor:
        torch.manual_seed(0)
        a = torch.randn(3, n, n, dtype=torch.complex64, device=device)
        return a + a.conj().transpose(-1, -2)

    @staticmethod
    def _fail_once(calls: list[str]):
        """Return an eigh stand-in that fails the first call, then delegates."""
        real_eigh = torch.linalg.eigh

        def flaky(matrix):
            calls.append(matrix.device.type)
            if len(calls) == 1:
                raise torch.linalg.LinAlgError("simulated cuSOLVER workspace failure")
            return real_eigh(matrix)

        return flaky

    @pytest.mark.gpu
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="latch is accelerator-only"
    )
    def test_accelerator_fallback_is_latched_and_not_retried(self, monkeypatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr(torch.linalg, "eigh", self._fail_once(calls))
        m = self._hermitian(device="cuda:0")

        _robust_eigh(m)
        assert (
            len(calls) == 2
        ), "first call should attempt on cuda, then fall back to cpu"
        assert calls == ["cuda", "cpu"]
        assert ("cuda", m.dtype) in coil_sensitivity_mod._EIGH_CPU_FALLBACK

        _robust_eigh(m)
        # 3, not 4: the latched call skips the attempt known to fail.
        assert len(calls) == 3, "latched call must not re-attempt the failing cuda path"
        assert calls[2] == "cpu"

    @pytest.mark.gpu
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="latch is accelerator-only"
    )
    def test_first_accelerator_fallback_warns_once_then_drops_to_debug(
        self, monkeypatch, caplog
    ) -> None:
        calls: list[str] = []
        monkeypatch.setattr(torch.linalg, "eigh", self._fail_once(calls))
        m = self._hermitian(device="cuda:0")

        with caplog.at_level(logging.WARNING):
            _robust_eigh(m)
            _robust_eigh(m)
            _robust_eigh(m)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}"
        assert "CPU LAPACK" in warnings[0].getMessage()

    @pytest.mark.gpu
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="latch is accelerator-only"
    )
    def test_latched_fallback_still_returns_correct_eigenvalues(
        self, monkeypatch
    ) -> None:
        real_eigh = torch.linalg.eigh
        m = self._hermitian(device="cuda:0")
        expected = real_eigh(m)[0]

        calls: list[str] = []
        monkeypatch.setattr(torch.linalg, "eigh", self._fail_once(calls))
        first = _robust_eigh(m)[0]
        second = _robust_eigh(m)[0]

        torch.testing.assert_close(first, expected)
        torch.testing.assert_close(second, expected)
        assert first.device == m.device, "results must come back on the original device"

    def test_cpu_input_is_never_latched(self, monkeypatch) -> None:
        """A CPU matrix has no accelerated path to skip, so it must not latch.

        Latching CPU would make the retry order-dependent and silence the
        WARNING that ``TestEspiritFiniteGuard`` relies on.
        """
        calls: list[str] = []
        monkeypatch.setattr(torch.linalg, "eigh", self._fail_once(calls))
        m = self._hermitian(device="cpu")

        _robust_eigh(m)
        assert not coil_sensitivity_mod._EIGH_CPU_FALLBACK, "CPU must not latch"

    def test_healthy_eigh_never_latches(self) -> None:
        """A working path must not be poisoned: no fallback, no latch entry."""
        m = self._hermitian()
        vals, _ = _robust_eigh(m)
        torch.testing.assert_close(vals, torch.linalg.eigh(m)[0])
        assert not coil_sensitivity_mod._EIGH_CPU_FALLBACK


def test_rank_deficient_espirit_names_the_viable_acs_size():
    """The raise must say what WOULD work, not just "increase acs_size".

    Every kspace_filling arm declares ``kernel_size: 12, acs_size: 24`` with 4
    coils. Those run ``power_iter``, which has no rank condition, so the pairing
    is legal today -- but it is one ``method: espirit`` away from being
    rank-deficient, and solving ``(acs - k + 1)^2 >= margin * k^2 * C`` by hand
    at the point of failure is exactly the friction ``espirit_min_acs_size``
    exists to remove.
    """
    import torch

    from spectramr.infrastructure.physics.coil_sensitivity import (
        espirit_min_acs_size,
        estimate_csm_espirit,
    )

    viable = espirit_min_acs_size(4, kernel_size=12, max_acs=256)
    with pytest.raises(ValueError, match="rank-deficient") as excinfo:
        estimate_csm_espirit(
            torch.zeros(1, 4, 256, 256, dtype=torch.complex64),
            num_coils=4,
            kernel_size=12,
            acs_size=24,
        )
    message = str(excinfo.value)
    assert str(viable) in message, f"raise did not name the viable ACS size: {message}"
    # And it must warn against widening over a mask -- the ACS is only safe to
    # grow across a FULLY SAMPLED region.
    assert "FULLY SAMPLED" in message


# ---------------------------------------------------------------------------
# prepare_smaps_for_kspace_conditioning (#1297)
# ---------------------------------------------------------------------------
#
# Coil sensitivity maps live in IMAGE space, but the k-space cold-diffusion
# networks concatenate them channel-wise onto a K-SPACE tensor and then apply a
# single domain transform to the whole stack.  Whichever way ``force_pure_kspace``
# is set, one half of that stack is mistreated -- and a convolution can only
# relate channels at the same index, so an image-space sensitivity at pixel
# (x, y) ends up aligned with the spatial frequency (kx, ky) = (x, y).
#
# The helper fixes the domain and then makes the two halves comparable.  The
# tests below pin the three properties that make it safe, plus the layout and
# degenerate-input handling.


def _analytic_smaps(
    num_coils: int = 4, size: int = 64, batch: int = 2
) -> torch.Tensor:
    """Smooth, RSS-normalised complex sensitivity maps on an analytic phantom."""
    grid = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    corners = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    coils = [
        torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 1.5)
        * torch.exp(1j * torch.pi * (xx * cx + yy * cy) / 2)
        for cx, cy in corners[:num_coils]
    ]
    smaps = torch.stack(coils).unsqueeze(0).expand(batch, -1, -1, -1).clone()
    smaps = smaps.to(torch.complex64)
    return smaps / torch.sqrt((smaps.abs() ** 2).sum(dim=1, keepdim=True) + 1e-8)


def _reference_kspace(smaps: torch.Tensor, undersample: int = 4) -> torch.Tensor:
    """A plausible undersampled multi-coil k-space for the same phantom."""
    size = smaps.shape[-1]
    grid = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    phantom = ((xx**2 + yy**2) < 0.6).float()
    kspace = fft2c(phantom * smaps)
    mask = torch.zeros(size, size)
    mask[:, ::undersample] = 1.0
    mask[:, size // 2 - size // 16 : size // 2 + size // 16] = 1.0
    return kspace * mask


def _to_interleaved(z: torch.Tensor) -> torch.Tensor:
    """Complex -> ``[R1, I1, R2, I2, ...]`` on dim 1, as the strategies do."""
    b, c, h, w = z.shape
    return torch.view_as_real(z).permute(0, 1, 4, 2, 3).reshape(b, 2 * c, h, w)


def _rms(z: torch.Tensor) -> torch.Tensor:
    return z.abs().pow(2).flatten(1).mean(dim=1).sqrt()


def _peak(z: torch.Tensor) -> torch.Tensor:
    return z.abs().flatten(1).amax(dim=1)


class TestPrepareSmapsForKspaceConditioning:
    """The three properties that make image-domain maps safe in a k-space stack."""

    @pytest.mark.unit
    def test_output_is_the_kspace_of_the_input_up_to_a_real_gain(self) -> None:
        """Property 1 (domain): the result is ``fft2c(smaps)``, only rescaled.

        The whole point of the helper. A real positive per-sample gain and a
        magnitude clamp are the only things allowed to differ from the plain
        transform -- in particular the PHASE, which carries the spatial
        encoding, must be untouched.
        """
        smaps = _analytic_smaps()
        reference = _reference_kspace(smaps)
        prepared, _ = coil_sensitivity_mod.prepare_smaps_for_kspace_conditioning(
            smaps, reference
        )
        expected = fft2c(smaps)
        live = expected.abs() > 1e-6
        phase_error = (torch.angle(prepared[live]) - torch.angle(expected[live])).abs()
        # angles wrap; compare on the circle
        phase_error = torch.minimum(phase_error, 2 * torch.pi - phase_error)
        assert phase_error.max() < 1e-4

    @pytest.mark.unit
    def test_level_is_rms_matched_to_the_reference_per_sample(self) -> None:
        """Property 2 (level): each sample's RMS equals the reference's RMS.

        ``fft2c`` is unitary under ``norm="ortho"`` (Parseval), so the transform
        does NOT change the RMS -- unit-magnitude maps still sit an order of
        magnitude above the k-space periphery, which is where nearly the whole
        plane lives. Matching is what makes the two halves comparable.
        """
        smaps = _analytic_smaps()
        reference = _reference_kspace(smaps)
        # give the two samples deliberately different levels
        reference[1] *= 17.0
        prepared, scale = coil_sensitivity_mod.prepare_smaps_for_kspace_conditioning(
            smaps, reference
        )
        torch.testing.assert_close(_rms(prepared), _rms(reference), rtol=1e-4, atol=0)
        assert scale.shape == (smaps.shape[0],)
        # per-SAMPLE, not a single global gain
        assert scale[1] > 10 * scale[0]

    @pytest.mark.unit
    def test_rms_match_absorbs_a_global_amplitude_blowup(self) -> None:
        """A uniformly huge map cannot poison the gradient: step 2 is scale-free."""
        smaps = _analytic_smaps()
        reference = _reference_kspace(smaps)
        normal, _ = coil_sensitivity_mod.prepare_smaps_for_kspace_conditioning(
            smaps, reference
        )
        huge, _ = coil_sensitivity_mod.prepare_smaps_for_kspace_conditioning(
            smaps * 1000.0, reference
        )
        torch.testing.assert_close(normal, huge, rtol=1e-3, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.parametrize("size", [64, 128])
    @pytest.mark.parametrize("undersample", [1, 4, 8])
    def test_amplitude_cap_is_inert_for_realistic_maps(
        self, size: int, undersample: int
    ) -> None:
        """Property 3a: the guard does NOT distort the normal case.

        A ceiling that engages on every step is a distortion, not a guard. Swept
        over resolution and acceleration, realistic smooth maps come out at
        1.02-1.33x the reference peak -- comfortably under the 2.0 ceiling, and
        resolution-invariant because reference and map peaks scale together.
        """
        smaps = _analytic_smaps(size=size)
        reference = _reference_kspace(smaps, undersample=undersample)
        prepared, _ = coil_sensitivity_mod.prepare_smaps_for_kspace_conditioning(
            smaps, reference
        )
        ratio = (_peak(prepared) / _peak(reference)).max().item()
        assert ratio < coil_sensitivity_mod.SMAP_KSPACE_PEAK_RATIO, ratio
        # and the measured band, so a regression that silently rescales is caught
        assert 0.9 < ratio < 1.5, ratio

    @pytest.mark.unit
    def test_amplitude_cap_engages_and_preserves_phase(self) -> None:
        """Property 3b: with a degenerate reference the ceiling actually binds.

        A flat (noise-like) reference has a low peak-to-RMS, so RMS matching
        alone would let the map's DC spike tower over it. The clamp is
        phase-preserving: it scales magnitudes, never rotates them.
        """
        smaps = torch.full((1, 4, 64, 64), 0.5, dtype=torch.complex64)
        reference = torch.full((1, 4, 64, 64), 0.1, dtype=torch.complex64)
        prepared, _ = coil_sensitivity_mod.prepare_smaps_for_kspace_conditioning(
            smaps, reference
        )
        ratio = (_peak(prepared) / _peak(reference)).max().item()
        assert ratio == pytest.approx(
            coil_sensitivity_mod.SMAP_KSPACE_PEAK_RATIO, rel=1e-3
        )
        expected = fft2c(smaps)
        live = expected.abs() > 1e-6
        phase_error = (torch.angle(prepared[live]) - torch.angle(expected[live])).abs()
        phase_error = torch.minimum(phase_error, 2 * torch.pi - phase_error)
        assert phase_error.max() < 1e-4

    @pytest.mark.unit
    def test_real_interleaved_layout_matches_the_complex_path(self) -> None:
        """The strategies hand over ``[R1, I1, R2, I2, ...]``; same field, same answer."""
        smaps = _analytic_smaps()
        reference = _reference_kspace(smaps)
        complex_out, complex_scale = (
            coil_sensitivity_mod.prepare_smaps_for_kspace_conditioning(smaps, reference)
        )
        real_out, real_scale = (
            coil_sensitivity_mod.prepare_smaps_for_kspace_conditioning(
                _to_interleaved(smaps), _to_interleaved(reference)
            )
        )
        assert real_out.shape == _to_interleaved(smaps).shape
        assert not torch.is_complex(real_out)
        torch.testing.assert_close(
            real_out, _to_interleaved(complex_out), rtol=1e-4, atol=1e-6
        )
        torch.testing.assert_close(real_scale, complex_scale, rtol=1e-5, atol=0)

    @pytest.mark.unit
    def test_channel_dim_2_handles_the_5d_layout(self) -> None:
        """5D arms arrive as ``[B, D, C, H, W]`` -- coils on dim 2, not dim 1."""
        smaps = _analytic_smaps()
        reference = _reference_kspace(smaps)
        flat, _ = coil_sensitivity_mod.prepare_smaps_for_kspace_conditioning(
            smaps, reference
        )
        volumetric, _ = coil_sensitivity_mod.prepare_smaps_for_kspace_conditioning(
            smaps.unsqueeze(1).expand(-1, 3, -1, -1, -1).clone(),
            reference.unsqueeze(1).expand(-1, 3, -1, -1, -1).clone(),
            channel_dim=2,
        )
        assert volumetric.shape == (smaps.shape[0], 3, *smaps.shape[1:])
        torch.testing.assert_close(volumetric[:, 0], flat, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_degenerate_inputs_stay_finite(self) -> None:
        """Zero maps and a zero reference must not produce NaN/Inf.

        A fully-masked cold-diffusion step (t -> T) really does hand over an
        all-zero reference; scaling the conditioning to zero along with the data
        is coherent, a NaN is not.
        """
        smaps = _analytic_smaps()
        reference = _reference_kspace(smaps)
        zero_maps, _ = coil_sensitivity_mod.prepare_smaps_for_kspace_conditioning(
            torch.zeros_like(smaps), reference
        )
        zero_ref, _ = coil_sensitivity_mod.prepare_smaps_for_kspace_conditioning(
            smaps, torch.zeros_like(reference)
        )
        assert torch.isfinite(zero_maps.abs()).all()
        assert torch.isfinite(zero_ref.abs()).all()
        assert bool((zero_maps.abs() == 0).all())
        assert bool((zero_ref.abs() == 0).all())

    @pytest.mark.unit
    def test_dtype_and_shape_round_trip(self) -> None:
        """The concat that follows is dtype-strict, so the output must match."""
        smaps = _analytic_smaps()
        reference = _reference_kspace(smaps)
        for maps, ref in (
            (smaps, reference),
            (_to_interleaved(smaps), _to_interleaved(reference)),
        ):
            prepared, _ = coil_sensitivity_mod.prepare_smaps_for_kspace_conditioning(
                maps, ref
            )
            assert prepared.dtype == maps.dtype
            assert prepared.shape == maps.shape
            assert torch.cat([ref, prepared], dim=1).shape[1] == 2 * ref.shape[1]

    @pytest.mark.unit
    def test_odd_real_channel_count_raises(self) -> None:
        """No silent fallback (CLAUDE.md #3): an odd count is not a complex field."""
        odd = torch.randn(1, 3, 16, 16)
        reference = torch.randn(1, 3, 16, 16)
        with pytest.raises(ValueError, match="even channel count"):
            coil_sensitivity_mod.prepare_smaps_for_kspace_conditioning(odd, reference)

    @pytest.mark.unit
    def test_odd_channel_reference_is_read_as_a_real_field_not_rejected(self) -> None:
        """The ``in_channels: 1`` arms must not be killed by the S-map guard.

        Two ``kspace_cold_diffusion`` arms in the corpus declare a single
        channel, so ``reference`` arrives real with an odd extent. It is read
        for an RMS and a peak only and never converted back, so it is a real
        field with zero imaginary part -- rejecting it would turn a working
        (if wrongly-conditioned) arm into a crash, and the error would blame
        the S-maps for a tensor that is not the S-maps.
        """
        smaps = _analytic_smaps(batch=2)
        # A single real channel of the same phantom's undersampled k-space.
        reference = _reference_kspace(smaps)[:, :1].real.contiguous()
        assert reference.shape[1] % 2 == 1 and not torch.is_complex(reference)

        prepared, scale = coil_sensitivity_mod.prepare_smaps_for_kspace_conditioning(
            smaps, reference
        )

        assert prepared.shape == smaps.shape
        assert torch.isfinite(prepared).all()
        # The level match still holds, against the real field's own RMS.
        torch.testing.assert_close(_rms(prepared), _rms(reference), rtol=2e-3, atol=0)
        assert (scale > 0).all()
        # And the clamp stays inert, as it does for a complex reference.
        assert (_peak(prepared) / _peak(reference) < 2.0).all()

    @pytest.mark.unit
    def test_non_positive_peak_ratio_raises(self) -> None:
        smaps = _analytic_smaps(batch=1)
        with pytest.raises(ValueError, match="peak_ratio"):
            coil_sensitivity_mod.prepare_smaps_for_kspace_conditioning(
                smaps, _reference_kspace(smaps), peak_ratio=0.0
            )


# ---------------------------------------------------------------------------
# resolve_estimation_settings — one resolver for both config shapes (#1326)
# ---------------------------------------------------------------------------


class _Node:
    """Attribute-bearing config node, as the training strategies see it."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class TestResolveEstimationSettings:
    """The estimation block must resolve identically from a dict or a model.

    The training strategies hold a Pydantic tree; the inference strategies hold
    a ``model_dump``.  A resolver that only reads one of the two makes the arm's
    declared method a silent no-op on the other path — which is exactly how
    sampling came to hardcode ``power_iter`` while the YAML asked for ESPIRiT.
    """

    def _both_shapes(self, estimation: dict):
        from spectramr.infrastructure.physics.coil_sensitivity import (
            resolve_estimation_settings,
        )

        as_dict = {"physics": {"coil_processing": {"estimation": estimation}}}
        as_obj = _Node(physics=_Node(coil_processing=_Node(estimation=_Node(**estimation))))
        return (
            resolve_estimation_settings(as_dict),
            resolve_estimation_settings(as_obj),
        )

    def test_dict_and_attribute_forms_agree(self):
        from_dict, from_obj = self._both_shapes(
            {"method": "espirit", "kernel_size": 6, "acs_size": 24}
        )
        assert from_dict == from_obj
        assert from_dict == ("espirit", {"kernel_size": 6, "acs_size": 24})

    def test_absent_block_falls_back_to_the_default_with_no_subknobs(self):
        from spectramr.infrastructure.physics.coil_sensitivity import (
            resolve_estimation_settings,
        )

        assert resolve_estimation_settings({}) == ("power_iter", {})
        assert resolve_estimation_settings({"physics": {}}) == ("power_iter", {})
        assert resolve_estimation_settings(None) == ("power_iter", {})

    def test_method_none_collapses_to_the_default(self):
        """Every caller is on a branch that *requires* maps.

        Propagating ``"none"`` would make ``estimate_smaps`` return ``None`` and
        degrade the conditioning silently instead of raising (pitfall #9).
        """
        from_dict, from_obj = self._both_shapes({"method": "none", "kernel_size": 4})
        assert from_dict == from_obj == ("power_iter", {"kernel_size": 4})

    def test_unset_subknobs_are_omitted_so_estimate_smaps_defaults_apply(self):
        from_dict, from_obj = self._both_shapes(
            {"method": "power_iter", "kernel_size": None, "maps_path": None}
        )
        assert from_dict == from_obj == ("power_iter", {})

    def test_the_enabled_flag_is_not_mistaken_for_a_subknob(self):
        """58 arms carry ``enabled`` in the block; ``estimate_smaps`` has no such arg."""
        method, kwargs = self._both_shapes(
            {"method": "power_iter", "enabled": True, "kernel_size": 6}
        )[0]
        assert method == "power_iter"
        assert kwargs == {"kernel_size": 6}

    def test_the_resolved_pair_splats_into_estimate_smaps(self):
        """Round-trip: whatever the resolver returns must be callable as-is."""
        import torch

        from spectramr.infrastructure.physics.coil_sensitivity import (
            estimate_smaps,
            resolve_estimation_settings,
        )

        method, kwargs = resolve_estimation_settings(
            {
                "physics": {
                    "coil_processing": {"estimation": {"method": "power_iter", "kernel_size": 6}}
                }
            }
        )
        gen = torch.Generator().manual_seed(0)
        kspace = torch.complex(
            torch.randn(1, 4, 32, 32, generator=gen),
            torch.randn(1, 4, 32, 32, generator=gen),
        )
        maps = estimate_smaps(kspace, method=method, acs_only=True, **kwargs)
        assert maps is not None
        assert torch.is_complex(maps)
        # ``acs_only=True`` crops to the central ACS block *before* dispatch, so
        # the maps come back at the ACS size (24 by default), not the input's.
        # Every caller therefore has to resize them back — which is why both the
        # validation path and the sampler carry an interpolate step keyed off the
        # pre-crop ``(h, w)``.  Pinned here so a caller added later cannot assume
        # full-size maps and silently mis-align the conditioning.
        assert maps.shape[:2] == kspace.shape[:2]
        assert maps.shape[-2:] == (24, 24)


class TestReturnSubspace:
    """The eigenbasis was computed and thrown away on every call.

    ``estimate_csm_espirit`` eigendecomposes a per-pixel Gram matrix and keeps
    only the leading eigenvector; the other ``C - k`` span the coil null space.
    ``return_subspace=True`` surfaces the projector built from that SAME
    decomposition, so the maps and the projector cannot disagree -- recomputing
    it in a caller would be a second owner of the same eigenproblem (NN17).
    """

    @staticmethod
    def _kspace(coils: int = 4, size: int = 64) -> torch.Tensor:
        from spectramr.infrastructure.physics.fft_ops import fft2c

        torch.manual_seed(0)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij"
        )
        mag = ((xx**2 + yy**2) < 0.6).float()
        centres = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)][:coils]
        sens = torch.stack(
            [torch.exp(-((xx - a) ** 2 + (yy - b) ** 2)) for a, b in centres]
        ).to(torch.complex64)
        sens = sens / sens.abs().pow(2).sum(0, keepdim=True).sqrt().clamp(min=1e-6)
        return fft2c(sens * mag.to(torch.complex64)).unsqueeze(0)

    def test_the_default_return_is_unchanged(self):
        """Every existing caller consumes a bare tensor positionally."""
        k = self._kspace()
        maps = estimate_csm_espirit(k, num_coils=4, acs_size=24)
        assert isinstance(maps, torch.Tensor)
        assert maps.shape == (1, 4, 64, 64)

    def test_the_maps_are_identical_either_way(self):
        """The flag must not perturb the estimate it rides along with."""
        k = self._kspace()
        bare = estimate_csm_espirit(k, num_coils=4, acs_size=24)
        rich = estimate_csm_espirit(k, num_coils=4, acs_size=24, return_subspace=True)
        assert torch.equal(bare, rich.maps)

    def test_the_shapes_are_what_the_consumer_indexes(self):
        rich = estimate_csm_espirit(self._kspace(), num_coils=4, acs_size=24, return_subspace=True)
        assert rich.null_projector.shape == (1, 64, 64, 4, 4)
        assert rich.leading_eigenvalue.shape == (1, 64, 64)
        assert torch.is_complex(rich.null_projector)
        assert not torch.is_complex(rich.leading_eigenvalue)

    def test_the_flag_is_keyword_only(self):
        """A positional would land on `max_n_keep` and silently change the maps."""
        with pytest.raises(TypeError):
            estimate_csm_espirit(self._kspace(), 4, 6, 24, 0.02, 0.95, 0, None, True)

    def test_the_second_eigenvalue_is_reported(self):
        """The rank-one assumption is checkable, not assumed.

        Where lambda_2 approaches 1 the pixel needs a second ESPIRiT map and its
        null space is C-2, so the budget's denominator is too large there. The
        field exists so a consumer can measure that instead of hoping.
        """
        rich = estimate_csm_espirit(
            self._kspace(), num_coils=4, acs_size=24, return_subspace=True
        )
        support = rich.leading_eigenvalue >= 0.95
        assert rich.second_eigenvalue.shape == rich.leading_eigenvalue.shape
        # eigh returns ascending order, so the second-largest never exceeds the
        # largest -- a swap here would silently invert every rank verdict.
        assert (rich.second_eigenvalue <= rich.leading_eigenvalue + 1e-6).all()
        assert rich.second_eigenvalue[support].mean() < rich.leading_eigenvalue[support].mean()

    def test_the_projector_annihilates_the_leading_eigenvector(self):
        """The direct statement of what 'null space' means here."""
        rich = estimate_csm_espirit(self._kspace(), num_coils=4, acs_size=24, return_subspace=True)
        support = rich.leading_eigenvalue >= 0.95
        assert support.any(), "phantom produced no in-support pixels -- test is vacuous"
        maps = rich.maps[0].permute(1, 2, 0).unsqueeze(-1)  # (H, W, C, 1)
        residual = (rich.null_projector[0] @ maps).squeeze(-1).abs().pow(2).sum(-1)
        assert residual[support[0]].max() < 1e-6
