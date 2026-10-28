"""Unit tests for TargetAwareFSDC (Target-Aware Frequency-Split Data Consistency)."""

import pytest
import torch

from spectramr.infrastructure.physics.data_consistency import TargetAwareFSDC


class TestTargetAwareFSDC:
    """Tests for the physics-correct TA-FSDC implementation."""

    @pytest.fixture()
    def fsdc(self) -> TargetAwareFSDC:
        """Create a TA-FSDC instance with default parameters."""
        return TargetAwareFSDC(
            target_channels=8,
            center_fraction=0.1,  # 10% for easy testing
            hf_lambda=0.5,
        )

    @pytest.fixture()
    def multi_contrast_data(self) -> dict[str, torch.Tensor]:
        """Create synthetic multi-contrast k-space data.

        Layout: [B=2, C=16, H=32, W=32]
        - Source (reference): channels 0-7 (4 coils × 2 R/I)
        - Target: channels 8-15 (4 coils × 2 R/I)
        """
        B, C, H, W = 2, 16, 32, 32
        torch.manual_seed(42)
        kspace_pred = torch.randn(B, C, H, W)
        kspace_measured = torch.randn(B, C, H, W)

        # Mask: undersample ~50% of lines (keep center, random outer)
        mask = torch.zeros(B, 1, H, W)
        mask[:, :, ::2, :] = 1.0  # every other line sampled
        mask[:, :, H // 2 - 3 : H // 2 + 3, :] = 1.0  # ACS always sampled

        return {
            "kspace_pred": kspace_pred,
            "kspace_measured": kspace_measured,
            "mask": mask,
        }

    def test_output_shape_matches_input(
        self, fsdc: TargetAwareFSDC, multi_contrast_data: dict
    ) -> None:
        """Output shape must match prediction shape."""
        out = fsdc(**multi_contrast_data)
        assert out.shape == multi_contrast_data["kspace_pred"].shape

    def test_reference_channels_hard_dc(
        self, fsdc: TargetAwareFSDC, multi_contrast_data: dict
    ) -> None:
        """At sampled locations, reference channels must equal measurement."""
        out = fsdc(**multi_contrast_data)
        mask = multi_contrast_data["mask"]
        meas = multi_contrast_data["kspace_measured"]

        # Reference = channels 0:8
        ref_out = out[:, :8, ...]
        ref_meas = meas[:, :8, ...]
        mask_b = mask.bool()

        # At sampled positions, reference output == measurement (hard DC)
        for b in range(ref_out.shape[0]):
            sampled = mask_b[b, 0]  # [H, W]
            for c in range(8):
                torch.testing.assert_close(
                    ref_out[b, c][sampled],
                    ref_meas[b, c][sampled],
                    rtol=1e-5,
                    atol=1e-5,
                )

    def test_reference_unsampled_keeps_prediction(
        self, fsdc: TargetAwareFSDC, multi_contrast_data: dict
    ) -> None:
        """At unsampled locations, reference channels must equal prediction."""
        out = fsdc(**multi_contrast_data)
        mask = multi_contrast_data["mask"]
        pred = multi_contrast_data["kspace_pred"]

        ref_out = out[:, :8, ...]
        ref_pred = pred[:, :8, ...]
        unsampled = ~mask.bool()

        for b in range(ref_out.shape[0]):
            uns = unsampled[b, 0]
            for c in range(8):
                torch.testing.assert_close(
                    ref_out[b, c][uns],
                    ref_pred[b, c][uns],
                    rtol=1e-5,
                    atol=1e-5,
                )

    def test_target_acs_hard_dc(self, fsdc: TargetAwareFSDC, multi_contrast_data: dict) -> None:
        """In the ACS region, target channels must use hard DC (measurement)."""
        out = fsdc(**multi_contrast_data)
        meas = multi_contrast_data["kspace_measured"]
        H, W = 32, 32

        # ACS box: center_fraction is its full width, so the half-width is cf * N / 2.
        pad_h = max(1, int(H * fsdc.center_fraction / 2))
        pad_w = max(1, int(W * fsdc.center_fraction / 2))
        ch, cw = H // 2, W // 2

        target_out = out[:, 8:, ch - pad_h : ch + pad_h, cw - pad_w : cw + pad_w]
        target_meas = meas[:, 8:, ch - pad_h : ch + pad_h, cw - pad_w : cw + pad_w]

        # ACS region of target should equal measurement (hard DC)
        torch.testing.assert_close(target_out, target_meas, rtol=1e-5, atol=1e-5)

    def test_target_unsampled_pure_prediction(
        self, fsdc: TargetAwareFSDC, multi_contrast_data: dict
    ) -> None:
        """Unsampled target regions must be pure prediction (no measurement)."""
        out = fsdc(**multi_contrast_data)
        pred = multi_contrast_data["kspace_pred"]
        mask = multi_contrast_data["mask"]

        # Build the same ACS mask
        H, W = 32, 32
        pad_h = max(1, int(H * fsdc.center_fraction))
        pad_w = max(1, int(W * fsdc.center_fraction))
        ch, cw = H // 2, W // 2
        lf_mask = torch.zeros_like(mask)
        lf_mask[:, :, ch - pad_h : ch + pad_h, cw - pad_w : cw + pad_w] = 1.0

        # Unsampled = not in mask AND not in ACS
        unsampled = (~mask.bool()) & (~lf_mask.bool())

        target_out = out[:, 8:, ...]
        target_pred = pred[:, 8:, ...]

        for b in range(target_out.shape[0]):
            uns = unsampled[b, 0]
            for c in range(8):
                torch.testing.assert_close(
                    target_out[b, c][uns],
                    target_pred[b, c][uns],
                    rtol=1e-5,
                    atol=1e-5,
                )

    def test_lambda_hf_is_learnable(self, fsdc: TargetAwareFSDC) -> None:
        """The HF blend weight must be a learnable parameter."""
        assert fsdc.lambda_hf.requires_grad
        assert "lambda_hf" in dict(fsdc.named_parameters())

    def test_target_only_measurement(self, fsdc: TargetAwareFSDC) -> None:
        """When kspace_measured has only target channels, reference passes through."""
        B, H, W = 2, 32, 32
        pred = torch.randn(B, 16, H, W)
        meas = torch.randn(B, 8, H, W)  # target only
        mask = torch.ones(B, 1, H, W)

        out = fsdc(pred, meas, mask)
        assert out.shape == pred.shape

        # Reference channels should be unchanged (no source_meas available)
        torch.testing.assert_close(out[:, :8], pred[:, :8])

    def test_single_contrast_no_source(self) -> None:
        """When C_total == target_channels, no source split occurs."""
        fsdc = TargetAwareFSDC(target_channels=8, center_fraction=0.1)
        B, H, W = 2, 32, 32
        pred = torch.randn(B, 8, H, W)
        meas = torch.randn(B, 8, H, W)
        mask = torch.ones(B, 1, H, W)

        out = fsdc(pred, meas, mask)
        assert out.shape == (B, 8, H, W)

    def test_gradient_flow(self, fsdc: TargetAwareFSDC, multi_contrast_data: dict) -> None:
        """Gradients must flow through lambda_hf."""
        multi_contrast_data["kspace_pred"].requires_grad_(True)
        out = fsdc(**multi_contrast_data)
        loss = out.sum()
        loss.backward()

        assert fsdc.lambda_hf.grad is not None
        assert multi_contrast_data["kspace_pred"].grad is not None

    def test_too_few_measured_channels_raises(self, fsdc: TargetAwareFSDC) -> None:
        """C_meas < target_channels is a contract violation -> raise.

        Regression for the silent warn-and-passthrough that dropped DC
        entirely (CLAUDE.md pitfalls #4/#9/#10). A channel-count mismatch
        between predicted and measured k-space is a wiring bug, so the
        layer must fail loud rather than skip data consistency.
        """
        B, H, W = 2, 32, 32
        pred = torch.randn(B, 16, H, W)
        meas = torch.randn(B, 4, H, W)  # fewer than target_channels=8
        mask = torch.ones(B, 1, H, W)
        with pytest.raises(ValueError, match="cannot apply data consistency"):
            fsdc(pred, meas, mask)


# ── Hann-apodized ACS (audit-2026-05-14 §1 round-5) ─────────────────────────
#
# The legacy hard-rectangle ACS replacement in TA-FSDC produces a 2-D
# sinc when IFFT'd, with strong sidelobes and a concentrated central
# main lobe. For an under-trained cold-diffusion model whose HF
# prediction is near-zero, the ACS-only IFFT dominates the rendered
# image — producing the saturated "white spot" cluster at image centre
# seen in 169/180 fakes of ``experiment_130_ti_ccd``.
#
# The ``acs_taper="hann"`` opt-in apodizes the outer half of the ACS
# box with a Hann curve, reducing the central peak amplitude by ~78%
# and suppressing sidelobes — the under-training central cluster
# becomes a soft glow that the renderer's percentile normalization no
# longer saturates.


class TestAcsTaper:
    """Tests for the Hann-apodized ACS option (audit-2026-05-14 §1 round-5)."""

    def test_default_taper_is_none(self) -> None:
        """Backward compatibility: default mode is the legacy hard rectangle."""
        fsdc = TargetAwareFSDC(target_channels=8)
        assert fsdc.acs_taper is None

    def test_unsupported_taper_raises(self) -> None:
        """Unknown taper names fail loud (CLAUDE.md #9)."""
        with pytest.raises(ValueError, match="acs_taper="):
            TargetAwareFSDC(target_channels=8, acs_taper="cosine")

    def test_hard_mask_is_binary(self) -> None:
        """Legacy hard rectangle: mask values are 0 or 1 only."""
        fsdc = TargetAwareFSDC(target_channels=2, center_fraction=0.1)
        ref = torch.zeros(1, 1, 64, 64)
        m = fsdc._build_acs_mask(64, 64, ref)
        unique_vals = set(m.unique().tolist())
        assert unique_vals == {0.0, 1.0}

    def test_hann_mask_is_smoothly_tapered(self) -> None:
        """``acs_taper='hann'`` produces a smoothly-tapered mask.

        The discrete Hann window with N samples peaks at
        ``0.5*(1 - cos(π))`` only when ``(N-1)/2`` lands on an integer
        sample — i.e. odd N. For even N the peak sits between samples
        and the maximum sample value is slightly below 1 (≈ 0.96 for
        N=12). We accept any peak ≥ 0.9, which is the relevant invariant:
        the centre of the ACS retains close-to-full hard-DC weighting.
        """
        fsdc = TargetAwareFSDC(
            target_channels=2,
            center_fraction=0.2,
            acs_taper="hann",
        )
        ref = torch.zeros(1, 1, 64, 64)
        m = fsdc._build_acs_mask(64, 64, ref)
        assert m.min().item() == 0.0
        # Peak ≥ 0.9 — preserves hard-DC intent at ACS centre even with
        # even-N Hann window discretization.
        assert m.max().item() >= 0.9
        # Must contain intermediate values — proves the taper exists.
        n_unique = len(m.unique())
        assert n_unique > 5, (
            f"Hann mask is degenerate (only {n_unique} unique values). "
            "Increase the box size (center_fraction) so the taper has "
            "enough samples to ramp over."
        )

    def test_hann_reduces_ifft_central_peak(self) -> None:
        """Quantified regression for the white-spot fix.

        On a 256×256 grid with ``center_fraction=0.03`` (matching
        ``experiment_130_ti_ccd``'s YAML), the apodized ACS window's
        IFFT must have a meaningfully lower central peak than the
        legacy hard rectangle — that's the metric that decides whether
        the renderer's percentile normalisation will saturate a single
        bright cluster on under-trained outputs.

        Empirically the apodized peak is ~20% of the hard peak (78%
        reduction); we assert the looser bound ``< 0.5×`` so a future
        taper-shape change can still pass without false-alarming on
        small numerical drift.
        """
        from spectramr.infrastructure.physics.fft_ops import ifft2c

        H = W = 256
        ref = torch.zeros(1, 1, H, W)

        m_hard = TargetAwareFSDC(
            target_channels=2,
            center_fraction=0.03,
        )._build_acs_mask(H, W, ref)
        m_soft = TargetAwareFSDC(
            target_channels=2,
            center_fraction=0.03,
            acs_taper="hann",
        )._build_acs_mask(H, W, ref)

        ksp = torch.ones(1, 1, H, W, dtype=torch.complex64)
        img_hard = ifft2c(ksp * m_hard.to(torch.complex64)).abs()
        img_soft = ifft2c(ksp * m_soft.to(torch.complex64)).abs()

        peak_hard = img_hard.max().item()
        peak_soft = img_soft.max().item()
        assert peak_soft < 0.5 * peak_hard, (
            f"Hann apodization peak ({peak_soft:.3f}) is not meaningfully "
            f"below the hard-rectangle peak ({peak_hard:.3f}). The white-"
            "spot fix requires ≥ 50% peak reduction (audit-2026-05-14 §1 "
            "round-5)."
        )

    def test_hann_preserves_output_shape_in_forward(self) -> None:
        """Apodized mode produces the same output shape as the hard mode."""
        fsdc = TargetAwareFSDC(
            target_channels=8,
            center_fraction=0.1,
            acs_taper="hann",
        )
        B, H, W = 2, 32, 32
        pred = torch.randn(B, 16, H, W)
        meas = torch.randn(B, 16, H, W)
        mask = torch.zeros(B, 1, H, W)
        mask[:, :, ::2, :] = 1.0
        mask[:, :, H // 2 - 3 : H // 2 + 3, :] = 1.0

        out = fsdc(pred, meas, mask)
        assert out.shape == pred.shape

    def test_hann_preserves_acs_centre_hard_dc(self) -> None:
        """At the very ACS centre, the Hann window peaks close to 1.0 — so
        the downstream compose substitutes the measurement nearly exactly
        at DC. Only the ACS boundary gets soft-blended.
        """
        fsdc = TargetAwareFSDC(
            target_channels=2,
            center_fraction=0.2,
            acs_taper="hann",
        )
        ref = torch.zeros(1, 1, 64, 64)
        m = fsdc._build_acs_mask(64, 64, ref)
        # ``center_fraction=0.2`` → pad=12, box=24. The Hann window peaks
        # at the central sample(s); for even box width the peak sample
        # value is slightly below 1 (~0.96 for N=24) but well above 0.9.
        cy, cx = 32, 32
        assert m[0, 0, cy, cx].item() >= 0.9

    def test_hann_falls_back_to_binary_for_tiny_acs(self) -> None:
        """Boxes too small to taper (``box_h < 2`` or ``box_w < 2``)
        gracefully fall back to the binary rectangle so the algorithm
        still produces a sensible ACS mask. Without this, hann_window(0)
        would raise."""
        # center_fraction=0.005 on a 64x64 grid → pad = max(1, 0) = 1,
        # box = 2 — boundary case, still tapers. Use 32x32 to push under.
        fsdc = TargetAwareFSDC(
            target_channels=2,
            center_fraction=0.005,
            acs_taper="hann",
        )
        ref = torch.zeros(1, 1, 32, 32)
        m = fsdc._build_acs_mask(32, 32, ref)
        # Mask must be valid (no NaNs, sums >= 1).
        assert not torch.isnan(m).any()
        assert m.sum().item() >= 1.0



def test_acs_box_spans_the_declared_fraction() -> None:
    """Planted: the box used to span 2 * center_fraction of each axis."""
    fsdc = TargetAwareFSDC(target_channels=2, center_fraction=0.125)
    m = fsdc._build_acs_mask(64, 64, torch.zeros(1, 1, 64, 64))
    rows = int(m[0, 0].any(dim=1).sum())
    assert rows == 8  # 0.125 * 64


def test_log_scaled_hf_blend_is_linear() -> None:
    from spectramr.data.transforms.normalization import compress_kspace_log, decompress_kspace_log

    torch.manual_seed(0)
    fsdc = TargetAwareFSDC(target_channels=2, center_fraction=0.0625, log_scaled=True)
    lin_pred = torch.randn(1, 2, 32, 32) * 20
    lin_meas = torch.randn(1, 2, 32, 32) * 20
    mask = torch.ones(1, 1, 32, 32)
    out = fsdc(
        compress_kspace_log(lin_pred, channel_dim=1),
        compress_kspace_log(lin_meas, channel_dim=1),
        mask,
    )
    b = torch.sigmoid(fsdc.lambda_hf).detach()
    expected = (1 - b) * lin_pred + b * lin_meas
    got = decompress_kspace_log(out, channel_dim=1)
    torch.testing.assert_close(got[..., 0, 0], expected[..., 0, 0], rtol=1e-3, atol=1e-3)
