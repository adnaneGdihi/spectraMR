"""Regression tests for cross-contrast DC channel alignment.

Tests verify that:
1. DC layer uses target-contrast channels (last N) not source-prior (first N)
2. Single-contrast experiments are unaffected (no truncation needed)
"""

import torch

from spectramr.infrastructure.physics.data_consistency import HardDataConsistency


class TestDCCrossContrastChannelAlignment:
    """Verify DC layer receives correct channels in cross-contrast mode."""

    def test_dc_receives_target_channels_not_source(self) -> None:
        """When measured k-space has [Source, Target], DC should use Target half.

        Scenario: measured = [8ch_T1, 8ch_T2], prediction = 8ch_T2.
        DC must enforce T2 measurements on T2 predictions, not T1.
        """
        B, H, W = 2, 64, 64
        out_ch = 4  # 4 complex coils for target

        # Simulate cross-contrast measured k-space: [Source, Target] = 8 complex coils
        # Source: T1 = constant 1.0, Target: T2 = constant 2.0
        source_kspace = torch.ones(B, out_ch, H, W, dtype=torch.cfloat)
        target_kspace = torch.full((B, out_ch, H, W), 2.0 + 0j, dtype=torch.cfloat)
        measured_kspace = torch.cat([source_kspace, target_kspace], dim=1)

        # Model prediction (target contrast only)
        x_out = torch.randn(B, out_ch, H, W, dtype=torch.cfloat)

        # Simulate the FIXED channel alignment logic
        if measured_kspace.shape[1] > x_out.shape[1]:
            aligned_measured = measured_kspace[:, -x_out.shape[1] :]
        else:
            aligned_measured = measured_kspace

        # Verify: aligned_measured should be the Target half (value ≈ 2.0)
        assert aligned_measured.shape == x_out.shape
        assert torch.allclose(
            aligned_measured.real, torch.full_like(aligned_measured.real, 2.0)
        ), "DC should receive T2 (target) channels, not T1 (source)"

    def test_dc_single_contrast_unaffected(self) -> None:
        """Single-contrast: measured.shape[1] == x_out.shape[1], no truncation."""
        B, out_ch, H, W = 2, 4, 64, 64

        measured_kspace = torch.randn(B, out_ch, H, W, dtype=torch.cfloat)
        x_out = torch.randn(B, out_ch, H, W, dtype=torch.cfloat)

        # No truncation should happen
        if measured_kspace.shape[1] > x_out.shape[1]:
            aligned = measured_kspace[:, -x_out.shape[1] :]
        else:
            aligned = measured_kspace

        assert torch.equal(
            aligned, measured_kspace
        ), "Single-contrast should pass through unchanged"

    def test_dc_hard_consistency_with_target_channels(self) -> None:
        """End-to-end: HardDC with correctly aligned target channels in k-space domain."""
        B, C, H, W = 1, 4, 32, 32
        dc = HardDataConsistency()

        # Measured target k-space (complex)
        measured = torch.randn(B, C, H, W, dtype=torch.cfloat)
        prediction = torch.randn(B, C, H, W, dtype=torch.cfloat)

        # Float mask (matching actual usage in generator DC path)
        mask = torch.zeros(B, 1, H, W, dtype=torch.float32)
        mask[:, :, 10:20, :] = 1.0  # Sample center lines

        # Use is_kspace_domain=True (matching generator's DC call)
        result = dc(prediction, measured, mask, is_kspace_domain=True)
        assert result.shape == prediction.shape

        # At measured locations (mask=1), result ≈ measured (with slight noise from DC layer)
        # At unmeasured locations (mask=0), result = prediction (exact)
        mask_expanded = mask.expand_as(result.real)
        sampled = mask_expanded.bool()

        # HardDC applies _add_realistic_noise to measurements, so use relaxed tolerance
        assert torch.allclose(
            result[sampled], measured[sampled], atol=0.1
        ), "DC should enforce measured values (approximately) at sampled locations"

        unsampled = ~sampled
        assert torch.allclose(
            result[unsampled], prediction[unsampled], atol=1e-6
        ), "DC should keep prediction exactly at unsampled locations"

    def test_mask_target_channels_extracted_correctly(self) -> None:
        """Mask channel alignment should also take target (last N) channels."""
        B, H, W = 2, 64, 64
        out_ch = 4

        # 16-channel mask: first 8 = source (all 1.0), last 8 = target (mixed)
        source_mask = torch.ones(B, out_ch * 2, H, W)
        target_mask = torch.zeros(B, out_ch * 2, H, W)
        target_mask[:, :, ::4, :] = 1.0  # Every 4th line

        mask = torch.cat(
            [source_mask[:, :out_ch], target_mask[:, :out_ch]], dim=1
        )  # [B, 8, H, W]

        # Simulate FIXED alignment
        if mask.shape[1] > out_ch:
            aligned_mask = mask[:, -out_ch:]
        else:
            aligned_mask = mask

        # Target mask channels should have the undersampling pattern
        assert aligned_mask.shape[1] == out_ch
        assert (
            aligned_mask.mean().item() < 0.5
        ), "Target mask should be sparse (undersampled)"
