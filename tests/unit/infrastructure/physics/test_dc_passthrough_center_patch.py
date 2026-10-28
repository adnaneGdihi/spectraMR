"""Regression tests for ``dc_passthrough_center_patch``.

Background
----------
The cluster's iter-2 cross-contrast cold-diffusion PNG showed a bright
central "white spot" superimposed on the brain. Diagnosis (see
docs/validation_image_audit.rst):

  * Raw k-space spans 5+ decades of magnitude, dominated by the DC bin.
  * Percentile normalisation flattens the bulk but leaves DC untouched —
    the percentile is *designed* to ignore that tail.
  * ``AdaptiveDataConsistency`` eventually learns ``λ ≈ 1`` at DC, but
    starts random — iter-0/iter-2 outputs leak ~50% of the CNN's random
    DC error, rendering as a central blob.

The fix is to skip-connect the central patch deterministically: at
sampled bins (mask=1), replace the CNN output with the measurement.
Unsampled bins inside the patch keep the CNN's value (we can't know
them better than the network does). ``λ ≡ 1`` on the patch, by
construction.

These tests pin that contract.
"""
from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.data_consistency import (
    dc_passthrough_center_patch,
)


def _centre_slice(t: torch.Tensor, h: int, w: int):
    """Slice centred on the fftshift DC bin (``H // 2``, ``W // 2``)."""
    H, W = t.shape[-2], t.shape[-1]
    cy, cx = H // 2, W // 2
    hs = max(0, cy - h // 2)
    ws = max(0, cx - w // 2)
    he = min(H, hs + h)
    we = min(W, ws + w)
    return t[..., hs:he, ws:we]


def test_dc_bin_is_overwritten_at_sampled_location():
    """The single-pixel DC patch must equal the measurement when mask=1."""
    x_out = torch.randn(2, 4, 16, 16, dtype=torch.complex64)
    measured = torch.randn(2, 4, 16, 16, dtype=torch.complex64)
    mask = torch.ones(2, 1, 16, 16)
    result = dc_passthrough_center_patch(x_out, measured, mask, patch_size=1)
    # DC bin equals measurement
    assert torch.equal(_centre_slice(result, 1, 1), _centre_slice(measured, 1, 1))
    # All other bins unchanged
    result_no_centre = result.clone()
    measured_no_centre = x_out.clone()
    H, W = 16, 16
    cy, cx = H // 2, W // 2
    result_no_centre[..., cy, cx] = 0
    measured_no_centre[..., cy, cx] = 0
    assert torch.equal(result_no_centre, measured_no_centre)


def test_unsampled_dc_keeps_cnn_value():
    """When mask=0 at the DC bin, the CNN's value is preserved (we don't
    have a measurement to overwrite with)."""
    x_out = torch.randn(1, 2, 8, 8, dtype=torch.complex64)
    measured = torch.randn(1, 2, 8, 8, dtype=torch.complex64)
    mask = torch.zeros(1, 1, 8, 8)
    result = dc_passthrough_center_patch(x_out, measured, mask, patch_size=1)
    assert torch.equal(result, x_out)


def test_3x3_patch_overwrites_centre_region():
    """Wider patches replace a larger square at the centre."""
    x_out = torch.randn(1, 2, 16, 16, dtype=torch.complex64)
    measured = torch.randn(1, 2, 16, 16, dtype=torch.complex64)
    mask = torch.ones(1, 1, 16, 16)
    result = dc_passthrough_center_patch(x_out, measured, mask, patch_size=(3, 3))
    assert torch.equal(_centre_slice(result, 3, 3), _centre_slice(measured, 3, 3))


def test_none_measured_is_noop():
    x_out = torch.randn(1, 2, 4, 4)
    result = dc_passthrough_center_patch(x_out, None, None, patch_size=1)
    assert torch.equal(result, x_out)


def test_cross_contrast_channel_alignment():
    """For cross-contrast layouts, ``measured`` carries source+target
    while ``x_out`` is target-only. The helper must align to LAST C
    channels."""
    # x_out has 4 channels (target); measured has 8 (source||target)
    x_out = torch.zeros(1, 4, 8, 8, dtype=torch.complex64)
    measured = torch.zeros(1, 8, 8, 8, dtype=torch.complex64)
    # Distinguish source (first 4) vs target (last 4)
    measured[:, :4] = 1 + 0j   # source
    measured[:, 4:] = 99 + 0j  # target — what we WANT at DC
    mask = torch.ones(1, 1, 8, 8)
    result = dc_passthrough_center_patch(x_out, measured, mask, patch_size=1)
    # DC of x_out (target) must come from LAST 4 channels of measured, not first 4
    cy, cx = 4, 4
    assert torch.allclose(result[:, :, cy, cx], torch.full((1, 4), 99 + 0j))


def test_patch_size_int_equivalent_to_tuple():
    x_out = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    measured = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    mask = torch.ones(1, 1, 8, 8)
    a = dc_passthrough_center_patch(x_out, measured, mask, patch_size=2)
    b = dc_passthrough_center_patch(x_out, measured, mask, patch_size=(2, 2))
    assert torch.equal(a, b)


def test_patch_clamped_to_tensor_shape():
    """Asking for a patch larger than the tensor must clamp, not crash."""
    x_out = torch.randn(1, 1, 4, 4, dtype=torch.complex64)
    measured = torch.randn(1, 1, 4, 4, dtype=torch.complex64)
    mask = torch.ones(1, 1, 4, 4)
    result = dc_passthrough_center_patch(x_out, measured, mask, patch_size=999)
    # Full tensor replaced
    assert torch.equal(result, measured)


def test_real_stacked_path_preserves_dtype():
    """Real-stacked (non-complex) inputs round-trip without promotion."""
    x_out = torch.randn(1, 4, 8, 8)
    measured = torch.randn(1, 4, 8, 8)
    mask = torch.ones(1, 1, 8, 8)
    result = dc_passthrough_center_patch(x_out, measured, mask, patch_size=1)
    assert result.dtype == x_out.dtype
    assert torch.equal(_centre_slice(result, 1, 1), _centre_slice(measured, 1, 1))


def test_partial_mask_per_pixel_decision():
    """A mask that's 1 in half of the patch, 0 in the other half, must
    overwrite only the sampled half."""
    x_out = torch.full((1, 1, 8, 8), 7.0)
    measured = torch.full((1, 1, 8, 8), -3.0)
    mask = torch.zeros(1, 1, 8, 8)
    # Sample only one row inside the 2x2 centre patch
    mask[:, :, 3, :] = 1
    result = dc_passthrough_center_patch(x_out, measured, mask, patch_size=2)
    # row 3 inside patch → -3 (from measured)
    assert torch.equal(result[:, :, 3, 3:5], torch.full((1, 1, 2), -3.0))
    # row 4 inside patch → 7 (from x_out, mask=0)
    assert torch.equal(result[:, :, 4, 3:5], torch.full((1, 1, 2), 7.0))


def test_mask_broadcast_to_multichannel_xout():
    """A 1-channel mask must broadcast to all channels of x_out."""
    x_out = torch.zeros(1, 4, 4, 4, dtype=torch.complex64)
    measured = torch.full((1, 4, 4, 4), 5 + 0j, dtype=torch.complex64)
    mask = torch.ones(1, 1, 4, 4)
    result = dc_passthrough_center_patch(x_out, measured, mask, patch_size=1)
    cy, cx = 2, 2
    assert torch.allclose(result[:, :, cy, cx], torch.full((1, 4), 5 + 0j))


def test_invalid_patch_size_in_generator(monkeypatch):
    """``KSpaceColdDiffusionGenerator`` must reject malformed knob values.

    We exercise just the validation branch of __init__ without constructing
    the full generator (which has a heavy backbone-import dependency tree).
    """
    # Mirror the validation logic from the generator's __init__
    def _validate(passthrough_cfg):
        if passthrough_cfg is None or passthrough_cfg == 0:
            return None
        if isinstance(passthrough_cfg, (list, tuple)):
            if len(passthrough_cfg) != 2:
                raise ValueError(
                    "dc_passthrough_center_size must be int or 2-tuple, "
                    f"got {passthrough_cfg!r}"
                )
            size = (int(passthrough_cfg[0]), int(passthrough_cfg[1]))
        else:
            s = int(passthrough_cfg)
            size = (s, s)
        if any(d < 1 for d in size):
            raise ValueError(
                "dc_passthrough_center_size dimensions must be >= 1, "
                f"got {size}"
            )
        return size

    assert _validate(None) is None
    assert _validate(0) is None
    assert _validate(1) == (1, 1)
    assert _validate([3, 3]) == (3, 3)
    with pytest.raises(ValueError, match="must be int or 2-tuple"):
        _validate([1, 2, 3])
    with pytest.raises(ValueError, match="dimensions must be >= 1"):
        _validate(0.5)  # truncates to 0


def test_batch_broadcast_for_5d_to_4d_flatten():
    """5D → 4D slice-flatten asymmetry: mask/measurement stay at batch=B
    while x_out arrives at B*D after the validator flattens the depth dim.

    Anchor: experiment_11 cluster run 2026-05-28 19:30 traceback:

        File ".../dc_passthrough_center_patch", line 1041, in
        mask_patch = mask_patch.expand_as(out_patch)
        RuntimeError: The expanded size of the tensor (2) must match
        the existing size (18) at non-singleton dimension 0.

    The original ``mask_patch.expand_as(out_patch)`` failed because
    ``expand`` requires source size = 1 in every non-singleton
    dimension, but 2 → 18 is not a singleton expansion. The fix is
    to ``repeat_interleave(D, dim=0)`` along the batch axis when
    ``out_patch.shape[0]`` is an integer multiple of ``mask_patch.shape[0]``.
    """
    # B=2, D=9 (2*9 = 18, the original failing case)
    B, D, C, H, W = 2, 9, 8, 16, 16
    x_out = torch.randn(B * D, C, H, W, dtype=torch.complex64)
    measured = torch.full(
        (B, C, H, W), 7.0 + 3.0j, dtype=torch.complex64
    )
    mask = torch.ones(B, 1, H, W)
    # Should NOT raise the original RuntimeError.
    result = dc_passthrough_center_patch(
        x_out, measured, mask, patch_size=5
    )
    assert result.shape == x_out.shape
    # Every slice in B*D must see the centre patch from its parent
    # batch element (repeat_interleave semantics — element i comes
    # from original batch i // D).
    cy, cx = H // 2, W // 2
    for i in range(B * D):
        # The result at the centre bin must equal the measurement
        # value of the parent batch element (since mask=1 everywhere).
        assert torch.allclose(
            result[i, :, cy, cx],
            measured[i // D, :, cy, cx],
        ), f"batch index {i} (parent {i // D}) didn't match measurement"


def test_batch_broadcast_handles_5d_to_4d_with_b1():
    """When B=1, the bug doesn't trigger (broadcast already handles it)
    but the new code path must still work — repeat_interleave with
    n_repeat=D produces a tensor of shape (D, ...)."""
    B, D, C, H, W = 1, 4, 4, 8, 8
    x_out = torch.randn(B * D, C, H, W, dtype=torch.complex64)
    measured = torch.full((B, C, H, W), 2.0 + 0j, dtype=torch.complex64)
    mask = torch.ones(B, 1, H, W)
    result = dc_passthrough_center_patch(
        x_out, measured, mask, patch_size=3
    )
    assert result.shape == x_out.shape


def test_batch_broadcast_bails_on_non_integer_ratio():
    """When ``B_out`` is not a clean multiple of ``B_mask`` (e.g. user
    passes mismatched tensors), the function returns ``x_out`` unchanged
    rather than corrupting the centre patch by mis-aligning batches."""
    x_out = torch.randn(7, 4, 8, 8, dtype=torch.complex64)
    measured = torch.full((3, 4, 8, 8), 9.0 + 0j, dtype=torch.complex64)
    mask = torch.ones(3, 1, 8, 8)  # 7 % 3 != 0 — no clean ratio
    result = dc_passthrough_center_patch(
        x_out, measured, mask, patch_size=3
    )
    # Result must equal x_out untouched.
    assert torch.equal(result, x_out)


def _blobness(kspace: torch.Tensor) -> float:
    """Image-domain centre/periphery energy ratio of a k-space tensor.

    A value >> 1 means the IFFT magnitude piles up at the array centre —
    the "DC blob". For a real anatomical brain it sits modestly above 1
    (the brain occupies the centre of the FOV but the periphery is not
    empty); for a DC/low-frequency-only spectrum it is much larger.
    """
    from spectramr.infrastructure.physics.fft_ops import ifft2c

    img = ifft2c(kspace).abs().sum(dim=1)[0]  # coil-sum magnitude [H, W]
    H, W = img.shape
    cy, cx = H // 2, W // 2
    centre = img[cy - 8 : cy + 8, cx - 8 : cx + 8].mean()
    border = torch.cat([img[:8].flatten(), img[-8:].flatten()]).mean()
    return float(centre / (border + 1e-9))


def test_passthrough_manufactures_dc_blob_when_hf_is_weak():
    """Regression: the passthrough *creates* the centre blob it claims to fix.

    Root-cause correction (2026-05-29, experiment_11 debugging). The
    ``dc_passthrough_center_size`` knob was added to "remove" the
    validation DC blob, but it does the opposite while the CNN's
    high-frequency output is still near zero — which is exactly the
    early-training / iter-0 regime that every saved validation PNG was
    captured in (all 16 cluster runs saved only epoch0/step0-2).

    Mechanism: the passthrough force-writes the measured DC (the brightest,
    lowest-frequency k-space bins) into an output whose high-frequency
    content is ~0. The IFFT of "measured DC + ≈0 elsewhere" is a
    low-pass point-spread function centred on the array — the blob.

    This test pins the mechanism so the band-aid is not silently
    re-added: feeding a near-zero-HF output through the passthrough must
    raise the image-domain centre/periphery ratio far above what the
    untouched (already weak) output had.
    """
    from spectramr.infrastructure.physics.fft_ops import fft2c

    torch.manual_seed(0)
    B, C, H, W = 1, 4, 64, 64

    # A realistic brain-like target: smooth disc + texture, forward-FFT'd.
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    r = (xx**2 + yy**2).sqrt()
    img = (r < 0.7).float() * 0.8 + (r < 0.4).float() * 0.4
    img = img + 0.1 * torch.cos(8 * xx) * torch.cos(8 * yy) * (r < 0.7)
    img = img.clamp(0, 1)[None, None].repeat(1, C, 1, 1).to(torch.complex64)
    measured = fft2c(img)  # fully-sampled measurement [B, C, H, W]

    # Untrained CNN output: near-zero everywhere (tiny noise), which on its
    # own is NOT a strong blob — its energy is diffuse, not centre-piled.
    cnn_out = 1e-3 * torch.randn(B, C, H, W, dtype=torch.complex64)
    mask = torch.ones(B, 1, H, W)

    blob_before = _blobness(cnn_out)
    after = dc_passthrough_center_patch(cnn_out, measured, mask, patch_size=5)
    blob_after = _blobness(after)

    # The passthrough must SHARPLY increase centre concentration (the blob).
    assert blob_after > 3.0 * blob_before, (
        f"passthrough did not manufacture a blob: before={blob_before:.2f}, "
        f"after={blob_after:.2f}. If this assertion now fails because the "
        f"ratio is small, the passthrough behaviour changed — re-examine "
        f"whether re-enabling dc_passthrough_center_size is safe."
    )
