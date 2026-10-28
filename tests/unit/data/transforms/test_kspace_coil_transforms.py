"""Regression tests for :mod:`spectramr.data.transforms.kspace_coil_transforms`.

Audit anchor: 2026-05-14 F5 (centro_symmetric mosaic class).

Background
----------
The legacy ``CoilCombineTransform(method="rss")`` does

    IFFT → RSS-magnitude → FFT-back-to-k-space

The third step ``fft2c(|x|)`` is *phase-stripping*: the resulting k-space
is Hermitian-symmetric, so any downstream IFFT-then-magnitude produces a
centro-symmetric ("doubled brain") image. This was the signature flagged
in 10 of 89 smoke-run mosaics on 2026-05-14.

The fix adds ``method="rss_image"`` which stops at step 2 and returns the
real magnitude image directly.

These tests:

- pin the new ``rss_image`` mode's output domain (image, 1-channel real),
- assert it is NOT centro-symmetric for a clearly-asymmetric coil image
  (the legacy ``rss`` mode IS centro-symmetric on the same input),
- exercise the transform via the ``tio.Subject`` apply path used by the
  TorchIO transform builder.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torchio as tio

from spectramr.data.transforms.kspace_coil_transforms import CoilCombineTransform
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c


def _make_asymmetric_coil_kspace(
    n_coils: int = 4, H: int = 16, W: int = 16, D: int = 2
) -> torch.Tensor:
    """Build an asymmetric multi-coil k-space tensor.

    Image-domain: a single bright pixel in the upper-left quadrant per coil
    (different intensity per coil). Then FFT to k-space. The resulting
    magnitude image is asymmetric — a clean negative-control for the
    centro-symmetry test.
    """
    torch.manual_seed(0)
    img = torch.zeros(n_coils, D, H, W, dtype=torch.complex64)
    for c in range(n_coils):
        img[c, :, 3 + c % 2, 4 + c] = torch.complex(
            torch.tensor(1.0 + 0.25 * c), torch.tensor(0.0)
        )
    ksp_chw = fft2c(img)            # (C, D, H, W) complex
    return ksp_chw.permute(0, 2, 3, 1)  # (C, H, W, D) complex


def _is_hermitian_magnitude(k: torch.Tensor, atol: float = 1e-4) -> bool:
    """Return True if ``|K[i,j]| == |K[(-i)%H, (-j)%W]|`` everywhere.

    This is the Hermitian-symmetry property of the k-space of a real
    signal — ``fft2c(real)`` always satisfies it. ``torch.flip(k)``
    places ``k[N-1-i]`` where Hermitian symmetry asks for ``k[-i]``
    (i.e. ``k[(N-i) % N]``), so we ``roll`` by 1 to align the indices.
    """
    flipped = torch.flip(k, dims=(-2, -1))
    flipped = torch.roll(flipped, shifts=(1, 1), dims=(-2, -1))
    return torch.allclose(k, flipped, atol=atol, rtol=0.0)


def test_rss_image_combine_returns_real_image_domain() -> None:
    """``_rss_image_combine`` returns a single-channel real magnitude tensor."""
    ksp = _make_asymmetric_coil_kspace()
    t = CoilCombineTransform(method="rss_image")
    out = t._rss_image_combine(ksp)

    assert not torch.is_complex(out), "rss_image must be real-valued"
    assert out.shape == (1, 16, 16, 2), f"unexpected shape {out.shape}"
    assert torch.all(out >= 0), "RSS magnitude must be non-negative"


def test_rss_image_avoids_centro_symmetry_regression() -> None:
    """The new ``rss_image`` mode does NOT produce centro-symmetric output.

    Regression for audit-2026-05-14 F5.

    Two distinct properties, pinned independently:

    1. Legacy ``_rss_combine`` returns *k-space* whose magnitude is
       Hermitian-symmetric (rotationally symmetric about the centre).
       That k-space tensor was being rendered directly by the validation
       writer as if it were an image — hence the "doubled brain".
    2. New ``_rss_image_combine`` returns an *image-domain* magnitude
       tensor whose magnitude is NOT centro-symmetric on an
       intentionally-asymmetric input.

    The combination — legacy output is centro-symmetric magnitude, fixed
    output is not — is what proves the fix actually changes the visual.
    """
    ksp = _make_asymmetric_coil_kspace()

    # Legacy ``rss`` — Hermitian k-space whose magnitude IS centro-symmetric.
    legacy = CoilCombineTransform(method="rss")
    legacy_ksp = legacy._rss_combine(ksp)              # (1, H, W, D) complex
    legacy_mag = legacy_ksp.abs().permute(0, 3, 1, 2)  # (1, D, H, W) real

    # New ``rss_image`` — image-domain magnitude, no FFT round-trip.
    fixed = CoilCombineTransform(method="rss_image")
    fixed_img = fixed._rss_image_combine(ksp).permute(0, 3, 1, 2)  # (1, D, H, W)

    assert _is_hermitian_magnitude(legacy_mag), (
        "Sanity check: legacy ``rss`` mode returns Hermitian k-space whose "
        "magnitude must satisfy |K[i,j]| == |K[-i,-j]| (audit-2026-05-14 F5 "
        "signature). If this fails, the legacy phase-strip pitfall has "
        "shifted — re-verify the FFT(magnitude) round-trip on ``_rss_combine``."
    )
    assert not _is_hermitian_magnitude(fixed_img), (
        "Regression: new ``rss_image`` magnitude is Hermitian-symmetric. "
        "The FFT(magnitude) round-trip has crept back into ``_rss_image_combine``."
    )


def test_rss_image_apply_transform_pipeline() -> None:
    """End-to-end through a ``tio.Subject``: the image is converted in-place."""
    ksp = _make_asymmetric_coil_kspace()
    subject = tio.Subject(
        input=tio.ScalarImage(tensor=ksp, affine=np.eye(4)),
    )
    t = CoilCombineTransform(method="rss_image", keys=["input"])
    out_subj = t(subject)

    out_data = out_subj["input"].data
    assert not torch.is_complex(out_data), "rss_image must produce real tensor"
    assert out_data.shape[0] == 1, (
        f"rss_image must collapse coils to 1 channel; got C={out_data.shape[0]}"
    )


def test_unknown_method_raises() -> None:
    """Unknown ``method`` raises — keeps CLAUDE.md #9 (no silent fallbacks)."""
    ksp = _make_asymmetric_coil_kspace()
    t = CoilCombineTransform(method="not_a_real_method")  # type: ignore[arg-type]
    subject = tio.Subject(input=tio.ScalarImage(tensor=ksp, affine=np.eye(4)))
    try:
        t(subject)
    except ValueError as e:
        assert "Unknown method" in str(e)
    else:
        raise AssertionError("Expected ValueError for unknown method")


# ============================================================================ #
# F1 regression — single-coil real pass-through must NOT warn
#
# `tests_experiments/smoke_test/smoke_test_all_20260516_111315.log` showed
# 8,604 lines of CoilCombine warnings — six fingerprints all firing when
# single-coil real data reached the transform. CLAUDE.md pitfall #10
# treats those warnings as silent regressions. The pass-through for the
# "already combined" case (1 or 2 channel real) is the EXPECTED behavior
# when the transform is mounted on a single-coil dataset; it is no longer
# warning-worthy. Real data with >2 channels is genuinely ambiguous and
# must raise.
#
# Audit anchor: TODO/audit/smoke_audit_20260516.md F1.
# ============================================================================ #


def test_single_coil_real_passes_through_without_warning(caplog) -> None:
    """1-channel real input is treated as already-combined; no WARNING."""
    import logging as _logging
    data = torch.randn(1, 2, 16, 16)  # (1 channel, D=2, H=16, W=16)
    subject = tio.Subject(input=tio.ScalarImage(tensor=data, affine=np.eye(4)))
    t = CoilCombineTransform(method="rss", keys=["input"])

    with caplog.at_level(_logging.WARNING, logger="spectramr.data.transforms.kspace_coil_transforms"):
        out = t(subject)

    # Data passes through unchanged.
    assert torch.equal(out["input"].data, data), (
        "single-coil real input must pass through unchanged"
    )
    # No WARNING records — the previous code emitted "cannot combine. Skipping."
    coilcombine_warns = [
        r for r in caplog.records
        if r.levelno >= _logging.WARNING and "CoilCombine" in r.getMessage()
    ]
    assert not coilcombine_warns, (
        f"Expected no CoilCombine WARNINGs for single-coil real input, "
        f"got: {[r.getMessage() for r in coilcombine_warns]}"
    )


def test_two_channel_real_already_combined_passes_through(caplog) -> None:
    """2-channel real input (interleaved R/I) is pass-through; no WARNING."""
    import logging as _logging
    data = torch.randn(2, 2, 16, 16)  # (2 channels = real/imag, D=2, H=16, W=16)
    subject = tio.Subject(input=tio.ScalarImage(tensor=data, affine=np.eye(4)))
    t = CoilCombineTransform(method="rss", keys=["input"])

    with caplog.at_level(_logging.WARNING, logger="spectramr.data.transforms.kspace_coil_transforms"):
        out = t(subject)

    assert torch.equal(out["input"].data, data)
    coilcombine_warns = [
        r for r in caplog.records
        if r.levelno >= _logging.WARNING and "CoilCombine" in r.getMessage()
    ]
    assert not coilcombine_warns


def test_three_channel_real_raises_value_error() -> None:
    """3-channel real input is genuinely ambiguous and MUST raise.

    Per CLAUDE.md pitfall #9: silent fallbacks forbidden. The previous
    code emitted a warning and skipped; the correct behavior is to fail
    loud so the YAML / pipeline is fixed at audit-time.
    """
    data = torch.randn(3, 2, 16, 16)  # (3 channels, D=2, H=16, W=16)
    subject = tio.Subject(input=tio.ScalarImage(tensor=data, affine=np.eye(4)))
    t = CoilCombineTransform(method="rss", keys=["input"])

    try:
        t(subject)
    except (ValueError, RuntimeError) as e:
        # TorchIO may wrap the ValueError in a RuntimeError; check the chain.
        msg = str(e) + " " + str(e.__cause__ or "")
        assert "3 channels" in msg or "cannot disambiguate" in msg, (
            f"Expected ValueError naming the 3-channel cause, got: {e!r}"
        )
    else:
        raise AssertionError(
            "Expected ValueError for 3-channel real input (ambiguous)"
        )


def test_sense_combine_raises_when_no_sensitivity_maps() -> None:
    """``method='sense'`` with no sensitivity maps MUST raise (CLAUDE.md #9).

    The previous behaviour silently logged a warning and fell back to RSS,
    delivering a physically different output (RSS loses coil phase coherence)
    to a caller who explicitly asked for SENSE. NN#3: no silent fallbacks.
    """
    ksp = _make_asymmetric_coil_kspace(n_coils=4)
    subject = tio.Subject(input=tio.ScalarImage(tensor=ksp, affine=np.eye(4)))
    # No 'sensitivity' key in the subject → sensitivity is None at combine time.
    t = CoilCombineTransform(method="sense", keys=["input"])
    try:
        t(subject)
    except (ValueError, RuntimeError) as e:
        # TorchIO may wrap the ValueError in a RuntimeError; check the chain.
        msg = str(e) + " " + str(e.__cause__ or "")
        assert "requires sensitivity maps" in msg, (
            f"Expected ValueError naming the missing sensitivity maps, got: {e!r}"
        )
    else:
        raise AssertionError(
            "Expected ValueError for method='sense' with no sensitivity maps"
        )


def test_sense_combine_direct_call_raises_without_sensitivity() -> None:
    """Direct ``_sense_combine(kspace, None)`` raises a clear ValueError."""
    ksp = _make_asymmetric_coil_kspace(n_coils=4)
    t = CoilCombineTransform(method="sense", keys=["input"])
    try:
        t._sense_combine(ksp, None)
    except ValueError as e:
        assert "requires sensitivity maps" in str(e)
        assert "sensitivity" in str(e)
    else:
        raise AssertionError("Expected ValueError from _sense_combine(..., None)")


def test_complex_multi_coil_input_still_combines() -> None:
    """The pass-through path must NOT short-circuit valid complex input.

    Regression guard: F1 added a fail-loud branch for real-with->2-channels,
    but the existing complex multi-coil path must still produce the
    canonical 2-channel real (R, I) k-space output.
    """
    ksp = _make_asymmetric_coil_kspace(n_coils=4)
    subject = tio.Subject(input=tio.ScalarImage(tensor=ksp, affine=np.eye(4)))
    t = CoilCombineTransform(method="rss", keys=["input"])

    out = t(subject)
    out_data = out["input"].data

    assert not torch.is_complex(out_data), "rss output is real R/I-interleaved"
    assert out_data.shape[0] == 2, (
        f"rss output must be 2-channel (R, I); got C={out_data.shape[0]}"
    )


# ---------------------------------------------------------------------------
# rss_per_channel — Shen 2024's own coil combination (#2093)
# ---------------------------------------------------------------------------


def test_rss_per_channel_matches_the_authors_own_call() -> None:
    """The mode reproduces ``fastmri.rss`` on the upstream's ``[Nc,H,W,2]`` layout.

    This is the oracle for a lifted behaviour: there is no upstream call left to
    spy on, so equivalence to the original at a fixed seed is the evidence.
    """
    fastmri = pytest.importorskip("fastmri")

    torch.manual_seed(0)
    kspace = torch.randn(4, 16, 16, 1, dtype=torch.complex64)

    combined = CoilCombineTransform(method="rss_per_channel")._rss_per_channel_combine(kspace)
    ours = ifft2c(combined.permute(0, 3, 1, 2))[0, 0]

    # Shen's path: ifft2c per coil, then fastmri.rss over the coil axis of
    # a real/imag-last tensor (utils/data_transform.py:51 and :62).
    per_coil = ifft2c(kspace.permute(0, 3, 1, 2))
    theirs = fastmri.rss(torch.stack([per_coil.real, per_coil.imag], dim=-1)[:, 0])

    assert torch.allclose(theirs[..., 0], ours.real, atol=1e-5)
    assert torch.allclose(theirs[..., 1], ours.imag, atol=1e-5)


def test_rss_per_channel_leaves_both_channels_non_negative() -> None:
    """``sqrt(sum(x**2))`` per channel cannot return a negative part.

    Pins the property that separates this mode from a complex-magnitude RSS.
    Swapping the implementation to ``fastmri.rss_complex`` turns this red, which
    is the point: the published numbers came from the per-channel reduction.
    """
    torch.manual_seed(1)
    kspace = torch.randn(4, 16, 16, 1, dtype=torch.complex64)

    combined = CoilCombineTransform(method="rss_per_channel")._rss_per_channel_combine(kspace)
    image = ifft2c(combined.permute(0, 3, 1, 2))

    assert image.real.min() >= 0.0
    assert image.imag.min() >= 0.0


def test_rss_per_channel_is_not_the_magnitude_rss() -> None:
    """Non-vacuity: the new mode is a different reduction, not a second spelling."""
    torch.manual_seed(2)
    kspace = torch.randn(4, 16, 16, 1, dtype=torch.complex64)

    transform = CoilCombineTransform(method="rss_per_channel")
    per_channel = transform._rss_per_channel_combine(kspace)
    magnitude = transform._rss_combine(kspace)

    assert per_channel.shape == magnitude.shape
    assert not torch.allclose(per_channel, magnitude, atol=1e-4)


def test_rss_per_channel_apply_transform_emits_two_real_channels() -> None:
    """The subject path returns ``(2, H, W, D)`` real, the layout the arms need."""
    kspace = _make_asymmetric_coil_kspace(n_coils=4, H=16, W=16, D=2)
    subject = tio.Subject(kspace=tio.ScalarImage(tensor=kspace))

    out = CoilCombineTransform(method="rss_per_channel")(subject)["kspace"].data

    assert out.shape == (2, 16, 16, 2)
    assert not torch.is_complex(out)
