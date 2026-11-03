"""Render a recorded tensor as a ``(B, 1, H, W)`` magnitude preview, in its own domain."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch

from spectramr.core.snapshot_trace import ChannelSegment
from spectramr.infrastructure.training.utils.domain_inference import (
    looks_like_kspace as _looks_like_kspace,
)

logger = logging.getLogger(__name__)


def _is_in_kspace_domain(x: torch.Tensor) -> bool:
    """Is a complex tensor in k-space domain (-> needs an IFFT to view as image)?

    Decides by the **spectrum**, not the raw DC magnitude. A non-negative
    image FFTs to a sharp-DC k-space; a k-space — even one whose DC has been
    flattened by ``data.normalize_kspace`` (phase-preserving ``log1p`` /
    robust magnitude compression) — FFTs to a spread, no-DC field. So ``x`` is
    already image-domain iff ``fft2c(x)`` carries the k-space DC signature.

    This is **normalization-invariant**, unlike the legacy
    ``_looks_like_kspace(x.abs())`` raw-DC test, which false-negated on
    normalized k-space and rendered the experiment_11 debug ``target.png`` as
    raw k-space (DC blob / "doubled brain", 2026-06-17 downloaded-run triage).
    VF / IB image-output targets still FFT to a sharp DC, so they are
    correctly recognised as image-domain and left untransformed.
    """
    from spectramr.infrastructure.physics.fft_ops import fft2c

    return not _looks_like_kspace(fft2c(x).abs())


def split_channel_segments(
    name: str,
    t: torch.Tensor,
    segments: Sequence[ChannelSegment],
) -> list[tuple[str, torch.Tensor, bool, bool | None]] | None:
    """Slice a superposed tensor into ``(render_name, sub, is_kspace, log)``.

    ``log`` is the segment's declared compression state, or ``None`` to inherit
    the parent tensor's -- see :data:`ChannelSegment` for why a segment may need
    to disagree with its own domain.

    Returns ``None`` when the declaration cannot be honoured -- no channel axis,
    or the segment widths do not sum to the tensor's channel count. The caller
    then renders the tensor whole, having logged why. Silently re-normalising a
    mis-declared split would put a wrong picture under a right-looking filename,
    which is the failure mode this whole mechanism exists to remove.
    """
    if t.dim() < 2:
        return None
    total = t.shape[1]
    declared = sum(seg[1] for seg in segments)
    if declared != total:
        logger.warning(
            "[snapshot] channel_segments for %r declare %d channels but "
            "the tensor has %d; rendering it whole. The split is a claim about "
            "what the emitting strategy concatenated -- fix the declaration at "
            "the call site rather than the tensor.",
            name,
            declared,
            total,
        )
        return None

    out: list[tuple[str, torch.Tensor, bool, bool | None]] = []
    start = 0
    for seg in segments:
        label, width, is_kspace = seg[0], seg[1], seg[2]
        log_compressed = seg[3] if len(seg) == 4 else None
        out.append(
            (
                f"{name}__{label}",
                t[:, start : start + width],
                is_kspace,
                log_compressed,
            )
        )
        start += width
    return out


def render_image_preview(
    t: torch.Tensor,
    *,
    in_kspace: bool,
    authoritative: bool = False,
    log_scaled: bool = False,
) -> torch.Tensor | None:
    """Reduce a tensor to ``(B, 1, H, W)`` magnitude suitable for save_image.

    Args:
        t: Input tensor of any layout.
        in_kspace: When True, apply a centred IFFT before magnitude.
        authoritative: When True, the ``in_kspace`` flag came from the config
            SSOT (``model.target_domain`` / ``KNOWN_KSPACE_OUTPUT_MODELS`` via
            ``needs_ifft_for_visualization``), not a name/substring guess — so
            the IFFT is applied UNCONDITIONALLY and the fragile spectrum veto
            (``_is_in_kspace_domain``) is bypassed. The veto false-negates on
            real *normalized multicoil* M4Raw k-space and rendered the
            experiment_11 ``input_prepared`` / ``model_output`` snapshots as
            raw, off-centre k-space instead of a brain (2026-06-27 audit).
        log_scaled: When True, ``data.processing.enable_log_scaling`` compressed
            this k-space with ``log1p``, so the magnitude is decompressed before
            the IFFT (#682). Without it the transform runs on a per-bin
            nonlinearity and the render is a high-pass artifact, not a scaled
            image. Threaded in as an argument because this module receives
            ``config.logging`` and cannot reach ``config.data``.

    Returns:
        ``(B, 1, H, W)`` real magnitude in [0, 1] *per-sample* normalised,
        or ``None`` if the tensor is not displayable as an image.
    """
    try:
        x = t.detach().cpu()
        if x.dim() == 5:
            # 3-D volume -> take central slice along the depth axis
            x = x[..., x.shape[-1] // 2]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            return None

        # Re-pair real-stacked complex -> complex tensor ONLY when the caller
        # has declared this tensor to be k-space. In image domain we cannot
        # safely assume an even channel count means (R,I) interleaved: it
        # might be (ULF,HF) paired modalities, or multi-contrast stacked
        # channels, or per-coil magnitudes already RSS'd elsewhere. RSS of
        # the raw real-stacked channels is mathematically identical to the
        # per-coil-magnitude+RSS chain for properly-encoded complex data
        # (sqrt(Σ R_c² + Σ I_c²) = sqrt(Σ |C_c|²)), so we lose nothing by
        # skipping the pairing in image domain.
        if in_kspace and not x.is_complex() and x.shape[1] >= 2 and x.shape[1] % 2 == 0:
            ev = x[:, 0::2]
            od = x[:, 1::2]
            x = torch.complex(ev, od)

        # Optional IFFT (k-space -> image). Guard against a spurious IFFT on a
        # tensor that is flagged k-space but is already image domain (VF / IB
        # image-output arms): only transform when the tensor is genuinely
        # k-space. The decision is spectrum-based (see _is_in_kspace_domain),
        # so it is invariant to data.normalize_kspace's DC-flattening — the
        # raw-DC test used here previously false-negated on normalized k-space
        # and rendered experiment_11 debug target.png as raw k-space (DC blob /
        # "doubled brain", 2026-06-17), while still IFFT'ing an image-domain
        # complex target into a black DC-spike PNG was the older failure (smoke
        # 2026-06-12) the guard prevents.
        if in_kspace and x.is_complex():
            if authoritative or _is_in_kspace_domain(x):
                from spectramr.infrastructure.physics.fft_ops import ifft2c
                from spectramr.infrastructure.training.utils.kspace_view import (
                    decompress_for_view,
                )

                # Undo log1p BEFORE the transform (#682). A no-op when the arm
                # does not compress, so the call is unconditional.
                x = decompress_for_view(x, log_scaled=log_scaled, channel_dim=1)
                x = ifft2c(x)
            else:
                logger.debug(
                    "debug-snapshot: tensor flagged k-space but its spectrum "
                    "carries the k-space DC signature (already image domain); "
                    "skipping IFFT."
                )

        # Magnitude
        x = x.abs() if x.is_complex() else x.float().abs()

        # Multi-channel -> RSS reduce. Works for per-coil magnitudes,
        # real-stacked complex (mathematically equivalent), and is the
        # least-wrong fallback for paired modalities (still mixes them,
        # but at least produces a deterministic single-channel image; the
        # caller should slice to one modality upstream when that matters).
        if x.dim() == 4 and x.shape[1] > 1:
            x = torch.sqrt((x**2).sum(dim=1, keepdim=True) + 1e-12)

        # Per-sample [0, 1]
        b = x.shape[0]
        mn = x.view(b, -1).min(1, keepdim=True)[0].view(b, 1, 1, 1)
        mx = x.view(b, -1).max(1, keepdim=True)[0].view(b, 1, 1, 1)
        x = (x - mn) / (mx - mn + 1e-12)
        return x.clamp(0, 1)
    except Exception as exc:
        logger.debug("debug-snapshot preview failed for shape %s: %s", tuple(t.shape), exc)
        return None


#: A declared-k-space tensor whose rendered *image* has at least this fraction
#: of interior rows-or-columns fully dark is flagged as a domain superposition
#: (a k-space line mask applied in the image domain). A correct k-space
#: undersampling IFFTs to a *dense* aliased image (~0 zero lines); only a
#: line mask multiplied against IMAGE data zeroes whole image rows/cols.
MASK_IN_IMAGE_MIN_ZERO_FRAC = 0.30


def image_domain_mask_fraction(preview: torch.Tensor) -> float:
    """Fraction of interior image rows-or-cols that are ~fully zero.

    Operates on the FINAL rendered preview ``(B, 1, H, W)`` in ``[0, 1]`` — the
    exact image a reviewer looks at. This is the k-space-mask-in-image-domain
    signature: applying a Cartesian phase-encode line mask to IMAGE data zeroes
    whole rows (or columns) of the image, whereas applying it to k-space and
    IFFT-ing produces a dense, aliased image with no zero lines.

    The FOV borders (outer 15% each side) are excluded — air rows there are
    legitimately dark. A degenerate (near-constant / all-zero) preview returns
    ``0.0`` so a blank tensor is never misread as striped.

    Returns ``max(row_fraction, col_fraction)`` (a mask lives on one axis).
    """
    if preview is None or preview.dim() != 4 or preview.shape[1] != 1:
        return 0.0
    x = preview.detach().float()
    if not torch.isfinite(x).all():
        return 0.0
    if float(x.max() - x.min()) < 1e-6:  # constant / blank frame: no line structure
        return 0.0
    _, _, h, w = x.shape
    eps = 1e-4
    # A row/col is "dark" iff it is dark across every sample and the other axis.
    row_max = x.amax(dim=(0, 1, 3))  # (H,)
    col_max = x.amax(dim=(0, 1, 2))  # (W,)
    ry0, ry1 = int(0.15 * h), int(0.85 * h)
    cx0, cx1 = int(0.15 * w), int(0.85 * w)
    interior_rows = row_max[ry0:ry1]
    interior_cols = col_max[cx0:cx1]
    row_frac = float((interior_rows < eps).float().mean()) if interior_rows.numel() else 0.0
    col_frac = float((interior_cols < eps).float().mean()) if interior_cols.numel() else 0.0
    return max(row_frac, col_frac)


__all__ = [
    "MASK_IN_IMAGE_MIN_ZERO_FRAC",
    "image_domain_mask_fraction",
    "render_image_preview",
    "split_channel_segments",
]
