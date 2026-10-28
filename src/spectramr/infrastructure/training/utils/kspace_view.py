"""One decompression step for every k-space *visualization* path (#682).

With ``data.processing.enable_kspace_normalization`` on,
``enable_log_scaling`` compresses k-space magnitude with ``m -> log1p(m)``
(``data/transforms/normalization.py::compress_kspace_log``).
The inverse, :func:`decompress_kspace_log`, had exactly ONE caller in the tree --
inside the diffusion strategy, gated on ``is_cold_diffusion`` -- so every other
arm inverse-FFT'd the *compressed* spectrum.

That is not a scaled image. ``log1p`` is a per-bin nonlinearity, so it reweights
the spectrum before the transform: the low-|k| bins that carry contrast are
squashed toward the high-|k| bins that carry noise, and the IFFT of the result is
a high-pass-like artifact dominated by whatever DC survives. The render looks like
a plausible-but-wrong reconstruction, which is why it went unnoticed -- the
snapshot is *supposed* to show something odd early in training.

This module exists so the step is written once. `debug_snapshot` in particular
cannot reach it any other way: it receives ``config.logging`` and has no access to
``config.data`` at all, which is why the flag is threaded in as an argument rather
than read here.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = ["decompress_for_view", "decompress_scored_against", "log_scaling_enabled"]


def log_scaling_enabled(config: Any) -> bool:
    """Whether this arm's k-space reaches the model ``log1p``-compressed.

    Resolved by :attr:`KSpaceNormalizationSpec.log_compressed`, not read off
    ``enable_log_scaling``: the compression runs only when
    ``enable_kspace_normalization`` is also on, and the declaration alone would
    ``expm1`` a spectrum that was never compressed (#2353).

    Returns ``False`` when there is no ``data.processing`` block at all -- a
    strategy constructed standalone in a unit test, or a paradigm with no data
    section. Defaulting to False is the safe direction there: it renders exactly
    what the caller passed, rather than applying ``expm1`` to a tensor that was
    never compressed (which would blow the dynamic range apart).

    But when the block DOES exist and a normalization knob does not, this
    raises. A plain ``getattr(..., False)`` over a *declared* field is how a
    rename disables a mechanism in silence: the guard keeps the old spelling,
    every call returns False, #682 comes back, and nothing goes red because
    "absent" and "off" are the same boolean. Absent-block and absent-field are
    different facts and get different answers.

    Raises:
        AttributeError: ``data.processing`` exists but omits one of the knobs
            :meth:`KSpaceNormalizationSpec.from_data_config` resolves -- the
            field was renamed or removed, and every k-space visualization is
            silently rendering a compressed spectrum.
    """
    from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

    data = getattr(config, "data", None)
    processing = getattr(data, "processing", None) if data is not None else None
    if processing is None:
        return False
    return KSpaceNormalizationSpec.from_data_config(data).log_compressed


def decompress_scored_against(
    reference: torch.Tensor,
    others: dict[str, torch.Tensor | None],
    *,
    log_scaled: bool,
    channel_dim: int = 1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
    """Put a reference and every tensor scored against it on ONE level.

    :func:`decompress_for_view` decides per tensor, which is right for a render
    and wrong for a metric: a prediction whose compressed |k|max passes
    ``DECOMPRESS_MAGNITUDE_CEILING`` was left compressed while its target was
    decompressed, so the two were scored in different units. Here the decision is
    made once, on ``reference`` (the pipeline's target, whose compression is
    known), and the same ``decompress_kspace_log`` is applied to every tensor.
    Its clamp turns a diverged prediction into an honestly bad finite score.

    Args:
        reference: The target the others are compared with.
        others: Name -> tensor to transform identically; ``None`` passes through.
        log_scaled: Whether ``compress_kspace_log`` was applied upstream.
        channel_dim: Channel axis for real-stacked input; ignored for complex.

    Returns:
        ``(reference, others)`` after the shared decision.
    """
    decided = decompress_for_view(reference, log_scaled=log_scaled, channel_dim=channel_dim)
    if decided is reference:
        return reference, dict(others)

    from spectramr.data.transforms.normalization import decompress_kspace_log

    return decided, {
        name: None if t is None else decompress_kspace_log(t, channel_dim=channel_dim)
        for name, t in others.items()
    }


def decompress_for_view(x: torch.Tensor, *, log_scaled: bool, channel_dim: int = 1) -> torch.Tensor:
    """Undo log-magnitude compression before an IFFT, if it was applied.

    A no-op when ``log_scaled`` is False, so every call site can invoke it
    unconditionally instead of repeating the branch.

    ``log_scaled`` is a *declaration*, and this verifies it against the tensor
    before acting on it. The two can disagree -- ``data.processing`` says the arm
    compresses k-space, but a given snapshot may have captured a tensor from the
    uncompressed side of that transform -- and acting on the declaration alone is
    what rendered ``experiment_11_attention_none``'s ground truth as an edge map:
    ``expm1`` under the ``DECOMPRESS_MAGNITUDE_CEILING`` clamp collapses every
    coefficient above the ceiling to the SAME magnitude while phase survives, so
    the IFFT draws a phase-only reconstruction with no low-frequency content.

    **Do not "fix" a bad render by narrowing the caller's ``log_scaled_keys``**
    (e.g. passing ``set()`` for ``diffusion_step`` in ``strategies/base.py``).
    That tag's tensors are post-normalization *by design*, so once the arm's
    normalization actually runs they WILL be compressed and all-keys ``expm1``
    becomes the correct behaviour. Narrowing the key set fixes today's pictures
    and silently re-introduces #682 later; the magnitude check below is what
    distinguishes the two cases at the moment of rendering, every time.

    Args:
        x: Complex or real-stacked interleaved k-space.
        log_scaled: Whether ``compress_kspace_log`` was applied upstream, as
            resolved by :func:`log_scaling_enabled`.
        channel_dim: Channel axis for real-stacked input; ignored for complex.

    Returns:
        The tensor in physical k-space magnitude, ready for ``ifft2c``.
    """
    if not log_scaled:
        return x

    # Only complex or even-channel (interleaved real/imag) tensors are k-space in
    # this codebase's layout; anything else was not compressed and must not be
    # expm1'd. Mirrors the guard the diffusion strategy already applies.
    if not (torch.is_complex(x) or (x.dim() >= 2 and x.shape[channel_dim] % 2 == 0)):
        return x

    from spectramr.data.transforms.normalization import (
        DECOMPRESS_MAGNITUDE_CEILING,
        decompress_kspace_log,
    )

    before = float(x.abs().max())

    # A compressed magnitude cannot plausibly exceed the ceiling: `log1p` of a
    # few hundred is <= ~6, and the constant's own docstring states that 30.0
    # "never touches legitimate data". Above it, `expm1` is guaranteed to be the
    # wrong operation -- it would clamp the whole band to `expm1(30) ~ 1e13` --
    # so skip it and say so. The previous guard fired only when decompression
    # failed to EXPAND `|k|max`; spurious `expm1` always expands, so the
    # catastrophic direction was the one direction that passed in silence.
    if before > DECOMPRESS_MAGNITUDE_CEILING:
        logger.warning(
            "[kspace-view] refusing to decompress: |k|max %.4g exceeds the "
            "compressed-domain ceiling %.4g, so expm1 would clamp the entire "
            "contrast-carrying band to one magnitude and render a phase-only "
            "image. Rendering the tensor as-is. Two things produce this, and "
            "they need different fixes: (a) for a pipeline tensor (target, "
            "input_*, noisy_kspace) it means the tensor was declared log-scaled "
            "but never compressed -- check that the arm's k-space normalization "
            "actually reached this tensor; (b) for a model_output* tensor it may "
            "instead mean the prediction DIVERGED in compressed units (this "
            "arm's kernelized-attention sibling reached ~1750 at iter 1000). "
            "Either way the render is not a decompression artifact.",
            before,
            DECOMPRESS_MAGNITUDE_CEILING,
        )
        return x

    out = decompress_kspace_log(x, channel_dim=channel_dim)
    after = float(out.abs().max())
    # expm1 is strictly expanding above 0, so |k|max MUST grow. These two host
    # transfers used to feed a `logger.debug` line and nothing else: the comment
    # stated the tell while the code never tested it, so at the default log level
    # they bought nothing at all. Now they buy the check they describe -- a
    # no-op decompression means the render is still a compressed-domain artifact,
    # which is #682 wearing the fix's own clothes and is invisible in the image.
    if before > 0.0 and after <= before:
        logger.warning(
            "[kspace-view] decompression did not expand |k|max (%.4g -> %.4g). "
            "The tensor was declared log-scaled but expm1 changed nothing, so "
            "this render is still a compressed spectrum (#682). Check "
            "`data.processing.enable_log_scaling` against what the pipeline "
            "actually applied.",
            before,
            after,
        )
    else:
        logger.debug("[kspace-view] decompressed |k|max %.4g -> %.4g", before, after)
    return out
