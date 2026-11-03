"""Real acquisition-asset preparation for the sim2rank ``MetricContext``.

The sim2rank meta-evaluation grades a no-reference / physics metric battery that
needs the *measurement* (acquired k-space, coil maps, sampling geometry) and a
*reconstructor* — not just a magnitude image. When the input is genuine
multi-coil k-space (fastMRI brain ``.h5``), those assets exist, but the pipeline
historically collapsed everything to a p99-normalized magnitude reference and let
``_build_sample_context`` **fabricate** a synthetic birdcage stand-in. Metrics
were then graded against a made-up acquisition.

This module assembles a :class:`MetricContext` from the *real* reference the
acquisition provides — the exact ``(coil_images, smaps, x_gt_mag, p99)`` tuple
:func:`spectramr.data.datasets.pseudo_gt_reference.synthesize_pseudo_gt`
returns. Because ``coil_images`` is the image-domain reference, the fully-sampled
acquired k-space is recovered exactly as ``fft2c(coil_images)`` (no producer
signature change needed). Fields the acquisition genuinely cannot supply
(``prior_model`` / ``encoder``) are left unset — a metric needing them stays
honestly ``not_applicable`` rather than scored against a stand-in.
"""

from __future__ import annotations

import torch
from torch import Tensor

from spectramr.core.metrics.context import MetricContext


def _to_bchw_complex(coil_images: Tensor) -> Tensor:
    """Coerce coil images to complex ``[1, C, H, W]``."""
    c = coil_images
    if c.dim() == 3:  # [C, H, W]
        c = c.unsqueeze(0)
    if c.dim() != 4:
        raise ValueError(f"coil_images must be [C,H,W] or [1,C,H,W]; got shape {tuple(c.shape)}")
    if not c.is_complex():
        c = c.to(torch.complex64)
    return c


def estimate_noise_covariance(coil_images: Tensor, corner_frac: float = 0.1) -> Tensor:
    """Per-coil noise covariance ``[C, C]`` from a background corner of the FOV.

    The top-left ``corner_frac`` x ``corner_frac`` block of the image-domain FOV
    is air (no anatomy) for a centred brain acquisition, so it samples the
    thermal receive noise. Returns the complex sample covariance across coils —
    the ``Sigma`` a CRLB metric (``qvcr``) needs. Mean-centred so a residual DC
    offset in the corner does not inflate the diagonal.
    """
    c = _to_bchw_complex(coil_images)
    _, ncoils, h, w = c.shape
    hh = max(1, int(round(h * corner_frac)))
    ww = max(1, int(round(w * corner_frac)))
    patch = c[0, :, :hh, :ww].reshape(ncoils, -1)  # [C, P] complex
    patch = patch - patch.mean(dim=1, keepdim=True)
    denom = max(patch.shape[1] - 1, 1)
    return (patch @ patch.conj().transpose(0, 1)) / denom  # [C, C] complex


def prepare_metric_context(
    *,
    coil_images: Tensor,
    smaps: Tensor,
    p99: float | Tensor = 1.0,
    acq_params: dict[str, object] | None = None,
    foreground_rel: float = 0.05,
) -> MetricContext:
    """Assemble a real-asset :class:`MetricContext` from a fully-sampled acquisition.

    Args:
        coil_images: complex image-domain coil images ``[1, C, H, W]`` (or
            ``[C, H, W]``) — the first element ``synthesize_pseudo_gt`` returns.
        smaps: real ESPIRiT sensitivity maps ``[1, C, H, W]`` complex.
        p99: the 99th-percentile scale the magnitude reference was divided by, so
            the k-space and reconstructor live on the same normalized scale as the
            graded (p99-normalized) images.
        acq_params: optional ``{TR, TE, flip_angle, ...}`` parsed from the
            acquisition header (unlocks ``brc``/``qvcr`` when present).
        foreground_rel: relative-threshold for the foreground mask ``Omega``.

    Returns:
        A :class:`MetricContext` carrying real ``coil_maps``, acquired
        ``y_kspace`` (fully sampled, ``mask=None``, ``acceleration=1.0``),
        ``foreground_mask``, a SENSE ``reconstructor``, and a ``noise_cov``
        estimate. Prior/encoder fields are left ``None`` on purpose.
    """
    from spectramr.infrastructure.physics.coil_sensitivity import coil_combine_sense
    from spectramr.infrastructure.physics.fft_ops import fft2c, sense_adjoint

    c = _to_bchw_complex(coil_images)
    smaps_c = smaps if smaps.dim() == 4 else smaps.unsqueeze(0)
    if isinstance(p99, Tensor):
        scale = float(p99.item())
    else:
        scale = float(p99)
    scale = scale if abs(scale) > 1e-12 else 1.0
    coil_n = c / scale  # match the p99-normalized image scale the metrics grade

    y_kspace = fft2c(coil_n)  # real acquired k-space, fully sampled

    if coil_n.shape[1] > 1:
        x_complex = coil_combine_sense(coil_n, smaps_c)
    else:
        x_complex = coil_n
    mag = x_complex.abs()
    foreground = (mag > foreground_rel * float(mag.amax())).float()

    noise_cov = estimate_noise_covariance(coil_n)

    def _recon(y: Tensor, _s: Tensor = smaps_c) -> Tensor:
        return sense_adjoint(y, _s)

    return MetricContext(
        coil_maps=smaps_c,
        y_kspace=y_kspace,
        mask=None,  # fully-sampled reference
        acceleration=1.0,
        foreground_mask=foreground,
        reconstructor=_recon,
        noise_cov=noise_cov,
        acq_params=acq_params,
    )


__all__ = [
    "estimate_noise_covariance",
    "prepare_metric_context",
]
