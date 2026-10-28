"""Receive-chain amplitude axes: gain drift, ADC clipping, ADC quantisation, coil dropout.

Gain, full scale and clip level are per subject (the whole stack or volume shares one ADC range
and one scan), so none of these factorises over slices and all are native in every mode.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _ensure_complex
from spectramr.infrastructure.physics.degradations.catalogue.receive_common import (
    acquisition_ranks,
    check_n_coils,
    control_generator,
    positive,
    receive_op,
    subject_peak,
    two_d,
)
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.fft_ops import fftnc, ifftnc
from spectramr.infrastructure.physics.per_sample_rng import per_sample_rand

__all__ = [
    "adc_clipping",
    "adc_clipping_2d",
    "adc_enob",
    "adc_quantization",
    "adc_quantization_2d",
    "apply_line_gain",
    "coil_dropout",
    "coil_dropout_2d",
    "drift_gain_map",
    "receiver_gain_drift",
    "receiver_gain_drift_2d",
]

_DROPOUT_MODES = ("full", "partial")


def drift_gain_map(
    tau: Tensor, drift: float, step_exp: Tensor, t_step: Tensor, *, per_slice: bool = True
) -> Tensor:
    """Per-line gain referenced to the k-space centre, ``[B, 1, D, H, 1]``.

    ``g = (1 + drift tau) 2^(step_exp [tau >= t_step]) / g(centre)``: a slow linear drift plus a
    receiver gain-stage switch of ``2^step_exp`` (6 dB per unit exponent). The image intensity
    scale is set by the gain on the centre line, so the centre has gain exactly 1 and only the
    line-to-line modulation remains. The centre is line ``H // 2`` of each slice (``per_slice``,
    2-D and multislice) or the ``(D // 2, H // 2)`` sample (3-D). ``tau`` is ``[D, H]`` (fraction
    of the scan), ``step_exp`` and ``t_step`` are ``[B]``.
    """
    on = (tau[None] >= t_step.view(-1, 1, 1)).to(tau.dtype)
    g = (1.0 + drift * tau[None]) * torch.exp2(step_exp.view(-1, 1, 1) * on)
    h, d = tau.shape[1], tau.shape[0]
    centre = g[:, :, h // 2 : h // 2 + 1] if per_slice else g[:, d // 2, h // 2].view(-1, 1, 1)
    return (g / centre)[:, None, :, :, None]


def apply_line_gain(x: Tensor, gain: Tensor, dims: tuple[int, ...]) -> Tensor:
    """Scale every k-space line by its gain (``[B, 1, D, H, 1]`` or broadcastable)."""
    return ifftnc(fftnc(x, dims) * gain.to(x.real.dtype), dims)


@receive_op
def receiver_gain_drift(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    dims: tuple[int, ...],
    *,
    drift_max: float = 0.05,
    step_max_db: float = 6.0,
) -> Tensor:
    """Uncorrected receiver gain-stage switch plus a slow drift, applied to every k-space line.

    The dominant failure is a discrete gain switch (stages of about 6 dB): the lines after a
    seeded time ``t0`` are scaled by ``2^(+-theta)`` for ``step_max_db = 6`` (sign seeded), on top of
    a linear drift ``1 + 0.05 theta t/T``. Both are divided by the gain at the k-space centre line
    so the centre has gain 1 (see ``drift_gain_map``). The gain is common to all coils and
    multiplies the k-space line acquired at ``t`` (Bernstein et al., Handbook of MRI Pulse
    Sequences, 2004, ch. 13: gain and amplitude instability produce line-to-line modulation, i.e.
    ghosts along phase encode). Line times come from ``line_times`` for the acquisition mode.
    """
    drift = theta * positive("drift_max", drift_max)
    step_exp = theta * positive("step_max_db", step_max_db) / 6.0
    if theta <= 0.0:
        return x
    x = _ensure_complex(x)
    b, _, d, h, _ = x.shape
    _, tau = acquisition_ranks(geometry, d, h, x.device)
    u = per_sample_rand(b, 2, generator=control_generator(x.device, seed), device=x.device)
    sign = torch.where(u[:, 0] < 0.5, -1.0, 1.0).to(tau.dtype)
    per_slice = geometry.mode is not AcquisitionMode.VOLUME_3D
    gain = drift_gain_map(tau, drift, sign * step_exp, u[:, 1].to(tau.dtype), per_slice=per_slice)
    return apply_line_gain(x, gain, dims)


@receive_op
def adc_clipping(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    dims: tuple[int, ...],
    *,
    depth: float = 0.9,
) -> Tensor:
    """ADC saturation: Re and Im of k-space clipped at ``(1 - theta depth)`` of the subject's peak.

    The converter has a fixed full scale and clips symmetrically beyond it (Bernstein et al.
    2004, ch. 13); the clip level is relative to the per-subject peak ``|Re|`` or ``|Im|``, so
    theta = 0 clips nothing.
    """
    if not 0.0 < depth < 1.0:
        raise ValueError(
            f"depth must lie in (0, 1) so the clip level stays positive, got {depth!r}"
        )
    if theta <= 0.0:
        return x
    x = _ensure_complex(x)
    k = fftnc(x, dims)
    level = (1.0 - theta * depth) * subject_peak(k)
    re = torch.minimum(torch.maximum(k.real, -level), level)
    im = torch.minimum(torch.maximum(k.imag, -level), level)
    return ifftnc(torch.complex(re, im), dims)


def adc_enob(theta: float, enob_max: float = 16.0, span: float = 12.0) -> float:
    """Effective number of bits ``16 - 12 theta``: real-valued, as for a real ADC."""
    return enob_max - span * theta


@receive_op
def adc_quantization(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    dims: tuple[int, ...],
) -> Tensor:
    """Uniform ADC quantisation of Re and Im at ``ENOB = 16 - 12 theta`` effective bits.

    Mid-tread quantiser, ``delta = P / 2^(ENOB - 1)`` for the subject's full scale ``P`` (its
    peak ``|Re|`` or ``|Im|``), so the error is bounded by ``delta/2`` with variance
    ``delta^2/12`` for a busy signal (Widrow & Kollar, Quantization Noise, Cambridge Univ. Press,
    2008, ch. 4). ENOB is continuous, so every theta is a distinct level; theta = 0 (16 bit)
    bypasses the quantiser exactly.
    """
    if theta <= 0.0:
        return x
    x = _ensure_complex(x)
    k = fftnc(x, dims)
    step = subject_peak(k) / 2.0 ** (adc_enob(theta) - 1.0)

    def quantise(v: Tensor) -> Tensor:
        return torch.round(v / step) * step

    return ifftnc(torch.complex(quantise(k.real), quantise(k.imag)), dims)


@receive_op
def coil_dropout(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    dims: tuple[int, ...],
    *,
    mode: str = "full",
    n_coils: int | None = None,
    max_fraction: float = 0.5,
) -> Tensor:
    """A seeded subset of coils goes dead: for the whole scan (``full``) or from a seeded line on.

    ``ceil(theta max_fraction C)`` coils, capped at ``C // 2`` so a coil always survives, are
    chosen by a seeded ranking (nested in theta). ``partial`` zeroes them from a seeded line in
    acquisition order (a cable or preamplifier failing mid-scan); the other coils are untouched.
    """
    if mode not in _DROPOUT_MODES:
        raise ValueError(f"coil_dropout mode must be one of {_DROPOUT_MODES}, got {mode!r}")
    check_n_coils(x, n_coils)
    max_fraction = positive("max_fraction", max_fraction)
    if max_fraction > 0.5:
        raise ValueError(f"max_fraction must be <= 0.5, got {max_fraction}")
    if theta <= 0.0:
        return x
    b, c, d, h, _ = x.shape
    n_drop = min(math.ceil(theta * max_fraction * c - 1e-9), c // 2)
    u = per_sample_rand(b, c + 1, generator=control_generator(x.device, seed), device=x.device)
    coil_rank = u[:, :c].argsort(dim=1).argsort(dim=1)
    dead = (coil_rank < n_drop).view(b, c, 1, 1, 1)
    x = _ensure_complex(x)
    if mode == "full":
        return torch.where(dead, torch.zeros_like(x), x)
    rank, _ = acquisition_ranks(geometry, d, h, x.device)
    first = (u[:, c] * (d * h)).floor().long().view(b, 1, 1)
    lines = (rank[None] >= first).view(b, 1, d, h, 1)
    k = fftnc(x, dims)
    out = ifftnc(torch.where(dead & lines, torch.zeros_like(k), k), dims)
    return torch.where(dead, out, x)


receiver_gain_drift_2d = two_d(receiver_gain_drift)
adc_clipping_2d = two_d(adc_clipping)
adc_quantization_2d = two_d(adc_quantization)
coil_dropout_2d = two_d(coil_dropout)
