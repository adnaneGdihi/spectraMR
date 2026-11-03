r"""Rep-BV by spatial frequency: which scales an estimator keeps, and which it loses.

A dominant Rep-BV squared bias (:mod:`~spectramr.core.metrics.rep_bias_variance`)
reads as an output smoother than the anatomy or as a reference that moved between
repetitions. Radial k-space bands separate the two: blur takes energy from the high
band, a change of tissue contrast shows in the low band.

Bands use the radius normalised to Nyquist on each axis,
:math:`\rho=\sqrt{(k_y/(H/2))^2+(k_x/(W/2))^2}`: ``low`` below 1/8 (periods above
16 pixels, the scale of tissue contrast), ``mid`` from 1/8 to 3/8, ``high`` from
3/8 (periods under 5.3 pixels: edges and fine detail). The three partition the
grid, and every energy is divided by the batch's pixel count, so the bands of a
column sum to the Rep-BV total it splits.

With :math:`\hat f,\hat t` the (orthonormal) k-space of prediction and reference and
:math:`\psi` the reference's global phase, per band :math:`K`:

``band_corrected_mse_<K>``
    :math:`\sum_{k\in K}\lVert\hat f-e^{i\psi}\hat t\rVert^2-\nu_K`; the bands sum
    to ``reference_noise_corrected_mse``.
``band_bias_squared_<K>``
    The corrected band error minus the band's repetition variance; the bands sum
    to ``bias_squared``.
``band_cross_power_<K>``
    :math:`\sum_{k\in K}\operatorname{Re}\langle\hat f,e^{i\psi}\hat t\rangle`. The
    reference's noise is zero-mean and independent of :math:`f`, so this estimates
    :math:`\operatorname{Re}\langle\hat f,\hat x\rangle_K` with no correction.
``band_signal_deficit_<K>``
    The reference's noise-corrected power minus the cross power: the share of the
    reference's signal in :math:`K` the output does not reproduce.
``band_input_cross_power_<K>``
    The cross power of the model's k-space input :math:`y`. For a fully sampled
    input repetition it estimates the signal input and reference share, motion
    between them included, with no noise model at all.

The band transfer :math:`T_K=\text{cross}/(\text{cross}+\text{deficit})` is
:math:`g` for an output that scales the band by :math:`g`, whatever noise it keeps.
Form it from column means over slices or subjects, never per slice: the corrected
high-band reference power is a small difference of large terms. On a fully sampled
input, cross / input cross is the transfer relative to the identity with no noise
model; the identity's own deficit-route transfer is the reference floor per band,
lowered by motion between repetitions as well as by any noise-estimate bias.

**Noise per band.** :math:`\nu_K=\kappa\sum_{k\in K}\lVert F(\sqrt w\,P^{\perp}t)\rVert^2/\sum w`
(:math:`w=\lVert s\rVert^2`, :math:`\kappa=C/(C-1)`) spreads the null-space estimate
``reference_noise_corrected_mse`` subtracts over k-space with its own spectrum, so
noise confined to the acquired columns stays there; the bands sum to the total.

Cross powers and deficit carry the directions of an estimator that shrinks; an
amplified band reads a negative deficit, which ``band_corrected_mse`` charges.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch import Tensor

from spectramr.core.metrics.coil_manifold import (
    as_complex_coils,
    require_complex_maps,
    unit_maps,
)
from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.registry import register_metric
from spectramr.core.metrics.rep_bias_variance import _align, _context
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

__all__ = ["BAND_EDGES", "BAND_NAMES", "band_masks", "band_terms"]

#: Upper radius of ``low`` and ``mid`` on the Nyquist-normalised radius.
BAND_EDGES: tuple[float, float] = (0.125, 0.375)
BAND_NAMES: tuple[str, str, str] = ("low", "mid", "high")


@lru_cache(maxsize=8)
def band_masks(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """``[3, H, W]`` indicator of each band on the centred ``fft2c`` grid."""
    ky = (torch.arange(height, device=device, dtype=dtype) - height // 2) / (height / 2)
    kx = (torch.arange(width, device=device, dtype=dtype) - width // 2) / (width / 2)
    rho = torch.sqrt(ky[:, None].square() + kx[None, :].square())
    low, mid = BAND_EDGES
    return torch.stack([rho < low, (rho >= low) & (rho < mid), rho >= mid]).to(dtype)


def _power(z: Tensor) -> Tensor:
    """Coil-summed ``|z|^2`` per bin, ``[B, H, W]``."""
    return (z.real.square() + z.imag.square()).sum(1)


def _band_sums(values: Tensor) -> Tensor:
    """``[B, H, W]`` per-bin values -> ``[3]`` band sums per pixel of the batch."""
    b, h, w = values.shape
    masks = band_masks(h, w, values.device, values.dtype)
    return torch.einsum("bhw,khw->k", values, masks) / (b * h * w)


def _kspace(tensor: Tensor | None, *, field: str, metric: str) -> Tensor:
    if tensor is None:
        raise ValueError(f"{metric} needs MetricContext.{field} and it was not supplied.")
    return as_complex_coils(tensor, name=field)


def _reference_noise(t: Tensor, maps: Tensor | None, *, metric: str) -> Tensor:
    """``[3]`` noise energy of the reference per band, from its coil null space."""
    images = ifft2c(t)
    maps = require_complex_maps(maps, images, metric=metric)
    coils = images.shape[1]
    if coils < 2:
        raise ValueError(f"{metric}: a single coil has no null space to read noise from.")
    unit, norm2 = unit_maps(maps)
    residual = images - unit * (unit.conj() * images).sum(1, keepdim=True)
    spectrum = _power(fft2c(residual * norm2.sqrt().unsqueeze(1)))
    total = norm2.sum()
    scale = coils / (coils - 1) * norm2.numel() / total.clamp(min=torch.finfo(total.dtype).tiny)
    noise = _band_sums(spectrum) * scale
    return torch.where(total > 0, noise, torch.full_like(noise, float("nan")))


def _pair(metric: str, prediction: Tensor, target: Tensor | None) -> tuple[Tensor, Tensor]:
    """``(f, e^{i psi} t)``: complex k-space of both, the reference phase-aligned to ``f``."""
    f = _kspace(prediction, field="prediction", metric=metric)
    t = _kspace(target, field="target", metric=metric)
    if t.shape != f.shape:
        raise ValueError(f"{metric}: target {tuple(t.shape)} vs prediction {tuple(f.shape)}.")
    return f, _align(t, f)


def _rep_variance(
    metric: str, prediction: Tensor, target: Tensor | None, ctx: MetricContext
) -> Tensor:
    del target
    f = _kspace(prediction, field="prediction", metric=metric)
    g = _kspace(ctx.pred_alt, field="pred_alt", metric=metric)
    if g.shape != f.shape:
        raise ValueError(f"{metric}: pred_alt {tuple(g.shape)} vs prediction {tuple(f.shape)}.")
    return 0.5 * _band_sums(_power(f - _align(g, f)))


def _cross_power(
    metric: str, prediction: Tensor, target: Tensor | None, ctx: MetricContext
) -> Tensor:
    del ctx
    f, t = _pair(metric, prediction, target)
    return _band_sums((f * t.conj()).real.sum(1))


def _input_cross_power(
    metric: str, prediction: Tensor, target: Tensor | None, ctx: MetricContext
) -> Tensor:
    del prediction
    y = _kspace(ctx.y_kspace, field="y_kspace", metric=metric)
    return _cross_power(metric, y, target, ctx)


def _signal_deficit(
    metric: str, prediction: Tensor, target: Tensor | None, ctx: MetricContext
) -> Tensor:
    f, t = _pair(metric, prediction, target)
    noise = _reference_noise(t, ctx.coil_maps, metric=metric)
    return _band_sums(_power(t)) - noise - _band_sums((f * t.conj()).real.sum(1))


def _corrected_mse(
    metric: str, prediction: Tensor, target: Tensor | None, ctx: MetricContext
) -> Tensor:
    f, t = _pair(metric, prediction, target)
    return _band_sums(_power(f - t)) - _reference_noise(t, ctx.coil_maps, metric=metric)


def _bias_squared(
    metric: str, prediction: Tensor, target: Tensor | None, ctx: MetricContext
) -> Tensor:
    corrected = _corrected_mse(metric, prediction, target, ctx)
    return corrected - _rep_variance(metric, prediction, target, ctx)


_TERMS = {
    "corrected_mse": _corrected_mse,
    "bias_squared": _bias_squared,
    "cross_power": _cross_power,
    "signal_deficit": _signal_deficit,
    "input_cross_power": _input_cross_power,
}


def band_terms(
    metric: str, term: str, prediction: Tensor, target: Tensor | None, ctx: MetricContext
) -> Tensor:
    """``[3]`` values of ``term`` (low, mid, high) for one batch; an unknown term raises."""
    if term not in _TERMS:
        raise ValueError(f"{metric}: unknown band term {term!r}; known: {sorted(_TERMS)}.")
    return _TERMS[term](metric, prediction, target, ctx)


class _BandMetric:
    """One band of one term: ``band_terms(...)[band]``."""

    name, term, band = "", "", 0

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> Tensor:
        ctx = _context(self.name, context, kwargs)
        return band_terms(self.name, self.term, prediction, target, ctx)[self.band]


def _contract(needs: tuple[str, ...], direction: str) -> dict[str, Any]:
    # The context flag routes the map-free cross power through the k-space path too.
    flags = {"requires_reference": True, "requires_measurement_context": True}
    return {**flags, "needs": needs, "direction": direction}


_MSE = _DEFICIT = _contract(("coil_maps",), "lower")
_BIAS = _contract(("coil_maps", "pred_alt"), "lower")
_CROSS = _contract((), "higher")
_INPUT = _contract(("y_kspace",), "higher")


@register_metric("band_corrected_mse_low", **_MSE)
class BandCorrectedMSELow(_BandMetric):
    name, term, band = "band_corrected_mse_low", "corrected_mse", 0


@register_metric("band_corrected_mse_mid", **_MSE)
class BandCorrectedMSEMid(_BandMetric):
    name, term, band = "band_corrected_mse_mid", "corrected_mse", 1


@register_metric("band_corrected_mse_high", **_MSE)
class BandCorrectedMSEHigh(_BandMetric):
    name, term, band = "band_corrected_mse_high", "corrected_mse", 2


@register_metric("band_bias_squared_low", **_BIAS)
class BandBiasSquaredLow(_BandMetric):
    name, term, band = "band_bias_squared_low", "bias_squared", 0


@register_metric("band_bias_squared_mid", **_BIAS)
class BandBiasSquaredMid(_BandMetric):
    name, term, band = "band_bias_squared_mid", "bias_squared", 1


@register_metric("band_bias_squared_high", **_BIAS)
class BandBiasSquaredHigh(_BandMetric):
    name, term, band = "band_bias_squared_high", "bias_squared", 2


@register_metric("band_cross_power_low", **_CROSS)
class BandCrossPowerLow(_BandMetric):
    name, term, band = "band_cross_power_low", "cross_power", 0


@register_metric("band_cross_power_mid", **_CROSS)
class BandCrossPowerMid(_BandMetric):
    name, term, band = "band_cross_power_mid", "cross_power", 1


@register_metric("band_cross_power_high", **_CROSS)
class BandCrossPowerHigh(_BandMetric):
    name, term, band = "band_cross_power_high", "cross_power", 2


@register_metric("band_signal_deficit_low", **_DEFICIT)
class BandSignalDeficitLow(_BandMetric):
    name, term, band = "band_signal_deficit_low", "signal_deficit", 0


@register_metric("band_signal_deficit_mid", **_DEFICIT)
class BandSignalDeficitMid(_BandMetric):
    name, term, band = "band_signal_deficit_mid", "signal_deficit", 1


@register_metric("band_signal_deficit_high", **_DEFICIT)
class BandSignalDeficitHigh(_BandMetric):
    name, term, band = "band_signal_deficit_high", "signal_deficit", 2


@register_metric("band_input_cross_power_low", **_INPUT)
class BandInputCrossPowerLow(_BandMetric):
    name, term, band = "band_input_cross_power_low", "input_cross_power", 0


@register_metric("band_input_cross_power_mid", **_INPUT)
class BandInputCrossPowerMid(_BandMetric):
    name, term, band = "band_input_cross_power_mid", "input_cross_power", 1


@register_metric("band_input_cross_power_high", **_INPUT)
class BandInputCrossPowerHigh(_BandMetric):
    name, term, band = "band_input_cross_power_high", "input_cross_power", 2
