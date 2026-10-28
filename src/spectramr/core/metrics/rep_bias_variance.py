r"""Rep-BV: bias, variance and NEX-equivalence of a reconstruction, from repetitions.

All four read multi-coil **k-space** and work on coil images
(:func:`~spectramr.core.metrics.coil_manifold.coil_images`); every energy is the
coil-summed squared magnitude per pixel, averaged over all pixels, so the four
share one unit and ``bias_squared`` is a plain difference.

``rep_variance``
    :math:`\tfrac12\,\overline{\lVert f(y)-e^{i\varphi}f(y')\rVert^2}` for two
    independent repetitions :math:`y, y'` of one slice. When :math:`f(y), f(y')`
    are i.i.d. given the anatomy, :math:`E\lVert f(y)-f(y')\rVert^2 =
    2E\lVert f(y)-Ef(y)\rVert^2`, so this is an unbiased estimate of the
    estimator variance. :math:`\varphi` removes the per-acquisition receiver
    phase (the dot-product angle, ``_average_reps``' convention) -- one real
    degree of freedom out of :math:`2CHW`.

``reference_noise_corrected_mse``
    :math:`\overline{\lVert f-e^{i\psi}t\rVert^2}-\widehat{\operatorname{tr}\Sigma_e}`.
    With :math:`t=x+e`, :math:`e` independent of :math:`f`,
    :math:`E\lVert f-t\rVert^2=\lVert f-x\rVert^2+\operatorname{tr}\Sigma_e`.
    The reference's noise is read off its own coil null space: :math:`P^{\perp}x=0`,
    so :math:`P^{\perp}t=P^{\perp}e`, and for **isotropic** :math:`\Sigma_e=\sigma^2I`
    :math:`E\lVert P^{\perp}e\rVert^2=(C-1)\sigma^2`, giving
    :math:`\operatorname{tr}\Sigma_e\approx\frac{C}{C-1}\,\overline{\lVert
    P^{\perp}t\rVert^2}` (support-weighted, as ``null_space_residual``). Thermal
    noise under an orthonormal FFT is spatially stationary, which is what lets
    the support estimate stand for every pixel. With the committed per-coil
    :math:`\Sigma` one would use :math:`\operatorname{tr}(P^{\perp}\Sigma P^{\perp})`
    in place of :math:`(C-1)\sigma^2`. :math:`\psi` removes the reference's
    receiver phase, which the n2n leave-one-out targets carry (up to 176 deg on
    M4Raw) and no reconstruction can infer.

``nex_equivalent``
    The input's own noise floor, :math:`\frac{C}{C-1}\overline{\lVert
    P^{\perp}y\rVert^2}`, over the corrected MSE: 1 for the identity, :math:`N` for
    an :math:`N`-repetition average. Needs a fully sampled input -- aliasing is
    not rank one and would read as noise.

``bias_squared``
    Corrected MSE minus ``rep_variance``, from :math:`E\lVert f-x\rVert^2=
    \lVert Ef-x\rVert^2+E\lVert f-Ef\rVert^2`.

Every metric raises on a missing context field and returns a device scalar.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from spectramr.core.metrics.coil_manifold import (
    coil_images,
    null_space_energy,
    require_complex_maps,
    require_kspace_domain,
    weighted_mean,
)
from spectramr.core.metrics.context import MetricContext, resolve_context
from spectramr.core.metrics.registry import register_metric

__all__ = [
    "BiasSquared",
    "NexEquivalent",
    "ReferenceNoiseCorrectedMSE",
    "RepVariance",
]


def _energy(z: Tensor) -> Tensor:
    """Coil-summed ``|z|^2`` per pixel, ``[B, H, W]``."""
    return (z.real.square() + z.imag.square()).sum(1)


def _align(moving: Tensor, fixed: Tensor) -> Tensor:
    """``moving`` rotated by the per-sample global phase that best matches ``fixed``."""
    inner = (moving.conj() * fixed).sum(dim=(1, 2, 3), keepdim=True)
    magnitude = inner.abs()
    rotation = inner / magnitude.clamp(min=torch.finfo(magnitude.dtype).tiny)
    return moving * torch.where(magnitude > 0, rotation, torch.ones_like(rotation))


def _images(tensor: Tensor | None, *, field: str, metric: str) -> Tensor:
    if tensor is None:
        raise ValueError(f"{metric} needs MetricContext.{field} and it was not supplied.")
    return coil_images(tensor, name=field)


def _noise_trace(images: Tensor, maps: Tensor, *, metric: str) -> Tensor:
    """Per-pixel ``tr Sigma`` of ``images``' noise, from their coil null space."""
    coils = images.shape[1]
    if coils < 2:
        raise ValueError(f"{metric}: a single coil has no null space to read noise from.")
    residual, weights = null_space_energy(images, maps)
    return coils / (coils - 1) * weighted_mean(residual, weights)


def _context(metric: str, context: MetricContext | None, kwargs: dict[str, Any]) -> MetricContext:
    ctx = resolve_context(context, kwargs)
    require_kspace_domain(ctx.signal_domain, metric=metric)
    return ctx


def _rep_variance(metric: str, prediction: Tensor, ctx: MetricContext) -> Tensor:
    f = coil_images(prediction)
    g = _images(ctx.pred_alt, field="pred_alt", metric=metric)
    if g.shape != f.shape:
        raise ValueError(f"{metric}: pred_alt {tuple(g.shape)} vs prediction {tuple(f.shape)}.")
    return 0.5 * _energy(f - _align(g, f)).mean()


def _corrected_mse(
    metric: str, prediction: Tensor, target: Tensor | None, ctx: MetricContext
) -> Tensor:
    f = coil_images(prediction)
    t = _images(target, field="target", metric=metric)
    if t.shape != f.shape:
        raise ValueError(f"{metric}: target {tuple(t.shape)} vs prediction {tuple(f.shape)}.")
    maps = require_complex_maps(ctx.coil_maps, t, metric=metric)
    return _energy(f - _align(t, f)).mean() - _noise_trace(t, maps, metric=metric)


class _RepBVMetric:
    """Shared call shape: k-space prediction, optional reference, a context."""

    name = ""

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> Tensor:
        return self.compute(prediction, target, _context(self.name, context, kwargs))

    def compute(self, prediction: Tensor, target: Tensor | None, ctx: MetricContext) -> Tensor:
        raise NotImplementedError


@register_metric("rep_variance", requires_reference=False, needs=("pred_alt",), direction="lower")
class RepVariance(_RepBVMetric):
    """Estimator variance per pixel, from the model's outputs on two repetitions."""

    name = "rep_variance"

    def compute(self, prediction: Tensor, target: Tensor | None, ctx: MetricContext) -> Tensor:
        del target
        return _rep_variance(self.name, prediction, ctx)


@register_metric(
    "reference_noise_corrected_mse",
    requires_reference=True,
    needs=("coil_maps",),
    direction="lower",
)
class ReferenceNoiseCorrectedMSE(_RepBVMetric):
    """Coil-image MSE with the reference's own noise energy removed."""

    name = "reference_noise_corrected_mse"

    def compute(self, prediction: Tensor, target: Tensor | None, ctx: MetricContext) -> Tensor:
        return _corrected_mse(self.name, prediction, target, ctx)


@register_metric(
    "nex_equivalent",
    requires_reference=True,
    needs=("coil_maps", "y_kspace"),
    direction="higher",
)
class NexEquivalent(_RepBVMetric):
    """How many averaged repetitions the reconstruction's error is worth."""

    name = "nex_equivalent"

    def compute(self, prediction: Tensor, target: Tensor | None, ctx: MetricContext) -> Tensor:
        y = _images(ctx.y_kspace, field="y_kspace", metric=self.name)
        maps = require_complex_maps(ctx.coil_maps, y, metric=self.name)
        floor = _noise_trace(y, maps, metric=self.name)
        corrected = _corrected_mse(self.name, prediction, target, ctx)
        # A corrected MSE at or below zero (noise estimate >= error) has no repetition count; NaN, not inf or negative.
        return torch.where(corrected > 0, floor / corrected, torch.full_like(corrected, float("nan")))


@register_metric(
    "bias_squared",
    requires_reference=True,
    needs=("coil_maps", "pred_alt"),
    direction="lower",
)
class BiasSquared(_RepBVMetric):
    """Squared bias per pixel: corrected MSE minus the repetition variance."""

    name = "bias_squared"

    def compute(self, prediction: Tensor, target: Tensor | None, ctx: MetricContext) -> Tensor:
        corrected = _corrected_mse(self.name, prediction, target, ctx)
        return corrected - _rep_variance(self.name, prediction, ctx)
