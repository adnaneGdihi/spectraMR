r"""NuCert: the coil null-space residual as a reference-free error certificate.

For any reconstruction :math:`f` and the rank-one truth :math:`x=sm`,
:math:`\lVert f-x\rVert^2\ge\lVert P^{\perp}f\rVert^2` per pixel
(:mod:`~spectramr.core.metrics.coil_manifold`), so the null-space energy of the
output alone lower-bounds its error. No reference is read.

* ``null_space_residual`` -- the support-weighted mean of
  :math:`\lVert P^{\perp}f\rVert^2`, weights :math:`w=\lVert s\rVert^2` (ESPIRiT's
  soft-SENSE confidence), exactly the reduction ``coil_subspace_residual`` trains
  on. In the data's squared units.
* ``null_space_residual_fraction`` -- the same weighted sums of
  :math:`\lVert P^{\perp}f\rVert^2` over :math:`\lVert f\rVert^2`: scale-free, 0 on
  a realisable image and :math:`(C-1)/C` on isotropic noise.

Both read the prediction as multi-coil **k-space** -- ``[B, 2C, H, W]``
interleaved or ``[B, C, H, W]`` complex -- and take coil images through
``ifft2c``. Both raise when the maps are missing, real, or mis-shaped; neither
syncs to the host.
"""

from __future__ import annotations

from typing import Any

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

__all__ = ["NullSpaceResidual", "NullSpaceResidualFraction"]


def _residual_and_weights(
    metric: str, prediction: Tensor, context: MetricContext | None, kwargs: dict[str, Any]
) -> tuple[Tensor, Tensor, Tensor]:
    """``(||P_perp f||^2, ||f||^2, w)`` per pixel for the prediction."""
    ctx = resolve_context(context, kwargs)
    require_kspace_domain(ctx.signal_domain, metric=metric)
    images = coil_images(prediction)
    maps = require_complex_maps(ctx.coil_maps, images, metric=metric)
    residual, weights = null_space_energy(images, maps)
    energy = (images.real.square() + images.imag.square()).sum(1)
    return residual, energy, weights


@register_metric(
    "null_space_residual",
    requires_reference=False,
    needs=("coil_maps",),
    direction="lower",
)
class NullSpaceResidual:
    """Support-weighted mean coil null-space energy of the prediction (lower is better)."""

    name = "null_space_residual"

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> Tensor:
        del target
        residual, _, weights = _residual_and_weights(self.name, prediction, context, kwargs)
        return weighted_mean(residual, weights)


@register_metric(
    "null_space_residual_fraction",
    requires_reference=False,
    needs=("coil_maps",),
    direction="lower",
)
class NullSpaceResidualFraction:
    """Share of the prediction's supported energy that lies off the coil manifold."""

    name = "null_space_residual_fraction"

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> Tensor:
        del target
        residual, energy, weights = _residual_and_weights(self.name, prediction, context, kwargs)
        return weighted_mean(residual, weights) / weighted_mean(energy, weights)
