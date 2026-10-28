"""Wiener data consistency with the signal power of the fitted decay (Algorithm G).

``dc_method: noise_adaptive`` blends each measured bin toward ``y`` with weight
``P / (P + beta sigma^2)`` and takes ``P = |y|^2``. On a noise-dominated outer
line ``|y|^2`` is mostly noise power, so the layer trusts the noise about as
much as the signal. ``reverse_dc_trust: fitted_decay`` takes ``P`` from the
separable envelope the measured rows fit instead,

.. math::

    P(k_y, k_x) = S(|k_y|)^2\\, q(k_x),

with ``S`` the power law of ``reverse_decay`` and ``q`` the column profile of the
fitted rows, normalised to mean one. The row ``ky = 0`` keeps ``|y|^2``: the law
is not defined there and that row's signal dominates its noise.

Before the blend, the layer moves the prediction on every measured bin into the
disc ``|k_pred - y| <= 3 sigma``; the blend is convex, so the result stays in
the disc, and the ``observed_discrepancy`` invariant checks it. Both run in
LINEAR k-space -- a Wiener weight on ``log1p``-compressed values is not a
Wiener weight -- and ``sigma^2`` comes from the one estimator the layer owns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from spectramr.infrastructure.physics.data_consistency import NoiseAdaptiveDataConsistency
from spectramr.models.diffusion.reverse_decay import measured_power, row_index

if TYPE_CHECKING:
    from torch import Tensor

    from spectramr.models.diffusion.kspace_process import PhysicsInformedColdDiffusion
    from spectramr.models.diffusion.reverse_decay import DecayEnvelope

#: ``c`` of the ``|k_pred - y| <= c sigma`` disc on measured bins.
TRUST_RADIUS = 3.0


@dataclass(frozen=True)
class FittedTrust:
    """Linear-domain signal power ``[B, 1, H, W]``, noise std ``[B]`` and disc radius."""

    signal_power: Tensor
    sigma: Tensor

    @property
    def radius(self) -> Tensor:
        """``TRUST_RADIUS * sigma`` as ``[B, 1, 1, 1]``."""
        return (TRUST_RADIUS * self.sigma).view(-1, 1, 1, 1)


def build_fitted_trust(
    decay: DecayEnvelope, measurement: Tensor, obs: Tensor, *, log_scaled: bool, eps: float = 1e-8
) -> FittedTrust:
    """``P = S(|k_line|)^2 q(k_across)`` from ``decay`` and the fitted lines of ``measurement``.

    The law is written for rows; a column envelope (a mask that removes columns,
    #2160) is the same construction on the transposed plane.
    """
    if decay.axis.bins is not None:
        raise ValueError(
            f"build_fitted_trust needs a line envelope; this one fits the {decay.axis.name!r} axis."
        )
    if decay.axis.line_dim == -1:
        trust = _row_trust(
            decay, measurement.transpose(-1, -2), obs.transpose(-1, -2), log_scaled, eps
        )
        return FittedTrust(signal_power=trust.signal_power.transpose(-1, -2), sigma=trust.sigma)
    return _row_trust(decay, measurement, obs, log_scaled, eps)


def _row_trust(
    decay: DecayEnvelope, measurement: Tensor, obs: Tensor, log_scaled: bool, eps: float
) -> FittedTrust:
    """``P = S(|ky|)^2 q(kx)`` over rows: ``decay``'s bins index ``measurement``'s dim -2."""
    _, power, support = measured_power(measurement, obs, log_scaled)
    sigma2 = decay.sigma2.view(-1, 1, 1, 1)
    rows = decay.fitted_bins[:, None, :, None] & support
    row_signal = ((power - sigma2) * rows).sum(dim=-1, keepdim=True) / rows.sum(
        dim=-1, keepdim=True
    ).clamp_min(1)
    ratio = (power - sigma2).clamp_min(0.0) / row_signal.clamp_min(eps)
    q = (ratio * rows).sum(dim=-2, keepdim=True) / rows.sum(dim=-2, keepdim=True).clamp_min(1)
    q = q / q.mean(dim=-1, keepdim=True).clamp_min(eps)

    height = measurement.shape[-2]
    ky = row_index(height, measurement.device)
    law = torch.exp(
        2.0 * (decay.log_amp[:, None] + decay.slope[:, None] * torch.log(ky.clamp_min(1.0)))
    )
    signal = law[:, None, :, None] * q
    signal = torch.where((ky < 1).view(1, 1, -1, 1), power, signal)
    return FittedTrust(signal_power=signal, sigma=decay.sigma2.sqrt())


def apply_fitted_trust(
    sampler: PhysicsInformedColdDiffusion,
    trust: FittedTrust,
    x0: Tensor,
    measurement: Tensor,
    obs: Tensor,
) -> Tensor:
    """The model's ``noise_adaptive`` layer with ``trust``'s power and disc, on ``obs`` only.

    ``x0`` and ``measurement`` stay in the sampler's domain; the layer's ``log_scaled``
    moves them to linear k-space for the blend, so it must match ``kspace_log_scaled``.
    """
    layer = getattr(sampler.model, "dc_layer", None)
    if not isinstance(layer, NoiseAdaptiveDataConsistency):
        raise ValueError(
            "reverse_dc_trust='fitted_decay' drives the model's NoiseAdaptiveDataConsistency "
            f"layer, but the model carries {type(layer).__name__}."
        )
    # The layer owns the log1p round trip; converting here as well would decompress twice.
    if layer.log_scaled != sampler.kspace_log_scaled:
        raise ValueError(
            f"The dc_layer declares log_scaled={layer.log_scaled}, but the sampler's k-space "
            f"has kspace_log_scaled={sampler.kspace_log_scaled}."
        )
    support = obs.amax(dim=-3, keepdim=True)
    out = layer(
        x0,
        measurement,
        support,
        is_kspace_domain=True,
        noise_sigma=trust.sigma,
        signal_power=trust.signal_power,
        trust_radius=TRUST_RADIUS,
    )
    return x0 * (1.0 - obs) + out * obs


__all__ = ["TRUST_RADIUS", "FittedTrust", "apply_fitted_trust", "build_fitted_trust"]
