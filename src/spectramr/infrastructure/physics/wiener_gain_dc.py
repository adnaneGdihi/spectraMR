r"""Wiener data consistency with a closed-form gain fit, on linear k-space.

A single-NEX input carries the full receiver noise while an NEX-averaged target
does not, so hard data consistency (``(1 - m) k_pred + m y``) writes that noise
back into every acquired bin. This layer blends instead, per bin, by how much
of the measured power is signal, after first matching the prediction's scale to
the measurement:

.. math::

    \alpha &= \operatorname{clip}\Bigl(\frac{\operatorname{Re}\langle M_c\hat k,\,M_c y\rangle}
              {\langle M_c\hat k,\,M_c\hat k\rangle + \epsilon},\ \alpha_{\min},\ \alpha_{\max}\Bigr), \\
    w &= \frac{|\alpha\hat k|^2}{|\alpha\hat k|^2 + \operatorname{softplus}(\beta)\,\sigma^2}, \\
    k_{\text{out}} &= (1 - M)\,\alpha\hat k + M\,\bigl(w\,y + (1 - w)\,\alpha\hat k\bigr).

:math:`M_c` is the measured support inside the central disc, where the signal
dominates the noise, so the least-squares gain is not biased by noise power.
:math:`\alpha` is fitted in closed form and detached: the optimiser never
trades image content for scale. :math:`\sigma^2` is the per-sample noise power
of the measurement, estimated as :class:`NoiseAdaptiveDataConsistency` does.
As :math:`\sigma^2 \to 0`, :math:`w \to 1` on every measured bin with non-zero
prediction and the layer reduces to hard DC.

All three steps act on LINEAR complex k-space. The k-space the cold-diffusion
model holds is ``log1p``-compressed when ``data.processing.enable_log_scaling``
is on, and a Wiener blend or a least-squares gain computed on compressed values
is a different operator, so the compression flag is a required argument.
"""

from __future__ import annotations

import weakref

import torch
from torch import Tensor

from spectramr.infrastructure.physics.data_consistency import NoiseAdaptiveDataConsistency
from spectramr.infrastructure.physics.dc_mask import align_dc_mask

__all__ = ["WienerGainDataConsistency"]


class WienerGainDataConsistency(NoiseAdaptiveDataConsistency):
    """Noise-aware soft DC with a gain fit between prediction and measurement.

    Args:
        log_scaled: Whether the k-space this layer receives is ``log1p``-compressed
            (``data.processing.enable_log_scaling``). Required: guessing the domain
            would silently compute the blend on the wrong values.
        beta: Initial value of the softplus-wrapped trust temperature; ``dc_weight``.
        learnable_beta: Whether ``beta`` is optimised.
        gain_radius: Radius of the central disc used for the gain fit, as a
            fraction of the largest k-space radius.
        gain_bounds: ``(min, max)`` clamp on the fitted gain.
        noise_support_frac: Forwarded to :class:`NoiseAdaptiveDataConsistency`.
        eps: Numerical floor.
    """

    def __init__(
        self,
        *,
        log_scaled: bool,
        beta: float = 1.0,
        learnable_beta: bool = True,
        gain_radius: float = 0.08,
        gain_bounds: tuple[float, float] = (0.5, 2.0),
        noise_support_frac: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        if not isinstance(log_scaled, bool):
            raise TypeError(
                "WienerGainDataConsistency needs log_scaled as a bool (the arm's "
                f"data.processing.enable_log_scaling), got {log_scaled!r}."
            )
        if not 0.0 < float(gain_radius) <= 1.0:
            raise ValueError(f"gain_radius must lie in (0, 1], got {gain_radius!r}")
        lo, hi = (float(gain_bounds[0]), float(gain_bounds[1]))
        if not 0.0 < lo <= 1.0 <= hi:
            raise ValueError(f"gain_bounds must satisfy 0 < min <= 1 <= max, got {gain_bounds!r}")
        super().__init__(
            beta=beta,
            learnable_beta=learnable_beta,
            noise_support_frac=noise_support_frac,
            eps=eps,
        )
        self.log_scaled = log_scaled
        self.gain_radius = float(gain_radius)
        self.gain_bounds = (lo, hi)
        # Telemetry: mean fitted gain of the last forward, a 0-d device tensor.
        self.register_buffer("_last_gain", torch.ones(()), persistent=False)
        # sigma^2 of the last measurement, keyed on the identity and version of the
        # caller's measurement and mask: the reverse sampler passes the same two
        # tensors at every step, so the estimate runs once per trajectory.
        self._sigma2_key: tuple | None = None
        self._sigma2: Tensor | None = None

    def _trajectory_sigma2(
        self, measured_kspace: Tensor, mask: Tensor, y: Tensor, support: Tensor
    ) -> Tensor:
        """Noise variance ``[B, 1, 1, 1]`` of ``y``, reused while its inputs are unchanged."""
        key = self._sigma2_key
        if (
            key is not None
            and key[0]() is measured_kspace
            and key[1]() is mask
            and key[2:] == (measured_kspace._version, mask._version)
        ):
            assert self._sigma2 is not None
            return self._sigma2
        sigma2 = self._noise_power(self._measured_power(y), support, None)
        self._sigma2_key = (
            weakref.ref(measured_kspace),
            weakref.ref(mask),
            measured_kspace._version,
            mask._version,
        )
        self._sigma2 = sigma2
        return sigma2

    def fit_gain(
        self, k_pred: Tensor, y: Tensor, measured: Tensor, power: Tensor | None = None
    ) -> Tensor:
        """Least-squares gain ``[B, 1, 1, 1]`` over the measured central disc.

        Args:
            k_pred: Complex prediction ``[B, C, H, W]``.
            y: Complex measurement ``[B, C, H, W]``.
            measured: Boolean measured support, broadcastable to ``[B, 1, H, W]``.
            power: ``|k_pred|^2`` when the caller already holds it.
        """
        _, _, H, W = k_pred.shape
        if self._coord_cache_shape != (H, W):
            self._build_coords(H, W, k_pred.device)
        centre = (self._coord_radius <= self.gain_radius).view(1, 1, H, W)
        region = (measured & centre).to(k_pred.real.dtype)
        with torch.no_grad():
            if power is None:
                power = k_pred.real.square() + k_pred.imag.square()
            cross = k_pred.real * y.real + k_pred.imag * y.imag  # Re(conj(k) y)
            num = (cross * region).sum(dim=(1, 2, 3), keepdim=True)
            den = (power * region).sum(dim=(1, 2, 3), keepdim=True)
            alpha = num / (den + self.eps)
            # A sample with no measured centre bin has no gain to fit: keep unit gain.
            alpha = torch.where(den > self.eps, alpha, torch.ones_like(alpha))
            return alpha.clamp(*self.gain_bounds)

    def forward(
        self,
        img_pred: Tensor,
        measured_kspace: Tensor | None = None,
        mask: Tensor | None = None,
        sensitivity_maps: Tensor | None = None,
        is_kspace_domain: bool = False,
        acs_mask: Tensor | None = None,
        noise_sigma: Tensor | float | None = None,
    ) -> Tensor:
        """Apply the gain fit and the Wiener blend; returns the layout of ``img_pred``.

        Only k-space input is supported: the cold-diffusion generator and its
        sampler always pass ``is_kspace_domain=True``, and an image-domain input
        would need an FFT whose domain this layer cannot verify.
        """
        del sensitivity_maps, acs_mask  # coil-wise, like the parent layer
        if measured_kspace is None or mask is None:
            return img_pred
        if not is_kspace_domain:
            raise ValueError(
                "WienerGainDataConsistency acts on k-space only; call it with "
                "is_kspace_domain=True."
            )
        # Function-local: importing ``models.diffusion`` at module level closes a
        # cycle through the generator, which imports this layer.
        from spectramr.models.diffusion.reverse_domain import linear_kspace, stacked_kspace

        k_pred = linear_kspace(img_pred, log_scaled=self.log_scaled)
        y = linear_kspace(measured_kspace, log_scaled=self.log_scaled).to(k_pred.dtype)
        mask_real = mask.real if torch.is_complex(mask) else mask
        measured = align_dc_mask(mask_real, k_pred.shape[-3]) > 0
        measured_any = measured.any(dim=1, keepdim=True)

        power = k_pred.real.square() + k_pred.imag.square()
        alpha = self.fit_gain(k_pred, y, measured_any, power=power.detach())
        k_scaled = k_pred * alpha
        # |alpha k|^2 averaged over coils, as the parent's measured power is.
        # Detached: the trust weight must not steer the prediction it blends.
        signal_power = alpha.square() * (
            power.detach().mean(dim=1, keepdim=True) if power.shape[1] > 1 else power.detach()
        )
        support = measured_any.to(y.real.dtype)
        sigma2 = (
            self._trajectory_sigma2(measured_kspace, mask, y, support)
            if noise_sigma is None
            else None
        )
        lam = self._compute_lambda(y, support, noise_sigma, signal_power, sigma2=sigma2)
        k_blend = (1.0 - lam) * k_scaled + lam * y
        k_out = torch.where(measured, k_blend, k_scaled)

        with torch.no_grad():
            self._last_gain = alpha.mean().detach()
        return stacked_kspace(k_out, img_pred, log_scaled=self.log_scaled)
