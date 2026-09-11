r"""LCAH -- Lipschitz-Certified Acquisition Hypernetwork encoder.

A target encoder conditioned on the continuous acquisition vector
:math:`\boldsymbol\varphi=(\mathrm{TE},\mathrm{TR},\mathrm{TI},\alpha,B_0)`
through a hypernetwork, with both networks spectral-normalised so the
extrapolation certificate

.. math::

   \sup_{\mathbf x}\bigl\|f_{h_\psi(\boldsymbol\varphi)}(\mathbf x)
       - f_{h_\psi(\boldsymbol\varphi^\ast)}(\mathbf x)\bigr\|
   \;\le\; L_w L_h\,\lVert\boldsymbol\varphi - \boldsymbol\varphi^\ast\rVert

holds by construction (Section LCAH of the bundle design). Unlike HyperMorph or
the neural-CDE acquisition-independent estimator, the discriminating point is the
certificate, not the conditioning: the certified radius is reported at inference
at no training cost.

The hypernetwork emits FiLM modulation :math:`(\gamma,\beta)` for the target's
feature map -- a real, Lipschitz-analysable weight-generation seam (the existing
``field_film_modulation`` pattern). The acquisition vector matches the SSOT
``ConditioningContext.acquisition`` layout. That is continuous acquisition
physics, not the framework's discrete ``contrast_idx``, so this encoder does
not declare ``supports_contrast_conditioning`` (#1938).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm as _spectral_norm

from spectramr.models.blocks.spectral_lipschitz import (
    extrapolation_radius,
    spectral_norm_product,
)
from spectramr.models.registry import register_model


def _maybe_sn(layer: nn.Module, enabled: bool) -> nn.Module:
    return _spectral_norm(layer) if enabled else layer


@register_model(
    name="lcah_encoder",
    training_mode="acq_hypernetwork",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
)
class LCAHEncoder(nn.Module):
    """Spectral-normalised acquisition hypernetwork encoder with a certificate.

    Args:
        in_channels: Input image channels.
        acq_dim: Acquisition-vector dimension (5 = TE, TR, TI, FA, B0).
        hidden_channels: Target feature width (also the FiLM channel count).
        out_channels: Output feature channels.
        hyper_hidden: Hypernetwork hidden width.
        spectral_norm: Spectral-normalise both networks (required for the
            certificate to be valid).
    """

    def __init__(
        self,
        in_channels: int = 1,
        acq_dim: int = 5,
        hidden_channels: int = 16,
        out_channels: int = 1,
        hyper_hidden: int = 32,
        *,
        spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.spectral_norm = bool(spectral_norm)

        self.hypernet = nn.Sequential(
            _maybe_sn(nn.Linear(acq_dim, hyper_hidden), spectral_norm),
            nn.ReLU(),
            _maybe_sn(nn.Linear(hyper_hidden, 2 * hidden_channels), spectral_norm),
        )
        self.target_pre = _maybe_sn(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1), spectral_norm
        )
        self.target_post = nn.Sequential(
            nn.ReLU(),
            _maybe_sn(nn.Conv2d(hidden_channels, out_channels, 3, padding=1), spectral_norm),
        )

    def forward(self, x: torch.Tensor, acquisition: torch.Tensor) -> torch.Tensor:
        """Run the FiLM-modulated target conditioned on the acquisition vector.

        Args:
            x: Input image ``[B, in_channels, H, W]``.
            acquisition: Acquisition vectors ``[B, acq_dim]``.

        Returns:
            Feature map ``[B, out_channels, H, W]``.
        """
        gamma, beta = self.hypernet(acquisition).chunk(2, dim=-1)
        h = self.target_pre(x)
        h = gamma[:, :, None, None] * h + beta[:, :, None, None]
        return self.target_post(h)

    def lipschitz_hyper(self) -> float:
        r"""Hypernetwork Lipschitz bound :math:`L_h=\prod\sigma_{\max}(W^h_\ell)`."""
        return spectral_norm_product(self.hypernet)

    def lipschitz_target(self) -> float:
        r"""Target Lipschitz bound :math:`L_w=\prod\sigma_{\max}(W^f_\ell)`."""
        return spectral_norm_product(self.target_pre) * spectral_norm_product(self.target_post)

    def certified_radius(self, phi: torch.Tensor, phi_train: torch.Tensor) -> torch.Tensor:
        r"""Certified drift radius :math:`L_w L_h\min_i\lVert\varphi-\varphi_i\rVert`.

        Args:
            phi: Query acquisition vector ``[acq_dim]``.
            phi_train: Training acquisition vectors ``[N, acq_dim]``.

        Returns:
            The certified worst-case prediction drift (zero at a training point).
        """
        lipschitz = self.lipschitz_hyper() * self.lipschitz_target()
        return extrapolation_radius(lipschitz, phi, phi_train)


__all__ = ["LCAHEncoder"]
