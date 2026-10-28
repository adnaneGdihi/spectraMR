r"""SENSE proximal gradient with a learned backbone as the prox: CS with a network in the loop.

``cs_l1_wavelet_sense`` replaces the prox by a wavelet soft-threshold and has no weights, so
a self-supervised strategy has nothing to train on it. This model keeps its data term and
step, and replaces the prox by a residual backbone shared across iterations:

.. math::

    z_k=x_k-\tfrac1L A^H(Ax_k-y),\qquad x_{k+1}=z_k+f_\theta(z_k),\qquad x_0=A^Hy,

with :math:`L=\max_r\sum_c|S_c(r)|^2` as in the wavelet arm. It reads :math:`b=A^Hy`, the
mask and the coil maps exactly as ``cs_l1_wavelet_sense`` does, so the two share a seam and
a cohort. The data step runs in float32 with autocast off; the backbone runs under the
ambient autocast.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from spectramr.infrastructure.physics.multicoil_ei_operator import (
    MulticoilEIOperator,
    interleaved_to_complex,
)
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

__all__ = ["CSPnPUNetSENSE"]


@register_model(
    name="cs_pnp_unet_sense",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    expects_real_imag_interleaved=True,
    requires_paired_data=False,
)
class CSPnPUNetSENSE(nn.Module, IGenerator):
    """Proximal gradient on SENSE whose prox is a shared residual backbone.

    Args:
        in_channels: must be 2 (one complex image, real/imag interleaved).
        out_channels: must be 2.
        num_iterations: unrolled proximal-gradient steps; the backbone is applied once per step.
        backbone: registered model name of the denoiser, e.g. ``standard_unet``.
        backbone_kwargs: kwargs for ``backbone``; channels are set here, not by the caller.

    The three knobs have no defaults: a misspelled ``model_kwargs`` key is dropped by the
    factory, and a default would then stand in for it silently.

    Raises:
        ValueError: on any knob outside its declared range or set.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        *,
        num_iterations: int,
        backbone: str,
        backbone_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        if in_channels != 2 or out_channels != 2:
            raise ValueError(
                "cs_pnp_unet_sense reconstructs one complex image, real/imag interleaved: "
                f"in_channels and out_channels must be 2, got {in_channels} and {out_channels}."
            )
        if isinstance(num_iterations, bool) or not isinstance(num_iterations, int):
            raise ValueError(
                f"cs_pnp_unet_sense num_iterations must be a positive integer, got {num_iterations!r}."
            )
        if num_iterations < 1:
            raise ValueError(
                f"cs_pnp_unet_sense num_iterations must be >= 1, got {num_iterations}."
            )
        if not isinstance(backbone, str) or not backbone:
            raise ValueError(f"cs_pnp_unet_sense backbone must be a model name, got {backbone!r}.")
        if backbone.startswith("cs_"):
            raise ValueError(
                f"cs_pnp_unet_sense backbone {backbone!r} is itself a solver; "
                "the denoiser must be a learned network."
            )
        if not isinstance(backbone_kwargs, dict):
            raise ValueError(
                f"cs_pnp_unet_sense backbone_kwargs must be a mapping, got {backbone_kwargs!r}."
            )
        self.num_iterations = num_iterations
        self.backbone_name = backbone
        # Deferred import: the factory imports the model packages.
        from spectramr.models.factories.model_factory import get_model_factory

        self.backbone = get_model_factory().create_generator(
            backbone, **{**backbone_kwargs, "in_channels": 2, "out_channels": 2}
        )

    @property
    def solver_record(self) -> dict[str, Any]:
        """What the run solved, stamped into provenance beside the parameter count."""
        return {
            "solver": "proximal_gradient_unrolled",
            "objective": "0.5*||M F S x - y||^2 + learned prox",
            "regularization": f"learned:{self.backbone_name}",
            "num_iterations": self.num_iterations,
            "step_size": "1/L, L = max_r sum_c |S_c(r)|^2",
            "prox": "residual, one backbone shared across iterations",
        }

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        sensitivity_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct from ``x = A^H y`` (``[B, 2, H, W]``); returns ``[B, 2, H, W]``."""
        if mask is None:
            raise ValueError(
                "cs_pnp_unet_sense needs the sampling mask its A^H y was built with; "
                "none reached forward(). Run it under a strategy that passes `mask`."
            )
        with torch.autocast(device_type=x.device.type, enabled=False):
            maps = sensitivity_maps
            if maps is not None and torch.is_complex(maps):
                maps = maps.to(torch.complex64)
            operator = MulticoilEIOperator(maps, mask, where="at cs_pnp_unet_sense.forward()")
            step = 1.0 / max(operator.lipschitz_bound(), 1e-12)
            b = interleaved_to_complex(x.float()).to(torch.complex64)
        est = b
        for _ in range(self.num_iterations):
            with torch.autocast(device_type=x.device.type, enabled=False):
                z = est - step * (operator.adjoint(operator.forward(est)) - b)
                z_ri = torch.cat([z.real, z.imag], dim=1)
            delta = self.backbone(z_ri)
            with torch.autocast(device_type=x.device.type, enabled=False):
                z_ri = z_ri + delta.float()
                est = torch.complex(z_ri[:, 0:1], z_ri[:, 1:2])
        return torch.cat([est.real, est.imag], dim=1).to(x.dtype)
