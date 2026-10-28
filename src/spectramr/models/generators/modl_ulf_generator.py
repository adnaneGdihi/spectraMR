"""MoDL for ULF→HF on magnitude images, with SIMULATED low-pass data consistency.

Aggarwal et al. (2019, IEEE TMI) unroll ``x_{k+1} = argmin_x lambda ||A x - y||^2 +
||x - D_w(x_k)||^2`` with a shared CNN denoiser ``D_w`` and a learnable ``lambda``.
The paired 64 mT / 3 T data are magnitude NIfTI with no acquired k-space, so ``A``
here is the ULF image-domain operator the ULF MAP strategy uses: a Gaussian low-pass
in the fft2c domain (``infrastructure/physics/ulf_lowpass.py``), and ``y`` is the
64 mT image. The DC step is then the closed-form Fourier prox, not a CG solve. It is
data consistency against a simulated operator, not against measured k-space.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.physics.ulf_lowpass import _lowpass_otf, lowpass_dc_prox
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

#: ``model_kwargs.data_consistency`` vocabulary.
DATA_CONSISTENCY_MODES = ("lowpass", "none")


class _ResidualDenoiser(nn.Module):
    """``D_w(x) = x + CNN(x)``: the MoDL regulariser, shared across iterations."""

    def __init__(self, channels: int, features: int, num_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(channels, features, 3, padding=1), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(features, features, 3, padding=1),
                nn.BatchNorm2d(features),
                nn.ReLU(),
            ]
        layers.append(nn.Conv2d(features, channels, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


@register_model(
    name="modl_ulf",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class MoDLULFGenerator(nn.Module, IGenerator):
    """Unrolled MoDL: shared residual CNN denoiser alternating with low-pass DC.

    Args:
        in_channels: image channels; must equal ``out_channels`` (DC stays in image space).
        out_channels: see ``in_channels``.
        features: CNN width.
        num_layers: CNN depth (>= 3).
        iterations: unrolled iterations K.
        data_consistency: ``lowpass`` (the MoDL step) or ``none`` (the same CNN
            applied K times, the control). Anything else raises.
        blur_cutoff: cutoff of the low-pass ``A``; required with ``lowpass`` and
            refused with ``none``.
        lambda_init: initial DC weight ``lambda`` (learned, kept positive via ``exp``).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: int = 32,
        num_layers: int = 5,
        iterations: int = 5,
        data_consistency: str = "lowpass",
        blur_cutoff: float | None = None,
        lambda_init: float = 1.0,
    ) -> None:
        super().__init__()
        if data_consistency not in DATA_CONSISTENCY_MODES:
            raise ValueError(
                f"modl_ulf data_consistency={data_consistency!r}; expected one of "
                f"{DATA_CONSISTENCY_MODES}."
            )
        if data_consistency == "lowpass" and blur_cutoff is None:
            raise ValueError("modl_ulf data_consistency='lowpass' needs model_kwargs.blur_cutoff.")
        if data_consistency == "none" and blur_cutoff is not None:
            raise ValueError("modl_ulf blur_cutoff is read only with data_consistency='lowpass'.")
        if in_channels != out_channels:
            raise ValueError(
                f"modl_ulf needs in_channels == out_channels (got {in_channels}, {out_channels}): "
                "the DC step compares the estimate with the input in one image space."
            )
        if num_layers < 3 or iterations < 1 or lambda_init <= 0:
            raise ValueError("modl_ulf needs num_layers >= 3, iterations >= 1, lambda_init > 0.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = int(iterations)
        self.data_consistency = data_consistency
        self.blur_cutoff = None if blur_cutoff is None else float(blur_cutoff)
        self.denoiser = _ResidualDenoiser(in_channels, features, num_layers)
        self.log_lambda = (
            nn.Parameter(torch.tensor(math.log(lambda_init)))
            if data_consistency == "lowpass"
            else None
        )

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """``x`` is the 64 mT image ``y``, ``[B, C, H, W]`` or TorchIO's ``[B, C, H, W, 1]``."""
        depth = x.shape[-1] if x.dim() == 5 else None
        if depth is not None:
            b, c, h, w, _ = x.shape
            x = x.permute(0, 4, 1, 2, 3).reshape(b * depth, c, h, w)
        y = x
        y_k = h_otf = lam = None
        if self.log_lambda is not None:
            y_k = fft2c(y)
            h_otf = _lowpass_otf(
                y.shape[-2], y.shape[-1], self.blur_cutoff, y.device, torch.float32
            )
            lam = self.log_lambda.exp()
        z = y
        for _ in range(self.iterations):
            z = self.denoiser(z)
            if lam is not None:
                z = lowpass_dc_prox(y_k, z, h_otf, lam, 1.0)
        if depth is not None:
            z = z.reshape(-1, depth, *z.shape[1:]).permute(0, 2, 3, 4, 1)
        return z


__all__ = ["DATA_CONSISTENCY_MODES", "MoDLULFGenerator"]
