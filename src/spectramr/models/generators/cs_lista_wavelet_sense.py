r"""Unrolled ISTA on SENSE with a learned wavelet soft-threshold: CS whose prior is still L1-wavelet.

``cs_l1_wavelet_sense`` solves :math:`\min_x\tfrac12\lVert MFSx-y\rVert^2+\lambda\lVert D\Psi x\rVert_1`
with FISTA and has no weights. This model keeps the same operator, orthonormal :math:`\Psi`
and step :math:`1/L`, truncates the iteration at :math:`K` steps and learns what the fixed
solver hard-codes (ISTA-Net / LISTA):

.. math::

    z_k=x_k-\tfrac{\rho_k}{L}A^H(Ax_k-b'),\qquad
    x_{k+1}=\Psi^H\,\mathcal S_{\theta_{k,\ell,o}\rho_k/L}(\Psi z_k),\qquad x_0=b,

with :math:`\rho_k` a per-iteration step scale and :math:`\theta_{k,\ell,o}>0` one threshold
per iteration, wavelet level :math:`\ell` and detail orientation :math:`o`. The approximation
band is never thresholded. At initialisation (:math:`\rho_k=1`, :math:`\theta=\lambda_0`) it is
ISTA with the wavelet arm's threshold, so training starts from the classical solver.

The weights are fitted by the strategy's own loss (equivariant imaging through
:math:`A=MFS`), which needs no reference. It reads :math:`b=A^Hy`, the mask and the coil maps
exactly as ``cs_l1_wavelet_sense`` does. The solve runs in float32 with autocast off.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from spectramr.infrastructure.physics.multicoil_ei_operator import (
    MulticoilEIOperator,
    interleaved_to_complex,
)
from spectramr.infrastructure.physics.wavelet_ops import (
    WAVELET_FILTERS,
    OrthogonalWavelet2D,
    complex_soft_threshold,
)
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

__all__ = ["CSLISTAWaveletSENSE"]

_ORIENTATIONS = 3  # horizontal, vertical, diagonal detail quadrants


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"cs_lista_wavelet_sense {name} must be a positive integer, got {value!r}."
        )
    return value


def _band_index(h: int, w: int, levels: int) -> torch.Tensor:
    """``[H, W]`` map to ``level * 3 + orientation`` on detail coefficients, ``-1`` on the approximation."""
    index = torch.full((h, w), -1, dtype=torch.long)
    for level in range(levels):
        hh, ww = h >> level, w >> level
        hh2, ww2 = hh // 2, ww // 2
        base = level * _ORIENTATIONS
        index[:hh2, ww2:ww] = base
        index[hh2:hh, :ww2] = base + 1
        index[hh2:hh, ww2:ww] = base + 2
    return index


@register_model(
    name="cs_lista_wavelet_sense",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    expects_real_imag_interleaved=True,
    requires_paired_data=False,
)
class CSLISTAWaveletSENSE(nn.Module, IGenerator):
    """K unrolled ISTA steps on SENSE with learned step scales and per-band thresholds.

    Args:
        in_channels: must be 2 (one complex image, real/imag interleaved).
        out_channels: must be 2.
        num_iterations: unrolled steps ``K``; every step has its own parameters.
        lambda_reg: initial threshold, in the units of ``A^H y`` (the wavelet arm's lambda).
        wavelet: one of ``WAVELET_FILTERS``.
        levels: decomposition levels.

    The four knobs have no defaults: a misspelled ``model_kwargs`` key is dropped by the
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
        lambda_reg: float,
        wavelet: str,
        levels: int,
    ) -> None:
        super().__init__()
        if in_channels != 2 or out_channels != 2:
            raise ValueError(
                "cs_lista_wavelet_sense reconstructs one complex image, real/imag "
                f"interleaved: in_channels and out_channels must be 2, got {in_channels} and {out_channels}."
            )
        if isinstance(lambda_reg, bool) or not isinstance(lambda_reg, (int, float)):
            raise ValueError(
                f"cs_lista_wavelet_sense lambda_reg must be a number, got {lambda_reg!r}."
            )
        if not lambda_reg > 0.0:
            raise ValueError(f"cs_lista_wavelet_sense lambda_reg must be > 0, got {lambda_reg}.")
        if wavelet not in WAVELET_FILTERS:
            raise ValueError(
                f"cs_lista_wavelet_sense wavelet {wavelet!r} is unknown; choose one of "
                f"{sorted(WAVELET_FILTERS)}."
            )
        self.num_iterations = _positive_int("num_iterations", num_iterations)
        self.lambda_reg = float(lambda_reg)
        self.transform = OrthogonalWavelet2D(wavelet, _positive_int("levels", levels))
        # theta = softplus(raw) > 0; raw is set so theta starts at lambda_reg.
        raw0 = math.log(math.expm1(self.lambda_reg))
        self.raw_threshold = nn.Parameter(
            torch.full((self.num_iterations, self.transform.levels * _ORIENTATIONS), raw0)
        )
        self.log_step_scale = nn.Parameter(torch.zeros(self.num_iterations))
        self._band_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    @property
    def solver_record(self) -> dict[str, Any]:
        """What the run solved, stamped into provenance beside the parameter count."""
        return {
            "solver": "ista_unrolled_learned",
            "objective": "0.5*||M F S x - y||^2 + learned per-band wavelet L1",
            "regularization": "wavelet",
            "wavelet": self.transform.wavelet,
            "levels": self.transform.levels,
            "lambda_init": self.lambda_reg,
            "lambda_policy": "learned per iteration, level and orientation",
            "num_iterations": self.num_iterations,
            "step_size": "rho_k / L, rho_k learned, L = max_r sum_c |S_c(r)|^2",
        }

    def _bands(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w, str(device))
        if key not in self._band_cache:
            self._band_cache[key] = _band_index(h, w, self.transform.levels).to(device)
        return self._band_cache[key]

    def _threshold_map(self, k: int, step: torch.Tensor, bands: torch.Tensor) -> torch.Tensor:
        """``[B, 1, H, W]`` threshold of iteration ``k``: zero on the approximation band."""
        theta = F.softplus(self.raw_threshold[k])
        theta = torch.cat([theta, theta.new_zeros(1)])  # index -1 -> 0
        return theta[bands] * step.view(-1, 1, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        sensitivity_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct from ``x = A^H y`` (``[B, 2, H, W]``); returns ``[B, 2, H, W]``."""
        if mask is None:
            raise ValueError(
                "cs_lista_wavelet_sense needs the sampling mask its A^H y was built with; "
                "none reached forward(). Run it under a strategy that passes `mask`."
            )
        with torch.autocast(device_type=x.device.type, enabled=False):
            b = interleaved_to_complex(x.float()).to(torch.complex64)
            maps = sensitivity_maps
            if maps is not None and torch.is_complex(maps):
                maps = maps.to(torch.complex64)
            operator = MulticoilEIOperator(maps, mask, where="at cs_lista_wavelet_sense.forward()")
            inv_l = 1.0 / max(operator.lipschitz_bound(), 1e-12)
            bands = self._bands(b.shape[-2], b.shape[-1], b.device)
            est = b
            for k in range(self.num_iterations):
                step = torch.exp(self.log_step_scale[k]) * inv_l
                z = est - step * (operator.adjoint(operator.forward(est)) - b)
                coeffs = complex_soft_threshold(
                    self.transform.forward(z),
                    self._threshold_map(k, step.expand(b.shape[0]), bands),
                )
                est = self.transform.inverse(coeffs)
        return torch.cat([est.real, est.imag], dim=1).to(x.dtype)
