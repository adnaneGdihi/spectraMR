r"""SENSE + L1-wavelet compressed sensing as a parameter-free reconstruction model.

The model solves, per slice,

.. math::

    \hat x=\arg\min_x\tfrac12\lVert MFSx-y\rVert_2^2+\lambda\lVert D\Psi x\rVert_1,

with :math:`\Psi` an orthonormal periodized wavelet and :math:`D` selecting its
detail coefficients, by FISTA (:class:`~spectramr.infrastructure.physics.mbir.FISTAMBIRSolver`,
the one FISTA owner) at step :math:`1/L`, :math:`L=\max_r\sum_c|S_c(r)|^2\ge\lVert A\rVert^2`.

It has no weights. It sits where a network sits so that a reconstruction strategy
grades it on the cohort's own validation path: the strategy hands it
:math:`b=A^Hy` (the SENSE-combined zero-filled image an image network receives),
the mask and the coil maps, and the data-term gradient :math:`A^HAz-b` needs
nothing else. Reading :math:`b` rather than :math:`y` also keeps the repetition
forward honest: on ``input_alt`` the strategy rebuilds :math:`b` from the second
repetition, and the solve follows it.

The solve runs in float32/complex64 with autocast off: a fixed-point iteration
gains nothing from half precision and FFTs at fp16 lose the small coefficients the
threshold decides on.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import nn

from spectramr.infrastructure.physics.mbir import FISTAMBIRSolver
from spectramr.infrastructure.physics.multicoil_ei_operator import (
    MulticoilEIOperator,
    interleaved_to_complex,
)
from spectramr.infrastructure.physics.wavelet_ops import WAVELET_FILTERS, OrthogonalWavelet2D
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

__all__ = ["CSL1WaveletSENSE"]


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"cs_l1_wavelet_sense {name} must be a positive integer, got {value!r}.")
    return value


@register_model(
    name="cs_l1_wavelet_sense",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    expects_real_imag_interleaved=True,
    requires_paired_data=False,
)
class CSL1WaveletSENSE(nn.Module, IGenerator):
    """FISTA on SENSE + L1-wavelet; consumes ``A^H y``, ``mask`` and ``sensitivity_maps``.

    Args:
        in_channels: must be 2 (one complex image, real/imag interleaved).
        out_channels: must be 2.
        lambda_reg: the fixed regularization weight, in the units of ``A^H y``.
        num_iterations: FISTA iterations.
        wavelet: one of ``WAVELET_FILTERS``.
        levels: decomposition levels.

    The four solver knobs have no defaults: a misspelled ``model_kwargs`` key is
    dropped by the factory, and a default would then stand in for it silently.

    Raises:
        ValueError: on any knob outside its declared range or set.
    """

    #: No weights: graded by ``--validate-only`` without a checkpoint
    #: (:mod:`spectramr.core.parameter_free`).
    parameter_free: ClassVar[bool] = True

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        *,
        lambda_reg: float,
        num_iterations: int,
        wavelet: str,
        levels: int,
    ) -> None:
        super().__init__()
        if in_channels != 2 or out_channels != 2:
            raise ValueError(
                "cs_l1_wavelet_sense reconstructs one complex image, real/imag "
                f"interleaved: in_channels and out_channels must be 2, got "
                f"{in_channels} and {out_channels}."
            )
        if isinstance(lambda_reg, bool) or not isinstance(lambda_reg, (int, float)):
            raise ValueError(
                f"cs_l1_wavelet_sense lambda_reg must be a number, got {lambda_reg!r}."
            )
        if not lambda_reg >= 0.0:
            raise ValueError(f"cs_l1_wavelet_sense lambda_reg must be >= 0, got {lambda_reg}.")
        if wavelet not in WAVELET_FILTERS:
            raise ValueError(
                f"cs_l1_wavelet_sense wavelet {wavelet!r} is unknown; choose one of "
                f"{sorted(WAVELET_FILTERS)}."
            )
        self.lambda_reg = float(lambda_reg)
        self.num_iterations = _positive_int("num_iterations", num_iterations)
        self.transform = OrthogonalWavelet2D(wavelet, _positive_int("levels", levels))

    @property
    def solver_record(self) -> dict[str, Any]:
        """What the run solved, stamped into provenance beside the parameter count."""
        return {
            "solver": "fista",
            "objective": "0.5*||M F S x - y||^2 + lambda*||D Psi x||_1",
            "regularization": "wavelet",
            "wavelet": self.transform.wavelet,
            "levels": self.transform.levels,
            "lambda_reg": self.lambda_reg,
            "lambda_policy": "fixed",
            "num_iterations": self.num_iterations,
            "step_size": "1/L, L = max_r sum_c |S_c(r)|^2",
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
                "cs_l1_wavelet_sense needs the sampling mask its A^H y was built with; "
                "none reached forward(). Run it under a strategy that passes `mask`."
            )
        with torch.autocast(device_type=x.device.type, enabled=False):
            b = interleaved_to_complex(x.float()).to(torch.complex64)
            maps = sensitivity_maps
            if maps is not None and torch.is_complex(maps):
                maps = maps.to(torch.complex64)
            # The operator the EI strategy built A^H y with: the same maps and mask.
            operator = MulticoilEIOperator(maps, mask, where="at cs_l1_wavelet_sense.forward()")
            solver = FISTAMBIRSolver(
                operator,
                num_iterations=self.num_iterations,
                lambda_reg=self.lambda_reg,
                step_size=None,
                regularization="wavelet",
                wavelet=self.transform,
                lipschitz=operator.lipschitz_bound(),
            )
            image = solver(adjoint_measurement=b)
        return torch.cat([image.real, image.imag], dim=1).to(x.dtype)
