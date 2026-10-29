"""The calibrated hardware warp as a registered model: the operator the calibration arm produces.

The ``hardware_warp_calibration`` strategy learns nothing by gradient; its deliverable is the
harmonic-gradient displacement field ``v`` of one scanner unit at one software version. Until this
module existed the arm's ``model:`` block named a U-Net that was constructed and never called, so
the Tier-2 forward probe reported a green that certified nothing. This operator is what the block
names now: it holds the fitted coefficients, the strategy loads the fit into it once the
least-squares solve has run, and it evaluates ``v`` on scanner-frame point clouds. It does not
act on image batches -- an image carries no scanner frame without its header -- so it declares
the probe's ``forward`` opt-out and the probe reports *not applicable* instead of green; image
resampling stays with the ``hardware_unwarp`` transform, the one owner of that step.

The coefficients are a frozen ``nn.Parameter`` rather than a buffer because the harness builds
an optimizer over ``parameters()`` and torch refuses an empty list; a parameter that never
requires a gradient is skipped by every optimizer step and never decayed.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from spectramr.data.hardware_warp_artifacts import load_artifact, validate_artifact
from spectramr.infrastructure.physics.hardware_warp_commutator import (
    HarmonicGradientBasis,
    ball_points,
)
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

__all__ = ["HardwareWarpOperator"]


@register_model(
    name="hardware_warp_operator",
    training_mode="hardware_warp_calibration",
    # No spatial_dims: the operator acts on point clouds, so an image dimensionality would be a
    # claim the audit checks against data.sampling.patch_size and finds false either way.
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=False,
    supports_contrast_conditioning=False,
)
class HardwareWarpOperator(nn.Module, IGenerator):
    """Displacement field ``v(r)`` of a calibrated hardware warp on RAS points in mm.

    Construct it empty for a calibration run (the strategy calls :meth:`load_artifact_record`
    after the fit) or from ``artifact_path`` for a run that applies an existing artifact. An
    uncalibrated operator refuses to evaluate rather than returning zeros: an identity warp is a
    measurement (``n_pairs == 0`` in ``identity_artifact``), not the absence of one.
    """

    #: The Tier-2 probe forwards an image batch; this operator has no image contract, so the
    #: probe reports ``no_probe`` for it instead of a green that would certify nothing.
    synthetic_forward_probe_skip = frozenset({"forward"})

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        artifact_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._in_channels, self._out_channels = int(in_channels), int(out_channels)
        self.coeffs = nn.Parameter(torch.zeros(0, dtype=torch.float64), requires_grad=False)
        self.register_buffer("center_mm", torch.zeros(3, dtype=torch.float64))
        self._basis: HarmonicGradientBasis | None = None
        self.artifact: dict[str, Any] | None = None
        if artifact_path is not None:
            self.load_artifact_record(load_artifact(artifact_path))

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def name(self) -> str:
        return "hardware_warp_operator"

    @property
    def calibrated(self) -> bool:
        """``True`` once an artifact record has been loaded."""
        return self._basis is not None

    def load_artifact_record(self, artifact: dict[str, Any]) -> None:
        """Take the coefficients, basis and centre of a validated artifact record.

        The canonical basis is rebuilt from the artifact's ``basis_points_per_axis`` and
        ``ball_radius_mm`` exactly as every other consumer does, so ``coeffs`` pair with the
        fields they were fitted on. Assigning ``coeffs.data`` keeps the parameter object the
        harness's optimizer already holds.
        """
        artifact = validate_artifact(artifact)
        self._basis = HarmonicGradientBasis(
            ball_points(int(artifact["basis_points_per_axis"]), float(artifact["ball_radius_mm"])),
            int(artifact["max_degree"]),
            int(artifact["min_degree"]),
        )
        self.coeffs.data = torch.tensor(artifact["coeffs"], dtype=torch.float64)
        self.center_mm.copy_(torch.tensor(artifact["center_mm"], dtype=torch.float64))
        self.artifact = dict(artifact)

    def displacement(self, points: torch.Tensor) -> torch.Tensor:
        """``v`` at ``points [..., 3]`` given in RAS mm (absolute, the centre is subtracted here).

        The basis is a float64 CPU object, so the evaluation runs there and the result returns
        on the caller's device and dtype; the operator is a calibration product, not a hot path.
        """
        if self._basis is None:
            raise RuntimeError(
                "HardwareWarpOperator is not calibrated: run the hardware_warp_calibration arm "
                "or construct it with artifact_path"
            )
        if points.ndim < 1 or points.shape[-1] != 3:
            raise TypeError(
                f"HardwareWarpOperator acts on point clouds [..., 3] in RAS mm, got shape "
                f"{tuple(points.shape)}; an image batch carries no scanner frame, apply the "
                "artifact to images through the hardware_unwarp transform"
            )
        flat = points.detach().reshape(-1, 3).to("cpu", torch.float64) - self.center_mm.cpu()
        out = self._basis.synthesize(self.coeffs.detach().cpu(), flat)
        return out.reshape(points.shape).to(points.device, points.dtype)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.displacement(x)

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(input_shape)
