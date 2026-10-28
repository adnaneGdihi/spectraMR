"""Frozen-parameter orchestrator: identify the hardware warp from a repeat-session pose set.

No learned parameters. The strategy walks under the standard training harness -- the
``QualityMatchingStrategy`` precedent -- and its deliverable is the warp artifact of
:mod:`spectramr.data.hardware_warp_artifacts`, not a checkpoint. Every knob of
``training.*`` is read, validated and stamped at construction, so a misdeclared arm fails
before the dataloader has consumed any memory; the fit itself runs once, from the first
training step, and every step returns an exactly-zero differentiable loss so the harness's
backward/step cycle stays valid while nothing moves.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch

from spectramr.data.hardware_warp_artifacts import (
    load_pair_residuals,
    load_pose_set,
    write_artifact,
)
from spectramr.infrastructure.physics.hardware_warp_calibration import calibrate_hardware_warp

from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)

__all__ = [
    "HardwareWarpCalibrationStrategy",
    "HardwareWarpKnobs",
    "require_hardware_warp_config",
    "run_hardware_warp_calibration",
]

_FIELDS = (
    "max_degree",
    "min_degree",
    "ball_radius_mm",
    "basis_points_per_axis",
    "identifiability_floor",
    "pose_set_path",
    "artifact_path",
    "readout_term",
    "unit_id",
    "software_version",
)


@dataclass(frozen=True)
class HardwareWarpKnobs:
    """The resolved ``training.*`` knobs of one calibration arm."""

    max_degree: int
    min_degree: int
    ball_radius_mm: float
    basis_points_per_axis: int
    identifiability_floor: float
    pose_set_path: Path
    artifact_path: Path
    readout_term: str
    unit_id: str
    software_version: str


def require_hardware_warp_config(config: Any) -> HardwareWarpKnobs:
    """Read every knob of ``config.training`` or RAISE, naming the field.

    Attributes are read directly rather than defaulted: a knob the schema advertises but the
    strategy never reads is indistinguishable from one that works (pitfall #15), and a default
    here would let an arm run a calibration it did not declare.
    """
    training = getattr(config, "training", None)
    if training is None:
        raise ValueError("config.training is required by HardwareWarpCalibrationStrategy")
    values: dict[str, Any] = {}
    for field in _FIELDS:
        try:
            values[field] = getattr(training, field)
        except AttributeError:
            raise ValueError(
                f"training.{field} is required by HardwareWarpCalibrationStrategy but the "
                "config does not carry it"
            ) from None
    if not (1 <= int(values["min_degree"]) <= int(values["max_degree"])):
        raise ValueError(
            f"need 1 <= training.min_degree <= training.max_degree, got "
            f"{values['min_degree']}, {values['max_degree']}"
        )
    if float(values["ball_radius_mm"]) <= 0:
        raise ValueError(f"training.ball_radius_mm must be > 0, got {values['ball_radius_mm']}")
    if int(values["basis_points_per_axis"]) < 2:
        raise ValueError(
            f"training.basis_points_per_axis must be >= 2, got {values['basis_points_per_axis']}"
        )
    if float(values["identifiability_floor"]) <= 0:
        raise ValueError(
            f"training.identifiability_floor must be > 0, got {values['identifiability_floor']}"
        )
    if values["readout_term"] != "off":
        # The pose-set contract carries no within-session T1w/T2w residuals, so the readout
        # term ``f e_ro`` cannot be estimated here; accepting the value and skipping the
        # subtraction would be the silent fallback of non-negotiable 3.
        raise ValueError(
            f"training.readout_term={values['readout_term']!r} is not implemented: the pose set "
            "carries no within-session residuals to estimate the readout term from. Declare "
            "readout_term: off."
        )
    for field in ("unit_id", "software_version"):
        if not isinstance(values[field], str) or not values[field]:
            raise ValueError(f"training.{field} must be a non-empty string, got {values[field]!r}")
    pose_set_path = Path(values["pose_set_path"])
    if not pose_set_path.is_file():
        raise FileNotFoundError(
            f"training.pose_set_path resolved to {pose_set_path}, which is not a file. The "
            "calibration cannot run without the repeat-session pose set."
        )
    if not str(values["artifact_path"]):
        raise ValueError("training.artifact_path must name where the warp artifact is written")
    return HardwareWarpKnobs(
        max_degree=int(values["max_degree"]),
        min_degree=int(values["min_degree"]),
        ball_radius_mm=float(values["ball_radius_mm"]),
        basis_points_per_axis=int(values["basis_points_per_axis"]),
        identifiability_floor=float(values["identifiability_floor"]),
        pose_set_path=pose_set_path,
        artifact_path=Path(values["artifact_path"]),
        readout_term=str(values["readout_term"]),
        unit_id=values["unit_id"],
        software_version=values["software_version"],
    )


def run_hardware_warp_calibration(knobs: HardwareWarpKnobs) -> dict[str, Any]:
    """Load the pose set and residuals, fit, certify against the floor, write the artifact.

    The floor is enforced on the artifact's own ``constant_per_degree`` -- the certificate the
    fit was made under -- and a degree below it raises before anything is written, because an
    unidentifiable block would be reported as a measured warp.
    """
    pose_set = load_pose_set(knobs.pose_set_path)
    residuals = load_pair_residuals(pose_set, knobs.pose_set_path.parent)
    artifact = calibrate_hardware_warp(
        pose_set,
        residuals,
        max_degree=knobs.max_degree,
        min_degree=knobs.min_degree,
        ball_radius_mm=knobs.ball_radius_mm,
        basis_points_per_axis=knobs.basis_points_per_axis,
        unit_id=knobs.unit_id,
        software_version=knobs.software_version,
    )
    below = {
        degree: value
        for degree, value in artifact["constant_per_degree"].items()
        if value < knobs.identifiability_floor
    }
    if below:
        raise ValueError(
            f"identifiability constant below the floor {knobs.identifiability_floor:g} for "
            f"degree(s) {below}: the pose set does not determine those blocks. Add repeat "
            "sessions with a different head pose or lower training.max_degree."
        )
    path = write_artifact(artifact, knobs.artifact_path)
    logger.info(
        "HardwareWarpCalibration: %d pairs, constant per degree %s (floor %g), %d coefficients "
        "written to %s",
        artifact["n_pairs"],
        artifact["constant_per_degree"],
        knobs.identifiability_floor,
        len(artifact["coeffs"]),
        path,
    )
    return artifact


class HardwareWarpCalibrationStrategy(ReconstructionTrainingStrategy):
    """Identify the scanner's hardware warp from repeat-session commutators.

    A frozen-parameter orchestrator: it LEARNS NOTHING. Construction resolves and stamps every
    knob and confirms the pose set exists; the first training step runs
    :func:`run_hardware_warp_calibration` once and every step returns an exactly-zero,
    differentiable loss.
    """

    #: Loss ownership (issue #1918): computes no loss term and folds nothing.
    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = False

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("hardware_warp_calibration",))
        self._knobs = require_hardware_warp_config(self.config)
        self._artifact: dict[str, Any] | None = None
        # Resolved knobs stamped into provenance (pitfall #15); the fit's certificate and pair
        # count join them once the calibration has run.
        self.hardware_warp_provenance: dict[str, Any] = {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(self._knobs).items()
        }
        logger.info("HardwareWarpCalibrationStrategy knobs: %s", self.hardware_warp_provenance)

    @property
    def artifact(self) -> dict[str, Any] | None:
        """The written artifact, or ``None`` before the first step."""
        return self._artifact

    def _zero_loss(self) -> torch.Tensor:
        """An exactly-zero loss that is still differentiable w.r.t. the model, so the harness's
        ``backward()`` has a graph to walk while no parameter moves."""
        params = [p for p in self.env.generator.parameters() if p.requires_grad]
        if not params:
            return torch.zeros((), requires_grad=True)
        return sum(p.sum() for p in params) * 0.0

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if self._artifact is None:
            self._artifact = run_hardware_warp_calibration(self._knobs)
            self.hardware_warp_provenance.update(
                constant_per_degree=self._artifact["constant_per_degree"],
                n_pairs=self._artifact["n_pairs"],
                n_basis=len(self._artifact["coeffs"]),
            )
        return {"loss_total": self._zero_loss()}
