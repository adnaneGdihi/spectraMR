"""Tier-1 check: the repeat-session pose set identifies every harmonic degree of the warp.

The hardware-warp calibration (``training_mode: hardware_warp_calibration``) observes the warp
``v`` only through ``ad_xi v`` for the rigid motions ``xi`` between sessions, and Theorem 2.1 of
``TODO/ULF_paired_64mT_3T_breakthrough_candidate.md`` gives the per-degree constant
``omega^2 (1 - |P_l(cos theta)|)`` two rotation axes at angle ``theta`` guarantee. Rotations
about one axis leave the axisymmetric harmonics in the kernel, so the least-squares fit would
still return an answer -- a minimum-norm one, with an unconstrained direction -- and nothing
downstream would notice. This check certifies the measured pose set before the fit runs:
:func:`~spectramr.infrastructure.physics.hardware_warp_commutator.identifiability_constant`
on the declared ball and degree band, compared against ``training.identifiability_floor``.

The verdict is a plain record; :class:`~.config_health_checker.ConfigHealthChecker` owns the
emitted ``check_name`` so the check-names ledger (``scripts/ci/check_health_check_names.py``)
sees it. The pose set is read by its one owner, :mod:`spectramr.data.hardware_warp_artifacts`,
and turned into generators by :func:`~.hardware_warp_calibration.pose_set_generators`, the same
two calls the training strategy makes -- so the check certifies the file the fit will read, not
a re-parse of it. A pose set that is missing or malformed is a failed verdict naming the
defect, never a pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import ValidationError

from spectramr.config.schemas.training.hardware_warp import (
    TrainingConfigHardwareWarpCalibration,
)
from spectramr.data.hardware_warp_artifacts import load_pose_set
from spectramr.infrastructure.physics.hardware_warp_calibration import pose_set_generators
from spectramr.infrastructure.physics.hardware_warp_commutator import (
    HarmonicGradientBasis,
    ball_points,
    identifiability_constant,
)

if TYPE_CHECKING:
    from spectramr.config.settings import TrainingSettings

TRAINING_MODE = "hardware_warp_calibration"
YAML_KEYS = ["training.pose_set_path", "training.identifiability_floor", "training.max_degree"]
FIX_HINT = "add repeat sessions with a different head pose, or lower training.max_degree"


@dataclass(frozen=True)
class PoseSpreadVerdict:
    """What the checker turns into a ``HealthCheckResult``."""

    passed: bool
    message: str
    severity: str = "info"
    yaml_keys: list[str] = field(default_factory=list)
    fix_hint: str | None = None


def pose_spread_verdict(config: TrainingSettings) -> PoseSpreadVerdict:
    """Certify the pose set of a hardware-warp calibration arm; not applicable to any other."""
    training = getattr(config, "training", None)
    if str(getattr(training, "training_mode", "")).lower() != TRAINING_MODE:
        return PoseSpreadVerdict(True, f"not applicable: training_mode is not {TRAINING_MODE!r}.")
    try:
        knobs = TrainingConfigHardwareWarpCalibration.from_training_block(training)
    except ValidationError as exc:
        return PoseSpreadVerdict(
            False,
            f"training block does not validate as {TRAINING_MODE}: {exc}",
            "error",
            yaml_keys=YAML_KEYS,
            fix_hint="declare every required knob of TrainingConfigHardwareWarpCalibration",
        )
    # The file owner raises FileNotFoundError for an absent file and ValueError (which
    # json.JSONDecodeError subclasses) for anything that departs from the contract.
    try:
        generators = pose_set_generators(load_pose_set(knobs.pose_set_path))
    except (OSError, ValueError) as exc:
        return PoseSpreadVerdict(
            False,
            f"training.pose_set_path cannot be certified: {exc}",
            "error",
            yaml_keys=["training.pose_set_path"],
            fix_hint="point training.pose_set_path at a pose set that follows the JSON contract",
        )
    basis = HarmonicGradientBasis(
        ball_points(knobs.basis_points_per_axis, knobs.ball_radius_mm),
        knobs.max_degree,
        knobs.min_degree,
    )
    constants = identifiability_constant(basis, generators)
    below = sorted(d for d, c in constants.items() if c < knobs.identifiability_floor)
    table = ", ".join(f"l={d}: {c:.3e}" for d, c in sorted(constants.items()))
    summary = (
        f"{len(generators)} session pair(s); per-degree identifiability constants {{{table}}} "
        f"against floor {knobs.identifiability_floor:.3e}"
    )
    if below:
        return PoseSpreadVerdict(
            False,
            f"degree(s) {below} fall below the identifiability floor: {summary}. Rotations about "
            "one axis leave that degree's axisymmetric harmonics unobserved.",
            "error",
            yaml_keys=YAML_KEYS,
            fix_hint=FIX_HINT,
        )
    return PoseSpreadVerdict(True, f"pose set certified: {summary}.")


__all__ = [
    "FIX_HINT",
    "TRAINING_MODE",
    "YAML_KEYS",
    "PoseSpreadVerdict",
    "pose_spread_verdict",
]
