"""``validation.scoring.measurement_maps`` needs a model validated with coil maps.

The measured-map scoring family is produced only on the cold-diffusion branch
that calibrates coil maps (``takes_coil_maps``). Declared on any other model the
knob would be read and never fire, so the arm is refused at audit time rather
than reporting a family that is silently absent (non-negotiable 8). The schema's
``Literal`` owns the vocabulary; this witness owns the model compatibility.

Registration is by import (the witness package walk).
"""

from __future__ import annotations

from spectramr.infrastructure.training.strategies.measured_map_scoring import (
    measurement_maps_defect,
)
from spectramr.infrastructure.validation.witness.registry import (
    Severity,
    Stage,
    Subject,
    Tier,
    WitnessVerdict,
    register_witness,
)
from spectramr.infrastructure.validation.witness.subject import WitnessSubject

__all__ = ["KNOB", "measurement_maps_need_coil_maps"]

_CATEGORY = "dead_knob"
_NAME = "measurement_maps_need_coil_maps"
KNOB = "validation.scoring.measurement_maps"
_FIX = f"Set {KNOB}: off, or use a kspace_cold_diffusion model."


@register_witness(
    _NAME,
    category=_CATEGORY,
    stage=Stage.DECLARE,
    tiers=(Tier.T0, Tier.T1),
    subjects=(Subject.CONFIG,),
    severity=Severity.ERROR,
    description="validation.scoring.measurement_maps is declared only on a coil-map model",
    fix_hint=_FIX,
)
def measurement_maps_need_coil_maps(subject: WitnessSubject) -> WitnessVerdict:
    """Error when the measured-map family is declared where it cannot be produced."""
    raw = subject.raw_config or {}
    scoring = ((raw.get("validation") or {}).get("scoring")) or {}
    mode = scoring.get("measurement_maps") if isinstance(scoring, dict) else None
    model_type = (raw.get("model") or {}).get("model_type")
    message = measurement_maps_defect(mode, model_type)
    if message is None:
        return WitnessVerdict(
            witness_name=_NAME,
            passed=True,
            message=f"{KNOB}={mode or 'off'!r} is honoured by model_type={model_type!r}",
            severity=Severity.INFO,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T0,
        )
    return WitnessVerdict(
        witness_name=_NAME,
        passed=False,
        message=message,
        severity=Severity.ERROR,
        category=_CATEGORY,
        stage=Stage.DECLARE,
        tier=Tier.T0,
        yaml_keys=(KNOB, "model.model_type"),
        fix_hint=_FIX,
    )
