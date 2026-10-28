"""``kspace_cold_diffusion`` arms whose ``model_kwargs`` their backbone cannot take (#2047).

``force_pure_kspace`` on an image-domain-DC backbone and ``kspace_feature_norm`` /
the radial-band-token knobs off ``complex_unet`` make the generator raise at
construction. Before this witness ``audit --strict`` reported 146/146 passed on
four such arms, which then died seconds into the run. The rules are not restated
here: :func:`backbone_kwargs_defects` owns them and the constructor raises on the
same list, so the audit and the build cannot disagree.

Registration is by import (the witness package walk).
"""

from __future__ import annotations

from spectramr.infrastructure.validation.witness.registry import (
    Applicability,
    Severity,
    Stage,
    Subject,
    Tier,
    WitnessVerdict,
    register_witness,
)
from spectramr.infrastructure.validation.witness.subject import WitnessSubject

__all__ = ["kspace_backbone_kwargs_compatible", "kspace_backbone_kwargs_defects"]

_CATEGORY = "model_kwargs"
_NAME = "kspace_backbone_kwargs_compatible"
_MODEL_TYPE = "kspace_cold_diffusion"
_FIX_HINT = (
    "Change model.model_kwargs so backbone_type can consume every key, as each "
    "message says; the generator raises on the same rules at construction."
)


def kspace_backbone_kwargs_defects(raw: dict | None) -> list[str]:
    """The generator's own verdict on ``model.model_kwargs``; empty when buildable."""
    model = (raw or {}).get("model") or {}
    # Function-local: keeps model construction code out of the witness walk's
    # import graph until an applicable arm is actually checked.
    from spectramr.models.generators.kspace_cold_diffusion_generator import (
        backbone_kwargs_defects,
    )

    return backbone_kwargs_defects(model.get("model_kwargs") or {})


@register_witness(
    _NAME,
    category=_CATEGORY,
    stage=Stage.CONSTRUCT,
    tiers=(Tier.T0, Tier.T1),
    subjects=(Subject.CONFIG,),
    severity=Severity.ERROR,
    description="model_kwargs are legal for the declared kspace_cold_diffusion backbone_type",
    applicability=Applicability(model_types=frozenset({_MODEL_TYPE})),
    fix_hint=_FIX_HINT,
)
def kspace_backbone_kwargs_compatible(subject: WitnessSubject) -> WitnessVerdict:
    """Error on every backbone x model_kwargs combination the generator refuses."""
    defects = kspace_backbone_kwargs_defects(subject.raw_config)
    if not defects:
        return WitnessVerdict(
            witness_name=_NAME,
            passed=True,
            message="model_kwargs are legal for the declared backbone_type",
            severity=Severity.INFO,
            category=_CATEGORY,
            stage=Stage.CONSTRUCT,
            tier=Tier.T1,
        )
    return WitnessVerdict(
        witness_name=_NAME,
        passed=False,
        message=" | ".join(defects),
        severity=Severity.ERROR,
        category=_CATEGORY,
        stage=Stage.CONSTRUCT,
        tier=Tier.T1,
        yaml_keys=("model.model_kwargs",),
        fix_hint=_FIX_HINT,
    )
