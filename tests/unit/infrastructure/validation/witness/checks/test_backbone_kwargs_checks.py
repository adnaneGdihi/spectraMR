"""Planted violations for ``kspace_backbone_kwargs_compatible`` (#2047)."""

from __future__ import annotations

import pytest

from spectramr.infrastructure.validation.witness import Tier, WitnessSubject, run_witnesses
from spectramr.infrastructure.validation.witness.checks import backbone_kwargs_checks as bkc

_TIERS = frozenset({Tier.T0, Tier.T1})


def _raw(model_kwargs: dict, model_type: str = "kspace_cold_diffusion") -> dict:
    return {"model": {"model_type": model_type, "model_kwargs": model_kwargs}}


def _subject(raw: dict) -> WitnessSubject:
    return WitnessSubject.for_ci("arm.yaml", raw)


# The four arms of #2047, reduced to the keys that kill them.
@pytest.mark.parametrize(
    ("model_kwargs", "needle"),
    [
        ({"backbone_type": "swin_diff_rec", "force_pure_kspace": True}, "wrong domain"),
        ({"backbone_type": "diff_varnet", "kspace_feature_norm": "rms"}, "not 'diff_varnet'"),
        ({"backbone_type": "nafnet", "kspace_feature_norm": "rms"}, "not 'nafnet'"),
    ],
)
def test_the_issue_arms_are_rejected(model_kwargs: dict, needle: str) -> None:
    verdict = bkc.kspace_backbone_kwargs_compatible(_subject(_raw(model_kwargs)))
    assert verdict.passed is False and needle in verdict.message
    assert verdict.severity.value == "error"


def test_a_buildable_arm_passes() -> None:
    raw = _raw(
        {"backbone_type": "complex_unet", "force_pure_kspace": True, "kspace_feature_norm": "rms"}
    )
    assert bkc.kspace_backbone_kwargs_compatible(_subject(raw)).passed is True


def test_the_audit_gate_schedules_it_and_fails_the_arm() -> None:
    """Reachability: the T0/T1 ladder ``spectramr audit`` runs calls the witness."""
    raw = _raw({"backbone_type": "diff_varnet", "kspace_feature_norm": "rms"})
    verdicts = [
        v for v in run_witnesses(_subject(raw), tiers=_TIERS) if v.witness_name == bkc._NAME
    ]
    assert len(verdicts) == 1 and verdicts[0].passed is False


def test_other_model_types_are_not_judged_by_these_rules() -> None:
    raw = _raw({"backbone_type": "diff_varnet", "kspace_feature_norm": "rms"}, model_type="unet")
    names = {v.witness_name for v in run_witnesses(_subject(raw), tiers=_TIERS)}
    assert bkc._NAME not in names
