"""Adapters that republish the three pre-existing check stacks as witnesses.

Nothing is rewritten. `ConfigHealthChecker` keeps its 126 methods,
`ValidatorRegistry` keeps its 17+ rules, `compatibility_matrix` keeps its
`_RULES` list. Each is wrapped once so a single gate can run all three, which is
the actual fix for the split:

* ``bootstrap.build_container`` ran ``ValidatorRegistry`` only — it had never run
  ``ConfigHealthChecker``, so a config could start training with health findings
  nothing had looked at,
* ``cli/app.py::_audit_one`` ran ``ConfigHealthChecker`` + ``compatibility_matrix``
  only, with zero references to ``get_validator_registry`` — so the audit could
  pass a config that `train` would then reject at startup.

Both directions were real, which is why this is not "the audit is a superset of
train". Two stacks that never compare notes is the ``META`` failure class: each
looks thorough, and the union is what nobody checked.

Subject kinds are what let one gate serve both. The registry adapter reads the
raw dict, the other two need resolved settings, and the gate only schedules a
witness whose subjects the surface can actually provide.
"""

from __future__ import annotations

import logging

from spectramr.infrastructure.validation.witness.registry import (
    Severity,
    Stage,
    Subject,
    Tier,
    WitnessVerdict,
    register_witness,
)
from spectramr.infrastructure.validation.witness.subject import WitnessSubject

logger = logging.getLogger(__name__)

__all__ = ["health_result_to_verdict", "verdict_to_health_result"]

_SEVERITY = {
    "error": Severity.ERROR,
    "warning": Severity.WARNING,
    "info": Severity.INFO,
}


def health_result_to_verdict(result, *, prefix: str) -> WitnessVerdict:
    """Republish a ``HealthCheckResult`` without changing its meaning."""
    return WitnessVerdict(
        witness_name=f"{prefix}:{result.check_name}",
        passed=bool(result.passed),
        message=result.message,
        severity=_SEVERITY.get(str(result.severity), Severity.WARNING),
        category=result.category or "unclassified",
        yaml_keys=tuple(getattr(result, "yaml_keys", ()) or ()),
        fix_hint=getattr(result, "fix_hint", None),
        tier=Tier.T1,
    )


def verdict_to_health_result(verdict: WitnessVerdict):
    """Bridge back, so the audit's rendering / --json / exit codes are untouched.

    The CLI only ever consumed ``HealthCheckResult``; converting at the boundary
    keeps the unification additive instead of a CLI rewrite.
    """
    from spectramr.infrastructure.validation.config_health_checker import (
        HealthCheckResult,
    )

    return HealthCheckResult(
        passed=verdict.passed,
        check_name=verdict.witness_name,
        message=verdict.message,
        severity=str(verdict.severity),
        category=verdict.category,
        yaml_keys=list(verdict.yaml_keys),
        fix_hint=verdict.fix_hint,
    )


@register_witness(
    "legacy.config_health_checker",
    category="legacy_adapter",
    stage=Stage.PARSE,
    tiers=(Tier.T0, Tier.T1),
    subjects=(Subject.SETTINGS,),
    description="The 126 ConfigHealthChecker methods, republished as verdicts.",
)
def config_health_checker(subject: WitnessSubject) -> list[WitnessVerdict]:
    from spectramr.infrastructure.validation.config_health_checker import (
        validate_config_health,
    )

    # ``log_summary=False``: every result below is republished as a verdict and
    # rendered by the witness surface. Letting the checker narrate as well
    # printed "Config Health: n/m checks passed" twice in a train run —
    # once from the pipeline's own fail-fast gate, once from here via bootstrap.
    report = validate_config_health(subject.settings, log_summary=False)
    return [health_result_to_verdict(r, prefix="health") for r in report.results]


@register_witness(
    "legacy.compatibility_matrix",
    category="legacy_adapter",
    stage=Stage.DISPATCH,
    tiers=(Tier.T0, Tier.T1),
    subjects=(Subject.SETTINGS,),
    description="Cross-component compatibility rules, republished as verdicts.",
)
def compatibility_matrix(subject: WitnessSubject) -> list[WitnessVerdict]:
    from spectramr.infrastructure.validation.compatibility_matrix import (
        validate_experiment_compatibility,
    )

    # CompatMessage is a near-twin of HealthCheckResult but carries `rule`
    # instead of `check_name` and has no `passed` field: severity IS the verdict.
    return [
        WitnessVerdict(
            witness_name=f"compat:{m.rule}",
            passed=(m.severity != "error"),
            message=m.message,
            severity=_SEVERITY.get(str(m.severity), Severity.WARNING),
            category=m.category,
            yaml_keys=tuple(getattr(m, "yaml_keys", ()) or ()),
            fix_hint=m.fix_hint,
            tier=Tier.T1,
        )
        for m in validate_experiment_compatibility(subject.settings)
    ]


@register_witness(
    "legacy.validator_registry",
    category="legacy_adapter",
    stage=Stage.DECLARE,
    tiers=(Tier.T0, Tier.T1),
    subjects=(Subject.SETTINGS,),
    severity=Severity.WARNING,
    description=(
        "ValidatorRegistry + training-mode dispatchability, previously reachable "
        "only from the train/bootstrap path and never from `spectramr audit`."
    ),
)
def validator_registry(subject: WitnessSubject) -> list[WitnessVerdict]:
    """Wraps ``ConfigValidator.validate``, not the registry directly.

    Going through ``ConfigValidator`` preserves two things a bare registry call
    would drop: the training-mode dispatchability check (owned by
    ``STRATEGY_CLASS_PATHS``, so it lives outside the config-layer registry) and
    the exact error text bootstrap has always raised.
    """
    from spectramr.infrastructure.validation.config_validation import (
        ConfigValidationError,
        ConfigValidator,
    )

    try:
        ConfigValidator.validate(subject.settings)
    except ConfigValidationError as exc:
        return [
            WitnessVerdict(
                witness_name="legacy.validator_registry",
                passed=False,
                message=str(exc),
                # WARNING, not ERROR: bootstrap already raises on these at startup,
                # and the audit has not yet been run corpus-wide at ERROR to show the
                # rule set is satisfiable. Ratchet once it has.
                severity=Severity.WARNING,
                category="config_validation",
                tier=Tier.T1,
                fix_hint=(
                    "Findings from ValidatorRegistry, which `spectramr audit` never "
                    "ran before; `spectramr train` raises on the same findings at startup."
                ),
            )
        ]
    return [
        WitnessVerdict(
            witness_name="legacy.validator_registry",
            passed=True,
            message="ValidatorRegistry and training-mode dispatch are satisfied.",
            severity=Severity.INFO,
            category="config_validation",
            tier=Tier.T1,
        )
    ]
