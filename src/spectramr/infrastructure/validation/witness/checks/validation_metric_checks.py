"""``validation_metric_names_resolve``: every checkpoint selector has a producer (T0.5).

``metrics.best_metric_name`` and ``early_stopping.metric`` name a key the
validation pass must produce. One arm once selected its checkpoint on
``val_nmse``, a key no path produced, so early stopping never fired. A selector
resolves when it is registered, emitted by the strategy
(``capabilities.emitted_metrics``) or written undeclared by the strategy's
validation path (``val_loss``, which ``MetricsMixin.validation_step`` does not
write, #2838).
It may also carry a ``val_`` prefix and a ``_<n>x`` / ``_mean`` cascade suffix,
because the validation pass reports per-rung and aggregated copies of a metric.
Error when the strategy has declared its emitted set (opt-in strictness); INFO
census line when it has not.

``validation.scoring.compute`` is not checked here. Those names are handed to
the metrics computer verbatim, so no suffix is stripped from them, and
``check_metric_names_are_registered`` owns them through the runtime's own
predicate (#2819).
"""

from __future__ import annotations

import re

from spectramr.core.metrics.validation_names import UNIVERSAL_VALIDATION_KEYS
from spectramr.infrastructure.validation.witness.registry import (
    Severity,
    Stage,
    Subject,
    Tier,
    WitnessVerdict,
    register_witness,
)
from spectramr.infrastructure.validation.witness.subject import WitnessSubject

__all__ = ["unresolved_metric_names", "validation_metric_names_resolve", "validation_owners_for"]

_NAME = "validation_metric_names_resolve"
_CATEGORY = "metrics_misconfiguration"
_SUFFIX = re.compile(r"_(mean|min|max|median|heldout_\d+x|\d+x)$")


def _bases(name: str) -> set[str]:
    raw = str(name)
    stripped = _SUFFIX.sub("", raw)
    out = {raw, stripped}
    for candidate in list(out):
        out.add(candidate.removeprefix("val_"))
        out.add(f"val_{candidate.removeprefix('val_')}")
    return out


def unresolved_metric_names(
    names: list[str], registered, emitted: frozenset[str], *, universal: frozenset[str]
) -> list[str]:
    """Names with no owner: not registered, not emitted, not written by the validation path."""
    known = set(emitted) | set(universal)
    out = []
    for name in names:
        bases = _bases(name)
        if any(registered(b) for b in bases) or bases & known:
            continue
        out.append(str(name))
    return out


def _declared_names(settings) -> list[str]:
    names: list[str] = []
    metrics = getattr(settings, "metrics", None)
    best = getattr(metrics, "best_metric_name", None)
    if best:
        names.append(str(best))
    early = getattr(settings, "early_stopping", None)
    monitor = getattr(early, "metric", None)
    if monitor:
        names.append(str(monitor))
    return names


def validation_owners_for(settings) -> tuple[frozenset[str], frozenset[str], str]:
    """``(declared emitted keys, undeclared keys its validation path writes, strategy name)``.

    An unresolvable strategy keeps the ``val_loss`` exemption: which path it
    takes is unknown, and the audit already reports the strategy as
    ``"unresolved"`` rather than blaming the selector.
    """
    try:
        from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

        cls = TrainingStrategyFactory().get_strategy_class(settings)
    except Exception:
        return frozenset(), UNIVERSAL_VALIDATION_KEYS, "unresolved"
    from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
        universal_validation_keys,
    )

    caps = getattr(cls, "capabilities", None)
    emitted = frozenset(getattr(caps, "emitted_metrics", frozenset()) or ())
    return emitted, universal_validation_keys(cls), cls.__name__


@register_witness(
    _NAME,
    category=_CATEGORY,
    stage=Stage.DECLARE,
    tiers=(Tier.T1,),
    subjects=(Subject.SETTINGS,),
    severity=Severity.ERROR,
    description="Every checkpoint selector name has an owner (registry or strategy)",
    fix_hint=(
        "Fix the name, register the metric with @register_metric, or declare it in the "
        "strategy's capabilities.emitted_metrics if its validation_step writes it."
    ),
)
def validation_metric_names_resolve(subject: WitnessSubject) -> WitnessVerdict:
    """Error (declared strategy) / advisory (undeclared) on an ownerless selector name."""
    from spectramr.core.metrics.registry import MetricsRegistry

    settings = subject.settings
    names = _declared_names(settings)
    if not names:
        return WitnessVerdict(
            witness_name=_NAME,
            passed=True,
            message="no checkpoint selector declared",
            severity=Severity.INFO,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T1,
        )
    emitted, universal, strategy_name = validation_owners_for(settings)
    unresolved = unresolved_metric_names(
        names, MetricsRegistry.is_registered, emitted, universal=universal
    )
    if not unresolved:
        return WitnessVerdict(
            witness_name=_NAME,
            passed=True,
            message=f"all {len(names)} selector name(s) resolve (strategy {strategy_name})",
            severity=Severity.ERROR,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T1,
        )
    if emitted:
        return WitnessVerdict(
            witness_name=_NAME,
            passed=False,
            message=(
                f"{len(unresolved)} name(s) are neither registered nor emitted by "
                f"{strategy_name}: {sorted(unresolved)}. A selector no path produces means the "
                "best checkpoint is never chosen."
            ),
            severity=Severity.ERROR,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T1,
            yaml_keys=(
                "metrics.best_metric_name",
                "early_stopping.metric",
            ),
        )
    return WitnessVerdict(
        witness_name=_NAME,
        passed=True,
        message=(
            f"UNVERIFIED: {sorted(unresolved)} are not registered and {strategy_name} declares "
            "no emitted_metrics; if its validation_step writes them, declare them."
        ),
        severity=Severity.INFO,
        category=_CATEGORY,
        stage=Stage.DECLARE,
        tier=Tier.T1,
    )
