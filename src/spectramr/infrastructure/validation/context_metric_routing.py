"""Which configured validation metrics a strategy can actually score (#2703).

A metric that declares it needs context (coil maps, the measurement, an alternate input)
raises without it. Only some validation paths build that context: the shared
``ModelValidationMixin`` path for every metric whose needs it can fill, and diffusion for
its fixed set of trust functionals. A context metric configured on any other strategy
raises inside that strategy's validation step, where ``gan.py`` turns the exception into a
warning and drops every validation metric of the run.

This module answers the question from the strategy CLASS, so the audit can refuse the
combination before a job is submitted.
"""

from __future__ import annotations

from collections.abc import Iterable

from spectramr.core.metrics.registry import MetricsRegistry


def configured_metric_names(config: object) -> list[str]:
    """The metric names ``validation.scoring.compute`` lists, specs reduced to their names."""
    validation = getattr(config, "validation", None)
    scoring = getattr(validation, "scoring", None)
    compute = getattr(scoring, "compute", None) or []
    return [m if isinstance(m, str) else str(getattr(m, "name", m)) for m in compute]


#: Metric output transforms that read the coil maps off the validation batch. Only the
#: shared validation path hands ``_metric_transform_views`` that batch; every other
#: strategy calls it with the tensors alone, so the transform raises
#: ``SenseMapsMissingError`` in the validation step (#2703).
BATCH_MAP_TRANSFORMS: frozenset[str] = frozenset({"sense_magnitude"})

#: A metric whose only need is the batch's coil maps; whether a strategy routes it is
#: whether it hands the batch to the metric transform.
_BATCH_MAP_PROBE_METRIC = "sense_psnr"


def configured_output_transform(config: object) -> str | None:
    """``validation.scoring.output_transform``, or ``None`` when unset."""
    scoring = getattr(getattr(config, "validation", None), "scoring", None)
    name = getattr(scoring, "output_transform", None)
    return name if isinstance(name, str) else None


def unrouted_output_transform(strategy_cls: type, transform: str | None) -> str | None:
    """``transform`` when it needs the validation batch's coil maps and ``strategy_cls`` does
    not pass the batch on; otherwise ``None``."""
    if transform not in BATCH_MAP_TRANSFORMS:
        return None
    routes = getattr(strategy_cls, "routes_context_metric", None)
    return None if callable(routes) and routes(_BATCH_MAP_PROBE_METRIC) else transform


def unrouted_context_metrics(strategy_cls: type, names: Iterable[str]) -> list[str]:
    """Configured context metrics that ``strategy_cls`` never gives their context.

    A class without ``routes_context_metric`` routes nothing. A name unknown to the
    registry needs no context as far as the registry can say; the metric-name audit owns it.
    """
    routes = getattr(strategy_cls, "routes_context_metric", None)
    return [
        name
        for name in names
        if MetricsRegistry.needs_context(name) and not (callable(routes) and routes(name))
    ]
