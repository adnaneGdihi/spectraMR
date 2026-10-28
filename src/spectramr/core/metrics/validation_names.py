"""Which ``validation.scoring.compute`` names are legal: the one owner of that rule (#2819).

A name in that list is owned by exactly one of two producers. The metrics
registry computes it, or the strategy's own ``validation_step`` writes it. The
strategy declares the second kind in ``capabilities.emitted_metrics``, and
``val_loss`` is written by ``ModelValidationMixin.validation_step``. It is not
written by ``MetricsMixin.validation_step``, which returns only the registry
computer's keys, so the caller says which universal keys its strategy's
validation path writes (#2838). A name neither producer owns is a column that
never appears.

The training runtime (``MetricsMixin._registry_backed_validation_metrics``) and
the audit (``ConfigHealthChecker.check_metric_names_are_registered``) both read
this module. They used to keep separate copies, and the audit's registry-only
copy rejected ``val_loss``, which every run produces. The two still differ in
polarity, which is deliberate: the runtime only logs an unowned name on a
strategy that has declared no emitted set, while the audit errors on it.

The strategy is passed in as its emitted set, so this module stays in ``core``
and never imports a strategy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

__all__ = ["UNIVERSAL_VALIDATION_KEYS", "is_emitted", "partition_validation_compute"]

#: Written by ``ModelValidationMixin.validation_step`` without a declaration;
#: ``MetricsMixin.validation_step`` does not write them.
UNIVERSAL_VALIDATION_KEYS: frozenset[str] = frozenset({"loss", "val_loss"})


def is_emitted(name: str, emitted: Iterable[str], *, universal: Iterable[str]) -> bool:
    """True when the strategy's validation path writes ``name`` itself.

    ``universal`` is the undeclared set that path writes: ``UNIVERSAL_VALIDATION_KEYS``
    or empty. It has no default so no caller inherits an exemption its path does
    not earn. Only the ``val_`` prefix is normalised. A cascade suffix such as
    ``_4x`` is not stripped: the metrics computer is handed the name verbatim, so
    ``psnr_4x`` is not ``psnr`` here.
    """
    owned = frozenset(emitted) | frozenset(universal)
    candidates = {name, f"val_{name}", name.removeprefix("val_")}
    return bool(candidates & owned)


def partition_validation_compute(
    declared: Iterable[str],
    emitted: Iterable[str],
    is_registered: Callable[[str], bool],
    *,
    universal: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Split declared names into ``(registry_names, unowned)``.

    ``registry_names`` are the names the metrics computer builds; emitted names
    are dropped from it because the strategy produces them. ``unowned`` are the
    names no producer owns, in declaration order.
    """
    emitted_set = frozenset(emitted)
    universal_set = frozenset(universal)
    registry_names: list[str] = []
    unowned: list[str] = []
    for raw in declared:
        name = str(raw)
        if is_registered(name):
            registry_names.append(name)
        elif not is_emitted(name, emitted_set, universal=universal_set):
            unowned.append(name)
    return registry_names, unowned
