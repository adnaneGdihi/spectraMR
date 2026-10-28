"""The ``MetricContext`` seam — measurement + priors for no-reference metrics.

Full-reference metrics answer "how close is ``pred`` to ``target``?". A
no-reference (NR) metric has *no* ``target``; it answers "is ``pred``
self-consistent with the acquisition physics and with learned priors?".
That question needs the *measurement* (acquired k-space, sampling mask,
coil maps), the *acquisition geometry* (acceleration, line order), and
optionally *learned assets* (a reconstructor, a diffusion prior, an SSL
encoder). None of those fit the ``(pred, target)`` metric contract.

``MetricContext`` is the single immutable bundle that carries them. It is
passed as an optional keyword to a metric's ``__call__`` — full-reference
metrics ignore it (their ``__call__`` already absorbs ``**kwargs``), so
backward compatibility is total. The registry records, per metric, which
fields it ``needs`` (see :func:`MetricsRegistry.needs`); the validation
loop, the audit ladder, and the sim2rank sweep use that to decide whether
a metric can be computed at all on a given cohort.

The field-name → consumer mapping in the comments is the SSOT for the
``needs=(...)`` tuples passed to ``@register_metric``; keep them in sync.

This module is dependency-free on purpose (only ``torch`` / stdlib) so it
can be imported from ``core/`` without dragging in ``infrastructure``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class MetricContext:
    """Immutable measurement + prior bundle for no-reference metrics.

    Every field is optional: a metric reads only the subset it declared via
    ``@register_metric(..., needs=(...))``. The trailing comment on each
    field lists the metric keys that consume it.
    """

    # --- measurement (forward model y = M F S x + n) ------------------------
    y_kspace: Tensor | None = (
        None  # acquired k-space y    (ndcr, prp_snr, are, csoe, aoqv, hallucination_index)
    )
    mask: Tensor | None = None  # sampling mask M       (ndcr, are, ssel, fmrs)
    coil_maps: Tensor | None = None  # ESPIRiT maps S        (ndcr, csoe, prp_snr)
    acceleration: float | None = None  # R, fixes Delta_R      (are)
    acq_order: Tensor | None = None  # k-space line acq order (aoqv)
    noise_cov: Tensor | None = None  # Sigma for CRLB        (qvcr)
    # The prediction's own coil k-space, linear, in y_kspace's layout (ndcr). Supplied
    # by the strategy that produced it rather than declared in ``needs``: without it
    # ndcr re-projects the coil-combined image, which floors on multi-coil noise.
    pred_kspace: Tensor | None = None

    # --- multi-contrast / quantitative --------------------------------------
    sibling_contrasts: dict[str, Tensor] | None = (
        None  # {contrast: tensor} (ccsa, ccpr, hallucination_index)
    )
    acq_params: dict[str, Any] | None = None  # TR/TE/FA per contrast (brc, qvcr)

    # --- foreground / region ------------------------------------------------
    foreground_mask: Tensor | None = None  # Omega (psdc, pegc, are, cpge, ssel, savs, ecfd, bnac)

    # --- learned assets -----------------------------------------------------
    reconstructor: Callable[..., Tensor] | None = (
        None  # R(.) (fmrs, aoqv, lrjs, fmed, pedu, prp_snr)
    )
    prior_model: Any | None = None  # diffusion prior (dpst, fitc, ksdp, pfld, hallucination_index)
    encoder: Any | None = None  # SSL encoder    (pmmd, ncpd)
    group_actions: list[Any] | None = None  # G for equivariance (fmed)

    # --- per-metric numeric tunables (from NRMetricConfig) ------------------
    # A flat, read-only view of cfg.metrics.nr_metrics so a metric can pull its
    # own knob without the schema object leaking into ``core``.
    nr_params: dict[str, Any] | None = None

    # --- stochasticity ------------------------------------------------------
    generator: torch.Generator | None = None  # RNG for stochastic metrics (pedu, fitc, pfld, ...)

    # --- multi-sample ensembles ---------------------------------------------
    ensemble_std: Tensor | None = None  # per-pixel std over N reverse samples (empirical_coverage)

    # --- repeated acquisitions ----------------------------------------------
    # The model's output on a second repetition of the same slice, same layout as
    # the prediction (rep_variance, bias_squared).
    pred_alt: Tensor | None = None
    # What prediction/target/pred_alt ARE -- 'kspace' or 'image' -- as the caller
    # states it; None leaves the metric's documented default (null_space_residual,
    # rep_variance and the rest of the coil-manifold family read k-space).
    signal_domain: str | None = None

    def has(self, *names: str) -> bool:
        """True iff every named field is non-None.

        Used by metrics to validate their declared ``needs`` are satisfied
        before computing, and to raise a precise error (no silent fallback,
        CLAUDE.md pitfall #9) when a required field is missing.
        """
        return all(getattr(self, n, None) is not None for n in names)

    def require(self, metric_key: str, *names: str) -> None:
        """Raise ``ValueError`` if any named field is missing.

        Preferred over a silent ``return nan`` inside a metric — an NR metric
        listed in the config but starved of its measurement context is a
        wiring bug the audit ladder should surface, not paper over.
        """
        missing = [n for n in names if getattr(self, n, None) is None]
        if missing:
            raise ValueError(
                f"metric {metric_key!r} requires MetricContext fields {missing} "
                f"but they were not supplied. Wire data.retain_kspace / coil maps "
                f"(see config_health_checker.check_nr_metric_context_wired)."
            )

    @classmethod
    def from_kwargs(cls, kwargs: dict[str, Any]) -> MetricContext:
        """Build a context from loose ``**kwargs`` (engine back-compat).

        The legacy sim2rank engine and ``GFactor``-style metrics pass
        ``sensitivity_maps=`` / ``acceleration=`` / ``y_kspace=`` as flat
        keyword arguments. This collects any field-named entries so an NR
        metric can accept either an explicit ``context=`` or the loose form.
        Unknown keys are ignored.
        """
        known = {f.name for f in fields(cls)}
        # Accept the common ``sensitivity_maps`` alias for ``coil_maps``.
        if "sensitivity_maps" in kwargs and "coil_maps" not in kwargs:
            kwargs = {**kwargs, "coil_maps": kwargs["sensitivity_maps"]}
        return cls(**{k: v for k, v in kwargs.items() if k in known})


def resolve_context(context: MetricContext | None, kwargs: dict[str, Any]) -> MetricContext:
    """Return ``context`` if given, else assemble one from loose ``kwargs``.

    The one-liner every NR metric uses at the top of ``__call__`` so it works
    whether the caller threads an explicit ``MetricContext`` (validation loop,
    in-package sim2rank) or flat keyword arguments (legacy engine).
    """
    if isinstance(context, MetricContext):
        return context
    return MetricContext.from_kwargs(kwargs)


__all__ = ["MetricContext", "resolve_context"]
