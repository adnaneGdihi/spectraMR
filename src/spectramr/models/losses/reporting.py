"""Render an arm's declared objective from the loss-weight SSOT.

The training strategies used to answer "what does this arm optimize?" from
hand-written lists of ``(name, enable_<name>, lambda_<name>)`` triples. Those
lists read only ``losses.reconstruction`` / ``losses.physics``, so an arm that
declares its objective in the domain lists — ``losses.image_losses``,
``losses.kspace_losses``, ``losses.complex_losses``, ``losses.latent_losses`` —
was reported as having no losses at all while training a full set (#1919).

Everything here reads :class:`~spectramr.models.losses.weights.LossWeightTable`,
which already resolves every declaration surface and records which one won. This
module adds no declaration surface of its own: it has no list of loss names.

**Read the DECLARED weight, never** ``table.weight(name)``. That accessor returns
the *effective* weight at an iteration and defaults to ``iteration=0``, so every
warmup-gated loss reads ``0.0`` — a banner built on it prints ``λ=0.0000`` for a
loss the YAML declares at ``1.0``, which is worse than printing nothing. The
declared value is ``LossWeightSpec.weight``; warmup is reported separately, as an
annotation, by :func:`format_loss_objective`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from spectramr.models.losses.weights import LossWeightSpec, LossWeightTable

__all__ = [
    "active_loss_names",
    "declared_losses",
    "format_loss_objective",
    "is_active",
]


def is_active(spec: LossWeightSpec) -> bool:
    """Whether ``spec`` contributes to the objective at its declared weight.

    A loss is active when it is enabled AND declared at a non-zero weight. This
    is the same predicate the old hand-written banners applied
    (``if enabled and weight > 0``), lifted onto the SSOT so it is stated once.

    ``warmup_gated`` deliberately does NOT enter this predicate: a warmup-gated
    loss is part of the objective, it simply ramps. Excluding it here would drop
    it from the CSV columns and reintroduce the mid-run column instability the
    zero-fill exists to prevent.
    """
    return bool(spec.enabled) and spec.weight > 0


def declared_losses(table: LossWeightTable) -> Iterator[LossWeightSpec]:
    """Every loss ``table`` resolves, active first, then alphabetically.

    Yields the table's own :class:`LossWeightSpec` objects — no copy, no second
    representation of a weight.
    """
    yield from sorted(table.values(), key=lambda s: (not is_active(s), s.name))


def active_loss_names(table: LossWeightTable) -> frozenset[str]:
    """The canonical names of every loss this arm actually optimizes.

    This is the set a per-step consumer needs (the CSV zero-fill in
    ``DiffusionTrainingStrategy._compute_losses_impl``). Build it ONCE and cache
    it: it is derived from the frozen config and cannot change between steps
    (non-negotiable 9).
    """
    return frozenset(spec.name for spec in table.values() if is_active(spec))


def _warmup_note(spec: LossWeightSpec, warmup_iterations: int) -> str:
    if not spec.warmup_gated or warmup_iterations <= 0:
        return ""
    return f"  [warmup: contributes 0 for the first {warmup_iterations} iterations]"


def _render(specs: Iterable[LossWeightSpec], warmup_iterations: int, mark: str) -> list[str]:
    return [
        f"  {mark} {spec.name:35s} λ={spec.weight:.4f}  ({spec.source})"
        f"{_warmup_note(spec, warmup_iterations)}"
        for spec in specs
    ]


def format_loss_objective(table: LossWeightTable, *, prefix: str) -> list[str]:
    """The startup banner for ``table``, as lines ready to log.

    Pure: takes a table, returns strings, touches no logger and no config. The
    caller decides the log level and the sink.

    Reports three things the hand-written banners could not:

    - the **source** of every weight, so a surprising number can be traced to the
      key that set it without re-reading the YAML;
    - **warmup gating**, annotated rather than folded into the number;
    - an arm that declares **nothing**, said out loud. The old banner emitted no
      lines at all in that case, which is indistinguishable from a banner that
      did not run (non-negotiable 18 — absent is a state to report, never one to
      infer).
    """
    active = [spec for spec in declared_losses(table) if is_active(spec)]
    inactive = [spec for spec in declared_losses(table) if not is_active(spec)]

    if not active:
        reason = (
            "this arm declares no loss weights at all"
            if not inactive
            else f"all {len(inactive)} declared losses are disabled or weighted 0"
        )
        lines = [f"{prefix} Configured Losses (0): {reason}."]
    else:
        lines = [f"{prefix} Configured Losses ({len(active)}):"]
        lines += _render(active, table.warmup_iterations, "✓")

    if inactive:
        lines.append(f"{prefix} Declared but inactive ({len(inactive)}):")
        lines += _render(inactive, table.warmup_iterations, "·")

    return lines
