"""SSOT for the cascading-acceleration validation sweep (issue #697).

The severity levels were declared **twice** -- `pipelines/training_loop.py` and
`infrastructure/training/strategies/diffusion.py` -- and agreed only by
coincidence. They must agree: the strategy evaluates at one list and the CSV
was labelled from the other, so a divergence mislabels every column silently.
That is worse than a missing column, because a mislabelled number is still a
number and gets read as one.

Lives in `core/` because both `pipelines/` and `infrastructure/` may import
rightward from here; `diffusion.py` importing from `training_loop.py` would be
a leftward import (non-negotiable #5).

## Why the record is tall

The sweep used to be flattened into `val_<metric>_<R>x` column names built from
a hardcoded 15-name list -- 45 names that were assembled and never added to the
header, so every value was discarded by the row writer's
``extrasaction="ignore"``. Encoding the level in the NAME has two costs beyond
that bug:

* the name list has to be maintained by hand and drifts from what an arm
  actually computes (a drained `kspace_filling` arm emits `val_hfen_8x`, which
  was in no list), and the held-out grid used a third naming convention
  (`_heldout_<R>x`) that appeared in none of them;
* it hides the relationship the reader needs. `diffusion.py` documents a real
  bug where a `step` schedule decoded R=8 back to R=4 -- ``R=8 -> t=200 ->
  step decodes back to R=4`` -- so the `val_*_8x` columns were labelled 8x
  while holding 4x measurements. With `acceleration_level` and `timestep` as
  separate columns that inconsistency is visible **in the data**; as a column
  name it required reading the schedule code to find.

So one row per (iteration, severity point), level and timestep as values.
"""

# Two of the three concerns live in sibling modules (300-LOC ceiling, NN20) and
# are re-exported here under their original names, against one definition each
# (NN17), so all 24 importers resolve them through this path unchanged.

from __future__ import annotations

import math
from typing import Any

from spectramr.core.cascade_levels import (
    CASCADE_ID_COLUMNS,
    CASCADING_LEVELS,
    UNRECORDED_SKIP_REASON,
    normalize_cascade_levels,
    reconcile_skipped_levels,
    resolve_cascade_levels,
)
from spectramr.core.cascade_round_trip import (
    IDENTITY_ACCELERATION,
    RoundTrip,
    check_round_trip,
    legacy_linear_timestep,
    training_band,
)

__all__ = [
    "CASCADE_ID_COLUMNS",
    "CASCADING_LEVELS",
    "IDENTITY_ACCELERATION",
    "UNRECORDED_SKIP_REASON",
    "RoundTrip",
    "aggregate_cascade_rows",
    "build_cascade_row",
    "check_round_trip",
    "flatten_band_records",
    "legacy_linear_timestep",
    "normalize_cascade_levels",
    "reconcile_skipped_levels",
    "resolve_cascade_levels",
    "training_band",
]


def build_cascade_row(
    *,
    acceleration_level: float,
    heldout: bool,
    timestep: float | None,
    metrics: dict[str, Any],
    acceleration_realized: float | None = None,
) -> dict[str, Any]:
    """One tall row: the severity point as data, metrics under their own names.

    `metrics` arrives UNSUFFIXED -- the same dict the per-level evaluation
    produced. No name list is consulted, which is the point: whatever the arm
    computed is what gets a column, so this cannot drift from the arm's
    declared metric set the way the retired 15-name list did.

    ``iteration``/``epoch`` are added by the writer, which knows them; the
    strategy that builds these rows does not.

    ``acceleration_level`` is what was REQUESTED; ``acceleration_realized`` is
    what the schedule decoded ``timestep`` back to. The module docstring above
    argues that carrying the timestep as data makes a declared-vs-realised
    divergence "visible in the data" -- but only to a reader willing to re-run
    the schedule inversion by hand. Carrying the decoded value makes it visible
    by SUBTRACTION (#1295). It is optional so a caller with no accelerator
    wired (the linear-fallback path) records ``None`` rather than restating the
    request and turning an unknown into a false confirmation.
    """
    row: dict[str, Any] = {
        "acceleration_level": float(acceleration_level),
        "heldout": bool(heldout),
        "timestep": None if timestep is None else float(timestep),
        "acceleration_realized": (
            None if acceleration_realized is None else float(acceleration_realized)
        ),
    }
    # Identity keys win: a metric literally named `timestep` must not silently
    # overwrite the severity point this row is about.
    row.update({k: v for k, v in metrics.items() if k not in row})
    return row


def aggregate_cascade_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse a validation's per-batch rows to one row per severity point.

    ``_run_validation`` calls ``validation_step`` once per val batch and the
    cascade is NOT gated on ``batch_idx``, so a sweep arrives as
    ``n_batches x n_levels`` rows. The suffixed ``val_*_<R>x`` keys beside them
    are accumulated and divided by the batch count, so publishing one batch here
    would put two different numbers under two labels that claim to be the same
    measurement -- the failure this record shape exists to prevent.

    Numeric values are averaged; non-numeric ones pass through from the first
    row of the group (dropping them is how the retired implementation lost 45
    columns). ``n_batches`` is emitted so the row states its own provenance; it
    is the GROUP size, so a column only some batches computed is averaged over
    the batches that had it while ``n_batches`` still reports the whole sweep.

    Grouping is by ``(acceleration_level, heldout)``: an R=8 held-out probe and
    an R=8 in-distribution point are different measurements, and averaging them
    would fold a robustness probe into the training regime. Group order follows
    first appearance, so the ascending sweep stays ascending.
    """
    grouped: dict[tuple[float, bool], list[dict[str, Any]]] = {}
    for row in rows:
        key = (float(row["acceleration_level"]), bool(row["heldout"]))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (level, heldout), group in grouped.items():
        merged: dict[str, Any] = {
            "acceleration_level": level,
            "heldout": heldout,
            "n_batches": len(group),
        }
        # UNION over the group, not `group[0]`'s keys: a metric only a later
        # batch computed would otherwise be dropped -- the retired 45-name
        # list's failure one level up. Order follows first appearance so the
        # column order stays stable across sweeps.
        keys: dict[str, None] = {}
        for row in group:
            keys.update(dict.fromkeys(row))
        for key in keys:
            if key in merged:
                continue
            values = [r[key] for r in group if r.get(key) is not None]
            # `bool` is an `int` subclass; excluded above as a grouping key, but
            # guard anyway so a boolean diagnostic is not averaged into 0.5.
            numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if numeric and len(numeric) == len(values):
                merged[key] = sum(numeric) / len(numeric)
            else:
                merged[key] = values[0] if values else None
        out.append(merged)
    return out


def flatten_band_records(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """Summarise per-reveal-band attribution into cascade-row columns.

    The full per-band table is the artifact a reader inspects for STEP
    DISCONTINUITIES — a gain that jumps at a reveal boundary is a band written
    by a differently-biased model call, and a jump in |k| along the phase-encode
    axis is what produces coherent ghosting there. A CSV row cannot hold that
    curve, so it carries the summary and the caller logs the table.

    Bands the reverse loop revealed nothing at (``n_bins == 0`` — the inert
    steps) contribute nothing: their gain is ``nan`` by construction and
    averaging it in would report an unmeasured step as a measured one.

    The observed support arrives as the ``step == -1`` record and is reported
    SEPARATELY rather than pooled: under ``dc_method='hard'`` it is pinned to the
    measurement, so its gain is the irreducible floor the inferred bands are read
    against, not another sample of the same quantity.

    Args:
        records: :func:`~spectramr.models.diffusion.reveal_attribution.attribute_reveal_bands`
            output.

    Returns:
        Flat ``val_band_*`` columns. Values are ``None`` when no band was
        measurable, which is distinct from a gain of zero.
    """
    inferred = [r for r in records if r["step"] >= 0 and int(r["n_bins"]) > 0]
    observed = next((r for r in records if r["step"] == -1), None)
    # `worst` is selected from the MEASURED records, and the reported minimum is
    # then read off it. Selecting from `inferred` while filtering the moduli
    # separately let the two columns name different bands: `min` seeds on
    # element 0 and every `k < nan` is False, so a leading nan record survived
    # as "worst" while `val_band_gain_modulus_min` came from a real one -- the
    # column labelled worst pointing at the band that was never measured. A band
    # with `n_bins > 0` and zero target energy produces exactly that
    # (reveal_attribution.py returns `(nan, nan)` for it), which is reachable at
    # high |k| on a cropped or zero-padded acquisition.
    measured = [r for r in inferred if not math.isnan(r["gain_modulus"])]
    moduli = [float(r["gain_modulus"]) for r in measured]
    phases = [
        float(r["gain_phase_rad"]) for r in inferred if not math.isnan(r["gain_phase_rad"])
    ]
    worst = min(measured, key=lambda r: float(r["gain_modulus"]), default=None)
    return {
        "val_band_count": float(len(inferred)),
        "val_band_gain_modulus_min": float(worst["gain_modulus"]) if worst is not None else None,
        "val_band_gain_modulus_mean": (sum(moduli) / len(moduli)) if moduli else None,
        "val_band_phase_rad_absmax": max(abs(p) for p in phases) if phases else None,
        "val_band_phase_rad_absmean": (sum(abs(p) for p in phases) / len(phases))
        if phases
        else None,
        "val_band_worst_timestep": float(worst["timestep"]) if worst is not None else None,
        "val_band_observed_gain_modulus": (
            None if observed is None else float(observed["gain_modulus"])
        ),
        "val_band_observed_phase_rad": (
            None if observed is None else float(observed["gain_phase_rad"])
        ),
    }
