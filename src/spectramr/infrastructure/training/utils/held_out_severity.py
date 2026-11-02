"""One owner for the held-out severity grid (``validation.gates.held_out_severity_*``).

Each reader interprets the grid: cold diffusion as accelerations, motion_meta as multipliers of
``training.motion``, virtual_fiducial as multipliers of the twin's motion, B0 and B1 strengths.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


def held_out_severity_points(config: Any) -> list[float]:
    """Severities to evaluate beyond the training range; ``[]`` unless the gate is on.

    The grid lives at ``validation.gates.held_out_severity_grid``; arms that predate it keep it
    under ``model.model_kwargs.digital_twin_kwargs.held_out_severity_grid``, which is read when
    the gate field is unset.
    """
    val = config.validation
    gates = getattr(val, "gates", None) if val is not None else None
    if not (gates.held_out_severity_eval if gates else False):
        return []
    grid = gates.held_out_severity_grid
    if grid is None:
        mk = config.model.model_kwargs or {}
        dtk = (
            mk.get("digital_twin_kwargs")
            if isinstance(mk, dict)
            else getattr(mk, "digital_twin_kwargs", None)
        ) or {}
        grid = (
            dtk.get("held_out_severity_grid")
            if isinstance(dtk, dict)
            else getattr(dtk, "held_out_severity_grid", None)
        )
    if not grid:
        return []
    try:
        return [float(s) for s in grid]
    except (TypeError, ValueError):
        return []


#: Written on every validation step of a reading strategy, so "no grid" and "gate off" are
#: told apart from "no held-out rows" in the log.
HELD_OUT_COUNT_KEY = "val_heldout_severities"


def held_out_metric_key(scale: float, name: str) -> str:
    """``val_hfen_heldout_x2`` for ``(2.0, "val_hfen")``; the ``val_`` prefix is not doubled."""
    return f"val_{name.removeprefix('val_')}_heldout_x{scale:g}"


def held_out_severity_readout(
    simulator: Any,
    scales: Sequence[float],
    score: Callable[[], Mapping[str, float]],
) -> dict[str, float]:
    """Score once per multiplier with the twin held at it (``simulator.at_severity``)."""
    out: dict[str, float] = {}
    for scale in scales:
        with simulator.at_severity(scale):
            scored = score()
        for name, value in scored.items():
            out[held_out_metric_key(scale, name)] = float(value)
    out[HELD_OUT_COUNT_KEY] = float(len(scales))
    return out
