"""One owner for the held-out severity grid (``validation.gates.held_out_severity_*``)."""

from __future__ import annotations

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
