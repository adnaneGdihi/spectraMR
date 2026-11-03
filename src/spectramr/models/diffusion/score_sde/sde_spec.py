"""Validation of a score generator's declared SDE, and the refusal of the discrete path."""

from __future__ import annotations

from typing import Any

#: SDE families a score generator can be trained and sampled as; ``None`` keeps discrete DDPM.
SUPPORTED_SDE_TYPES = frozenset({"vesde"})


def resolve_sde(
    sde_type: str | None, sigma_min: float | None, sigma_max: float | None
) -> tuple[str | None, float | None, float | None]:
    """Validate the declared SDE; raise on an unknown family or an incomplete schedule."""
    if sde_type is None:
        if sigma_min is not None or sigma_max is not None:
            raise ValueError("sigma_min/sigma_max are declared without an sde_type.")
        return None, None, None
    sde = str(sde_type).lower()
    if sde not in SUPPORTED_SDE_TYPES:
        raise ValueError(
            f"Unknown sde_type {sde_type!r}; supported: {sorted(SUPPORTED_SDE_TYPES)}."
        )
    if sigma_min is None or sigma_max is None:
        raise ValueError(f"sde_type={sde!r} needs both sigma_min and sigma_max declared.")
    if not 0.0 < float(sigma_min) < float(sigma_max):
        raise ValueError(f"Need 0 < sigma_min < sigma_max, got {sigma_min}, {sigma_max}.")
    return sde, float(sigma_min), float(sigma_max)


def refuse_discrete_path(model: Any, what: str) -> None:
    """Raise: a VE-SDE model driven by the discrete DDPM machinery would ignore its SDE."""
    raise RuntimeError(
        f"{type(model).__name__} is configured with sde_type={model.sde_type!r} but was driven "
        f"through {what}. Its training and validation run through the score-SDE engine "
        "(infrastructure/training/strategies/score_sde_engine.py), which conditions on t * (N - 1) for t in [eps, 1]."
    )
