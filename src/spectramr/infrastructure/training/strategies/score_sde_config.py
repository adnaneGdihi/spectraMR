"""Config-only rules of the score-SDE path, shared by the engine build and the static audit.

The engine (``score_sde_engine.build_score_sde_engine``) and the health check
``score_sde_declaration`` both call :func:`validate_score_sde_config`, so a YAML that cannot
build is rejected when it loads and the two never drift.
"""

from __future__ import annotations

from typing import Any

from spectramr.domain.exceptions import ConfigurationError
from spectramr.models.diffusion.samplers import SamplerRegistry
from spectramr.models.diffusion.score_sde.sde_spec import resolve_sde

#: Registry names of the generator that reads ``sde_type``/``sigma_min``/``sigma_max``.
SCORE_SDE_MODEL_TYPES = frozenset(
    {"score_based_diffusion", "score_based_diffusion_generator", "score_based_unet"}
)


def declared_sde(config: Any) -> tuple[str | None, float | None, float | None]:
    """The SDE ``model.model_kwargs`` declares for a score generator; raise if it is illegal."""
    model = getattr(config, "model", None)
    if str(getattr(model, "model_type", "") or "").lower() not in SCORE_SDE_MODEL_TYPES:
        return None, None, None
    kwargs = getattr(model, "model_kwargs", None) or {}
    try:
        return resolve_sde(kwargs.get("sde_type"), kwargs.get("sigma_min"), kwargs.get("sigma_max"))
    except ValueError as exc:
        raise ConfigurationError(f"model.model_kwargs: {exc}") from exc


def validate_score_sde_config(config: Any) -> None:
    """Raise ``ConfigurationError`` on any config the score-SDE engine cannot run."""
    physics = getattr(config, "physics", None)
    dc = getattr(physics, "data_consistency", None) if physics else None
    if dc is None or not dc.enabled:
        raise ConfigurationError("sde_type needs physics.data_consistency enabled with a sampler.")
    if not SamplerRegistry.is_registered(dc.method):
        raise ConfigurationError(
            f"physics.data_consistency.method={dc.method!r} is not a registered sampler "
            "(for a VE-SDE arm: 'pc_sampler_consistency')."
        )
    if getattr(config.data, "image_undersampling", False):
        raise ConfigurationError(
            "The score-SDE path builds its own measurement from the delivered image; "
            "data.image_undersampling would mask an already-aliased magnitude a second time."
        )
    us = getattr(config, "undersampling", None)
    if us is None or len(us.mask_types) != 1:
        raise ConfigurationError(
            "The score-SDE path needs an undersampling block with exactly one mask_types entry."
        )
    if dc.eval_noise_level != 0.0 or dc.interval != 1:
        raise ConfigurationError(
            "pc_sampler_consistency applies noise-free DC at every update; "
            "physics.data_consistency.eval_noise_level/interval are not read by it."
        )
    line_axis(config)


def line_axis(config: Any) -> int:
    """Axis the sampled lines run along: the phase-encode axis, or the other one for 'readout'."""
    acquisition = getattr(config.data, "acquisition", None)
    pe_axis = int(getattr(acquisition, "phase_encode_axis", -2) or -2)
    direction = getattr(config.undersampling, "mask_direction", None)
    if direction is None or direction == "phase":
        return pe_axis
    if direction == "readout":
        return -1 if pe_axis == -2 else -2
    raise ConfigurationError(
        f"undersampling.mask_direction={direction!r}: the score-SDE measurement takes "
        "'phase' or 'readout'."
    )
