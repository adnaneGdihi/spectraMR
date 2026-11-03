"""resolve_cascade_schedule reads a training.diffusion block or its dict dump the same way."""

from __future__ import annotations

import pytest

from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
from spectramr.infrastructure.training.utils.cascade_sampler import resolve_cascade_schedule

_DECLARED = {
    "timesteps": 1000,
    "noise_schedule": "linear",
    "prediction_type": "sample",
    "sampler": "ddim",
    "sampling_steps": 50,
}


def _block(**overrides: object) -> object:
    return TrainingStrategyConfigSchema.model_validate(
        {"diffusion": {**_DECLARED, **overrides}}
    ).diffusion


def test_schema_and_dump_resolve_identically() -> None:
    block = _block()
    from_schema = resolve_cascade_schedule(block, None)
    assert from_schema == resolve_cascade_schedule(block.model_dump(), None)  # type: ignore[attr-defined]
    assert (from_schema.num_timesteps, from_schema.sampling_steps) == (1000, 50)
    assert from_schema.beta_schedule == "linear"


def test_validation_steps_win_and_clamp_to_the_horizon() -> None:
    assert resolve_cascade_schedule(_block(), 7).sampling_steps == 7
    assert resolve_cascade_schedule(_block(), 5000).sampling_steps == 1000


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"sampler": "ddpm"}, "sampler"),
        ({"prediction_type": "epsilon"}, "prediction_type"),
        ({"noise_schedule": "sqrt"}, "noise_schedule"),
    ],
)
def test_unimplemented_choices_raise(override: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_cascade_schedule(_block(**override), None)


def test_missing_block_raises() -> None:
    with pytest.raises(ValueError, match=r"training\.diffusion"):
        resolve_cascade_schedule(None, None)
