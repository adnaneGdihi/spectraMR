"""``m4raw_coil_map_recipe``: which repetitions a group's coil maps come from.

Pinned: the closed set raises on anything else, and ``input_repetition`` reads
the first repetition only, so the maps of an arm graded against the
leave-one-out mean of the others do not depend on that reference.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.data.datasets.m4raw_coil_map_recipe import (
    DEFAULT_RECIPE,
    VALID_RECIPES,
    check_recipe,
    recipe_source,
)


def _reps(seed: int) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    shape = (1, 4, 12, 20)  # non-square: H != W
    return [torch.randn(shape, dtype=torch.complex64, generator=gen) for _ in range(3)]


def test_the_default_is_the_legacy_recipe() -> None:
    assert DEFAULT_RECIPE == "all_reps_mean"
    assert {"all_reps_mean", "input_repetition"} == VALID_RECIPES


def test_the_schema_and_the_dataset_offer_exactly_these_recipes() -> None:
    """``config/`` cannot import ``data/``, so the schema's Literal is a copy; pin it."""
    import inspect
    from typing import get_args

    from spectramr.config.schemas.data import DataConfigSchema
    from spectramr.data.datasets.m4raw_dataset import M4RawRepetitionDataset

    field = DataConfigSchema.model_fields["coil_maps_recipe"]
    assert set(get_args(field.annotation)) == VALID_RECIPES
    assert field.default == DEFAULT_RECIPE
    signature = inspect.signature(M4RawRepetitionDataset.__init__)
    assert signature.parameters["coil_maps_recipe"].default == DEFAULT_RECIPE


def test_an_unknown_recipe_raises() -> None:
    with pytest.raises(ValueError, match="unknown coil-map recipe 'median'"):
        check_recipe("median")
    with pytest.raises(ValueError, match="valid"):
        recipe_source("median")


def test_input_repetition_is_the_first_repetition_alone() -> None:
    reps = _reps(0)
    other = [reps[0], *_reps(1)[1:]]
    assert recipe_source("input_repetition")(reps) is reps[0]
    assert torch.equal(recipe_source("input_repetition")(other), reps[0])


def test_the_mean_reads_every_repetition() -> None:
    """Planted contrast: the default recipe moves when a non-input repetition does."""
    reps = _reps(0)
    other = [reps[0], *_reps(1)[1:]]
    mean = recipe_source("all_reps_mean")
    assert not torch.allclose(mean(reps), mean(other))
