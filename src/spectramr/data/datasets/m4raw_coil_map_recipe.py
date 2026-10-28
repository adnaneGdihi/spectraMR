"""Which repetitions of an M4Raw group its coil maps are estimated from.

``all_reps_mean`` (the default) is the phase-aligned mean of every repetition,
the lowest-noise estimate. ``input_repetition`` is the repetition the dataset
serves as ``input`` alone: an arm graded against the leave-one-out mean of the
other repetitions must not be handed maps estimated from that reference.

A map file written before the recipe was recorded carries none and reads as
``all_reps_mean``, the only recipe that existed then, so ``MAP_FORMAT_VERSION``
did not change.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

__all__ = ["DEFAULT_RECIPE", "VALID_RECIPES", "check_recipe", "recipe_source"]

DEFAULT_RECIPE: str = "all_reps_mean"


def _all_reps_mean(kspace_reps: Sequence[torch.Tensor]) -> torch.Tensor:
    from spectramr.data.datasets.m4raw_dataset import _average_reps

    return _average_reps(list(kspace_reps), "phase_aligned_mean")


def _input_repetition(kspace_reps: Sequence[torch.Tensor]) -> torch.Tensor:
    # The dataset serves its first sorted repetition as `input`, and the
    # precompute groups files with the same sort, so index 0 is that file.
    return kspace_reps[0]


_RECIPE_SOURCES: dict[str, Callable[[Sequence[torch.Tensor]], torch.Tensor]] = {
    "all_reps_mean": _all_reps_mean,
    "input_repetition": _input_repetition,
}

#: The recipes a consumer may declare (``data.coil_maps_recipe``).
VALID_RECIPES: frozenset[str] = frozenset(_RECIPE_SOURCES)


def check_recipe(recipe: str) -> str:
    """``recipe`` itself, or a ``ValueError`` naming the valid set."""
    if recipe not in VALID_RECIPES:
        raise ValueError(f"unknown coil-map recipe {recipe!r}; valid: {sorted(VALID_RECIPES)}.")
    return recipe


def recipe_source(recipe: str) -> Callable[[Sequence[torch.Tensor]], torch.Tensor]:
    """The function that reduces a group's repetitions to the k-space ESPIRiT reads."""
    return _RECIPE_SOURCES[check_recipe(recipe)]
