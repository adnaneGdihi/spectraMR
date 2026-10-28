"""Unit tests for :mod:`spectramr.infrastructure.training.strategies.n2n_strategy`.

The strategy runs the shared reconstruction step and owns only how the training
target is formed from the loader's repetition stack (``data.n2n_pairing``). The
tests pin that the target is never the input's own slot, that each pairing reads
the slots it names, and that a run without a stack refuses to start.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.config.schemas.enums import Task
from spectramr.infrastructure.training.strategies.n2n_strategy import (
    NoiseToNoiseStrategy,
    pair_from_stack,
)
from spectramr.infrastructure.training.strategies.reconstruction import ReconstructionStepMixin

B, C, H, W = 4, 8, 16, 16


def _stack(n_slots: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Slot s is filled with the value s, so a target names its slot exactly."""
    slots = torch.stack([torch.full((B, C, H, W, 1), float(s)) for s in range(n_slots)], dim=1)
    return slots[:, 0], slots.reshape(B, n_slots * C, H, W, 1)


def _bare(pairing: str | None) -> NoiseToNoiseStrategy:
    strategy = NoiseToNoiseStrategy.__new__(NoiseToNoiseStrategy)
    strategy.config = SimpleNamespace(data=SimpleNamespace(n2n_pairing=pairing))
    strategy.logging_service = None
    return strategy


def test_ordered_takes_the_second_half():
    x, stack = _stack(2)
    assert torch.equal(pair_from_stack(x, stack, "ordered"), torch.ones_like(x))


def test_one_vs_rest_mean_averages_the_other_slots():
    x, stack = _stack(3)
    assert torch.equal(pair_from_stack(x, stack, "one_vs_rest_mean"), torch.full_like(x, 1.5))


def test_random_pair_draws_every_other_slot_and_never_the_input():
    x, stack = _stack(3)
    g = torch.Generator().manual_seed(0)
    seen = set()
    for _ in range(50):
        target = pair_from_stack(x, stack, "random_pair", generator=g)
        values = target.flatten(1)[:, 0]
        # Each sample's target is one whole slot, drawn per sample.
        assert torch.equal(target, values.view(B, 1, 1, 1, 1).expand_as(target))
        seen |= set(values.tolist())
    assert seen == {1.0, 2.0}


@pytest.mark.parametrize(
    ("n_slots", "pairing", "match"),
    [
        (1, "random_pair", "at least two"),
        (3, "ordered", "two halves"),
        (2, "bogus", "Unknown n2n pairing"),
    ],
)
def test_pairing_refuses_a_stack_it_cannot_pair(n_slots, pairing, match):
    x, stack = _stack(n_slots)
    with pytest.raises(ValueError, match=match):
        pair_from_stack(x, stack, pairing)


def test_a_stack_that_does_not_tile_the_input_raises():
    x, stack = _stack(2)
    with pytest.raises(ValueError, match="not a multiple"):
        pair_from_stack(x, stack[:, :-1], "ordered")


def test_strategy_composes_the_shared_step_and_declares_denoising():
    assert ReconstructionStepMixin in NoiseToNoiseStrategy.__mro__
    assert NoiseToNoiseStrategy.capabilities.tasks == frozenset({Task.DENOISING})


@pytest.mark.parametrize("pairing", [None, "rep_pair"])
def test_strategy_refuses_to_start_without_a_pairing(pairing):
    with pytest.raises(ValueError, match=r"data\.n2n_pairing"):
        _bare(pairing)._setup_strategy_specific_components()


def test_unpack_forms_the_target_from_the_stack():
    strategy = _bare("ordered")
    strategy._setup_strategy_specific_components()
    x, stack = _stack(2)
    inp, target = strategy._unpack_batch({"input": {"data": x}, "kspace_reps": {"data": stack}})
    assert inp is x
    assert torch.equal(target, torch.ones_like(x))


def test_unpack_without_a_stack_reads_the_served_target():
    """The validation split serves input/target, never a stack."""
    strategy = _bare("random_pair")
    strategy._setup_strategy_specific_components()
    x, y = torch.zeros(B, C, H, W), torch.ones(B, C, H, W)
    inp, target = strategy._unpack_batch({"input": {"data": x}, "target": {"data": y}})
    assert inp is x and target is y


def test_loop_batch_round_trips_the_stack_through_the_adapter():
    """The loop runs BatchAdapter.from_dict before the strategy sees the batch."""
    from spectramr.data.batch_types import BatchAdapter

    strategy = _bare("ordered")
    strategy._setup_strategy_specific_components()
    x, stack = _stack(2)
    batch = BatchAdapter.from_dict({"input": {"data": x}, "kspace_reps": {"data": stack}})
    assert batch.target is None
    inp, target = strategy._unpack_batch(batch.to("cpu"))
    assert torch.equal(inp, x)
    assert torch.equal(target, torch.ones_like(x))
