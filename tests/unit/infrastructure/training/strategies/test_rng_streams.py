"""Per-rank, checkpointed random streams for the self-supervised strategies (#2762)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.training.strategies import rng_streams
from spectramr.infrastructure.training.strategies.rng_streams import (
    generator_states,
    rank_seeded_generator,
    restore_generator_states,
)


def _on_rank(monkeypatch: pytest.MonkeyPatch, rank: int | None) -> None:
    monkeypatch.setattr(rng_streams, "distributed_rank", lambda: rank)


def _draws(generator: torch.Generator, n: int = 8) -> torch.Tensor:
    return torch.randint(0, 1 << 30, (n,), generator=generator)


def test_a_single_process_run_keeps_the_declared_seed(monkeypatch) -> None:
    _on_rank(monkeypatch, None)
    reference = torch.Generator().manual_seed(7)
    assert torch.equal(_draws(rank_seeded_generator(7)), _draws(reference))


def test_two_ranks_draw_different_streams(monkeypatch) -> None:
    _on_rank(monkeypatch, 0)
    rank0 = _draws(rank_seeded_generator(7))
    _on_rank(monkeypatch, 1)
    rank1 = _draws(rank_seeded_generator(7))
    assert not torch.equal(rank0, rank1)


def test_rank_zero_resumes_its_saved_stream_exactly(monkeypatch) -> None:
    _on_rank(monkeypatch, 0)
    saved = SimpleNamespace(_rng=rank_seeded_generator(3))
    _draws(saved._rng)  # advance past the start
    states = generator_states(saved, ("_rng",))
    expected = _draws(saved._rng)

    resumed = SimpleNamespace(_rng=rank_seeded_generator(3))
    restore_generator_states(resumed, states)
    assert torch.equal(_draws(resumed._rng), expected)


def test_another_rank_continues_neither_replaying_nor_copying(monkeypatch) -> None:
    _on_rank(monkeypatch, 0)
    saved = SimpleNamespace(_rng=rank_seeded_generator(3))
    _draws(saved._rng)
    states = generator_states(saved, ("_rng",))
    rank0_next = _draws(saved._rng)

    _on_rank(monkeypatch, 1)
    replay = _draws(rank_seeded_generator(3))
    first = SimpleNamespace(_rng=rank_seeded_generator(3))
    restore_generator_states(first, states)
    second = SimpleNamespace(_rng=rank_seeded_generator(3))
    restore_generator_states(second, states)
    continued = _draws(first._rng)
    assert not torch.equal(continued, rank0_next)
    assert not torch.equal(continued, replay)
    assert torch.equal(continued, _draws(second._rng)), "the continuation is deterministic"


def test_an_old_checkpoint_leaves_the_fresh_stream(monkeypatch) -> None:
    _on_rank(monkeypatch, 0)
    owner = SimpleNamespace(_rng=rank_seeded_generator(5))
    restore_generator_states(owner, None)
    restore_generator_states(owner, {})
    assert torch.equal(_draws(owner._rng), _draws(torch.Generator().manual_seed(5)))


def test_a_stream_the_owner_lacks_is_refused(monkeypatch) -> None:
    _on_rank(monkeypatch, 0)
    states = {"_other": torch.Generator().get_state()}
    with pytest.raises(ValueError, match="_other"):
        restore_generator_states(SimpleNamespace(_rng=torch.Generator()), states)


def test_model_parallel_peers_share_their_replicas_stream(monkeypatch) -> None:
    """Under a mesh, global ranks 0 and 1 (tensor peers of dp replica 0) draw alike."""
    from spectramr.core.parallel_groups import DPLayout, activate_dp_layout, reset_dp_layout

    draws = []
    try:
        for global_rank, dp in ((0, 0), (1, 0), (2, 1)):
            _on_rank(monkeypatch, global_rank)
            activate_dp_layout(
                DPLayout(
                    rank=dp,
                    world_size=2,
                    group=None,
                    src_rank=0,
                    writer=True,
                    dim_names=("dp_shard", "tp"),
                    shape=(2, 2),
                )
            )
            draws.append(_draws(rank_seeded_generator(7)))
    finally:
        reset_dp_layout()
    assert torch.equal(draws[0], draws[1]), "tensor peers of one replica must draw alike"
    assert not torch.equal(draws[0], draws[2]), "data replicas must still differ"
