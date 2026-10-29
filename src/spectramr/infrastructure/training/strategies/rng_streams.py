"""Per-rank CPU random streams that survive a checkpoint resume.

Several strategies (SSDU split and noise, the EI group draw, the ambient
diffusion split, the physics-equivariant SSL operator draw) own a
``torch.Generator`` seeded from the config. Seeded identically on every rank,
each rank draws the same split or group element at every step, so data
parallelism divides the stochastic diversity by the world size; never saved,
a requeued run replays the sequence from step 0. These helpers seed the stream
per rank and carry its state through the base ``strategy_state_dict``, which
saves the attributes a strategy names in ``checkpointed_streams``.

The checkpoint holds rank 0's state only (rank 0 is the only process that
reaches the save), the same constraint the global ``rng_state`` restore in
``checkpointing.resume`` is rank-gated on. Rank 0 therefore resumes its
saved stream exactly, and every other rank reseeds from a draw of that saved
state offset by its rank: a deterministic continuation, distinct across ranks,
that neither replays the rank's original sequence nor copies rank 0's.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

from spectramr.core.env import distributed_rank
from spectramr.core.parallel_groups import current_dp_layout

__all__ = ["generator_states", "rank_seeded_generator", "restore_generator_states", "stream_rank"]

_SEED_SPAN = 1 << 62


def stream_rank() -> int:
    """This process's data-parallel rank, ``0`` outside a distributed launch.

    Ranks that share one data replica (tensor, context, spatial or pipeline peers)
    must draw the same timesteps and masks, so a non-trivial mesh keys the stream by
    data-parallel rank. Without an active mesh it is the global rank, read from the
    launcher's environment so it is correct before ``init_process_group``.
    """
    layout = current_dp_layout()
    return layout.rank if layout is not None else (distributed_rank() or 0)


def rank_seeded_generator(seed: int) -> torch.Generator:
    """A CPU generator seeded ``seed + rank``; single-process runs keep ``seed``."""
    return torch.Generator(device="cpu").manual_seed(int(seed) + stream_rank())


def generator_states(owner: Any, names: Iterable[str]) -> dict[str, torch.Tensor]:
    """``{name: state}`` for the named generator attributes of ``owner``."""
    return {name: getattr(owner, name).get_state() for name in names}


def restore_generator_states(owner: Any, states: Mapping[str, torch.Tensor] | None) -> None:
    """Restore the generators captured by :func:`generator_states` (rank 0's).

    A missing or empty mapping (a checkpoint written before the streams were
    saved) leaves the freshly seeded generators untouched.
    """
    if not states:
        return
    rank = stream_rank()
    for name, state in states.items():
        generator = getattr(owner, name, None)
        if not isinstance(generator, torch.Generator):
            raise ValueError(
                f"checkpoint carries a random stream {name!r} that "
                f"{type(owner).__name__} does not own."
            )
        # set_state takes a CPU ByteTensor; a CUDA map_location moves it.
        saved = state.detach().to("cpu", torch.uint8)
        if rank == 0:
            generator.set_state(saved)
            continue
        source = torch.Generator(device="cpu")
        source.set_state(saved)
        base = int(torch.randint(0, _SEED_SPAN, (1,), generator=source))
        generator.manual_seed((base + rank) % _SEED_SPAN)
