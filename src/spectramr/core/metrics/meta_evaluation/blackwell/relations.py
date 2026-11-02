"""A catalogue that brings its own certified relations and its own displays.

The diagonal ranker derives relations from information profiles. Other catalogues
cannot: multi-coil operators are not diagonal, and a nuisance mixture's information
is not tractable. Each certifies its relations its own way (coil witnesses, unitary
absorption) and hands the ranker this bundle.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .evaluate import Realiser
from .order import RelationGraph


@dataclass(frozen=True)
class RelationCatalogue:
    """Experiments, their certified relations, and how to display them.

    ``realiser(n_draws, seed)`` builds the :data:`~.evaluate.Realiser` for one run (draw
    count and seed fix any per-draw nuisance or mask). ``noise_dims`` are the extra
    leading noise axes per draw (the coils).
    """

    experiments: Sequence[Any]
    graph: RelationGraph
    realiser: Callable[[int, int], Realiser]
    noise_dims: tuple[int, ...] = ()
    provenance: dict[str, object] = field(default_factory=dict)


__all__ = ["RelationCatalogue"]
