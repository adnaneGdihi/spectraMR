"""The certified Blackwell relations of a Fourier-diagonal catalogue.

For diagonal Fisher operators the Loewner order is the elementwise order of the profiles,
and by Theorem C1.1 that is Blackwell's order: ``a`` dominates ``b`` exactly when some
kernel that does not depend on the image turns ``a``'s data into ``b``'s. Elementwise order
is already transitive, so no closure step is needed. A pair is incomparable when each
experiment is strictly more informative at some coefficient. By Theorem C1.2 no quality
functional orders it without fixing a task.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch

from .catalogue import DiagonalExperiment


class Relation(StrEnum):
    DOMINATES = "dominates"
    DOMINATED = "dominated"
    EQUIVALENT = "equivalent"
    INCOMPARABLE = "incomparable"


def compare(a: torch.Tensor, b: torch.Tensor, *, atol: float = 1e-9) -> Relation:
    """Relation of profile ``a`` to profile ``b`` (both ``log n``, ``-inf`` allowed)."""
    if a.shape != b.shape:
        raise ValueError(f"profile shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}")
    same = (a == b) | ((a - b).abs() <= atol)  # a == b covers the -inf == -inf case
    a_ge = bool(((a > b) | same).all())
    b_ge = bool(((b > a) | same).all())
    if a_ge and b_ge:
        return Relation.EQUIVALENT
    if a_ge:
        return Relation.DOMINATES
    if b_ge:
        return Relation.DOMINATED
    return Relation.INCOMPARABLE


@dataclass(frozen=True)
class RelationGraph:
    """Index pairs into the catalogue. ``strict`` holds ``(better, worse)``.

    ``uncertified`` pairs could be shown neither comparable nor incomparable. The
    diagonal class has none. With coil overlap they occur (see ``multicoil``).
    """

    strict: tuple[tuple[int, int], ...]
    equivalent: tuple[tuple[int, int], ...]
    incomparable: tuple[tuple[int, int], ...]
    uncertified: tuple[tuple[int, int], ...] = ()

    def summary(self, experiments: Sequence[Any]) -> dict[str, object]:
        """Counts overall and the certified task-free fraction of cross-family pairs.

        Uncertified pairs stay in the denominator, so the fraction is a lower bound.
        """
        fam = [e.family for e in experiments]
        cross = Counter(
            kind
            for kind, pairs in (
                ("comparable", self.strict),
                ("comparable", self.equivalent),
                ("incomparable", self.incomparable),
                ("uncertified", self.uncertified),
            )
            for i, j in pairs
            if fam[i] != fam[j]
        )
        n_cross = sum(cross.values())
        return {
            "n_strict": len(self.strict),
            "n_equivalent": len(self.equivalent),
            "n_incomparable": len(self.incomparable),
            "n_uncertified": len(self.uncertified),
            "cross_family_pairs": n_cross,
            "cross_family_task_free_fraction": (
                cross["comparable"] / n_cross if n_cross else float("nan")
            ),
        }


def certify(experiments: Sequence[DiagonalExperiment], *, atol: float = 1e-9) -> RelationGraph:
    """Classify every unordered pair of the catalogue."""
    strict: list[tuple[int, int]] = []
    equivalent: list[tuple[int, int]] = []
    incomparable: list[tuple[int, int]] = []
    for i in range(len(experiments)):
        for j in range(i + 1, len(experiments)):
            rel = compare(experiments[i].log_info, experiments[j].log_info, atol=atol)
            if rel is Relation.DOMINATES:
                strict.append((i, j))
            elif rel is Relation.DOMINATED:
                strict.append((j, i))
            elif rel is Relation.EQUIVALENT:
                equivalent.append((i, j))
            else:
                incomparable.append((i, j))
    return RelationGraph(tuple(strict), tuple(equivalent), tuple(incomparable))


__all__ = ["Relation", "RelationGraph", "certify", "compare"]
