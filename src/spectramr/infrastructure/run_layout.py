"""What inside a results tree is the arm's own output, and what merely lives there.

``experiments/results/<experiment>/`` is not homogeneous: alongside the arm's
real checkpoints, metrics and images, ``spectramr profile`` files a **throwaway**
run of that same arm under ``profile/<run_id>/run/``. That child is a genuine
training run — it writes the same markers, the same ``metrics/real_images``, the
same ``final_metrics.json`` — so nothing about its *contents* distinguishes it
from the arm's own. Only its location does.

Every consumer that walks a results tree therefore needs the same predicate, and
this module is its single owner (non-negotiable 17).

Its placement is forced twice over. It cannot live in any of its three consumers:
:mod:`~spectramr.infrastructure.reporting.batch` imports
:mod:`~spectramr.infrastructure.reporting.pipeline`, which imports
:mod:`~spectramr.infrastructure.reporting.cases.loader`, so a constant shared by
all three would close a cycle. And it cannot sit anywhere inside
``infrastructure/reporting/`` either, even as a leaf: that package's ``__init__``
eagerly imports the whole pipeline, so ``cli/profile_paths.py`` importing from
there pulls torch, scipy, pandas and matplotlib into the ``--help`` path and
breaks ``test_startup_budget.py`` — measured, not assumed. ``infrastructure/``
itself has a deliberately empty ``__init__``, and dependencies point inward
(non-negotiable 5), so ``cli/`` may import this while the reverse could not.
"""

from __future__ import annotations

from pathlib import Path

#: The results-tree segment under which ``spectramr profile`` files its runs, i.e.
#: ``<arm run dir>/PROFILE_SUBDIR/<run_id>/run/``. Named once here and read by both
#: the writer (``cli.profile_paths.resolve_profile_paths``) and every reader that
#: walks a results tree.
PROFILE_SUBDIR = "profile"

#: Every segment a reader must treat as profiling output: the current one, and the
#: ``profiles`` spelling that trees written before the rename still carry.
PROFILE_SUBDIRS: frozenset[str] = frozenset({PROFILE_SUBDIR, "profiles"})


def is_profiling_artifact(path: str | Path) -> bool:
    """True when ``path`` lies inside a profiling run's throwaway output.

    Matches on the path *segment*, never a substring: an experiment legitimately
    named ``profiles_ablation`` is not a profiling artifact, and a check written
    as ``"profiles" in str(path)`` would silently erase it from every report.
    """
    return not PROFILE_SUBDIRS.isdisjoint(Path(path).parts)


def drop_profiling_artifacts(paths: list[Path]) -> list[Path]:
    """Filter :func:`is_profiling_artifact` out of a discovered path list.

    Order-preserving, because callers that take ``[0]`` are choosing the
    shallowest match and must keep doing so after the filter.
    """
    return [p for p in paths if not is_profiling_artifact(p)]


__all__ = [
    "PROFILE_SUBDIR",
    "PROFILE_SUBDIRS",
    "drop_profiling_artifacts",
    "is_profiling_artifact",
]
