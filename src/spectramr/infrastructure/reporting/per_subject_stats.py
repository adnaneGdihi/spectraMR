r"""Paired per-subject comparison of two arms from their per-slice rows.

Two arms are paired slice by slice on :data:`~.per_slice.SLICE_KEY`, each
arm's matched slices are averaged within a subject, and the tests run on the
subject means, because slices of one subject are not independent samples. The
statistics are the ones in :class:`~spectramr.core.metrics.statistical_tests.StatisticalTests`;
this module only decides what is paired with what.

A reciprocal metric (``nex_equivalent``) is compared on ``1/N_eq``, where lower
is better; the result carries ``scale`` so a reader does not misread the sign.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from spectramr.core.metrics.statistical_tests import StatisticalTests

from .per_slice import SLICE_KEY, SlicePass, is_reciprocal, slice_values


@dataclass(frozen=True)
class PairedSubjects:
    """Two arms' values on the slices both scored, reduced to subjects."""

    table: pd.DataFrame
    n_slices: int
    nonfinite_dropped: dict[str, int]
    unpaired_slices: dict[str, int]
    scale: str

    @property
    def n_subjects(self) -> int:
        return len(self.table)

    def describe(self) -> dict[str, Any]:
        """What the comparison was run on, for the report beside its numbers."""
        return {
            "source": "per_slice_rows",
            "unit": "subjects",
            "n_subjects": self.n_subjects,
            "n_slices_paired": self.n_slices,
            "nonfinite_dropped": dict(self.nonfinite_dropped),
            "unpaired_slices": dict(self.unpaired_slices),
            "scale": self.scale,
        }


def pair_on_slices(a: SlicePass, b: SlicePass, column_a: str, column_b: str) -> PairedSubjects:
    """Join two arms on slice identity, then average each subject's matched slices.

    Raises:
        ValueError: both arms scored slices and not one identity matches.
    """
    va, dropped_a = slice_values(a, column_a)
    vb, dropped_b = slice_values(b, column_b)
    joined = pd.concat({"a": va, "b": vb}, axis=1, join="inner")
    if joined.empty and len(va) and len(vb):
        # Both arms scored slices and none match: the identities disagree (a
        # loader that keys a field differently per arm), not a small overlap.
        raise ValueError(
            f"no slice of {a.files[0]} matches one of {b.files[0]} on {SLICE_KEY}; "
            f"e.g. {va.index[0]} against {vb.index[0]} (#2478)"
        )
    table = joined.groupby(level="subject_id").mean()
    return PairedSubjects(
        table=table,
        n_slices=len(joined),
        nonfinite_dropped={"a": dropped_a, "b": dropped_b},
        unpaired_slices={"a": len(va) - len(joined), "b": len(vb) - len(joined)},
        scale="1/N_eq (lower is better)" if is_reciprocal(column_a) else "metric",
    )


def compare_subjects(
    paired: PairedSubjects,
    *,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> dict[str, dict[str, Any]]:
    """Paired t-test, Wilcoxon, Cohen's d and a subject-resampling bootstrap CI.

    The bootstrap resamples subject-level differences, so the interval is for
    the mean over subjects, the same estimand the t-test tests.

    Raises:
        ValueError: fewer than two subjects were scored by both arms.
    """
    if paired.n_subjects < 2:
        raise ValueError(
            f"{paired.n_subjects} subject(s) scored by both arms; a paired test needs two"
        )
    va = paired.table["a"].to_numpy(dtype=np.float64)
    vb = paired.table["b"].to_numpy(dtype=np.float64)
    return {
        "paired_ttest": StatisticalTests.paired_ttest(va, vb),
        "wilcoxon": StatisticalTests.wilcoxon_signed_rank(va, vb),
        "effect_size": StatisticalTests.cohens_d(va, vb, paired=True),
        "bootstrap_ci": StatisticalTests.bootstrap_ci(
            va, vb, n_bootstrap=n_bootstrap, confidence=confidence
        ),
    }


__all__ = ["PairedSubjects", "compare_subjects", "pair_on_slices"]
