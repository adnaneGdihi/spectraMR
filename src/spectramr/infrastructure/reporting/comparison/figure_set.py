"""Which figures a comparison draws: the default set, every registered figure, or a list.

``report --compare`` draws :data:`COMPARISON_FIGURES` by default. A campaign report asks
for ``"all"``: every registered plotter is attempted, and one without the data it needs
soft-skips with its reason in the manifest. Figures in :data:`DRAWN_ELSEWHERE` take
inputs only their own block builds (phase tables, paired statistics), so they are drawn
there and never dispatched a second time from the generic loop.
"""

from __future__ import annotations

from collections.abc import Sequence

from .. import plotters
from ..plotters.differences import IDENTITY_FIGURES
from ..plotters.dynamics import FIGURES as DYNAMICS_FIGURES

#: Figure ids rendered by a comparison, in report order.
COMPARISON_FIGURES: tuple[str, ...] = (
    "cmp_metric_bars",
    "fig_1_2_learning_curves",
    "acceleration_sweep",
    "cmp_mosaic",
    "metric_distribution",
    "fig_3_2_subject_bars",
    "fig_3_3_metric_scatter",
    "cmp_paired_slice_scatter",
    "cmp_rank_heatmap",
    "fig_3_6_error_decomposition",
    "fig_3_7_metric_embedding",
    "fig_4_1_weight_trajectory",
    "fig_4_2_layer_drift",
    "fig_4_4_image_embedding",
    *DYNAMICS_FIGURES,
    *IDENTITY_FIGURES,
)

#: drawn by ``cmp_phase_metrics``'s own call and by ``all_metric_figures.cohort_figures``
DRAWN_ELSEWHERE: frozenset[str] = frozenset(
    {
        "cmp_phase_metrics",
        "cmp_coverage",
        "cmp_rank_reversal",
        "cmp_phase_bars",
        "cmp_group_bars",
        "cmp_effect_forest",
        "cmp_subject_lines",
        "cmp_effect_bars",
        "cmp_metric_group_bars",
        "cmp_metric_ladder",
    }
)


def comparison_figure_ids(figures: str | Sequence[str] | None) -> tuple[str, ...]:
    """``None`` -> the default set; ``"all"`` -> every registered figure; a list -> exactly it.

    An id the registry does not know raises: a typo must not quietly drop a figure.
    """
    if figures is None:
        return COMPARISON_FIGURES
    available = plotters.list_available()
    if figures == "all" or list(figures) == ["all"]:
        rest = [f for f in available if f not in COMPARISON_FIGURES and f not in DRAWN_ELSEWHERE]
        return (*COMPARISON_FIGURES, *rest)
    plotters.reject_retired(figures)
    unknown = sorted(set(figures) - set(available))
    if unknown:
        raise ValueError(f"unknown figure id(s) {unknown}; registered: {available}")
    return tuple(f for f in figures if f not in DRAWN_ELSEWHERE)


__all__ = ["COMPARISON_FIGURES", "DRAWN_ELSEWHERE", "comparison_figure_ids"]
