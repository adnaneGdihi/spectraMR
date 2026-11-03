r"""Cross-run comparison report (``spectramr report --compare``).

Arms are read at one checkpoint rule (:mod:`..views`), aligned by reference
content for the mosaic, and rendered through the plotter registry.
"""

from .arms import Arms, load_arms, resolve_runs, short_labels
from .figure_set import COMPARISON_FIGURES, comparison_figure_ids
from .report import COMPARISON_TABLES, generate_comparison

__all__ = [
    "COMPARISON_FIGURES",
    "COMPARISON_TABLES",
    "Arms",
    "comparison_figure_ids",
    "generate_comparison",
    "load_arms",
    "resolve_runs",
    "short_labels",
]
