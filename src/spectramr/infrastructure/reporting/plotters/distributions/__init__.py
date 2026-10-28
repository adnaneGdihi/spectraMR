"""Distribution, bivariate, embedding and background figures built on per-slice rows.

``FIGURES`` maps a figure id to its ``make``; :mod:`..__init__` registers every entry.
"""

from . import (
    background_zoom,
    embedding,
    error_decomposition,
    intensity_joint,
    metric_scatter,
    paired_scatter,
    rank_heatmap,
    subject_bars,
    violin,
)

#: figures fed the per-slice frame / the image cases by the pipeline and the comparison
SLICE_FIGURES = (
    "fig_3_1_metric_violin",
    "fig_3_2_subject_bars",
    "fig_3_3_metric_scatter",
    "cmp_paired_slice_scatter",
    "cmp_rank_heatmap",
    "fig_3_6_error_decomposition",
    "fig_3_7_metric_embedding",
)
CASE_FIGURES = ("fig_3_8_intensity_joint", "fig_3_9_background_zoom")

FIGURES = {
    "fig_3_1_metric_violin": violin.make,
    "fig_3_2_subject_bars": subject_bars.make,
    "fig_3_3_metric_scatter": metric_scatter.make,
    "cmp_paired_slice_scatter": paired_scatter.make,
    "cmp_rank_heatmap": rank_heatmap.make,
    "fig_3_6_error_decomposition": error_decomposition.make,
    "fig_3_7_metric_embedding": embedding.make,
    "fig_3_8_intensity_joint": intensity_joint.make,
    "fig_3_9_background_zoom": background_zoom.make,
}
