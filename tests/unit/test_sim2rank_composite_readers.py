"""Every reader of the published ``composite_rank`` treats lower as better."""

from __future__ import annotations

import math

_DATA = {
    "final_leaderboard": [
        {"category": "Pixel", "metrics": [
            {"key": "mae", "composite_rank": 4.0},
            {"key": "ssim", "composite_rank": 1.5},
            {"key": "psnr", "composite_rank": None},
        ]},
        {"category": "Edge", "metrics": [{"key": "hfen", "composite_rank": 2.0}]},
    ]
}  # fmt: skip


def test_clustering_spanning_set_picks_the_lowest_composite_rank() -> None:
    from scripts.generate_sim2rank_clustering import best_per_category, get_composite_rank

    assert best_per_category(_DATA, ["mae", "ssim", "psnr", "hfen"]) == {
        "Pixel": "ssim",
        "Edge": "hfen",
    }
    assert math.isinf(get_composite_rank("psnr", _DATA))


def test_extended_plots_annotate_the_best_ranked_metrics() -> None:
    from scripts.sim2rank_extended_plots import top_by_composite_rank

    metrics = ["mae", "ssim", "psnr", "hfen"]
    assert top_by_composite_rank(metrics, _DATA, 2) == [(1, 1.5), (3, 2.0)]
