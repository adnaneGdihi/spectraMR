"""The campaign report: the cross-arm comparison over a campaign's finished runs, then its site.

``campaign evaluate`` and ``campaign watch`` call this, so a campaign is reported by the same
figures and pages as ``spectramr report --compare`` + ``--dashboard``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .runs import _run_dir
from .site import build_site

logger = logging.getLogger(__name__)


def build_campaign_report(
    campaign_dir: str | Path, *, figures: str | list[str] = "all"
) -> dict[str, Path]:
    """Report every run on disk, compare them into ``comparison/``, and render the dashboard.

    ``figures="all"`` (the default) attempts every registered figure, per run and in the
    comparison; one without the data it needs is skipped with its reason in the manifest.
    The baseline is the run whose campaign role is ``baseline``. With fewer than two
    finished runs there is nothing to compare, and the runs are reported on their own.
    """
    from ..batch import discover_run_dirs
    from ..comparison import generate_comparison
    from ..comparison.arms import short_labels
    from ..pipeline import generate_report

    root = Path(campaign_dir)
    state = json.loads((root / "campaign_state.json").read_text())
    # The runs that finished on disk, not the state's statuses: a downloaded state is often
    # stale (a resubmission, a status never updated). The state only names the baseline.
    runs = [d for d in discover_run_dirs(root) if d.parent == root]
    by_dir = {d: e for e in state.get("experiments", []) if (d := _run_dir(root, e)) is not None}
    found = [(by_dir.get(d, {}), d) for d in runs]
    wanted = [figures] if isinstance(figures, str) else list(figures)
    for _, run in found:
        try:
            generate_report(run, method_name=run.name, figures=wanted)
        except Exception as exc:  # one run's failure must not cost the others their report
            logger.warning("campaign report: %s failed: %s", run, exc)
    if len(found) >= 2:
        labels = short_labels([d.name for _, d in found])
        baseline = next(
            (lab for lab, (e, _) in zip(labels, found, strict=True) if e.get("role") == "baseline"),
            None,
        )
        # "all" also means every metric, so the cohort figures (effect forest, ladder, phase
        # panels) that only draw for a metric selection are drawn too.
        generate_comparison(
            [d for _, d in found],
            root / "comparison",
            labels=labels,
            baseline=baseline,
            figures=figures,
            metrics="all" if figures == "all" else None,
        )
    else:
        logger.info("campaign report: %d finished run(s) under %s; no comparison", len(found), root)
    return build_site(root)
