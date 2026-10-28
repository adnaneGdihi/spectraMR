"""Plotting-SSOT ratchet guard.

Figures must be produced ONLY inside ``src/spectramr/infrastructure/reporting/``
(the plotting SSOT), with two sanctioned second surfaces — the sim2rank
meta-evaluation figures and dataset EDA — plus a frozen baseline of pre-existing
emitters kept out of scope by the report-only consolidation. The test fails on any
NEW figure-emitting call added elsewhere under ``src/spectramr/`` so plotting cannot
drift back out of the reporting pipeline (the ``no_new_*`` ratchet pattern).
"""

from __future__ import annotations

import re
from pathlib import Path

# Figure-emitting call sites forbidden outside the reporting SSOT. Require a
# trailing ``(`` so a docstring mentioning "savefig" is not a false positive.
_EMIT = re.compile(r"\.(?:savefig|write_html|write_image)\s*\(|\bplt\.show\s*\(")

_SRC = Path("src/spectramr")
_REPORTING = _SRC / "infrastructure" / "reporting"

# Permanent exceptions: sim2rank meta-evaluation and dataset EDA are sanctioned
# second surfaces (their own ``spectramr meta`` / EDA paths, never ``spectramr report``).
_ALLOWED_DIRS = (
    _SRC / "core" / "metrics" / "meta_evaluation",
    _SRC / "data" / "eda",
)

# Frozen baseline: pre-existing emitters left out of scope by the report-only
# consolidation. New violations must NOT be added; these are migration candidates.
_BASELINE: dict[str, str] = {
    "infrastructure/orchestration/campaign_report_generator.py":
        "campaign dashboards (fold-in candidate)",
    "models/analysis/operator_id_report.py":
        "operator-id figures in models/ (fold-in candidate)",
    "models/diffusion/noise_schedules.py":
        "noise-schedule viz incl. plt.show (extract candidate)",
    "application/use_cases/nr_validation/report.py":
        "NR-validation report (sim2rank orbit)",
    "infrastructure/validation/phantom_builder.py":
        "Tier-2 forward-probe debug snapshot",
}


def _is_allowed(path: Path) -> bool:
    if _REPORTING in path.parents:
        return True
    if any(d in path.parents for d in _ALLOWED_DIRS):
        return True
    try:
        rel = path.relative_to(_SRC).as_posix()
    except ValueError:
        return True
    return rel in _BASELINE


def test_no_new_plotting_outside_reporting_ssot() -> None:
    offenders: list[str] = []
    for py in _SRC.rglob("*.py"):
        if _is_allowed(py):
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        if _EMIT.search(text):
            offenders.append(py.relative_to(_SRC).as_posix())
    assert not offenders, (
        "figure-emitting call(s) found OUTSIDE the reporting SSOT — route the figure "
        "through infrastructure/reporting/plotters (register it), or, only if it is a "
        "genuinely sanctioned second surface, add it to the allowlist in this test: "
        f"{sorted(offenders)}"
    )


def test_baseline_entries_still_exist() -> None:
    # Prune a baseline entry once its emitter is migrated/removed so the ratchet
    # tightens — a stale exception would silently permit a regression.
    stale = sorted(rel for rel in _BASELINE if not (_SRC / rel).exists())
    assert not stale, f"stale SSOT-guard baseline entries (remove them): {stale}"
