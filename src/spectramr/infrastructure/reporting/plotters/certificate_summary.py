"""Certificate summary figure (R1 / R2 / R4 / R5 + V.3 badge).

Plan: TODO/backlog_ulf_radiologist_acceptance_roadmap.md §V.3.

Renders a compact regulatory-package figure: a 2×3 grid showing the
empirical-coverage curve (R1), DKW slack against beta (R2), Hoeffding
recall lower-bound per pathology class (R4), the McAllester PAC-Bayes
bound (R5), the validation badge eligibility flag, and the per-gate
pass/fail strip.

Consumes the JSON artefacts emitted by ``ConformalCalibrationStrategy``
(under ``certification.<letter>.output_artefact``).

Uses the shared reporting style (:mod:`infrastructure.reporting.style`)
+ metadata stamper (:mod:`infrastructure.reporting.metadata`) — the
SSOT for project-wide figure aesthetics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.style import use_default_style, save_plot


def _safe_load(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def render_certificate_summary(
    out_path: str | Path,
    r1_path: str | Path | None = None,
    r2_path: str | Path | None = None,
    r4_path: str | Path | None = None,
    r5_path: str | Path | None = None,
    badge_path: str | Path | None = None,
    title: str = "Regulatory certificate summary",
    formats: tuple[str, ...] | None = None,
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path:
    """Render the 2×3 certificate-summary figure to PNG.

    Returns the output path. Missing artefacts produce empty panels with
    a "—" label so the figure still renders for partial certificates.
    """
    import matplotlib.pyplot as plt

    use_default_style()

    payloads = {
        "R1 — Conformal coverage": _safe_load(r1_path),
        "R2 — CHD slack": _safe_load(r2_path),
        "R4 — Pathology recall": _safe_load(r4_path),
        "R5 — PAC-Bayes bound": _safe_load(r5_path),
        "V.3 — Eligibility flag": _safe_load(badge_path),
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(title, fontsize=14)

    # R1 — empirical coverage
    ax = axes[0, 0]
    p = payloads["R1 — Conformal coverage"]
    if p:
        cov = p.get("empirical_coverage")
        target = 1.0 - p.get("alpha", 0.1)
        ax.bar(["empirical", "target"], [cov, target], color=["steelblue", "lightgray"])
        ax.set_ylim(0, 1.05)
        # 0.9 reference guide belongs on the panel that HAS coverage bars to
        # read against it. The old ``... if not p else None`` drew it only when
        # the panel was empty (R1 data missing) and omitted it exactly when it
        # was useful — an inverted conditional-expression statement.
        ax.axhline(0.9, ls="--", color="red", lw=0.8)
    ax.set_title("R1 — Conformal coverage")

    # R2 — DKW slack
    ax = axes[0, 1]
    p = payloads["R2 — CHD slack"]
    if p:
        ax.bar(
            ["dkw_slack", "beta"],
            [p.get("dkw_slack", 0), p.get("beta", 0.05)],
            color=["coral", "lightgray"],
        )
    ax.set_title("R2 — CHD slack vs beta")

    # R4 — per-class recall lower bound
    ax = axes[0, 2]
    p = payloads["R4 — Pathology recall"]
    if p:
        per_class = p.get("per_class", {})
        names = list(per_class.keys())
        bounds = [per_class[k].get("lower_bound", 0) for k in names]
        ax.bar(names, bounds, color="seagreen")
        ax.axhline(p.get("target_recall", 0.9), ls="--", color="red", lw=0.8)
        ax.set_ylim(0, 1.05)
    ax.set_title("R4 — Pathology recall LB")

    # R5 — PAC-Bayes upper bound
    ax = axes[1, 0]
    p = payloads["R5 — PAC-Bayes bound"]
    if p:
        ax.bar(
            ["empirical_loss", "pac_bayes_bound"],
            [p.get("empirical_loss", 0), p.get("pac_bayes_bound", 0)],
            color=["steelblue", "darkorange"],
        )
    ax.set_title("R5 — McAllester upper bound")

    # V.3 — gate pass/fail strip
    ax = axes[1, 1]
    p = payloads["V.3 — Eligibility flag"]
    if p:
        gates = p.get("gates", {})
        names = list(gates.keys())
        passed = [1 if gates[g].get("passed", False) else 0 for g in names]
        colours = ["seagreen" if x else "crimson" for x in passed]
        ax.bar(names, passed, color=colours)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["fail", "pass"])
        ax.tick_params(axis="x", rotation=45)
    ax.set_title("V.3 — Per-gate pass/fail")

    # Eligibility headline
    ax = axes[1, 2]
    ax.axis("off")
    if p:
        eligible = p.get("eligible", False)
        colour = "seagreen" if eligible else "crimson"
        ax.text(
            0.5,
            0.5,
            f"eligible = {eligible}",
            fontsize=24,
            ha="center",
            va="center",
            color=colour,
            fontweight="bold",
        )
    ax.set_title("Validation badge")

    fig.tight_layout()
    if metadata is not None:
        stamp_figure(fig, metadata)
    out = Path(out_path)
    return save_plot(fig, out, formats=formats, dpi=dpi, metadata=metadata)


__all__ = ["render_certificate_summary"]
