"""KSD defensibility certificate — one-page PDF (Idea 7 of audit_plan_novel.md).

Renders a compact regulatory-style page summarising the outcome of the
Tier-3 KSD audit:

- Outcome badge (PASS / FAIL).
- Reported KSD² value, bootstrap p-value, threshold.
- Number of samples and (optional) bandwidth.
- Free-text message from the audit hook.

The PDF is intended to be attached to a submission package alongside
the existing certificate-summary figure rendered by
``certificate_summary.py``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from spectramr.infrastructure.reporting.metadata import (
    RunMetadata,
    stamp_figure,
    write_sidecar,
)
from spectramr.infrastructure.reporting.style import use_default_style, save_plot

logger = logging.getLogger(__name__)


def render_ksd_certificate(
    out_path: str | Path,
    *,
    result: Any,
    config_path: str | Path | None = None,
    samples_path: str | Path | None = None,
    title: str = "KSD defensibility certificate",
    formats: tuple[str, ...] | None = None,
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path:
    """Render a one-page PDF certificate for a KSD audit outcome.

    Args:
        out_path: Destination PDF.
        result: :class:`KSDDefensibilityResult` (or any object with the
            same field names: ``passed``, ``ksd_squared``, ``p_value``,
            ``threshold``, ``n_samples``, ``bandwidth``, ``message``).
        config_path: Optional experiment config the audit ran against.
        samples_path: Optional samples file path.
        title: Page title.
        metadata: Provenance stamp.

    Returns:
        The output path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    use_default_style()

    fig = plt.figure(figsize=(8.5, 5.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")

    passed = bool(getattr(result, "passed", False))
    badge_color = "#1f8b3a" if passed else "#b22222"
    badge_text = "PASS" if passed else "FAIL"

    # Title bar and outcome badge.
    ax.text(
        0.02,
        0.96,
        title,
        transform=ax.transAxes,
        fontsize=13,
        weight="bold",
        va="top",
    )
    ax.text(
        0.98,
        0.96,
        badge_text,
        transform=ax.transAxes,
        fontsize=18,
        weight="bold",
        color="white",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=badge_color, edgecolor="none"),
    )

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    ax.text(
        0.02,
        0.91,
        f"Generated: {timestamp}",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
        va="top",
    )

    # Body table.
    rows = [
        ("KSD²", f"{float(getattr(result, 'ksd_squared', 0.0)):.4e}"),
        ("Bootstrap p-value", f"{float(getattr(result, 'p_value', 0.0)):.4f}"),
        ("Threshold", f"{float(getattr(result, 'threshold', 0.05)):.4f}"),
        ("Samples", str(int(getattr(result, "n_samples", 0)))),
        (
            "Kernel bandwidth",
            f"{float(getattr(result, 'bandwidth', 0.0) or 0.0):.4f} (median heuristic)"
            if getattr(result, "bandwidth", None) is None
            else f"{float(result.bandwidth):.4f}",
        ),
    ]
    if config_path is not None:
        rows.append(("Experiment config", str(config_path)))
    if samples_path is not None:
        rows.append(("Samples source", str(samples_path)))

    y = 0.80
    line_h = 0.045
    for k, v in rows:
        ax.text(
            0.04,
            y,
            k,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            weight="bold",
        )
        ax.text(
            0.32,
            y,
            v,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            family="monospace",
        )
        y -= line_h

    # Free-text message.
    msg = str(getattr(result, "message", ""))
    if msg:
        ax.text(
            0.04,
            y - 0.03,
            "Audit message:",
            transform=ax.transAxes,
            fontsize=10,
            weight="bold",
            va="top",
        )
        ax.text(
            0.04,
            y - 0.08,
            msg,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            wrap=True,
            family="monospace",
        )

    # Footer.
    ax.text(
        0.5,
        0.04,
        (
            "Kernelised Stein Discrepancy goodness-of-fit test "
            "(Liu, Lee & Jordan, ICML 2016) with wild bootstrap "
            "(Chwialkowski et al., ICML 2016). Idea 7 of audit_plan_novel.md."
        ),
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
        ha="center",
        va="bottom",
        wrap=True,
    )

    if metadata is not None:
        stamp_figure(fig, metadata)
    fig.tight_layout()
    return save_plot(fig, out_path, formats=formats, dpi=dpi, metadata=metadata)


__all__ = ["render_ksd_certificate"]
