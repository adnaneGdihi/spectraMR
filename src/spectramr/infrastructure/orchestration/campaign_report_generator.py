"""Campaign Report Generator — HTML dashboards and comparison plots.

Cross-experiment campaign dashboards; paper comparison figures are in
``infrastructure.reporting.comparison``."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from spectramr.infrastructure.orchestration.campaign_evaluator import (
    CampaignReport,
    UncertaintyReport,
)

logger = logging.getLogger(__name__)
__all__ = ["CampaignReportGenerator"]


class CampaignReportGenerator:
    """Generate HTML dashboard and plots for cross-experiment comparison."""

    def __init__(self, output_dir: str | Path, figsize: tuple[int, int] = (12, 7), dpi: int = 120):
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi
        sns.set_style("whitegrid")

    def generate(self, report: CampaignReport) -> str:
        """Generate all plots and HTML dashboard.

        Returns path to dashboard HTML.
        """
        plot_paths: list[tuple[str, str]] = []  # (title, relative_path)

        if report.leaderboard is not None and not report.leaderboard.empty:
            p = self._plot_leaderboard_bars(report.leaderboard)
            if p:
                plot_paths.append(("Leaderboard", p))
            p = self._plot_metric_distributions(report.leaderboard)
            if p:
                plot_paths.append(("Metric Distributions", p))

        for metric_name, sig_df in report.significance_matrix.items():
            p = self._plot_significance_heatmap(sig_df, metric_name)
            if p:
                plot_paths.append((f"Significance — {metric_name.upper()}", p))

        if report.pairwise_results:
            p = self._plot_effect_sizes(report.pairwise_results)
            if p:
                plot_paths.append(("Effect Sizes (Cohen's d)", p))

        if report.uncertainty_reports:
            calibrated = [ur for ur in report.uncertainty_reports if ur.ece is not None]
            if calibrated:
                p = self._plot_uncertainty_calibration(calibrated)
                if p:
                    plot_paths.append(("Uncertainty Calibration", p))

        html_path = self._generate_html(report, plot_paths)

        report_json = self.output_dir / "evaluation_report.json"
        report.save(report_json)

        logger.info(f"Dashboard: {html_path}")
        return str(html_path)

    # ── Plot: Leaderboard bars ───────────────────────────────────

    def _plot_leaderboard_bars(self, lb: pd.DataFrame) -> str | None:
        metrics = lb["metric"].unique()
        n_metrics = len(metrics)
        if n_metrics == 0:
            return None

        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6), squeeze=False)
        lower_better = {"lpips", "mse", "nmse", "loss", "mae"}

        for idx, metric in enumerate(metrics):
            ax = axes[0, idx]
            mdf = lb[lb["metric"] == metric].sort_values(
                "mean", ascending=metric.lower() in lower_better
            )
            colors = sns.color_palette("viridis", n_colors=len(mdf))
            bars = ax.barh(
                mdf["experiment"],
                mdf["mean"],
                xerr=mdf["std"],
                color=colors,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_xlabel(metric.upper())
            ax.set_title(metric.upper())
            for bar, (_, row) in zip(bars, mdf.iterrows(), strict=False):
                ax.text(
                    bar.get_width() + row["std"] * 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{row['mean']:.3f}",
                    va="center",
                    fontsize=8,
                )

        plt.tight_layout()
        path = self.plots_dir / "leaderboard.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path.relative_to(self.output_dir))

    # ── Plot: Metric distributions ───────────────────────────────

    def _plot_metric_distributions(self, lb: pd.DataFrame) -> str | None:
        metrics = lb["metric"].unique()
        if len(metrics) == 0:
            return None

        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5), squeeze=False)
        for idx, metric in enumerate(metrics):
            ax = axes[0, idx]
            mdf = lb[lb["metric"] == metric]
            ax.bar(
                range(len(mdf)),
                mdf["mean"],
                yerr=mdf["std"],
                color=sns.color_palette("muted", len(mdf)),
                capsize=3,
            )
            ax.set_xticks(range(len(mdf)))
            ax.set_xticklabels(mdf["experiment"], rotation=45, ha="right", fontsize=7)
            ax.set_ylabel(metric.upper())
            ax.set_title(metric.upper())

        plt.tight_layout()
        path = self.plots_dir / "distributions.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path.relative_to(self.output_dir))

    # ── Plot: Significance heatmap ───────────────────────────────

    def _plot_significance_heatmap(self, sig_df: pd.DataFrame, metric: str) -> str | None:
        fig, ax = plt.subplots(figsize=self.figsize)
        mask = np.triu(np.ones_like(sig_df, dtype=bool))
        sns.heatmap(
            sig_df,
            mask=mask,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=0.1,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "p-value"},
        )
        ax.set_title(f"Pairwise p-values — {metric.upper()}")
        plt.tight_layout()
        path = self.plots_dir / f"significance_{metric}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path.relative_to(self.output_dir))

    # ── Plot: Effect sizes ───────────────────────────────────────

    def _plot_effect_sizes(self, results: list) -> str | None:
        data = []
        for r in results:
            d = r.effect_size.get("cohens_d")
            if d is not None:
                data.append(
                    {
                        "comparison": f"{r.experiment_b}",
                        "metric": r.metric,
                        "cohens_d": d,
                        "interpretation": r.effect_size.get("interpretation", ""),
                    }
                )
        if not data:
            return None
        df = pd.DataFrame(data)
        metrics = df["metric"].unique()
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5), squeeze=False)
        for idx, metric in enumerate(metrics):
            ax = axes[0, idx]
            mdf = df[df["metric"] == metric].sort_values("cohens_d")
            colors = [
                "#e74c3c" if abs(d) >= 0.8 else "#f39c12" if abs(d) >= 0.5 else "#3498db"
                for d in mdf["cohens_d"]
            ]
            ax.barh(mdf["comparison"], mdf["cohens_d"], color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            for threshold, label in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
                ax.axvline(threshold, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
                ax.axvline(-threshold, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
            ax.set_xlabel("Cohen's d (vs baseline)")
            ax.set_title(f"Effect Size — {metric.upper()}")
        plt.tight_layout()
        path = self.plots_dir / "effect_sizes.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path.relative_to(self.output_dir))

    # ── Plot: Uncertainty calibration ─────────────────────────────

    def _plot_uncertainty_calibration(self, reports: list[UncertaintyReport]) -> str | None:
        names = [r.experiment_name for r in reports]
        ece_vals = [r.ece or 0.0 for r in reports]
        sharpness_vals = [r.sharpness or 0.0 for r in reports]
        corr_vals = [r.uncertainty_error_correlation or 0.0 for r in reports]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # ECE (lower is better)
        colors = sns.color_palette("Reds_r", n_colors=len(names))
        axes[0].barh(names, ece_vals, color=colors)
        axes[0].set_xlabel("ECE")
        axes[0].set_title("Expected Calibration Error ↓")

        # Sharpness (lower is sharper)
        colors = sns.color_palette("Blues_r", n_colors=len(names))
        axes[1].barh(names, sharpness_vals, color=colors)
        axes[1].set_xlabel("Sharpness")
        axes[1].set_title("Prediction Sharpness ↓")

        # Uncertainty-Error Correlation (higher is better)
        colors = sns.color_palette("Greens", n_colors=len(names))
        axes[2].barh(names, corr_vals, color=colors)
        axes[2].set_xlabel("Correlation")
        axes[2].set_title("Uncertainty–Error Correlation ↑")

        plt.tight_layout()
        path = self.plots_dir / "uncertainty_calibration.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path.relative_to(self.output_dir))

    # ── HTML Dashboard ───────────────────────────────────────────

    def _generate_html(self, report: CampaignReport, plot_paths: list[tuple[str, str]]) -> Path:
        html_path = self.output_dir / "campaign_dashboard.html"
        plots_html = ""
        for title, rel_path in plot_paths:
            plots_html += f'<div class="plot-card"><img src="{rel_path}" alt="{title}"><div class="plot-title">{title}</div></div>\n'

        lb_html = ""
        if report.leaderboard is not None and not report.leaderboard.empty:
            lb_html = report.leaderboard.to_html(
                index=False, classes="data-table", float_format="%.4f"
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Campaign: {report.campaign_name}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f0f23;color:#e0e0e0;padding:20px}}
.container{{max-width:1400px;margin:0 auto}}
.header{{background:linear-gradient(135deg,#1a1a3e,#2d2d5e);border-radius:12px;padding:40px;margin-bottom:24px;text-align:center}}
.header h1{{font-size:2.2em;background:linear-gradient(90deg,#00d4ff,#7b68ee);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.header p{{color:#a0a0c0;margin-top:8px}}
.summary{{background:#1a1a3e;border-radius:12px;padding:24px;margin-bottom:24px;white-space:pre-line;font-family:monospace;font-size:0.9em;border-left:4px solid #7b68ee}}
.plots-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:20px;margin-bottom:24px}}
.plot-card{{background:#1a1a3e;border-radius:12px;overflow:hidden;transition:transform 0.2s}}
.plot-card:hover{{transform:translateY(-4px)}}
.plot-card img{{width:100%;display:block}}
.plot-title{{padding:12px;text-align:center;font-weight:600;color:#7b68ee}}
.data-table{{width:100%;border-collapse:collapse;background:#1a1a3e;border-radius:12px;overflow:hidden;margin-bottom:24px}}
.data-table th{{background:#2d2d5e;padding:12px;text-align:left;color:#7b68ee}}
.data-table td{{padding:10px 12px;border-top:1px solid #2d2d5e}}
.data-table tr:hover{{background:#2d2d5e}}
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>🔬 {report.campaign_name}</h1>
<p>Experiment Campaign Evaluation Dashboard</p>
</div>
<div class="summary">{report.summary}</div>
<h2 style="color:#7b68ee;margin-bottom:16px">📊 Leaderboard</h2>
{lb_html}
<h2 style="color:#7b68ee;margin:24px 0 16px">📈 Visualisations</h2>
<div class="plots-grid">{plots_html}</div>
</div>
</body>
</html>"""
        html_path.write_text(html, encoding="utf-8")
        return html_path
