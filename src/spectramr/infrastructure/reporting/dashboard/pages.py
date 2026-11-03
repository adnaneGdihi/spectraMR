"""Comparison and contrast pages, rendered from a folder ``spectramr report --compare`` wrote.

Both read only what is on disk (``data/``, ``figures/``, ``tables/``, ``contrast.json``), so
the same function serves a fresh comparison and a downloaded campaign tree.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from . import charts
from .figure_index import FAMILIES, OTHER, caption, family, slug
from .shell import Section, esc, figure_html, gallery, rel, render_page, table_html

HEADLINE = ("psnr", "ssim", "nmse", "lpips", "nrmse", "sense_psnr")
PHASES = ("test", "val", "train")


def figure_ids(folder: Path) -> list[str]:
    """Figure stems in manifest order (then any extra files on disk), each once."""
    ids: list[str] = []
    manifest = folder / "report_manifest.json"
    if manifest.exists():
        for a in json.loads(manifest.read_text()).get("artifacts", []):
            if a.get("kind") == "figure" and a.get("status") == "ok":
                ids.append(Path(a["path"]).stem)
    figdir = folder / "figures"
    if figdir.is_dir():
        ids += sorted({p.stem for p in figdir.iterdir() if p.suffix in (".png", ".pdf", ".svg")})
    return list(dict.fromkeys(ids))


def gallery_sections(folder: Path, page: Path, ids: Sequence[str]) -> list[Section]:
    """One gallery section per figure family, in family order, for the figures in ``folder``."""
    groups: dict[str, list[str]] = {}
    for fid in ids:
        groups.setdefault(family(fid), []).append(fid)
    out = []
    for fam in [name for name, _ in FAMILIES] + [OTHER]:
        tiles = [
            figure_html(
                folder / "figures" / fid, page, caption=caption(fid), wide=family(fid) == "Images"
            )
            for fid in groups.get(fam, [])
        ]
        body = gallery(tiles)
        if body:
            out.append(Section(slug(fam), fam, body))
    return out


def read_long(folder: Path) -> pd.DataFrame:
    p = folder / "data" / "metrics_long.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def pick_phase(long_df: pd.DataFrame) -> str | None:
    present = set(long_df.get("phase", pd.Series(dtype=str)))
    return next((p for p in PHASES if p in present), None)


def headline_metrics(long_df: pd.DataFrame, phase: str) -> list[str]:
    dirs = charts.directions(long_df)
    have = [m for m in long_df.loc[long_df["phase"] == phase, "metric"].unique() if m in dirs]
    return [m for m in HEADLINE if m in have][:4] or have[:4]


def results_table(
    long_df: pd.DataFrame, phase: str, metrics: list[str], baseline: str | None
) -> str:
    wide = long_df[long_df["phase"] == phase].pivot_table(
        index="arm", columns="metric", values="mean", aggfunc="first"
    )
    cols = [m for m in metrics if m in wide.columns]
    order = charts.rank_table(long_df, phase, metrics).index
    wide = wide.reindex(
        [a for a in order if a in wide.index] + [a for a in wide.index if a not in order]
    )
    best = {m: charts.best_arm(long_df, phase, m) for m in cols}
    return table_html(wide[cols].reset_index(), best=best, baseline=baseline)


def metric_sections(long_df: pd.DataFrame, baseline: str | None) -> list[Section]:
    phase = pick_phase(long_df)
    if phase is None:
        return []
    heads = headline_metrics(long_df, phase)
    out = []
    lb = charts.leaderboard_div(long_df, phase=phase, metrics=heads, baseline=baseline)
    sub = long_df[long_df["phase"] == phase]
    units = ", ".join(sorted(set(sub["unit"].dropna().astype(str)))) if "unit" in sub else "rows"
    if lb:
        out.append(
            Section(
                "leaderboard",
                "Leaderboard",
                f"<div class='panel'>{lb}</div><p class='howto'><b>How to read this.</b> One row per arm, "
                "best mean rank over these metrics at the top. Dots are means and bars one SD across "
                f"the {esc(units)} each arm logged: spread, not a test. Paired tests over subjects are "
                "under Contrasts. The orange dot is the best "
                "arm on that metric and the grey one is the baseline.</p>",
                note=f"{phase.capitalize()} split, {long_df.loc[long_df['phase'] == phase, 'arm'].nunique()} arms.",
            )
        )
    hm = charts.rank_heatmap_div(long_df, phase=phase)
    if hm:
        out.append(
            Section(
                "ranks",
                "Ranks across every metric",
                f"<div class='panel'>{hm}</div>",
                note="Darker is better. Only metrics with a declared direction are ranked.",
            )
        )
    out.append(
        Section(
            "results",
            "Results table",
            results_table(long_df, phase, heads, baseline),
            note="Headline means per arm, best value per column highlighted; every metric is in data/metrics_long.csv.",
        )
    )
    return out


def downloads(folder: Path, page: Path) -> str:
    files = sorted(p for d in ("data", "tables") for p in (folder / d).glob("*") if p.is_file())
    files += [p for p in (folder / "comparison_summary.md", folder / "contrast.json") if p.exists()]
    links = "".join(
        f"<li><a href='{esc(rel(p, page))}'>{esc(p.parent.name)}/{esc(p.name)}</a></li>"
        for p in files
    )
    return f"<ul class='meta'>{links}</ul>" if links else ""


def comparison_page(
    folder: str | Path,
    *,
    baseline: str | None = None,
    title: str | None = None,
    crumbs: Sequence[tuple[str, str | None]] = (),
    plotly_js: Path | None = None,
    filename: str = "comparison.html",
) -> Path:
    """Write the comparison (or, when ``contrast.json`` exists, contrast) page for ``folder``."""
    folder = Path(folder)
    page = folder / filename
    cpath = folder / "contrast.json"
    contrast = json.loads(cpath.read_text()) if cpath.exists() else None
    c = (contrast or {}).get("contrast", {})
    long_df = read_long(folder)
    sections: list[Section] = []
    lede = None
    if contrast is not None:
        baseline = baseline or c.get("baseline")
        others = [a["label"] for a in c.get("arms", []) if a.get("label") != baseline]
        lede = f"{', '.join(others)} tested against {baseline}, paired over subjects."
        sections += contrast_sections(folder, long_df)
    sections += metric_sections(long_df, baseline)
    sections += gallery_sections(folder, page, figure_ids(folder))
    dl = downloads(folder, page)
    if dl:
        sections.append(Section("files", "Data and tables", dl))
    prov = {}
    manifest = folder / "report_manifest.json"
    if manifest.exists():
        p = json.loads(manifest.read_text()).get("provenance", {})
        prov = {
            "Commit": (p.get("git_commit") or "")[:10] + ("*" if p.get("git_dirty") else ""),
            "Data": p.get("dataset_version"),
            "Generated": (p.get("timestamp_utc") or "")[:16],
        }
    prov["Baseline"] = baseline
    return render_page(
        page,
        title=title or contrast_title(c) or folder.parent.name,
        lede=lede,
        sections=sections,
        crumbs=crumbs,
        provenance=prov,
        plotly_js=plotly_js,
    )


def contrast_title(c: dict) -> str | None:
    """``variant vs baseline`` from a contrast block, or None when it has no name."""
    name = c.get("name")
    return name.replace("__vs__", " vs ") if name else None


def contrast_sections(folder: Path, long_df: pd.DataFrame) -> list[Section]:
    stats_path = folder / "data" / "paired_stats.csv"
    if not stats_path.exists():
        return []
    stats = pd.read_csv(stats_path)
    dirs = charts.directions(long_df)
    out = []
    for phase in [p for p in PHASES if p in set(stats["phase"])]:
        eff = charts.oriented_effects(stats[stats["phase"] == phase], dirs)
        if eff.empty:
            continue
        # One panel, metrics as rows: d is unitless, so every metric shares the axis.
        div = charts.effect_dots_div(
            eff.rename(columns={"metric": "row"}).assign(metric="effect size d"),
            rows="row",
            metrics=["effect size d"],
        )
        n_sig = int(eff["significant"].sum())
        out.append(
            Section(
                f"effects-{phase}",
                f"Effects on the {phase} split",
                (f"<div class='panel'>{div}</div>" if div else "")
                + "<p class='howto'><b>How to read this.</b> Each row is a metric. Right of zero means the "
                "variant beats the baseline; filled dots survive Holm correction at 0.05.</p>",
                note=f"{n_sig} of {len(eff)} metrics differ after Holm correction.",
            )
        )
    keep = ["phase", "metric", "mean_diff", "ci_low", "ci_high", "cohens_d", "p_holm", "n_subjects"]
    tbl = stats[[c for c in keep if c in stats.columns]]
    out.append(Section("paired-stats", "Paired statistics", table_html(tbl)))
    return out
