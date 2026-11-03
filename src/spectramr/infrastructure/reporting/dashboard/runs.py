"""The runs table: one row per run, from the campaign state, a batch, or plain discovery."""

from __future__ import annotations

from pathlib import Path

from .shell import Section, esc, rel


def _run_dir(root: Path, exp: dict) -> Path | None:
    """The folder under ``root`` holding ``exp``: its recorded ``output_dir``, else by name.

    A name matches itself or ``name_<suffix>`` (``x_r4`` -> ``x_r4_m4raw``), never a longer
    sibling (``theta1`` must not claim ``theta10``).
    """
    recorded = Path(exp.get("output_dir") or "").name
    if recorded and (root / recorded).is_dir():
        return root / recorded
    name = exp.get("name", "")
    hits = sorted(
        d
        for d in root.iterdir()
        if d.is_dir() and (d.name == name or d.name.startswith(name + "_"))
    )
    exact = [d for d in hits if (d / "report").is_dir()]
    return (exact or hits or [None])[0]


def _report_of(run: Path) -> Path | None:
    p = run / "report" / "qc_report.html"
    return p if p.exists() else None


def run_records(root: Path, state: dict, batch_rows: list[dict] | None) -> list[dict]:
    """One record per run from, in order of preference, the campaign state, a batch, or discovery."""
    if state.get("experiments"):
        out = []
        claimed: set[Path] = set()
        for e in state["experiments"]:
            d = _run_dir(root, e)
            if d is not None:
                claimed.add(d)
            out.append(
                {
                    "name": e.get("name", ""),
                    "role": e.get("role"),
                    "status": e.get("status", "unknown"),
                    "wall": e.get("wall_time_seconds"),
                    "job": e.get("slurm_job_id"),
                    "report": _report_of(d) if d is not None else None,
                    "error": None,
                }
            )
        # A campaign state is often stale (resubmitted arms, a status never updated): runs that
        # finished on disk but the state does not name still belong in the table.
        from ..batch import discover_run_dirs  # batch imports this module; resolve at call time

        for d in discover_run_dirs(root):
            if d not in claimed and d.parent == root:
                out.append(
                    {
                        "name": d.name,
                        "role": None,
                        "status": "on disk",
                        "wall": None,
                        "job": None,
                        "report": _report_of(d),
                        "error": None,
                    }
                )
        return out
    if batch_rows is None:
        from ..batch import discover_run_dirs  # batch imports this module; resolve at call time

        batch_rows = [{"run": d, "ok": _report_of(d) is not None} for d in discover_run_dirs(root)]
    out = []
    for r in batch_rows:
        run = Path(r["run"])
        html = Path(r["html"]) if r.get("html") else _report_of(run)
        status = "reported" if r.get("ok") else "failed" if r.get("error") else "no report"
        out.append(
            {
                "name": str(run.relative_to(root)) if run != root else run.name,
                "role": None,
                "status": status,
                "wall": None,
                "job": None,
                "report": html,
                "error": r.get("error"),
            }
        )
    return out


def runs_section(page: Path, records: list[dict]) -> Section | None:
    if not records:
        return None
    rows = []
    for e in records:
        report = e["report"]
        link = (
            f"<a href='{esc(rel(report, page))}'>Open report</a>"
            if report is not None and Path(report).exists()
            else "<span class='meta'>No report</span>"
        )
        status = str(e["status"])
        cls = (
            "ok"
            if status in ("completed", "reported", "on disk")
            else "bad"
            if status in ("failed", "cancelled", "timeout")
            else "meta"
        )
        title = f" title='{esc(e['error'])}'" if e.get("error") else ""
        wall = e.get("wall")
        wall_txt = f"{wall / 60:.0f} min" if isinstance(wall, (int, float)) else ""
        rows.append(
            f"<tr><td>{esc(e['name'])}</td><td>{esc(e.get('role') or '')}</td>"
            f"<td><span class='tag {cls}'{title}>{esc(status)}</span></td><td class='num'>{esc(wall_txt)}</td>"
            f"<td class='num'>{esc(e.get('job') or '')}</td><td>{link}</td></tr>"
        )
    n_done = sum(1 for e in records if e["status"] in ("completed", "reported", "on disk"))
    head = "".join(
        f"<th scope='col'>{h}</th>" for h in ("Run", "Role", "Status", "Wall time", "Job", "Report")
    )
    body = (
        f"<div class='tbl-wrap'><table class='tbl'><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )
    return Section("runs", "Runs", body, note=f"{n_done} of {len(records)} runs completed.")
