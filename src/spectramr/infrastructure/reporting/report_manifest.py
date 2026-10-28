r"""Authoritative artifact index + provenance for a report bundle.

Wires the figure/table outputs + metadata into a single
``report_manifest.json`` (every artifact: id, paths, format, sha256,
status). Also emits an optional submission/ bundle (TIFF + caption .txt).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .metadata import RunMetadata


def _sha256(path: Path) -> str | None:
    if not path or not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def write_report_manifest(
    out_dir: str | Path,
    *,
    figures: dict,
    tables: dict,
    metadata: RunMetadata,
    captions: dict | None = None,
    figure_reasons: dict | None = None,
    runs: list[dict] | None = None,
) -> Path:
    """Write ``report_manifest.json`` under ``out_dir`` and return its path.

    ``figure_reasons`` maps a skipped figure id to WHY it produced no file
    (``unregistered`` / ``no_data`` / ``raised:<err>``, from
    ``plotters.dispatch_detailed``). A bare ``status: skipped`` cannot tell a
    report that legitimately has no learning curve to draw from one whose plotter
    crashed, and the two need different responses from whoever reads this file.
    Optional so existing callers are unaffected; absent reasons simply omit the
    key rather than inventing one.

    ``runs`` lists the runs the report read, each with its completion state
    (:meth:`.run_status.RunStatus.to_dict`) and whether it was ``included``,
    so a crashed arm left out of a comparison is recorded rather than absent.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    captions = captions or {}
    figure_reasons = figure_reasons or {}
    artifacts: list[dict] = []
    for fid, path in (figures or {}).items():
        if path is None:
            entry = {"id": fid, "kind": "figure", "status": "skipped"}
            if fid in figure_reasons:
                entry["reason"] = figure_reasons[fid]
            artifacts.append(entry)
            continue
        path = Path(path)
        artifacts.append(
            {
                "id": fid,
                "kind": "figure",
                "status": "ok",
                "path": str(path.relative_to(out_dir)) if out_dir in path.parents else str(path),
                "format": path.suffix.lstrip("."),
                "sha256": _sha256(path),
                "caption": captions.get(fid, ""),
            }
        )
    for tid, group in (tables or {}).items():
        if not group:
            artifacts.append({"id": tid, "kind": "table", "status": "skipped"})
            continue
        for fmt, path in group.items():
            path = Path(path)
            artifacts.append(
                {
                    "id": f"{tid}.{fmt}",
                    "kind": "table",
                    "status": "ok",
                    "path": str(path.relative_to(out_dir))
                    if out_dir in path.parents
                    else str(path),
                    "format": fmt,
                    "sha256": _sha256(path),
                }
            )
    manifest: dict = {"provenance": metadata.to_dict(), "artifacts": artifacts}
    if runs is not None:
        manifest["runs"] = runs
    path = out_dir / "report_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def write_submission_bundle(
    out_dir: str | Path, figures: dict, captions: dict | None = None
) -> Path:
    """Emit submission/ with 600-dpi TIFF + caption .txt per figure.

    Re-uses the already-emitted TIFFs when present; otherwise links the PDF.
    """
    out_dir = Path(out_dir)
    sub = out_dir / "submission"
    sub.mkdir(parents=True, exist_ok=True)
    captions = captions or {}
    for fid, path in (figures or {}).items():
        if path is None:
            continue
        path = Path(path)
        tiff = path.with_suffix(".tiff")
        src = tiff if tiff.exists() else path
        (sub / src.name).write_bytes(src.read_bytes())
        if captions.get(fid):
            (sub / f"{fid}.txt").write_text(captions[fid])
    return sub
