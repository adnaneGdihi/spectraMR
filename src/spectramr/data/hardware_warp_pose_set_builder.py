"""Build a hardware-warp pose set from the repeat sessions of a BIDS-style scanner directory.

The file->record half of Phase 3 (non-negotiable 7): this module walks the raw tree, decides
which sessions of one head form a pair, reads the two volumes, hands them to
:func:`~spectramr.infrastructure.physics.repeat_session_registration.register_session_pair`,
saves each pair's ``{"points", "residual"}`` observation and writes the pose set through its
owner, :func:`spectramr.data.hardware_warp_artifacts.write_pose_set`. The registration itself
lives in physics; the layout knowledge lives here.

Pairing. Proposition 2.3 of the candidate counts *disjoint* pairs, so the default pairs sessions
``(1, 2), (3, 4), ...`` within a subject; ``scheme="all"`` takes every pair and is offered for
small cohorts where the correlation between pairs sharing a session is accepted knowingly.
Sessions are ordered by their BIDS ``ses-`` label.

Frame. Every pair is expressed about the field of view's centre (RAS mm), which one pose set
shares: the first pair fixes ``center_mm`` and a later pair whose centre differs by more than
``frame_tolerance_mm`` is refused rather than folded in.

Provenance. A warp is specific to one unit and one software version, and the pose set must say
which. The builder reads ``DeviceSerialNumber`` and ``SoftwareVersions`` from the first
session's JSON sidecar when present; an explicit argument overrides, and having neither raises
rather than stamping a placeholder (non-negotiable 3).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from spectramr.data.hardware_warp_artifacts import write_pose_set
from spectramr.infrastructure.physics import repeat_session_registration as rsr

__all__ = [
    "LOWFIELD_DIR_SPELLINGS",
    "RepeatSession",
    "SessionPair",
    "build_pose_set",
    "discover_repeat_sessions",
    "pair_sessions",
    "read_sidecar_provenance",
    "resolve_lowfield_root",
]

logger = logging.getLogger(__name__)

# The Leiden paired 64 mT / 3 T release lays low-field scans out as
# ``64mT_data/sub-XXXX/ses-NN/anat/sub-XXXX_ses-NN_run-N_<contrast>.nii.gz``; ``run-`` is
# optional in older exports, so both spellings are matched and the first run per session is kept.
# The Zenodo archive itself names the subtree ``64mT data`` and the curated cluster copy
# ``64mT_data`` (#2418), so both are tried unless the caller names one.
_SESSION_RE = re.compile(r"_ses-(?P<ses>[A-Za-z0-9]+)_")
LOWFIELD_DIR_SPELLINGS: tuple[str, ...] = ("64mT_data", "64mT data")


@dataclass(frozen=True)
class RepeatSession:
    subject: str
    session: str
    path: Path


@dataclass(frozen=True)
class SessionPair:
    fixed: RepeatSession
    moving: RepeatSession

    @property
    def stem(self) -> str:
        return f"{self.fixed.subject}_ses-{self.fixed.session}_ses-{self.moving.session}"


def resolve_lowfield_root(raw_root: Path | str, lowfield_dir: str | None = None) -> Path:
    """``raw_root/<name>`` for the first spelling that exists; an explicit name is tried alone."""
    names = (lowfield_dir,) if lowfield_dir else LOWFIELD_DIR_SPELLINGS
    for name in names:
        root = Path(raw_root) / name
        if root.is_dir():
            return root
    tried = ", ".join(repr(n) for n in names)
    raise FileNotFoundError(f"low-field directory not found under {raw_root}: tried {tried}")


def discover_repeat_sessions(
    raw_root: Path | str,
    *,
    contrast: str = "T1w",
    subdir: str = "anat",
    lowfield_dir: str | None = None,
) -> dict[str, list[RepeatSession]]:
    """Subjects with two or more sessions of ``contrast``, each session's first run, in label order."""
    root = resolve_lowfield_root(raw_root, lowfield_dir)
    found: dict[str, dict[str, Path]] = {}
    for path in sorted(root.glob(f"sub-*/ses-*/{subdir}/sub-*_ses-*_{contrast}.nii.gz")):
        match = _SESSION_RE.search(path.name)
        if match is None:
            continue
        subject = path.name.split("_")[0]
        found.setdefault(subject, {}).setdefault(match.group("ses"), path)
    out = {
        subject: [RepeatSession(subject, ses, sessions[ses]) for ses in sorted(sessions)]
        for subject, sessions in sorted(found.items())
        if len(sessions) >= 2
    }
    logger.info("%d subject(s) with repeat %s sessions under %s", len(out), contrast, root)
    return out


def pair_sessions(
    sessions: Sequence[RepeatSession], *, scheme: str = "disjoint"
) -> list[SessionPair]:
    """``(1,2), (3,4), ...`` for ``disjoint``; every ordered-by-label pair for ``all``."""
    if scheme == "disjoint":
        return [SessionPair(sessions[i], sessions[i + 1]) for i in range(0, len(sessions) - 1, 2)]
    if scheme == "all":
        return [
            SessionPair(sessions[i], sessions[j])
            for i in range(len(sessions))
            for j in range(i + 1, len(sessions))
        ]
    raise ValueError(f"pairing scheme must be 'disjoint' or 'all', got {scheme!r}")


def read_sidecar_provenance(nifti_path: Path) -> dict[str, str | None]:
    """``{"unit_id", "software_version"}`` from the BIDS JSON sidecar, ``None`` where absent."""
    sidecar = nifti_path.with_name(nifti_path.name.replace(".nii.gz", ".json"))
    if not sidecar.is_file():
        return {"unit_id": None, "software_version": None}
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    unit = meta.get("DeviceSerialNumber") or meta.get("StationName")
    return {
        "unit_id": str(unit) if unit else None,
        "software_version": str(meta["SoftwareVersions"]) if meta.get("SoftwareVersions") else None,
    }


def _resolve_provenance(
    pairs: Iterable[SessionPair], unit_id: str | None, software_version: str | None
) -> tuple[str, str]:
    first = next(iter(pairs), None)
    sidecar = (
        read_sidecar_provenance(first.fixed.path)
        if first
        else {"unit_id": None, "software_version": None}
    )
    unit = unit_id or sidecar["unit_id"]
    version = software_version or sidecar["software_version"]
    missing = [k for k, v in (("unit_id", unit), ("software_version", version)) if not v]
    if missing:
        raise ValueError(
            f"{missing} neither given nor present in the first session's sidecar; a warp is specific "
            "to a unit and a software version, so the pose set cannot be stamped"
        )
    return str(unit), str(version)


def build_pose_set(
    pairs: Sequence[SessionPair],
    out_dir: Path | str,
    *,
    ball_radius_mm: float,
    unit_id: str | None = None,
    software_version: str | None = None,
    registration_kwargs: dict[str, Any] | None = None,
    pose_set_name: str = "pose_set.json",
) -> tuple[Path, dict[str, Any]]:
    """Register every pair, save its residual beside the pose set and write the pose set.

    Returns ``(pose_set_path, pose_set)``. Residuals are saved under ``out_dir/residuals/`` and
    referenced relative to ``out_dir``, which is what ``load_pair_residuals`` resolves against.
    """
    rsr.require_sitk()
    import SimpleITK as sitk  # noqa: N813

    if not pairs:
        raise ValueError("no session pairs to register; the pose set would be empty")
    out = Path(out_dir)
    (out / "residuals").mkdir(parents=True, exist_ok=True)
    unit, version = _resolve_provenance(pairs, unit_id, software_version)
    tolerance = float((registration_kwargs or {}).get("frame_tolerance_mm", 1.0))
    center: torch.Tensor | None = None
    records = []
    for k, pair in enumerate(pairs):
        fixed = sitk.ReadImage(str(pair.fixed.path), sitk.sitkFloat32)
        moving = sitk.ReadImage(str(pair.moving.path), sitk.sitkFloat32)
        reg = rsr.register_session_pair(
            fixed, moving, ball_radius_mm=ball_radius_mm, **(registration_kwargs or {})
        )
        if center is None:
            center = reg.center
        elif float((reg.center - center).norm()) > tolerance:
            raise ValueError(
                f"{pair.stem}: FOV centre {reg.center.tolist()} differs from the pose set's "
                f"{center.tolist()} by more than {tolerance:g} mm; one pose set is one frame"
            )
        rel = Path("residuals") / f"{pair.stem}.pt"
        torch.save({"points": reg.points, "residual": reg.residual}, out / rel)
        records.append(
            {
                "subject_id": pair.fixed.subject,
                "sessions": [pair.fixed.session, pair.moving.session],
                "omega": reg.omega.tolist(),
                "translation": reg.translation.tolist(),
                "residual_path": rel.as_posix(),
            }
        )
        logger.info("pair %d/%d (%s) written to %s", k + 1, len(pairs), pair.stem, rel)
    pose_set = {
        "unit_id": unit,
        "software_version": version,
        "ball_radius_mm": float(ball_radius_mm),
        "center_mm": [float(c) for c in center],
        "pairs": records,
    }
    return write_pose_set(pose_set, out / pose_set_name), pose_set
