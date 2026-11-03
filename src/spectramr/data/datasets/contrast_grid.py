"""One content model for multi-subject, multi-contrast reference discovery.

Real-data Sim2Rank references are a :class:`ContrastGrid` of :class:`ContentKey`
cells with explicit ``(subject, contrast)`` identity, replacing the positional
``s * C + c`` index that misaligned on incomplete subjects.

This module is the single owner of contrast parsing from file names (M4Raw
``<subject>_<contrast><rep>``, fastMRI ``AX*``) and of the fastMRI header-versus-
filename resolution, which raises on disagreement. The unlinked brain grid is
built in :mod:`spectramr.data.datasets.fastmri_brain_groups`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "ContentKey",
    "ContrastGrid",
    "ContrastMismatchError",
    "IncompleteSubjectError",
    "brain_filename_contrast",
    "choose_contrasts",
    "contrast_label",
    "discover_m4raw_grid",
    "h5_files",
    "parse_m4raw_stem",
    "resolve_brain_contrast",
]

_M4RAW_STEM = re.compile(r"^(?P<subject>.+)_(?P<contrast>[A-Za-z]+\d*?)(?P<rep>\d{2})$")
_AX_TOKEN = re.compile(r"^AX(?:T1|T2|FLAIR|PD)[A-Z0-9]*$")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class ContrastMismatchError(ValueError):
    """The h5 ``acquisition`` attr and the filename disagree, or the name is ambiguous."""


class IncompleteSubjectError(ValueError):
    """A subject lacks one of the grid's contrasts."""


def _stem(path: Path | str) -> str:
    return Path(path).name.split(".")[0]


def parse_m4raw_stem(path: Path | str) -> tuple[str, str, int]:
    """``(subject, contrast, rep)`` of an M4Raw ``<subject>_<contrast><rep>`` name; else raises."""
    m = _M4RAW_STEM.match(_stem(path))
    if m is None:
        raise ValueError(
            f"{Path(path).name!r} is not an M4Raw '<subject>_<contrast><rep>' name "
            "(e.g. 2022091411_T201.h5); refusing to group it."
        )
    return m.group("subject"), m.group("contrast").upper(), int(m.group("rep"))


def _ax_tokens(path: Path | str) -> list[str]:
    return [t.upper() for t in _stem(path).split("_") if _AX_TOKEN.match(t.upper())]


def brain_filename_contrast(path: Path | str) -> str:
    """The single exact ``AX*`` underscore token of a fastMRI-brain name; raises on 0 or >= 2."""
    found = _ax_tokens(path)
    if len(found) != 1:
        raise ContrastMismatchError(
            f"{Path(path).name!r} contains {len(found)} recognised acquisition tokens "
            f"({found}); expected exactly 1 AX* token. Tokens: {sorted(_stem(path).split('_'))}."
        )
    return found[0]


def resolve_brain_contrast(path: Path | str, acquisition: str | bytes | None) -> str:
    """Header attr versus filename token; disagreement raises, absence of both raises."""
    tokens = _ax_tokens(path)
    from_name = brain_filename_contrast(path) if tokens else None
    from_attr = None
    if acquisition is not None:
        raw = acquisition.decode() if isinstance(acquisition, bytes) else str(acquisition)
        from_attr = raw.strip().upper() or None
    if from_attr and from_name and from_attr != from_name:
        raise ContrastMismatchError(
            f"{Path(path).name}: the h5 'acquisition' attr says {from_attr!r} but the "
            f"filename token says {from_name!r}. A file whose header and name disagree "
            "about its contrast has broken provenance; fix the file or exclude it."
        )
    label = from_attr or from_name
    if label is None:
        raise ValueError(
            f"Cannot determine a contrast for {Path(path).name}: no 'acquisition' attr "
            "and no AX* token in the filename."
        )
    return label


def contrast_label(path: Path | str) -> str:
    """Contrast of either naming scheme: the fastMRI ``AX*`` token, else the M4Raw token."""
    if _ax_tokens(path):
        return brain_filename_contrast(path)
    return parse_m4raw_stem(path)[1]


@dataclass(frozen=True)
class ContentKey:
    """One clean reference: who, which contrast, which slice (``None`` = central), which files."""

    subject_id: str
    contrast: str
    slice_index: int | None
    file_ids: tuple[str, ...]

    @property
    def paths(self) -> list[Path]:
        return [Path(f) for f in self.file_ids]

    @property
    def content_id(self) -> str:
        """Stable, filename-safe id: ``<subject>_<contrast>[_s<k>]``."""
        base = f"{self.subject_id}_{self.contrast}"
        if self.slice_index is not None:
            base += f"_s{self.slice_index}"
        return _UNSAFE.sub("-", base)


@dataclass(frozen=True)
class ContrastGrid:
    """Ordered contents, row-major over ``rows()``, contrast-minor in ``contrasts`` order.

    ``linked=True``: a row is one subject imaged in every contrast (M4Raw).
    ``linked=False``: contrasts come from different exams (fastMRI brain), so a
    row is a positional ``index`` and ``subjects`` lists exam ids.
    """

    keys: tuple[ContentKey, ...]
    subjects: tuple[str, ...]
    contrasts: tuple[str, ...]
    linked: bool
    dropped_subjects: tuple[str, ...] = field(default=())
    min_reps: int = 1

    def __post_init__(self) -> None:
        ids = [k.content_id for k in self.keys]
        if len(set(ids)) != len(ids):
            raise ValueError(f"ContrastGrid content_ids are not unique: {ids}")
        c = len(self.contrasts)
        if (c == 0 and self.keys) or (c and len(self.keys) % c):
            raise ValueError(f"{len(self.keys)} keys do not tile {c} contrasts")
        for row in self.rows():
            if tuple(k.contrast for k in row) != self.contrasts:
                raise ValueError(f"row {[k.content_id for k in row]} not in {self.contrasts}")
            if self.linked and len({k.subject_id for k in row}) != 1:
                raise ValueError(f"linked row spans subjects: {[k.content_id for k in row]}")

    @property
    def row_label(self) -> str:
        return "subject" if self.linked else "index"

    @property
    def n_rows(self) -> int:
        return len(self.keys) // len(self.contrasts) if self.contrasts else 0

    def rows(self) -> list[tuple[ContentKey, ...]]:
        c = len(self.contrasts)
        return [self.keys[r * c : (r + 1) * c] for r in range(self.n_rows)]

    def select(self, contrast: str | None = None, per_subject: int | None = None) -> ContrastGrid:
        """Sub-grid of one contrast and/or the first ``per_subject`` contents of each subject."""
        keys = list(self.keys)
        contrasts = self.contrasts
        if contrast is not None:
            want = contrast.strip().upper()
            if want not in self.contrasts:
                raise ValueError(f"contrast {want!r} not in grid {list(self.contrasts)}")
            keys = [k for k in keys if k.contrast == want]
            contrasts = (want,)
        if per_subject is not None:
            if per_subject < 1:
                raise ValueError(f"per_subject must be >= 1, got {per_subject}")
            rank = {
                k.content_id: sum(j.subject_id == k.subject_id for j in keys[:i])
                for i, k in enumerate(keys)
            }
            keys = [k for k in keys if rank[k.content_id] < per_subject]
            if self.linked and per_subject < len(contrasts):
                contrasts = contrasts[:per_subject]
        subjects = tuple(dict.fromkeys(k.subject_id for k in keys))
        return ContrastGrid(
            tuple(keys), subjects, contrasts, self.linked, self.dropped_subjects, self.min_reps
        )

    def provenance(self) -> list[dict]:
        """Per-content table for the results JSON (file basenames, not absolute paths)."""
        return [
            {
                "content_id": k.content_id,
                "subject_id": k.subject_id,
                "contrast": k.contrast,
                "slice_index": k.slice_index,
                "file_ids": [Path(f).name for f in k.file_ids],
                "n_reps": len(k.file_ids),
            }
            for k in self.keys
        ]


def h5_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("*.h5")) or sorted(data_dir.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No H5 files found in {data_dir}")
    return files


def choose_contrasts(
    present: dict[str, int], requested: list[str] | None, cap: int, where: Path
) -> list[str]:
    """``requested`` in user order (each must be present), else most-populated first."""
    if requested:
        want = [c.strip().upper() for c in requested]
        if len(want) > cap:
            raise ValueError(
                f"{len(want)} contrasts requested ({want}) but --max-contrasts is {cap}; "
                "raise --max-contrasts or narrow --contrasts."
            )
        missing = [c for c in want if c not in present]
        if missing:
            raise ValueError(
                f"Requested contrast(s) {missing} not found in {where}. Present: {sorted(present)}"
            )
        chosen = want
    else:
        chosen = sorted(present, key=lambda c: (-present[c], c))[:cap]
    if not chosen:
        raise ValueError(f"No usable contrasts discovered in {where}")
    return chosen


def discover_m4raw_grid(
    data_dir: Path,
    contrasts: list[str] | None = None,
    max_subjects: int = 5,
    max_contrasts: int = 3,
    allow_incomplete: bool = False,
) -> ContrastGrid:
    """Linked ``subject x contrast`` grid of M4Raw repetition groups.

    A cell needs >= 2 repetitions when any cell in the tree has them (1 otherwise,
    the singleton law, stamped as ``min_reps``). A subject lacking a chosen
    contrast raises :class:`IncompleteSubjectError`, or is dropped and recorded in
    ``dropped_subjects`` when ``allow_incomplete``.
    """
    cells: dict[str, dict[str, list[Path]]] = {}
    for f in h5_files(data_dir):
        subject, contrast, _ = parse_m4raw_stem(f)
        cells.setdefault(subject, {}).setdefault(contrast, []).append(f)
    min_reps = 2 if any(len(r) >= 2 for c in cells.values() for r in c.values()) else 1
    if min_reps == 1:
        logger.warning("No multi-repetition groups in %s; using single acquisitions.", data_dir)
    usable = {
        s: {c: sorted(r) for c, r in cs.items() if len(r) >= min_reps} for s, cs in cells.items()
    }
    present: dict[str, int] = {}
    for cs in usable.values():
        for c in cs:
            present[c] = present.get(c, 0) + 1
    chosen = choose_contrasts(present, contrasts, max_contrasts, data_dir)

    keys: list[ContentKey] = []
    subjects: list[str] = []
    dropped: list[str] = []
    for subject in sorted(usable):
        if len(subjects) >= max_subjects:
            break
        missing = [c for c in chosen if c not in usable[subject]]
        if missing:
            if not allow_incomplete:
                raise IncompleteSubjectError(
                    f"M4Raw subject {subject!r} lacks contrast(s) {missing} (with >= {min_reps} "
                    f"reps) of the grid {chosen}. Pass --allow-incomplete-subjects to drop it "
                    "(the dropped list is stamped), or narrow --contrasts."
                )
            dropped.append(subject)
            continue
        subjects.append(subject)
        keys.extend(
            ContentKey(subject, c, None, tuple(str(p) for p in usable[subject][c])) for c in chosen
        )
    if dropped:
        logger.warning("Dropped %d incomplete M4Raw subject(s): %s", len(dropped), dropped)
    if not subjects:
        raise ValueError(f"No complete M4Raw subject for contrasts {chosen} in {data_dir}")
    logger.info("M4Raw grid: %d subjects x %d contrasts %s", len(subjects), len(chosen), chosen)
    return ContrastGrid(tuple(keys), tuple(subjects), tuple(chosen), True, tuple(dropped), min_reps)
