"""SSOT guard: data loading must route through ``src/data/``.

CLAUDE.md pitfall #11 mandates that any code turning a file (``.h5``,
``.pt``, ``.nii.gz``, ``.npy``, ``.pkl``) into a ``torch.utils.data.Dataset``
or ``DataLoader`` must live under ``src/data/``.

This test pins the current state of forbidden file-load patterns
outside ``src/data/``. The intent:

- **New violations fail the test immediately** (regression guard).
- **Fixing an allowlisted violation also fails** the test (force the
  fixer to update this allowlist, so the documentation stays honest).

Allowlist categories (legitimate, not pipeline data loading):

- Physics helpers loading static config-like artifacts (coil-sensitivity
  maps, sampling patterns) — not Dataset construction.
- I/O adapter primitives at the bottom of the data layer
  (``src/infrastructure/io/adapters.py``).
- Checkpoint service loading model weights (not data).
- CLI utilities + reporting plotters that operate on artifacts after
  training, not during it.
- Docstring examples (we look for the actual call site, but accept
  files whose only hits are in docstrings).

Phase-1 targets (planned removals, currently allowlisted with the
``"phase1_target"`` marker so the test reminds us to remove them when
the eval bypass is plugged):

- ``src/infrastructure/orchestration/campaign_evaluator.py``
- ``src/pipelines/infer.py`` (Phase-2 target, also currently allowlisted)
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


class Violation(NamedTuple):
    path: str  # repo-relative
    pattern: str  # the matched pattern


# ── Forbidden patterns (data-load primitives) ───────────────────────────────

_PATTERNS = [
    r"\bnib\.load\(",
    r"\bnibabel\.load\(",
    r"\bh5py\.File\(",
    r"\bpickle\.load\(",
    r"\bnp\.load\(",
    # torch.load is mostly checkpoints — covered by checkpoint allowlist
]


# ── Allowlist: file → reason. Entries here intentionally permit the pattern.
# Phase-1 targets are flagged so the fix landing in Phase 1 will trip the test.

_ALLOWLIST: dict[str, str] = {
    # Physics layer: loads coil maps / sampling patterns (config-like, not Datasets).
    "src/spectramr/infrastructure/physics/coil_sensitivity.py": "loads coil-sensitivity maps as physics primitive",
    "src/spectramr/infrastructure/physics/integration.py": "loads pre-computed coil maps for sense_forward",
    "src/spectramr/infrastructure/physics/sampling.py": "loads externally-provided sampling masks",

    # IO adapter layer: bottom-of-stack primitives the data layer composes from.
    "src/spectramr/infrastructure/io/adapters.py": "low-level IO primitives consumed by src/data/",

    # Domain helpers using optional imports.
    "src/spectramr/domain/entities/optional_imports.py": "guarded h5py.File for optional dependency",
    "src/spectramr/domain/entities/entities_3d.py": "Sphinx doctest example (>>> nib.load) — not executable code",

    # (Checkpoint service uses pickle.loads on bytes, not pickle.load on a
    # file handle — not matched by the current patterns. No allowlist entry
    # needed.)

    # Reporting plotters operate on already-trained-artifacts (post-hoc).
    "src/spectramr/infrastructure/reporting/plotters/failure_gallery.py": "loads sample artifacts for failure-gallery reports",

    # CLI tools / utilities — not part of the training pipeline.
    "src/spectramr/tools/cluster_explorer.py": "CLI utility for cluster filesystem inspection",
    "src/spectramr/tools/inspect_cluster_data.py": "CLI utility for cluster filesystem inspection",
    "src/spectramr/tools/index_datasets.py": "CLI utility that BUILDS the manifests src/data/ consumes",
    "src/spectramr/shared/utils/preprocessing_qc.py": "QC tool — operates on artifacts post-preprocessing",
    "src/spectramr/cli/audit_plan_novel_cli.py": "ad-hoc audit CLI: loads precomputed sample tensors / scores (not training data)",

    # Diagnostic block — loads pre-trained statistical model, not data.
    "src/spectramr/models/blocks/metric_sfc.py": "loads pre-trained SFC metric model parameters",

    # ─── Phase 1 (LANDED) — campaign_evaluator's data sites are now ───
    # routed through src/data/builders/eval_dataset_builder.py. Only the
    # pkl manifest load remains, which is config per Amendment C of the
    # data-layer unification plan.
    "src/spectramr/infrastructure/orchestration/campaign_evaluator.py": "pickle.load on manifest (paths/config, not MRI data — Amendment C)",

    # ─── Phase-2 target — inference data path will route through SSOT ───
    "src/spectramr/pipelines/infer.py": "phase2_target: inference data path to route through DataPipelineDirector",
}


def _grep_pattern(pattern: str) -> list[Violation]:
    """Use git-grep to find ``pattern`` in ``src/``; return repo-relative hits."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "grep",
                "-n",
                "-E",
                pattern,
                "--",
                "src/spectramr/",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git not available")
    if result.returncode not in (0, 1):  # 1 = no matches; both are fine
        pytest.skip(f"git grep failed: {result.stderr.strip()}")
    hits: list[Violation] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        path = line.split(":", 1)[0]
        # Filter out anything that lives under src/data/ — that's the SSOT.
        if path.startswith("src/spectramr/data/"):
            continue
        # Filter out tests directory (mirrored grep often misses but be safe).
        if "/tests/" in path or path.startswith("tests/"):
            continue
        hits.append(Violation(path=path, pattern=pattern))
    return hits


# ── Tests ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pattern", _PATTERNS)
def test_no_unallowlisted_data_loading(pattern: str) -> None:
    """For each forbidden pattern, fail if any unallowlisted file uses it."""
    hits = _grep_pattern(pattern)
    unallowed = [h for h in hits if h.path not in _ALLOWLIST]
    if unallowed:
        msg_lines = [
            f"Forbidden data-load pattern {pattern!r} appears outside "
            f"src/spectramr/data/ in unallowlisted files (CLAUDE.md #11):",
        ]
        for h in unallowed:
            msg_lines.append(f"  - {h.path}")
        msg_lines.append(
            "\nFix options:\n"
            "  (a) route the load through src/data/ (preferred — file becomes\n"
            "      a Dataset), OR\n"
            "  (b) if the file is legitimately a physics/IO/CLI helper, add\n"
            "      it to _ALLOWLIST in this test with a reason."
        )
        pytest.fail("\n".join(msg_lines))


def test_allowlist_entries_are_still_relevant() -> None:
    """Every allowlist entry must still match at least one forbidden pattern.

    When a Phase-1 (or later) refactor removes a load from an
    allowlisted file, this test trips — forcing the fixer to delete
    the stale allowlist entry. Keeps the allowlist honest.
    """
    all_hits: set[str] = set()
    for pat in _PATTERNS:
        for h in _grep_pattern(pat):
            all_hits.add(h.path)
    stale: list[str] = []
    for entry, reason in _ALLOWLIST.items():
        if entry not in all_hits:
            stale.append(f"  - {entry}  (reason: {reason})")
    if stale:
        pytest.fail(
            "Stale _ALLOWLIST entries — these files no longer match any "
            "forbidden pattern. Remove them from _ALLOWLIST:\n"
            + "\n".join(stale)
        )


def test_phase1_targets_documented() -> None:
    """Sanity: phase1_target entries must still exist (until Phase 1 lands).

    This is the audit trail — when Phase 1 removes the eval bypass, the
    'allowlist still relevant' test above will flag the stale entry
    AND this test will flip green by virtue of the entry being deleted.
    """
    p1_targets = [
        e for e, reason in _ALLOWLIST.items() if "phase1_target" in reason
    ]
    p2_targets = [
        e for e, reason in _ALLOWLIST.items() if "phase2_target" in reason
    ]
    # Phase 1 (eval bypass plug) landed: data-bearing sites in
    # campaign_evaluator now route through eval_dataset_builder.
    # Only the pkl manifest load remains, which is allowlisted with
    # a non-phase1_target reason ("config not data, per Amendment C").
    assert len(p1_targets) == 0, (
        f"Phase 1 landed — there should be 0 phase1_target allowlist "
        f"entries, found {len(p1_targets)}: {p1_targets}."
    )
    assert len(p2_targets) == 1, (
        f"Expected 1 phase2_target (inference data path), found "
        f"{len(p2_targets)}: {p2_targets}. Update this assertion when "
        "Phase 2 progresses."
    )
