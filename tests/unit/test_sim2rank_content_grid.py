"""Sim2Rank's real-data grouping goes through one aligned content grid (C1, C3)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("scripts.sim2rank")

_SUBJECTS = ("2022061001", "2022061002", "2022061003")


def _tree(root: Path, *, with_incomplete: bool = False) -> Path:
    for subj in _SUBJECTS:
        for c, n in (("FLAIR", 2), ("T1", 3), ("T2", 3)):
            for r in range(1, n + 1):
                (root / f"{subj}_{c}{r:02d}.h5").write_bytes(b"")
    if with_incomplete:
        for c in ("T1", "T2"):
            for r in (1, 2, 3):
                (root / f"2022061004_{c}{r:02d}.h5").write_bytes(b"")
    return root


def _args(root: Path, **kw) -> SimpleNamespace:
    base = {"data_dir": str(root), "dataset_kind": "m4raw", "contrasts": None}
    base.update(kw)
    return SimpleNamespace(**base)


def _subject_contrast(group: list[Path]) -> tuple[str, str]:
    names = {tuple(Path(p).stem[:-2].split("_")) for p in group}
    assert len(names) == 1, group
    return names.pop()


def test_one_contrast_meta_eval_groups_are_distinct_subjects(tmp_path: Path) -> None:
    from scripts.sim2rank.sim2rank import _discover_native_groups

    groups = _discover_native_groups(_args(_tree(tmp_path)), max_subjects=3, max_contrasts=1)
    assert [_subject_contrast(g)[0] for g in groups] == list(_SUBJECTS)


def test_max_contrasts_two_rows_stay_aligned(tmp_path: Path) -> None:
    from scripts.sim2rank.sim2rank import _discover_native_grid

    grid = _discover_native_grid(_args(_tree(tmp_path)), max_subjects=2, max_contrasts=2)
    for row in grid.rows():
        cells = [_subject_contrast(k.paths) for k in row]
        assert len({s for s, _ in cells}) == 1
        assert [c for _, c in cells] == list(grid.contrasts)


def test_contrasts_knob_is_read_on_m4raw(tmp_path: Path) -> None:
    from scripts.sim2rank.sim2rank import _discover_native_grid

    grid = _discover_native_grid(
        _args(_tree(tmp_path), contrasts=["T2"]), max_subjects=3, max_contrasts=3
    )
    assert grid.contrasts == ("T2",)
    assert {_subject_contrast(k.paths)[1] for k in grid.keys} == {"T2"}


def test_incomplete_subject_raises_unless_allowed(tmp_path: Path) -> None:
    from scripts.sim2rank.sim2rank import _discover_native_grid

    root = _tree(tmp_path, with_incomplete=True)
    with pytest.raises(ValueError, match="2022061004"):
        _discover_native_grid(_args(root), max_subjects=4, max_contrasts=3)
    grid = _discover_native_grid(
        _args(root, allow_incomplete_subjects=True), max_subjects=4, max_contrasts=3
    )
    assert grid.dropped_subjects == ("2022061004",)


def test_allow_incomplete_flag_parses() -> None:
    import sys

    from scripts.sim2rank.sim2rank import parse_args

    old = sys.argv
    try:
        sys.argv = ["sim2rank.py", "--allow-incomplete-subjects"]
        assert parse_args().allow_incomplete_subjects is True
    finally:
        sys.argv = old


def test_generations_and_meta_eval_select_from_the_grid() -> None:
    import inspect

    from scripts.sim2rank import run_all_generations, sim2rank

    gen = inspect.getsource(run_all_generations)
    assert "rep_groups[: args.max_subjects]" not in gen
    assert "_discover_native_grid(" in gen
    meta = inspect.getsource(sim2rank.run_meta_evaluation_pipeline)
    assert "rep_groups[: args.max_subjects]" not in meta


def test_generations_cli_reads_and_stamps_allow_incomplete(monkeypatch) -> None:
    import inspect
    import sys

    from scripts.sim2rank import run_all_generations as gen

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_all_generations.py", "--output-dir", "/tmp/x", "--allow-incomplete-subjects"],
    )
    assert gen.parse_args().allow_incomplete_subjects is True
    src = inspect.getsource(gen)
    assert '"content_grid": content_grid' in src
    assert '"dropped_subjects": list(grid.dropped_subjects)' in src


def test_probe_cli_drops_and_prints_incomplete_subjects(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import argparse
    import inspect

    import torch

    from scripts.sim2rank import probe_metrics

    assert "--allow-incomplete-subjects" in inspect.getsource(probe_metrics.main)
    root = _tree(tmp_path, with_incomplete=True)
    monkeypatch.setattr(
        "scripts.sim2rank.ground_truth.synthesize_pseudo_gt",
        lambda group, device=None: (None, None, torch.ones((1, 1, 4, 4)), 1.0),
    )
    args = argparse.Namespace(
        synthetic=False,
        population=4,
        data_dir=str(root),
        dataset_kind="m4raw",
        contrasts=["T1", "FLAIR"],
        allow_incomplete_subjects=False,
    )
    with pytest.raises(ValueError, match="2022061004"):
        probe_metrics._load_reference(args, device="cpu")
    args.allow_incomplete_subjects = True
    assert len(probe_metrics._load_reference(args, device="cpu")) == 4  # rows x contrasts, capped
    assert "dropped incomplete subjects: ['2022061004']" in capsys.readouterr().out
