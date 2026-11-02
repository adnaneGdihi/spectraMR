"""Unit tests for the meta-evaluate clean-reference loader.

The loader's contract is strict by design (CLAUDE.md pitfalls #9, #10):

* ``--input`` omitted          → deterministic synthetic phantoms.
* ``--input`` set & missing    → :class:`FileNotFoundError`.
* ``--input`` set & empty      → :class:`RuntimeError`.
* ``--input`` set & .pt files  → direct tensor load.
* ``--input`` set & .h5 files  → M4Raw pseudo-GT, multi-contrast aware.

These tests fence the no-silent-fallback property and verify that the
H5 path tags volumes with their contrast (the on-disk grouping the
M4Raw training dataset also uses — ``<subject>_<contrast><rep>.h5``
→ group by stripping the last 2 chars of the stem).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from spectramr.cli.app import (
    _load_clean_volumes,
    _load_clean_volumes_from_h5,
    _load_clean_volumes_from_pt,
    _resolve_device,
    _synthesize_phantom_volumes,
    meta_evaluate,
)
from spectramr.core.compute_device import AcceleratorRequiredError


def test_missing_input_directory_raises_file_not_found(tmp_path: Path) -> None:
    """A non-existent ``--input`` must fail loud, not silently synthesize."""
    target = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="--input directory not found"):
        _load_clean_volumes(target, max_subjects=4)


def test_empty_input_directory_raises_runtime_error(tmp_path: Path) -> None:
    """A directory that exists but has no .pt/.h5 files must raise."""
    (tmp_path / "README.txt").write_text("nothing here", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no .pt or .h5 files"):
        _load_clean_volumes(tmp_path, max_subjects=4)


def test_pt_files_load_happy_path(tmp_path: Path) -> None:
    """`.pt` slices load directly, capped by max_subjects, in sorted order."""
    for i in range(3):
        torch.save(
            torch.ones((1, 8, 8)) * float(i),
            tmp_path / f"subject_{i:02d}.pt",
        )
    volumes = _load_clean_volumes_from_pt(tmp_path, max_subjects=2)
    assert [tag for tag, _ in volumes] == ["subject_00", "subject_01"]
    assert all(isinstance(t, torch.Tensor) for _, t in volumes)
    assert volumes[0][1].shape == (1, 8, 8)


def test_pt_files_accept_dict_with_image_key(tmp_path: Path) -> None:
    """Dict payloads with an ``image`` key unwrap cleanly."""
    torch.save({"image": torch.zeros((1, 4, 4)), "extra": 7}, tmp_path / "a.pt")
    volumes = _load_clean_volumes_from_pt(tmp_path, max_subjects=1)
    assert len(volumes) == 1
    assert volumes[0][0] == "a"
    assert volumes[0][1].shape == (1, 4, 4)


def test_phantom_fallback_only_when_input_omitted_explicitly() -> None:
    """The phantom path is reachable only by the helper, not the strict loader."""
    volumes = _synthesize_phantom_volumes(n=2, seed=7)
    assert [tag for tag, _ in volumes] == ["phantom_0", "phantom_1"]
    assert volumes[0][1].shape == (1, 64, 64)


def test_h5_loader_tags_volumes_with_subject_and_contrast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-contrast tagging: each (subject, contrast) becomes one volume.

    M4Raw on-disk layout: ``<subject>_<contrast><rep>.h5``. The sim2rank
    grouper strips the last 2 chars of the stem so a single group
    represents one (subject, contrast). The meta-evaluate H5 path
    reuses that convention so downstream figures can stratify per
    contrast.
    """
    # Fake H5 files covering 2 subjects x {T1, T2, FLAIR}. M4Raw grouping reads
    # names only, so the files need not be valid HDF5; the pseudo-GT is stubbed.
    subjects = ["2022091411", "2022091415"]
    contrasts = ["T1", "T2", "FLAIR"]
    reps = [1, 2]
    paths_by_group: dict[str, list[Path]] = {}
    for subj in subjects:
        for contrast in contrasts:
            group_key = f"{subj}_{contrast}"
            paths_by_group[group_key] = []
            for rep in reps:
                p = tmp_path / f"{subj}_{contrast}{rep:02d}.h5"
                p.write_bytes(b"fake-h5")
                paths_by_group[group_key].append(p)

    def fake_pseudo_gt(group, device=torch.device("cpu")):
        # Return shapes matching synthesize_pseudo_gt: (y, smaps, x_gt_mag, p99).
        x_gt_mag = torch.ones((1, 1, 8, 8), device=device) * float(len(group))
        return None, None, x_gt_mag, torch.tensor(1.0)

    monkeypatch.setattr(
        "spectramr.data.datasets.pseudo_gt_reference.synthesize_pseudo_gt",
        fake_pseudo_gt,
    )

    volumes = _load_clean_volumes_from_h5(
        tmp_path,
        max_subjects=2,
        max_contrasts=3,
    )

    tags = [tag for tag, _ in volumes]
    assert tags == [
        "2022091411_FLAIR",
        "2022091411_T1",
        "2022091411_T2",
        "2022091415_FLAIR",
        "2022091415_T1",
        "2022091415_T2",
    ]
    # x_gt_mag shape is (1, 1, H, W) → loader drops batch axis to (1, H, W).
    assert all(t.shape == (1, 8, 8) for _, t in volumes)


def test_h5_loader_contrast_whitelist_filters_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--contrasts T1 T2`` skips FLAIR without aborting the run."""
    subjects = ["s1"]
    contrasts = ["T1", "T2", "FLAIR"]
    groups: list[list[Path]] = []
    for subj in subjects:
        for contrast in contrasts:
            p = tmp_path / f"{subj}_{contrast}01.h5"
            p.write_bytes(b"fake-h5")
            groups.append([p])

    monkeypatch.setattr(
        "spectramr.data.datasets.pseudo_gt_reference.synthesize_pseudo_gt",
        lambda group, device=torch.device("cpu"): (
            None,
            None,
            torch.zeros((1, 1, 4, 4), device=device),
            torch.tensor(1.0),
        ),
    )

    volumes = _load_clean_volumes_from_h5(
        tmp_path,
        max_subjects=1,
        max_contrasts=3,
        contrasts=("T1", "T2"),
    )
    assert [tag for tag, _ in volumes] == ["s1_T1", "s1_T2"]


def test_loader_dispatches_pt_before_h5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both .pt and .h5 are present, .pt wins (no H5 cost)."""
    torch.save(torch.zeros((1, 4, 4)), tmp_path / "a.pt")
    (tmp_path / "b.h5").write_bytes(b"fake-h5")

    sentinel = {"called": False}

    def trip(*_a, **_k):
        sentinel["called"] = True
        raise AssertionError("H5 path must not run when .pt files exist")

    monkeypatch.setattr(
        "spectramr.data.datasets.native_grid.discover_native_grid",
        trip,
    )

    volumes = _load_clean_volumes(tmp_path, max_subjects=1)
    assert len(volumes) == 1
    assert sentinel["called"] is False


# ---------------------------------------------------------------------------
# eval_mode / betting run-shape knobs (pitfall #15: wired + validated + stamped)
# ---------------------------------------------------------------------------


def _meta_args(tmp_path: Path, **overrides):
    """Minimal argparse.Namespace for ``meta_evaluate`` (defaults = a 2d real run)."""
    import argparse

    base = {
        "input": None,
        "metrics": None,
        "severities": 2,
        "max_subjects": 1,
        "max_contrasts": 1,
        "contrasts": None,
        "synthetic_n": 2,
        "device": "cpu",
        "eval_mode": "2d",
        "betting": False,
        "betting_alpha": 0.05,
        "seed": 0,
        "out": tmp_path / "summary.json",
        "figures": False,
        "figures_dir": None,
        "tables": False,
        "tables_dir": None,
        "nr_battery": False,
        "tiers": None,
        "register_aggregator": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_meta_evaluate_3d_fails_loud_before_any_io(tmp_path: Path) -> None:
    """``--eval-mode 3d`` must raise at the CLI boundary, before loading data."""
    with pytest.raises(NotImplementedError, match="3d"):
        meta_evaluate(_meta_args(tmp_path, eval_mode="3d"))
    # And nothing was written.
    assert not (tmp_path / "summary.json").exists()


def test_meta_evaluate_rejects_bad_betting_alpha(tmp_path: Path) -> None:
    """An out-of-range α fails loud rather than silently clamping."""
    with pytest.raises(ValueError, match="betting-alpha"):
        meta_evaluate(_meta_args(tmp_path, betting=True, betting_alpha=1.5))


def test_meta_evaluate_synthetic_2d_betting_stamps_summary(tmp_path: Path) -> None:
    """Smoke the full synthetic 2-D + betting path; eval_mode + betting land in JSON."""
    import json

    rc = meta_evaluate(_meta_args(tmp_path, betting=True))
    assert rc == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["eval_mode"] == "2d"
    assert "betting" in summary and "aggregate" in summary["betting"]


def test_meta_evaluate_tiers_routes_to_full_harness(tmp_path: Path) -> None:
    """``--tiers`` routes to the §8 harness: cards JSON, not the bare ranking."""
    import json

    rc = meta_evaluate(
        _meta_args(
            tmp_path,
            tiers=["concordance", "fr_proxy"],
            metrics=["gri", "csr"],
            severities=3,
        )
    )
    assert rc == 0
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["research_mode"] is True
    assert result["tiers_run"] == ["concordance", "fr_proxy"]
    assert set(result["cards"]) == {"gri", "csr"}
    for card in result["cards"].values():
        assert set(card["tiers"]) == {"concordance", "fr_proxy"}


def test_meta_evaluate_tiers_writes_report_and_persists_model(tmp_path: Path) -> None:
    """``--tiers --tables`` emits the §8.8 cards CSV/MD and persists the model."""
    import json

    rc = meta_evaluate(
        _meta_args(
            tmp_path,
            tiers=["concordance", "fr_proxy"],
            metrics=["gri", "csr"],
            severities=3,
            tables=True,
        )
    )
    assert rc == 0
    # §8.8 report artifacts land next to the JSON.
    assert (tmp_path / "nr_cards.csv").exists()
    assert (tmp_path / "nr_cards.md").exists()
    # §8.7 fitted aggregator persisted (JSON-able, NRQualityIndexModel.from_dict-able).
    result = json.loads((tmp_path / "summary.json").read_text())
    model = result["aggregator_model"]
    assert model is not None and "weights" in model
    from spectramr.application.use_cases.nr_validation.aggregator import (
        NRQualityIndexModel,
    )

    restored = NRQualityIndexModel.from_dict(model)
    assert restored.feature_names == result["nr_metrics"]


def test_meta_evaluate_tiers_all_expands(tmp_path: Path) -> None:
    """``--tiers all`` expands to every advertised tier."""
    import json

    from spectramr.application.use_cases.nr_metric_validation_use_case import (
        HARNESS_TIERS,
    )

    rc = meta_evaluate(
        _meta_args(
            tmp_path,
            tiers=["all"],
            metrics=["gri"],
            severities=3,
        )
    )
    assert rc == 0
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["tiers_run"] == list(HARNESS_TIERS)


def test_meta_evaluate_tiers_rejects_unknown(tmp_path: Path) -> None:
    """An unknown --tiers value fails with a non-zero return code, no cards written."""
    rc = meta_evaluate(_meta_args(tmp_path, tiers=["bogus_tier"], metrics=["gri"]))
    assert rc == 1
    assert not (tmp_path / "summary.json").exists()


# ---------------------------------------------------------------------------
# Device resolution & device routing
# ---------------------------------------------------------------------------


def test_resolve_device_auto_without_accelerator_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``auto`` on a heavy pipeline (``_resolve_device`` defaults to ``evaluate``)
    must RAISE when no accelerator is visible, never silently downgrade to CPU —
    a silent CPU run burns the wall-clock allocation at ~100x while reporting
    success (accelerated-run contract, CLAUDE.md #9b / #10)."""
    monkeypatch.delenv("FORCE_CPU", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(AcceleratorRequiredError, match="requires an accelerator"):
        _resolve_device("auto")


def test_resolve_device_auto_cpu_opt_in_via_force_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one sanctioned CPU escape hatch: ``FORCE_CPU=true`` lets ``auto`` fall
    to CPU on an accelerator-less host (logged + stamped as an opt-in)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("FORCE_CPU", "true")
    assert _resolve_device("auto") == torch.device("cpu")


def test_resolve_device_explicit_cuda_without_cuda_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cuda`` must NOT silently downgrade to CPU — that masks a wasted GPU alloc.
    Not relaxed by FORCE_CPU: asking for cuda on a cuda-less host is a broken
    environment, not a preference."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(AcceleratorRequiredError, match="CUDA is not available"):
        _resolve_device("cuda")


def test_resolve_device_unknown_raises() -> None:
    with pytest.raises(ValueError, match="device='tpu' unrecognized"):
        _resolve_device("tpu")


def test_resolve_device_cpu_always_returns_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _resolve_device("cpu") == torch.device("cpu")


def test_pt_loader_moves_tensors_to_requested_device(tmp_path: Path) -> None:
    """The CPU smoke path of the device routing — exercised on every machine."""
    torch.save(torch.ones((1, 4, 4)), tmp_path / "subj.pt")
    volumes = _load_clean_volumes_from_pt(
        tmp_path, max_subjects=1, device=torch.device("cpu")
    )
    assert volumes[0][1].device.type == "cpu"


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA for the device-routing path"
)
def test_pt_loader_routes_to_cuda(tmp_path: Path) -> None:
    """When CUDA is available, .pt volumes land on the GPU."""
    torch.save(torch.ones((1, 4, 4)), tmp_path / "subj.pt")
    volumes = _load_clean_volumes_from_pt(
        tmp_path, max_subjects=1, device=torch.device("cuda")
    )
    assert volumes[0][1].device.type == "cuda"


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA for the simulator device-correctness path",
)
def test_simulator_runs_end_to_end_on_cuda() -> None:
    """Round-trip: every degradation family must produce a CUDA output
    when fed a CUDA clean — this is the property the simulator patches
    in this PR are protecting (CPU-pinned helpers used to cross-device
    crash on the first multiply)."""
    from spectramr.core.metrics.meta_evaluation.simulator import (
        SimulatorConfig,
        run_simulator,
    )

    clean = torch.rand((1, 32, 32), device="cuda")
    samples = run_simulator(
        [("subj_cuda", clean)],
        SimulatorConfig(n_severities_per_family=2, seed=0),
    )
    # 8 families × 2 severities = 16 samples expected
    assert len(samples) == 16
    for s in samples:
        assert (
            s.clean.device.type == "cuda"
        ), f"family {s.degradation_family}: clean is on {s.clean.device}, expected cuda"
        assert (
            s.degraded.device.type == "cuda"
        ), f"family {s.degradation_family}: degraded is on {s.degraded.device}, expected cuda"


def _brain_tree(root: Path) -> None:
    import h5py
    import numpy as np

    for pid in ("200_6002001", "200_6002002", "201_6002011"):
        with h5py.File(str(root / f"file_brain_AXT1_{pid}.h5"), "w") as f:
            f.create_dataset("kspace", data=np.zeros((1, 4, 4), dtype=np.complex64))
            f.attrs["acquisition"] = "AXT1"


def test_brain_filenames_are_never_nex_averaged_across_exams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C4: each brain exam is its own reference; the M4Raw stem law never runs on it."""
    _brain_tree(tmp_path)
    seen: list[int] = []

    def fake_pseudo_gt(group, device=None):
        seen.append(len(group))
        return None, None, torch.ones((1, 1, 4, 4)), torch.tensor(1.0)

    monkeypatch.setattr(
        "spectramr.data.datasets.pseudo_gt_reference.synthesize_pseudo_gt", fake_pseudo_gt
    )
    grid: dict[str, object] = {}
    volumes = _load_clean_volumes_from_h5(
        tmp_path,
        max_subjects=5,
        max_contrasts=1,
        contrasts=("AXT1",),
        dataset_kind="fastmri_brain",
        grid_out=grid,
    )
    assert seen == [1, 1, 1]
    assert [t for t, _ in volumes] == [
        "200_6002001_AXT1",
        "200_6002002_AXT1",
        "201_6002011_AXT1",
    ]
    assert grid["grid_linked"] is False and grid["row_label"] == "index"
    assert grid["min_reps"] == 1 and len(grid["contents"]) == 3
    with pytest.raises(ValueError, match="not an M4Raw"):
        _load_clean_volumes_from_h5(tmp_path, max_subjects=5, dataset_kind="m4raw")


def test_contrasts_apply_before_the_subject_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M2: ``--contrasts T2`` with three subjects yields three T2 volumes, not one."""
    for subj in ("s1", "s2", "s3"):
        for c in ("FLAIR", "T1", "T2"):
            for r in (1, 2):
                (tmp_path / f"{subj}_{c}{r:02d}.h5").write_bytes(b"")
    monkeypatch.setattr(
        "spectramr.data.datasets.pseudo_gt_reference.synthesize_pseudo_gt",
        lambda group, device=None: (None, None, torch.ones((1, 1, 4, 4)), torch.tensor(1.0)),
    )
    volumes = _load_clean_volumes_from_h5(
        tmp_path, max_subjects=3, max_contrasts=1, contrasts=("T2",)
    )
    assert [t for t, _ in volumes] == ["s1_T2", "s2_T2", "s3_T2"]


def test_unknown_dataset_kind_is_rejected_by_the_one_dispatch(tmp_path: Path) -> None:
    (tmp_path / "s1_T101.h5").write_bytes(b"")
    with pytest.raises(ValueError, match="Unknown dataset_kind"):
        _load_clean_volumes_from_h5(tmp_path, max_subjects=1, dataset_kind="knee")


def test_skipped_contents_are_stamped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for subj in ("s1", "s2"):
        for r in (1, 2):
            (tmp_path / f"{subj}_T1{r:02d}.h5").write_bytes(b"")

    def flaky(group, device=None):
        if "s2" in group[0].name:
            raise RuntimeError("bad file")
        return None, None, torch.ones((1, 1, 4, 4)), torch.tensor(1.0)

    monkeypatch.setattr("spectramr.data.datasets.pseudo_gt_reference.synthesize_pseudo_gt", flaky)
    grid: dict[str, object] = {}
    _load_clean_volumes_from_h5(tmp_path, max_subjects=2, max_contrasts=1, grid_out=grid)
    assert [r["content_id"] for r in grid["contents"]] == ["s1_T1"]
    assert grid["skipped_content_ids"] == ["s2_T1"]
