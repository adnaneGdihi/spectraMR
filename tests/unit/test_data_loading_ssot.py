"""Phase-4 data-loading SSOT regression tests.

Locks ``TODO/backlog_ssot_and_layering_cleanup.md`` Phase 4 + CLAUDE.md
pitfall #11: every file → tensor transition lives in :mod:`spectramr.data`
or under a designated builder SSOT.

The five sites the audit flagged are all now routed through the SSOT:

- ``src/pipelines/make.py:make_dataloader`` — delegates to ``DataPipelineDirector``.
- ``src/pipelines/parallel.py:_apply_distributed_samplers`` — delegates to
  ``spectramr.data.builders.data_loader_builder.wrap_with_distributed_sampler``.
- ``src/cli/app.py:_load_clean_volumes_from_pt`` — delegates to
  ``spectramr.data.datasets.meta_eval_clean_refs.load_clean_volumes``.
- ``src/cli/app.py:_load_clean_volumes_from_h5`` — imports
  ``synthesize_pseudo_gt`` from ``spectramr.data.datasets.pseudo_gt_reference``
  instead of ``scripts.sim2rank.ground_truth``; the physics module opens no file.
- ``src/infrastructure/inference/streaming_inference.py`` — uses
  ``spectramr.data.io_strategies.lazy_load_nifti`` instead of ``nibabel.load``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest


def test_pipelines_make_no_longer_instantiates_dataloader_directly() -> None:
    """``make_dataloader`` delegates to the director, not raw ``DataLoader(...)``."""
    from spectramr.pipelines import make

    src = inspect.getsource(make.make_dataloader)
    # Check for the CALL (with paren), not docstring mentions.
    assert "torch.utils.data.DataLoader(" not in src
    assert "DataPipelineDirector" in src


def test_pipelines_parallel_no_longer_instantiates_dataloader_directly() -> None:
    """``_apply_distributed_samplers`` delegates to the builder SSOT."""
    from spectramr.pipelines import parallel

    src = inspect.getsource(parallel._apply_distributed_samplers)
    # Allow type-annotation mentions; check for the constructor call only.
    assert "DataLoader(\n" not in src and " DataLoader(" not in src
    assert "wrap_with_distributed_sampler" in src


def test_cli_meta_eval_pt_loader_delegates_to_dataset_ssot() -> None:
    """The CLI's ``.pt`` clean-refs path imports from the canonical dataset."""
    from spectramr.cli import app

    src = inspect.getsource(app._load_clean_volumes_from_pt)
    # No raw torch.load here any more.
    assert "torch.load(" not in src
    assert "load_clean_volumes" in src
    assert "spectramr.data.datasets.meta_eval_clean_refs" in src


def test_cli_h5_loader_imports_pseudo_gt_from_the_data_layer() -> None:
    """The CLI's H5 path imports ``synthesize_pseudo_gt`` from ``data/`` (rule 7)."""
    from spectramr.cli import app

    src = inspect.getsource(app._load_clean_volumes_from_h5)
    assert (
        "from spectramr.data.datasets.pseudo_gt_reference import synthesize_pseudo_gt"
        in src
    )


def test_synthesize_pseudo_gt_canonical_home() -> None:
    """``synthesize_pseudo_gt`` lives in ``data/``; the physics module keeps no file read."""
    import spectramr.infrastructure.physics.m4raw_pseudo_gt as physics
    from spectramr.data.datasets.pseudo_gt_reference import synthesize_pseudo_gt

    assert synthesize_pseudo_gt.__module__ == "spectramr.data.datasets.pseudo_gt_reference"
    assert "h5py" not in inspect.getsource(physics)
    assert not hasattr(physics, "synthesize_pseudo_gt")


def test_scripts_ground_truth_is_thin_shim() -> None:
    """``scripts.sim2rank.ground_truth`` re-exports the canonical function."""
    pytest.importorskip("scripts.sim2rank.ground_truth")  # not in the public export
    from scripts.sim2rank.ground_truth import (
        synthesize_pseudo_gt as legacy,
    )
    from spectramr.data.datasets.pseudo_gt_reference import (
        synthesize_pseudo_gt as canonical,
    )

    assert legacy is canonical


def test_streaming_inference_uses_io_strategies_for_nifti() -> None:
    """``streaming_inference.py`` no longer imports ``nibabel`` directly."""
    from spectramr.infrastructure.inference import streaming_inference

    src = inspect.getsource(streaming_inference)
    assert "import nibabel" not in src
    assert "lazy_load_nifti" in src


def test_meta_eval_dataset_round_trips_pt_files(tmp_path: Path) -> None:
    """End-to-end smoke: build the dataset on a tmp dir of ``.pt`` files."""
    import torch

    from spectramr.data.datasets.meta_eval_clean_refs import load_clean_volumes

    for i in range(3):
        torch.save(torch.randn(1, 4, 4), tmp_path / f"subj_{i}.pt")

    volumes = load_clean_volumes(tmp_path, max_subjects=10)
    assert len(volumes) == 3
    for sid, t in volumes:
        assert sid.startswith("subj_")
        assert t.shape == (1, 4, 4)
        assert t.dtype is torch.float32


def test_meta_eval_dataset_handles_dict_payloads(tmp_path: Path) -> None:
    """Dict payloads with ``image`` / ``clean`` keys are unwrapped."""
    import torch

    from spectramr.data.datasets.meta_eval_clean_refs import load_clean_volumes

    torch.save({"image": torch.randn(1, 4, 4)}, tmp_path / "with_image.pt")
    torch.save({"clean": torch.randn(1, 4, 4)}, tmp_path / "with_clean.pt")
    torch.save({"other_key": torch.randn(1, 4, 4)}, tmp_path / "fallback.pt")

    volumes = load_clean_volumes(tmp_path, max_subjects=10)
    assert len(volumes) == 3


def test_meta_eval_dataset_skips_non_tensor_payloads(tmp_path: Path) -> None:
    """Files whose payload isn't a tensor (after dict-unwrap) are silently skipped."""
    import torch

    from spectramr.data.datasets.meta_eval_clean_refs import load_clean_volumes

    torch.save({"image": torch.randn(1, 4, 4)}, tmp_path / "ok.pt")
    torch.save({"image": "not_a_tensor"}, tmp_path / "bad.pt")

    volumes = load_clean_volumes(tmp_path, max_subjects=10)
    assert len(volumes) == 1


def test_meta_eval_dataset_raises_on_missing_dir(tmp_path: Path) -> None:
    """Missing directory raises ``FileNotFoundError`` (no silent fallback)."""
    from spectramr.data.datasets.meta_eval_clean_refs import MetaEvalCleanRefsDataset

    with pytest.raises(FileNotFoundError):
        MetaEvalCleanRefsDataset(tmp_path / "does_not_exist")


def test_layering_lint_phase4_check_passes() -> None:
    """The lint script's ``no direct DataLoader outside src/data/`` check is green."""
    import subprocess

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", str(repo_root / "scripts" / "ci" / "check_layering.sh")],
        capture_output=True, text=True, timeout=60,
    )
    output = result.stdout + result.stderr
    assert "PASS  no direct DataLoader outside src/data/" in output, output
