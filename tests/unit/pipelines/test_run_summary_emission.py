"""Smoke test for the Phase 4 ``run_summary.json`` artefact.

Every successful training run must drop a self-describing JSON footer
into the run output dir so downstream tooling (mosaic, ranking,
dispatch logs) can introspect the run without re-parsing the source
YAML. The emit helper soft-fails by design -- the test exercises the
happy path; failure modes are logged but do not abort training.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.utils.config_block_stub import block_stub

#: ``logging`` routed through the ELECTED stub, not hand-rolled.
#:
#: ``_emit_run_summary`` reads ``config.logging.identity.run``; these stubs set a
#: FLAT ``run_name`` attribute, so the read raised ``AttributeError`` -- and the
#: helper soft-fails by design, so it logged and wrote nothing. The test then
#: failed on "run_summary.json must land in the run output dir", naming the
#: symptom and not the moved leaf. 3 of cluster job 8004252's failures.
#:
#: ``run_name`` was the right legacy spelling all along; it just was not ROUTED.
#: ``block_stub`` resolves it through ``FLAT_TO_CANONICAL`` -> ``identity.run``,
#: which is the whole reason that helper exists (an unrouted kwarg lands as a
#: stray attribute while the reader keeps its default).
#:
#: The two sites that passed did so only because their ``provenance`` supplied
#: ``run_name`` and short-circuited the ``or``. Routed here too -- a test that
#: passes by never reaching the read is not evidence the read works.


def test_run_summary_stamp_round_trip(tmp_path: Path) -> None:
    """``_emit_run_summary`` writes a JSON with the expected keys."""
    from spectramr.pipelines.train import _emit_run_summary

    logger_stub = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
    )
    config = SimpleNamespace(
        logging=block_stub("logging", run_name="exp_smoke"),
        training=SimpleNamespace(
            strategy_class="spectramr.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy"
        ),
        model=SimpleNamespace(model_type="latent_diffusion_residual_head"),
        metadata=None,
    )
    result = {"success": True, "best_metrics": {"psnr": 32.4}}

    _emit_run_summary(config, run_dir=tmp_path, result=result, seed=42, logger_=logger_stub)
    path = tmp_path / "run_summary.json"
    assert path.exists(), "run_summary.json must land in the run output dir"

    parsed = json.loads(path.read_text())
    assert parsed["run_name"] == "exp_smoke"
    assert parsed["seed"] == 42
    assert parsed["success"] is True
    assert parsed["strategy_class"].endswith("DiffusionTrainingStrategy")
    assert parsed["model_type"] == "latent_diffusion_residual_head"
    assert parsed["best_metrics"] == {"psnr": 32.4}
    assert parsed["error"] is None
    assert parsed["launch"] is None  # not started via `spectramr launch` → no-op


def test_run_summary_folds_in_provenance_and_timing(tmp_path: Path) -> None:
    """With a provenance record + started_at, the footer becomes a full
    traceability record: run_id, git, host, config hash, and wall-clock."""
    from datetime import datetime

    from spectramr.pipelines.train import _emit_run_summary

    logger_stub = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    config = SimpleNamespace(
        logging=block_stub("logging", run_name="exp_smoke"),
        training=SimpleNamespace(strategy_class="X.DiffusionTrainingStrategy"),
        model=SimpleNamespace(model_type="latent_diffusion_residual_head"),
        metadata=None,
    )
    provenance = {
        "run_id": "exp_smoke-20260611_200500-abc123",
        "run_name": "exp_smoke",
        "config_sha256": "deadbeef0123",
        "git": {"sha_short": "abc123", "branch": "dev", "dirty": True},
        "env": {"hostname": "node07"},
        "batch": {"effective": 16},
        "model": {"total": 12_345_678},
        "slurm": {"SLURM_JOB_ID": "999"},
        "launch": {"backend": "slurm", "account": "test_alloc", "gpus": 2},
    }
    # The loop fills these onto result; run_training_pipeline overwrites
    # training_time with the real wall-clock.
    result = {
        "success": True,
        "training_time": "01:02:05",
        "duration_sec": 3725.0,
        "iterations_per_sec": 4.2,
        "train_iterations_per_sec": 9.5,
        "final_loss": 0.12,
    }
    started = datetime(2026, 6, 11, 20, 5, 0).astimezone()

    _emit_run_summary(
        config,
        run_dir=tmp_path,
        result=result,
        seed=42,
        logger_=logger_stub,
        provenance=provenance,
        started_at=started,
    )
    parsed = json.loads((tmp_path / "run_summary.json").read_text())
    assert parsed["run_id"] == "exp_smoke-20260611_200500-abc123"
    assert parsed["config_sha256"] == "deadbeef0123"
    assert parsed["git"]["dirty"] is True
    assert parsed["host"] == "node07"
    assert parsed["effective_batch"] == 16
    assert parsed["model_params"] == 12_345_678
    assert parsed["duration"] == "01:02:05"
    assert parsed["duration_sec"] == 3725.0
    assert parsed["iterations_per_sec"] == 4.2
    assert parsed["train_iterations_per_sec"] == 9.5
    assert parsed["final_loss"] == 0.12
    assert parsed["started_at"].startswith("2026-06-11T20:05:00")
    assert parsed["ended_at"]  # stamped at write time
    assert parsed["slurm"] == {"SLURM_JOB_ID": "999"}
    # the launcher backend + resolved resources are stamped (pitfall #15c)
    assert parsed["launch"] == {"backend": "slurm", "account": "test_alloc", "gpus": 2}
    # No node hardware in this record → the key is omitted, not stamped as nulls.
    assert parsed["hardware"] is None


def test_run_summary_stamps_node_hardware(tmp_path: Path) -> None:
    """``iterations_per_sec`` is not comparable across arms without the GPU
    count/type and the cores/RAM the run actually had, so the footer carries a
    compact census of ``env.node`` (2026-07-25)."""
    from spectramr.pipelines.train import _emit_run_summary

    logger_stub = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    config = SimpleNamespace(
        logging=block_stub("logging", run_name="exp_smoke"),
        training=SimpleNamespace(strategy_class="X.DiffusionTrainingStrategy"),
        model=SimpleNamespace(model_type="unet"),
        metadata=None,
    )
    provenance = {
        "run_id": "exp_smoke-20260725_120000-abc123",
        "env": {
            "hostname": "node07",
            "node": {
                "gpu": {
                    "count": 2,
                    "types": {"NVIDIA A100-SXM4-40GB": 2},
                    "node_count": 4,
                    "driver_version": "550.54.15",
                    "total_mem_gb": 80.0,
                },
                "cpu": {
                    "model": "AMD EPYC 7763",
                    "usable_cores": 8,
                    "logical_cores": 128,
                },
                "memory": {"usable_gb": 64.0, "total_gb": 1007.5},
            },
        },
    }

    _emit_run_summary(
        config,
        run_dir=tmp_path,
        result={"success": True, "iterations_per_sec": 4.2},
        seed=42,
        logger_=logger_stub,
        provenance=provenance,
    )
    hw = json.loads((tmp_path / "run_summary.json").read_text())["hardware"]
    assert hw["gpu_count"] == 2
    assert hw["gpu_types"] == {"NVIDIA A100-SXM4-40GB": 2}
    assert hw["gpu_node_count"] == 4, "the node had 4; this job was given 2"
    assert hw["gpu_driver_version"] == "550.54.15"
    assert hw["cpu_cores_usable"] == 8
    assert hw["cpu_cores_total"] == 128
    assert hw["memory_usable_gb"] == 64.0
    assert hw["memory_total_gb"] == 1007.5


def test_run_summary_records_failure(tmp_path: Path) -> None:
    """A failed-run result must surface the error string in the summary."""
    from spectramr.config.schemas.base import ExperimentMetadataSchema
    from spectramr.pipelines.train import _emit_run_summary

    logger_stub = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
    )
    config = SimpleNamespace(
        logging=block_stub("logging", run_name=None),
        # `training` genuinely may be None — train.py guards it explicitly.
        training=None,
        # `model` may NOT. It is a required field on TrainingSettings, and #370
        # dropped the getattr fallback around `config.model.model_type`
        # (train.py:779) accordingly. `model=None` here modelled a config that
        # cannot exist, and because _emit_run_summary soft-fails by design the
        # resulting AttributeError was swallowed into "no summary file written"
        # rather than a visible error.
        model=SimpleNamespace(model_type="unet"),
        metadata=ExperimentMetadataSchema(name="fallback_name"),
    )
    result = {"success": False, "error": "boom"}

    _emit_run_summary(config, run_dir=tmp_path, result=result, seed=0, logger_=logger_stub)
    parsed = json.loads((tmp_path / "run_summary.json").read_text())
    assert parsed["run_name"] == "fallback_name"
    assert parsed["success"] is False
    assert parsed["error"] == "boom"


def test_run_summary_soft_fails_on_bad_dir(tmp_path: Path) -> None:
    """A path-creation failure must not raise -- training wrap-up must not abort.

    Point the helper at a file rather than a directory; ``mkdir`` will
    fail but the helper must swallow the exception with a warning.
    """
    from spectramr.pipelines.train import _emit_run_summary

    warnings: list[str] = []

    def _capture_warning(*args, **_kwargs):
        warnings.append(args[0] if args else "")

    logger_stub = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=_capture_warning,
    )
    bad_path = tmp_path / "not_a_dir"
    bad_path.write_text("blocking file")  # exists as a file, not a dir
    config = SimpleNamespace(
        logging=block_stub("logging", run_name="x"),
        training=None,
        model=None,
        metadata=None,
    )

    # Must not raise.
    _emit_run_summary(
        config, run_dir=bad_path, result={"success": True}, seed=0, logger_=logger_stub
    )
    assert warnings, "soft-fail must emit a warning so the issue is discoverable"


def test_a_failed_best_weights_restore_is_stamped(tmp_path: Path) -> None:
    """restore_best_weights that failed leaves the LAST weights in the final checkpoint;
    the summary says so instead of the warning being the only record."""
    from spectramr.pipelines.train import _emit_run_summary

    logger_stub = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    config = SimpleNamespace(
        logging=block_stub("logging", run_name="r"),
        training=SimpleNamespace(strategy_class="x.Y"),
        model=SimpleNamespace(model_type="m"),
        metadata=None,
    )
    for restored in (True, False, None):
        result = {"success": True, "best_weights_restored": restored}
        _emit_run_summary(config, run_dir=tmp_path, result=result, seed=1, logger_=logger_stub)
        parsed = json.loads((tmp_path / "run_summary.json").read_text())
        assert parsed["best_weights_restored"] is restored


@pytest.mark.parametrize("validate_only, warned", [(False, True), (True, False)])
def test_empty_checkpoint_dir_warns_only_for_a_run_that_trains(
    tmp_path: Path, validate_only: bool, warned: bool
) -> None:
    """#503 flags a training run that saved nothing; a validate-only run never saves."""
    from spectramr.pipelines.train import _emit_run_summary

    warnings: list[str] = []
    logger_stub = SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: warnings.append(a[0])
    )
    (tmp_path / "checkpoints").mkdir()
    config = SimpleNamespace(
        logging=block_stub("logging", run_name="r"),
        training=SimpleNamespace(strategy_class="x.Y"),
        model=SimpleNamespace(model_type="m"),
        metadata=None,
        checkpoint=SimpleNamespace(enabled=True),
    )
    result = {"success": True, "validate_only": validate_only}
    _emit_run_summary(config, run_dir=tmp_path, result=result, seed=1, logger_=logger_stub)
    assert any("#503" in w for w in warnings) is warned
