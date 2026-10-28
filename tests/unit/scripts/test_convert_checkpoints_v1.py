"""The v1 converter on files the pre-1.0 writers actually wrote (Task 1 fixtures)."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch

from spectramr.infrastructure.checkpointing import codecs
from spectramr.infrastructure.checkpointing.capture import check_strategy_models
from spectramr.infrastructure.checkpointing.resume import resume
from tests.unit.infrastructure.checkpointing.conftest import (
    FakeNativeAdapter,
    make_pipeline,
    native_runtime,
)
from tests.unit.scripts.convert_v1_helpers import (
    EXPECTED,
    ck_names,
    converter,
    copy_run,
    listing,
    resume_cfg,
)

FIXTURE_RUNS = ["director_run", "deepspeed_run", "service_pth_run", "service_safetensors_run"]


@pytest.fixture
def run(tmp_path):
    return lambda name: copy_run(tmp_path, name)


def test_director_run_converts_to_step_and_best_files(run) -> None:
    run_dir = run("director_run")
    assert converter().main([str(run_dir)]) == 0
    assert ck_names(run_dir) == [
        "best.safetensors",
        "step_00000002.safetensors",
        "step_00000004.safetensors",
    ]
    meta = codecs.read(run_dir / "checkpoints" / "best.safetensors")["meta"]
    assert meta["best_metric"] == {"name": "val_psnr", "value": 21.5}
    assert meta["conversion"]["envelope"] == "director"
    assert set(meta["conversion"]["dropped"]) >= {"metrics", "counter_state"}
    assert meta["native_tag"] is None


@pytest.mark.parametrize(
    "name", ["checkpoint_epoch_0000_step_000002.pt", "checkpoint_epoch_0001_step_000004.pt"]
)
def test_converted_step_resumes_at_the_same_step_with_the_same_rng_stream(run, name) -> None:
    """Review Focus 5: an in-flight carya run, converted, then resumed."""
    run_dir = run("director_run")
    converter().main([str(run_dir)])
    expected = EXPECTED[f"director_run/{name}"]
    step = expected["step"]
    result = resume(
        run_dir / "checkpoints" / f"step_{step:08d}.safetensors",
        make_pipeline(),
        strategy=None,
        parallel=None,
        config=resume_cfg(),
    )
    assert (result.epoch, result.step) == (expected["epoch"], step)
    assert torch.rand(3).tolist() == expected["draws"]["torch"]
    assert np.random.rand(3).tolist() == expected["draws"]["numpy"]
    assert random.random() == expected["draws"]["python"]


def test_best_metric_seeds_early_stopping_when_absent(run) -> None:
    run_dir = run("director_run")
    converter().main([str(run_dir)])
    seeded = codecs.read(run_dir / "checkpoints" / "step_00000002.safetensors")["early_stopping"]
    assert seeded == {"monitor": "val_psnr", "best_value": 21.5, "best_iteration": 4}
    kept = codecs.read(run_dir / "checkpoints" / "step_00000004.safetensors")["early_stopping"]
    assert kept["wait_count"] == 0 and kept["mode"] == "max"


def test_an_early_stopped_run_stays_stopped(run) -> None:
    """Without the stop signal a converted run would resume and train its remaining budget."""
    run_dir = run("director_run")
    path = run_dir / "checkpoints" / "checkpoint_epoch_0001_step_000004.pt"
    payload = torch.load(path, weights_only=False)
    payload["early_stopping_state"] |= {"stop_signal": True, "wait_count": 3}
    torch.save(payload, path)
    assert converter().main([str(run_dir)]) == 0
    state = codecs.read(run_dir / "checkpoints" / "step_00000004.safetensors")["early_stopping"]
    assert state["stop_signal"] is True and state["wait_count"] == 3


def test_the_ema_counter_is_split_out(run) -> None:
    run_dir = run("director_run")
    converter().main([str(run_dir)])
    state = codecs.read(run_dir / "checkpoints" / "step_00000004.safetensors")
    assert "_ema_num_updates" not in state["ema"]["generator"] and state["ema"]["num_updates"] == 1
    assert state["meta"]["conversion"]["ema_num_updates"] == "recorded"


def test_format_pt_replaces_the_alias_with_the_converted_best(run) -> None:
    run_dir = run("director_run")
    assert converter().main([str(run_dir), "--format", "pt"]) == 0
    best = run_dir / "checkpoints" / "best.pt"
    assert best.is_file() and not best.is_symlink()
    assert codecs.read(best)["meta"]["role"] == "best"
    assert ck_names(run_dir) == ["best.pt", "step_00000002.pt", "step_00000004.pt"]


def test_deepspeed_run_folds_the_sidecar_and_renames_the_tags(run) -> None:
    run_dir = run("deepspeed_run")
    assert converter().main([str(run_dir)]) == 0
    assert ck_names(run_dir) == ["best.safetensors", "deepspeed", "step_00000004.safetensors"]
    best = codecs.read(run_dir / "checkpoints" / "best.safetensors")
    assert best["meta"]["conversion"]["envelope"] == "consolidated"
    assert (best["meta"]["step"], best["meta"]["native_tag"]) == (4, "best")
    assert best["meta"]["best_metric"] == {"name": "val_psnr", "value": 21.5}
    assert set(best["ema"]["generator"]) == {"weight", "bias"}
    assert best["ema"]["num_updates"] == 1
    assert best["meta"]["conversion"]["ema_num_updates"] == (
        "from checkpoint_epoch_0001_step_000004.pt"
    )
    # Inherited from the step file the same run recorded, not the CLI default.
    assert best["meta"]["topology"] == {"world_size": 1, "parallel_strategy": "deepspeed"}
    tags = run_dir / "checkpoints" / "deepspeed" / "generator"
    assert sorted(p.name for p in tags.iterdir()) == ["best", "latest", "step_00000004"]
    assert (tags / "latest").read_text() == "step_00000004"


def test_deepspeed_engine_optimizer_and_scheduler_stay_in_the_tag(run) -> None:
    """PF1: the payload's engine optimizer is dropped; the plain discriminator's is kept."""
    run_dir = run("deepspeed_run")
    converter().main([str(run_dir)])
    state = codecs.read(run_dir / "checkpoints" / "step_00000004.safetensors")
    assert state["meta"]["native_tag"] == "step_00000004"
    assert set(state["optim"]) == {"discriminator"} and "sched" not in state
    assert {"optimizer_g", "scheduler_g"} <= set(state["meta"]["conversion"]["dropped"])


def test_deepspeed_step_resumes_from_its_tag_with_the_same_rng_stream(run) -> None:
    run_dir = run("deepspeed_run")
    converter().main([str(run_dir)])
    expected = EXPECTED["deepspeed_run/checkpoint_epoch_0001_step_000004.pt"]
    native = FakeNativeAdapter(frozenset({"generator"}))
    result = resume(
        run_dir / "checkpoints" / "step_00000004.safetensors",
        make_pipeline(),
        strategy=None,
        parallel=native_runtime(native),
        config=resume_cfg(),
    )
    assert (native.seen["tag"], native.seen["checkpoint_dir"]) == (
        "step_00000004",
        run_dir / "checkpoints",
    )
    assert result.step == expected["step"]
    assert torch.rand(3).tolist() == expected["draws"]["torch"]


@pytest.mark.parametrize("fmt", ["pth", "safetensors"])
def test_service_runs_convert_with_both_optimizers(run, fmt) -> None:
    run_dir = run(f"service_{fmt}_run")
    assert converter().main([str(run_dir)]) == 0
    assert ck_names(run_dir) == ["best.safetensors", "step_00000004.safetensors"]
    state = codecs.read(run_dir / "checkpoints" / "step_00000004.safetensors")
    assert set(state["optim"]) == {"generator", "discriminator"}
    assert set(state["models"]["generator"]) == {"weight", "bias"}
    assert state["ema"]["num_updates"] == 1
    assert state["meta"]["conversion"]["envelope"] == "service"
    best = codecs.read(run_dir / "checkpoints" / "best.safetensors")["meta"]
    # The safetensors writer dropped the metric name; manifest.json kept it.
    assert best["best_metric"] == {"name": "val_psnr", "value": 21.5}


@pytest.mark.parametrize("name", FIXTURE_RUNS)
def test_every_converted_file_holds_only_pipeline_models(run, name) -> None:
    """Any other models key reads as a stage, and a GAN or diffusion resume refuses it."""
    run_dir = run(name)
    assert converter().main([str(run_dir)]) == 0
    for path in sorted((run_dir / "checkpoints").glob("*.safetensors")):
        models = codecs.read(path)["models"]
        assert set(models) <= {"generator", "discriminator"}
        check_strategy_models(models, {}, source=path)


def test_topology_flags_apply_only_when_the_file_has_none(run) -> None:
    run_dir = run("service_pth_run")
    converter().main([str(run_dir), "--world-size", "4", "--parallel-strategy", "ddp"])
    meta = codecs.read(run_dir / "checkpoints" / "step_00000004.safetensors")["meta"]
    assert meta["topology"] == {"world_size": 4, "parallel_strategy": "ddp"}
    assert meta["conversion"]["topology_assumed"] is True
    run_dir = run("director_run")
    converter().main([str(run_dir), "--world-size", "4"])
    meta = codecs.read(run_dir / "checkpoints" / "step_00000004.safetensors")["meta"]
    assert meta["topology"]["world_size"] == 1  # the director recorded it
    assert meta["conversion"]["topology_assumed"] is False


def test_wrapper_prefixes_are_stripped(run) -> None:
    run_dir = run("director_run")
    path = run_dir / "checkpoints" / "checkpoint_epoch_0001_step_000004.pt"
    payload = torch.load(path, weights_only=False)
    payload["generator"] = {f"module.{k}": v for k, v in payload["generator"].items()}
    torch.save(payload, path)
    converter().main([str(run_dir)])
    models = codecs.read(run_dir / "checkpoints" / "step_00000004.safetensors")["models"]
    assert set(models["generator"]) == {"weight", "bias"}


def test_dry_run_changes_nothing(run, capsys) -> None:
    run_dir = run("deepspeed_run")
    before = listing(run_dir)
    assert converter().main([str(run_dir), "--dry-run"]) == 0
    assert listing(run_dir) == before
    assert "dry run" in capsys.readouterr().out


def test_a_second_run_is_a_no_op(run) -> None:
    run_dir = run("director_run")
    converter().main([str(run_dir)])
    after = ck_names(run_dir)
    assert converter().main([str(run_dir)]) == 0
    assert ck_names(run_dir) == after


def test_a_multi_stage_run_is_named_in_the_report(run, capsys) -> None:
    run_dir = run("director_run")
    (run_dir / "resolved_config.json").write_text(
        json.dumps({"training": {"training_mode": "multi"}})
    )
    assert converter().main([str(run_dir)]) == 0
    assert "#2962" in capsys.readouterr().out


def test_conversion_records_what_it_seeded_kept_and_dropped(run) -> None:
    """Concerns the report raised, written into the file so a reader of it can tell."""
    director = run("director_run")
    converter().main([str(director)])
    ck = director / "checkpoints"
    assert (
        codecs.read(ck / "step_00000002.safetensors")["meta"]["conversion"]["early_stopping"]
        == "seeded from checkpoint_best.pt"
    )
    assert (
        codecs.read(ck / "step_00000004.safetensors")["meta"]["conversion"]["early_stopping"]
        == "recorded"
    )
    service = run("service_pth_run")
    converter().main([str(service)])
    conv = codecs.read(service / "checkpoints" / "step_00000004.safetensors")["meta"]["conversion"]
    assert {"metric_name", "metric_value", "loss"} <= set(conv["dropped"])
    assert conv["notes"] == ["discriminator optimizer kept without discriminator weights"]
    deepspeed = run("deepspeed_run")
    converter().main([str(deepspeed)])
    best = codecs.read(deepspeed / "checkpoints" / "best.safetensors")["meta"]["conversion"]
    assert "sidecar.metrics" in best["dropped"]


def test_a_service_safetensors_step_lists_its_metric_value_as_dropped(run) -> None:
    """The safetensors writer kept the value but not the name; the value is still dropped."""
    run_dir = run("service_safetensors_run")
    converter().main([str(run_dir)])
    conv = codecs.read(run_dir / "checkpoints" / "step_00000004.safetensors")["meta"]["conversion"]
    assert {"metric_value", "loss"} <= set(conv["dropped"])
