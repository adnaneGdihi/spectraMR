"""PF7 and its neighbours: a run the converter cannot convert cleanly is left exactly as it was."""

from __future__ import annotations

import os
import shutil

import pytest
import torch

from spectramr.infrastructure.checkpointing import codecs
from tests.unit.scripts.convert_v1_helpers import ck_names, converter, copy_run, listing


@pytest.fixture
def run(tmp_path):
    return lambda name: copy_run(tmp_path, name)


def _corrupt_reads(monkeypatch) -> None:
    real_read = codecs.read

    def corrupt(path):
        state = real_read(path)
        state["models"]["generator"]["weight"] = state["models"]["generator"]["weight"] + 1
        return state

    monkeypatch.setattr(codecs, "read", corrupt)


@pytest.mark.parametrize("name", ["director_run", "deepspeed_run", "service_safetensors_run"])
def test_a_round_trip_mismatch_leaves_the_run_untouched(run, monkeypatch, capsys, name) -> None:
    run_dir = run(name)
    before = listing(run_dir)
    _corrupt_reads(monkeypatch)
    assert converter().main([str(run_dir)]) == 1
    assert "RoundTripMismatchError" in capsys.readouterr().err
    # Sources, aliases, sidecar, manifest, tag directories and `latest`: byte for byte.
    assert listing(run_dir) == before


def test_a_mismatch_on_the_last_target_also_removes_the_verified_ones(run, monkeypatch) -> None:
    """best and step 2 stage and verify before step 4 fails; none of the three may remain."""
    run_dir = run("director_run")
    real_read, seen = codecs.read, []

    def corrupt_last(path):
        seen.append(path.name)
        state = real_read(path)
        if path.name.startswith("step_00000004"):
            state["rng"]["python"] = list(state["rng"]["python"])  # tuple -> list
        return state

    monkeypatch.setattr(codecs, "read", corrupt_last)
    before = listing(run_dir)
    assert converter().main([str(run_dir)]) == 1
    assert seen == [
        "best.converting.safetensors",
        "step_00000002.converting.safetensors",
        "step_00000004.converting.safetensors",
    ]
    assert listing(run_dir) == before


def test_a_mismatch_in_one_run_does_not_stop_the_next(run, monkeypatch, capsys) -> None:
    good, bad = run("service_pth_run"), run("director_run")
    real_read = codecs.read

    def corrupt_director(path):
        state = real_read(path)
        if "director_run" in str(path):
            state["meta"]["step"] += 1
        return state

    monkeypatch.setattr(codecs, "read", corrupt_director)
    before = listing(bad)
    assert converter().main([str(bad), str(good)]) == 1
    assert "meta.step" in capsys.readouterr().err
    assert listing(bad) == before
    assert ck_names(good) == ["best.safetensors", "step_00000004.safetensors"]


@pytest.mark.parametrize("link", [shutil.copy2, os.link])
@pytest.mark.parametrize("fmt", ["safetensors", "pt"])
def test_a_regular_file_alias_is_never_a_second_best(run, link, fmt) -> None:
    """The old writers fell back to a hardlink, then a copy, where symlinks failed."""
    run_dir = run("director_run")
    ck = run_dir / "checkpoints"
    (ck / "best.pt").unlink()
    link(ck / "checkpoint_best.pt", ck / "best.pt")
    assert converter().main([str(run_dir), "--format", fmt]) == 0
    assert ck_names(run_dir) == [f"best.{fmt}", f"step_00000002.{fmt}", f"step_00000004.{fmt}"]
    meta = codecs.read(ck / f"best.{fmt}")["meta"]
    assert meta["conversion"]["from"] == "checkpoint_best.pt"


def test_a_last_file_equal_to_its_periodic_twin_is_retired(run) -> None:
    run_dir = run("service_pth_run")
    ck = run_dir / "checkpoints"
    shutil.copy2(ck / "checkpoint_step_4.pth", ck / "checkpoint_last.pth")
    assert converter().main([str(run_dir)]) == 0
    assert ck_names(run_dir) == ["best.safetensors", "step_00000004.safetensors"]
    conv = codecs.read(ck / "step_00000004.safetensors")["meta"]["conversion"]
    assert (conv["from"], conv["duplicates"]) == ("checkpoint_step_4.pth", ["checkpoint_last.pth"])


def test_two_different_files_for_one_step_are_refused(run, capsys) -> None:
    run_dir = run("service_pth_run")
    ck = run_dir / "checkpoints"
    payload = torch.load(ck / "checkpoint_step_4.pth", weights_only=False)
    payload["model_state_dict"]["bias"] = payload["model_state_dict"]["bias"] + 1
    torch.save(payload, ck / "checkpoint_last.pth")
    before = listing(run_dir)
    assert converter().main([str(run_dir)]) == 1
    assert "checkpoint_last.pth vs step_00000004" in capsys.readouterr().err
    assert listing(run_dir) == before


def test_an_unrecognised_pre_v1_file_refuses_the_run(run, capsys) -> None:
    run_dir = run("director_run")
    torch.save({"x": 1}, run_dir / "checkpoints" / "checkpoint_mystery.pt")
    before = listing(run_dir)
    assert converter().main([str(run_dir)]) == 1
    assert "checkpoint_mystery.pt" in capsys.readouterr().err
    assert listing(run_dir) == before


def test_an_existing_schema_1_file_is_never_overwritten(run, capsys) -> None:
    run_dir = run("director_run")
    (run_dir / "checkpoints" / "step_00000004.safetensors").write_bytes(b"not mine")
    before = listing(run_dir)
    assert converter().main([str(run_dir)]) == 1
    assert "FileExistsError" in capsys.readouterr().err
    assert listing(run_dir) == before


def test_a_stale_sidecar_is_not_folded(run, capsys) -> None:
    run_dir = run("deepspeed_run")
    sidecar = run_dir / "checkpoints" / "checkpoint_best_ema.pt"
    payload = torch.load(sidecar, weights_only=False)
    payload["global_step"] = 2
    torch.save(payload, sidecar)
    before = listing(run_dir)
    assert converter().main([str(run_dir)]) == 1
    assert "stale EMA" in capsys.readouterr().err
    assert listing(run_dir) == before


def test_a_crashed_attempts_staging_files_are_cleared(run) -> None:
    run_dir = run("director_run")
    staging = run_dir / "checkpoints" / ".convert_v1_staging"
    staging.mkdir()
    (staging / "step_00000004.converting.safetensors").write_bytes(b"partial")
    (staging / "tmpk2j4x9.tmp").write_bytes(b"killed mid-write")  # atomic_publish's temp
    (run_dir / "checkpoints" / "tmpuser.tmp").write_bytes(b"not the converter's")
    assert converter().main([str(run_dir)]) == 0
    assert not staging.exists()
    assert (run_dir / "checkpoints" / "tmpuser.tmp").read_bytes() == b"not the converter's"
