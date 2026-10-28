"""Content the converter cannot place is refused, never dropped; one test per shape."""

from __future__ import annotations

import json
import shutil

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from spectramr.infrastructure.checkpointing import codecs
from tests.unit.scripts.convert_v1_helpers import ck_names, converter, copy_run, listing


@pytest.fixture
def run(tmp_path):
    return lambda name: copy_run(tmp_path, name)


def _rewrite_safetensors(path, edit) -> None:
    with safe_open(str(path), framework="pt") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118
    edit(tensors)
    save_file(tensors, str(path))


def _assert_refused(run_dir, capsys, reason: str) -> None:
    before = listing(run_dir)
    assert converter().main([str(run_dir)]) == 1
    assert reason in capsys.readouterr().err
    assert listing(run_dir) == before


def test_a_bare_state_dict_under_a_step_name_is_refused(run, capsys) -> None:
    """It would otherwise take the best's step and EMA, and its source would be deleted."""
    run_dir = run("deepspeed_run")
    ck = run_dir / "checkpoints"
    (ck / "checkpoint_epoch_0001_step_000004.pt").unlink()  # so no dedupe refuses first
    shutil.copy2(ck / "checkpoint_best.pt", ck / "checkpoint_epoch_0000_step_000002.pt")
    _assert_refused(run_dir, capsys, "bare state dict")


def test_a_name_that_disagrees_with_its_payload_step_is_refused(run, capsys) -> None:
    run_dir = run("director_run")
    ck = run_dir / "checkpoints"
    (ck / "checkpoint_epoch_0000_step_000002.pt").rename(
        ck / "checkpoint_epoch_0000_step_000003.pt"
    )
    _assert_refused(run_dir, capsys, "names step 3 but holds step 2")


def test_an_unknown_director_key_is_refused(run, capsys) -> None:
    run_dir = run("director_run")
    path = run_dir / "checkpoints" / "checkpoint_epoch_0001_step_000004.pt"
    payload = torch.load(path, weights_only=False)
    payload["mystery_state"] = {"x": torch.ones(1)}
    torch.save(payload, path)
    _assert_refused(run_dir, capsys, "mystery_state")


def test_an_unknown_extra_optimizer_is_refused(run, capsys) -> None:
    run_dir = run("service_pth_run")
    path = run_dir / "checkpoints" / "checkpoint_step_4.pth"
    payload = torch.load(path, weights_only=False)
    payload["extra_optimizer_states"]["critic"] = payload["extra_optimizer_states"]["discriminator"]
    torch.save(payload, path)
    _assert_refused(run_dir, capsys, "extra_optimizer_states.critic")


def test_an_unknown_safetensors_key_is_refused(run, capsys) -> None:
    run_dir = run("service_safetensors_run")
    path = run_dir / "checkpoints" / "checkpoint_step_4.safetensors"
    _rewrite_safetensors(path, lambda t: t.update(mystery_tensor=torch.zeros(2)))
    _assert_refused(run_dir, capsys, "safetensors:mystery_tensor")


def test_an_optimizer_without_param_groups_is_refused(run, capsys) -> None:
    run_dir = run("service_safetensors_run")
    path = run_dir / "checkpoints" / "checkpoint_step_4.safetensors"
    _rewrite_safetensors(path, lambda t: t.pop("_meta_optimizer_state_dict_param_groups"))
    _assert_refused(run_dir, capsys, "has no param_groups")


def test_the_older_param_groups_spelling_is_the_main_optimizer(run) -> None:
    """The deleted loader read ``_meta_optimizer_param_groups`` as the generator's groups."""
    run_dir = run("service_safetensors_run")
    ck = run_dir / "checkpoints"
    original = None
    for name in ("checkpoint_step_4.safetensors", "checkpoint_best.safetensors"):

        def respell(tensors):
            nonlocal original
            groups = tensors.pop("_meta_optimizer_state_dict_param_groups")
            original = json.loads(groups.numpy().tobytes())
            tensors["_meta_optimizer_param_groups"] = groups

        _rewrite_safetensors(ck / name, respell)
    assert converter().main([str(run_dir)]) == 0
    assert ck_names(run_dir) == ["best.safetensors", "step_00000004.safetensors"]
    optim = codecs.read(ck / "step_00000004.safetensors")["optim"]["generator"]
    assert optim["param_groups"] == original


def _v1_best(tmp_path):
    source = copy_run(tmp_path / "v1", "service_pth_run")
    assert converter().main([str(source), "--format", "pt"]) == 0
    return source / "checkpoints" / "best.pt"


@pytest.mark.parametrize("fmt", ["pt", "safetensors"])
def test_a_schema_1_best_beside_the_legacy_best_is_refused(run, tmp_path, capsys, fmt) -> None:
    """C1: a real v1 best.pt is never taken for the alias, overwritten or swept."""
    run_dir = run("director_run")
    alias = run_dir / "checkpoints" / "best.pt"
    alias.unlink()
    shutil.copy2(_v1_best(tmp_path), alias)
    before = listing(run_dir)
    assert converter().main([str(run_dir), "--format", fmt]) == 1
    assert "a schema-1 best already exists" in capsys.readouterr().err
    assert listing(run_dir) == before


def test_a_regular_best_that_is_not_the_alias_is_refused(run, capsys) -> None:
    run_dir = run("director_run")
    alias = run_dir / "checkpoints" / "best.pt"
    alias.unlink()
    alias.write_bytes(b"something else")
    _assert_refused(run_dir, capsys, "is not the alias")


def test_a_missing_run_dir_exits_2(tmp_path, capsys) -> None:
    assert converter().main([str(tmp_path / "no_such_run")]) == 2
    captured = capsys.readouterr()
    assert "does not exist" in captured.err and "nothing to convert" not in captured.out


def test_a_checkpoints_dir_exits_2_naming_its_parent(run, capsys) -> None:
    run_dir = run("director_run")
    before = listing(run_dir)
    assert converter().main([str(run_dir / "checkpoints")]) == 2
    captured = capsys.readouterr()
    assert str(run_dir) in captured.err and "nothing to convert" not in captured.out
    assert listing(run_dir) == before


def test_an_existing_directory_that_is_no_run_exits_2(tmp_path, capsys) -> None:
    """A typo'd run path that happens to exist is the same mistake as a missing one."""
    (tmp_path / "experiments").mkdir()
    assert converter().main([str(tmp_path / "experiments")]) == 2
    captured = capsys.readouterr()
    assert "not a run directory" in captured.err and "nothing to convert" not in captured.out


def test_a_run_with_nothing_to_convert_exits_0(tmp_path, capsys) -> None:
    (tmp_path / "arm" / "checkpoints").mkdir(parents=True)
    assert converter().main([str(tmp_path / "arm")]) == 0
    assert "nothing to convert" in capsys.readouterr().out
