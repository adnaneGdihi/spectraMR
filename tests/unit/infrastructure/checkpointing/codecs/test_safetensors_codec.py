"""safetensors holds the whole RunState: tensors by name, everything else as JSON."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from spectramr.infrastructure.checkpointing import codecs
from spectramr.infrastructure.checkpointing.payload import (
    CONVERTER,
    NotARunStateError,
    SchemaVersionError,
)


@pytest.fixture
def round_trip(tmp_path, run_state):
    state = run_state("safetensors")
    path = tmp_path / "step_00000150.safetensors"
    codecs.write(path, state)
    return state, codecs.read(path)


def test_tensors_come_back_equal(round_trip) -> None:
    state, back = round_trip
    for k, v in state["models"]["generator"].items():
        assert torch.equal(back["models"]["generator"][k], v)
    assert torch.equal(back["strategy"]["params"]["sigma"], torch.tensor(0.5))


def test_optimizer_state_keeps_int_keys_and_hyperparameters(round_trip) -> None:
    state, back = round_trip
    want, got = state["optim"]["generator"], back["optim"]["generator"]
    assert got["param_groups"] == want["param_groups"]
    assert sorted(got["state"]) == sorted(want["state"])  # ints, not "0", "1"
    for idx, slots in want["state"].items():
        for name, value in slots.items():
            assert torch.equal(got["state"][idx][name], value)


def test_rng_state_is_accepted_by_every_generator(round_trip) -> None:
    _, back = round_trip
    py, npy = back["rng"]["python"], back["rng"]["numpy"]
    assert isinstance(py, tuple) and isinstance(py[1], tuple)
    assert isinstance(npy, tuple) and isinstance(npy[1], np.ndarray)
    assert npy[1].dtype == np.uint32
    random.setstate(py)
    np.random.set_state(npy)
    torch.set_rng_state(back["rng"]["torch"])


def test_rng_streams_continue_identically(tmp_path, run_state) -> None:
    random.seed(3)
    np.random.seed(3)
    state = run_state("safetensors")
    state["rng"] = {"numpy": np.random.get_state(), "python": random.getstate()}
    expected = (random.random(), np.random.rand())
    path = tmp_path / "best.safetensors"
    state["meta"] = run_state("safetensors", role="best")["meta"]
    codecs.write(path, state)
    back = codecs.read(path)
    random.setstate(back["rng"]["python"])
    np.random.set_state(back["rng"]["numpy"])
    assert (random.random(), np.random.rand()) == expected


def test_tied_weights_are_written_and_stay_tied(round_trip) -> None:
    _, back = round_trip
    tied = back["models"]["stage:tied"]
    assert torch.equal(tied["embed.weight"], tied["head.weight"])
    assert tied["embed.weight"] is tied["head.weight"]


def test_partially_overlapping_views_are_copied_not_refused(tmp_path, run_state) -> None:
    base = torch.arange(10.0)
    state = run_state("safetensors")
    state["strategy"] = {"a": base[:6], "b": base[4:]}
    path = tmp_path / "step_00000150.safetensors"
    codecs.write(path, state)
    back = codecs.read(path)
    assert torch.equal(back["strategy"]["a"], base[:6])
    assert torch.equal(back["strategy"]["b"], base[4:])


def test_non_finite_floats_survive(round_trip) -> None:
    _, back = round_trip
    assert math.isnan(back["early_stopping"]["best_value"])
    assert back["ema"]["num_updates"] == 7


def test_unsupported_values_are_refused_by_path(tmp_path, run_state) -> None:
    state = run_state("safetensors")
    state["strategy"] = {"fn": print}
    with pytest.raises(TypeError, match="strategy"):
        codecs.write(tmp_path / "step_00000150.safetensors", state)


def test_a_headerless_safetensors_file_is_refused_by_name(tmp_path) -> None:
    path = tmp_path / "best.safetensors"
    save_file({"model_state_dict_w": torch.zeros(2)}, str(path))  # the pre-1.0 service shape
    with pytest.raises(SchemaVersionError, match="convert_checkpoints_v1"):
        codecs.read(path)


def test_a_headerless_foreign_state_dict_is_named_foreign(tmp_path) -> None:
    path = tmp_path / "pretrained_encoder.safetensors"
    save_file({"encoder.conv.weight": torch.zeros(2)}, str(path))
    with pytest.raises(NotARunStateError, match="not a spectramr RunState") as caught:
        codecs.read(path)
    assert CONVERTER not in str(caught.value)


def test_a_headerless_file_under_a_pre_1_0_name_names_both_remedies(tmp_path) -> None:
    path = tmp_path / "checkpoint_best.safetensors"
    save_file({"encoder.conv.weight": torch.zeros(2)}, str(path))
    with pytest.raises(SchemaVersionError, match=CONVERTER) as caught:
        codecs.read(path)
    assert "external weights" in str(caught.value)


def test_a_failed_write_leaves_no_file(tmp_path, run_state) -> None:
    state = run_state("safetensors")
    state["strategy"] = {"fn": print}
    path = tmp_path / "step_00000150.safetensors"
    with pytest.raises(TypeError):
        codecs.write(path, state)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "value",
    [torch.zeros(2, dtype=torch.complex128), 1j, np.complex64(1j)],
    ids=["complex128-tensor", "complex-scalar", "numpy-complex-scalar"],
)
def test_values_safetensors_cannot_hold_are_refused_with_path_and_way_out(
    tmp_path, run_state, value
) -> None:
    state = run_state("safetensors")
    state["strategy"] = {"x": value}
    with pytest.raises(TypeError, match=r"strategy\.x.*format.*pt"):
        codecs.write(tmp_path / "step_00000150.safetensors", state)
