"""pt and pth: one torch.save container, loaded weights-only."""

from __future__ import annotations

import pickle
import random

import numpy as np
import pytest
import torch

from spectramr.infrastructure.checkpointing import codecs
from spectramr.infrastructure.checkpointing.payload import SchemaVersionError


@pytest.mark.parametrize("fmt", ["pt", "pth"])
def test_round_trip_keeps_every_part(tmp_path, run_state, fmt) -> None:
    state = run_state(fmt)
    path = tmp_path / f"step_00000150.{fmt}"
    codecs.write(path, state)
    back = codecs.read(path)
    assert back["meta"] == state["meta"]
    for k, v in state["models"]["generator"].items():
        assert torch.equal(back["models"]["generator"][k], v)
    assert back["optim"]["generator"]["param_groups"] == state["optim"]["generator"]["param_groups"]
    assert back["sched"] == state["sched"]
    random.setstate(back["rng"]["python"])
    np.random.set_state(back["rng"]["numpy"])
    torch.set_rng_state(back["rng"]["torch"])


def test_a_pre_1_0_envelope_is_refused_by_name(tmp_path) -> None:
    path = tmp_path / "best.pt"
    torch.save({"generator": {}, "optimizer_g": {}, "epoch": 1}, path)
    with pytest.raises(SchemaVersionError, match="convert_checkpoints_v1"):
        codecs.read(path)


def test_an_object_torch_will_not_load_weights_only_names_the_converter(tmp_path) -> None:
    path = tmp_path / "best.pt"
    torch.save({"meta": {"schema_version": 1}, "models": {}, "x": object()}, path)
    with pytest.raises(SchemaVersionError, match="convert_checkpoints_v1") as caught:
        codecs.read(path)
    assert isinstance(caught.value.__cause__, pickle.UnpicklingError)
    # Its keys cannot be read, so it may as well be a foreign file: say both.
    assert "external weights" in str(caught.value)


def test_a_legacy_envelope_with_numpy_metrics_names_the_converter(tmp_path) -> None:
    path = tmp_path / "best.pt"
    torch.save(
        {
            "epoch": 3,
            "generator": {},
            "metrics": {"psnr": np.float64(30.1)},
            "rng_state": {"numpy": np.random.get_state()},
        },
        path,
    )
    with pytest.raises(SchemaVersionError, match="convert_checkpoints_v1"):
        codecs.read(path)
