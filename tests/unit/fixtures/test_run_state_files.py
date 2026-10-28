"""The shared RunState writer: models and EMA by default, any other part on request."""

from pathlib import Path

import pytest
import torch

from spectramr.infrastructure.checkpointing import codecs
from tests.fixtures.run_state_files import write_run_state


def test_the_default_file_holds_meta_and_models_only(tmp_path: Path) -> None:
    path = write_run_state(tmp_path / "best.pt", {"generator": {"w": torch.ones(1)}})
    assert set(codecs.read(path)) == {"meta", "models"}


@pytest.mark.parametrize("suffix", ["pt", "safetensors"])
def test_extra_parts_are_written_beside_models_and_ema(tmp_path: Path, suffix: str) -> None:
    optim = {"state": {}, "param_groups": [{"lr": 0.1, "params": [0]}]}
    sched = {"last_epoch": 4}
    path = write_run_state(
        tmp_path / f"step_00000010.{suffix}",
        {"generator": {"w": torch.ones(1)}},
        ema={"w": torch.zeros(1)},
        role="step",
        extra_parts={"optim": {"generator": optim}, "sched": {"generator": sched}},
    )
    state = codecs.read(path)
    assert {"meta", "models", "ema", "optim", "sched"} <= set(state)
    assert state["sched"]["generator"]["last_epoch"] == 4
    assert state["optim"]["generator"]["param_groups"][0]["lr"] == 0.1


def test_extra_parts_cannot_overwrite_meta_models_or_ema(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="models"):
        write_run_state(tmp_path / "best.pt", {"generator": {}}, extra_parts={"models": {"x": {}}})
