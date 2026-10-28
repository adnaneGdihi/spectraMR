"""The dice_risk synthesiser is read from the generator part of a RunState, strictly."""

from pathlib import Path

import pytest
import torch

from spectramr.infrastructure.checkpointing.restore import CheckpointPartMissingError
from spectramr.infrastructure.training.strategies.conformal_calibration_strategy import (
    ConformalCalibrationStrategy,
)
from tests.fixtures.run_state_files import write_run_state

_load = ConformalCalibrationStrategy._load_generator_checkpoint


def test_the_generator_part_is_loaded(tmp_path: Path) -> None:
    src = torch.nn.Linear(2, 2)
    path = write_run_state(tmp_path / "best.safetensors", {"generator": src.state_dict()})
    target = torch.nn.Linear(2, 2)
    _load(target, str(path))
    assert torch.equal(target.weight, src.weight)


def test_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dice_risk checkpoint not found"):
        _load(torch.nn.Linear(2, 2), str(tmp_path / "absent.pt"))


def test_a_file_holding_no_generator_raises(tmp_path: Path) -> None:
    path = write_run_state(tmp_path / "best.pt", {"stage:enc": torch.nn.Linear(2, 2).state_dict()})
    with pytest.raises(CheckpointPartMissingError):
        _load(torch.nn.Linear(2, 2), str(path))


def test_a_partial_state_is_refused_not_half_loaded(tmp_path: Path) -> None:
    """``exact``: the certificate must be computed on the trained synthesiser, whole."""
    path = write_run_state(tmp_path / "best.pt", {"generator": {"weight": torch.zeros(2, 2)}})
    with pytest.raises(RuntimeError, match="bias"):
        _load(torch.nn.Linear(2, 2), str(path))
