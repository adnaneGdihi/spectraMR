"""A stage loads its own stage:<name> part when the file has one, else the generator."""

from pathlib import Path

import pytest
import torch

from spectramr.infrastructure.training.strategies.pipeline_strategy import MultiTrainingStrategy
from tests.fixtures.run_state_files import write_run_state


def _strategy() -> MultiTrainingStrategy:
    return MultiTrainingStrategy.__new__(MultiTrainingStrategy)


def test_a_named_stage_part_wins(tmp_path: Path) -> None:
    a, b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    path = write_run_state(
        tmp_path / "best.pt", {"generator": a.state_dict(), "stage:enc": b.state_dict()}
    )
    target = torch.nn.Linear(2, 2)
    _strategy()._load_stage_weights(target, str(path), "enc")
    assert torch.equal(target.weight, b.weight)


def test_without_a_stage_part_the_generator_is_used(tmp_path: Path) -> None:
    a = torch.nn.Linear(2, 2)
    path = write_run_state(tmp_path / "best.pt", {"generator": a.state_dict()})
    target = torch.nn.Linear(2, 2)
    _strategy()._load_stage_weights(target, str(path), "enc")
    assert torch.equal(target.weight, a.weight)


def test_zero_overlap_now_raises(tmp_path: Path) -> None:
    path = write_run_state(tmp_path / "best.pt", {"generator": {"other.w": torch.zeros(1)}})
    with pytest.raises(RuntimeError, match="zero parameter names"):
        _strategy()._load_stage_weights(torch.nn.Linear(2, 2), str(path), "enc")


def test_a_stage_declaring_a_stage1_vae_is_refused() -> None:
    """Stages are built by GeneratorBuilder, which never reads ``vae_checkpoint_path``;
    accepting the knob would train the stage in a random latent space (NN3/NN8)."""
    from types import SimpleNamespace

    stage = SimpleNamespace(
        name="ldm",
        model_type="latent_diffusion",
        model_kwargs={"vae_checkpoint_path": "stage1/best.safetensors"},
    )
    with pytest.raises(ValueError, match="vae_checkpoint_path"):
        _strategy()._create_stage_model(stage)
