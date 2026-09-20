"""Tests for checkpoint resume state persistence.

Validates that AMP GradScaler state and iteration counter state
are correctly saved/loaded during checkpoint save/load roundtrips.

Also verifies backward compatibility with old checkpoints that
do not contain scaler_state or counter_state keys.
"""

import copy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


class _SimpleGenerator(nn.Module):
    """Minimal generator for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _FakePipeline:
    """Lightweight stand-in for TrainingPipeline."""

    def __init__(
        self,
        *,
        device: torch.device,
        generator: nn.Module,
        discriminator: nn.Module | None = None,
        optimizer_g: torch.optim.Optimizer | None = None,
        optimizer_d: torch.optim.Optimizer | None = None,
        scheduler_g: Any | None = None,
        scheduler_d: Any | None = None,
        scaler: torch.amp.GradScaler | None = None,
    ) -> None:
        self.device = device
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_g = optimizer_g or torch.optim.Adam(
            generator.parameters(), lr=1e-3
        )
        self.optimizer_d = optimizer_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.scaler = scaler


@pytest.fixture()
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture()
def generator(device: torch.device) -> nn.Module:
    return _SimpleGenerator().to(device)


@pytest.fixture()
def pipeline(generator: nn.Module, device: torch.device) -> _FakePipeline:
    return _FakePipeline(device=device, generator=generator)


@pytest.fixture()
def pipeline_with_scaler(generator: nn.Module, device: torch.device) -> _FakePipeline:
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    return _FakePipeline(device=device, generator=generator, scaler=scaler)


@pytest.fixture()
def tmp_checkpoint_dir(tmp_path: Path) -> str:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    return str(ckpt_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCheckpointSavesScalerState:
    """Verify scaler_state is persisted in checkpoint files."""

    def test_checkpoint_saves_scaler_state(
        self, pipeline_with_scaler: _FakePipeline, tmp_checkpoint_dir: str
    ) -> None:
        """Checkpoint file must contain 'scaler_state' when scaler is set."""
        from spectramr.infrastructure.builders.directors.checkpoint_director import (
            CheckpointDirector,
        )

        director = CheckpointDirector.__new__(CheckpointDirector)
        director._config = MagicMock()
        director._checkpoint_dir = Path(tmp_checkpoint_dir)
        director._pipeline = pipeline_with_scaler
        director._epoch = 1
        director._global_step = 100
        director._metrics = {}
        director._scaler = pipeline_with_scaler.scaler
        director._counter_state = None
        director._checkpoint_path = None
        director._product = None
        director._is_validated = True

        saved_path = director.save()
        checkpoint = torch.load(saved_path, map_location="cpu", weights_only=False)

        assert "scaler_state" in checkpoint
        assert isinstance(checkpoint["scaler_state"], dict)

    def test_checkpoint_without_scaler_has_no_key(
        self, pipeline: _FakePipeline, tmp_checkpoint_dir: str
    ) -> None:
        """Checkpoint must NOT contain 'scaler_state' when scaler is None."""
        from spectramr.infrastructure.builders.directors.checkpoint_director import (
            CheckpointDirector,
        )

        director = CheckpointDirector.__new__(CheckpointDirector)
        director._config = MagicMock()
        director._checkpoint_dir = Path(tmp_checkpoint_dir)
        director._pipeline = pipeline
        director._epoch = 0
        director._global_step = 0
        director._metrics = {}
        director._scaler = None
        director._counter_state = None
        director._checkpoint_path = None
        director._product = None
        director._is_validated = True

        saved_path = director.save()
        checkpoint = torch.load(saved_path, map_location="cpu", weights_only=False)

        assert "scaler_state" not in checkpoint


class TestCheckpointRestoresScalerState:
    """Verify scaler state roundtrip save→load identity."""

    def test_scaler_state_roundtrip(
        self, pipeline_with_scaler: _FakePipeline, tmp_checkpoint_dir: str
    ) -> None:
        """Scaler state_dict must match after save→load roundtrip."""
        from spectramr.infrastructure.builders.directors.checkpoint_director import (
            CheckpointDirector,
        )

        original_scaler_state = copy.deepcopy(pipeline_with_scaler.scaler.state_dict())

        # Save
        save_director = CheckpointDirector.__new__(CheckpointDirector)
        save_director._config = MagicMock()
        save_director._checkpoint_dir = Path(tmp_checkpoint_dir)
        save_director._pipeline = pipeline_with_scaler
        save_director._epoch = 2
        save_director._global_step = 500
        save_director._metrics = {"loss": 0.5}
        save_director._scaler = pipeline_with_scaler.scaler
        save_director._counter_state = None
        save_director._checkpoint_path = None
        save_director._product = None
        save_director._is_validated = True

        saved_path = save_director.save()

        # Create fresh pipeline with new scaler
        fresh_gen = _SimpleGenerator()
        fresh_scaler = torch.amp.GradScaler("cpu", enabled=True)
        fresh_pipeline = _FakePipeline(
            device=torch.device("cpu"),
            generator=fresh_gen,
            scaler=fresh_scaler,
        )

        # Load
        load_director = CheckpointDirector.__new__(CheckpointDirector)
        load_director._config = MagicMock()
        load_director._checkpoint_dir = Path(tmp_checkpoint_dir)
        load_director._pipeline = fresh_pipeline
        load_director._epoch = 0
        load_director._global_step = 0
        load_director._metrics = {}
        load_director._scaler = None
        load_director._counter_state = None
        load_director._checkpoint_path = None
        load_director._product = None
        load_director._is_validated = True

        success = load_director.load_from(str(saved_path))

        assert success is True
        restored_state = fresh_pipeline.scaler.state_dict()
        assert restored_state == original_scaler_state


class TestCheckpointSavesCounterState:
    """Verify counter_state is persisted."""

    def test_counter_state_saved(
        self, pipeline: _FakePipeline, tmp_checkpoint_dir: str
    ) -> None:
        """Checkpoint must contain 'counter_state' when set."""
        from spectramr.infrastructure.builders.directors.checkpoint_director import (
            CheckpointDirector,
        )

        counter_state = {"current_step": 1234, "current_epoch": 5}

        director = CheckpointDirector.__new__(CheckpointDirector)
        director._config = MagicMock()
        director._checkpoint_dir = Path(tmp_checkpoint_dir)
        director._pipeline = pipeline
        director._epoch = 5
        director._global_step = 1234
        director._metrics = {}
        director._scaler = None
        director._counter_state = counter_state
        director._checkpoint_path = None
        director._product = None
        director._is_validated = True

        saved_path = director.save()
        checkpoint = torch.load(saved_path, map_location="cpu", weights_only=False)

        assert "counter_state" in checkpoint
        assert checkpoint["counter_state"]["current_step"] == 1234
        assert checkpoint["counter_state"]["current_epoch"] == 5


class TestBackwardCompatibility:
    """Old checkpoints without scaler_state/counter_state must load cleanly."""

    def test_old_checkpoint_loads_without_scaler(
        self, pipeline_with_scaler: _FakePipeline, tmp_checkpoint_dir: str
    ) -> None:
        """Loading an old checkpoint without scaler_state must not crash."""
        from spectramr.infrastructure.builders.directors.checkpoint_director import (
            CheckpointDirector,
        )

        # Create a "legacy" checkpoint (no scaler_state, no counter_state)
        legacy_path = Path(tmp_checkpoint_dir) / "checkpoint_epoch_0001_step_000050.pt"
        legacy_data = {
            "epoch": 1,
            "global_step": 50,
            "generator": pipeline_with_scaler.generator.state_dict(),
            "optimizer_g": pipeline_with_scaler.optimizer_g.state_dict(),
            "metrics": {},
        }
        torch.save(legacy_data, legacy_path)

        # Load into a pipeline WITH scaler — should not crash
        load_director = CheckpointDirector.__new__(CheckpointDirector)
        load_director._config = MagicMock()
        load_director._checkpoint_dir = Path(tmp_checkpoint_dir)
        load_director._pipeline = pipeline_with_scaler
        load_director._epoch = 0
        load_director._global_step = 0
        load_director._metrics = {}
        load_director._scaler = None
        load_director._counter_state = None
        load_director._checkpoint_path = None
        load_director._product = None
        load_director._is_validated = True

        success = load_director.load_from(str(legacy_path))

        assert success is True
        assert load_director._global_step == 50
        assert load_director._counter_state is None  # Not in old checkpoint

    def test_old_checkpoint_loads_without_counter(
        self, pipeline: _FakePipeline, tmp_checkpoint_dir: str
    ) -> None:
        """Loading an old checkpoint without counter_state must not crash."""
        from spectramr.infrastructure.builders.directors.checkpoint_director import (
            CheckpointDirector,
        )

        legacy_path = Path(tmp_checkpoint_dir) / "checkpoint_epoch_0002_step_000100.pt"
        legacy_data = {
            "epoch": 2,
            "global_step": 100,
            "generator": pipeline.generator.state_dict(),
            "optimizer_g": pipeline.optimizer_g.state_dict(),
            "metrics": {"loss": 0.3},
        }
        torch.save(legacy_data, legacy_path)

        load_director = CheckpointDirector.__new__(CheckpointDirector)
        load_director._config = MagicMock()
        load_director._checkpoint_dir = Path(tmp_checkpoint_dir)
        load_director._pipeline = pipeline
        load_director._epoch = 0
        load_director._global_step = 0
        load_director._metrics = {}
        load_director._scaler = None
        load_director._counter_state = None
        load_director._checkpoint_path = None
        load_director._product = None
        load_director._is_validated = True

        success = load_director.load_from(str(legacy_path))

        assert success is True
        assert load_director._global_step == 100
        assert load_director._counter_state is None


class TestCheckpointStartIteration:
    """Verify that start_iteration is correctly extracted on resume."""

    def test_global_step_recovered(
        self, pipeline: _FakePipeline, tmp_checkpoint_dir: str
    ) -> None:
        """global_step must be recoverable after load_from()."""
        from spectramr.infrastructure.builders.directors.checkpoint_director import (
            CheckpointDirector,
        )

        # Save at step 5000
        save_director = CheckpointDirector.__new__(CheckpointDirector)
        save_director._config = MagicMock()
        save_director._checkpoint_dir = Path(tmp_checkpoint_dir)
        save_director._pipeline = pipeline
        save_director._epoch = 10
        save_director._global_step = 5000
        save_director._metrics = {"loss": 0.2}
        save_director._scaler = None
        save_director._counter_state = {
            "current_step": 5000,
            "current_epoch": 10,
        }
        save_director._checkpoint_path = None
        save_director._product = None
        save_director._is_validated = True

        saved_path = save_director.save()

        # Load into fresh pipeline
        fresh_gen = _SimpleGenerator()
        fresh_pipeline = _FakePipeline(
            device=torch.device("cpu"),
            generator=fresh_gen,
        )

        load_director = CheckpointDirector.__new__(CheckpointDirector)
        load_director._config = MagicMock()
        load_director._checkpoint_dir = Path(tmp_checkpoint_dir)
        load_director._pipeline = fresh_pipeline
        load_director._epoch = 0
        load_director._global_step = 0
        load_director._metrics = {}
        load_director._scaler = None
        load_director._counter_state = None
        load_director._checkpoint_path = None
        load_director._product = None
        load_director._is_validated = True

        success = load_director.load_from(str(saved_path))

        assert success is True
        # This is the value that train.py uses as start_iteration
        assert load_director._global_step == 5000
        assert load_director._epoch == 10
        assert load_director._counter_state is not None
        assert load_director._counter_state["current_step"] == 5000
        assert load_director._counter_state["current_epoch"] == 10


class TestCheckpointPersistsEmaWarmupCounter:
    """The saved ``ema_state`` must carry the warmup counter (#2172).

    ``ModelEma.state_dict`` is overridden to inject ``_ema_num_updates`` so a
    requeue continues the decay ramp instead of restarting it. The director used
    to save ``unwrap_model(ema).state_dict()``, and ``unwrap_model`` peels
    ModelEma's own ``.module`` -- so the override never ran and the counter was
    dropped on every save. Nothing failed; the ramp just silently restarted at 0
    on each resume, which on the kspace_filling arms means a resumed run has
    different EMA semantics from an uninterrupted one.

    This pins the spelling against that regression: revert the director to the
    unwrapped form and this turns red.
    """

    @staticmethod
    def _director_with_ema(pipeline: _FakePipeline, ckpt_dir: str, num_updates: int):
        from spectramr.infrastructure.builders.directors.checkpoint_director import (
            CheckpointDirector,
        )
        from spectramr.infrastructure.optimization.ema import ModelEma

        ema = ModelEma(pipeline.generator, decay=0.99, warmup=True)
        ema.num_updates = num_updates
        pipeline.ema = ema

        director = CheckpointDirector.__new__(CheckpointDirector)
        director._config = MagicMock()
        director._checkpoint_dir = Path(ckpt_dir)
        director._pipeline = pipeline
        director._epoch = 1
        director._global_step = 100
        director._metrics = {}
        director._scaler = None
        director._counter_state = None
        director._checkpoint_path = None
        director._product = None
        director._is_validated = True
        return director

    def test_saved_ema_state_carries_the_counter(
        self, pipeline: _FakePipeline, tmp_checkpoint_dir: str
    ) -> None:
        from spectramr.infrastructure.optimization.ema import ModelEma

        director = self._director_with_ema(pipeline, tmp_checkpoint_dir, num_updates=4321)
        checkpoint = torch.load(
            director.save(), map_location="cpu", weights_only=False
        )

        assert checkpoint["ema_state"] is not None
        assert checkpoint["ema_state"][ModelEma._NUM_UPDATES_KEY] == 4321

    def test_saved_ema_weight_keys_stay_bare(
        self, pipeline: _FakePipeline, tmp_checkpoint_dir: str
    ) -> None:
        """The on-disk weight-key contract is unchanged, so old checkpoints load."""
        from spectramr.infrastructure.optimization.ema import ModelEma

        director = self._director_with_ema(pipeline, tmp_checkpoint_dir, num_updates=1)
        ema_state = torch.load(
            director.save(), map_location="cpu", weights_only=False
        )["ema_state"]

        weight_keys = [k for k in ema_state if k != ModelEma._NUM_UPDATES_KEY]
        assert weight_keys
        assert not any(
            k.startswith(("module.", "_orig_mod.", "_fsdp_wrapped_module."))
            for k in weight_keys
        )

    def test_counter_survives_a_save_load_roundtrip(
        self, pipeline: _FakePipeline, tmp_checkpoint_dir: str
    ) -> None:
        """The half that matters: a resume continues the ramp."""
        from spectramr.infrastructure.optimization.ema import ModelEma

        director = self._director_with_ema(pipeline, tmp_checkpoint_dir, num_updates=999)
        ema_state = torch.load(
            director.save(), map_location="cpu", weights_only=False
        )["ema_state"]

        restored = ModelEma(_SimpleGenerator(), decay=0.99, warmup=True)
        restored.load_shadow_state_dict(ema_state)

        assert restored.num_updates == 999
