"""store: the only writer. Native artifact, writer rank, atomic write, retention."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from spectramr.infrastructure.checkpointing import codecs, store
from spectramr.infrastructure.checkpointing.async_writer import (
    CheckpointWriteError,
    CheckpointWriter,
)
from spectramr.infrastructure.checkpointing.layout import CheckpointLayout
from spectramr.infrastructure.distributed.checkpoint_adapters import DefaultCheckpointAdapter
from spectramr.infrastructure.distributed.state_io import PassthroughStateIO
from tests.unit.infrastructure.checkpointing.conftest import FakeEarlyStopping, make_pipeline


def _config(fmt: str = "pt", keep_last_n: object = 2):
    return SimpleNamespace(checkpoint=SimpleNamespace(format=fmt, keep_last_n=keep_last_n))


def _step(tmp_path: Path, step: int, *, fmt: str = "pt", parallel=None, **kw) -> Path:
    return store.save_step(
        make_pipeline(),
        layout=CheckpointLayout.for_run(tmp_path),
        config=_config(fmt, **kw),
        strategy=None,
        parallel=parallel,
        early_stopping=FakeEarlyStopping(),
        reason="periodic",
        epoch=0,
        step=step,
    )


@pytest.mark.parametrize("fmt", ["safetensors", "pt", "pth"])
def test_save_step_writes_the_declared_format(tmp_path: Path, fmt: str) -> None:
    path = _step(tmp_path, 7, fmt=fmt)
    assert path == tmp_path / "checkpoints" / f"step_00000007.{fmt}"
    assert codecs.read(path)["meta"]["reason"] == "periodic"


def test_save_step_then_resume_round_trips(tmp_path: Path) -> None:
    state = codecs.read(_step(tmp_path, 3))
    assert (state["meta"]["epoch"], state["meta"]["step"]) == (0, 3)


def test_a_killed_write_leaves_the_previous_file(tmp_path: Path) -> None:
    path = _step(tmp_path, 3)
    before = path.read_bytes()
    with patch("torch.save", side_effect=OSError("disk full")), pytest.raises(RuntimeError):
        _step(tmp_path, 3)
    assert path.read_bytes() == before
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_a_non_writer_rank_writes_nothing(tmp_path: Path) -> None:
    with patch.object(store, "is_writer_rank", return_value=False):
        path = _step(tmp_path, 3)
    assert not path.exists()


def test_native_tag_is_recorded_and_passed_to_save_native(tmp_path: Path) -> None:
    seen: dict = {}

    class Native:
        name = "deepspeed"
        writes_native_artifact = True
        requires_all_ranks = True
        state_io = PassthroughStateIO()

        def full_model_state(self, model):
            return model.state_dict()

        def engine_owned_keys(self, models):
            return frozenset()

        def ema_is_whole(self, models):
            return True

        def native_tag_roots(self, *, models, checkpoint_dir):
            return (checkpoint_dir / "deepspeed" / "generator",)

        def save_native(self, *, models, checkpoint_dir, tag, client_state):
            seen.update(tag=tag, client_state=dict(client_state))
            (checkpoint_dir / "deepspeed" / "generator" / tag).mkdir(parents=True)

    path = _step(
        tmp_path, 12, parallel=SimpleNamespace(strategy="deepspeed", checkpoint_adapter=Native())
    )
    assert seen["tag"] == "step_00000012"
    assert seen["client_state"] == {"epoch": 0, "global_step": 12}
    assert codecs.read(path)["meta"]["native_tag"] == "step_00000012"


class TestRetention:
    def test_keeps_the_last_n_steps(self, tmp_path: Path) -> None:
        for step in (1, 2, 3):
            _step(tmp_path, step)
        names = sorted(p.name for p in (tmp_path / "checkpoints").iterdir())
        assert names == ["step_00000002.pt", "step_00000003.pt"]

    def test_an_invalid_keep_last_n_raises_before_anything_is_written(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="keep_last_n"):
            _step(tmp_path, 1, keep_last_n=0)
        assert not (tmp_path / "checkpoints").exists()

    def test_a_retention_error_never_fails_a_landed_save(self, tmp_path: Path) -> None:
        _step(tmp_path, 1)
        with patch(
            "spectramr.infrastructure.checkpointing.retention._units",
            side_effect=OSError("listing failed"),
        ):
            path = _step(tmp_path, 2)
        assert path.exists()

    def test_a_best_save_prunes_nothing(self, tmp_path: Path) -> None:
        for step in (1, 2):
            _step(tmp_path, step)
        _best(tmp_path, 3)
        assert len(list((tmp_path / "checkpoints").glob("step_*"))) == 2


def _best(tmp_path: Path, step: int, value: float = 21.5) -> Path:
    return store.save_best(
        make_pipeline(),
        layout=CheckpointLayout.for_run(tmp_path),
        config=_config(),
        strategy=None,
        parallel=None,
        early_stopping=FakeEarlyStopping(),
        epoch=0,
        step=step,
        metric_name="val_psnr",
        metric_value=value,
    )


class TestSaveBest:
    def test_creates_best_in_the_declared_format(self, tmp_path: Path) -> None:
        assert _best(tmp_path, 4) == tmp_path / "checkpoints" / "best.pt"

    def test_records_the_metric_and_position(self, tmp_path: Path) -> None:
        meta = codecs.read(_best(tmp_path, 4))["meta"]
        assert meta["best_metric"] == {"name": "val_psnr", "value": 21.5}
        assert (meta["role"], meta["reason"], meta["step"]) == ("best", "best", 4)

    def test_a_second_best_replaces_the_first(self, tmp_path: Path) -> None:
        _best(tmp_path, 4)
        path = _best(tmp_path, 6, value=22.0)
        assert codecs.read(path)["meta"]["step"] == 6
        assert [p.name for p in path.parent.iterdir()] == ["best.pt"]


def test_a_non_writer_rank_still_enters_the_native_save(tmp_path: Path) -> None:
    """The native save is collective: skipping it on a non-writer rank would hang rank 0."""
    calls: list[str] = []

    class Recording(DefaultCheckpointAdapter):
        name = "deepspeed"
        requires_all_ranks = True

        def save_native(self, *, models, checkpoint_dir, tag, client_state):
            calls.append(tag)

    parallel = SimpleNamespace(strategy="deepspeed", checkpoint_adapter=Recording())
    with patch.object(store, "is_writer_rank", return_value=False):
        path = _step(tmp_path, 5, parallel=parallel)
    assert calls == ["step_00000005"]
    assert not path.exists()


class TestWithAWriter:
    """``writer=`` moves the file write and the prune off the calling thread."""

    def _save(self, tmp_path: Path, step: int, writer) -> Path:
        return store.save_step(
            make_pipeline(),
            layout=CheckpointLayout.for_run(tmp_path),
            config=_config(),
            strategy=None,
            parallel=None,
            early_stopping=FakeEarlyStopping(),
            reason="periodic",
            epoch=0,
            step=step,
            writer=writer,
        )

    def test_save_step_lands_after_drain_and_prunes(self, tmp_path: Path) -> None:
        writer = CheckpointWriter()
        for step in (1, 2, 3):
            self._save(tmp_path, step, writer)
        writer.drain()
        names = sorted(p.name for p in (tmp_path / "checkpoints").iterdir())
        assert names == ["step_00000002.pt", "step_00000003.pt"]
        assert codecs.read(tmp_path / "checkpoints" / names[-1])["meta"]["step"] == 3

    def test_save_best_lands_after_drain(self, tmp_path: Path) -> None:
        writer = CheckpointWriter()
        path = store.save_best(
            make_pipeline(),
            layout=CheckpointLayout.for_run(tmp_path),
            config=_config(),
            strategy=None,
            parallel=None,
            early_stopping=FakeEarlyStopping(),
            epoch=0,
            step=4,
            metric_name="val_psnr",
            metric_value=21.5,
            writer=writer,
        )
        writer.drain()
        assert codecs.read(path)["meta"]["best_metric"]["value"] == 21.5

    def test_a_non_writer_rank_submits_nothing(self, tmp_path: Path) -> None:
        writer = CheckpointWriter()
        with patch.object(store, "is_writer_rank", return_value=False):
            path = self._save(tmp_path, 3, writer)
        writer.drain()
        assert not path.exists()
        assert writer._executor is None

    def _best(self, tmp_path: Path, step: int, writer) -> Path:
        return store.save_best(
            make_pipeline(),
            layout=CheckpointLayout.for_run(tmp_path),
            config=_config(),
            strategy=None,
            parallel=None,
            early_stopping=FakeEarlyStopping(),
            epoch=0,
            step=step,
            metric_name="val_psnr",
            metric_value=21.5,
            writer=writer,
        )

    def test_a_failed_periodic_write_surfaces_at_the_next_save_naming_itself(
        self, tmp_path: Path
    ) -> None:
        writer = CheckpointWriter()
        with patch("torch.save", side_effect=OSError("disk full")):
            failed = self._save(tmp_path, 1, writer)
            with pytest.raises(CheckpointWriteError) as caught:
                self._best(tmp_path, 2, writer)
        assert caught.value.path == failed
        assert str(caught.value) == f"Failed to save checkpoint {failed}: disk full"

    def test_a_failed_best_write_does_not_stop_the_next_save(self, tmp_path: Path) -> None:
        writer = CheckpointWriter()
        with patch("torch.save", side_effect=OSError("disk full")):
            best = self._best(tmp_path, 1, writer)
            writer.drain()
        self._save(tmp_path, 2, writer)
        writer.close()
        assert [f.path for f in writer.soft_failures] == [best]
        assert (tmp_path / "checkpoints" / "step_00000002.pt").exists()

    def test_a_loss_schedule_write_is_not_fatal(self, tmp_path: Path) -> None:
        writer = CheckpointWriter()
        with patch("torch.save", side_effect=OSError("disk full")):
            store.save_step(
                make_pipeline(),
                layout=CheckpointLayout.for_run(tmp_path),
                config=_config(),
                strategy=None,
                parallel=None,
                early_stopping=FakeEarlyStopping(),
                reason="loss_schedule",
                epoch=0,
                step=1,
                writer=writer,
            )
            writer.drain()
        assert [f.role for f in writer.soft_failures] == ["step"]
