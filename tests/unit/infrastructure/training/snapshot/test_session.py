"""A session selects steps, records the batch and every model call inside, and writes one snapshot."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from spectramr.config.schemas.logging import LoggingConfigSchema
from spectramr.core import snapshot_trace as st
from spectramr.data.batch_types import TrainingBatch
from spectramr.infrastructure.training.snapshot import session as session_mod
from spectramr.infrastructure.training.snapshot.session import SnapshotSession
from spectramr.infrastructure.training.snapshot.writer import MANIFEST

pytestmark = pytest.mark.unit


class _Strategy:
    def __init__(self, tmp_path, **snaps) -> None:
        self.config = SimpleNamespace(
            logging=LoggingConfigSchema(snapshots={"max_calls": 1, **snaps}),
            training=SimpleNamespace(output_dir=None),
        )
        self.env = SimpleNamespace(
            run_output_dir=str(tmp_path), models={"generator": nn.Conv2d(1, 1, 1)}
        )

    def _snapshot_provenance(self, phase: str) -> dict:
        return {"source": phase}


@pytest.fixture
def strategy(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod.ViewPolicy, "from_config", classmethod(lambda cls, c: cls()))
    return _Strategy(tmp_path)


def _batch() -> TrainingBatch:
    return TrainingBatch(
        input=torch.rand(2, 1, 8, 8), target=torch.rand(2, 1, 8, 8), mask=torch.ones(2, 1, 8, 8)
    )


def test_a_due_step_writes_batch_model_and_strategy_stages(strategy, tmp_path) -> None:
    s = SnapshotSession.build(strategy)
    with s.step("train", 1, _batch()):
        st.record("prepare", "input_prepared", {"input_prepared": torch.rand(2, 1, 8, 8)})
        strategy.env.models["generator"](torch.rand(2, 1, 8, 8))
    (manifest,) = tmp_path.rglob(MANIFEST)
    m = json.loads(manifest.read_text())
    names = [f"{x['group']}/{x['name']}" for x in m["stages"]]
    assert names == [
        "batch/input_raw",
        "prepare/input_prepared",
        "model/input[generator]",
        "model/output[generator]",
    ]
    assert m["provenance"] == {"source": "train"} and m["strategy"] == "_Strategy"
    assert m["snapshots"]["max_calls"] == 1
    assert m["models"]["generator"]["in_channels"] == 1
    assert manifest.parent.parent.name == "train"


def test_steps_past_the_budget_record_nothing(strategy, tmp_path) -> None:
    s = SnapshotSession.build(strategy)
    with s.step("train", 1, _batch()) as first:
        assert first is not None
    with s.step("train", 2, _batch()) as second:
        assert second is None and st.current() is None
    assert len(list(tmp_path.rglob(MANIFEST))) == 1


def test_the_snapshot_is_written_when_the_step_raises(strategy, tmp_path) -> None:
    s = SnapshotSession.build(strategy)
    with pytest.raises(RuntimeError), s.step("val", 5, _batch()):
        raise RuntimeError("diverged")
    (manifest,) = tmp_path.rglob(MANIFEST)
    assert manifest.parent.parent.name == "val"


def test_disabled_snapshots_build_no_session(tmp_path) -> None:
    assert SnapshotSession.build(_Strategy(tmp_path, enabled=False)) is None


def test_a_mocked_run_dir_is_refused(strategy) -> None:
    from unittest.mock import MagicMock

    strategy.env.run_output_dir = MagicMock()
    with pytest.raises(TypeError, match="path"):
        SnapshotSession.build(strategy)


class _OneSampleDataset:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def __len__(self) -> int:
        return 1

    def __getitem__(self, i: int):
        if self.fail:
            raise OSError("archive truncated")
        st.record(
            "loader", "dataset_output", {"input": torch.rand(4, 8, 8), "mask": torch.ones(1, 8, 8)}
        )
        return {}


def test_loader_stages_come_first_and_masks_are_not_ifftd(strategy, tmp_path) -> None:
    strategy.env.data_loaders = {"train": SimpleNamespace(dataset=_OneSampleDataset())}
    s = SnapshotSession.build(strategy)
    with s.step("train", 1, _batch()) as trace:
        pass
    assert [f"{x.group}/{x.name}" for x in trace.stages][:2] == [
        "loader/dataset_output",
        "batch/input_raw",
    ]
    views = trace.stages[0].views
    assert views["input"].kspace and not views["mask"].kspace
    assert trace.stages[0].tensors["input"].shape == (4, 8, 8), "the loader sample is kept whole"
    assert trace.extra["loader_trace"] == {"index": 0, "seed": 1}


def test_a_failing_loader_trace_is_noted_and_the_step_still_writes(strategy, tmp_path) -> None:
    strategy.env.data_loaders = {"train": SimpleNamespace(dataset=_OneSampleDataset(fail=True))}
    s = SnapshotSession.build(strategy)
    with s.step("train", 1, _batch()) as trace:
        pass
    assert "archive truncated" in trace.extra["loader_trace"]
    assert len(list(tmp_path.rglob(MANIFEST))) == 1
