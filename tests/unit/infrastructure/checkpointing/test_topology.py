"""topology_record and check_resume_topology: the world size and backend a checkpoint was
written at, and the resume guard over them."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from spectramr.infrastructure.checkpointing.topology import check_resume_topology, topology_record


def test_topology_record_coerces_no_runtime_to_none() -> None:
    assert topology_record(None) == {"world_size": 1, "parallel_strategy": "none"}
    assert topology_record(SimpleNamespace(strategy="ddp"))["parallel_strategy"] == "ddp"


def _guard(data: dict, *, allow=False, allow_strategy=False, parallel=None) -> None:
    cfg = SimpleNamespace(
        allow_world_size_change=allow, allow_parallel_strategy_change=allow_strategy
    )
    check_resume_topology(data.get("topology"), cfg, parallel, source=Path("ckpt.pt"))


class TestResumeTopologyGuard:
    """A checkpoint written at one world size must not silently resume at another.

    ``max_iterations`` is per rank and the LR schedule is indexed by iteration, so the same
    ``global_step`` at 4 ranks has seen 4x the samples it had at 1.
    """

    @staticmethod
    def _world(monkeypatch, size):
        from spectramr.infrastructure.distributed.distributed_training import RankUtility

        monkeypatch.setattr(RankUtility, "get_world_size", staticmethod(lambda: size))

    def test_a_different_world_size_is_refused(self, monkeypatch):
        self._world(monkeypatch, 4)
        data = {"topology": {"world_size": 1, "parallel_strategy": "none"}}
        with pytest.raises(RuntimeError, match=r"world_size=1.*world_size=4"):
            _guard(data)

    def test_the_refusal_names_the_opt_out(self, monkeypatch):
        self._world(monkeypatch, 2)
        with pytest.raises(RuntimeError, match="allow_world_size_change"):
            _guard({"topology": {"world_size": 8}})

    def test_the_same_world_size_resumes(self, monkeypatch):
        self._world(monkeypatch, 4)
        _guard({"topology": {"world_size": 4}})

    def test_an_older_checkpoint_without_a_topology_is_accepted(self, monkeypatch):
        """Unknown is not different: it predates the record."""
        self._world(monkeypatch, 4)
        _guard({})
        _guard({"topology": "garbage"})

    def test_the_opt_out_resumes_and_says_so(self, monkeypatch, caplog):
        self._world(monkeypatch, 4)
        with caplog.at_level("WARNING"):
            _guard({"topology": {"world_size": 1}}, allow=True)
        assert "allow_world_size_change is true" in caplog.text

    def test_a_truthy_accident_is_not_the_opt_out(self, monkeypatch):
        """``is True``, not truthiness: a stray object must not waive the guard."""
        self._world(monkeypatch, 4)
        with pytest.raises(RuntimeError):
            _guard({"topology": {"world_size": 1}}, allow="yes")

    def test_the_schema_knob_defaults_off(self):
        from spectramr.config.schemas.checkpoint import CheckpointConfigSchema

        assert CheckpointConfigSchema().allow_world_size_change is False
        assert CheckpointConfigSchema(allow_world_size_change=True).allow_world_size_change


class TestResumeParallelStrategyGuard:
    """A checkpoint's optimizer state is only loadable by the backend that wrote it, unless
    both sides are plain (none, dp, ddp)."""

    @staticmethod
    def _check(saved, *, now_strategy, allow=False):
        parallel = None if now_strategy is None else SimpleNamespace(strategy=now_strategy)
        _guard(
            {"topology": {"world_size": 1, "parallel_strategy": saved}},
            allow_strategy=allow,
            parallel=parallel,
        )

    @pytest.fixture(autouse=True)
    def _one_rank(self, monkeypatch):
        from spectramr.infrastructure.distributed.distributed_training import RankUtility

        monkeypatch.setattr(RankUtility, "get_world_size", staticmethod(lambda: 1))

    @pytest.mark.parametrize(
        ("saved", "now"),
        [
            ("fsdp", "none"),
            ("none", "fsdp"),
            ("deepspeed", "ddp"),
            ("ddp", "deepspeed"),
            ("fsdp", "deepspeed"),
        ],
    )
    def test_a_swap_involving_a_sharded_backend_is_refused(self, saved, now):
        with pytest.raises(RuntimeError, match="allow_parallel_strategy_change"):
            self._check(saved, now_strategy=now)

    @pytest.mark.parametrize(
        ("saved", "now"), [("none", "ddp"), ("ddp", "dp"), ("dp", "none"), ("ddp", "none")]
    )
    def test_plain_backends_are_interchangeable(self, saved, now):
        self._check(saved, now_strategy=now)

    @pytest.mark.parametrize("strategy", ["none", "ddp", "fsdp", "deepspeed"])
    def test_the_same_backend_resumes(self, strategy):
        self._check(strategy, now_strategy=strategy)

    def test_no_runtime_is_the_plain_path(self):
        """A resume with no ParallelRuntime attached is a single-process run."""
        self._check("none", now_strategy=None)
        with pytest.raises(RuntimeError, match="fsdp"):
            self._check("fsdp", now_strategy=None)

    def test_the_opt_out_resumes_and_says_so(self, caplog):
        with caplog.at_level("WARNING"):
            self._check("fsdp", now_strategy="ddp", allow=True)
        assert "allow_parallel_strategy_change is true" in caplog.text

    def test_a_truthy_accident_is_not_the_opt_out(self):
        with pytest.raises(RuntimeError):
            self._check("fsdp", now_strategy="ddp", allow="yes")

    def test_the_schema_knob_defaults_off(self):
        from spectramr.config.schemas.checkpoint import CheckpointConfigSchema

        assert CheckpointConfigSchema().allow_parallel_strategy_change is False
