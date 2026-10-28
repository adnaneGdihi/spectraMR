"""The sink registry, the directory rule, and the LoggingService wired onto them."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from spectramr.config.schemas.logging import LoggingConfigSchema
from spectramr.infrastructure.logging.sinks import (
    ArtifactSink,
    FileSink,
    SinkRegistry,
    TensorBoardSink,
    resolve_run_dir,
    resolve_sinks_dir,
)
from spectramr.infrastructure.services.logging_service import LoggingServiceFactory


def _cfg(**blocks) -> LoggingConfigSchema:
    return LoggingConfigSchema(**blocks)


class TestDirectories:
    def test_undeclared_dir_follows_the_run_directory(self, tmp_path):
        assert resolve_sinks_dir(_cfg(), tmp_path) == (str(tmp_path / "logs"), "run_dir")

    def test_declared_dir_wins_even_when_it_equals_the_default(self, tmp_path):
        assert resolve_sinks_dir(_cfg(sinks={"dir": "./logs"}), tmp_path) == ("./logs", "declared")

    def test_run_dir_default_override_and_sanity_redirect(self):
        none = SimpleNamespace(training=None)
        set_ = SimpleNamespace(training=SimpleNamespace(output_dir="out/run"))
        assert resolve_run_dir(none) == Path("experiments/outputs")
        assert resolve_run_dir(set_) == Path("out/run")
        assert resolve_run_dir(set_, True) == Path("out/run/sanity_check")


class TestRegistry:
    def test_duplicate_name_raises_and_replace_closes_the_old_sink(self):
        closed = []

        class _S(ArtifactSink):
            def close(self):
                closed.append(self.name)

        reg = SinkRegistry()
        reg.add(_S("a", lambda: "x"))
        with pytest.raises(ValueError, match="already registered"):
            reg.add(_S("a", lambda: "y"))
        reg.add(_S("a", lambda: "y"), replace=True)
        assert closed == ["a"]

    def test_unknown_sink_raises_naming_the_registered_ones(self):
        reg = SinkRegistry()
        reg.add(ArtifactSink("loss_csv", lambda: None))
        with pytest.raises(KeyError, match="loss_csv"):
            reg.get("nope")

    def test_records_report_enabled_and_path_and_close_all_empties(self):
        reg = SinkRegistry()
        reg.add(ArtifactSink("metrics_csv", lambda: "/r/m.csv"))
        reg.add(ArtifactSink("loss_csv", lambda: None))
        assert [(r.name, r.enabled, r.path) for r in reg.records()] == [
            ("metrics_csv", True, "/r/m.csv"),
            ("loss_csv", False, None),
        ]
        reg.close_all()
        assert reg.records() == []

    def test_a_disabled_tensorboard_writer_is_a_disabled_sink(self):
        class _Off:
            event_dir = "/never"

            def __bool__(self):
                return False

        sink = TensorBoardSink(_Off())
        assert (sink.enabled, sink.path) == (False, None)


class TestFileSink:
    def test_writes_and_detaches(self, tmp_path):
        log = logging.getLogger("sinks.file.ok")
        sink = FileSink(log, str(tmp_path / "d"), "x.log")
        log.warning("hello")
        assert "hello" in (tmp_path / "d" / "x.log").read_text()
        sink.close()
        assert not any(isinstance(h, logging.FileHandler) for h in log.handlers)
        assert not sink.enabled

    def test_an_unwritable_dir_relocates_loudly(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("not a dir")
        log = logging.getLogger("sinks.file.relocated")
        with pytest.warns(RuntimeWarning, match="relocated"):
            sink = FileSink(log, str(blocker / "sub"), "y.log")
        try:
            assert sink.relocated_from == str(blocker / "sub")
            assert Path(sink.resolved_path).exists()
        finally:
            sink.close()


class TestServiceOwnsTheSinks:
    def test_undeclared_dir_puts_the_log_beside_the_run(self, tmp_path):
        svc = LoggingServiceFactory.create(_cfg(), "SinksFollowRun", run_dir=tmp_path)
        try:
            svc.log_warning("tracked")
            assert svc.sinks_dir_source == "run_dir"
            assert svc.resolved_log_path == str(tmp_path / "logs" / "SinksFollowRun.log")
            assert "tracked" in Path(svc.resolved_log_path).read_text()
            assert {
                "name": "file",
                "enabled": True,
                "path": svc.resolved_log_path,
            } in svc.sink_records()
        finally:
            svc.close()

    def test_declared_dir_still_wins(self, tmp_path):
        declared = tmp_path / "elsewhere"
        svc = LoggingServiceFactory.create(
            _cfg(sinks={"dir": str(declared)}), "SinksDeclared", run_dir=tmp_path / "run"
        )
        try:
            assert svc.sinks_dir_source == "declared"
            assert svc.resolved_log_path == str(declared / "SinksDeclared.log")
        finally:
            svc.close()

    def test_tensorboard_and_artifact_sinks_register_and_close_with_the_service(self, tmp_path):
        svc = LoggingServiceFactory.create(_cfg(), "SinksTb", run_dir=tmp_path)
        writer = svc.open_tensorboard(tmp_path, _cfg(tracking={"enabled": False}))
        svc.attach_artifact_sink("loss_csv", lambda: tmp_path / "t.csv")
        by_name = {r["name"]: r for r in svc.sink_records()}
        assert by_name["tensorboard"]["enabled"] is False and not writer
        assert by_name["loss_csv"]["path"] == str(tmp_path / "t.csv")
        svc.close()
        assert svc.sink_records() == []
