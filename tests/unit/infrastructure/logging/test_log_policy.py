"""Tests for :mod:`spectramr.infrastructure.logging.log_policy` (rank floor, precedence, console).

Regression (2026-08-15): a 4-GPU ``train-distributed`` launch opened with four
interleaved copies of every startup line. ``setup_distributed`` did clamp
non-zero ranks to WARNING — but it runs *after* the config loads and the backend
resolves, and all four duplicated lines are emitted before that. The clamp moved
to the entry point, keyed off the rank torchrun exports into the environment
before the process even starts.
"""

from __future__ import annotations

import logging

import pytest

from spectramr.infrastructure.logging import log_policy
from spectramr.infrastructure.logging.log_policy import (
    quiet_secondary_ranks,
    rank_floor,
)


@pytest.fixture(autouse=True)
def _isolated_root_level(monkeypatch):
    """Restore the root level; these tests mutate global logging state."""
    original = logging.root.level
    original_handlers = list(logging.root.handlers)
    logging.root.handlers[:] = []  # no leaked console handler decides what gets installed
    log_policy._entry.update(cli=None, env=None)
    log_policy._policy = log_policy.ConsolePolicy()
    monkeypatch.delenv("SPECTRAMR_LOG_LEVEL", raising=False)
    monkeypatch.delenv("TORCH_LOGS", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    yield
    logging.captureWarnings(False)
    logging.root.handlers[:] = original_handlers
    logging.root.setLevel(original)
    log_policy._entry.update(cli=None, env=None)
    log_policy._policy = log_policy.ConsolePolicy()


# Assertions read ``logging.root`` (a plain attribute), not ``logging.getLogger()``.


def test_no_op_when_not_distributed():
    """A single-process run must be provably untouched."""
    logging.root.setLevel(logging.INFO)
    assert quiet_secondary_ranks() is False
    assert logging.root.level == logging.INFO


def test_no_op_on_rank_zero(monkeypatch):
    """Rank 0 is the rank whose narration we want to keep."""
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    logging.root.setLevel(logging.INFO)
    assert quiet_secondary_ranks() is False
    assert logging.root.level == logging.INFO


@pytest.mark.parametrize("rank", ["1", "2", "3"])
def test_clamps_secondary_rank(rank, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", rank)
    logging.root.setLevel(logging.INFO)
    assert quiet_secondary_ranks() is True
    assert logging.root.level == logging.ERROR


def test_only_rank_zero_logs_but_a_rank_failure_still_surfaces(monkeypatch):
    """The default policy is rank 0 only, with ERROR kept on the others.

    A rank-specific failure (OOM, NaN) is a fact about *that* rank and must not
    vanish; everything below ERROR is rank 0's to say.
    """
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    quiet_secondary_ranks()
    assert logging.root.isEnabledFor(logging.ERROR)
    assert not logging.root.isEnabledFor(logging.WARNING)


def test_secondary_level_is_the_users_to_set(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    log_policy._policy = log_policy.ConsolePolicy(secondary_level=logging.WARNING)
    quiet_secondary_ranks()
    assert logging.root.isEnabledFor(logging.WARNING)
    assert not logging.root.isEnabledFor(logging.INFO)


def test_all_ranks_policy_does_not_quiet_anyone(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    log_policy._policy = log_policy.ConsolePolicy(console_ranks="all")
    logging.root.setLevel(logging.INFO)
    assert quiet_secondary_ranks() is False
    assert logging.root.level == logging.INFO


def test_is_idempotent(monkeypatch):
    """It runs at the entry point AND inside ``setup_distributed``."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    assert quiet_secondary_ranks() is True
    assert quiet_secondary_ranks() is True
    assert logging.root.level == logging.ERROR


def test_setup_distributed_uses_the_shared_helper():
    """``python -m spectramr.pipelines.distributed`` bypasses the CLI entry point.

    Pinned as source inspection because exercising it would require a real
    process group. The point is that there is ONE policy, not two that drift.
    """
    import inspect

    from spectramr.pipelines import distributed

    src = inspect.getsource(distributed)
    assert "quiet_secondary_ranks()" in src
    assert "logging.getLogger().setLevel" not in src, (
        "the inline clamp is back; it duplicates log_policy's policy"
    )


def test_the_bypass_path_is_clamped_before_anything_logs():
    """Merely *calling* the helper somewhere is not enough — WHERE decides.

    ``setup_distributed`` cannot run until the backend is known, and the
    backend comes from the config. So on the ``python -m`` path the config
    load and ``[Parallel] process-group backend=...`` both log at INFO
    *before* the only clamp that used to exist. The clamp therefore has to sit
    at the top of ``run_distributed_training`` — the funnel BOTH entry points
    share — not merely somewhere in the module.

    Pinned by source order, the same technique the startup-notice test uses:
    a refactor that moves the call down is exactly the regression, and it
    would leave the coarse ``in src`` assertion above still passing.
    """
    import inspect

    from spectramr.pipelines import distributed

    body = inspect.getsource(distributed.run_distributed_training)

    clamp = body.index("quiet_secondary_ranks()")
    assert clamp < body.index("TrainingSettings.from_yaml"), (
        "the config load logs before the clamp; rank 1-3 will echo it"
    )
    assert clamp < body.index("resolve_distributed_backend("), (
        "[Parallel] process-group backend=... logs before the clamp"
    )
    assert clamp < body.index("setup_distributed("), (
        "the clamp must precede the process group, not ride along inside it"
    )


class TestRankFloor:
    """``rank_floor`` is the half of the policy for code that RE-SETS a level.

    Clamping the root logger once is not enough: ``LoggingService.setup``
    resolves a level from the config and pushes it onto the root logger, every
    logger in ``loggerDict``, and every handler. That undoes an earlier
    ``quiet_secondary_ranks()`` the moment training configures logging — so the
    4x duplication would return for everything logged from that point on, which
    is most of a training run.
    """

    def test_single_process_is_untouched(self):
        assert rank_floor(logging.DEBUG) == logging.DEBUG
        assert rank_floor(logging.INFO) == logging.INFO

    def test_rank_zero_is_untouched(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", "0")
        assert rank_floor(logging.INFO) == logging.INFO

    @pytest.mark.parametrize("rank", ["1", "2", "3"])
    def test_secondary_rank_is_floored(self, monkeypatch, rank):
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", rank)
        assert rank_floor(logging.INFO) == logging.ERROR
        assert rank_floor(logging.DEBUG) == logging.ERROR
        assert rank_floor(logging.INFO, floor=logging.WARNING) == logging.WARNING

    def test_a_stricter_level_is_not_loosened(self, monkeypatch):
        """It is a FLOOR, not an assignment.

        A config asking for ERROR on a secondary rank must stay ERROR; clamping
        it down to WARNING would make a rank noisier than the operator asked.
        """
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", "2")
        assert rank_floor(logging.ERROR) == logging.ERROR
        assert rank_floor(logging.CRITICAL) == logging.CRITICAL


class TestTheClampSurvivesLoggingConfiguration:
    """Both writers of the root level must go through the floor.

    Source-inspected rather than exercised: ``LoggingService.setup`` installs
    handlers and patches ``logging.getLogger`` globally, so running it for real
    inside the suite leaks into every later test. What must not regress is
    structural — that neither writer resolves a level the floor has not seen.
    """

    def test_logging_service_floors_the_level_it_resolves(self):
        import inspect

        from spectramr.infrastructure.services.logging_service import LoggingService

        body = inspect.getsource(LoggingService.setup)
        assert "effective_level(log_level)" in body, (
            "setup() resolves a raw level again; it will re-verbose "
            "secondary ranks the moment training configures logging"
        )

    def test_bootstrap_floors_the_level_it_resolves(self):
        """``bootstrap_console_logging`` deliberately LOWERS an existing level.

        Its ``has_colored`` branch exists to never silence handlers a caller
        already installed, so without the floor it would happily undo the rank
        clamp. Today ``cli/app.py`` bootstraps before clamping, which hides
        this — the floor is what makes the two order-independent.
        """
        import inspect

        body = inspect.getsource(log_policy.bootstrap_console_logging)
        floor = body.index("rank_floor(")
        assert floor < body.index("root.setLevel("), (
            "the floor must be applied before any setLevel in this function"
        )


class TestPrecedence:
    """``--log-level`` > ``SPECTRAMR_LOG_LEVEL`` > config > INFO, and a bad value raises."""

    def test_default_is_info(self):
        r = log_policy.resolve_level()
        assert (r.name, r.source) == ("info", "default")

    def test_config_beats_default(self):
        r = log_policy.resolve_level("warning", config_declared=True)
        assert (r.value, r.source) == (logging.WARNING, "config")

    def test_undeclared_config_is_labelled_default(self):
        assert log_policy.resolve_level("info", config_declared=False).source == "default"

    def test_env_beats_config(self, monkeypatch):
        monkeypatch.setenv("SPECTRAMR_LOG_LEVEL", "error")
        log_policy.configure_entry(None)
        r = log_policy.resolve_level("debug", config_declared=True)
        assert (r.value, r.source) == (logging.ERROR, "env")

    def test_cli_beats_env_and_config(self, monkeypatch):
        monkeypatch.setenv("SPECTRAMR_LOG_LEVEL", "error")
        log_policy.configure_entry("debug")
        r = log_policy.resolve_level("warning", config_declared=True)
        assert (r.value, r.source) == (logging.DEBUG, "cli")

    def test_setup_cannot_undo_a_cli_level(self):
        """``LoggingService.setup`` asks ``effective_level``; the CLI choice must win there too."""
        log_policy.configure_entry("error")
        assert log_policy.effective_level("debug") == logging.ERROR

    @pytest.mark.parametrize("bad", ["verbose", "warn!", "10x", True])
    def test_unknown_level_raises(self, bad):
        with pytest.raises(ValueError, match="Unsupported log level"):
            log_policy.parse_level(bad)

    def test_bad_cli_value_raises_before_anything_is_applied(self):
        with pytest.raises(ValueError, match="Unsupported log level"):
            log_policy.configure_entry("loud")

    def test_secondary_rank_floors_whatever_was_asked(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", "2")
        log_policy.configure_entry("debug")
        assert logging.root.level == logging.ERROR


class TestApplyLevel:
    def test_warning_silences_info_and_handlers_follow(self):
        log_policy.bootstrap_console_logging(logging.INFO)
        applied = log_policy.apply_config_level("warning", declared=True)
        assert applied.value == logging.WARNING
        assert logging.root.level == logging.WARNING
        assert all(h.level in (logging.NOTSET, logging.WARNING) for h in logging.root.handlers)

    def test_debug_is_not_blocked_by_an_info_handler(self):
        """The bootstrap handler is born at INFO; raising verbosity must lower it too."""
        log_policy.bootstrap_console_logging(logging.INFO)
        log_policy.apply_config_level("debug", declared=True)
        assert all(h.level in (logging.NOTSET, logging.DEBUG) for h in logging.root.handlers)

    def test_configure_entry_installs_the_colored_console(self):
        from spectramr.infrastructure.logging.console import ColoredConsoleFormatter

        log_policy.configure_entry(None)
        assert any(isinstance(h.formatter, ColoredConsoleFormatter) for h in logging.root.handlers)

    def test_accepts_the_schema_enum(self):
        from spectramr.config.schemas.enums import LogLevel

        assert log_policy.apply_config_level(LogLevel.ERROR).value == logging.ERROR

    def test_current_level_records_the_source(self):
        log_policy.apply_config_level("error", declared=True)
        cur = log_policy.current_level()
        assert (cur.name, cur.source) == ("error", "config")

    def test_apply_logging_config_reads_declared_from_fields_set(self):
        from spectramr.config.schemas.logging import LoggingSinksConfigSchema

        declared = _logging_cfg(sinks=LoggingSinksConfigSchema(level="warning"))
        assert log_policy.apply_logging_config(declared).source == "config"
        assert log_policy.apply_logging_config(_logging_cfg()).source == "default"


_VERBS_THAT_APPLY_THE_CONFIG_LEVEL = [
    ("spectramr.main", "__common_train_setup"),
    ("spectramr.main", "begin_inference_run"),
    ("spectramr.main", "experiment_command"),
    ("spectramr.pipelines.distributed", "run_distributed_training"),
    ("spectramr.pipelines.ablation", "run_ablation_pipeline"),
    ("spectramr.pipelines.ablation", "run_ablation_study"),
    ("spectramr.pipelines.infer", "run_inference_pipeline"),
    ("spectramr.cli.app", "export"),
    ("spectramr.cli.app", "_audit_one"),
    ("spectramr.cli.app", "report"),
]


@pytest.mark.parametrize(("module", "func"), _VERBS_THAT_APPLY_THE_CONFIG_LEVEL)
def test_every_config_loading_verb_applies_the_declared_level(module, func):
    """A verb that loads a config but never applies its level is the original bug.

    Bulk audit and the ``hpo`` parent are left out on purpose: one level cannot
    stand for many configs, and each hpo trial applies its own.
    """
    import ast
    import importlib
    import inspect

    mod = importlib.import_module(module)
    node = ast.parse(inspect.getsource(mod))
    fn = next(
        n
        for n in ast.walk(node)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == func
    )
    called = {
        c.func.id if isinstance(c.func, ast.Name) else getattr(c.func, "attr", "")
        for c in ast.walk(fn)
        if isinstance(c, ast.Call)
    }
    assert called & {"apply_logging_config", "_apply_log_level"}, (
        f"{module}.{func} loads a config but never applies logging.sinks.level"
    )


def _logging_cfg(**blocks):
    from spectramr.config.schemas.logging import LoggingConfigSchema

    return LoggingConfigSchema(**blocks)


class TestDistributedPolicy:
    """``logging.distributed`` and ``logging.sinks.console_stream`` are read, not decorative."""

    def test_defaults_are_rank0_only_error_floor_split(self):
        assert log_policy.current_policy() == {
            "console_ranks": "rank0",
            "secondary_level": "error",
            "console_stream": "split",
            "file_ranks": "rank0",
        }

    def test_config_is_adopted(self):
        log_policy.adopt_logging_policy(
            _logging_cfg(
                distributed={
                    "console_ranks": "all",
                    "secondary_level": "warning",
                    "file_ranks": "all",
                },
                sinks={"console_stream": "stdout"},
            )
        )
        assert log_policy.current_policy() == {
            "console_ranks": "all",
            "secondary_level": "warning",
            "console_stream": "stdout",
            "file_ranks": "all",
        }

    def test_apply_logging_config_applies_the_stream_to_the_console(self):
        from spectramr.infrastructure.logging.console import _MANAGED_ATTR

        # pytest puts its own StreamHandlers on the root during the call; those would be
        # upgraded in place rather than replaced, so the test owns the root's handlers.
        saved = list(logging.root.handlers)
        logging.root.handlers[:] = []
        try:
            log_policy.configure_entry(None)
            log_policy.apply_logging_config(_logging_cfg(sinks={"console_stream": "stderr"}))
            managed = [h for h in logging.root.handlers if hasattr(h, _MANAGED_ATTR)]
            assert managed and all(getattr(h, _MANAGED_ATTR) == "stderr" for h in managed)
        finally:
            logging.root.handlers[:] = saved

    def test_silent_config_does_not_install_a_console(self):
        from spectramr.infrastructure.logging.console import _MANAGED_ATTR

        for h in [h for h in logging.root.handlers if hasattr(h, _MANAGED_ATTR)]:
            logging.root.removeHandler(h)
        log_policy.apply_logging_config(_logging_cfg(sinks={"silent": True}))
        assert not [h for h in logging.root.handlers if hasattr(h, _MANAGED_ATTR)]

    @pytest.mark.parametrize(
        ("rank", "scope", "expected"),
        [(None, "rank0", True), ("0", "rank0", True), ("2", "rank0", False), ("2", "all", True)],
    )
    def test_console_enabled_on_this_rank(self, monkeypatch, rank, scope, expected):
        if rank is not None:
            monkeypatch.setenv("WORLD_SIZE", "4")
            monkeypatch.setenv("RANK", rank)
        log_policy._policy = log_policy.ConsolePolicy(console_ranks=scope)
        assert log_policy.console_enabled_on_this_rank() is expected

    def test_text_log_path_single_process_and_rank_zero(self, monkeypatch):
        assert log_policy.rank_log_path("/r/x.log") == "/r/x.log"
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", "0")
        assert log_policy.rank_log_path("/r/x.log") == "/r/x.log"

    def test_other_ranks_write_no_text_log_by_default(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", "2")
        assert log_policy.rank_log_path("/r/x.log") is None

    def test_file_ranks_all_gives_each_rank_its_own_file(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", "2")
        log_policy._policy = log_policy.ConsolePolicy(file_ranks="all")
        assert log_policy.rank_log_path("/r/x.log") == "/r/x.rank2.log"

    def test_logging_service_setup_uses_the_policy_for_file_and_stream(self):
        import inspect

        from spectramr.infrastructure.services.logging_service import LoggingService

        body = inspect.getsource(LoggingService.setup)
        assert "adopt_logging_policy(logging_config)" in body
        assert "rank_log_path(" in body
        assert body.count("stream=console_stream()") == 2, "root and service-logger consoles"


class TestWarningsFollowThePolicy:
    """``warnings.warn`` bypasses logging, so it printed from every rank to stderr."""

    def test_entry_routes_python_warnings_through_logging(self):
        import warnings

        log_policy.configure_entry(None)
        assert warnings.showwarning is not warnings._showwarning_orig
