"""Tests for the ``spectramr profile`` verb: registration, dry-run, and outcome.

The config loader is stubbed throughout — these cover the verb's control flow,
not config parsing. The one test that must not be stubbed is the registration
check: a verb that builds a perfect argv but never appears on the parser is the
"registered but unwired" shape non-negotiable 16 exists for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spectramr.cli import profile_cli
from spectramr.cli.profile_command import PROFILE_CHILD_OVERRIDES

# ------------------------------------------------------------- registration


def test_profile_is_registered_on_the_real_parser():
    from spectramr.cli.app import build_parser

    sub = next(a for a in build_parser()._actions if a.dest == "command")
    assert "profile" in sub.choices


def test_profile_dispatches_to_the_handler():
    """``main`` dispatches through ``args.func``; a subparser that forgets
    ``set_defaults`` parses fine and then crashes on AttributeError."""
    from spectramr.cli.app import build_parser

    args = build_parser().parse_args(["profile", "--config", "a.yaml"])
    assert args.func is profile_cli.profile


def test_profile_is_flagged_as_a_heavy_startup_command():
    """The parent loads the arm's config before it can resolve the run dir, so
    it pays the torch import too — without this the CLI looks frozen."""
    from spectramr.cli.app import _HEAVY_STARTUP_COMMANDS

    assert "profile" in _HEAVY_STARTUP_COMMANDS


@pytest.mark.parametrize(
    ("flag", "expected"),
    [("--target", "train"), ("--mode", "full"), ("--focus", "all")],
)
def test_defaults(flag, expected):
    from spectramr.cli.app import build_parser

    args = build_parser().parse_args(["profile", "--config", "a.yaml"])
    assert getattr(args, flag.lstrip("-").replace("-", "_")) == expected


@pytest.mark.parametrize(
    "bad", [["--target", "hpo"], ["--mode", "gpu-only"], ["--focus", "everything"]]
)
def test_unknown_option_values_are_rejected_not_defaulted(bad):
    """A closed choice set must raise, never degrade to a default (NN3)."""
    from spectramr.cli.app import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["profile", "--config", "a.yaml", *bad])


# ----------------------------------------------------------------- dry run


def stub_settings(**over):
    """A settings-shaped stub: enough for the verb and both pre-flight checks.

    Defaults are deliberately the *permissive* ones — ``strategy: none`` and an
    epoch-derived (``None``) budget — so a test that does not care about
    pre-flight is not silently exercising it. Tests that DO care override.
    """
    from types import SimpleNamespace

    base = {
        "training": SimpleNamespace(output_dir="experiments/results/exp_11", max_iterations=None),
        "validation": SimpleNamespace(schedule=SimpleNamespace(interval_steps=100, on_epoch=False)),
        "parallel": SimpleNamespace(strategy="none"),
    }
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture
def stub_config(monkeypatch):
    """Stub the config load; these tests are about the verb, not the schema."""
    monkeypatch.setattr(profile_cli, "_load_arm", lambda p: stub_settings())
    # Hermetic: the backend choice must not depend on the GPU of the machine running the tests.
    monkeypatch.setattr(profile_cli, "cuda_visible", lambda: False)
    # The stub path names no file, and the iteration cap reads the raw YAML.
    monkeypatch.setattr(
        "spectramr.cli.manifest_dispatch.compute_iter_cap_overrides",
        lambda cfg, n: (["--override", f"training.max_iterations={n}"], []),
    )


def _args(**over):
    import argparse

    base = {
        "config": Path("experiments/validated/dummy_gan.yaml"),
        "target": "train",
        "mode": "cpu-only",
        "backend": "auto",
        "focus": "train-loop",
        "experiment": None,
        "device": "cpu",
        "val_batches": None,
        "iterations": None,
        "no_phase_split": False,
        "dry_run": True,
        "extra": [],
    }
    base.update(over)
    return argparse.Namespace(**base)


def test_dry_run_prints_the_command_and_writes_nothing(stub_config, capsys, tmp_path):
    created_before = set(Path("experiments/results").glob("exp_11/profile/*"))
    assert profile_cli.profile(_args()) == 0
    out = capsys.readouterr().out
    assert "-m scalene run" in out
    assert "--program-path" in out
    assert set(Path("experiments/results").glob("exp_11/profile/*")) == created_before


def test_dry_run_works_without_scalene(stub_config, monkeypatch, capsys):
    """The command should be inspectable on a box that cannot run it."""
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert profile_cli.profile(_args()) == 0
    assert "scalene run" in capsys.readouterr().out


def test_auto_backend_swaps_scalene_gpu_mode_for_torch_on_cuda(stub_config, monkeypatch, capsys):
    monkeypatch.setattr(profile_cli, "cuda_visible", lambda: True)
    assert profile_cli.profile(_args(mode="full")) == 0
    out = capsys.readouterr().out
    assert "profile_torch_runner" in out
    assert "scalene run" not in out


def test_backend_scalene_forces_scalene_even_for_a_gpu_mode(stub_config, monkeypatch, capsys):
    monkeypatch.setattr(profile_cli, "cuda_visible", lambda: True)
    assert profile_cli.profile(_args(mode="full", backend="scalene")) == 0
    assert "scalene run" in capsys.readouterr().out


def test_backend_torch_without_cuda_raises_instead_of_profiling_the_cpu(stub_config):
    with pytest.raises(RuntimeError, match="no CUDA device"):
        profile_cli.profile(_args(backend="torch"))


def test_backend_torch_rejects_an_iteration_cap_too_short_for_its_window(stub_config, monkeypatch):
    monkeypatch.setattr(profile_cli, "cuda_visible", lambda: True)
    with pytest.raises(ValueError, match="--iterations >= 16"):
        profile_cli.profile(_args(backend="torch", iterations=10))


def test_real_run_raises_without_scalene(stub_config, monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="not installed"):
        profile_cli.profile(_args(dry_run=False))


# ------------------------------------------------------------ outcome paths


@pytest.fixture
def fake_run(monkeypatch, tmp_path, stub_config):
    """Redirect the results root into tmp and stub the child process."""
    import spectramr.cli.profile_paths as pp

    monkeypatch.setattr(pp, "RESULTS_ROOT", tmp_path / "experiments" / "results")
    monkeypatch.setattr(profile_cli, "require_scalene", lambda: "2.3.0")
    return tmp_path / "experiments" / "results" / "exp_11" / "profile"


def _only_profile_dir(profiles_root: Path) -> Path:
    (found,) = list(profiles_root.iterdir())
    return found


def test_successful_run_writes_manifest_and_reports_the_modes(fake_run, monkeypatch):
    def child(argv, log_path, **_):
        # Stand in for scalene: emit the outfile the real one would write.
        Path(argv[argv.index("--outfile") + 1]).write_text("{}", encoding="utf-8")
        log_path.write_text("ok\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(profile_cli, "_run_child", child)
    assert profile_cli.profile(_args(dry_run=False)) == 0

    manifest = json.loads((_only_profile_dir(fake_run) / "profile_manifest.json").read_text())
    assert manifest["modes"]["mode"] == "cpu-only"
    assert manifest["modes"]["focus"] == "train-loop"
    assert manifest["outcome"] == {
        "exit_code": 0,
        "duration_s": pytest.approx(manifest["outcome"]["duration_s"]),
        "outfile_written": True,
        "retries": 0,
        # A green run carries no diagnosis. Asserted rather than omitted: a
        # classifier that fired on success would attach a failure explanation to
        # a profile that worked.
        "diagnosis": None,
    }
    # Arms of one cohort share <experiment>; the arm in the id keeps them apart.
    assert _only_profile_dir(fake_run).name.startswith("dummy_gan-train-cpu-only-")
    assert manifest["modes"]["single_process"] == []
    assert manifest["modes"]["child_overrides"] == list(PROFILE_CHILD_OVERRIDES)


def test_manifest_survives_a_child_that_never_returns_a_profile(fake_run, monkeypatch):
    """The manifest is written BEFORE the run, so a killed job still records
    which modes were in flight."""

    def child(argv, log_path, **_):
        log_path.write_text("boom\n", encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(profile_cli, "_run_child", child)
    with pytest.raises(KeyboardInterrupt):
        profile_cli.profile(_args(dry_run=False))

    manifest = json.loads((_only_profile_dir(fake_run) / "profile_manifest.json").read_text())
    assert manifest["modes"]["mode"] == "cpu-only"
    assert "outcome" not in manifest


def test_green_child_with_no_profile_written_is_a_failure(fake_run, monkeypatch, caplog):
    """The --program-path failure mode: the run succeeds and reports nothing.
    Exiting 0 there would read as 'nothing was slow'."""
    monkeypatch.setattr(
        profile_cli, "_run_child", lambda argv, log, **_: log.write_text("", "utf-8") or 0
    )
    assert profile_cli.profile(_args(dry_run=False)) == 1
    assert any("wrote no profile" in r.getMessage() for r in caplog.records)


def test_child_exit_code_is_propagated(fake_run, monkeypatch):
    monkeypatch.setattr(
        profile_cli, "_run_child", lambda argv, log, **_: log.write_text("", "utf-8") or 3
    )
    assert profile_cli.profile(_args(dry_run=False)) == 3


def test_a_failed_child_that_wrote_a_profile_still_gets_its_reports(fake_run, monkeypatch):
    """Regression: a late crash (exit 1 after Scalene wrote the outfile) used to skip
    hotspots.json and the phase split, leaving a 50 MB profile nobody could read."""

    def child(argv, log_path, **_):
        Path(argv[argv.index("--outfile") + 1]).write_text("{}", encoding="utf-8")
        log_path.write_text("", encoding="utf-8")
        return 1

    monkeypatch.setattr(profile_cli, "_run_child", child)
    assert profile_cli.profile(_args(dry_run=False)) == 1

    manifest = json.loads((_only_profile_dir(fake_run) / "profile_manifest.json").read_text())
    assert manifest["outcome"]["exit_code"] == 1
    assert "hotspots" in manifest
    assert "phase_split" in manifest


def test_a_recognized_crash_is_diagnosed_into_the_manifest(fake_run, monkeypatch, caplog):
    """The production wiring, not the classifier: registering a diagnostic that
    ``profile()`` never calls is the "capability, unwired" shape of
    non-negotiable 16. Driven through the real entry point, and asserted on the
    artifact an operator reads days later rather than only on the log."""
    crash = (
        '  File "<unknown>", line 0, in std::_Sp_counted_ptr<'
        "torch::profiler::impl::Result*, (__gnu_cxx::_Lock_policy)2>::_M_dispose()\n"
    )

    def child(argv, log_path, **_):
        # NOT `log.write_text(...) or 245`: write_text returns the character
        # count, so that idiom yields the exit code only for an empty string.
        log_path.write_text(crash, encoding="utf-8")
        return 245

    monkeypatch.setattr(profile_cli, "_run_child", child)
    assert profile_cli.profile(_args(dry_run=False)) == 245

    manifest = json.loads((_only_profile_dir(fake_run) / "profile_manifest.json").read_text())
    diagnosis = manifest["outcome"]["diagnosis"]
    assert diagnosis["status"] == "recognized"
    assert diagnosis["signature"] == "scalene_torch_profiler_segfault"
    assert any("cpu-only" in r.getMessage() for r in caplog.records)


def test_an_unrecognized_crash_still_propagates_and_records(fake_run, monkeypatch):
    """The classifier must not swallow a failure it cannot name."""

    def child(argv, log_path, **_):
        log_path.write_text("odd\n", encoding="utf-8")
        return 7

    monkeypatch.setattr(profile_cli, "_run_child", child)
    assert profile_cli.profile(_args(dry_run=False)) == 7

    manifest = json.loads((_only_profile_dir(fake_run) / "profile_manifest.json").read_text())
    assert manifest["outcome"]["diagnosis"]["status"] == "unrecognized"


# --------------------------------------------------- phase split + val cap


_SRC = "/repo/src/spectramr"
_TRAIN_FRAME = f"{_SRC}/pipelines/training_loop.py _execute_training_loop:812;"
_VAL_FRAME = f"{_SRC}/pipelines/train.py _run_validation:1955;"


def _profile_with_stacks() -> str:
    """A minimal profile in scalene 2.3.0's real shape."""

    def stack(frames, cpu):
        return [frames, {"cpu_samples": cpu, "c_time": 0.0, "python_time": cpu, "count": 1}]

    return json.dumps(
        {"stacks": [stack([_TRAIN_FRAME], 7.0), stack([_TRAIN_FRAME, _VAL_FRAME], 3.0)]}
    )


def _child_writing(payload: str):
    def child(argv, log_path, **_):
        Path(argv[argv.index("--outfile") + 1]).write_text(payload, encoding="utf-8")
        log_path.write_text("ok\n", encoding="utf-8")
        return 0

    return child


def test_phase_split_writes_separate_train_and_validation_reports(fake_run, monkeypatch):
    """The whole point of the feature: one run, two reports."""
    monkeypatch.setattr(profile_cli, "_run_child", _child_writing(_profile_with_stacks()))
    assert profile_cli.profile(_args(dry_run=False)) == 0

    profile_dir = _only_profile_dir(fake_run)
    phases = profile_dir / "phases"
    assert (phases / "train.json").exists()
    assert (phases / "validation.json").exists()

    summary = json.loads((phases / "summary.json").read_text())
    assert summary["phases"]["train"]["cpu_share"] == 7.0
    assert summary["phases"]["validation"]["cpu_share"] == 3.0

    manifest = json.loads((profile_dir / "profile_manifest.json").read_text())
    assert manifest["phase_split"]["status"] == "written"


def test_a_profile_without_stacks_degrades_instead_of_failing_the_run(fake_run, monkeypatch):
    """A splitter problem must not fail a run that already paid for the profile."""
    monkeypatch.setattr(profile_cli, "_run_child", _child_writing("{}"))
    assert profile_cli.profile(_args(dry_run=False)) == 0

    manifest = json.loads((_only_profile_dir(fake_run) / "profile_manifest.json").read_text())
    assert manifest["phase_split"]["status"] == "unavailable"
    assert "stacks" in manifest["phase_split"]["reason"]


def test_unreadable_profile_is_recorded_not_inferred(fake_run, monkeypatch):
    monkeypatch.setattr(profile_cli, "_run_child", _child_writing("not json at all"))
    assert profile_cli.profile(_args(dry_run=False)) == 0

    manifest = json.loads((_only_profile_dir(fake_run) / "profile_manifest.json").read_text())
    assert manifest["phase_split"]["status"] == "unavailable"


def test_no_phase_split_says_it_was_disabled_rather_than_unavailable(fake_run, monkeypatch):
    """'Disabled' and 'unavailable' are different findings and must not blur."""
    monkeypatch.setattr(profile_cli, "_run_child", _child_writing(_profile_with_stacks()))
    assert profile_cli.profile(_args(dry_run=False, no_phase_split=True)) == 0

    profile_dir = _only_profile_dir(fake_run)
    assert not (profile_dir / "phases").exists()
    manifest = json.loads((profile_dir / "profile_manifest.json").read_text())
    assert manifest["phase_split"]["status"] == "disabled"


def test_val_batches_is_refused_where_it_could_not_act(stub_config):
    """`infer` runs no validation loop, so the cap would be silently dropped."""
    with pytest.raises(ValueError, match="--val-batches is meaningless"):
        profile_cli.profile(_args(target="infer", val_batches=4))


# ------------------------------------------------------------------ --iterations


def test_train_target_defaults_to_the_profile_iteration_budget(stub_config, capsys):
    from spectramr.cli.profile_command import DEFAULT_PROFILE_ITERATIONS

    assert profile_cli.profile(_args()) == 0
    assert f"training.max_iterations={DEFAULT_PROFILE_ITERATIONS}" in capsys.readouterr().out


def test_iterations_is_rejected_for_a_target_that_runs_no_training_loop(stub_config):
    with pytest.raises(ValueError, match="--iterations is meaningless"):
        profile_cli.profile(_args(target="infer", iterations=10))


def test_iterations_below_one_is_rejected(stub_config):
    with pytest.raises(ValueError, match=">= 1"):
        profile_cli.profile(_args(iterations=0))


def test_train_target_defaults_to_a_validation_cap(stub_config, capsys):
    from spectramr.cli.profile_command import DEFAULT_PROFILE_VAL_BATCHES
    from spectramr.cli.val_batches import VAL_BATCHES_OVERRIDE_KEY

    assert profile_cli.profile(_args()) == 0
    assert f"{VAL_BATCHES_OVERRIDE_KEY}={DEFAULT_PROFILE_VAL_BATCHES}" in capsys.readouterr().out


def test_an_explicit_val_batches_beats_the_default(stub_config, capsys):
    from spectramr.cli.val_batches import VAL_BATCHES_OVERRIDE_KEY

    assert profile_cli.profile(_args(val_batches=9)) == 0
    assert f"{VAL_BATCHES_OVERRIDE_KEY}=9" in capsys.readouterr().out


def test_infer_target_gets_no_default_cap(stub_config, capsys):
    assert profile_cli.profile(_args(target="infer")) == 0
    assert "max_iterations" not in capsys.readouterr().out


def test_torch_backend_run_is_stamped_and_declares_the_reports_it_cannot_make(fake_run, monkeypatch):
    monkeypatch.setattr(profile_cli, "cuda_visible", lambda: True)

    def child(argv, log_path, **_):
        out = Path(argv[argv.index("--out") + 1])
        (out / "summary.json").write_text("{}", encoding="utf-8")
        log_path.write_text("ok\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(profile_cli, "_run_child", child)
    assert profile_cli.profile(_args(dry_run=False, mode="full", iterations=100)) == 0

    manifest = json.loads((_only_profile_dir(fake_run) / "profile_manifest.json").read_text())
    assert manifest["modes"]["backend_requested"] == "auto"
    assert manifest["modes"]["backend_applied"] == "torch"
    assert manifest["scalene"] is None
    assert manifest["torch_profiler"]["wait_warmup_active"] == [10, 5, 20]
    assert manifest["phase_split"]["status"] == "unavailable"
    assert manifest["hotspots"]["status"] == "unavailable"
    assert manifest["outcome"]["outfile_written"] is True


def test_torch_backend_run_that_wrote_nothing_is_a_failure_not_a_pass(fake_run, monkeypatch):
    monkeypatch.setattr(profile_cli, "cuda_visible", lambda: True)

    def child(argv, log_path, **_):
        log_path.write_text("ok\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(profile_cli, "_run_child", child)
    assert profile_cli.profile(_args(dry_run=False, mode="full", iterations=100)) == 1


# ------------------------------------------------------------ intermittent retry

_SEGV = "std::_Sp_counted_ptr<torch::profiler::impl::Result*>::_M_dispose()\n"


def test_the_intermittent_profiler_crash_is_rerun_once_and_can_recover(fake_run, monkeypatch):
    calls = []

    def child(argv, log_path, **_):
        calls.append(1)
        if len(calls) == 1:
            log_path.write_text(_SEGV, encoding="utf-8")
            return 245
        log_path.write_text("", encoding="utf-8")
        Path(argv[argv.index("--outfile") + 1]).write_text(json.dumps({"files": {}}), "utf-8")
        return 0

    monkeypatch.setattr(profile_cli, "_run_child", child)
    assert profile_cli.profile(_args(dry_run=False)) == 0
    assert len(calls) == 2
    manifest = json.loads((_only_profile_dir(fake_run) / "profile_manifest.json").read_text())
    assert manifest["outcome"]["retries"] == 1


def test_a_persistent_profiler_crash_is_not_retried_forever(fake_run, monkeypatch):
    calls = []

    def child(argv, log_path, **_):
        calls.append(1)
        log_path.write_text(_SEGV, encoding="utf-8")
        return 245

    monkeypatch.setattr(profile_cli, "_run_child", child)
    assert profile_cli.profile(_args(dry_run=False)) == 245
    assert len(calls) == 1 + profile_cli.MAX_INTERMITTENT_RETRIES


def test_a_retry_spends_only_what_is_left_of_the_cap(fake_run, monkeypatch):
    """--timeout-min bounds the whole profile: the retry gets the remainder, and
    none at all once the first attempt used the budget."""
    from types import SimpleNamespace

    clock = [1000.0]
    monkeypatch.setattr(profile_cli, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    budgets = []

    def child(argv, log_path, *, timeout_s=None):
        budgets.append(timeout_s)
        clock[0] += 50.0
        log_path.write_text(_SEGV, encoding="utf-8")
        return 245

    monkeypatch.setattr(profile_cli, "_run_child", child)
    profile_cli.profile(_args(dry_run=False, timeout_min=1))
    assert budgets == [60, 10.0]

    budgets.clear()
    clock[0] = 1000.0

    def slow_child(argv, log_path, *, timeout_s=None):
        budgets.append(timeout_s)
        clock[0] += 61.0
        log_path.write_text(_SEGV, encoding="utf-8")
        return 245

    monkeypatch.setattr(profile_cli, "_run_child", slow_child)
    profile_cli.profile(_args(dry_run=False, timeout_min=1))
    assert budgets == [60]


def test_an_unrecognized_failure_is_never_retried(fake_run, monkeypatch):
    calls = []

    def child(argv, log_path, **_):
        calls.append(1)
        log_path.write_text("odd\n", encoding="utf-8")
        return 7

    monkeypatch.setattr(profile_cli, "_run_child", child)
    assert profile_cli.profile(_args(dry_run=False)) == 7
    assert len(calls) == 1


# ------------------------------------------------------------ wall-clock cap


def test_a_hung_child_is_stopped_at_the_cap(tmp_path, monkeypatch):
    """Planted violation: a child that never exits (the idle-GPU shape)."""
    import sys
    import time

    monkeypatch.setattr(profile_cli, "TIMEOUT_GRACE_S", 5)
    argv = [sys.executable, "-c", "import time; time.sleep(60)"]
    started = time.monotonic()
    code = profile_cli._run_child(argv, tmp_path / "log", timeout_s=0.5)
    assert code == profile_cli.TIMEOUT_EXIT_CODE
    assert time.monotonic() - started < 30


def test_a_child_that_ignores_the_interrupt_is_killed_after_the_grace(tmp_path, monkeypatch):
    import sys
    import time

    monkeypatch.setattr(profile_cli, "TIMEOUT_GRACE_S", 1)
    deaf = "import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(60)"
    started = time.monotonic()
    code = profile_cli._run_child([sys.executable, "-c", deaf], tmp_path / "log", timeout_s=0.5)
    assert code == profile_cli.TIMEOUT_EXIT_CODE
    assert time.monotonic() - started < 30


def test_a_grandchild_holding_the_output_pipe_is_stopped_at_the_cap(tmp_path, monkeypatch):
    """Planted violation: the child leaves a grandchild that inherits stdout and
    outlives it, the shape of a DataLoader worker or a Scalene re-exec. Signalling
    only the child left the read loop blocked on the open pipe until SLURM's limit."""
    import os
    import sys
    import time

    monkeypatch.setattr(profile_cli, "TIMEOUT_GRACE_S", 1)
    pid_file = tmp_path / "grandchild.pid"
    child = (
        "import subprocess, sys, time\n"
        "g = subprocess.Popen([sys.executable, '-c', "
        "'import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(90)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(g.pid))\n"
        "time.sleep(90)\n"
    )
    started = time.monotonic()
    code = profile_cli._run_child([sys.executable, "-c", child], tmp_path / "log", timeout_s=1)
    assert code == profile_cli.TIMEOUT_EXIT_CODE
    assert time.monotonic() - started < 30
    grandchild = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("the grandchild outlived the cap")


def test_a_child_inside_the_cap_keeps_its_own_exit_code(tmp_path):
    import sys

    argv = [sys.executable, "-c", "raise SystemExit(3)"]
    assert profile_cli._run_child(argv, tmp_path / "log", timeout_s=30) == 3


def test_a_timed_out_run_is_named_in_the_manifest(fake_run, monkeypatch):
    monkeypatch.setattr(
        profile_cli, "_run_child", lambda argv, log, **_: log.write_text("", "utf-8") or 124
    )
    assert profile_cli.profile(_args(dry_run=False, timeout_min=5)) == 124
    manifest = json.loads((_only_profile_dir(fake_run) / "profile_manifest.json").read_text())
    assert manifest["modes"]["timeout_min"] == 5
    assert manifest["outcome"]["diagnosis"]["signature"] == "profile_timeout"


def test_a_negative_cap_is_refused(stub_config):
    with pytest.raises(ValueError, match="--timeout-min"):
        profile_cli.profile(_args(timeout_min=-1))


def test_the_cap_is_on_by_default():
    from spectramr.cli.app import build_parser

    args = build_parser().parse_args(["profile", "--config", "a.yaml"])
    assert args.timeout_min == profile_cli.DEFAULT_TIMEOUT_MIN > 0
