"""Unit tests for the SLURM-array per-task dispatcher (WS-4 PR-B).

These pin the logic that used to be a fragile ``grep | sed | … || true`` block
in ``dispatch_experiments.sbatch`` — now a torch-free YAML parse. The bash
DRYRUN tests (``test_dispatch_experiments.py``) cover the shell wiring; these
cover the resolution + cap math directly, which is faster and immune to shell
quoting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

import spectramr.cli.manifest_dispatch as md
from spectramr.cli.manifest_dispatch import (
    _override_keys,
    compute_iter_cap_overrides,
    read_overrides_file,
    resolve_manifest_config,
)


def _manifest(tmp_path: Path) -> Path:
    m = tmp_path / "manifest.txt"
    m.write_text(
        "experiments/inprogress/vf/exp_vf_04.yaml\n"
        "experiments/inprogress/vf/exp_vf_33.yaml\n"
    )
    return m


# --- manifest resolution -----------------------------------------------------


def test_resolve_by_index(tmp_path: Path) -> None:
    m = _manifest(tmp_path)
    assert resolve_manifest_config(m, 0).endswith("exp_vf_04.yaml")
    assert resolve_manifest_config(m, 1).endswith("exp_vf_33.yaml")


def test_resolve_ignores_blank_lines(tmp_path: Path) -> None:
    """A trailing newline / blank line must not become a phantom (missing) arm."""
    m = tmp_path / "m.txt"
    m.write_text("a.yaml\n\nb.yaml\n\n")
    assert resolve_manifest_config(m, 1).endswith("b.yaml")
    with pytest.raises(IndexError, match="out of range"):
        resolve_manifest_config(m, 2)  # only 2 real entries, not 4


def test_resolve_out_of_range_raises(tmp_path: Path) -> None:
    with pytest.raises(IndexError, match="out of range"):
        resolve_manifest_config(_manifest(tmp_path), 999)


# --- TRAIN_ITERS cap (the production-incident logic) -------------------------


def _cfg(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(body)
    return p


def test_cap_shrinks_both_iterations_and_eval_interval(tmp_path: Path) -> None:
    # max_iterations 500000 (> cap) and eval_interval 500 (> cap) → both capped.
    cfg = _cfg(
        tmp_path,
        "training:\n  max_iterations: 500000\nvalidation:\n  eval_interval: 500\n",
    )
    overrides, _ = compute_iter_cap_overrides(cfg, 100)
    assert "--override" in overrides
    assert "training.max_iterations=100" in overrides
    assert "validation.eval_interval=100" in overrides


def test_cap_does_not_inflate_short_calibration_arm(tmp_path: Path) -> None:
    # A 1-shot calibration arm (max_iterations 1) must NOT be inflated to 100.
    cfg = _cfg(
        tmp_path,
        "training:\n  max_iterations: 1\nvalidation:\n  eval_interval: 1\n",
    )
    overrides, messages = compute_iter_cap_overrides(cfg, 100)
    assert "training.max_iterations=100" not in overrides
    assert "validation.eval_interval=100" not in overrides
    assert any("keeping config max_iterations=1" in m for m in messages)


def test_cap_strips_inline_comment_without_sed(tmp_path: Path) -> None:
    # `max_iterations: 1  # VF plan CW-6` — yaml.safe_load drops the comment
    # natively, so the value is 1 (not "116"); the bash needed a sed hack.
    cfg = _cfg(
        tmp_path,
        "training:\n  max_iterations: 1  # VF plan CW-6\n"
        "validation:\n  eval_interval: 2500\n",
    )
    overrides, messages = compute_iter_cap_overrides(cfg, 100)
    assert "training.max_iterations=100" not in overrides  # NOT inflated
    assert any("keeping config max_iterations=1" in m for m in messages)
    assert "validation.eval_interval=1" in overrides  # eval capped to eff iters


def test_cap_survives_config_without_eval_interval(tmp_path: Path) -> None:
    # THE regression: a config that omits eval_interval. The bash grep exited 1
    # and `set -e` killed the task (56 mrixfields tasks, job 7145522). Here the
    # missing key is just None → the cap branch injects eval_interval cleanly.
    cfg = _cfg(tmp_path, "config_version: '1.0'\ntraining:\n  max_iterations: 500000\n")
    overrides, _ = compute_iter_cap_overrides(cfg, 100)
    assert "training.max_iterations=100" in overrides
    assert "validation.eval_interval=100" in overrides  # injected for the missing key


def test_cap_reads_the_canonical_schedule_block(tmp_path: Path) -> None:
    """A migrated arm declares ``validation.schedule.interval_steps``.

    Every other fixture in this file spells the interval FLAT, which is what
    hid the drift: the code read ``validation.eval_interval`` and so did the
    tests, so both stayed wrong together while the schema moved the leaf to
    ``validation.schedule.interval_steps`` (RENAMES, 2026-07-31). On a migrated
    arm the read returned ``None``, which reads as "absent" -- safe for the cap
    itself, but it made the already-fires branch unreachable and inflated any
    arm declaring a SMALLER interval than the cap, contradicting the
    cap-not-inflate contract in the docstring.

    50 < 100 here, so the correct answer is to leave it alone.
    """
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "training:\n"
        "  max_iterations: 500000\n"
        "validation:\n"
        "  schedule:\n"
        "    interval_steps: 50\n"
    )
    overrides, messages = compute_iter_cap_overrides(cfg, 100)
    assert "validation.eval_interval=100" not in overrides
    assert any("keeping config eval_interval=50" in m for m in messages)


def test_cap_prefers_the_canonical_spelling_over_the_legacy_one(
    tmp_path: Path,
) -> None:
    """Both present and disagreeing: the canonical block wins.

    The loader folds the legacy name INTO the block, so a file carrying both is
    resolved by the block. Reading the flat one first would cap against a value
    the run never uses.
    """
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "training:\n"
        "  max_iterations: 500000\n"
        "validation:\n"
        "  eval_interval: 5000\n"
        "  schedule:\n"
        "    interval_steps: 50\n"
    )
    _, messages = compute_iter_cap_overrides(cfg, 100)
    assert any("keeping config eval_interval=50" in m for m in messages)


def test_cap_leaves_eval_interval_that_already_fires(tmp_path: Path) -> None:
    # eval_interval (50) already fires within the capped run (eff=100) → no cap.
    # Legacy flat spelling: still read, as the fallback for unmigrated arms.
    cfg = _cfg(
        tmp_path,
        "training:\n  max_iterations: 500000\nvalidation:\n  eval_interval: 50\n",
    )
    overrides, messages = compute_iter_cap_overrides(cfg, 100)
    assert "validation.eval_interval=100" not in overrides
    assert any("keeping config eval_interval=50" in m for m in messages)


# --- audit gate (moved out of bash — must stay fail-loud) ---------------------


class _FakeRun:
    """Record subprocess.run calls and return a canned returncode per verb."""

    def __init__(self, audit_rc: int = 0, train_rc: int = 0) -> None:
        self.audit_rc = audit_rc
        self.train_rc = train_rc
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)

        class _R:
            returncode = self.train_rc if "train" in cmd else self.audit_rc

        return _R()


def _real_cfg(tmp_path: Path) -> tuple[Path, Path]:
    """A manifest + an actually-existing config (manifest_dispatch is_file-checks)."""
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  max_iterations: 5\n")
    man = tmp_path / "m.txt"
    man.write_text(str(cfg) + "\n")
    return man, cfg


def test_audit_failure_returns_2_and_skips_train(tmp_path, monkeypatch) -> None:
    man, _ = _real_cfg(tmp_path)
    fake = _FakeRun(audit_rc=2)
    monkeypatch.setattr(md.subprocess, "run", fake)
    rc = md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=None,
        resume=False,
        no_audit=False,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=False,
    )
    assert rc == 2
    assert not any("train" in c for c in fake.calls)  # train never reached


def test_runs_audit_then_train_in_order(tmp_path, monkeypatch) -> None:
    man, _ = _real_cfg(tmp_path)
    fake = _FakeRun(audit_rc=0, train_rc=0)
    monkeypatch.setattr(md.subprocess, "run", fake)
    rc = md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=None,
        resume=False,
        no_audit=False,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=False,
    )
    assert rc == 0
    verbs = [
        "audit" if "audit" in c else "train" if "train" in c else "?"
        for c in fake.calls
    ]
    assert verbs == ["audit", "train"]  # audit pre-flight precedes train


def test_no_audit_skips_preflight(tmp_path, monkeypatch) -> None:
    man, _ = _real_cfg(tmp_path)
    fake = _FakeRun()
    monkeypatch.setattr(md.subprocess, "run", fake)
    md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=None,
        resume=False,
        no_audit=True,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=False,
    )
    assert not any("audit" in c for c in fake.calls)
    assert any("train" in c for c in fake.calls)


def test_dry_run_never_shells_out(tmp_path, monkeypatch) -> None:
    man, _ = _real_cfg(tmp_path)
    fake = _FakeRun()
    monkeypatch.setattr(md.subprocess, "run", fake)
    rc = md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=100,
        resume=False,
        no_audit=False,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=True,
    )
    assert rc == 0
    assert fake.calls == []  # dry-run plans only — no audit/train subprocess


# --- --prod: run the actual experiment (full max_iterations, no smoke cap) ----
# --prod is the opposite of the TRAIN_ITERS smoke cap: it asserts a production run
# (no override) and stamps the mode into the SLURM .out. Combining it with a smoke
# cap is contradictory and must fail loud, not silently let one win (#9/#15).


def test_prod_rejects_train_iters(tmp_path) -> None:
    man, _ = _real_cfg(tmp_path)
    rc = md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=100,
        resume=False,
        no_audit=True,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=True,
        prod=True,
    )
    assert rc == 2  # mutually-exclusive: --prod + --train-iters is rejected


def test_prod_dry_run_stamps_mode_and_no_override(tmp_path, capsys) -> None:
    man, _ = _real_cfg(tmp_path)
    rc = md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=None,
        resume=False,
        no_audit=True,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=True,
        prod=True,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "[MODE] PROD" in out
    assert "--override" not in out  # production run — no smoke-cap overrides


def test_prod_passes_no_override_to_train(tmp_path, monkeypatch) -> None:
    man, _ = _real_cfg(tmp_path)
    fake = _FakeRun()
    monkeypatch.setattr(md.subprocess, "run", fake)
    rc = md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=None,
        resume=False,
        no_audit=True,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=False,
        prod=True,
    )
    assert rc == 0
    train_cmd = next(c for c in fake.calls if "train" in c)
    assert "--override" not in train_cmd  # --prod runs the FULL config max_iterations


def test_prod_flag_parsed_by_main(tmp_path, monkeypatch) -> None:
    """The CLI exposes --prod and forwards it through to _dispatch."""
    man, _ = _real_cfg(tmp_path)
    seen = {}

    def _fake_dispatch(**kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(md, "_dispatch", _fake_dispatch)
    rc = md.main(["--manifest", str(man), "--index", "0", "--prod", "--dry-run"])
    assert rc == 0
    assert seen.get("prod") is True


# --- --output-base: route each task's output into a campaign tree -------------
# When the campaign orchestrator submits a cohort as ONE array job, each task
# must write its checkpoints/CSVs into <campaign_dir>/<config-stem> so the
# orchestrator's _discover_results finds them. --output-base appends a
# `training.output_dir` override pointing there. None (default) keeps the
# config's own output_dir untouched, so the standalone array dispatcher is
# byte-for-byte unchanged.


def test_output_base_overrides_training_output_dir(tmp_path, monkeypatch) -> None:
    man, _cfg = _real_cfg(tmp_path)  # _cfg stem == "arm"
    fake = _FakeRun()
    monkeypatch.setattr(md.subprocess, "run", fake)
    base = tmp_path / "campaign"
    rc = md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=None,
        resume=False,
        no_audit=True,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=False,
        prod=True,
        output_base=str(base),
    )
    assert rc == 0
    train_cmd = next(c for c in fake.calls if "train" in c)
    joined = " ".join(train_cmd)
    assert "--override" in train_cmd
    # routed into <base>/<stem>
    assert f"training.output_dir={base / 'arm'}" in joined


def test_output_base_none_leaves_output_dir_untouched(tmp_path, monkeypatch) -> None:
    man, _ = _real_cfg(tmp_path)
    fake = _FakeRun()
    monkeypatch.setattr(md.subprocess, "run", fake)
    rc = md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=None,
        resume=False,
        no_audit=True,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=False,
        prod=True,
        output_base=None,
    )
    assert rc == 0
    train_cmd = next(c for c in fake.calls if "train" in c)
    assert "training.output_dir" not in " ".join(train_cmd)


def test_output_base_composes_with_smoke_cap(tmp_path, monkeypatch) -> None:
    # --output-base routing and the TRAIN_ITERS cap overrides must both land.
    man, _ = _real_cfg(tmp_path)
    fake = _FakeRun()
    monkeypatch.setattr(md.subprocess, "run", fake)
    base = tmp_path / "camp"
    md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=3,
        resume=False,
        no_audit=True,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=False,
        prod=False,
        output_base=str(base),
    )
    train_cmd = next(c for c in fake.calls if "train" in c)
    joined = " ".join(train_cmd)
    assert "training.max_iterations=3" in joined  # smoke cap
    assert f"training.output_dir={base / 'arm'}" in joined  # routing


def test_output_base_flag_parsed_by_main(tmp_path, monkeypatch) -> None:
    man, _ = _real_cfg(tmp_path)
    seen = {}

    def _fake_dispatch(**kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(md, "_dispatch", _fake_dispatch)
    rc = md.main(
        [
            "--manifest",
            str(man),
            "--index",
            "0",
            "--dry-run",
            "--output-base",
            "/camp/dir",
        ]
    )
    assert rc == 0
    assert seen.get("output_base") == "/camp/dir"


# --- launch verb: the array's 36/45 failure ----------------------------------
#
# Array 8589967 (2026-09-12) lost 36 of 45 tasks because the dispatcher spelled
# `spectramr train` for every arm. `train` never calls `setup_distributed`, so
# each kspace_filling arm -- all 60 declare `parallel.strategy: deepspeed` --
# paid for the audit and the model build and then raised out of
# `_require_process_group` at Stage B. The framework was right to refuse; the
# dispatcher simply did not know the second verb.


def _cfg_with_strategy(tmp_path: Path, strategy: str | None) -> Path:
    """An arm whose `parallel:` block is spelled exactly as *strategy* asks.

    `None` writes a NULL `parallel:` key rather than omitting it -- the shape
    that reads as a dict everywhere except where it is `None`, and the one that
    broke the old bash grep.
    """
    body = "training:\n  max_iterations: 5\n"
    if strategy is None:
        body += "parallel:\n"
    else:
        body += f"parallel:\n  strategy: {strategy}\n"
    cfg = tmp_path / f"arm_{strategy or 'null'}.yaml"
    cfg.write_text(body)
    return cfg


def test_declared_strategy_is_read_from_the_yaml(tmp_path: Path) -> None:
    assert md.read_parallel_strategy(_cfg_with_strategy(tmp_path, "deepspeed")) == "deepspeed"
    assert md.read_parallel_strategy(_cfg_with_strategy(tmp_path, "ddp")) == "ddp"


@pytest.mark.parametrize("body", ["training:\n  max_iterations: 5\n", "parallel:\n", "{}\n", ""])
def test_absent_or_null_parallel_block_reads_as_the_schema_default(
    tmp_path: Path, body: str
) -> None:
    """`none` is what the schema itself defaults to, so the raw read and the
    loaded config agree -- an arm with no `parallel:` block is not distributed."""
    cfg = tmp_path / "arm.yaml"
    cfg.write_text(body)
    assert md.read_parallel_strategy(cfg) == "none"


def test_process_group_predicate_comes_from_the_strategy_registry() -> None:
    """`dp` is the case a hardcoded name list gets wrong in the expensive
    direction.

    `nn.DataParallel` is genuinely single-process, so the tempting
    `strategy != "none"` shorthand would launch it under torchrun for no reason.
    The registry answers per plugin, via `requires_process_group`, which is why
    this is a query and not a literal set here (non-negotiable 17).
    """
    assert md.requires_process_group("deepspeed") is True
    assert md.requires_process_group("ddp") is True
    assert md.requires_process_group("fsdp") is True
    assert md.requires_process_group("dp") is False
    assert md.requires_process_group("none") is False


def test_single_process_arm_keeps_the_train_verb() -> None:
    argv = md.build_launch_argv("a.yaml", [], False, distributed=False)
    assert argv[1:] == ["-m", "spectramr.cli", "train", "--config", "a.yaml"]
    assert "torch.distributed.run" not in argv


def test_process_group_arm_gets_torchrun_and_the_distributed_verb() -> None:
    """The exact defect: a deepspeed arm must not reach the `train` verb."""
    argv = md.build_launch_argv("a.yaml", [], False, distributed=True, nproc_per_node=4)
    assert argv[1:3] == ["-m", "torch.distributed.run"]
    assert "--nproc_per_node=4" in argv
    assert "train-distributed" in argv
    assert "train" not in argv, "the single-process verb must not survive"


def test_the_rendezvous_is_static_on_loopback() -> None:
    """`--standalone` selects the c10d DYNAMIC backend, whose store server
    resolves `socket.gethostname()` whatever endpoint it is given.

    On a host whose own name is not in /etc/hosts that hangs for the full 300 s
    store timeout and dies with `DistNetworkError: client socket has timed out
    … trying to connect to (<hostname>, <port>)`, naming neither the cause nor
    the fix — observed on this machine with both `--standalone` and an explicit
    `--rdzv_endpoint=127.0.0.1:0`, so the endpoint is not the lever, the backend
    is. A single-node launch needs loopback and nothing else.
    """
    argv = md.build_launch_argv("a.yaml", [], False, distributed=True, master_port=29511)
    assert "--standalone" not in argv
    assert "--master_addr=127.0.0.1" in argv
    assert "--master_port=29511" in argv
    assert "--nnodes=1" in argv


def test_rendezvous_port_is_derived_from_the_per_task_job_id() -> None:
    """`SLURM_JOB_ID` differs per array task; `SLURM_ARRAY_JOB_ID` is the shared
    one. Deriving from the shared id would have two tasks on one node — which
    happens: 31 and 35 of array 8589967 both ran on compute-2-5 — fight over a
    single socket.
    """
    a = md.resolve_rendezvous_port({"SLURM_JOB_ID": "8590053"})
    b = md.resolve_rendezvous_port({"SLURM_JOB_ID": "8590054"})
    assert a != b
    assert 20000 <= a < 40000 and 20000 <= b < 40000


def test_rendezvous_port_falls_back_when_there_is_no_scheduler() -> None:
    """A local run has no SLURM_JOB_ID; asking the OS beats a fixed constant."""
    port = md.resolve_rendezvous_port({})
    assert 1024 < port < 65536


def test_the_launcher_is_the_venvs_own_interpreter() -> None:
    """`sys.executable -m torch.distributed.run`, never a bare `torchrun`.

    A `torchrun` found on PATH can belong to a different interpreter than the
    one the .sbatch activated -- the same shadowing failure the FreeSurfer
    LD_LIBRARY_PATH scrub exists for, arriving by a different route.
    """
    argv = md.build_launch_argv("a.yaml", [], False, distributed=True)
    assert argv[0] == sys.executable
    assert "torchrun" not in argv


@pytest.mark.parametrize("distributed", [False, True])
def test_resume_and_overrides_are_spelled_the_same_for_both_verbs(distributed: bool) -> None:
    """`train-distributed` takes the same `--resume` / `--override` flags, so a
    smoke cap or a campaign output route must not be lost by taking one branch."""
    argv = md.build_launch_argv(
        "a.yaml",
        ["--override", "training.max_iterations=7"],
        True,
        distributed=distributed,
    )
    assert argv[-4:] == [
        "--resume",
        "if-present",
        "--override",
        "training.max_iterations=7",
    ]


@pytest.mark.unit
def test_the_dispatcher_never_spells_resume_auto() -> None:
    """`auto` raises when the checkpoint dir is empty, and a requeued array task
    re-runs this identical command — so the first attempt of every chain would
    fail outright. `if-present` is the only spelling a chain can carry."""
    for distributed in (False, True):
        argv = md.build_launch_argv("a.yaml", [], True, distributed=distributed)
        assert "auto" not in argv, argv
        assert "if-present" in argv


def test_dispatch_launches_a_deepspeed_arm_under_torchrun(tmp_path, monkeypatch) -> None:
    """End to end through `_dispatch`, which is where the verb was hardcoded."""
    cfg = _cfg_with_strategy(tmp_path, "deepspeed")
    man = tmp_path / "m.txt"
    man.write_text(str(cfg) + "\n")
    fake = _FakeRun()
    monkeypatch.setattr(md.subprocess, "run", fake)
    rc = md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=None,
        resume=False,
        no_audit=True,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=False,
        nproc_per_node=2,
    )
    assert rc == 0
    launch = " ".join(fake.calls[-1])
    assert "torch.distributed.run" in launch
    assert "--nproc_per_node=2" in launch
    assert "train-distributed" in launch


def test_dispatch_keeps_train_for_an_arm_that_declares_no_parallelism(
    tmp_path, monkeypatch
) -> None:
    """Anti-vacuity for the test above: the fix must not send everything to
    torchrun, which would be a slower, differently-seeded run for every arm in
    the corpus that never asked for one."""
    cfg = _cfg_with_strategy(tmp_path, None)
    man = tmp_path / "m.txt"
    man.write_text(str(cfg) + "\n")
    fake = _FakeRun()
    monkeypatch.setattr(md.subprocess, "run", fake)
    md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=None,
        resume=False,
        no_audit=True,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=False,
    )
    launch = fake.calls[-1]
    assert "train" in launch
    assert "torch.distributed.run" not in " ".join(launch)


def test_nproc_per_node_flag_reaches_dispatch(tmp_path, monkeypatch) -> None:
    """The .sbatch passes the derived GPU count through this flag; a name change
    that argparse accepts but `_dispatch` ignores would silently run one rank."""
    man, _ = _real_cfg(tmp_path)
    seen = {}
    monkeypatch.setattr(md, "_dispatch", lambda **kw: seen.update(kw) or 0)
    assert md.main(["--manifest", str(man), "--index", "0", "--dry-run", "--nproc-per-node", "8"]) == 0
    assert seen.get("nproc_per_node") == 8


# ---------------------------------------------------------------------------
# --overrides-file: a frozen key=value list applied to every array task.
# ---------------------------------------------------------------------------


class TestOverridesFile:
    """The passthrough exists because `sbatch --export` splits on commas."""

    def test_reads_key_value_lines(self, tmp_path):
        f = tmp_path / "ov.txt"
        f.write_text("training.max_iterations=2\nvalidation.loader.num_batches=2\n")
        assert read_overrides_file(f) == [
            "--override",
            "training.max_iterations=2",
            "--override",
            "validation.loader.num_batches=2",
        ]

    def test_comma_in_value_survives(self, tmp_path):
        """The shape an --export-borne knob would truncate."""
        f = tmp_path / "ov.txt"
        f.write_text("data.transforms=[a,b,c]\n")
        assert read_overrides_file(f) == ["--override", "data.transforms=[a,b,c]"]

    def test_blank_and_comment_lines_skipped(self, tmp_path):
        f = tmp_path / "ov.txt"
        f.write_text("# a note\n\ntraining.max_iterations=2\n\n")
        assert read_overrides_file(f) == ["--override", "training.max_iterations=2"]

    @pytest.mark.parametrize("bad", ["training.max_iterations", "=2", "   =  3"])
    def test_malformed_line_raises_rather_than_dropping(self, tmp_path, bad):
        """A silently-skipped override is indistinguishable from one applied."""
        f = tmp_path / "ov.txt"
        f.write_text(f"{bad}\n")
        with pytest.raises(ValueError, match="must be 'key=value'"):
            read_overrides_file(f)

    def test_error_names_the_line_number(self, tmp_path):
        f = tmp_path / "ov.txt"
        f.write_text("training.max_iterations=2\nbroken\n")
        with pytest.raises(ValueError, match=r":2:"):
            read_overrides_file(f)

    def test_override_keys_extracts_dotted_keys(self):
        args = ["--override", "a.b=1", "--override", "c.d=2"]
        assert _override_keys(args) == ["a.b", "c.d"]


class TestOverridesFileCapCollision:
    """A key the TRAIN_ITERS cap already set is refused, not resolved."""

    def _cfg(self, tmp_path):
        cfg = tmp_path / "arm.yaml"
        cfg.write_text("training:\n  max_iterations: 70000\n")
        man = tmp_path / "manifest.txt"
        man.write_text(f"{cfg}\n")
        return man

    def test_collision_with_cap_exits_2(self, tmp_path, capsys):
        ov = tmp_path / "ov.txt"
        ov.write_text("training.max_iterations=2\n")
        rc = md._dispatch(
            manifest=str(self._cfg(tmp_path)),
            index=0,
            train_iters=50,
            resume=False,
            no_audit=True,
            dispatch_dir=str(tmp_path),
            dry_run=True,
            overrides_file=str(ov),
        )
        assert rc == 2
        assert "already set" in capsys.readouterr().err

    def test_non_colliding_key_is_appended(self, tmp_path, capsys):
        ov = tmp_path / "ov.txt"
        ov.write_text("validation.loader.num_batches=2\n")
        rc = md._dispatch(
            manifest=str(self._cfg(tmp_path)),
            index=0,
            train_iters=50,
            resume=False,
            no_audit=True,
            dispatch_dir=str(tmp_path),
            dry_run=True,
            overrides_file=str(ov),
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert "validation.loader.num_batches=2" in out
        assert "training.max_iterations=50" in out  # the cap still applies

    def test_no_cap_means_no_collision_possible(self, tmp_path, capsys):
        ov = tmp_path / "ov.txt"
        ov.write_text("training.max_iterations=2\n")
        rc = md._dispatch(
            manifest=str(self._cfg(tmp_path)),
            index=0,
            train_iters=None,
            resume=False,
            no_audit=True,
            dispatch_dir=str(tmp_path),
            dry_run=True,
            overrides_file=str(ov),
        )
        assert rc == 0
        assert "training.max_iterations=2" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# --verb / --verb-args-file: the array runs a named pipeline, not only `train`.
#
# The submit wrapper's command form (`… spectramr <verb> <yamls> <flags>`) ends
# here: the verb reaches every task through TASK_VERB and the trailing flags
# through a frozen file. Two things must hold whatever the verb — the config is
# still injected per task, and a verb that cannot spell a knob this submission
# carries is refused ONCE rather than by argparse on every task.
# ---------------------------------------------------------------------------


class TestVerbSelection:
    def test_the_named_verb_reaches_the_launch(self) -> None:
        argv = md.build_launch_argv("a.yaml", [], False, distributed=False, verb="infer")
        assert argv[3] == "infer"
        assert argv[4:6] == ["--config", "a.yaml"]

    def test_verb_args_land_after_the_overrides(self) -> None:
        """Last wins in argparse, so the submitter's own flags must come last:
        anything they spell has to beat what the dispatcher added."""
        argv = md.build_launch_argv(
            "a.yaml",
            ["--override", "training.max_iterations=5"],
            True,
            distributed=False,
            verb_args=["--seed", "7"],
        )
        assert argv[-2:] == ["--seed", "7"]
        assert argv.index("--override") < argv.index("--seed")

    def test_the_process_group_upgrade_is_train_only(self) -> None:
        """`parallel.strategy` describes how the arm TRAINS. Pairing the
        torchrun upgrade with another verb would launch `train-distributed` for
        a submission that asked for inference, so it is a caller bug, not a
        default to resolve."""
        with pytest.raises(ValueError, match="train-only"):
            md.build_launch_argv("a.yaml", [], False, distributed=True, verb="infer")

    def test_a_deepspeed_arm_under_a_non_train_verb_runs_single_process(
        self, tmp_path, monkeypatch
    ) -> None:
        cfg = _cfg_with_strategy(tmp_path, "deepspeed")
        man = tmp_path / "m.txt"
        man.write_text(str(cfg) + "\n")
        fake = _FakeRun()
        monkeypatch.setattr(md.subprocess, "run", fake)
        rc = md._dispatch(
            manifest=str(man),
            index=0,
            train_iters=None,
            resume=False,
            no_audit=True,
            dispatch_dir=str(tmp_path / "d"),
            dry_run=False,
            verb="infer",
        )
        assert rc == 0
        launch = " ".join(fake.calls[-1])
        assert "torch.distributed.run" not in launch
        assert "train-distributed" not in launch
        assert " infer --config " in launch

    def test_verb_and_args_flags_reach_dispatch(self, tmp_path, monkeypatch) -> None:
        man, _ = _real_cfg(tmp_path)
        seen: dict = {}
        monkeypatch.setattr(md, "_dispatch", lambda **kw: seen.update(kw) or 0)
        assert (
            md.main(
                [
                    "--manifest",
                    str(man),
                    "--index",
                    "0",
                    "--dry-run",
                    "--verb",
                    "infer",
                    "--verb-args-file",
                    "/tmp/verb_args.txt",
                ]
            )
            == 0
        )
        assert seen.get("verb") == "infer"
        assert seen.get("verb_args_file") == "/tmp/verb_args.txt"

    def test_dispatch_forwards_the_frozen_args_to_the_launch(
        self, tmp_path, capsys
    ) -> None:
        cfg = tmp_path / "arm.yaml"
        cfg.write_text("training:\n  max_iterations: 5\n")
        man = tmp_path / "m.txt"
        man.write_text(str(cfg) + "\n")
        args = tmp_path / "verb_args.txt"
        args.write_text("--debug\n--device\ncuda\n")
        rc = md._dispatch(
            manifest=str(man),
            index=0,
            train_iters=None,
            resume=False,
            no_audit=True,
            dispatch_dir=str(tmp_path / "d"),
            dry_run=True,
            verb_args_file=str(args),
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert out.rstrip().endswith("--debug --device cuda")


class TestVerbArgsFile:
    def test_one_argv_element_per_line_keeps_its_spaces(self, tmp_path) -> None:
        """The wrapper freezes one element per line precisely so a value with a
        space in it survives; splitting here would undo that."""
        f = tmp_path / "a.txt"
        f.write_text("--note\nrun 3 of 4\n")
        assert md.read_verb_args_file(f) == ["--note", "run 3 of 4"]

    def test_blank_lines_and_comments_are_dropped(self, tmp_path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("# why\n\n--debug\n")
        assert md.read_verb_args_file(f) == ["--debug"]


class TestVerbPlanValidation:
    def test_train_is_accepted_with_every_knob(self) -> None:
        assert md.validate_verb_plan("train", resume=True, overrides=True) is None

    def test_a_verb_whose_config_is_not_a_flag_is_refused(self) -> None:
        """`audit` takes its config POSITIONALLY, so the `--config <path>` this
        array injects is an argparse error on every task."""
        error = md.validate_verb_plan("audit", resume=False, overrides=False)
        assert error is not None
        assert "--config" in error
        assert "train" in error  # names what can be dispatched instead

    def test_predict_is_refused_and_points_at_infer(self) -> None:
        error = md.validate_verb_plan("predict", resume=False, overrides=False)
        assert error is not None and "infer" in error

    def test_overrides_need_a_verb_that_spells_override(self) -> None:
        assert md.validate_verb_plan("infer", resume=False, overrides=True) is not None
        assert md.validate_verb_plan("infer", resume=False, overrides=False) is None

    def test_resume_needs_a_verb_that_spells_resume(self) -> None:
        assert md.validate_verb_plan("infer", resume=True, overrides=False) is not None

    def test_the_option_set_comes_from_the_real_parser(self) -> None:
        """Anti-vacuity for the two checks above: the answers are the parser's,
        so a flag added to (or dropped from) a verb changes them without anyone
        editing a table here."""
        from spectramr.cli.app import build_parser

        for verb in ("train", "infer"):
            parser_opts = {
                opt
                for action in build_parser()._actions
                if isinstance(action, argparse._SubParsersAction)
                for a in action.choices[verb]._actions
                for opt in a.option_strings
            }
            assert md.verb_option_strings(verb) == parser_opts
        assert md.verb_option_strings("no-such-verb") == frozenset()

    def test_an_impossible_plan_is_refused_before_the_manifest_is_read(
        self, tmp_path
    ) -> None:
        """The refusal is identical on every task, so it must not cost a
        manifest read, an audit or a queue slot to discover."""
        rc = md._dispatch(
            manifest=str(tmp_path / "does-not-exist.txt"),
            index=0,
            train_iters=None,
            resume=False,
            no_audit=True,
            dispatch_dir=str(tmp_path / "d"),
            dry_run=True,
            verb="audit",
        )
        assert rc == 2

    def test_check_plan_needs_no_manifest(self, capsys) -> None:
        assert md.main(["--check-plan", "--verb", "train"]) == 0
        assert "plan accepted" in capsys.readouterr().out

    def test_check_plan_reports_the_refusal(self, capsys) -> None:
        assert md.main(["--check-plan", "--verb", "infer", "--resume"]) == 2
        assert "--resume" in capsys.readouterr().err

    def test_dispatching_still_requires_a_manifest(self) -> None:
        """--check-plan relaxed the requiredness; a real dispatch must not have
        become a run with no arms."""
        with pytest.raises(SystemExit):
            md.main(["--verb", "train"])
