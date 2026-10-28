"""Tests for the campaign-aware distributed launcher (PR-14)."""

from __future__ import annotations

from spectramr.infrastructure.distributed.launcher import (
    dispatch_for_campaign,
    launch_distributed,
)


class _Settings:
    def __init__(self, num_devices=1, num_nodes=1, strategy="none"):
        self.parallel = type("P", (), {
            "num_devices": num_devices,
            "num_nodes": num_nodes,
            "strategy": strategy,
            "backend": "nccl",
        })()


def test_dispatch_single_process_runs_directly() -> None:
    settings = _Settings(num_devices=1, num_nodes=1, strategy="none")
    log = []

    def run(rank, world_size, settings):
        log.append((rank, world_size))
        return "ok"

    out = dispatch_for_campaign(run, settings, arm_name="test")
    assert out == "ok"
    assert log == [(0, 1)]


def test_dispatch_no_parallel_block_runs_directly() -> None:
    class _NoParallel:
        pass

    log = []

    def run(rank, world_size, settings):
        log.append(rank)

    dispatch_for_campaign(run, _NoParallel(), arm_name="bare")
    assert log == [0]


def test_launch_distributed_world_size_1_short_circuit() -> None:
    """world_size=1 should skip the spawn machinery entirely."""
    log = []

    def run(rank, world_size, settings):
        log.append((rank, world_size))
        return "executed"

    out = launch_distributed(run, settings=None, num_devices=1, num_nodes=1)
    assert out == "executed"
    assert log == [(0, 1)]


def test_launch_multi_node_returns_torchrun_command() -> None:
    """Multi-node launch is a fan-out responsibility of the cluster job
    script — we return the torchrun cmd so it can be exec'd remotely."""
    cmd = launch_distributed(
        run_fn=lambda **k: None,
        settings=None,
        num_devices=4,
        num_nodes=2,
    )
    assert isinstance(cmd, str)
    assert "torchrun" in cmd
    assert "--nproc_per_node=4" in cmd
    assert "--nnodes=2" in cmd


# ---------------------------------------------------------------------------
# W6: the emitted torchrun line named a subcommand that cannot open a group.
#
# `test_launch_multi_node_returns_torchrun_command` above asserts "torchrun" is
# in the string and checks both `--nproc_per_node` and `--nnodes` -- and that is
# precisely why the wrong subcommand survived: every flag was right and the
# thing being launched was not. A user who copied this line got
# `_require_process_group` raising during `adopt`, at Stage B, after the model
# and data were already built -- a crash that reads like a DDP/DeepSpeed problem
# rather than a launcher typo.
# ---------------------------------------------------------------------------


def _emitted_command() -> str:
    return launch_distributed(
        run_fn=lambda **k: None,
        settings=None,
        num_devices=4,
        num_nodes=2,
    )


def test_the_emitted_command_launches_the_distributed_entry_point() -> None:
    """`train` never calls `setup_distributed`, so no process group exists."""
    cmd = _emitted_command()
    assert "-m spectramr.cli train-distributed" in cmd, (
        f"the launcher emits a subcommand that cannot open a process group: {cmd}"
    )


def test_it_does_not_emit_the_single_process_subcommand() -> None:
    """Anti-vacuity: `"train-distributed" in cmd` is also true of the broken
    string `train --config` if `train-distributed` appeared anywhere else, and a
    future edit could reintroduce plain `train` alongside it."""
    cmd = _emitted_command()
    tokens = cmd.split()
    idx = tokens.index("-m")
    assert tokens[idx + 2] == "train-distributed", (
        f"the token after the module is {tokens[idx + 2]!r}, not the "
        "distributed entry point"
    )


def test_the_subcommand_is_one_the_cli_actually_registers() -> None:
    """Pinned against the real parser, not a hard-coded string: the launcher and
    the CLI drifting apart is the whole defect, so a rename of either must fail
    here rather than ship a command line nobody runs locally."""
    from spectramr.cli.app import build_parser

    parser = build_parser()
    choices: set[str] = set()
    for action in parser._actions:
        if getattr(action, "choices", None) and action.dest == "command":
            choices = set(action.choices)
            break
    assert choices, "could not introspect the CLI subcommands"

    tokens = _emitted_command().split()
    subcommand = tokens[tokens.index("-m") + 2]
    assert subcommand in choices, (
        f"the launcher emits `{subcommand}`, which the CLI does not register. "
        f"Registered: {sorted(choices)}"
    )


class TestRendezvousPort:
    def test_free_port_is_a_usable_unprivileged_port(self):
        import socket

        from spectramr.infrastructure.distributed.launcher import _free_port

        port = _free_port()
        assert 1024 < port < 65536
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))  # still free

    def test_single_node_spawn_gets_a_port_unless_one_is_declared(self, monkeypatch):
        from unittest.mock import patch

        from spectramr.infrastructure.distributed import launcher

        monkeypatch.delenv("MASTER_PORT", raising=False)
        with patch("torch.multiprocessing.spawn") as spawn:
            launcher.launch_distributed(lambda **_: None, object(), num_devices=2)
        spawn.assert_called_once()
        import os

        assert os.environ["MASTER_PORT"] != "29500"
        monkeypatch.delenv("MASTER_PORT", raising=False)

    def test_a_declared_port_is_kept(self, monkeypatch):
        import os
        from unittest.mock import patch

        from spectramr.infrastructure.distributed import launcher

        monkeypatch.setenv("MASTER_PORT", "31234")
        with patch("torch.multiprocessing.spawn"):
            launcher.launch_distributed(lambda **_: None, object(), num_devices=2)
        assert os.environ["MASTER_PORT"] == "31234"

    def test_multi_node_command_carries_a_rendezvous_id_and_the_declared_port(self):
        from spectramr.infrastructure.distributed import launcher

        command = launcher.launch_distributed(
            lambda **_: None, object(), num_devices=2, num_nodes=2
        )
        assert "--rdzv_id=${SLURM_JOB_ID:-spectramr}" in command
        assert "$MASTER_ADDR:${MASTER_PORT:-29500}" in command
