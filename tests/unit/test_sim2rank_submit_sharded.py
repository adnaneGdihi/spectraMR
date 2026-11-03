"""``submit_sim2rank_sharded.sh``: the shard array, then one dependent merge job.

Driven against a stub ``sbatch`` on PATH that records its arguments, so the quoting of
the passthrough arguments is executed without a cluster.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SUBMIT = REPO / "scripts" / "sim2rank" / "submit_sim2rank_sharded.sh"


def _stub_sbatch(tmp_path: Path) -> tuple[dict, Path]:
    log = tmp_path / "sbatch.log"
    stub = tmp_path / "bin" / "sbatch"
    stub.parent.mkdir()
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\0" "$@" >> {log}\nprintf "\\n" >> {log}\n'
        'echo "$((1000 + $(grep -c "" ' + str(log) + ')))"\n'
    )
    stub.chmod(0o755)
    branch = subprocess.run(
        ["git", "-C", str(REPO), "branch", "--show-current"], capture_output=True, text=True
    ).stdout.strip()
    # The dispatch snapshot follows dev; the fixture names this checkout's branch instead.
    env = {
        **os.environ,
        "PATH": f"{stub.parent}:{os.environ['PATH']}",
        "SPECTRAMR_DISPATCH_RUN_BRANCH": branch,
    }
    env.pop("SIM2RANK_SBATCH_ARGS", None)
    return env, log


def _dispatch_dirs() -> set[Path]:
    root = REPO / "logs" / "sim2rank_sharded"
    return set(root.iterdir()) if root.is_dir() else set()


def _calls(log: Path) -> list[list[str]]:
    return [c.strip("\0").split("\0") for c in log.read_text().split("\0\n") if c.strip("\0\n")]


def test_submit_script_arrays_then_merges_with_passthrough(tmp_path) -> None:
    env, log = _stub_sbatch(tmp_path)
    before = _dispatch_dirs()
    try:
        _check_arrays_then_merges(env, log)
    finally:
        # The submit script writes its snapshot under the repo; never leave one behind.
        for d in _dispatch_dirs() - before:
            shutil.rmtree(d, ignore_errors=True)


def _check_arrays_then_merges(env: dict, log: Path) -> None:
    passthrough = ["--data-dir", "/data/a b", "--severity-axes", "x.yaml", "--seed", "7"]
    out = subprocess.run(
        ["bash", str(SUBMIT), "--shards", "3", *passthrough],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    shard_call, merge_call = _calls(log)
    for call in (shard_call, merge_call):
        assert not any(a in ("-p",) or a.startswith("--partition") for a in call)
        assert "--gres=gpu:1" in call
    assert "--array=0-2" in shard_call
    array_id = out.split("shard array: ")[1].split()[0]
    assert f"--dependency=afterok:{array_id}" in merge_call
    assert "--kill-on-invalid-dep=yes" in merge_call
    shard_wrap = next(a for a in shard_call if a.startswith("--wrap="))[len("--wrap=") :]
    merge_wrap = next(a for a in merge_call if a.startswith("--wrap="))[len("--wrap=") :]
    for wrap, script in ((shard_wrap, "sim2rank.py"), (merge_wrap, "merge_shards.py")):
        prelude, cmd = wrap.split(" && python ", 1)
        assert "source .venv/bin/activate" in prelude
        assert prelude.endswith("source .venv/bin/activate")
        assert cmd.startswith(f"scripts/sim2rank/{script} ")
        echoed = subprocess.run(
            ["bash", "-c", "printf '%s\\n' " + cmd.split(" ", 1)[1]],
            env={**env, "SLURM_ARRAY_TASK_ID": "2"},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        tail = ["--content-shard", "2/3"] if script == "sim2rank.py" else []
        manifest = Path(echoed[len(passthrough) + 1])
        assert echoed == [*passthrough, "--dispatch-manifest", str(manifest), *tail]
        snapshot = json.loads(manifest.with_name("manifest.snapshot.json").read_text())
        assert snapshot["head"] and list(snapshot["configs"]) == ["scripts/sim2rank/sim2rank.py"]


def test_submit_script_refuses_a_checkout_off_the_run_branch(tmp_path) -> None:
    env, log = _stub_sbatch(tmp_path)
    env["SPECTRAMR_DISPATCH_RUN_BRANCH"] = "no-such-branch"
    proc = subprocess.run(
        ["bash", str(SUBMIT), "--shards", "2"], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 1 and "code snapshot" in proc.stderr and not log.exists()


@pytest.mark.parametrize(
    "argv", [[], ["--shards", "0"], ["--shards", "2", "--content-shard", "0/2"]]
)
def test_submit_script_refuses_bad_shard_flags(tmp_path, argv) -> None:
    env, log = _stub_sbatch(tmp_path)
    proc = subprocess.run(["bash", str(SUBMIT), *argv], env=env, capture_output=True, text=True)
    assert proc.returncode == 2 and not log.exists()
