"""Tests for the pipeline-agnostic execution backends (WS-D).

These backends answer only "run this ``spectramr <verb>`` invocation *here*"
(local/docker/apptainer/slurm) — they know nothing about training. The
shell-out backends (Docker/Apptainer/SLURM) build a command/script from a
:class:`SpectraMRInvocation` + :class:`ResourceSpec`; ``build_*`` is pure and
testable without spawning anything.
"""

from __future__ import annotations

import os

import pytest

import spectramr.infrastructure.execution.backends as backends_mod
from spectramr.infrastructure.execution import (
    ApptainerBackend,
    DockerBackend,
    SpectraMRInvocation,
    ResourceSpec,
    SlurmBackend,
    export_launch_env,
    resolve_launch_provenance,
)


@pytest.fixture(autouse=True)
def _slurm_account(monkeypatch):
    """Supply the SLURM allocation these tests used to inherit from a literal.

    ``ResourceSpec`` ships no account default -- an allocation name is
    site-specific, so the tree carries none (#1146). Tests that render SLURM
    directives must therefore configure one, exactly as a user does.
    """
    monkeypatch.setenv("SPECTRAMR_SLURM_ACCOUNT", "test_alloc")
    monkeypatch.delenv("SPECTRAMR_SLURM_MAIL_USER", raising=False)


@pytest.fixture(autouse=True)
def _isolate_launch_env():
    """Snapshot + clear SPECTRAMR_LAUNCH_* around every test.

    ``export_launch_env`` writes os.environ directly (it must, so the child run
    inherits it), and the container backends now forward those vars — so a leak
    would pollute unrelated build-command tests. This isolates each test.
    """
    saved = {k: v for k, v in os.environ.items() if k.startswith("SPECTRAMR_LAUNCH_")}
    for k in saved:
        del os.environ[k]
    yield
    for k in [k for k in os.environ if k.startswith("SPECTRAMR_LAUNCH_")]:
        del os.environ[k]
    os.environ.update(saved)


def test_invocation_to_cli_args():
    inv = SpectraMRInvocation(verb="train", config="exp.yaml", extra_args=("--resume", "auto"))
    assert inv.to_cli_args() == ["train", "--config", "exp.yaml", "--resume", "auto"]


def test_invocation_without_config():
    inv = SpectraMRInvocation(verb="doctor")
    assert inv.to_cli_args() == ["doctor"]


def test_resourcespec_account_env_fallback(monkeypatch):
    """The account resolves from the environment and NOWHERE else.

    There is deliberately no built-in default: an allocation name is
    site-specific, so any value shipped here would be wrong everywhere but one
    cluster. Unset stays ``None`` rather than becoming somebody's account, and
    it is ``render_directives`` that reports it (see the SLURM backend tests).
    """
    monkeypatch.delenv("SPECTRAMR_SLURM_ACCOUNT", raising=False)
    assert ResourceSpec().account is None  # no site default is shipped
    monkeypatch.setenv("SPECTRAMR_SLURM_ACCOUNT", "myacct")
    assert ResourceSpec().account == "myacct"  # env supplies it
    assert ResourceSpec(account="explicit").account == "explicit"  # arg wins


def test_resourcespec_empty_env_reads_as_unset_not_as_empty(monkeypatch):
    """``SPECTRAMR_SLURM_ACCOUNT=""`` must land on *unset*, and be reported.

    An empty value is how a ``.env`` disables a variable without deleting the
    line, so it has to reach the same behaviour as absent. Assigning it through
    would leave ``account == ""``, which passes the ``is None`` guard in
    ``render_directives`` and then falls out of ``if resources.account:`` --
    the directive would be dropped with no error at all (non-negotiable 3).

    ``partition`` is pinned in the same shape because the two are resolved by
    adjacent branches of one ``__post_init__``; pinning only one is what lets
    the other drift apart from it.
    """
    monkeypatch.setenv("SPECTRAMR_SLURM_ACCOUNT", "")
    monkeypatch.setenv("SPECTRAMR_SLURM_PARTITION", "")
    spec = ResourceSpec()
    assert spec.account is None, "empty env became an empty account, not unset"
    assert spec.partition is None, "empty env became an empty partition, not unset"

    with pytest.raises(ValueError, match="SPECTRAMR_SLURM_ACCOUNT"):
        SlurmBackend().render_directives(spec, job_name="gm_test")


def test_docker_backend_builds_run_command():
    inv = SpectraMRInvocation(verb="train", config="exp.yaml")
    cmd = DockerBackend().build_command(inv, ResourceSpec())  # default gpus=1
    assert cmd[:3] == ["docker", "run", "--gpus"]
    assert "1" in cmd  # honors the COUNT (default 1), not "all"
    # bind-mounts the workdir and ends with the spectramr invocation
    assert any(":/workspace" in c for c in cmd)
    assert cmd[-3:] == ["train", "--config", "exp.yaml"]


@pytest.mark.parametrize(
    "gpus,expected", [(0, "all"), (1, "1"), (2, "2")]
)
def test_docker_backend_honors_gpu_count(gpus, expected):
    """``--gpus N`` for a count; the ``0`` sentinel means all visible GPUs."""
    cmd = DockerBackend().build_command(
        SpectraMRInvocation(verb="train", config="x.yaml"), ResourceSpec(gpus=gpus)
    )
    assert cmd[cmd.index("--gpus") + 1] == expected


def test_apptainer_backend_builds_run_command():
    inv = SpectraMRInvocation(verb="infer", config="exp.yaml")
    cmd = ApptainerBackend().build_command(inv, ResourceSpec())  # default gpus=1
    assert cmd[:3] == ["apptainer", "run", "--nv"]
    assert "--bind" in cmd
    # default count=1 restricts the container to device 0
    assert "CUDA_VISIBLE_DEVICES=0" in cmd
    assert cmd[-3:] == ["infer", "--config", "exp.yaml"]


@pytest.mark.parametrize(
    "gpus,visible", [(2, "CUDA_VISIBLE_DEVICES=0,1"), (1, "CUDA_VISIBLE_DEVICES=0")]
)
def test_apptainer_backend_restricts_to_gpu_count(gpus, visible):
    cmd = ApptainerBackend().build_command(
        SpectraMRInvocation(verb="train", config="x.yaml"), ResourceSpec(gpus=gpus)
    )
    assert "--nv" in cmd
    assert visible in cmd


def test_apptainer_backend_gpus_zero_is_unrestricted():
    """``gpus=0`` (all) → ``--nv`` with no CUDA_VISIBLE_DEVICES restriction."""
    cmd = ApptainerBackend().build_command(
        SpectraMRInvocation(verb="train", config="x.yaml"), ResourceSpec(gpus=0)
    )
    assert "--nv" in cmd
    assert not any(c.startswith("CUDA_VISIBLE_DEVICES") for c in cmd)


def test_resourcespec_rejects_negative_gpus():
    with pytest.raises(ValueError, match="gpus must be >= 0"):
        ResourceSpec(gpus=-1)


def test_docker_image_env_override(monkeypatch):
    monkeypatch.setenv("SPECTRAMR_DOCKER_IMAGE", "myrepo/spectramr:dev")
    cmd = DockerBackend().build_command(
        SpectraMRInvocation(verb="train", config="x.yaml"), ResourceSpec()
    )
    assert "myrepo/spectramr:dev" in cmd


def test_slurm_backend_builds_sbatch_script():
    inv = SpectraMRInvocation(verb="train", config="exp.yaml")
    res = ResourceSpec(account="test_alloc", gpus=2, mem="64G", time="08:00:00")
    script = SlurmBackend().build_script(inv, res)
    assert script.startswith("#!/")
    assert "#SBATCH --account=test_alloc" in script
    assert "#SBATCH --gpus=2" in script
    assert "#SBATCH --mem=64G" in script
    assert "#SBATCH --time=08:00:00" in script
    # the actual spectramr command is present
    assert "train --config exp.yaml" in script


def test_slurm_partition_only_emitted_when_set():
    inv = SpectraMRInvocation(verb="train", config="x.yaml")
    no_part = SlurmBackend().build_script(inv, ResourceSpec(partition=None))
    assert "--partition" not in no_part
    with_part = SlurmBackend().build_script(inv, ResourceSpec(partition="gpu"))
    assert "#SBATCH --partition=gpu" in with_part


def test_slurm_array_only_emitted_when_set():
    """WS-4: ``--array`` is the SSOT replacement for the hand-written array
    .sbatch files. Default (None) must leave the directive block byte-identical
    (the campaign golden depends on this)."""
    inv = SpectraMRInvocation(verb="train", config="x.yaml")
    no_array = SlurmBackend().build_script(inv, ResourceSpec())
    assert "--array" not in no_array
    with_array = SlurmBackend().build_script(inv, ResourceSpec(array="0-9"))
    assert "#SBATCH --array=0-9" in with_array


def test_slurm_array_line_follows_gres():
    """The array directive renders directly after ``--gres`` (stable ordering so
    the regenerated .sbatch headers match render_directives byte-for-byte)."""
    res = ResourceSpec(gpus=2, array="0-19%4")
    block = SlurmBackend().render_directives(res, job_name="arr")
    lines = block.splitlines()
    gres_i = lines.index("#SBATCH --gpus=2")
    array_i = lines.index("#SBATCH --array=0-19%4")
    assert array_i == gres_i + 1


def test_resourcespec_rejects_empty_array():
    """An empty/whitespace array is a mistake, not 'unset' — raise rather than
    emit an invalid ``#SBATCH --array=`` (pitfall #15)."""
    with pytest.raises(ValueError, match="array, when set, must be a non-empty"):
        ResourceSpec(array="   ")
    # None (the default) is the disable sentinel and must NOT raise.
    assert ResourceSpec(array=None).array is None


def test_backend_names():
    assert DockerBackend().name == "docker"
    assert ApptainerBackend().name == "apptainer"
    assert SlurmBackend().name == "slurm"


# --- submit_script: the single shared sbatch primitive (WS-E consolidation) ---


class _SbatchResult:
    def __init__(self, returncode, stdout, stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_submit_script_parses_job_id(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["input"] = kwargs.get("input")
        return _SbatchResult(0, "Submitted batch job 98765\n")

    monkeypatch.setattr(backends_mod.subprocess, "run", fake_run)
    jid = SlurmBackend().submit_script("#!/bin/bash\n", dependency_job_ids=[1, 2])
    assert jid == 98765
    assert seen["cmd"][0] == "sbatch"
    assert "--dependency" in seen["cmd"] and "afterok:1:2" in seen["cmd"]
    assert seen["input"] == "#!/bin/bash\n"  # piped via stdin, not a temp file


def test_submit_script_raises_on_sbatch_failure(monkeypatch):
    monkeypatch.setattr(
        backends_mod.subprocess, "run", lambda *a, **k: _SbatchResult(1, "", "boom")
    )
    with pytest.raises(RuntimeError, match="sbatch failed"):
        SlurmBackend().submit_script("#!/bin/bash\n")


def test_slurm_run_delegates_to_submit_script(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        SlurmBackend,
        "submit_script",
        lambda self, script, **k: captured.update(script=script) or 12345,
    )
    handle = SlurmBackend().run(SpectraMRInvocation("train", "x.yaml"), ResourceSpec())
    assert handle.id == "12345"
    assert "#SBATCH --job-name=spectramr-train" in captured["script"]


# ---------------------------------------------------------------------------
# Launch → provenance handoff (#9): SPECTRAMR_LAUNCH_* env round-trip + container
# forwarding, so a run started via `spectramr launch` records the backend +
# resolved resources in its run_summary.json (pitfall #15c).
# ---------------------------------------------------------------------------


def test_export_and_resolve_launch_provenance_roundtrip():
    export_launch_env("slurm", ResourceSpec(account="acct", partition="gpu", gpus=2, mem="64G"))
    prov = resolve_launch_provenance()
    assert prov["backend"] == "slurm"
    assert prov["account"] == "acct"
    assert prov["partition"] == "gpu"
    assert prov["mem"] == "64G"
    assert prov["gpus"] == 2  # int-typed, not the string "2"
    assert isinstance(prov["gpus"], int)


def test_resolve_launch_provenance_empty_when_not_launched():
    # No SPECTRAMR_LAUNCH_BACKEND (plain `spectramr train`) → strict no-op.
    assert resolve_launch_provenance() == {}


def test_resolve_launch_provenance_raises_on_bad_int(monkeypatch):
    monkeypatch.setenv("SPECTRAMR_LAUNCH_BACKEND", "slurm")
    monkeypatch.setenv("SPECTRAMR_LAUNCH_GPUS", "not-an-int")
    with pytest.raises(ValueError, match="not an integer"):
        resolve_launch_provenance()


@pytest.mark.parametrize("backend_cls", [DockerBackend, ApptainerBackend])
def test_container_forwards_launch_env(backend_cls):
    export_launch_env("docker", ResourceSpec(gpus=1))
    cmd = backend_cls().build_command(
        SpectraMRInvocation(verb="train", config="x.yaml"), ResourceSpec(gpus=1)
    )
    # the SPECTRAMR_LAUNCH_* vars are forwarded into the container as --env pairs
    assert "--env" in cmd
    assert any(c == "SPECTRAMR_LAUNCH_BACKEND=docker" for c in cmd)


@pytest.mark.parametrize("backend_cls", [DockerBackend, ApptainerBackend])
def test_container_no_launch_env_when_not_launched(backend_cls):
    # Without SPECTRAMR_LAUNCH_*, no launch --env pairs are injected.
    cmd = backend_cls().build_command(
        SpectraMRInvocation(verb="train", config="x.yaml"), ResourceSpec(gpus=0)
    )
    assert not any(c.startswith("SPECTRAMR_LAUNCH_") for c in cmd)


# ---------------------------------------------------------------------------
# Wall-clock chaining directives.
#
# `render_directives` is the SSOT the committed `.sbatch` headers are
# REGENERATED from, so adding these by hand to a header is the drift
# `test_header_matches_render_directives_ssot` exists to catch. Both default to
# emitting nothing, which is what keeps every existing caller -- and the
# campaign golden -- byte-identical.
# ---------------------------------------------------------------------------


def _lines(**spec_kwargs) -> list[str]:
    from spectramr.infrastructure.execution.backends import ResourceSpec, SlurmBackend

    block = SlurmBackend().render_directives(
        ResourceSpec(account="acct", **spec_kwargs), job_name="j"
    )
    return [ln for ln in block.splitlines() if ln.startswith("#SBATCH")]


@pytest.mark.unit
def test_neither_directive_is_emitted_by_default(monkeypatch):
    """Byte-identical for every caller that does not ask -- the same contract
    `array` already keeps."""
    monkeypatch.delenv("SPECTRAMR_SLURM_PARTITION", raising=False)
    lines = _lines()

    assert not [ln for ln in lines if "requeue" in ln or "open-mode" in ln]


@pytest.mark.unit
def test_requeue_is_emitted_when_asked(monkeypatch):
    """Slurm refuses `scontrol requeue` outright without it, so a wall-clock
    chain silently stops after one link."""
    monkeypatch.delenv("SPECTRAMR_SLURM_PARTITION", raising=False)

    assert "#SBATCH --requeue" in _lines(requeue=True)


@pytest.mark.unit
def test_open_mode_is_emitted_when_asked(monkeypatch):
    """Without append, the requeued task truncates its predecessor's .out --
    destroying the record of why the previous link stopped."""
    monkeypatch.delenv("SPECTRAMR_SLURM_PARTITION", raising=False)

    assert "#SBATCH --open-mode=append" in _lines(open_mode="append")


@pytest.mark.unit
def test_the_two_sit_after_partition_and_before_the_mail_lines(monkeypatch):
    """Position is the invariant, not just presence: the committed headers are
    regenerated from this order and compared line by line."""
    from spectramr.infrastructure.execution.backends import ResourceSpec, SlurmBackend

    monkeypatch.delenv("SPECTRAMR_SLURM_PARTITION", raising=False)
    block = SlurmBackend().render_directives(
        ResourceSpec(account="acct", partition="batch", requeue=True, open_mode="append"),
        job_name="j",
        mail_type="END,FAIL",
        mail_user="u",
    )
    lines = [ln for ln in block.splitlines() if ln.startswith("#SBATCH")]
    idx = {ln.split("=")[0]: i for i, ln in enumerate(lines)}

    assert idx["#SBATCH --partition"] < idx["#SBATCH --requeue"]
    assert idx["#SBATCH --requeue"] < idx["#SBATCH --open-mode"]
    assert idx["#SBATCH --open-mode"] < idx["#SBATCH --mail-type"]
