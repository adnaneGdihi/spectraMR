"""Campaign scripts must run the venv's torch, or stop with a legible message.

``module load torchvision`` put a second, module-provided torch on the path (and the
``PYTHONPATH`` its module sets wins over the venv's site-packages). A cluster job then ran
torch 2.7 against a package that requires >= 2.8 and failed at its first torch-dependent
import with a misleading ``cannot import name``. The bash is executed here with a stub
``python``, because a text match cannot tell a guard from a comment about one.
"""

from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path

import pytest

from spectramr.infrastructure.orchestration.slurm_backend import (
    SLURMBackend,
    _environment_guard_block,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[4]


@pytest.fixture(autouse=True)
def _account(monkeypatch):
    monkeypatch.setenv("SPECTRAMR_SLURM_ACCOUNT", "test_alloc")
    monkeypatch.delenv("SPECTRAMR_SLURM_PARTITION", raising=False)
    monkeypatch.delenv("SPECTRAMR_SLURM_MAIL_USER", raising=False)


def _per_arm(base: str, **kw) -> str:
    return SLURMBackend.generate_job_script(
        experiment_name="exp", config_path="c.yaml", output_dir="o", base_dir=base, **kw
    )


def _array(base: str) -> str:
    return SLURMBackend.generate_array_job_script(
        manifest_path="m.txt", n_tasks=2, output_base="o", base_dir=base, dispatch_dir="d"
    )


def _python_stub(tmp_path: Path, *, works: bool) -> Path:
    stubs = tmp_path / "bin"
    stubs.mkdir(exist_ok=True)
    python = stubs / "python"
    if works:
        # A wrapper, not a symlink: a venv interpreter reached through a symlink elsewhere
        # does not find its pyvenv.cfg and loses the venv's site-packages (torch).
        python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        python.chmod(python.stat().st_mode | stat.S_IEXEC)
    else:
        python.write_text("#!/bin/sh\necho 'ImportError: torch is too old' >&2\nexit 1\n")
        python.chmod(python.stat().st_mode | stat.S_IEXEC)
    return stubs


def _run(script: str, path: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/usr/bin/bash", "-c", script],
        env={"PATH": f"{path}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )


class TestNoModuleTorch:
    @pytest.mark.parametrize("make", [_per_arm, _array])
    def test_no_script_loads_a_site_torch_module(self, make) -> None:
        """Planted: the line that made the venv's torch lose to the cluster's 2.7."""
        script = make("/base")
        assert "module load" not in [ln.strip().rsplit("  ", 1)[0] for ln in script.splitlines()]
        assert not any(ln.lstrip().startswith("module load") for ln in script.splitlines())


class TestVenvIsMandatory:
    @staticmethod
    def _environment(script: str) -> str:
        start = script.index("# The project venv is the torch SSOT.")
        end = script.index("\n", script.index('source "', start)) + 1
        return script[start:end]

    @pytest.mark.parametrize("make", [_per_arm, _array])
    def test_a_missing_venv_fails_loudly_instead_of_using_system_python(
        self, make, tmp_path
    ) -> None:
        script = make(str(tmp_path / "base"))
        out = _run(self._environment(script), "")
        assert out.returncode == 3, out.stderr
        assert "venv activate script not found" in out.stderr

    def test_an_existing_venv_is_sourced(self, tmp_path) -> None:
        venv = tmp_path / "activate"
        venv.write_text("export VENV_SOURCED=yes\n")
        script = _per_arm(str(tmp_path / "base"), venv_path=str(venv))
        out = _run(self._environment(script) + 'echo "$VENV_SOURCED"\n', "")
        assert out.returncode == 0 and out.stdout.strip() == "yes", out.stderr


class TestEnvironmentGuard:
    def test_a_working_torch_passes_and_reports_its_version(self, tmp_path) -> None:
        out = _run(_environment_guard_block(str(REPO)), str(_python_stub(tmp_path, works=True)))
        assert out.returncode == 0, out.stderr
        assert "[ENV] torch" in out.stdout

    def test_planted_a_python_that_cannot_import_torch_stops_the_job(self, tmp_path) -> None:
        out = _run(_environment_guard_block(str(REPO)), str(_python_stub(tmp_path, works=False)))
        assert out.returncode == 3
        assert "'import torch' failed in the active environment" in out.stderr
        assert "shadow the venv's" in out.stderr

    def test_a_missing_helper_fails_loudly(self, tmp_path) -> None:
        out = _run(
            _environment_guard_block(str(tmp_path / "nowhere")),
            str(_python_stub(tmp_path, works=True)),
        )
        assert out.returncode == 3 and "missing" in out.stderr

    @pytest.mark.parametrize("make", [_per_arm, _array])
    def test_both_generators_run_the_guard_before_the_work(self, make) -> None:
        script = make("/base")
        assert "[ENV] python:" in script
        assert (
            script.index("# ── Environment guard ──") < script.index("manifest_dispatch")
            if ("manifest_dispatch" in script)
            else script.index("# ── Environment guard ──") < script.index("TRAIN_CMD=")
        )

    def test_the_generated_scripts_are_valid_bash(self) -> None:
        for script in (_per_arm("/base"), _per_arm("/base", resume=True), _array("/base")):
            checked = subprocess.run(
                ["/usr/bin/bash", "-n"], input=script, capture_output=True, text=True, check=False
            )
            assert checked.returncode == 0, checked.stderr
