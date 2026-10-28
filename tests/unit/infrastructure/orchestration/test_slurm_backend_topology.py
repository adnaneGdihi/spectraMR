"""Campaign scripts must use every GPU and core the allocation grants.

Two campaign generators forked ranks from a literal rather than from the grant: the
array script pinned ``CUDA_VISIBLE_DEVICES=0`` and passed no rank count, so a
``--gpus=4`` task billed four GPUs and trained on one, and the per-arm script handed
every node the same fixed rendezvous. The bash block is executed here against
controlled Slurm environments, because a text grep cannot tell a derivation from a
literal.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from spectramr.infrastructure.orchestration.slurm_backend import (
    SLURMBackend,
    _granted_topology_block,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[4]


@pytest.fixture(autouse=True)
def _account(monkeypatch):
    monkeypatch.setenv("SPECTRAMR_SLURM_ACCOUNT", "test_alloc")
    monkeypatch.delenv("SPECTRAMR_SLURM_PARTITION", raising=False)
    monkeypatch.delenv("SPECTRAMR_SLURM_MAIL_USER", raising=False)


def _run_block(env: dict[str, str], *, default_cpus: int = 8) -> subprocess.CompletedProcess[str]:
    script = _granted_topology_block(str(REPO), default_cpus) + (
        'echo "GPUS=${GPUS_PER_NODE} THREADS=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS}"\n'
    )
    return subprocess.run(
        ["/usr/bin/bash", "-c", script],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        check=False,
    )


class TestGrantedTopologyBlock:
    def test_ranks_come_from_the_grant_not_a_literal(self) -> None:
        out = _run_block({"SLURM_GPUS_ON_NODE": "4", "SLURM_CPUS_PER_TASK": "32"})
        assert out.returncode == 0, out.stderr
        assert "GPUS=4 THREADS=8 MKL=8" in out.stdout

    def test_the_threads_are_divided_across_the_ranks(self) -> None:
        """Every forked rank inherits the value, so the node's cores are split."""
        out = _run_block({"SLURM_GPUS_ON_NODE": "8", "SLURM_CPUS_PER_TASK": "64"})
        assert "GPUS=8 THREADS=8" in out.stdout

    def test_the_thread_floor_is_one(self) -> None:
        out = _run_block({"SLURM_GPUS_ON_NODE": "8", "SLURM_CPUS_PER_TASK": "4"})
        assert "GPUS=8 THREADS=1" in out.stdout

    def test_the_declared_cpu_default_applies_without_a_slurm_value(self) -> None:
        out = _run_block({"SLURM_GPUS_ON_NODE": "2"}, default_cpus=16)
        assert "GPUS=2 THREADS=8" in out.stdout

    def test_a_job_total_is_split_across_nodes(self) -> None:
        out = _run_block({"SLURM_GPUS": "8", "SLURM_NNODES": "2", "SLURM_CPUS_PER_TASK": "16"})
        assert "GPUS=4 THREADS=4" in out.stdout

    def test_planted_unknown_grant_refuses_instead_of_assuming_one_rank(self, tmp_path) -> None:
        """The silent single-rank run: four GPUs billed, one used, no error.

        PATH holds only ``grep``: a host with ``nvidia-smi`` would otherwise answer the
        last-resort probe and hide the refusal.
        """
        (tmp_path / "grep").symlink_to("/usr/bin/grep")
        out = _run_block({"PATH": str(tmp_path)})
        assert out.returncode == 1
        assert "could not determine GPUs per node" in out.stderr

    def test_a_missing_helper_fails_loudly(self) -> None:
        script = _granted_topology_block("/nonexistent/base", 8)
        out = subprocess.run(
            ["bash", "-c", script], env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True
        )
        assert out.returncode == 3 and "missing" in out.stderr


class TestArrayScript:
    @staticmethod
    def _script(**params) -> str:
        return SLURMBackend.generate_array_job_script(
            manifest_path="m.txt",
            n_tasks=3,
            output_base="out",
            base_dir="/base",
            dispatch_dir="disp",
            slurm_params=params or None,
        )

    def test_the_array_script_does_not_mask_to_the_first_gpu(self) -> None:
        """Planted: the literal export that pinned every task to device 0."""
        assert "export CUDA_VISIBLE_DEVICES=0" not in self._script(gpus=4)

    def test_the_array_script_passes_the_granted_rank_count(self) -> None:
        script = self._script(gpus=4)
        assert "--nproc-per-node ${GPUS_PER_NODE}" in script
        # Derived before it is used.
        assert script.index("derive_gpus_per_node") < script.index("--nproc-per-node")

    def test_the_array_script_bounds_the_threads_per_rank(self) -> None:
        script = self._script(gpus=4, cpus_per_task=32)
        assert "SLURM_CPUS_PER_TASK:-32" in script
        assert "OMP_NUM_THREADS" in script


class TestPerArmScript:
    @staticmethod
    def _script(**params) -> str:
        return SLURMBackend.generate_job_script(
            experiment_name="exp",
            config_path="c.yaml",
            output_dir="out",
            base_dir="/base",
            slurm_params=params,
        )

    def test_the_rank_count_is_the_grant(self) -> None:
        script = self._script(gpus=4)
        assert "NUM_GPUS=${GPUS_PER_NODE}" in script
        assert "NUM_GPUS=4" not in script

    def test_the_rendezvous_is_job_keyed_not_a_fixed_port(self) -> None:
        script = self._script(gpus=2)
        assert ":29500" not in script
        assert "--rdzv_id=${SLURM_JOB_ID:-spectramr}" in script
        assert 'MASTER_PORT="$(( 20000 + (${SLURM_JOB_ID:-0} % 20000) ))"' in script

    def test_multi_node_puts_a_launcher_on_every_node(self) -> None:
        script = self._script(gpus=1, nodes=2)
        assert "srun --kill-on-bad-exit=1 --ntasks=${NUM_NODES} --ntasks-per-node=1" in script

    def test_the_generated_script_is_valid_bash(self) -> None:
        for script in (self._script(gpus=4, nodes=2), self._script(gpus=1)):
            checked = subprocess.run(
                ["bash", "-n"], input=script, capture_output=True, text=True, check=False
            )
            assert checked.returncode == 0, checked.stderr
        checked = subprocess.run(
            ["bash", "-n"], input=TestArrayScript._script(gpus=4), capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stderr


class TestScriptsImportTheirOwnTree:
    """The package lives in ``src/``, so exporting the repo root selected nothing and the job
    imported whichever clone the venv's editable install pointed at."""

    @staticmethod
    def _line(script: str, prefix: str) -> str:
        return next(ln for ln in script.splitlines() if ln.startswith(prefix))

    @staticmethod
    def _per_arm(base: str) -> str:
        return SLURMBackend.generate_job_script(
            experiment_name="exp", config_path="c.yaml", output_dir="o", base_dir=base
        )

    @staticmethod
    def _array(base: str) -> str:
        return SLURMBackend.generate_array_job_script(
            manifest_path="m.txt", n_tasks=2, output_base="o", base_dir=base, dispatch_dir="d"
        )

    @pytest.mark.parametrize("make", ["_per_arm", "_array"])
    def test_the_job_imports_the_tree_it_was_generated_for(self, tmp_path, make) -> None:
        """Planted: a stub package under <base>/src must win over any installed copy."""
        import sys

        package = tmp_path / "src" / "spectramr"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        script = getattr(self, make)(str(tmp_path))
        export = self._line(script, "export PYTHONPATH=")
        probe = f"{sys.executable} -c 'import spectramr; print(spectramr.__file__)'"
        out = subprocess.run(
            ["/usr/bin/bash", "-c", f"{export}\n{probe}"],
            env={"PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == str(package / "__init__.py")

    @pytest.mark.parametrize("make", ["_per_arm", "_array"])
    def test_the_export_survives_an_unset_pythonpath_under_nounset(self, tmp_path, make) -> None:
        script = getattr(self, make)(str(tmp_path))
        export = self._line(script, "export PYTHONPATH=")
        out = subprocess.run(
            ["/usr/bin/bash", "-uc", f"{export}\necho ok"],
            env={"PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert out.returncode == 0 and "ok" in out.stdout, out.stderr

    def test_the_cuda_diagnostic_line_is_valid_python_and_names_the_imported_tree(
        self, tmp_path
    ) -> None:
        """The escaped quotes in this line made python raise SyntaxError on every job."""
        import sys

        stubs = tmp_path / "bin"
        stubs.mkdir()
        python = stubs / "python"
        python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        python.chmod(python.stat().st_mode | 0o111)
        script = self._per_arm(str(tmp_path))
        cuda = self._line(script, 'echo "CUDA:')
        origin = self._line(script, 'echo "spectramr:')
        out = subprocess.run(
            ["/usr/bin/bash", "-c", f"{cuda}\n{origin}"],
            env={"PATH": f"{stubs}:/usr/bin:/bin", "PYTHONPATH": str(REPO / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
        assert "SyntaxError" not in out.stderr, out.stderr
        assert out.stdout.splitlines()[0].startswith("CUDA: ")
        assert out.stdout.splitlines()[0].strip() != "CUDA:", "torch did not import"
        assert "spectramr/__init__.py" in out.stdout


class TestEmbeddedManifest:
    """The array script carries the manifest, so a moved campaign dir cannot break a task."""

    def test_task_reads_its_own_copy_when_the_campaign_dir_is_gone(self, tmp_path):
        import subprocess

        script = SLURMBackend.generate_array_job_script(
            manifest_path="/gone/campaign/array_manifest.txt",
            n_tasks=2,
            output_base="/gone/campaign",
            base_dir="/base",
            manifest_text="a.yaml\nb.yaml\n",
            snapshot_text='{"schema": 1}\n',
        )
        assert "--manifest ${ARRAY_MANIFEST_DIR}/array_manifest.txt" in script
        assert "/gone/campaign/array_manifest.txt" not in script.split("manifest_dispatch")[-1]
        start = script.index("ARRAY_MANIFEST_DIR=")
        end = script.index("# ── Resolve + audit")
        run = (
            f'export TMPDIR={tmp_path} USER=u SLURM_ARRAY_JOB_ID=7 SLURM_ARRAY_TASK_ID=1\n'
            + script[start:end]
            + '\ncat "${ARRAY_MANIFEST_DIR}/array_manifest.txt" "${ARRAY_MANIFEST_DIR}/array_manifest.snapshot.json"'
        )
        out = subprocess.run(["bash", "-c", run], capture_output=True, text=True, check=True)
        assert out.stdout == 'a.yaml\nb.yaml\n{"schema": 1}\n'

    def test_without_text_the_manifest_path_is_unchanged(self):
        script = SLURMBackend.generate_array_job_script(
            manifest_path="/camp/array_manifest.txt", n_tasks=1, output_base="/camp", base_dir="/b"
        )
        assert "--manifest /camp/array_manifest.txt" in script
        assert "ARRAY_MANIFEST_DIR" not in script
