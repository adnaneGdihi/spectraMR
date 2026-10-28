"""Unit tests for :mod:`spectramr.infrastructure.orchestration.slurm_backend`.

The real backend shells out to ``sbatch``/``sacct``/``squeue``; the tests
here only exercise:

* the pure ``JobStatus`` dataclass (terminal-state classification and
  the SLURM → campaign-state mapping)
* the ``dry_run=True`` branches of :class:`SLURMBackend` (no actual
  cluster interaction)
* the static ``generate_job_script`` helper, which is a string
  formatter and therefore CPU-only

Anything that requires a live SLURM controller is out of scope here and
lives behind the ``@pytest.mark.integration`` marker (none yet).
"""

from __future__ import annotations

import os
import re
import subprocess

import pytest

from spectramr.infrastructure.orchestration.slurm_backend import JobStatus, SLURMBackend


# ── JobStatus ───────────────────────────────────────────────────────


class TestJobStatusTerminal:
    """SLURM state-name → terminal classification."""

    @pytest.mark.parametrize(
        "state",
        ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"],
    )
    def test_terminal_states(self, state: str) -> None:
        s = JobStatus(job_id=1, state=state)
        assert s.is_terminal, f"{state} must be terminal"

    @pytest.mark.parametrize("state", ["PENDING", "RUNNING", "SUSPENDED", "UNKNOWN"])
    def test_non_terminal_states(self, state: str) -> None:
        s = JobStatus(job_id=1, state=state)
        assert not s.is_terminal, f"{state} must not be terminal"


class TestJobStatusNormalisation:
    """SLURM state-name → campaign state-name (str) mapping."""

    @pytest.mark.parametrize(
        "slurm_state, expected",
        [
            ("PENDING", "submitted"),
            ("RUNNING", "running"),
            ("COMPLETED", "completed"),
            ("FAILED", "failed"),
            ("CANCELLED", "cancelled"),
            ("CANCELLED+", "cancelled"),
            ("TIMEOUT", "timeout"),
            ("OUT_OF_MEMORY", "failed"),
            ("NODE_FAIL", "failed"),
            ("PREEMPTED", "submitted"),
        ],
    )
    def test_known_mappings(self, slurm_state: str, expected: str) -> None:
        s = JobStatus(job_id=1, state=slurm_state)
        assert s.normalised_state == expected

    def test_unknown_state_defaults_to_submitted(self) -> None:
        """An unrecognised SLURM state must not produce ``None`` or crash —
        the orchestrator depends on a non-null state string."""
        s = JobStatus(job_id=1, state="WEIRD_STATE")
        assert s.normalised_state == "submitted"


# ── SLURMBackend (dry-run path) ────────────────────────────────────


class TestSLURMBackendDryRun:
    """The dry-run paths avoid ``subprocess.run`` entirely."""

    def test_submit_job_returns_zero_in_dry_run(self) -> None:
        backend = SLURMBackend(dry_run=True)
        # No dependency.
        jid = backend.submit_job(script_content="#!/bin/bash\necho hi\n")
        assert jid == 0
        # With dependency string.
        jid2 = backend.submit_job(
            script_content="#!/bin/bash\necho hi\n",
            dependency_job_ids=[1, 2, 3],
        )
        assert jid2 == 0

    def test_query_job_status_returns_pending_in_dry_run(self) -> None:
        backend = SLURMBackend(dry_run=True)
        status = backend.query_job_status(42)
        assert status.job_id == 42
        assert status.state == "PENDING"
        assert status.normalised_state == "submitted"

    def test_query_batch_status_returns_pending_for_all_ids(self) -> None:
        backend = SLURMBackend(dry_run=True)
        out = backend.query_batch_status([10, 11, 12])
        assert set(out) == {10, 11, 12}
        for jid, status in out.items():
            assert status.job_id == jid
            assert status.state == "PENDING"

    def test_query_batch_status_empty_list_short_circuits(self) -> None:
        # An empty input list must short-circuit before any subprocess.
        backend = SLURMBackend(dry_run=False)
        assert backend.query_batch_status([]) == {}

    def test_cancel_job_dry_run_is_a_noop(self) -> None:
        # Just exercises the dry-run branch — nothing to assert beyond
        # "doesn't shell out / raise".
        backend = SLURMBackend(dry_run=True)
        backend.cancel_job(123)
        backend.cancel_batch([1, 2, 3])


# ── generate_job_script (pure formatter) ───────────────────────────


class TestGenerateJobScript:
    @pytest.fixture(autouse=True)
    def _account(self, monkeypatch):
        """The generators ship no account default -- an allocation name is
        site-specific (#1146). Configure one the way a real user does, so the
        directive assertions below test rendering rather than a baked-in site.
        """
        monkeypatch.setenv("SPECTRAMR_SLURM_ACCOUNT", "test_alloc")
        monkeypatch.delenv("SPECTRAMR_SLURM_PARTITION", raising=False)
        monkeypatch.delenv("SPECTRAMR_SLURM_MAIL_USER", raising=False)

    def _script(self, **kwargs: object) -> str:
        return SLURMBackend.generate_job_script(
            experiment_name="exp_alpha",
            config_path="experiments/inprogress/exp.yaml",
            output_dir="/scratch/results/exp_alpha",
            base_dir="/project/spectramr",
            **kwargs,  # type: ignore[arg-type]
        )

    def test_script_starts_with_shebang(self) -> None:
        out = self._script()
        assert out.startswith("#!/bin/bash")

    def test_default_sbatch_directives_present(self) -> None:
        out = self._script()
        for directive in [
            "#SBATCH --job-name=exp_alpha",
            "#SBATCH --account=test_alloc",
            "#SBATCH --time=120:00:00",
            "#SBATCH --mem=64GB",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --gpus=1",
        ]:
            assert directive in out

    def test_no_partition_directive_by_default(self) -> None:
        """GPUs are allocated via --gres; the cluster has no dedicated "gpu"
        partition, so no --partition directive is emitted unless requested."""
        assert "#SBATCH --partition" not in self._script()

    def test_partition_directive_emitted_when_requested(self) -> None:
        out = self._script(slurm_params={"partition": "batch"})
        assert "#SBATCH --partition=batch" in out

    def test_account_comes_from_the_environment(self) -> None:
        """No site allocation is baked in: the account is whatever
        ``$SPECTRAMR_SLURM_ACCOUNT`` says (set to ``test_alloc`` by the fixture).
        """
        assert "#SBATCH --account=test_alloc" in self._script()

    def test_custom_account_overrides_the_environment(self) -> None:
        out = self._script(slurm_params={"account": "other_alloc"})
        assert "#SBATCH --account=other_alloc" in out
        assert "#SBATCH --account=test_alloc" not in out

    def test_an_unconfigured_account_raises_instead_of_rendering_none(self, monkeypatch) -> None:
        """The discrimination leg for the guard added with #1146.

        Without it the generator emits the literal ``#SBATCH --account=None``:
        a script that looks generated, submits, and is rejected by sbatch with
        a message naming neither the knob nor the env var. Silent-fallback and
        silent-garbage are both forbidden (non-negotiable 3).
        """
        monkeypatch.delenv("SPECTRAMR_SLURM_ACCOUNT", raising=False)
        with pytest.raises(ValueError, match="SPECTRAMR_SLURM_ACCOUNT"):
            self._script()

    def test_mail_user_is_omitted_unless_configured(self) -> None:
        """No address is baked in; the directive is absent, not defaulted."""
        assert "--mail-user" not in self._script()

    def test_mail_user_is_emitted_when_configured(self, monkeypatch) -> None:
        monkeypatch.setenv("SPECTRAMR_SLURM_MAIL_USER", "me@example.org")
        assert "#SBATCH --mail-user=me@example.org" in self._script()
    def test_bytecode_cache_uses_node_local_prefix(self) -> None:
        # Regression guard: the per-arm script must NOT suppress the
        # bytecode cache (which forced a full source recompile of the
        # ~1000-module package on every array task). It should instead
        # redirect the cache to node-local scratch via PYTHONPYCACHEPREFIX
        # so each node compiles once and reuses the cache.
        out = self._script()
        assert "PYTHONDONTWRITEBYTECODE" not in out
        assert "export PYTHONPYCACHEPREFIX=" in out

    def test_output_override_redirects_training_output_dir(self) -> None:
        out = self._script()
        assert "-O training.output_dir=/scratch/results/exp_alpha" in out

    def test_resume_flag_when_requested(self) -> None:
        out_no = self._script(resume=False)
        out_yes = self._script(resume=True)
        assert "--resume auto" not in out_no
        assert "--resume auto" in out_yes

    def test_extra_overrides_are_appended(self) -> None:
        out = self._script(
            config_overrides={
                "training.checkpoint_path": "/parent/ckpt.pt",
                "optimization.optimizer.learning_rate": "1e-4",
            }
        )
        assert "-O training.checkpoint_path=/parent/ckpt.pt" in out
        assert "-O optimization.optimizer.learning_rate=1e-4" in out

    def test_custom_slurm_params_override_defaults(self) -> None:
        out = self._script(
            slurm_params={"mem": "32GB", "gpus": 2, "time": "01:00:00"}
        )
        assert "--mem=32GB" in out
        assert "--gpus=2" in out
        assert "--time=01:00:00" in out
        # Multi-GPU branch uses torchrun.
        assert "torchrun" in out
        assert "--nproc_per_node=" in out

    @staticmethod
    def _train_cmd(script: str, *, torchrun: bool) -> str:
        """The one ``TRAIN_CMD=`` line for the requested branch.

        Both branches are emitted into every script -- the selection is a bash
        ``if`` on ``NUM_GPUS`` at run time, not a generator-side choice -- so a
        bare ``in out`` assertion reads the wrong branch's text and passes
        whatever the other one says. That is precisely how the
        ``cli train``-under-torchrun defect stayed green (#2228): the multi-rank
        case asserted only that the word ``torchrun`` appeared somewhere.
        """
        lines = [ln for ln in script.splitlines() if "TRAIN_CMD=" in ln]
        picked = [ln for ln in lines if ("torchrun" in ln) is torchrun]
        assert len(picked) == 1, f"expected one branch, got {picked}"
        return picked[0]

    def test_the_multi_rank_branch_launches_train_distributed(self) -> None:
        """#2228: ``train`` never calls ``setup_distributed``.

        Under torchrun that spelling leaves every rank without a process group,
        so a DeepSpeed/DDP/FSDP arm builds its model and data and only then
        raises out of ``_require_process_group`` at Stage B.
        """
        cmd = self._train_cmd(self._script(slurm_params={"gpus": 4}), torchrun=True)
        assert "-m spectramr.cli train-distributed --config" in cmd
        # The planted spelling: `cli train` followed by anything but `-distributed`.
        assert not re.search(r"cli train(?!-distributed)", cmd), cmd

    def test_multi_node_takes_the_same_branch_as_multi_gpu(self) -> None:
        """The second shape of the same rule: ``nodes > 1`` at one GPU each."""
        cmd = self._train_cmd(
            self._script(slurm_params={"gpus": 1, "nodes": 2}), torchrun=True
        )
        assert "-m spectramr.cli train-distributed --config" in cmd

    def test_the_single_process_branch_keeps_the_plain_train_verb(self) -> None:
        """Not a copy of the rule above -- its inverse.

        One process with no rendezvous genuinely wants ``train``;
        ``train-distributed`` there would call ``setup_distributed`` with no
        launcher behind it. Pinning both directions is what stops a fix to one
        branch being pasted over the other.
        """
        cmd = self._train_cmd(self._script(slurm_params={"gpus": 1}), torchrun=False)
        assert "python -m spectramr.cli train --config" in cmd
        assert "train-distributed" not in cmd

    def test_single_gpu_uses_canonical_cli_entrypoint(self) -> None:
        out = self._script()
        # Single-GPU path must use the canonical module entrypoint. The
        # pre-refactor `python src/main.py` path was removed in the
        # src→spectramr migration and no longer exists — emitting it would
        # crash every single-GPU campaign/orchestrator job on the cluster.
        assert "python -m spectramr.cli train --config" in out
        assert "experiments/inprogress/exp.yaml" in out

    def test_no_dead_src_main_entrypoint(self) -> None:
        """Neither the train nor the inference block may reference the
        removed ``src/main.py`` entrypoint."""
        out = self._script(test_manifest="/shared/test_manifest.txt")
        assert "src/main.py" not in out

    def test_inference_uses_predict_subcommand(self) -> None:
        """Auto-inference must call ``predict --model`` (the real CLI), not
        the non-existent ``infer --checkpoint``/``--config`` form."""
        out = self._script(test_manifest="/shared/test_manifest.txt")
        assert "python -m spectramr.cli predict" in out
        assert "--model " in out
        assert "infer --checkpoint" not in out

    def test_inference_block_carries_no_checkpoint_pattern_of_its_own(self) -> None:
        """The block asks Python for the checkpoint; a bash glob would be a second owner of
        the discovery rule, and the old one picked the deleted writer's files by mtime (#2853)."""
        out = self._script(test_manifest="/shared/test_manifest.txt")
        assert "checkpoint_step_" not in out
        assert "checkpoint_epoch_" not in out
        assert "ls -t" not in out
        assert "python -m spectramr.cli.resolve_checkpoint" in out

    @staticmethod
    def _job_env() -> dict[str, str]:
        """The job's python on PATH and this checkout's ``src`` first, as the script exports."""
        import sys
        from pathlib import Path

        import spectramr

        return {
            **os.environ,
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
            "PYTHONPATH": str(Path(spectramr.__file__).resolve().parents[1]),
            "SPECTRAMR_SUPPRESS_CLINICAL_WARNING": "1",
        }

    @staticmethod
    def _v1(path, mtime: float | None = None):
        import torch

        from tests.fixtures.run_state_files import write_run_state

        path.parent.mkdir(parents=True, exist_ok=True)
        role = "best" if path.stem == "best" else "step"
        write_run_state(path, {"generator": {"w": torch.zeros(1)}}, role=role)
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def _resolved_checkpoint(self, tmp_path) -> subprocess.CompletedProcess[str]:
        """Run the rendered checkpoint lookup against ``tmp_path`` and echo what it chose."""
        script = SLURMBackend.generate_job_script(
            experiment_name="exp_alpha",
            config_path="experiments/inprogress/exp.yaml",
            output_dir=str(tmp_path),
            base_dir="/project/spectramr",
            test_manifest="/shared/m.txt",
        )
        block = script.split("# ── Automated Inference ──", 1)[1].split("MANIFEST=", 1)[0]
        return subprocess.run(
            ["bash", "-c", block + 'echo "CHECKPOINT=${CHECKPOINT}"'],
            env=self._job_env(),
            capture_output=True,
            text=True,
            check=True,
        )

    def test_rendered_lookup_picks_the_latest_step(self, tmp_path) -> None:
        self._v1(tmp_path / "checkpoints" / "step_00002000.pt", mtime=1_000)
        latest = self._v1(tmp_path / "checkpoints" / "step_00003000.safetensors")
        out = self._resolved_checkpoint(tmp_path).stdout
        assert f"CHECKPOINT={latest}" in out

    def test_rendered_lookup_prefers_the_best_checkpoint(self, tmp_path) -> None:
        self._v1(tmp_path / "checkpoints" / "step_00002000.pt")
        best = self._v1(tmp_path / "checkpoints" / "best.pt")
        assert f"CHECKPOINT={best}" in self._resolved_checkpoint(tmp_path).stdout

    def test_rendered_lookup_skips_loudly_without_a_checkpoint(self, tmp_path) -> None:
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "checkpoint_step_3000.safetensors").write_bytes(b"x")
        proc = self._resolved_checkpoint(tmp_path)
        assert "CHECKPOINT=\n" in proc.stdout
        assert "no checkpoint" in proc.stderr
        assert "convert_checkpoints_v1.py" in proc.stderr

    @staticmethod
    def _injection_block(script: str) -> str:
        start = script.index("PARENT_CHECKPOINT_0=")
        return script[start : script.index("\n", script.index("exit 3", start)) + 1]

    @pytest.mark.parametrize("parent_name", ["parent", "par ent $HOME"])
    def test_a_checkpoint_injection_resolves_at_job_start(self, tmp_path, parent_name) -> None:
        """Review Focus 4: a parent with no best still chains through its latest step.

        A space or a ``$`` in the campaign dir must reach the override verbatim.
        """
        parent = tmp_path / parent_name
        step = self._v1(parent / "checkpoints" / "step_00000500.safetensors")
        script = self._script(checkpoint_injections={"model.checkpoint_path": str(parent)})
        block = self._injection_block(script)
        # The single-process TRAIN_CMD line, assembled the way the job assembles it; the
        # injected value must survive its double quotes and the later `eval`.
        train_cmd = next(
            line
            for line in script.splitlines()
            if line.strip().startswith('TRAIN_CMD="python -m spectramr.cli train ')
        )
        proc = subprocess.run(
            ["bash", "-c", block + train_cmd.strip() + "\neval \"printf '%s\\n' $TRAIN_CMD\""],
            env=self._job_env(),
            capture_output=True,
            text=True,
            check=True,
        )
        assert f"model.checkpoint_path={step}" in proc.stdout.splitlines()

    def test_a_parent_without_any_checkpoint_fails_the_job(self, tmp_path) -> None:
        script = self._script(checkpoint_injections={"model.checkpoint_path": str(tmp_path / "p")})
        proc = subprocess.run(
            ["bash", "-c", self._injection_block(script) + "echo FELL_THROUGH"],
            env=self._job_env(),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 3
        assert f"could not resolve a checkpoint from {tmp_path / 'p'}" in proc.stderr
        assert "FELL_THROUGH" not in proc.stdout

    def test_injection_lines_sit_before_the_launch_verb_choice(self) -> None:
        script = self._script(checkpoint_injections={"model.checkpoint_path": "/runs/parent"})
        lines = script.splitlines()
        inject = next(i for i, ln in enumerate(lines) if ln.startswith("PARENT_CHECKPOINT_0="))
        assert lines[inject + 1].startswith('if [[ "${NUM_GPUS}" -gt 1')
        assert script.count('-O model.checkpoint_path=\\"\\${PARENT_CHECKPOINT_0}\\"') == 2

    def test_test_manifest_threads_through(self) -> None:
        out = self._script(test_manifest="/shared/test_manifest.txt")
        # Manifest is referenced in the auto-inference block.
        assert "/shared/test_manifest.txt" in out

    def test_venv_path_default_uses_base_dir(self) -> None:
        out = self._script()
        # default venv path = base_dir/.venv/bin/activate
        assert "/project/spectramr/.venv/bin/activate" in out

    def test_custom_venv_path_is_respected(self) -> None:
        out = self._script(venv_path="/opt/custom/.venv/bin/activate")
        assert "/opt/custom/.venv/bin/activate" in out


# ── generate_array_job_script (one array job for a whole campaign) ──────────


class TestGenerateArrayJobScript:
    """The campaign 'array' backend submits ONE script whose body resolves the
    config for THIS array task from a frozen manifest via spectramr.cli.manifest_dispatch.
    The header carries `#SBATCH --array=0-(N-1)%C`; the body routes output into
    the campaign tree via --output-base."""

    @pytest.fixture(autouse=True)
    def _account(self, monkeypatch):
        """The generators ship no account default -- an allocation name is
        site-specific (#1146). Configure one the way a real user does, so the
        directive assertions below test rendering rather than a baked-in site.
        """
        monkeypatch.setenv("SPECTRAMR_SLURM_ACCOUNT", "test_alloc")
        monkeypatch.delenv("SPECTRAMR_SLURM_PARTITION", raising=False)
        monkeypatch.delenv("SPECTRAMR_SLURM_MAIL_USER", raising=False)

    def _script(self, **kwargs: object) -> str:
        params = {
            "manifest_path": "/camp/array_manifest.txt",
            "n_tasks": 5,
            "concurrency": 8,
            "output_base": "/camp",
            "base_dir": "/project/spectramr",
            "dispatch_dir": "/camp/dispatch",
        }
        params.update(kwargs)
        return SLURMBackend.generate_array_job_script(**params)  # type: ignore[arg-type]

    def test_inference_mode_spells_the_checkpoint_list_and_not_prod(self) -> None:
        script = self._script(
            inference_checkpoints="/camp/inference_checkpoints.txt",
            inference_checkpoints_text="/r/best.safetensors\n",
            manifest_text="/a.yaml\n",
        )
        assert "--inference-checkpoints ${ARRAY_MANIFEST_DIR}/inference_checkpoints.txt" in script
        assert "/r/best.safetensors" in script
        assert "--prod" not in script.split("python -m spectramr.cli.manifest_dispatch", 1)[1]

    def test_inference_mode_excludes_resume_and_fresh(self) -> None:
        for kw in ({"resume": True}, {"fresh": True}):
            with pytest.raises(ValueError, match="inference array"):
                self._script(inference_checkpoints="/c.txt", **kw)

    def test_starts_with_shebang(self) -> None:
        assert self._script().startswith("#!/bin/bash")

    def test_emits_array_directive_with_concurrency(self) -> None:
        # N=5 indices 0..4, throttle %4 (below N, honoured verbatim).
        assert "#SBATCH --array=0-4%4" in self._script(n_tasks=5, concurrency=4)

    def test_single_task_array_is_zero_to_zero(self) -> None:
        # N=1 → %C capped to 1 (a %8 throttle on a 1-task array is misleading).
        assert "#SBATCH --array=0-0%1" in self._script(n_tasks=1)

    def test_concurrency_capped_at_task_count(self) -> None:
        # %C above the task count is pointless; cap it at N so the directive is
        # honest (SLURM tolerates %C>N but it reads misleadingly).
        assert "#SBATCH --array=0-2%3" in self._script(n_tasks=3, concurrency=8)

    def test_rejects_zero_tasks(self) -> None:
        # An empty array (no arms survived filtering) must raise, not emit
        # `--array=0--1` (#9/#15 — no silent malformed directive).
        with pytest.raises(ValueError, match="n_tasks"):
            self._script(n_tasks=0)

    def test_body_invokes_manifest_dispatch_with_task_id(self) -> None:
        out = self._script()
        assert "spectramr.cli.manifest_dispatch" in out
        assert "--manifest /camp/array_manifest.txt" in out
        # the per-task index is the SLURM array task id
        assert "--index ${SLURM_ARRAY_TASK_ID}" in out

    def test_body_runs_prod_and_routes_output(self) -> None:
        out = self._script()
        assert "--prod" in out  # campaign runs the real experiment
        assert "--output-base /camp" in out

    def test_body_writes_bytecode_cache_to_node_local_scratch(self) -> None:
        # The array path runs one task per arm; disabling the bytecode cache
        # (the old PYTHONDONTWRITEBYTECODE=1) forced a full recompile of the
        # ~1000 spectramr.* modules on EVERY task on the shared cluster FS.
        # It must redirect the cache to node-local scratch instead, matching
        # generate_job_script (see the single-job counterpart test).
        out = self._script()
        # The directive itself must be gone; the rationale comment may still
        # name the old ``PYTHONDONTWRITEBYTECODE=1`` for context, so match the
        # ``export`` line rather than the bare token.
        assert "export PYTHONDONTWRITEBYTECODE" not in out
        assert "export PYTHONPYCACHEPREFIX=" in out

    def test_array_account_and_gres_directives(self, monkeypatch) -> None:
        """Same account contract as the per-arm generator (#1146).

        The autouse ``_account`` fixture configures the env the way a real user
        does; this asserts the array header RENDERS it rather than carrying a
        baked-in site, and that a different value follows through.
        """
        out = self._script()
        assert "#SBATCH --account=test_alloc" in out
        assert "#SBATCH --gpus=1" in out
        monkeypatch.setenv("SPECTRAMR_SLURM_ACCOUNT", "my-alloc")
        assert "#SBATCH --account=my-alloc" in self._script()

    def test_custom_slurm_params_override_defaults(self) -> None:
        out = self._script(slurm_params={"mem": "128GB", "gpus": 2, "time": "48:00:00"})
        assert "#SBATCH --mem=128GB" in out
        assert "#SBATCH --gpus=2" in out
        assert "#SBATCH --time=48:00:00" in out

    # The dispatch argv is observed by running the rendered body with the
    # dispatcher replaced by `echo`, so the per-link bash guard is executed
    # rather than pattern-matched (#2542).
    @staticmethod
    def _dispatch_argv(script: str, restart_count: str | None) -> list[str]:
        body = script.split("cd /project/spectramr\n", 1)[1].split("TASK_EXIT_CODE=$?", 1)[0]
        body = body.replace("python -m spectramr.cli.manifest_dispatch", "echo DISPATCH")
        # The granted-GPU/core derivation is exercised on its own in
        # test_slurm_backend_topology.py; this helper observes the resume guard, so
        # the block is cut out and its one output is supplied.
        from spectramr.infrastructure.orchestration.slurm_backend import (
            _environment_guard_block,
            _granted_topology_block,
            _wall_clock_block,
        )

        body = body.replace(_granted_topology_block("/project/spectramr", 8), "")
        body = body.replace(_environment_guard_block("/project/spectramr"), "")
        # The deadline derivation is exercised in test_wall_clock_chaining.py.
        body = body.replace(_wall_clock_block("/project/spectramr"), "")
        env = {"PATH": os.environ["PATH"], "SLURM_ARRAY_TASK_ID": "0", "GPUS_PER_NODE": "1"}
        if restart_count is not None:
            env["SLURM_RESTART_COUNT"] = restart_count
        out = subprocess.run(
            ["bash", "-c", body], env=env, capture_output=True, text=True, check=True
        ).stdout
        line = next(ln for ln in out.splitlines() if ln.startswith("DISPATCH"))
        return line.split()[1:]

    @classmethod
    def _assert_fresh_chain(cls, script: str) -> None:
        """Fresh means: link 0 starts over, and every requeued link resumes."""
        for first in (None, "0"):
            argv = cls._dispatch_argv(script, first)
            assert "--resume-off" in argv and "--requeue-resumes" in argv, argv
            assert "--resume" not in argv, argv
        for later in ("1", "4"):
            argv = cls._dispatch_argv(script, later)
            assert "--resume" in argv and "--resume-off" not in argv, argv

    def test_fresh_starts_the_first_link_over_and_resumes_requeued_links(self) -> None:
        self._assert_fresh_chain(self._script(fresh=True))

    def test_resume_spells_resume_on_every_link(self) -> None:
        script = self._script(resume=True)
        for link in (None, "0", "3"):
            argv = self._dispatch_argv(script, link)
            assert "--resume" in argv and "--resume-off" not in argv, argv

    def test_default_spells_neither_flag(self) -> None:
        """--prod alone: the dispatcher's own if-present chaining applies."""
        script = self._script()
        for link in (None, "2"):
            argv = self._dispatch_argv(script, link)
            assert "--prod" in argv
            assert not {"--resume", "--resume-off", "--requeue-resumes"} & set(argv), argv

    def test_an_unguarded_fresh_script_fails_the_chain_check(self) -> None:
        """Planted: --resume-off on every link restarts each requeued link at 0."""
        unguarded = self._script(fresh=True).replace(
            '"${SLURM_RESTART_COUNT:-0}" -eq 0', "1 -eq 1"
        )
        with pytest.raises(AssertionError):
            self._assert_fresh_chain(unguarded)

    def test_a_bare_prod_script_fails_the_chain_check(self) -> None:
        """Planted: --prod with no guard resumes a stale checkpoint on link 0."""
        with pytest.raises(AssertionError):
            self._assert_fresh_chain(self._script())

    def test_array_tasks_can_be_requeued_and_keep_their_log(self) -> None:
        """Without these, the dispatcher's `scontrol requeue` is refused and a
        yielded link is the last one, so no chain (fresh or default) continues."""
        out = self._script()
        assert "#SBATCH --requeue" in out
        assert "#SBATCH --open-mode=append" in out

    def test_fresh_and_resume_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            self._script(fresh=True, resume=True)

    def test_output_and_error_use_array_tokens(self) -> None:
        # %A = array job id, %a = task id — so per-task logs don't collide.
        out = self._script()
        assert "%A_%a" in out


class TestClinicalDisclaimerSuppression:
    """The batch templates must set the knob the warning itself advertises.

    ``spectramr/__init__.py`` emits a ``UserWarning`` at import telling the reader
    to "Set SPECTRAMR_SUPPRESS_CLINICAL_WARNING=1 to silence this message in batch
    jobs" — and no submitter ever did, so every cluster job log opened with the
    paragraph, once per process (audit and train are separate processes). An
    advertised knob that nothing sets is pitfall #15 from the other direction.
    """

    @pytest.fixture(autouse=True)
    def _account(self, monkeypatch):
        monkeypatch.setenv("SPECTRAMR_SLURM_ACCOUNT", "test_alloc")

    def test_single_arm_script_exports_the_knob(self) -> None:
        script = SLURMBackend.generate_job_script(
            experiment_name="exp_alpha",
            config_path="experiments/inprogress/exp.yaml",
            output_dir="/scratch/results/exp_alpha",
            base_dir="/project/spectramr",
        )
        assert "export SPECTRAMR_SUPPRESS_CLINICAL_WARNING=1" in script
