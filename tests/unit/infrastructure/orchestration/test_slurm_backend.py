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

import re

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

    def test_resume_flag_threads_through(self) -> None:
        assert "--resume" in self._script(resume=True)
        assert "--resume" not in self._script(resume=False)

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
