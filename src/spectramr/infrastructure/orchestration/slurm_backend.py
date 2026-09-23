"""SLURM Backend — Thin wrapper around sbatch / sacct / squeue.

All cluster interaction is funnelled through this module so the
orchestrator remains testable with a mock backend.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["JobStatus", "SLURMBackend"]


@dataclass
class JobStatus:
    """Parsed SLURM job status."""

    job_id: int
    state: str  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT, etc.
    exit_code: int | None = None
    start_time: str | None = None
    end_time: str | None = None
    elapsed: str | None = None
    node: str | None = None
    reason: str | None = None

    @property
    def _base_state(self) -> str:
        """First whitespace token of the SLURM state.

        ``sacct`` reports cancelled jobs as ``"CANCELLED by <uid>"`` and may
        append a ``+`` suffix on truncation — normalise both so the state
        tables below match.
        """
        return (self.state or "").split()[0].rstrip("+") if self.state else ""

    @property
    def is_unknown(self) -> bool:
        """The job is absent from both ``sacct`` and ``squeue`` (vanished / aged
        out of the accounting retention). Distinct from ``PENDING`` — an unknown
        job has no scheduler record at all, so it must NOT keep normalising to
        the non-terminal ``submitted`` forever (that hangs ``campaign watch``)."""
        return self._base_state in ("", "UNKNOWN")

    @property
    def is_terminal(self) -> bool:
        terminal = {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
            "BOOT_FAIL",
            "DEADLINE",
        }
        return self._base_state in terminal

    @property
    def normalised_state(self) -> str:
        """Map SLURM state names to campaign state names."""
        mapping = {
            "PENDING": "submitted",
            "RUNNING": "running",
            "COMPLETED": "completed",
            "FAILED": "failed",
            "CANCELLED": "cancelled",
            "TIMEOUT": "timeout",
            "OUT_OF_MEMORY": "failed",
            "NODE_FAIL": "failed",
            "BOOT_FAIL": "failed",
            "DEADLINE": "failed",
            "PREEMPTED": "submitted",  # Will be rescheduled
            "REQUEUED": "submitted",
            "SUSPENDED": "running",
        }
        # Genuinely-unknown states keep the historical "submitted" default
        # (a transient new SLURM state should keep polling, not be dropped).
        # The fix here is that multi-token states like "CANCELLED by <uid>" and
        # other real terminal states (DEADLINE, BOOT_FAIL, …) now resolve via
        # ``_base_state`` instead of falling through to that default.
        return mapping.get(self._base_state, "submitted")


class SLURMBackend:
    """Interface to SLURM workload manager.

    Wraps ``sbatch``, ``sacct``, and ``squeue`` CLI tools.  When
    ``dry_run=True`` no commands are actually executed — useful for
    local testing.
    """

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run

    # ── Submit ───────────────────────────────────────────────────

    def submit_job(
        self,
        script_content: str,
        dependency_job_ids: list[int] | None = None,
    ) -> int:
        """Submit a job script to SLURM via sbatch.

        Args:
            script_content: Full bash script content (with #SBATCH directives).
            dependency_job_ids: If provided, submit with
                ``--dependency=afterok:id1:id2:...`` so this job only starts
                after all listed jobs complete successfully.

        Returns:
            SLURM job ID.

        Raises:
            RuntimeError: If sbatch fails.
        """
        if self.dry_run:
            dep_str = ""
            if dependency_job_ids:
                dep_str = f" (afterok:{':'.join(str(j) for j in dependency_job_ids)})"
            logger.info("[DRY RUN] Would submit%s:\n%s", dep_str, script_content[:200])
            return 0

        # WS-E: delegate the actual ``sbatch`` to the single shared submission
        # primitive (the launcher's SlurmBackend.submit_script) so the launcher
        # and the campaign submit through ONE path (one sbatch invocation + job-id
        # parse). Behavior is identical — stdin pipe, ``--dependency=afterok``.
        from spectramr.infrastructure.execution import SlurmBackend

        job_id = SlurmBackend().submit_script(script_content, dependency_job_ids=dependency_job_ids)
        dep_info = f" (depends on jobs {dependency_job_ids})" if dependency_job_ids else ""
        logger.info(f"Submitted SLURM job {job_id}{dep_info}")
        return job_id

    # ── Query ────────────────────────────────────────────────────

    def query_job_status(self, job_id: int) -> JobStatus:
        """Query a single job's status via sacct.

        Args:
            job_id: SLURM job ID.

        Returns:
            JobStatus dataclass.
        """
        if self.dry_run:
            return JobStatus(job_id=job_id, state="PENDING")

        result = subprocess.run(
            [
                "sacct",
                "-j",
                str(job_id),
                "--format=JobID,State,ExitCode,Start,End,Elapsed,NodeList,Reason",
                "--noheader",
                "--parsable2",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )

        if result.returncode != 0:
            logger.warning(f"sacct failed for job {job_id}: {result.stderr.strip()}")
            # Fall back to squeue
            return self._query_squeue(job_id)

        # Parse the main job line (skip .batch, .extern sub-steps)
        for line in result.stdout.strip().split("\n"):
            parts = line.split("|")
            if len(parts) < 8:
                continue
            raw_id = parts[0].strip()
            # Skip sub-steps like "12345.batch"
            if "." in raw_id:
                continue

            state = parts[1].strip()
            exit_code_raw = parts[2].strip()
            exit_code = None
            if exit_code_raw and ":" in exit_code_raw:
                try:
                    exit_code = int(exit_code_raw.split(":")[0])
                except ValueError:
                    pass

            return JobStatus(
                job_id=job_id,
                state=state,
                exit_code=exit_code,
                start_time=parts[3].strip() or None,
                end_time=parts[4].strip() or None,
                elapsed=parts[5].strip() or None,
                node=parts[6].strip() or None,
                reason=parts[7].strip() or None,
            )

        # If sacct returned nothing, try squeue
        return self._query_squeue(job_id)

    def query_batch_status(self, job_ids: list[int]) -> dict[int, JobStatus]:
        """Query status of multiple jobs in one call.

        Args:
            job_ids: List of SLURM job IDs.

        Returns:
            Dict mapping job_id → JobStatus.
        """
        if not job_ids:
            return {}

        if self.dry_run:
            return {jid: JobStatus(job_id=jid, state="PENDING") for jid in job_ids}

        # sacct with comma-separated job IDs
        id_str = ",".join(str(jid) for jid in job_ids)
        result = subprocess.run(
            [
                "sacct",
                "-j",
                id_str,
                "--format=JobID,State,ExitCode,Start,End,Elapsed,NodeList,Reason",
                "--noheader",
                "--parsable2",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

        statuses: dict[int, JobStatus] = {}

        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = line.split("|")
                if len(parts) < 8:
                    continue
                raw_id = parts[0].strip()
                if "." in raw_id:
                    continue
                try:
                    jid = int(raw_id)
                except ValueError:
                    continue

                exit_code = None
                exit_code_raw = parts[2].strip()
                if exit_code_raw and ":" in exit_code_raw:
                    try:
                        exit_code = int(exit_code_raw.split(":")[0])
                    except ValueError:
                        pass

                statuses[jid] = JobStatus(
                    job_id=jid,
                    state=parts[1].strip(),
                    exit_code=exit_code,
                    start_time=parts[3].strip() or None,
                    end_time=parts[4].strip() or None,
                    elapsed=parts[5].strip() or None,
                    node=parts[6].strip() or None,
                    reason=parts[7].strip() or None,
                )

        # Fill missing with individual queries
        for jid in job_ids:
            if jid not in statuses:
                statuses[jid] = self.query_job_status(jid)

        return statuses

    # ── Cancel ───────────────────────────────────────────────────

    def cancel_job(self, job_id: int) -> None:
        """Cancel a running or pending job.

        Args:
            job_id: SLURM job ID.
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would cancel job {job_id}")
            return

        result = subprocess.run(
            ["scancel", str(job_id)],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

        if result.returncode != 0:
            logger.warning(f"scancel failed for job {job_id}: {result.stderr.strip()}")
        else:
            logger.info(f"Cancelled SLURM job {job_id}")

    def cancel_batch(self, job_ids: list[int]) -> None:
        """Cancel multiple jobs."""
        for jid in job_ids:
            self.cancel_job(jid)

    # ── Internal ─────────────────────────────────────────────────

    def _query_squeue(self, job_id: int) -> JobStatus:
        """Fallback query via squeue (for pending/running jobs not yet in sacct)."""
        try:
            result = subprocess.run(
                [
                    "squeue",
                    "-j",
                    str(job_id),
                    "--format=%i|%T|%r",
                    "--noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split("|")
                if len(parts) >= 2:
                    return JobStatus(
                        job_id=job_id,
                        state=parts[1].strip(),
                        reason=parts[2].strip() if len(parts) > 2 else None,
                    )

        except Exception as e:
            logger.debug(f"squeue query failed for {job_id}: {e}")

        # Unknown status
        return JobStatus(job_id=job_id, state="UNKNOWN")

    # ── Script Generation ────────────────────────────────────────

    @staticmethod
    def generate_job_script(
        experiment_name: str,
        config_path: str,
        output_dir: str,
        base_dir: str,
        slurm_params: dict[str, Any] | None = None,
        resume: bool = False,
        venv_path: str | None = None,
        test_manifest: str | None = None,
        config_overrides: dict[str, str] | None = None,
    ) -> str:
        """Generate a SLURM job script for a single experiment.

        Args:
            experiment_name: Human-readable experiment name.
            config_path: Path to experiment YAML config.
            output_dir: Directory for SLURM output and results.
                Training output is redirected here via --override so that
                checkpoints, CSVs, and logs land inside the campaign tree.
            base_dir: Project root directory.
            slurm_params: SLURM resource parameters.
            resume: If True, add --resume auto to training command.
            venv_path: Path to Python virtualenv activate script.
            test_manifest: Path to shared test-set manifest for post-training
                inference.  When provided, the script runs inference
                automatically after training completes.
            config_overrides: Extra ``-O key=value`` overrides appended to
                the training command (e.g., for checkpoint injection from
                a parent stage).

        Returns:
            Complete bash script content.
        """
        params = {
            # No literal here: ResourceSpec is the single owner of account
            # resolution ($SPECTRAMR_SLURM_ACCOUNT). This generator used to
            # carry its own hardcoded value and read no env var at all, which
            # is the second-submitter half of #1146.
            "account": None,
            # No default partition: GPUs are allocated via --gres=gpu:N and this
            # cluster has no dedicated "gpu" partition. Pass partition=<name> in
            # slurm_params only if a specific cluster requires one.
            "partition": None,
            "time": "120:00:00",
            "mem": "64GB",
            "cpus_per_task": 8,
            "gpus": 1,
            "nodes": 1,
            "ntasks": 1,
            "mail_type": "END,FAIL",
        }
        if slurm_params:
            params.update(slurm_params)

        if venv_path is None:
            venv_path = f"{base_dir}/.venv/bin/activate"

        resume_flag = " --resume auto" if resume else ""

        # Override training.output_dir so checkpoints, CSVs, and logs
        # are written directly into the campaign directory structure.
        output_override = f" -O training.output_dir={output_dir}"

        # Extra config overrides (e.g., checkpoint injection from parent stage)
        extra_overrides = ""
        if config_overrides:
            for key, value in config_overrides.items():
                extra_overrides += f" -O {key}={value}"

        # WS-E consolidation: render the #SBATCH header via the shared
        # SlurmBackend SSOT (same generator the launcher uses), so the directive
        # format / partition-only-when-set rule no longer drifts between the two.
        from spectramr.core import env_names
        from spectramr.infrastructure.execution import ResourceSpec, SlurmBackend

        resources = ResourceSpec(
            account=params["account"],
            partition=params["partition"],
            mem=params["mem"],
            gpus=params["gpus"],
            time=params["time"],
            nodes=params["nodes"],
            ntasks=params["ntasks"],
            cpus_per_task=params["cpus_per_task"],
        )
        header = SlurmBackend().render_directives(
            resources,
            job_name=experiment_name,
            output=f"{output_dir}/slurm_%j.out",
            error=f"{output_dir}/slurm_%j.err",
            mail_type=params["mail_type"],
            mail_user=os.environ.get(env_names.SPECTRAMR_SLURM_MAIL_USER),
        )

        script = f"""{header}
# ── Diagnostics ──
echo "=========================================="
echo "SLURM Job: ${{SLURM_JOB_ID}} / {experiment_name}"
echo "Node: ${{SLURMD_NODENAME}}  |  GPU: ${{SLURM_GPUS}}"
echo "Start: $(date)"
echo "=========================================="

# ── Environment ──
module load torchvision 2>/dev/null || true

if [[ -f "{venv_path}" ]]; then
    source "{venv_path}"
fi

export PYTHONPATH={base_dir}:${{PYTHONPATH}}
# Bytecode cache: write .pyc to NODE-LOCAL scratch rather than disabling
# it. With ~1000 spectramr.* modules, disabling the cache forced a full
# source recompile on EVERY array task — a large, repeated startup tax on
# shared cluster filesystems. Redirecting the cache to per-node local disk
# means each node compiles once and reuses the cache on subsequent tasks
# (and avoids shared-FS __pycache__ write contention). Re-measure with
# "python -X importtime" cold vs warm on a compute node.
export PYTHONPYCACHEPREFIX="${{TMPDIR:-/tmp}}/${{USER}}/spectramr_pycache"
# The clinical-use disclaimer is a UserWarning emitted at ``import spectramr``.
# It advertises this knob as the way to silence it "in batch jobs", but no
# submitter ever set it, so every job log opened with the same paragraph
# (once per process — audit and train are separate processes).
export SPECTRAMR_SUPPRESS_CLINICAL_WARNING=1

cd {base_dir}

echo "Python: $(which python) ($(python --version))"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \\"N/A\\")')"
echo ""

# ── Training ──
# Note: training.output_dir is overridden to redirect all outputs
# (checkpoints, metrics CSVs) into the campaign directory tree.
# v6.2 PR-14: when the per-arm parallel block requests >1 GPU, dispatch
# through torchrun so FSDP / DDP land on every rank. Single-GPU path is
# unchanged.
#
# The multi-rank verb is `train-distributed`, NOT `train`: `train` never calls
# `setup_distributed`, so no process group exists and every group-requiring
# strategy raises out of `_require_process_group` during `adopt` -- at Stage B,
# after the model and the data are already built. `launcher.py` fixed the same
# defect in its own emitter; this was the copy that kept it (#2228).
NUM_GPUS={params["gpus"]}
NUM_NODES={params["nodes"]}
if [[ "${{NUM_GPUS}}" -gt 1 || "${{NUM_NODES}}" -gt 1 ]]; then
    TRAIN_CMD="torchrun --nproc_per_node=${{NUM_GPUS}} --nnodes=${{NUM_NODES}} --rdzv_backend=c10d --rdzv_endpoint=${{SLURMD_NODENAME:-127.0.0.1}}:29500 -m spectramr.cli train-distributed --config \\"{config_path}\\"{resume_flag}{output_override}{extra_overrides}"
else
    export CUDA_VISIBLE_DEVICES=0
    TRAIN_CMD="python -m spectramr.cli train --config \\"{config_path}\\"{resume_flag}{output_override}{extra_overrides}"
fi

echo "Config: {config_path}"
echo "Output: {output_dir}"
echo "Resume: {resume}"
echo ""

eval ${{TRAIN_CMD}}
TRAIN_EXIT_CODE=$?

if [[ ${{TRAIN_EXIT_CODE}} -eq 0 ]]; then
    echo ""
    echo "✅ Training completed: {experiment_name}"

    # ── Automated Inference ──
    # Resolve the checkpoint via the REAL writer conventions (best.{{pt,safetensors}}
    # alias, then checkpoint_best.*, then newest checkpoint_step_/epoch_*). The old
    # ``model_iter_*.pt`` glob matched nothing either writer produces, so the
    # fallback was dead and safetensors runs were never found.
    CKPT_DIR="{output_dir}/checkpoints"
    CHECKPOINT=""
    for cand in "${{CKPT_DIR}}/best.pt" "${{CKPT_DIR}}/best.safetensors" \\
                "${{CKPT_DIR}}/checkpoint_best.pt" "${{CKPT_DIR}}/checkpoint_best.safetensors"; do
        if [[ -f "${{cand}}" ]]; then CHECKPOINT="${{cand}}"; break; fi
    done
    if [[ -z "${{CHECKPOINT}}" ]]; then
        CHECKPOINT=$(ls -t "${{CKPT_DIR}}"/checkpoint_step_*.pt "${{CKPT_DIR}}"/checkpoint_step_*.safetensors \\
            "${{CKPT_DIR}}"/checkpoint_epoch_*.pt "${{CKPT_DIR}}"/checkpoint_epoch_*.safetensors 2>/dev/null | head -1)
    fi

    # Use campaign test manifest if available, fall back to experiment-local
    MANIFEST=""
    if [[ -f "{test_manifest or ""}" ]]; then
        MANIFEST="{test_manifest or ""}"
    elif [[ -f "{output_dir}/test_split_manifest.txt" ]]; then
        MANIFEST="{output_dir}/test_split_manifest.txt"
    fi

    if [[ -n "${{CHECKPOINT}}" && -f "${{CHECKPOINT}}" && -n "${{MANIFEST}}" ]]; then
        INFER_OUT="{output_dir}/inference_test_split"
        mkdir -p "${{INFER_OUT}}"
        python -m spectramr.cli predict \\
            --model "${{CHECKPOINT}}" \\
            --input "${{MANIFEST}}" \\
            --output "${{INFER_OUT}}"
        echo "Inference results: ${{INFER_OUT}}"
    else
        echo "Skipping auto-inference (checkpoint=${{CHECKPOINT:-missing}}, manifest=${{MANIFEST:-missing}})"
    fi
else
    echo ""
    echo "❌ Training failed: {experiment_name} (exit ${{TRAIN_EXIT_CODE}})"
fi

echo ""
echo "End: $(date)"
exit ${{TRAIN_EXIT_CODE}}
"""
        return script

    @staticmethod
    def generate_array_job_script(
        manifest_path: str,
        n_tasks: int,
        output_base: str,
        base_dir: str,
        *,
        concurrency: int = 8,
        dispatch_dir: str | None = None,
        slurm_params: dict[str, Any] | None = None,
        resume: bool = False,
        venv_path: str | None = None,
        no_audit: bool = False,
    ) -> str:
        """Generate ONE SLURM job-array script for a whole campaign cohort.

        Unlike :meth:`generate_job_script` (one script per arm), this emits a
        single script submitted with ``#SBATCH --array=0-(n_tasks-1)%concurrency``.
        Each array task resolves ITS config from ``manifest_path`` (line =
        ``$SLURM_ARRAY_TASK_ID``) via ``spectramr.cli.manifest_dispatch``, runs the
        Tier-0/1 audit pre-flight, then trains it as a production run
        (``--prod``), routing output into ``<output_base>/<config-stem>`` so the
        campaign's ``_discover_results`` finds the checkpoints.

        Args:
            manifest_path: Frozen newline-separated config list (one per arm),
                aligned with the array indices.
            n_tasks: Number of arms (array length). Must be >= 1.
            output_base: Campaign dir; each task writes to ``<base>/<stem>``.
            base_dir: Project root.
            concurrency: ``%N`` throttle — capped at ``n_tasks`` (a larger value
                is misleading). Must be >= 1.
            dispatch_dir: Per-task audit-log root (defaults to
                ``<output_base>/dispatch``).
            slurm_params: SLURM resource overrides (mem, gpus, time, …).
            resume: Append ``--resume`` to each task's train.
            venv_path: Virtualenv activate script (defaults to
                ``<base_dir>/.venv/bin/activate``).
            no_audit: Skip the per-task audit pre-flight.

        Returns:
            Complete bash script content (with the ``--array`` directive).

        Raises:
            ValueError: If ``n_tasks`` < 1 or ``concurrency`` < 1 — an empty or
                malformed array spec must fail loud, not emit ``--array=0--1``.
        """
        if n_tasks < 1:
            raise ValueError(
                f"generate_array_job_script needs n_tasks >= 1 (got {n_tasks}); "
                "an empty cohort cannot become a SLURM array."
            )
        if concurrency < 1:
            raise ValueError(
                f"generate_array_job_script needs concurrency >= 1 (got {concurrency})."
            )

        params: dict[str, Any] = {
            # No literal here: ResourceSpec is the single owner of account
            # resolution ($SPECTRAMR_SLURM_ACCOUNT). This generator used to
            # carry its own hardcoded value and read no env var at all, which
            # is the second-submitter half of #1146.
            "account": None,
            "partition": None,
            "time": "120:00:00",
            "mem": "64GB",
            "cpus_per_task": 8,
            "gpus": 1,
            "nodes": 1,
            "ntasks": 1,
            "mail_type": "END,FAIL",
        }
        if slurm_params:
            params.update(slurm_params)

        if venv_path is None:
            venv_path = f"{base_dir}/.venv/bin/activate"
        if dispatch_dir is None:
            dispatch_dir = f"{output_base}/dispatch"

        # %C above the task count reads misleadingly — cap it at N.
        eff_concurrency = min(concurrency, n_tasks)
        array_spec = f"0-{n_tasks - 1}%{eff_concurrency}"

        from spectramr.core import env_names
        from spectramr.infrastructure.execution import ResourceSpec, SlurmBackend

        resources = ResourceSpec(
            account=params["account"],
            partition=params["partition"],
            mem=params["mem"],
            gpus=params["gpus"],
            time=params["time"],
            nodes=params["nodes"],
            ntasks=params["ntasks"],
            cpus_per_task=params["cpus_per_task"],
            array=array_spec,
        )
        # %A = array job id, %a = task id — keep per-task logs from colliding.
        header = SlurmBackend().render_directives(
            resources,
            job_name="campaign_array",
            output=f"{dispatch_dir}/slurm_%A_%a.out",
            error=f"{dispatch_dir}/slurm_%A_%a.err",
            mail_type=params["mail_type"],
            mail_user=os.environ.get(env_names.SPECTRAMR_SLURM_MAIL_USER),
        )

        resume_flag = " --resume" if resume else ""
        audit_flag = " --no-audit" if no_audit else ""

        script = f"""{header}
# ── Diagnostics ──
echo "=========================================="
echo "Campaign array job ${{SLURM_ARRAY_JOB_ID}} task ${{SLURM_ARRAY_TASK_ID}}"
echo "Node: ${{SLURMD_NODENAME}}  |  GPU: ${{SLURM_GPUS}}"
echo "Start: $(date)"
echo "=========================================="

# ── Environment ──
module load torchvision 2>/dev/null || true

if [[ -f "{venv_path}" ]]; then
    source "{venv_path}"
fi

export PYTHONPATH={base_dir}:${{PYTHONPATH}}
# Bytecode cache: write .pyc to NODE-LOCAL scratch rather than disabling it.
# This is the array path — literally "EVERY array task" — so disabling the
# cache (the old PYTHONDONTWRITEBYTECODE=1) forced a full source recompile of
# the ~1000 spectramr.* modules on every task, the exact startup tax the
# single-arm script avoids. Redirect to per-node local disk so each node
# compiles once and reuses it (and avoids shared-FS __pycache__ contention).
export PYTHONPYCACHEPREFIX="${{TMPDIR:-/tmp}}/${{USER}}/spectramr_pycache"
# The clinical-use disclaimer is a UserWarning emitted at ``import spectramr``.
# It advertises this knob as the way to silence it "in batch jobs", but no
# submitter ever set it, so every job log opened with the same paragraph
# (once per process — audit and train are separate processes).
export SPECTRAMR_SUPPRESS_CLINICAL_WARNING=1
export CUDA_VISIBLE_DEVICES=0

cd {base_dir}

# ── Resolve + audit + train THIS array task's arm ──
# manifest_dispatch reads config line $SLURM_ARRAY_TASK_ID from the frozen
# manifest, runs the Tier-0/1 audit pre-flight, then trains it as the real
# experiment (--prod = full config max_iterations), routing output into the
# campaign tree (<output-base>/<config-stem>).
python -m spectramr.cli.manifest_dispatch \\
    --manifest {manifest_path} \\
    --index ${{SLURM_ARRAY_TASK_ID}} \\
    --prod{resume_flag}{audit_flag} \\
    --output-base {output_base} \\
    --dispatch-dir {dispatch_dir}
TASK_EXIT_CODE=$?

echo ""
echo "End: $(date)  (task ${{SLURM_ARRAY_TASK_ID}} exit ${{TASK_EXIT_CODE}})"
exit ${{TASK_EXIT_CODE}}
"""
        return script
