"""SLURM Backend — Thin wrapper around sbatch / sacct / squeue.

All cluster interaction is funnelled through this module so the
orchestrator remains testable with a mock backend.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spectramr.core.dispatch_snapshot import snapshot_path

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


def _array_resume_block(*, resume: bool, fresh: bool) -> str:
    """Render the bash that sets ``RESUME_FLAG`` for one array-task link.

    ``fresh`` decides per link because every link of a wall-clock chain re-runs
    the same script: only the first may start at iteration 0, and a requeued
    link that did not resume would restart the run it is meant to continue.
    """
    if fresh:
        return (
            "# --fresh: the first link starts at iteration 0; a requeued link resumes\n"
            "# the run that link started, so the wall-clock chain still continues.\n"
            'if [[ "${SLURM_RESTART_COUNT:-0}" -eq 0 ]]; then\n'
            '    RESUME_FLAG="--resume-off --requeue-resumes"\n'
            "else\n"
            '    RESUME_FLAG="--resume"\n'
            "fi\n"
            'echo "Resume: fresh (link ${SLURM_RESTART_COUNT:-0}) -> ${RESUME_FLAG}"\n'
        )
    if resume:
        return 'RESUME_FLAG="--resume"\necho "Resume: --resume (every link)"\n'
    return (
        "# Neither --fresh nor --resume: --prod implies --resume if-present on every\n"
        "# link, so a checkpoint already in the output dir is continued.\n"
        'RESUME_FLAG=""\n'
        'echo "Resume: default (--prod implies --resume if-present)"\n'
    )


def _environment_guard_block(base_dir: str) -> str:
    """Bash that makes sure the job runs the project venv's torch, or stops legibly.

    ``module load torchvision`` put a second, module-provided torch on the path (and the
    PYTHONPATH its module sets wins over the venv's site-packages), so a job ran torch 2.7
    against a package that requires >= 2.8 and failed at its first torch-dependent import
    with a misleading "cannot import name". ``dispatch_experiments.sbatch`` already forbids
    the module load, scrubs a leaked FreeSurfer ``LD_LIBRARY_PATH`` and probes ``import
    torch`` before spending GPU time; the campaign generators had none of that.
    """
    return f"""# ── Environment guard ──
LD_PATH_HELPER="{base_dir}/scripts/common/torch_ld_path.sh"
if [[ ! -f "${{LD_PATH_HELPER}}" ]]; then
    echo "FATAL: missing ${{LD_PATH_HELPER}}; is the campaign base_dir correct?" >&2
    exit 3
fi
source "${{LD_PATH_HELPER}}"
scrub_freesurfer_from_ld_path
echo "[ENV] python: $(command -v python)"
if ! python -c "import torch; print('[ENV] torch', torch.__version__, '| cuda_available', torch.cuda.is_available())"; then
    echo "FATAL: 'import torch' failed in the active environment." >&2
    echo "       active python: $(command -v python)" >&2
    echo "       LD_LIBRARY_PATH: ${{LD_LIBRARY_PATH:-<unset>}}" >&2
    echo "       A module-provided torch or a leaked site module can shadow the venv's;" >&2
    echo "       submit from a shell with no such module loaded, or install into the venv:" >&2
    echo "       pip install -e '.[dev]'." >&2
    exit 3
fi
"""


def _wall_clock_block(base_dir: str) -> str:
    """Export the allocation's deadline so the training loop yields ahead of it.

    The loop stops a margin before the end, saves and leaves the marker
    manifest_dispatch names; manifest_dispatch then requeues the task. Without
    the deadline a task that outlives its allocation is killed at TIMEOUT, which
    ``#SBATCH --requeue`` does not cover. Same helper as the per-arm script.
    """
    return f"""WALL_CLOCK_HELPER="{base_dir}/scripts/common/wall_clock.sh"
if [[ ! -f "${{WALL_CLOCK_HELPER}}" ]]; then
    echo "FATAL: missing ${{WALL_CLOCK_HELPER}}; is the campaign base_dir correct?" >&2
    exit 3
fi
source "${{WALL_CLOCK_HELPER}}"
derive_wall_clock_deadline
"""


def _granted_topology_block(base_dir: str, default_cpus: int) -> str:
    """Bash that derives the GPUs and cores THIS task was granted.

    One copy of the rule, shared by the per-arm and the array generators and by
    ``dispatch_experiments.sbatch`` / ``train_distributed.sbatch``
    (``scripts/common/gpu_count.sh``). The rank count has to come from Slurm's
    allocation, never from the request string and never from a literal: a
    four-GPU grant that forks one rank trains to completion with valid metrics
    and nothing says three quarters of the hardware sat idle (#1274). The thread
    budget is per RANK, because every forked rank inherits the value: handing
    each the node's whole core count is an N-fold oversubscription that reads as
    slow steps, not as an error.
    """
    return f"""# ── GPUs and cores granted to this task ──
GPU_COUNT_HELPER="{base_dir}/scripts/common/gpu_count.sh"
if [[ ! -f "${{GPU_COUNT_HELPER}}" ]]; then
    echo "FATAL: missing ${{GPU_COUNT_HELPER}}; is the campaign base_dir correct?" >&2
    exit 3
fi
source "${{GPU_COUNT_HELPER}}"
derive_gpus_per_node   # sets GPUS_PER_NODE, or exits 1 rather than assume 1
_CPUS="${{SLURM_CPUS_PER_TASK:-{default_cpus}}}"
_THREADS=$(( _CPUS / GPUS_PER_NODE ))
(( _THREADS < 1 )) && _THREADS=1
export OMP_NUM_THREADS="${{_THREADS}}"
export MKL_NUM_THREADS="${{_THREADS}}"
echo "[ENV] cpus-per-task=${{_CPUS}} gpus=${{GPUS_PER_NODE}} -> OMP_NUM_THREADS=${{_THREADS}} per rank"
"""


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

    def query_array_task_status(self, job_ids: list[int]) -> dict[tuple[int, int], JobStatus]:
        """Status of every task of the array jobs ``job_ids``, keyed ``(job, task)``.

        ``query_batch_status`` keys by job id and skips ``<job>_<task>`` lines, so a
        campaign array arm used to read task 0's state for every task: a TIMEOUT
        task was recorded ``completed`` once task 0 finished. Pending tasks appear
        as one ``<job>_[3-10%6]`` line and are expanded here.
        """
        if not job_ids:
            return {}
        if self.dry_run:
            return {}
        result = subprocess.run(
            [
                "sacct",
                "-j",
                ",".join(str(jid) for jid in job_ids),
                "--format=JobID,State,ExitCode,Start,End,Elapsed,NodeList,Reason",
                "--noheader",
                "--parsable2",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning(f"sacct failed for array jobs {job_ids}: {result.stderr.strip()}")
            return {}
        return parse_array_task_lines(result.stdout)

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
        checkpoint_injections: dict[str, str] | None = None,
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
            resume: If True, add ``--resume auto`` to the training command.
                False starts the arm fresh at iteration 0; this script has no
                wall-clock chain, so there is no later link to resume.
            venv_path: Path to Python virtualenv activate script.
            test_manifest: Path to shared test-set manifest for post-training
                inference.  When provided, the script runs inference
                automatically after training completes.
            config_overrides: Extra ``-O key=value`` overrides appended to
                the training command, with values fixed at submission.
            checkpoint_injections: ``inject_as`` -> parent run dir; the parent's
                checkpoint is resolved when the job starts, and the job exits 3
                if the parent left none.

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

        extra_overrides = ""
        if config_overrides:
            for key, value in config_overrides.items():
                extra_overrides += f" -O {key}={value}"

        # A parent's checkpoint is chosen when this job starts, after the dependency
        # finished: best, else its latest step. Resolving at submission would see nothing.
        # The path is shell-quoted because a campaign dir may hold spaces or a `$`.
        injection_lines = ""
        for i, (key, parent_dir) in enumerate(sorted((checkpoint_injections or {}).items())):
            var = f"PARENT_CHECKPOINT_{i}"
            fatal = f"FATAL: could not resolve a checkpoint from {parent_dir} to inject as {key}"
            injection_lines += (
                f'{var}="$(python -m spectramr.cli.resolve_checkpoint '
                f'{shlex.quote(f"{parent_dir}/checkpoints")})" '
                f"|| {{ echo {shlex.quote(fatal)} >&2; exit 3; }}\n"
            )
            # `\$` defers the expansion to `eval`, which then quotes the value once.
            extra_overrides += f' -O {key}=\\"\\${{{var}}}\\"'

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
            # A yield is a requeue of this job: Slurm refuses it without the directive, and
            # append keeps the earlier link's log. A requeued link resumes (see RESUME_ARG).
            requeue=True,
            open_mode="append",
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
# The project venv is the torch SSOT. No `module load torchvision`: a second, module-provided
# torch (and the PYTHONPATH its module sets) shadows the venv's, and the first torch-dependent
# import then fails with a misleading "cannot import name".
if [[ ! -f "{venv_path}" ]]; then
    echo "FATAL: venv activate script not found: {venv_path}" >&2
    exit 3
fi
source "{venv_path}"

# The package lives in src/, so the repo root alone selects nothing: the job then imports
# whichever clone the venv's editable install points at, which can be an older tree than
# the one this script was generated for.
export PYTHONPATH={base_dir}/src:{base_dir}:${{PYTHONPATH:-}}
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

{_granted_topology_block(base_dir, params["cpus_per_task"])}
{_environment_guard_block(base_dir)}
echo "Python: $(which python) ($(python --version))"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "spectramr: $(python -c 'import spectramr; print(spectramr.__file__)' 2>/dev/null)"
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
NUM_GPUS=${{GPUS_PER_NODE}}
NUM_NODES={params["nodes"]}
# The first link follows the arm's own setting. Every REQUEUED link (a wall-clock yield, or a
# node failure) must continue the run that link started, but tolerate an empty directory: the
# strict `auto` spelling raises when there is no checkpoint, which is the wall-clock chain's
# `if-present` spelling the error message itself names.
RESUME_ARG="{resume_flag}"
if [[ "${{SLURM_RESTART_COUNT:-0}}" -gt 0 ]]; then
    RESUME_ARG=" --resume if-present"
fi
# A process-group backend (ddp, fsdp, deepspeed) needs torchrun and `train-distributed` even at
# ONE rank: `train` never creates the group, so the arm builds its model and data and only then
# refuses at Stage B. The answer comes from the same two functions the array path uses (the
# arm's declared parallel.strategy, then the strategy registry), not from the GPU count.
NEEDS_GROUP="$(python -c 'import sys; from spectramr.cli.manifest_dispatch import read_parallel_strategy, requires_process_group; print(int(requires_process_group(read_parallel_strategy(sys.argv[1]))))' "{config_path}")" || {{
    echo "FATAL: could not read parallel.strategy from {config_path}; refusing to guess the launch verb." >&2
    exit 3
}}
{injection_lines}if [[ "${{NUM_GPUS}}" -gt 1 || "${{NUM_NODES}}" -gt 1 || "${{NEEDS_GROUP}}" -eq 1 ]]; then
    # Rendezvous on the first allocated node, keyed by the job id: a fixed port
    # on the node's own name made two jobs on one node join the same c10d group,
    # and gave every node of a multi-node job a different endpoint.
    MASTER_ADDR="$(scontrol show hostnames "${{SLURM_JOB_NODELIST:-localhost}}" | head -n1)"
    MASTER_PORT="$(( 20000 + (${{SLURM_JOB_ID:-0}} % 20000) ))"
    # One torchrun per node: a batch script runs on the first node only, so
    # nodes > 1 needs srun to put a launcher on each of them.
    LAUNCH=""
    [[ "${{NUM_NODES}}" -gt 1 ]] && LAUNCH="srun --kill-on-bad-exit=1 --ntasks=${{NUM_NODES}} --ntasks-per-node=1 "
    TRAIN_CMD="${{LAUNCH}}torchrun --nproc_per_node=${{NUM_GPUS}} --nnodes=${{NUM_NODES}} --rdzv_id=${{SLURM_JOB_ID:-spectramr}} --rdzv_backend=c10d --rdzv_endpoint=${{MASTER_ADDR}}:${{MASTER_PORT}} -m spectramr.cli train-distributed --config \\"{config_path}\\"${{RESUME_ARG}}{output_override}{extra_overrides}"
else
    TRAIN_CMD="python -m spectramr.cli train --config \\"{config_path}\\"${{RESUME_ARG}}{output_override}{extra_overrides}"
fi

echo "Config: {config_path}"
echo "Output: {output_dir}"
echo "Resume: {resume}"
echo ""

# ── Wall-clock yield ──
# A job killed at the wall loses everything since the last periodic save. The deadline is
# exported so the training loop stops itself a margin ahead of it, saves, and leaves a
# marker; the same two helpers the array path and train_distributed.sbatch use.
# The marker lives on SHARED storage, never ${{TMPDIR}}: under c10d rendezvous global rank 0
# is whichever node arrives first, not necessarily the one running this batch script.
WALL_CLOCK_HELPER="{base_dir}/scripts/common/wall_clock.sh"
if [[ ! -f "${{WALL_CLOCK_HELPER}}" ]]; then
    echo "FATAL: missing ${{WALL_CLOCK_HELPER}}; is the campaign base_dir correct?" >&2
    exit 3
fi
source "${{WALL_CLOCK_HELPER}}"
derive_wall_clock_deadline
if [[ -n "${{SPECTRAMR_WALL_CLOCK_DEADLINE:-}}" ]]; then
    mkdir -p "{base_dir}/experiments/results/dispatch"
    export SPECTRAMR_WALL_CLOCK_MARKER="{base_dir}/experiments/results/dispatch/yield_${{SLURM_JOB_ID:-$$}}.json"
    rm -f "${{SPECTRAMR_WALL_CLOCK_MARKER}}"
fi

eval ${{TRAIN_CMD}}
TRAIN_EXIT_CODE=$?

# The marker, not the exit code, says the run yielded on purpose: a yield is a clean partial
# run (every rank exits 0), so a nonzero exit WITH a marker is a crash after rank 0 wrote it.
if [[ -n "${{SPECTRAMR_WALL_CLOCK_MARKER:-}}" && -f "${{SPECTRAMR_WALL_CLOCK_MARKER}}" ]]; then
    if [[ ${{TRAIN_EXIT_CODE}} -ne 0 ]]; then
        echo "FATAL: yield marker present but training exited ${{TRAIN_EXIT_CODE}}; not requeueing." >&2
        exit ${{TRAIN_EXIT_CODE}}
    fi
    # The requeued link re-runs this command with SLURM_RESTART_COUNT > 0, which switches the
    # resume spelling above, so it continues from the checkpoint just written.
    echo "[WallClock] Yield marker found; requeueing job ${{SLURM_JOB_ID}}."
    if scontrol requeue "${{SLURM_JOB_ID}}"; then
        exit 0
    fi
    echo "FATAL: scontrol requeue failed. This run YIELDED and is NOT finished." >&2
    exit 1
fi

if [[ ${{TRAIN_EXIT_CODE}} -eq 0 ]]; then
    echo ""
    echo "✅ Training completed: {experiment_name}"

    # ── Automated Inference ──
    # Python chooses the checkpoint, with the same rule campaign evaluation uses; a bash
    # glob here was a second owner and picked the deleted writer's files by mtime (#2853).
    # With nothing to evaluate the resolver says so on stderr and inference is skipped.
    CHECKPOINT="$(python -m spectramr.cli.resolve_checkpoint "{output_dir}/checkpoints")" || CHECKPOINT=""

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
        fresh: bool = False,
        venv_path: str | None = None,
        no_audit: bool = False,
        manifest_text: str | None = None,
        snapshot_text: str | None = None,
        inference_checkpoints: str | None = None,
        inference_checkpoints_text: str | None = None,
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
            resume: Pass ``--resume`` on every link, so each task continues
                from the latest checkpoint in its output dir.
            fresh: Start each task at iteration 0: the first link
                (``SLURM_RESTART_COUNT`` unset or 0) passes ``--resume-off
                --requeue-resumes`` and every requeued link passes ``--resume``,
                so a wall-clock chain still continues its own run. With neither
                flag, ``--prod`` implies ``--resume if-present`` on every link,
                which resumes any checkpoint already in the output dir.
            venv_path: Virtualenv activate script (defaults to
                ``<base_dir>/.venv/bin/activate``).
            no_audit: Skip the per-task audit pre-flight.
            manifest_text: The manifest's content. When given, each task writes it
                (and ``snapshot_text``) to node-local scratch and reads that copy, so
                a campaign dir moved or cleaned while the array is pending cannot
                break the task.
            snapshot_text: The submit-time snapshot record, beside the manifest copy.
            inference_checkpoints: Frozen checkpoint list aligned with the manifest. Each task
                then grades its line once into ``<output_base>/<stem>/inference`` instead of
                training (``campaign infer``); ``resume`` and ``fresh`` do not apply.
            inference_checkpoints_text: The list's content, embedded beside the manifest copy
                like ``manifest_text`` so a moved campaign dir cannot break a pending task.

        Returns:
            Complete bash script content (with the ``--array`` directive).

        Raises:
            ValueError: If ``n_tasks`` < 1 or ``concurrency`` < 1 — an empty or
                malformed array spec must fail loud, not emit ``--array=0--1`` —
                or if both ``resume`` and ``fresh`` are set.
        """
        if resume and fresh:
            raise ValueError(
                "generate_array_job_script: resume and fresh are mutually exclusive."
            )
        if inference_checkpoints is not None and (resume or fresh):
            raise ValueError(
                "generate_array_job_script: an inference array grades fixed checkpoints; "
                "resume and fresh mean nothing there."
            )
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
        manifest_block = ""
        if manifest_text is not None:
            manifest_block = _embedded_manifest_block(manifest_path, manifest_text, snapshot_text)
            manifest_path = "${ARRAY_MANIFEST_DIR}/" + Path(manifest_path).name
            if inference_checkpoints is not None and inference_checkpoints_text is not None:
                manifest_block += _embedded_file_block(inference_checkpoints, inference_checkpoints_text)
                inference_checkpoints = "${ARRAY_MANIFEST_DIR}/" + Path(inference_checkpoints).name

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
            # Every link after the first is a requeue of this task, which SLURM
            # refuses without the directive; append keeps the earlier link's log.
            requeue=True,
            open_mode="append",
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

        resume_block = _array_resume_block(resume=resume, fresh=fresh)
        audit_flag = " --no-audit" if no_audit else ""
        mode_flags = (
            f"--inference-checkpoints {inference_checkpoints}"
            if inference_checkpoints is not None
            else "--prod ${RESUME_FLAG}"
        )

        script = f"""{header}
# ── Diagnostics ──
echo "=========================================="
echo "Campaign array job ${{SLURM_ARRAY_JOB_ID}} task ${{SLURM_ARRAY_TASK_ID}}"
echo "Node: ${{SLURMD_NODENAME}}  |  GPU: ${{SLURM_GPUS}}"
echo "Start: $(date)"
echo "=========================================="

# ── Environment ──
# The project venv is the torch SSOT. No `module load torchvision`: a second, module-provided
# torch (and the PYTHONPATH its module sets) shadows the venv's, and the first torch-dependent
# import then fails with a misleading "cannot import name".
if [[ ! -f "{venv_path}" ]]; then
    echo "FATAL: venv activate script not found: {venv_path}" >&2
    exit 3
fi
source "{venv_path}"

# The package lives in src/, so the repo root alone selects nothing: the job then imports
# whichever clone the venv's editable install points at, which can be an older tree than
# the one this script was generated for.
export PYTHONPATH={base_dir}/src:{base_dir}:${{PYTHONPATH:-}}
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
# The device mask is left to Slurm. A hard-coded mask to device 0 pinned every
# task to the first device whatever --gpus granted, so a four-GPU task saw one.

cd {base_dir}

{_granted_topology_block(base_dir, params["cpus_per_task"])}
{_environment_guard_block(base_dir)}
{_wall_clock_block(base_dir)}
{resume_block}
{manifest_block}
# ── Resolve + audit + train THIS array task's arm ──
# manifest_dispatch reads config line $SLURM_ARRAY_TASK_ID from the frozen
# manifest, runs the Tier-0/1 audit pre-flight, then trains it as the real
# experiment (--prod = full config max_iterations), routing output into the
# campaign tree (<output-base>/<config-stem>).
python -m spectramr.cli.manifest_dispatch \\
    --manifest {manifest_path} \\
    --index ${{SLURM_ARRAY_TASK_ID}} \\
    --nproc-per-node ${{GPUS_PER_NODE}} \\
    {mode_flags}{audit_flag} \\
    --output-base {output_base} \\
    --dispatch-dir {dispatch_dir}
TASK_EXIT_CODE=$?

echo ""
echo "End: $(date)  (task ${{SLURM_ARRAY_TASK_ID}} exit ${{TASK_EXIT_CODE}})"
exit ${{TASK_EXIT_CODE}}
"""
        return script


def _embedded_manifest_block(manifest_path: str, manifest_text: str, snapshot_text: str | None) -> str:
    """Bash that materialises the frozen manifest (+ snapshot) under node-local scratch."""
    name = Path(manifest_path).name
    parts = [
        'ARRAY_MANIFEST_DIR="${TMPDIR:-/tmp}/${USER}/array_manifest_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"',
        'mkdir -p "${ARRAY_MANIFEST_DIR}"',
        f"cat > \"${{ARRAY_MANIFEST_DIR}}/{name}\" <<'SPECTRAMR_MANIFEST_EOF'\n"
        f"{manifest_text.rstrip()}\nSPECTRAMR_MANIFEST_EOF",
    ]
    if snapshot_text is not None:
        snap = snapshot_path(manifest_path).name
        parts.append(
            f"cat > \"${{ARRAY_MANIFEST_DIR}}/{snap}\" <<'SPECTRAMR_SNAPSHOT_EOF'\n"
            f"{snapshot_text.rstrip()}\nSPECTRAMR_SNAPSHOT_EOF"
        )
    return "# Frozen at submit; the campaign dir may be moved before this task starts.\n" + "\n".join(parts)


def _embedded_file_block(path: str, text: str) -> str:
    """Bash that writes one more frozen list beside the embedded manifest copy."""
    return (
        f"\ncat > \"${{ARRAY_MANIFEST_DIR}}/{Path(path).name}\" <<'SPECTRAMR_LIST_EOF'\n"
        f"{text.rstrip()}\nSPECTRAMR_LIST_EOF"
    )


def _array_tasks(spec: str) -> list[int]:
    """Task indices of a sacct array element: ``7`` or ``[3-10%6]`` / ``[1,4-5]``."""
    if not spec.startswith("["):
        return [int(spec)]
    tasks: list[int] = []
    for part in spec.strip("[]").split("%")[0].split(","):
        lo, _, hi = part.partition("-")
        tasks.extend(range(int(lo), int(hi or lo) + 1))
    return tasks


def parse_array_task_lines(stdout: str) -> dict[tuple[int, int], JobStatus]:
    """``(job, task) -> JobStatus`` from ``sacct --parsable2`` output; steps skipped."""
    statuses: dict[tuple[int, int], JobStatus] = {}
    for line in stdout.strip().split("\n"):
        parts = line.split("|")
        if len(parts) < 8:
            continue
        raw_id = parts[0].strip()
        if "." in raw_id or "_" not in raw_id:
            continue
        job_raw, _, task_raw = raw_id.partition("_")
        try:
            job, tasks = int(job_raw), _array_tasks(task_raw)
        except ValueError:
            continue
        exit_raw = parts[2].strip()
        try:
            exit_code = int(exit_raw.split(":")[0]) if ":" in exit_raw else None
        except ValueError:
            exit_code = None
        for task in tasks:
            statuses[(job, task)] = JobStatus(
                job_id=job,
                state=parts[1].strip(),
                exit_code=exit_code,
                start_time=parts[3].strip() or None,
                end_time=parts[4].strip() or None,
                elapsed=parts[5].strip() or None,
                node=parts[6].strip() or None,
                reason=parts[7].strip() or None,
            )
    return statuses
