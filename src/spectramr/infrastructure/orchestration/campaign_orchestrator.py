"""Campaign Orchestrator — Submit, track, and manage experiment campaigns.

Coordinates the full campaign lifecycle:
  1. Parse campaign YAML
  2. Generate ablation configs (if axes defined)
  3. Submit all experiments via SLURM
  4. Track progress by polling SLURM
  5. Trigger evaluation when all experiments complete
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spectramr.config.schemas.campaign import (
    CampaignConfigSchema,
    CampaignExperimentSchema,
    SlurmDefaultsSchema,
)
from spectramr.infrastructure.orchestration.ablation_config_generator import (
    AblationConfigGenerator,
)
from spectramr.infrastructure.orchestration.campaign_state import (
    CampaignState,
    ExperimentStatus,
)
from spectramr.infrastructure.orchestration.slurm_backend import SLURMBackend

logger = logging.getLogger(__name__)

__all__ = ["CampaignOrchestrator"]


class CampaignOrchestrator:
    """Orchestrates experiment campaigns on a SLURM cluster.

    Usage::

        orchestrator = CampaignOrchestrator(base_dir="${SPECTRAMR_DATA_ROOT}")
        state = orchestrator.submit_campaign("experiments/campaigns/my_campaign.yaml")
        orchestrator.check_progress(state.campaign_dir)
    """

    #: Where campaign arms execute. ``slurm`` (default) submits sbatch jobs;
    #: ``docker`` / ``apptainer`` run each arm in a container, synchronously
    #: (sequentially). ``local`` is not supported here — the in-process backend
    #: lives in the cli layer (infrastructure may not import it); run a single
    #: config with ``spectramr train`` instead.
    _CAMPAIGN_WHERE = ("slurm", "docker", "apptainer")

    def __init__(
        self,
        base_dir: str = ".",
        dry_run: bool = False,
        resume: bool = False,
        only: list[str] | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        where: str = "slurm",
        slurm_overrides: dict[str, Any] | None = None,
    ) -> None:
        """Initialise orchestrator.

        Args:
            base_dir: Project root directory (for SLURM scripts).
            dry_run: If True, generate scripts but do not submit.
            resume: If True, add --resume auto to training commands.
            where: Execution target for each arm — ``slurm`` (default, sbatch),
                ``docker`` or ``apptainer`` (run each arm in a container,
                sequentially).
            only: If non-empty, restrict submission to arms whose ``name``
                appears in this list. Mutually allowed with the selectors
                below — they all combine via logical AND.
            include: Selector strings of the form ``"key=value"``,
                evaluated against each arm's ``name`` / ``role`` /
                ``tags``. ``key=name``, ``key=role`` or ``tag.<key>``
                are recognised. Repeated includes combine via OR
                (the arm matches if **any** include matches).
            exclude: Same syntax as ``include``; arms matching **any**
                exclude are dropped.
            slurm_overrides: SLURM resources to apply on top of the campaign's
                ``slurm_defaults`` and any per-group/per-arm override. The
                world size belongs here rather than in the YAML: it is a
                property of the allocation you have today, and under a sharded
                strategy it also sets the effective batch (``num_devices`` x
                ``data.batch_size``, with no world-size LR scaling in-tree), so
                committing one number to the corpus would fix an optimisation
                decision on behalf of every future submitter.
        """
        if where not in self._CAMPAIGN_WHERE:
            raise ValueError(
                f"Unknown campaign --where {where!r}. Choose from {list(self._CAMPAIGN_WHERE)}."
            )
        self.slurm_overrides = self._validate_slurm_overrides(slurm_overrides)
        self.base_dir = str(Path(base_dir).resolve())
        self.dry_run = dry_run
        self.resume = resume
        self.where = where
        self.only = list(only or [])
        self.include = list(include or [])
        self.exclude = list(exclude or [])
        self.slurm = SLURMBackend(dry_run=dry_run)

    # ── SLURM resources ──────────────────────────────────────────

    @staticmethod
    def _validate_slurm_overrides(raw: dict[str, Any] | None) -> dict[str, Any]:
        """Check keys and types against the schema, and raise on anything else.

        The job-script generator takes a plain dict, so an unrecognised key
        would be dropped in silence and the submitter would read the absence of
        an error as the override having applied (pitfall #15). Validating the
        merged result through ``SlurmDefaultsSchema`` also coerces ``"4"`` to
        ``4`` and rejects ``gpus=-1``, which argparse's ``key=value`` shape
        cannot do on its own.
        """
        if not raw:
            return {}
        allowed = set(SlurmDefaultsSchema.model_fields)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(
                f"Unknown SLURM override key(s): {', '.join(unknown)}. "
                f"Valid keys are {', '.join(sorted(allowed))}."
            )
        # Round-trip through the schema so a bad value fails here, on the login
        # node, rather than as an sbatch rejection per arm.
        return SlurmDefaultsSchema(**raw).model_dump(include=set(raw))

    def _resolve_slurm_params(
        self,
        config: CampaignConfigSchema,
        *layered: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Campaign defaults, then each *layered* override, then the CLI's.

        One owner for the precedence, because three call sites built it
        independently and only one of them consulted the per-arm layer
        (non-negotiable 17). The CLI goes last on purpose: it is the only layer
        that knows what allocation the submitter actually holds.
        """
        params = config.slurm_defaults.model_dump()
        for layer in layered:
            if layer:
                params.update(layer)
        params.update(self.slurm_overrides)
        return params

    # ── Filtering ────────────────────────────────────────────────

    @staticmethod
    def _selector_matches(selector: str, exp: CampaignExperimentSchema) -> bool:
        """Evaluate a single ``key=value`` selector against an arm.

        Recognised keys:
        ``name``        — match against ``exp.name``
        ``role``        — match against ``exp.role``
        ``tag.<key>``   — match against ``exp.tags[<key>]``
        """
        if "=" not in selector:
            raise ValueError(f"Invalid selector {selector!r}: expected 'key=value' form.")
        key, value = selector.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "name":
            return exp.name == value
        if key == "role":
            return exp.role == value
        if key.startswith("tag."):
            return exp.tags.get(key[4:]) == value
        raise ValueError(
            f"Unknown selector key {key!r} in {selector!r}. Use 'name=', 'role=', or 'tag.<key>='."
        )

    def _apply_filters(
        self, experiments: list[CampaignExperimentSchema]
    ) -> list[CampaignExperimentSchema]:
        """Return the subset of arms surviving --only / --include / --exclude."""
        kept: list[CampaignExperimentSchema] = []
        for exp in experiments:
            if self.only and exp.name not in self.only:
                continue
            if self.include and not any(self._selector_matches(s, exp) for s in self.include):
                continue
            if self.exclude and any(self._selector_matches(s, exp) for s in self.exclude):
                continue
            kept.append(exp)
        return kept

    # ── Submit ───────────────────────────────────────────────────

    def submit_campaign(
        self,
        campaign_config_path: str,
    ) -> CampaignState:
        """Submit all experiments in a campaign.

        Args:
            campaign_config_path: Path to campaign YAML.

        Returns:
            CampaignState with job IDs populated.
        """
        config = CampaignConfigSchema.from_yaml(campaign_config_path)
        campaign_dir = Path(self.base_dir) / config.output_dir
        campaign_dir.mkdir(parents=True, exist_ok=True)

        # Apply --only / --include / --exclude selectors to the parallel
        # experiments list. Sequential stage_groups are filtered inside
        # ``_submit_sequential``.
        filtered_count_before = len(config.enabled_experiments)
        filtered_experiments = self._apply_filters(config.enabled_experiments)
        filtered_count_after = len(filtered_experiments)

        logger.info("═══════════════════════════════════════════════════")
        logger.info(f"  Campaign: {config.name}")
        logger.info(f"  Description: {config.description}")
        if filtered_count_after != filtered_count_before:
            logger.info(
                f"  Experiments: {filtered_count_after} (filtered from {filtered_count_before})"
            )
        else:
            logger.info(f"  Experiments: {filtered_count_after}")
        logger.info(f"  Ablation axes: {len(config.ablation_axes)}")
        logger.info(f"  Output: {campaign_dir}")
        logger.info(f"  Dry run: {self.dry_run}")
        logger.info("═══════════════════════════════════════════════════")

        # Initialise campaign state
        state = CampaignState(
            campaign_name=config.name,
            campaign_config_path=str(campaign_config_path),
            campaign_dir=str(campaign_dir),
            description=config.description,
            test_manifest=config.test_manifest,
        )

        # Dispatch based on execution mode
        if config.execution == "sequential":
            if self.where != "slurm":
                raise NotImplementedError(
                    f"Sequential campaigns require --where slurm (got "
                    f"{self.where!r}); container fan-out runs arms synchronously "
                    f"and supports parallel-mode campaigns only in this version."
                )
            return self._submit_sequential(config, state, campaign_dir)

        if config.execution == "array":
            if self.where != "slurm":
                raise NotImplementedError(
                    f"Array campaigns require --where slurm (got {self.where!r}); "
                    f"a SLURM job array is a scheduler construct with no container "
                    f"equivalent. Use execution: parallel for container fan-out."
                )
            return self._submit_array(config, state, campaign_dir, filtered_experiments)

        # Campaign-level resume: skip arms a prior run already finished or still
        # has in flight (re-submitting them would duplicate jobs and clobber
        # per-arm outputs). Failed/pending arms are still re-submitted.
        handled = self._prior_handled(campaign_dir)

        # Collect all experiments to submit
        experiments_to_submit: list[tuple[str, str, str, dict]] = []
        # (name, config_path, role, slurm_overrides)

        # 1. Named experiments (post-filter)
        for exp in filtered_experiments:
            config_path = self._resolve_config_path(exp.config)
            experiments_to_submit.append((exp.name, config_path, exp.role, exp.slurm_overrides))

        # 2. Ablation experiments (generate configs)
        if config.ablation_axes:
            ablation_configs = self._generate_ablation_configs(config, campaign_dir)
            for abl_path in ablation_configs:
                abl_name = Path(abl_path).stem
                experiments_to_submit.append((abl_name, abl_path, "ablation", {}))

        # Submit each experiment
        submitted = 0
        failed = 0

        skipped = 0
        for name, cfg_path, role, overrides in experiments_to_submit:
            # Resume: carry the prior status forward (so the saved state stays
            # complete) and do NOT re-submit.
            if name in handled:
                state.experiments.append(handled[name])
                skipped += 1
                logger.info(f"↻ Resume: skipping {name} (already {handled[name].status})")
                continue

            # Validate config exists
            if not Path(cfg_path).exists():
                logger.error(f"✗ Config not found: {cfg_path}")
                state.experiments.append(
                    ExperimentStatus(
                        name=name,
                        config_path=cfg_path,
                        role=role,
                        status="failed",
                        error_message=f"Config not found: {cfg_path}",
                    )
                )
                failed += 1
                continue

            # Create experiment output directory
            exp_output = campaign_dir / name
            exp_output.mkdir(parents=True, exist_ok=True)

            slurm_params = self._resolve_slurm_params(config, overrides)

            # Resolve test manifest path for auto-inference
            test_manifest_path = None
            if config.test_manifest:
                test_manifest_path = self._resolve_config_path(config.test_manifest)

            # Submit/run one arm via the configured backend (slurm or container).
            try:
                exp_status = self._submit_one_arm(
                    name=name,
                    cfg_path=cfg_path,
                    exp_output=exp_output,
                    slurm_params=slurm_params,
                    test_manifest_path=test_manifest_path,
                    role=role,
                )
                state.experiments.append(exp_status)
                if exp_status.status == "failed":
                    failed += 1
                else:
                    submitted += 1
            except RuntimeError as e:
                logger.error(f"✗ Failed to submit {name}: {e}")
                state.experiments.append(
                    ExperimentStatus(
                        name=name,
                        config_path=cfg_path,
                        role=role,
                        status="failed",
                        error_message=str(e),
                    )
                )
                failed += 1

            # Small delay to avoid overwhelming the scheduler
            time.sleep(0.3)

        # Save state
        state.save()

        logger.info(
            f"\n  Submitted: {submitted}  |  Failed: {failed}  |  Skipped (resume): {skipped}"
        )
        logger.info(f"  State saved: {state.campaign_dir}/campaign_state.json")
        logger.info(f"  Monitor: python -m spectramr.cli campaign status {campaign_dir}")

        return state

    # ── Array submission (one SLURM job array for the whole cohort) ──

    def _submit_array(
        self,
        config: CampaignConfigSchema,
        state: CampaignState,
        campaign_dir: Path,
        filtered_experiments: list[CampaignExperimentSchema],
    ) -> CampaignState:
        """Submit the whole cohort as ONE SLURM job array.

        Unlike parallel mode (one ``sbatch`` per arm — N blocking login-node
        calls), this freezes a manifest of the surviving arms' configs and makes
        a SINGLE ``sbatch --array=0-(M-1)%C`` call. Each array task resolves its
        config by ``$SLURM_ARRAY_TASK_ID`` (line in the manifest) and routes its
        output into ``<campaign_dir>/<config-stem>``.

        Tracking is intentionally minimal — every arm shares the single array
        job id (``state.experiments[*].slurm_job_id``) and is distinguished by
        ``array_task_id``; ``campaign status`` polls the one job and
        ``_discover_results`` finds each arm's checkpoints by its recorded
        ``output_dir``.

        Missing configs are recorded ``failed`` and excluded from the array (so
        the indices stay contiguous). A file-stem collision among the surviving
        arms is fatal (two arms would write the same output dir) — raise rather
        than silently overwrite (#9/#15). Campaign-level ``--resume`` carries
        already-handled arms forward and drops them from the array.
        """
        handled = self._prior_handled(campaign_dir)

        # Resolve the surviving arms (existing configs, not already handled).
        # Each entry: (arm_name, role, resolved_config_path, stem, output_dir).
        manifest_arms: list[tuple[str, str, str, str, Path]] = []
        skipped = 0
        for exp in filtered_experiments:
            if exp.name in handled:
                state.experiments.append(handled[exp.name])
                skipped += 1
                logger.info(f"↻ Resume: skipping {exp.name} (already {handled[exp.name].status})")
                continue

            cfg_path = self._resolve_config_path(exp.config)
            if not Path(cfg_path).exists():
                logger.error(f"✗ Config not found: {cfg_path}")
                state.experiments.append(
                    ExperimentStatus(
                        name=exp.name,
                        config_path=cfg_path,
                        role=exp.role,
                        status="failed",
                        error_message=f"Config not found: {cfg_path}",
                    )
                )
                continue

            stem = Path(cfg_path).stem
            manifest_arms.append((exp.name, exp.role, cfg_path, stem, campaign_dir / stem))

        if not manifest_arms:
            logger.warning(
                "No runnable arms for the array (all missing / filtered / "
                "already handled); nothing submitted."
            )
            state.save()
            return state

        # Stem collision → two arms would write the same <campaign_dir>/<stem>.
        # Fail loud rather than silently clobber one arm's outputs (#9/#15).
        stem_to_arm: dict[str, str] = {}
        for name, _role, _cfg, stem, _out in manifest_arms:
            if stem in stem_to_arm:
                raise ValueError(
                    f"Array campaign has two arms sharing the config file stem "
                    f"{stem!r} ({stem_to_arm[stem]!r} and {name!r}); they would "
                    f"write the same output dir <campaign_dir>/{stem}. Rename one "
                    f"config so stems are unique."
                )
            stem_to_arm[stem] = name

        # Freeze the manifest (one resolved config per line, array-index order).
        manifest_path = campaign_dir / "array_manifest.txt"
        manifest_path.write_text("\n".join(cfg for _n, _r, cfg, _s, _o in manifest_arms) + "\n")

        # Pre-create per-arm output dirs so _discover_results has somewhere to look.
        for _n, _r, _cfg, _s, out in manifest_arms:
            out.mkdir(parents=True, exist_ok=True)

        # Generate ONE array script and submit it ONCE.
        script = SLURMBackend.generate_array_job_script(
            manifest_path=str(manifest_path),
            n_tasks=len(manifest_arms),
            output_base=str(campaign_dir),
            base_dir=self.base_dir,
            concurrency=config.array_concurrency,
            dispatch_dir=str(campaign_dir / "dispatch"),
            slurm_params=self._resolve_slurm_params(config),
            resume=self.resume,
        )
        array_job_id = self.slurm.submit_job(script)
        logger.info(
            f"✓ Submitted array of {len(manifest_arms)} arms → Job {array_job_id} "
            f"(%{min(config.array_concurrency, len(manifest_arms))} concurrency)"
        )

        # Record each arm against the single array job id + its task index.
        submit_time = datetime.now(UTC).isoformat()
        for idx, (name, role, cfg_path, _stem, out) in enumerate(manifest_arms):
            state.experiments.append(
                ExperimentStatus(
                    name=name,
                    config_path=cfg_path,
                    role=role,
                    slurm_job_id=array_job_id,
                    array_task_id=idx,
                    status="submitted",
                    submit_time=submit_time,
                    output_dir=str(out),
                )
            )

        state.save()
        logger.info(f"\n  Submitted (array): {len(manifest_arms)}  |  Skipped (resume): {skipped}")
        logger.info(f"  Array job id: {array_job_id}")
        logger.info(f"  Manifest: {manifest_path}")
        logger.info(f"  State saved: {state.campaign_dir}/campaign_state.json")
        logger.info(f"  Monitor: python -m spectramr.cli campaign status {campaign_dir}")
        return state

    # ── Per-arm dispatch (slurm submit vs container run) ──────────

    def _submit_one_arm(
        self,
        *,
        name: str,
        cfg_path: str,
        exp_output: Path,
        slurm_params: dict[str, Any],
        test_manifest_path: str | None,
        role: str,
    ) -> ExperimentStatus:
        """Submit (SLURM) or run (container) a single arm via ``self.where``."""
        if self.where == "slurm":
            script = SLURMBackend.generate_job_script(
                experiment_name=name,
                config_path=cfg_path,
                output_dir=str(exp_output),
                base_dir=self.base_dir,
                slurm_params=slurm_params,
                resume=self.resume,
                test_manifest=test_manifest_path,
            )
            job_id = self.slurm.submit_job(script)
            logger.info(f"✓ Submitted {name} → Job {job_id}")
            return ExperimentStatus(
                name=name,
                config_path=cfg_path,
                role=role,
                slurm_job_id=job_id,
                status="submitted",
                submit_time=datetime.now(UTC).isoformat(),
                output_dir=str(exp_output),
            )
        return self._run_arm_in_container(
            name=name,
            cfg_path=cfg_path,
            exp_output=exp_output,
            slurm_params=slurm_params,
            role=role,
        )

    def _run_arm_in_container(
        self,
        *,
        name: str,
        cfg_path: str,
        exp_output: Path,
        slurm_params: dict[str, Any],
        role: str,
    ) -> ExperimentStatus:
        """Run one arm in a container (docker/apptainer), synchronously.

        The arm runs ``spectramr train --config <arm> -O training.output_dir=<dir>``
        inside the container so checkpoints/CSVs land in the campaign tree. Runs
        block until the container exits, so arms execute sequentially. The
        container backends are pure infra (``spectramr.infrastructure.execution``),
        so the orchestrator imports them without a layer violation.
        """
        from spectramr.infrastructure.execution import (
            ApptainerBackend,
            DockerBackend,
            SpectraMRInvocation,
            ResourceSpec,
        )

        backend = {"docker": DockerBackend, "apptainer": ApptainerBackend}[self.where]()
        extra = ["-O", f"training.output_dir={exp_output}"]
        if self.resume:
            extra += ["--resume", "auto"]
        invocation = SpectraMRInvocation(verb="train", config=cfg_path, extra_args=tuple(extra))
        resources = ResourceSpec(
            account=slurm_params.get("account"),
            gpus=slurm_params.get("gpus", 1),
        )
        handle = backend.run(invocation, resources, dry_run=self.dry_run)

        if self.dry_run:
            logger.info(
                "[DRY RUN] %s → %s: %s",
                name,
                self.where,
                " ".join(handle.command or []),
            )
            return ExperimentStatus(
                name=name,
                config_path=cfg_path,
                role=role,
                status="submitted",
                submit_time=datetime.now(UTC).isoformat(),
                output_dir=str(exp_output),
            )

        ok = handle.returncode in (0, None)
        logger.info(
            "%s %s via %s (exit=%s)",
            "✓" if ok else "✗",
            name,
            self.where,
            handle.returncode,
        )
        return ExperimentStatus(
            name=name,
            config_path=cfg_path,
            role=role,
            status="completed" if ok else "failed",
            submit_time=datetime.now(UTC).isoformat(),
            output_dir=str(exp_output),
            error_message=None if ok else f"container exited {handle.returncode}",
        )

    def _prior_handled(self, campaign_dir: Path) -> dict[str, ExperimentStatus]:
        """Map experiment-name → prior status for arms that must NOT be
        re-submitted on ``--resume``: those already ``completed`` or still
        ``submitted`` / ``running``. ``pending`` / ``failed`` / ``cancelled`` /
        ``timeout`` arms are re-submittable and excluded. Returns ``{}`` when not
        resuming, when no prior ``campaign_state.json`` exists, or when it cannot
        be parsed (fail-open to submitting, never silently skipping everything).

        NOTE: covers the parallel execution mode. Sequential stage-group resume
        (which must also reconstruct ``afterok`` dependency chains for the arms
        it skips) remains a follow-up — see
        ``TODO/infrastructure_audit_2026_06_12.md``.
        """
        if not self.resume:
            return {}
        state_file = Path(campaign_dir) / "campaign_state.json"
        if not state_file.exists():
            return {}
        try:
            prior = CampaignState.load(state_file)
        except Exception as e:
            # Fail-open to full submission — never silently skip everything.
            logger.warning(
                f"Resume: could not load prior campaign state ({e}); submitting all arms"
            )
            return {}
        handled = {
            e.name: e
            for e in prior.experiments
            if e.status in ("completed", "submitted", "running")
        }
        if handled:
            logger.info(f"Resume: {len(handled)} arm(s) already handled — they will be skipped")
        return handled

    # ── Progress Check ───────────────────────────────────────────

    # Consecutive UNKNOWN polls before a vanished job is declared terminal. >1
    # so a just-submitted job briefly absent from sacct/squeue isn't false-failed.
    _UNKNOWN_TERMINAL_POLLS = 2

    def _has_completion_artifact(self, exp: ExperimentStatus) -> bool:
        """A job that vanished from SLURM most likely COMPLETED (and aged out of
        sacct's retention) if it left a checkpoint; otherwise treat it as FAILED.
        Probing an artifact avoids both a false success and silently discarding a
        finished run's results."""
        if not exp.output_dir:
            return False
        ckpt_dir = Path(exp.output_dir) / "checkpoints"
        return ckpt_dir.is_dir() and any(ckpt_dir.glob("*.pt"))

    def check_progress(
        self,
        campaign_dir: str,
    ) -> CampaignState:
        """Poll SLURM and update campaign state.

        Args:
            campaign_dir: Path to campaign output directory.

        Returns:
            Updated CampaignState.
        """
        state_path = Path(campaign_dir) / "campaign_state.json"
        state = CampaignState.load(state_path)

        # Collect all non-terminal job IDs
        active_jobs = [
            exp.slurm_job_id
            for exp in state.experiments
            if exp.slurm_job_id is not None and not exp.is_terminal
        ]

        if not active_jobs:
            logger.info("All experiments are in terminal state.")
            logger.info("\n%s", state.summary_table())
            return state

        # Batch query SLURM
        statuses = self.slurm.query_batch_status(active_jobs)

        # Update experiment states
        for exp in state.experiments:
            if exp.slurm_job_id is None or exp.is_terminal:
                continue

            job_status = statuses.get(exp.slurm_job_id)
            if job_status is None:
                continue

            new_state = job_status.normalised_state

            # A job that has vanished from both sacct and squeue (UNKNOWN) is
            # terminal — not still "submitted". Leaving it non-terminal re-polls
            # it forever and hangs `campaign watch`. Cap consecutive UNKNOWN polls
            # (a just-submitted job is briefly absent before the scheduler
            # registers it), then resolve completed-vs-failed by probing for an
            # output artifact rather than guessing (no false success).
            if job_status.is_unknown:
                polls = int(getattr(exp, "unknown_polls", 0)) + 1
                exp.unknown_polls = polls
                if polls >= self._UNKNOWN_TERMINAL_POLLS:
                    new_state = "completed" if self._has_completion_artifact(exp) else "failed"
                    logger.warning(
                        f"  {exp.name}: job {exp.slurm_job_id} vanished from SLURM "
                        f"after {polls} polls → {new_state}"
                    )
                else:
                    new_state = exp.status  # transient — poll again next cycle
            else:
                exp.unknown_polls = 0

            if new_state != exp.status:
                logger.info(f"  {exp.name}: {exp.status} → {new_state}")
                exp.status = new_state

            if job_status.start_time:
                exp.start_time = job_status.start_time
            if job_status.end_time:
                exp.end_time = job_status.end_time
            if job_status.exit_code is not None:
                exp.exit_code = job_status.exit_code
            if job_status.elapsed:
                exp.wall_time_seconds = _parse_elapsed(job_status.elapsed)

            # If completed, discover checkpoint and metrics
            if exp.status == "completed" and exp.output_dir:
                self._discover_results(exp)

        state.save()
        logger.info("\n%s", state.summary_table())

        return state

    # ── Cancel ───────────────────────────────────────────────────

    def cancel_campaign(self, campaign_dir: str) -> CampaignState:
        """Cancel all active jobs in a campaign.

        Args:
            campaign_dir: Path to campaign output directory.

        Returns:
            Updated CampaignState.
        """
        state_path = Path(campaign_dir) / "campaign_state.json"
        state = CampaignState.load(state_path)

        active_jobs = [
            exp.slurm_job_id
            for exp in state.experiments
            if exp.slurm_job_id is not None and not exp.is_terminal
        ]

        if not active_jobs:
            logger.info("No active jobs to cancel.")
            return state

        self.slurm.cancel_batch(active_jobs)

        for exp in state.experiments:
            if not exp.is_terminal:
                exp.status = "cancelled"

        state.save()
        logger.info(f"Cancelled {len(active_jobs)} jobs.")
        return state

    # ── Internal helpers ─────────────────────────────────────────

    def _resolve_config_path(self, config: str) -> str:
        """Resolve a config path relative to base_dir."""
        p = Path(config)
        if p.is_absolute():
            return str(p)
        resolved = Path(self.base_dir) / config
        return str(resolved)

    def _generate_ablation_configs(
        self,
        config: CampaignConfigSchema,
        campaign_dir: Path,
    ) -> list[str]:
        """Generate ablation configs from axes specification.

        Uses the first baseline experiment as the base config.
        """
        baselines = config.baseline_experiments
        if not baselines:
            # Use the first experiment as base
            base_config = config.experiments[0].config
        else:
            base_config = baselines[0].config

        base_config = self._resolve_config_path(base_config)
        ablation_dir = campaign_dir / "ablation_configs"

        configs = AblationConfigGenerator.generate(
            base_config_path=base_config,
            axes=config.ablation_axes,
            output_dir=str(ablation_dir),
            mode="grid",
        )

        logger.info(
            f"Generated {len(configs)} ablation configs from {len(config.ablation_axes)} axes"
        )

        return configs

    # ── Sequential campaign submission ───────────────────────────

    def _submit_sequential(
        self,
        config: CampaignConfigSchema,
        state: CampaignState,
        campaign_dir: Path,
    ) -> CampaignState:
        """Submit a sequential campaign with SLURM dependency chaining.

        Stage groups are topologically sorted so that dependencies are
        submitted first.  Each group's experiments are submitted in
        parallel, but the group as a whole waits for all parent groups
        via ``--dependency=afterok``.

        Checkpoint injection: experiments with ``checkpoint_from`` get
        the parent experiment's deterministic checkpoint path injected
        as a ``-O`` config override.
        """
        ordered_groups = self._topological_sort(config.stage_groups)

        # Track job IDs per stage group for dependency wiring
        group_job_ids: dict[str, list[int]] = {}

        # Campaign-level resume (skip already-handled arms) + broken-group
        # tracking: if a parent group has ANY submission failure its afterok
        # prerequisite can't be satisfied, so dependent groups must be skipped
        # rather than launched with a silently-shortened dependency chain.
        handled = self._prior_handled(campaign_dir)
        broken_groups: set[str] = set()
        skipped = 0

        # Resolve test manifest once
        test_manifest_path = None
        if config.test_manifest:
            test_manifest_path = self._resolve_config_path(config.test_manifest)

        submitted = 0
        failed = 0

        for group in ordered_groups:
            logger.info(f"\n── Stage group: {group.name} ──")
            if group.depends_on:
                logger.info(f"   Depends on: {group.depends_on}")

            # Collect parent job IDs from all depends_on groups
            dependency_ids: list[int] = []
            for dep_name in group.depends_on:
                dep_jobs = group_job_ids.get(dep_name, [])
                dependency_ids.extend(dep_jobs)

            # Filter out dry-run zeros
            dependency_ids = [j for j in dependency_ids if j > 0]

            current_group_jobs: list[int] = []

            stage_experiments = self._apply_filters([e for e in group.experiments if e.enabled])

            # If any parent group broke, this group cannot satisfy its afterok
            # prerequisite — skip it (and cascade the brokenness) rather than
            # launch its arms without a prerequisite.
            if any(dep in broken_groups for dep in group.depends_on):
                broken_groups.add(group.name)
                group_job_ids[group.name] = []
                for exp in stage_experiments:
                    logger.error(
                        f"  ✗ Skipping {exp.name}: parent group in "
                        f"{group.depends_on} failed to submit"
                    )
                    state.experiments.append(
                        ExperimentStatus(
                            name=exp.name,
                            config_path=exp.config,
                            role=exp.role,
                            stage_group=group.name,
                            status="failed",
                            error_message=(
                                f"Skipped: a parent group in {group.depends_on} "
                                "failed to submit (broken afterok chain)"
                            ),
                        )
                    )
                    failed += 1
                continue

            group_had_failure = False
            for exp in stage_experiments:
                # Resume: carry a prior already-handled arm forward instead of
                # re-submitting. A still-live parent job (submitted/running) must
                # be waited on by children; a completed one needs no dependency.
                if exp.name in handled:
                    prior = handled[exp.name]
                    state.experiments.append(prior)
                    if prior.status in ("submitted", "running") and prior.slurm_job_id:
                        current_group_jobs.append(prior.slurm_job_id)
                    skipped += 1
                    logger.info(f"  ↻ Resume: skipping {exp.name} (already {prior.status})")
                    continue

                config_path = self._resolve_config_path(exp.config)
                if not Path(config_path).exists():
                    logger.error(f"  ✗ Config not found: {config_path}")
                    state.experiments.append(
                        ExperimentStatus(
                            name=exp.name,
                            config_path=config_path,
                            role=exp.role,
                            stage_group=group.name,
                            status="failed",
                            error_message=f"Config not found: {config_path}",
                        )
                    )
                    failed += 1
                    group_had_failure = True
                    continue

                # Create experiment output directory
                exp_output = campaign_dir / exp.name
                exp_output.mkdir(parents=True, exist_ok=True)

                slurm_params = self._resolve_slurm_params(
                    config, group.slurm_overrides, exp.slurm_overrides
                )

                # Resolve checkpoint_from → config overrides
                extra_overrides: dict[str, str] | None = None
                if exp.checkpoint_from:
                    ref = exp.checkpoint_from
                    parent_output = campaign_dir / ref.experiment
                    # Deterministic checkpoint path. 'best.pt' is a symlink
                    # to 'checkpoint_best.pt' published by CheckpointService /
                    # CheckpointDirector (_publish_best_alias). Both names
                    # resolve to the same checkpoint.
                    ckpt_path = parent_output / "checkpoints" / "best.pt"
                    extra_overrides = {ref.inject_as: str(ckpt_path)}
                    logger.info(f"  ⬆ {exp.name}: injecting {ref.inject_as}={ckpt_path}")

                # Generate and submit
                script = SLURMBackend.generate_job_script(
                    experiment_name=exp.name,
                    config_path=config_path,
                    output_dir=str(exp_output),
                    base_dir=self.base_dir,
                    slurm_params=slurm_params,
                    resume=self.resume,
                    test_manifest=test_manifest_path,
                    config_overrides=extra_overrides,
                )

                try:
                    job_id = self.slurm.submit_job(
                        script,
                        dependency_job_ids=dependency_ids or None,
                    )
                    exp_status = ExperimentStatus(
                        name=exp.name,
                        config_path=config_path,
                        role=exp.role,
                        stage_group=group.name,
                        slurm_job_id=job_id,
                        depends_on_jobs=dependency_ids,
                        status="submitted",
                        submit_time=datetime.now(UTC).isoformat(),
                        output_dir=str(exp_output),
                    )
                    state.experiments.append(exp_status)
                    current_group_jobs.append(job_id)
                    submitted += 1
                    dep_str = ""
                    if dependency_ids:
                        dep_str = f" (after jobs {dependency_ids})"
                    logger.info(f"  ✓ {exp.name} → Job {job_id}{dep_str}")

                except RuntimeError as e:
                    logger.error(f"  ✗ Failed to submit {exp.name}: {e}")
                    state.experiments.append(
                        ExperimentStatus(
                            name=exp.name,
                            config_path=config_path,
                            role=exp.role,
                            stage_group=group.name,
                            status="failed",
                            error_message=str(e),
                        )
                    )
                    failed += 1
                    group_had_failure = True

                time.sleep(0.3)

            group_job_ids[group.name] = current_group_jobs
            if group_had_failure:
                # A dependent group's afterok chain would be incomplete — mark
                # this group broken so dependents are skipped (above).
                broken_groups.add(group.name)

        # Save state
        state.save()

        logger.info(
            f"\n  Submitted: {submitted}  |  Failed: {failed}  |  Skipped (resume): {skipped}"
        )
        logger.info("  Execution mode: sequential")
        logger.info(f"  Stage order: {' → '.join(g.name for g in ordered_groups)}")
        logger.info(f"  State saved: {state.campaign_dir}/campaign_state.json")

        return state

    @staticmethod
    def _topological_sort(
        groups: list,
    ) -> list:
        """Topological sort of stage groups by depends_on edges.

        Returns:
            Ordered list of stage groups (dependencies first).

        Raises:
            ValueError: If a cycle is detected.
        """
        name_to_group = {g.name: g for g in groups}
        in_degree: dict[str, int] = {g.name: 0 for g in groups}
        dependents: dict[str, list[str]] = {g.name: [] for g in groups}

        for g in groups:
            for dep in g.depends_on:
                in_degree[g.name] += 1
                dependents[dep].append(g.name)

        # Kahn's algorithm
        queue = [name for name, deg in in_degree.items() if deg == 0]
        result: list = []

        while queue:
            node = queue.pop(0)
            result.append(name_to_group[node])
            for child in dependents[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(result) != len(groups):
            visited = {g.name for g in result}
            cycle_nodes = {g.name for g in groups} - visited
            raise ValueError(f"Cycle detected in stage group dependencies: {cycle_nodes}")

        return result

    def _discover_results(self, exp: ExperimentStatus) -> None:
        """Discover checkpoint and metrics files for a completed experiment."""
        output = Path(exp.output_dir) if exp.output_dir else None
        if output is None:
            return

        # Find checkpoint via the shared discovery helper, which knows the real
        # writer conventions (best.{pt,safetensors} alias, checkpoint_best.*,
        # checkpoint_step_*/epoch_*) rather than the dead ``model_iter_*.pt`` glob.
        from spectramr.infrastructure.services.checkpoint_service import (
            discover_best_checkpoint,
        )

        ckpt = discover_best_checkpoint(output / "checkpoints")
        if ckpt is not None:
            exp.checkpoint_path = str(ckpt)

        # Find metrics CSV. MetricsTracker writes
        # ``{training,validation}_metrics_<model_type>.csv`` at the run root (and
        # some writers put them under ``logs/``), so glob both conventions rather
        # than hard-coding the default ``model`` model_type.
        csv_path = self._discover_metrics_csv(output)
        if csv_path is not None:
            exp.metrics_csv_path = str(csv_path)

        # Now that the path is known, read the run's best metrics out of it.
        self._extract_best_metrics(exp)

    @staticmethod
    def _discover_metrics_csv(output: Path) -> Path | None:
        """Find a metrics CSV, preferring validation over training, tolerating
        the ``_<model_type>`` suffix and the optional ``logs/`` subdirectory."""
        for base in (output, output / "logs"):
            if not base.exists():
                continue
            # Prefer an exact validation file, then any suffixed validation file,
            # then training as a last resort.
            for pattern in (
                "validation_metrics.csv",
                "validation_metrics_*.csv",
                "training_metrics.csv",
                "training_metrics_*.csv",
            ):
                matches = sorted(base.glob(pattern))
                if matches:
                    return matches[0]
        return None

    @staticmethod
    def _extract_best_metrics(exp: Any) -> None:
        """Populate ``exp.best_metrics`` from the run's metrics CSV.

        This body used to sit after an unconditional ``return None`` in
        ``_discover_metrics_csv``, so it had never executed and the campaign
        report's ``Best PSNR`` column was permanently empty (#1343). Moving it
        into a method that is actually called is the whole fix; the extraction
        logic itself is unchanged.

        Args:
            exp: The experiment record, mutated in place.
        """
        if not exp.metrics_csv_path:
            return
        try:
            import pandas as pd

            df = pd.read_csv(exp.metrics_csv_path)
            for metric in ["psnr", "ssim", "lpips", "mse"]:
                cols = [c for c in df.columns if metric in c.lower()]
                if cols:
                    col = cols[0]
                    numeric = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(numeric) > 0:
                        if metric in ("lpips", "mse"):
                            exp.best_metrics[metric] = float(numeric.min())
                        else:
                            exp.best_metrics[metric] = float(numeric.max())
        except Exception as e:
            logger.debug(f"Could not parse metrics CSV: {e}")


def _parse_elapsed(elapsed: str) -> float | None:
    """Parse SLURM elapsed time string to seconds.

    Formats: 'HH:MM:SS', 'D-HH:MM:SS', 'MM:SS'.
    """
    try:
        parts = elapsed.split("-")
        days = 0
        if len(parts) == 2:
            days = int(parts[0])
            time_str = parts[1]
        else:
            time_str = parts[0]

        segments = time_str.split(":")
        if len(segments) == 3:
            h, m, s = int(segments[0]), int(segments[1]), int(segments[2])
        elif len(segments) == 2:
            h, m, s = 0, int(segments[0]), int(segments[1])
        else:
            return None

        return float(days * 86400 + h * 3600 + m * 60 + s)
    except Exception:
        return None
