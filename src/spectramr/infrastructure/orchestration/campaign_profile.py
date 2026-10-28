"""``campaign profile``: profile every arm of a campaign into ``<campaign>/<arm>/profile/``.

One SLURM array, one task per arm; each task is ``spectramr profile --config <arm> --output-dir
<campaign>/<arm>``, so the profile lands in the directory the arm's training run writes (and is
skipped by every reader that walks it, ``run_layout.is_profiling_artifact``). It reads the campaign
YAML rather than ``campaign_state.json``: profiling is a pre-launch check and needs no prior submit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spectramr.config.schemas.campaign import CampaignConfigSchema, CampaignExperimentSchema
from spectramr.core.dispatch_snapshot import write_snapshot
from spectramr.infrastructure.orchestration.slurm_backend import SLURMBackend

logger = logging.getLogger(__name__)

MANIFEST = "profile_manifest.txt"
RUN_DIRS = "profile_run_dirs.txt"
ARGS = "profile_args.txt"

#: The preset of ``scripts/profiling/submit_profile_array.sh``: Scalene's GPU sampling ran
#: 12-25x slower and OOM-killed the host, so an array samples the CPU only (the run still uses
#: the accelerator), and ``--single-process`` lets a ddp/fsdp arm be profiled as one rank.
DEFAULT_PROFILE_ARGS: tuple[str, ...] = (
    "--mode",
    "cpu-only",
    "--val-batches",
    "4",
    "--single-process",
)
#: The profiled child is capped at 30 min, so the campaign's training-length allocation is not used.
PROFILE_SLURM: dict[str, Any] = {"time": "00:45:00", "mem": "64GB"}


@dataclass
class ArmProfile:
    name: str
    config: str
    run_dir: str
    skip_reason: str | None = None


def arm_run_dir(
    campaign_dir: Path, execution: str, exp: CampaignExperimentSchema, config: str
) -> Path:
    """The directory the arm's training run writes: ``<stem>`` for an array, ``<name>`` otherwise."""
    return campaign_dir / (Path(config).stem if execution == "array" else exp.name)


def _resolve(config: str, base: Path) -> str:
    return config if Path(config).is_absolute() else str(base / config)


def plan_profile(
    campaign_config: str | Path, base_dir: str | Path, *, only: list[str] | None = None
) -> tuple[CampaignConfigSchema, Path, list[ArmProfile]]:
    """The campaign, its directory and one plan per enabled arm (``only`` by name or run-dir name)."""
    cfg = CampaignConfigSchema.from_yaml(str(campaign_config))
    base = Path(base_dir).resolve()
    campaign_dir = base / cfg.output_dir
    plans: list[ArmProfile] = []
    for exp in cfg.enabled_experiments:
        config = _resolve(exp.config, base)
        run_dir = arm_run_dir(campaign_dir, cfg.execution, exp, config)
        plan = ArmProfile(exp.name, config, str(run_dir))
        if not Path(config).is_file():
            plan.skip_reason = f"config not found: {config}"
        plans.append(plan)
    wanted = set(only or [])
    unknown = wanted - {p.name for p in plans} - {Path(p.run_dir).name for p in plans}
    if unknown:
        raise ValueError(f"--only names no arm of this campaign: {sorted(unknown)}")
    if wanted:
        plans = [p for p in plans if {p.name, Path(p.run_dir).name} & wanted]
    dirs = [p.run_dir for p in plans]
    if len(set(dirs)) != len(dirs):
        raise ValueError("two arms resolve to the same run dir; their profiles would interleave")
    if cfg.ablation_axes:
        logger.warning(
            "ablation_axes are generated at submit time and are not profiled; profile a "
            "generated config directly with `spectramr profile`."
        )
    return cfg, campaign_dir, plans


def submit_profile(
    cfg: CampaignConfigSchema,
    campaign_dir: Path,
    plans: list[ArmProfile],
    base_dir: str | Path,
    *,
    profile_args: list[str],
    dry_run: bool = False,
    concurrency: int | None = None,
    slurm_overrides: dict[str, Any] | None = None,
) -> tuple[int | None, str]:
    """Freeze the manifest, run dirs and profile flags and submit ONE array: (job id, script).

    A dry run writes and submits nothing; its job id is ``None``.
    """
    base = Path(base_dir).resolve()
    runnable = [p for p in plans if p.skip_reason is None]
    if not runnable:
        raise ValueError("no arm can be profiled; nothing to submit")
    params = cfg.slurm_defaults.model_dump()
    params.update(PROFILE_SLURM)
    params.update(slurm_overrides or {})

    manifest_text = "".join(f"{p.config}\n" for p in runnable)
    run_dirs_text = "".join(f"{p.run_dir}\n" for p in runnable)
    args_text = "".join(f"{a}\n" for a in profile_args)
    manifest, run_dirs, args = campaign_dir / MANIFEST, campaign_dir / RUN_DIRS, campaign_dir / ARGS
    snapshot_text = None
    if not dry_run:  # a dry run writes nothing
        campaign_dir.mkdir(parents=True, exist_ok=True)
        manifest.write_text(manifest_text)
        run_dirs.write_text(run_dirs_text)
        args.write_text(args_text)
        snapshot_text = write_snapshot(manifest, str(base)).read_text()

    script = SLURMBackend.generate_array_job_script(
        manifest_path=str(manifest),
        n_tasks=len(runnable),
        output_base=str(campaign_dir),
        base_dir=str(base),
        concurrency=concurrency or cfg.array_concurrency,
        dispatch_dir=str(campaign_dir / "dispatch_profile"),
        slurm_params=params,
        manifest_text=manifest_text,
        snapshot_text=snapshot_text,
        profile_run_dirs=str(run_dirs),
        profile_run_dirs_text=run_dirs_text,
        profile_args=str(args) if profile_args else None,
        profile_args_text=args_text if profile_args else None,
    )
    if dry_run:
        return None, script
    return SLURMBackend(dry_run=False).submit_job(script), script


__all__ = [
    "DEFAULT_PROFILE_ARGS",
    "PROFILE_SLURM",
    "ArmProfile",
    "arm_run_dir",
    "plan_profile",
    "submit_profile",
]
