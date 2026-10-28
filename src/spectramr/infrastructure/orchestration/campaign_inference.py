"""``campaign infer``: grade every finished arm's best checkpoint on its held-out test split.

One SLURM array, one task per arm; each task is the arm's own ``train --validate-only`` on the
checkpoint ``checkpointing.select`` resolves (best, else latest), writing into
``<arm>/inference/``. Nothing here touches the arm's training-time ``test/`` or ``final_eval.json``.
An arm that cannot be graded (no checkpoint, config not found, ``inference/`` already there without
``force``) is listed with its reason and fails the command rather than shrinking the array silently.
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from spectramr.config.schemas.campaign import CampaignConfigSchema
from spectramr.core.dispatch_snapshot import write_snapshot
from spectramr.core.run_artifacts import INFERENCE_SUBDIR
from spectramr.infrastructure.checkpointing.select import best_then_latest, select
from spectramr.infrastructure.orchestration.campaign_state import CampaignState
from spectramr.infrastructure.orchestration.slurm_backend import SLURMBackend

logger = logging.getLogger(__name__)

MANIFEST = "inference_manifest.txt"
CHECKPOINTS = "inference_checkpoints.txt"
STATE = "inference_state.json"


@dataclass
class ArmPlan:
    name: str
    config: str | None
    run_dir: str
    checkpoint: str | None = None
    rule: str | None = None
    skip_reason: str | None = None
    #: The test manifest the training run's own pass scored (``test_metrics.json``), shown so the
    #: split being re-graded is visible; the task resolves its own from the arm's config.
    test_manifest: str | None = None
    #: A skip that is not a failure: the arm already has its result (rerun with ``force`` to redo).
    already_done: bool = False


def _resolve_config(recorded: str, base_dir: Path) -> str | None:
    """The recorded config path, else the same ``experiments/...`` tail under ``base_dir``."""
    if Path(recorded).is_file():
        return recorded
    tail = recorded.split("/experiments/", 1)
    local = base_dir / "experiments" / tail[1] if len(tail) == 2 else None
    return str(local) if local is not None and local.is_file() else None


def _checkpoint(run_dir: Path) -> tuple[Path | None, str | None]:
    """The weights the arm's own test pass scored (``test_metrics.json``), else best-then-latest.

    A run whose best weights were not restored scored its final ones, so grading ``best`` would
    compare two different networks.
    """
    scored = None
    with contextlib.suppress(OSError, ValueError, AttributeError):  # no summary: best-then-latest
        scored = json.loads((run_dir / "test_metrics.json").read_text()).get("weights")
    if scored == "final":
        latest = select(run_dir, "latest")
        return latest, "latest (the training test pass scored the final weights)"
    return best_then_latest(run_dir)


def _scored_manifest(run_dir: Path) -> str | None:
    try:
        manifest = json.loads((run_dir / "test_metrics.json").read_text()).get("manifest")
    except (OSError, ValueError, AttributeError):
        return None
    return str(manifest) if manifest else None


_NEVER_FINISHED = ("pending", "failed", "cancelled", "timeout", "conversion_pending")


def _unfinished(status: str, run_dir: Path) -> str | None:
    """Why the arm cannot be graded yet, or ``None``.

    ``campaign_state.json`` is only refreshed by ``campaign status``, so an arm that finished since
    the submit still reads ``submitted``/``running``; the run's own ``run_summary.json`` is the
    evidence that decides those two.
    """
    if status in _NEVER_FINISHED:
        return f"arm status is {status!r}, not completed"
    if status == "completed":
        return None
    try:
        summary = json.loads((run_dir / "run_summary.json").read_text())
    except (OSError, ValueError):
        return (
            f"arm status is {status!r} and {run_dir.name}/run_summary.json is missing or unreadable"
        )
    if summary.get("success") is True and not summary.get("wall_clock_yield"):
        return None
    return f"arm status is {status!r} and its run_summary.json does not report a finished run"


def plan_inference(
    campaign_dir: str | Path,
    base_dir: str | Path,
    *,
    only: list[str] | None = None,
    force: bool = False,
) -> list[ArmPlan]:
    """One plan per arm of ``campaign_state.json``; arms are located under ``campaign_dir``.

    The recorded ``output_dir`` is a cluster path, so only its last component is trusted: the arm's
    directory is ``<campaign_dir>/<that name>``, which also works on a downloaded tree.
    """
    campaign_dir, base = Path(campaign_dir), Path(base_dir)
    state = CampaignState.load(campaign_dir / "campaign_state.json")
    wanted = set(only or [])
    unknown = (
        wanted
        - {e.name for e in state.experiments}
        - {Path(e.output_dir or "").name for e in state.experiments}
    )
    if unknown:
        raise ValueError(f"--only names no arm of this campaign: {sorted(unknown)}")
    plans: list[ArmPlan] = []
    for exp in state.experiments:
        stem = Path(exp.output_dir or exp.name).name
        if wanted and not ({exp.name, stem} & wanted):
            continue
        run_dir = campaign_dir / stem
        plan = ArmPlan(exp.name, _resolve_config(exp.config_path, base), str(run_dir))
        plan.test_manifest = _scored_manifest(run_dir)
        plans.append(plan)
        if plan.config is None:
            plan.skip_reason = f"config not found: {exp.config_path}"
        elif (why := _unfinished(exp.status, run_dir)) is not None:
            plan.skip_reason = why
        elif plan.test_manifest is None:
            # Without a declared test set the task would run only the validation pass and write
            # no final_eval.json while reporting success.
            plan.skip_reason = f"{run_dir.name}/test_metrics.json is missing: the training run scored no test split"
        elif stem != Path(plan.config).stem:
            # The array writes <campaign>/<config stem>/inference, so the two names must agree.
            plan.skip_reason = (
                f"recorded output dir {stem!r} is not the config stem {Path(plan.config).stem!r}"
            )
        elif (run_dir / INFERENCE_SUBDIR / "final_eval.json").exists() and not force:
            plan.skip_reason = (
                f"{INFERENCE_SUBDIR}/ already holds a result; pass --force to redo it"
            )
            plan.already_done = True
        else:
            ckpt, rule = _checkpoint(run_dir)
            if ckpt is None:
                plan.skip_reason = f"no checkpoint under {run_dir}/checkpoints"
            else:
                plan.checkpoint, plan.rule = str(ckpt), rule
    return plans


def submit_inference(
    plans: list[ArmPlan],
    campaign_dir: str | Path,
    base_dir: str | Path,
    *,
    dry_run: bool = False,
    concurrency: int | None = None,
    slurm_overrides: dict[str, Any] | None = None,
) -> int | None:
    """Freeze the manifest + checkpoint list and submit ONE array; returns the job id (``None`` on dry run)."""
    campaign_dir, base = Path(campaign_dir), Path(base_dir).resolve()
    runnable = [p for p in plans if p.skip_reason is None]
    if not runnable:
        raise ValueError("no arm can be graded; nothing to submit")
    state = CampaignState.load(campaign_dir / "campaign_state.json")
    campaign_cfg = _campaign_config(state.campaign_config_path, base)
    params = campaign_cfg.slurm_defaults.model_dump()
    params.update(slurm_overrides or {})

    manifest_text = "\n".join(str(p.config) for p in runnable) + "\n"
    checkpoints_text = "\n".join(str(p.checkpoint) for p in runnable) + "\n"
    manifest, checkpoints = campaign_dir / MANIFEST, campaign_dir / CHECKPOINTS
    snapshot_text = None
    if not dry_run:  # a dry run writes nothing
        manifest.write_text(manifest_text)
        checkpoints.write_text(checkpoints_text)
        snapshot_text = write_snapshot(manifest, str(base)).read_text()

    script = SLURMBackend.generate_array_job_script(
        manifest_path=str(manifest),
        n_tasks=len(runnable),
        output_base=str(campaign_dir),
        base_dir=str(base),
        concurrency=concurrency or campaign_cfg.array_concurrency,
        dispatch_dir=str(campaign_dir / "dispatch_inference"),
        slurm_params=params,
        manifest_text=manifest_text,
        snapshot_text=snapshot_text,
        inference_checkpoints=str(checkpoints),
        inference_checkpoints_text=checkpoints_text,
    )
    job_id = SLURMBackend(dry_run=dry_run).submit_job(script)
    record = {
        "job_id": None if dry_run else job_id,
        "arms": {p.name: {**asdict(p), "task": i} for i, p in enumerate(runnable)},
    }
    if not dry_run:
        (campaign_dir / STATE).write_text(json.dumps(record, indent=2))
    return None if dry_run else job_id


def _campaign_config(recorded: str, base: Path) -> CampaignConfigSchema:
    path = Path(recorded)
    if not path.is_file():
        path = base / recorded
    if not path.is_file():
        raise FileNotFoundError(
            f"campaign config {recorded!r} not found (also tried under {base}); the array takes "
            "its SLURM defaults from it, so it is not guessed."
        )
    return CampaignConfigSchema.from_yaml(str(path))


__all__ = ["ArmPlan", "plan_inference", "submit_inference"]
