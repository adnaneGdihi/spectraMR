"""Campaign State — Persistent tracking of campaign progress.

Serialises campaign state to JSON so that progress survives across
``campaign status`` / ``campaign evaluate`` invocations.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = ["CampaignState", "ExperimentStatus"]

# ── Status type ──────────────────────────────────────────────────────

StatusType = Literal[
    "pending",
    "submitted",
    "running",
    "completed",
    "failed",
    "cancelled",
    "timeout",
    # The job ended, but its run is mid-conversion; polled again until converted.
    "conversion_pending",
]


# ── Per-experiment status ────────────────────────────────────────────


class ExperimentStatus(BaseModel):
    """Status of a single experiment within a campaign."""

    model_config = {"extra": "allow"}

    name: str
    config_path: str
    role: str = "variant"
    slurm_job_id: int | None = None
    #: Array-task index within ``slurm_job_id`` for ``execution: array``
    #: campaigns, where every arm shares ONE array job id and is distinguished
    #: by its task index (the sacct element is ``<slurm_job_id>_<array_task_id>``).
    #: ``None`` for parallel / sequential arms (each has its own job id).
    array_task_id: int | None = None
    status: StatusType = "pending"
    submit_time: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    wall_time_seconds: float | None = None
    exit_code: int | None = None

    # Result paths
    output_dir: str | None = None
    checkpoint_path: str | None = None
    metrics_csv_path: str | None = None
    #: ``validation_batches*.csv`` files (rank 0 first): one row per slice per
    #: pass, the only validation source the evaluator pairs across arms (#2478).
    slice_rows_paths: list[str] = Field(default_factory=list)

    # Sequential campaign support
    stage_group: str | None = Field(
        default=None,
        description="Stage group this experiment belongs to (sequential campaigns)",
    )
    depends_on_jobs: list[int] = Field(
        default_factory=list,
        description="SLURM job IDs this experiment depends on (afterok)",
    )

    # Aggregated best metrics (populated after training)
    best_metrics: dict[str, float] = Field(default_factory=dict)

    # Per-sample metric arrays (populated during evaluation)
    per_sample_metrics_path: str | None = None

    # Freeform tags
    tags: dict[str, str] = Field(default_factory=dict)
    error_message: str | None = None

    @property
    def is_terminal(self) -> bool:
        """Return True if the experiment is in a terminal state."""
        return self.status in ("completed", "failed", "cancelled", "timeout")

    @property
    def is_evaluable(self) -> bool:
        """Return True if the experiment completed and has a checkpoint."""
        return self.status == "completed" and self.checkpoint_path is not None


# ── Campaign-level state ─────────────────────────────────────────────


class CampaignState(BaseModel):
    """Persistent state of an entire campaign.

    Saved as ``campaign_state.json`` in the campaign output directory.
    """

    model_config = {"extra": "allow"}

    campaign_name: str
    campaign_config_path: str = ""
    campaign_dir: str
    description: str = ""
    test_manifest: str = ""
    # How this submission treated existing checkpoints: "resume", "fresh" or
    # "default" (array arms chain under --prod, other arms start fresh). Empty
    # on states written before the field existed.
    resume_mode: Literal["", "resume", "fresh", "default"] = ""
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    experiments: list[ExperimentStatus] = Field(default_factory=list)

    # ── Convenience properties ───────────────────────────────────

    @property
    def total(self) -> int:
        return len(self.experiments)

    @property
    def pending(self) -> list[ExperimentStatus]:
        return [e for e in self.experiments if e.status == "pending"]

    @property
    def submitted(self) -> list[ExperimentStatus]:
        return [e for e in self.experiments if e.status == "submitted"]

    @property
    def running(self) -> list[ExperimentStatus]:
        return [e for e in self.experiments if e.status == "running"]

    @property
    def completed(self) -> list[ExperimentStatus]:
        return [e for e in self.experiments if e.status == "completed"]

    @property
    def failed(self) -> list[ExperimentStatus]:
        return [e for e in self.experiments if e.status in ("failed", "timeout")]

    @property
    def evaluable(self) -> list[ExperimentStatus]:
        return [e for e in self.experiments if e.is_evaluable]

    @property
    def all_terminal(self) -> bool:
        return all(e.is_terminal for e in self.experiments)

    @property
    def progress_fraction(self) -> float:
        if not self.experiments:
            return 0.0
        terminal = sum(1 for e in self.experiments if e.is_terminal)
        return terminal / len(self.experiments)

    # ── Lookup ───────────────────────────────────────────────────

    def get_experiment(self, name: str) -> ExperimentStatus | None:
        """Find experiment by name."""
        for exp in self.experiments:
            if exp.name == name:
                return exp
        return None

    def get_by_job_id(self, job_id: int) -> ExperimentStatus | None:
        """Find experiment by SLURM job ID."""
        for exp in self.experiments:
            if exp.slurm_job_id == job_id:
                return exp
        return None

    def get_by_stage_group(self, group_name: str) -> list[ExperimentStatus]:
        """Return all experiments belonging to a stage group."""
        return [e for e in self.experiments if e.stage_group == group_name]

    # ── Mutation ──────────────────────────────────────────────────

    def update_experiment(self, name: str, **fields: Any) -> None:
        """Update fields of an experiment by name.

        Args:
            name: Experiment name.
            **fields: Fields to update.
        """
        exp = self.get_experiment(name)
        if exp is None:
            raise KeyError(f"Experiment '{name}' not found in campaign")
        for key, value in fields.items():
            setattr(exp, key, value)
        self.last_updated = datetime.now(UTC).isoformat()

    # ── Persistence ──────────────────────────────────────────────

    def save(self, path: str | Path | None = None) -> Path:
        """Save state to JSON.

        Args:
            path: File path. Defaults to ``<campaign_dir>/campaign_state.json``.

        Returns:
            Path to saved file.
        """
        if path is None:
            path = Path(self.campaign_dir) / "campaign_state.json"
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        self.last_updated = datetime.now(UTC).isoformat()

        # Atomic write: serialise to a temp file then os.replace. Writing
        # ``open(path, "w")`` directly truncates the target *before* dumping, so a
        # crash mid-serialisation (disk full, OOM) would leave an empty/partial
        # file — losing the campaign's entire job-id/status history.
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(self.model_dump(), f, indent=2, default=str)
            os.replace(tmp, path)  # atomic on POSIX; clobbers the old file
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

        logger.debug(f"Campaign state saved to {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> CampaignState:
        """Load state from JSON.

        Args:
            path: Path to campaign_state.json.

        Returns:
            CampaignState instance.

        Raises:
            FileNotFoundError: If path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Campaign state not found: {path}")

        with open(path) as f:
            data = json.load(f)

        return cls.model_validate(data)

    # ── Display ──────────────────────────────────────────────────

    def summary_table(self) -> str:
        """Return a formatted summary table of experiment statuses."""
        lines = [
            f"Campaign: {self.campaign_name}",
            f"Progress: {self.progress_fraction:.0%} "
            f"({len(self.completed)}/{self.total} completed, "
            f"{len(self.failed)} failed, "
            f"{len(self.running)} running, "
            f"{len(self.pending)} pending)",
            "",
            f"{'Name':<40} {'Role':<10} {'Status':<12} {'Job ID':<10} {'Best PSNR':<10}",
            "─" * 90,
        ]

        for exp in self.experiments:
            psnr = exp.best_metrics.get("psnr", "")
            psnr_str = f"{psnr:.2f}" if isinstance(psnr, float) else str(psnr)
            job_str = str(exp.slurm_job_id) if exp.slurm_job_id else "—"

            # Status with emoji
            status_icons = {
                "pending": "⏳",
                "submitted": "📤",
                "running": "🔄",
                "completed": "✅",
                "failed": "❌",
                "cancelled": "🚫",
                "timeout": "⏰",
                "conversion_pending": "⏸",
            }
            icon = status_icons.get(exp.status, "?")
            status_str = f"{icon} {exp.status}"

            lines.append(
                f"{exp.name:<40} {exp.role:<10} {status_str:<12} {job_str:<10} {psnr_str:<10}"
            )

        held = [e for e in self.experiments if e.status == "conversion_pending"]
        if held:
            lines += ["", "Held until their checkpoint conversion finishes:"]
            lines += [f"  {e.name}: {e.error_message}" for e in held]
        return "\n".join(lines)
