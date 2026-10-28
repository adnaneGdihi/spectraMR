"""Campaign Configuration Schema.

Pydantic v2 models for defining experiment campaigns — collections of
related experiments (baseline + variants) grouped by research question.

A campaign automates: define → submit → track → evaluate → compare.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "AblationAxisSchema",
    "CampaignConfigSchema",
    "CampaignExperimentSchema",
    "CheckpointFromSchema",
    "EvaluationConfigSchema",
    "SlurmDefaultsSchema",
    "StageGroupSchema",
]


class SlurmDefaultsSchema(BaseModel):
    """Default SLURM parameters applied to all experiments in a campaign."""

    model_config = {"extra": "forbid", "frozen": True}

    # None by default: the job-script generator (slurm_backend.generate_job_script)
    # only emits a ``#SBATCH --partition`` directive when one is explicitly set,
    # because the target cluster has no dedicated "gpu" partition (GPUs are
    # requested via --gres). A default of "gpu" silently overrode that None for
    # every campaign job → ``sbatch: invalid partition specified`` (pitfall #15).
    partition: str | None = Field(default=None)
    time: str = Field(default="120:00:00")
    mem: str = Field(default="64GB")
    cpus_per_task: int = Field(default=8, gt=0)
    gpus: int = Field(default=1, ge=0)
    nodes: int = Field(default=1, gt=0)
    ntasks: int = Field(default=1, gt=0)
    mail_type: str = Field(default="END,FAIL")


class CheckpointFromSchema(BaseModel):
    """Cross-stage checkpoint injection.

    Used in sequential campaigns to wire a parent stage's best checkpoint, else its
    latest step, into a child experiment's config. The file is chosen when the child
    job starts, after the parent finished; a parent that left none fails the child.
    """

    model_config = {"extra": "forbid", "frozen": True}

    stage: str = Field(
        description="Name of the parent stage_group to take the checkpoint from",
    )
    experiment: str = Field(
        description="Name of the experiment within that stage group",
    )
    inject_as: str = Field(
        default="model.checkpoint_path",
        description=(
            "Dot-separated config path to override with the resolved checkpoint "
            "(e.g. 'model.checkpoint_path' or 'training.multi.stages.0.checkpoint_path')"
        ),
    )


class CampaignExperimentSchema(BaseModel):
    """A single experiment within a campaign."""

    model_config = {"extra": "forbid", "frozen": True}

    name: str = Field(description="Human-readable label (e.g., 'Cold Diffusion VD')")
    config: str = Field(description="Path to experiment YAML config")
    role: Literal["baseline", "variant", "ablation"] = Field(default="variant")
    tags: dict[str, str] = Field(default_factory=dict)
    slurm_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-experiment SLURM overrides (time, mem, gpus, etc.)",
    )
    enabled: bool = Field(default=True, description="Set False to skip this experiment")
    checkpoint_from: CheckpointFromSchema | None = Field(
        default=None,
        description=(
            "Inject a checkpoint from a prior stage group's experiment. "
            "Only used when campaign execution is 'sequential'."
        ),
    )

    @field_validator("config")
    @classmethod
    def validate_config_extension(cls, value: str) -> str:
        """Ensure config path ends with .yaml."""
        if not value.endswith((".yaml", ".yml")):
            raise ValueError(f"Config must be a YAML file, got: {value}")
        return value


class AblationAxisSchema(BaseModel):
    """Parameter sweep axis for ablation studies.

    Specifies a config path and the values to sweep over.
    Multiple axes create a grid (or random sample) of configs.
    """

    model_config = {"extra": "forbid", "frozen": True}

    config_path: str = Field(
        description="Dot-separated path in the YAML config "
        "(e.g., 'objectives.reconstruction.lambda_l1')"
    )
    values: list[Any] = Field(
        min_length=2,
        description="Values to sweep over (at least 2)",
    )
    label: str = Field(
        default="",
        description="Short label for plots (defaults to last segment of config_path)",
    )

    @model_validator(mode="before")
    @classmethod
    def _auto_label(cls, data: Any) -> Any:
        """Auto-generate label from config_path if not provided."""
        if isinstance(data, dict):
            if not data.get("label") and data.get("config_path"):
                data["label"] = data["config_path"].rsplit(".", maxsplit=1)[-1]
        return data


class EvaluationConfigSchema(BaseModel):
    """Configuration for post-training comparative evaluation."""

    model_config = {"extra": "forbid", "frozen": True}

    metrics: list[str] = Field(
        default=["psnr", "ssim", "lpips"],
        description="Metrics to compute during evaluation",
    )
    n_bootstrap: int = Field(default=10_000, gt=0)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    significance_level: float = Field(default=0.05, gt=0, lt=1)
    # PR-6 H4: accept Holm-Bonferroni + Benjamini-Hochberg aliases.
    correction_method: Literal[
        "bonferroni",
        "holm_bonferroni",
        "fdr",
        "fdr_bh",
        "benjamini_hochberg",
    ] = Field(default="fdr")
    compute_uncertainty: bool = Field(default=True)
    generate_plots: bool = Field(default=True)
    export_latex: bool = Field(default=True)
    per_sample_inference: bool = Field(
        default=True,
        description="Run inference on shared test set to get per-sample metrics",
    )
    batch_size: int = Field(default=4, gt=0, description="Batch size for inference")


class StageGroupSchema(BaseModel):
    """A named group of experiments that execute together.

    Used in sequential campaigns to define execution ordering.
    All experiments within a stage group are submitted in parallel.
    Stage groups execute sequentially based on ``depends_on`` edges.

    Example::

        stage_groups:
          - name: pretrain_vae
            experiments:
              - name: vae_base
                config: experiments/vae.yaml
                role: baseline
          - name: train_ldm
            depends_on: [pretrain_vae]
            experiments:
              - name: ldm_ddpm
                config: experiments/ldm.yaml
                checkpoint_from:
                  stage: pretrain_vae
                  experiment: vae_base
    """

    model_config = {"extra": "forbid", "frozen": True}

    name: str = Field(description="Unique stage group identifier")
    description: str = Field(default="")
    depends_on: list[str] = Field(
        default_factory=list,
        description="Names of stage groups that must complete before this one starts",
    )
    experiments: list[CampaignExperimentSchema] = Field(
        min_length=1,
        description="Experiments to run in this stage (submitted in parallel)",
    )
    slurm_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="SLURM overrides applied to all experiments in this group",
    )


class CampaignConfigSchema(BaseModel):
    """Top-level campaign definition.

    Groups experiments under a single research question with shared
    evaluation protocol and optional ablation sweeps.

    Supports two execution modes:

    - ``parallel`` (default): All experiments submitted at once, no ordering.
    - ``sequential``: Experiments grouped into ``stage_groups`` with
      dependency edges.  Each stage group is submitted only after its
      parents complete (SLURM ``--dependency=afterok``).
    """

    model_config = {"extra": "forbid", "frozen": True}

    name: str = Field(description="Campaign identifier (e.g., 'kspace_recon_shootout')")
    description: str = Field(default="", description="Human-readable description")
    task: str = Field(
        default="",
        description="Reference to task YAML (optional, for documentation)",
    )
    test_manifest: str = Field(
        default="",
        description="Path to shared test set manifest for evaluation",
    )

    # Execution mode
    execution: Literal["parallel", "sequential", "array"] = Field(
        default="parallel",
        description=(
            "'parallel' submits each arm as its own sbatch job; 'sequential' "
            "uses stage_groups with afterok dependencies; 'array' submits the "
            "whole flat 'experiments' list as ONE SLURM job array (a single "
            "tracked job id + '%N' concurrency throttle, one sbatch call)."
        ),
    )

    #: Max array tasks running at once in ``execution: array`` mode — the ``%N``
    #: throttle on ``#SBATCH --array=0-(M-1)%N``. Ignored by the parallel /
    #: sequential paths. Default 8 mirrors the standalone array dispatcher
    #: (``scripts/training/submit_experiment_array.sh`` CONCURRENCY default).
    array_concurrency: int = Field(default=8, gt=0)

    # Parallel mode: flat list of experiments
    experiments: list[CampaignExperimentSchema] = Field(
        default_factory=list,
        description="List of experiments (used in parallel mode)",
    )

    # Sequential mode: ordered stage groups
    stage_groups: list[StageGroupSchema] = Field(
        default_factory=list,
        description="Ordered stage groups with dependencies (used in sequential mode)",
    )

    ablation_axes: list[AblationAxisSchema] = Field(
        default_factory=list,
        description="Parameter sweep axes for ablation mode",
    )

    evaluation: EvaluationConfigSchema = Field(
        default_factory=EvaluationConfigSchema,
    )

    slurm_defaults: SlurmDefaultsSchema = Field(
        default_factory=SlurmDefaultsSchema,
    )

    output_dir: str = Field(
        default="",
        description="Override output directory (defaults to experiments/results/campaigns/<name>)",
    )

    @model_validator(mode="before")
    @classmethod
    def _auto_output_dir(cls, data: Any) -> Any:
        """Auto-generate output_dir from campaign name if not specified."""
        if isinstance(data, dict):
            if not data.get("output_dir") and data.get("name"):
                data["output_dir"] = f"experiments/results/campaigns/{data['name']}"
        return data

    @model_validator(mode="after")
    def _validate_execution_mode(self) -> CampaignConfigSchema:
        """Ensure correct fields are populated for the chosen execution mode."""
        if self.execution == "sequential":
            if not self.stage_groups:
                raise ValueError("Sequential campaigns require 'stage_groups' to be defined")
            if self.experiments:
                raise ValueError("Sequential campaigns must use 'stage_groups', not 'experiments'")
            # Validate depends_on references
            group_names = {g.name for g in self.stage_groups}
            for group in self.stage_groups:
                for dep in group.depends_on:
                    if dep not in group_names:
                        raise ValueError(
                            f"Stage group '{group.name}' depends on unknown "
                            f"group '{dep}'. Available: {group_names}"
                        )
            # Validate checkpoint_from references
            for group in self.stage_groups:
                for exp in group.experiments:
                    if exp.checkpoint_from:
                        ref = exp.checkpoint_from
                        if ref.stage not in group_names:
                            raise ValueError(
                                f"Experiment '{exp.name}' references unknown "
                                f"stage '{ref.stage}' in checkpoint_from"
                            )
                        # Verify the referenced experiment exists
                        ref_group = next(g for g in self.stage_groups if g.name == ref.stage)
                        ref_names = {e.name for e in ref_group.experiments}
                        if ref.experiment not in ref_names:
                            raise ValueError(
                                f"Experiment '{exp.name}' references unknown "
                                f"experiment '{ref.experiment}' in stage "
                                f"'{ref.stage}'. Available: {ref_names}"
                            )
        elif self.execution == "array":
            # 'array' fans the flat experiments list over a single SLURM job
            # array; stage_groups (sequential-only) are a config mistake here.
            if not self.experiments:
                raise ValueError("Array campaigns require 'experiments' to be defined")
            if self.stage_groups:
                raise ValueError(
                    "Array campaigns use the flat 'experiments' list, not "
                    "'stage_groups' (those belong to execution: sequential)"
                )
        else:
            if not self.experiments and not self.stage_groups:
                raise ValueError("Parallel campaigns require 'experiments' to be defined")
        return self

    @property
    def all_declared_experiments(self) -> list[CampaignExperimentSchema]:
        """Return all experiments from both modes (flat + stage groups)."""
        all_exps: list[CampaignExperimentSchema] = list(self.experiments)
        for group in self.stage_groups:
            all_exps.extend(group.experiments)
        return all_exps

    @property
    def enabled_experiments(self) -> list[CampaignExperimentSchema]:
        """Return only enabled experiments (from both modes)."""
        return [e for e in self.all_declared_experiments if e.enabled]

    @property
    def baseline_experiments(self) -> list[CampaignExperimentSchema]:
        """Return experiments marked as baseline."""
        return [e for e in self.enabled_experiments if e.role == "baseline"]

    @property
    def variant_experiments(self) -> list[CampaignExperimentSchema]:
        """Return experiments marked as variant."""
        return [e for e in self.enabled_experiments if e.role == "variant"]

    @classmethod
    def from_yaml(cls, path: str) -> CampaignConfigSchema:
        """Load campaign config from YAML file.

        Args:
            path: Path to campaign YAML file.

        Returns:
            Validated CampaignConfigSchema instance.

        Raises:
            FileNotFoundError: If path does not exist.
            ValidationError: If YAML content is invalid.
        """
        from pathlib import Path as _Path

        import yaml

        config_path = _Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Campaign config not found: {path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        return cls.model_validate(data)
