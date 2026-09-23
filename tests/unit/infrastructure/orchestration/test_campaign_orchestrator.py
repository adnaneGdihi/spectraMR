"""Unit tests for :class:`CampaignOrchestrator` lifecycle methods.

The orchestrator's filter logic (``_selector_matches`` /
``_apply_filters``) is exhaustively covered in
:mod:`tests.unit.infrastructure.orchestration.test_campaign_filters`. The
topological-sort and stage-group sequencing live in
:mod:`tests.unit.infrastructure.orchestration.test_campaign_dag`.

This file fills the gap — the **lifecycle entry points** that the CLI
calls through:

* ``__init__`` stores the constructor flags and instantiates a
  dry-run ``SLURMBackend``.
* ``_resolve_config_path`` accepts absolute and relative inputs.
* ``submit_campaign`` (parallel mode, dry-run) returns a populated
  ``CampaignState`` and persists ``campaign_state.json`` next to the
  campaign output dir.
* ``submit_campaign`` records a "config not found" failure as an
  ``ExperimentStatus`` instead of crashing the whole campaign.
* ``check_progress`` short-circuits when every job is already in a
  terminal state (no SLURM polling).
* ``cancel_campaign`` short-circuits when there are no active jobs.
* ``_generate_ablation_configs`` proxies into
  :class:`AblationConfigGenerator` with the campaign's first
  baseline as the base config.

Everything runs under ``dry_run=True`` so no real ``sbatch`` shell-out
is attempted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from spectramr.config.schemas.campaign import (
    AblationAxisSchema,
    CampaignConfigSchema,
    StageGroupSchema,
)
from spectramr.infrastructure.orchestration.slurm_backend import JobStatus, SLURMBackend
from spectramr.infrastructure.orchestration.campaign_orchestrator import (
    CampaignOrchestrator,
)
from spectramr.infrastructure.orchestration.campaign_state import (
    CampaignState,
    ExperimentStatus,
)


# ── Helpers ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _slurm_account(monkeypatch):
    """Supply the SLURM allocation these tests used to inherit from a literal.

    ``ResourceSpec`` ships no account default -- an allocation name is
    site-specific, so the tree carries none (#1146). Tests that render SLURM
    directives must therefore configure one, exactly as a user does.
    """
    monkeypatch.setenv("SPECTRAMR_SLURM_ACCOUNT", "test_alloc")
    monkeypatch.delenv("SPECTRAMR_SLURM_MAIL_USER", raising=False)


def _write_min_experiment_yaml(path: Path) -> None:
    """Minimal experiment YAML — orchestrator only checks the file
    exists; it does not validate its contents until SLURM submission
    time, which the dry-run backend skips."""
    path.write_text(
        "config_version: '6.0'\n"
        "experiment:\n"
        "  name: stub\n"
    )


def _write_campaign_yaml(
    path: Path,
    *,
    name: str = "test_campaign",
    experiment_configs: list[Path] | None = None,
    output_dir: str | None = None,
    ablation_axes: list[dict] | None = None,
    execution: str = "parallel",
    array_concurrency: int | None = None,
    experiment_names: list[str] | None = None,
) -> None:
    data: dict = {"name": name, "execution": execution}
    if array_concurrency is not None:
        data["array_concurrency"] = array_concurrency
    if output_dir is not None:
        data["output_dir"] = output_dir
    if experiment_configs is not None:
        data["experiments"] = [
            {
                "name": (
                    experiment_names[i]
                    if experiment_names is not None
                    else f"arm_{i}"
                ),
                "config": str(cfg),
                "role": "variant",
            }
            for i, cfg in enumerate(experiment_configs)
        ]
    if ablation_axes is not None:
        data["ablation_axes"] = ablation_axes
    path.write_text(yaml.safe_dump(data, sort_keys=False))


# ── __init__ ────────────────────────────────────────────────────────


class TestConstruction:
    def test_stores_constructor_flags(self) -> None:
        o = CampaignOrchestrator(
            base_dir="/tmp",
            dry_run=True,
            resume=True,
            only=["a", "b"],
            include=["role=ablation"],
            exclude=["tag.arch=mamba"],
        )
        # base_dir is resolved to absolute.
        assert Path(o.base_dir).is_absolute()
        assert o.dry_run is True
        assert o.resume is True
        assert o.only == ["a", "b"]
        assert o.include == ["role=ablation"]
        assert o.exclude == ["tag.arch=mamba"]
        # SLURM backend honours the dry_run flag.
        assert o.slurm.dry_run is True

    def test_default_kwargs_are_none_safe(self) -> None:
        # only/include/exclude default to empty lists, not None.
        o = CampaignOrchestrator(base_dir=".", dry_run=True)
        assert o.only == []
        assert o.include == []
        assert o.exclude == []


# ── _resolve_config_path ───────────────────────────────────────────


class TestResolveConfigPath:
    def test_absolute_path_returned_unchanged(self, tmp_path: Path) -> None:
        o = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True)
        abs_path = "/etc/passwd"  # any absolute path; the orchestrator
                                  # only stringifies it.
        out = o._resolve_config_path(abs_path)
        assert Path(out).is_absolute()
        assert out == abs_path

    def test_relative_path_anchored_at_base_dir(self, tmp_path: Path) -> None:
        o = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True)
        out = o._resolve_config_path("experiments/foo.yaml")
        assert Path(out).is_absolute()
        # Drops down into base_dir/experiments/foo.yaml.
        assert out.endswith("experiments/foo.yaml")
        assert str(tmp_path) in out


# ── submit_campaign — parallel mode, dry run ───────────────────────


class TestSubmitCampaignParallel:
    def test_empty_experiments_rejected_by_schema(self, tmp_path: Path) -> None:
        """A parallel campaign with no experiments fails schema validation,
        not orchestrator logic — exercising the loud-fail at the SSOT."""
        cfg = tmp_path / "campaign.yaml"
        _write_campaign_yaml(cfg, experiment_configs=[], output_dir=str(tmp_path / "out"))
        o = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True)
        # Pydantic ValidationError surfaces from CampaignConfigSchema.
        with pytest.raises(Exception, match="Parallel campaigns require"):
            o.submit_campaign(str(cfg))

    def test_single_experiment_dry_run_records_state(self, tmp_path: Path) -> None:
        exp_yaml = tmp_path / "exp.yaml"
        _write_min_experiment_yaml(exp_yaml)
        cfg = tmp_path / "campaign.yaml"
        out_dir = tmp_path / "campaign_out"
        _write_campaign_yaml(
            cfg, experiment_configs=[exp_yaml], output_dir=str(out_dir)
        )

        o = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True)
        state = o.submit_campaign(str(cfg))

        # State populated with one submitted arm.
        assert isinstance(state, CampaignState)
        assert state.campaign_name == "test_campaign"
        # State directory matches the configured output_dir.
        assert state.campaign_dir.endswith("campaign_out")
        assert len(state.experiments) == 1
        exp = state.experiments[0]
        assert exp.name == "arm_0"
        # Dry-run SLURMBackend returns job_id=0 as the sentinel.
        assert exp.slurm_job_id == 0
        assert exp.status == "submitted"
        # State JSON persisted next to the campaign dir.
        assert (out_dir / "campaign_state.json").exists()

    def test_missing_experiment_config_recorded_as_failed(
        self, tmp_path: Path
    ) -> None:
        cfg = tmp_path / "campaign.yaml"
        out_dir = tmp_path / "campaign_out"
        ghost = tmp_path / "ghost.yaml"  # never written
        _write_campaign_yaml(
            cfg, experiment_configs=[ghost], output_dir=str(out_dir)
        )

        o = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True)
        state = o.submit_campaign(str(cfg))

        # Failure is recorded *per-experiment*, not raised as an exception.
        assert len(state.experiments) == 1
        exp = state.experiments[0]
        assert exp.status == "failed"
        assert exp.slurm_job_id is None
        assert exp.error_message is not None
        assert "Config not found" in exp.error_message

    def test_filter_only_restricts_submitted_arms(self, tmp_path: Path) -> None:
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        for p in (a, b):
            _write_min_experiment_yaml(p)
        cfg = tmp_path / "campaign.yaml"
        out_dir = tmp_path / "out"
        _write_campaign_yaml(
            cfg, experiment_configs=[a, b], output_dir=str(out_dir)
        )

        o = CampaignOrchestrator(
            base_dir=str(tmp_path), dry_run=True, only=["arm_1"]
        )
        state = o.submit_campaign(str(cfg))
        # only arm_1 went through the submitter.
        names = [e.name for e in state.experiments]
        assert names == ["arm_1"]


# ── submit_campaign — array mode (ONE SLURM array job) ─────────────


class TestSubmitCampaignArray:
    """``execution: array`` submits the whole cohort as a single SLURM job
    array: one ``sbatch`` call → one tracked array job id, with a per-arm task
    index. This is the fix for the parallel mode's N-serial-sbatch login-node
    stall."""

    def _three_arm_campaign(self, tmp_path: Path):
        configs = []
        for stem in ("alpha", "beta", "gamma"):
            p = tmp_path / f"{stem}.yaml"
            _write_min_experiment_yaml(p)
            configs.append(p)
        cfg = tmp_path / "campaign.yaml"
        out_dir = tmp_path / "camp_out"
        _write_campaign_yaml(
            cfg,
            experiment_configs=configs,
            output_dir=str(out_dir),
            execution="array",
        )
        return cfg, out_dir, configs

    def test_array_submits_one_job_and_records_task_ids(self, tmp_path: Path) -> None:
        cfg, out_dir, _configs = self._three_arm_campaign(tmp_path)

        o = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True)
        submitted_scripts: list[str] = []
        o.slurm.submit_job = lambda script, **kw: (  # type: ignore[method-assign]
            submitted_scripts.append(script) or 4242
        )

        state = o.submit_campaign(str(cfg))

        # ONE submission for the whole cohort (not three).
        assert len(submitted_scripts) == 1
        # All three arms recorded, sharing the single array job id.
        assert len(state.experiments) == 3
        assert {e.slurm_job_id for e in state.experiments} == {4242}
        # Distinct, contiguous task indices.
        assert sorted(e.array_task_id for e in state.experiments) == [0, 1, 2]
        # Each arm's output_dir is <campaign_dir>/<config-stem>.
        for exp in state.experiments:
            stem = Path(exp.config_path).stem
            assert exp.output_dir == str(out_dir / stem)
            assert exp.status == "submitted"
        # The submitted script is the array script.
        assert "#SBATCH --array=0-2%" in submitted_scripts[0]
        assert "manifest_dispatch" in submitted_scripts[0]
        # State persisted.
        assert (out_dir / "campaign_state.json").exists()

    def test_array_writes_frozen_manifest_in_order(self, tmp_path: Path) -> None:
        cfg, out_dir, _configs = self._three_arm_campaign(tmp_path)
        o = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True)
        o.slurm.submit_job = lambda script, **kw: 1  # type: ignore[method-assign]
        o.submit_campaign(str(cfg))

        manifest = out_dir / "array_manifest.txt"
        assert manifest.exists()
        lines = [ln for ln in manifest.read_text().splitlines() if ln.strip()]
        # One line per arm, in declaration order, resolved to absolute paths.
        assert len(lines) == 3
        assert [Path(ln).stem for ln in lines] == ["alpha", "beta", "gamma"]

    def test_array_concurrency_flows_into_directive(self, tmp_path: Path) -> None:
        configs = [tmp_path / f"{s}.yaml" for s in ("a", "b", "c", "d")]
        for p in configs:
            _write_min_experiment_yaml(p)
        cfg = tmp_path / "campaign.yaml"
        out_dir = tmp_path / "out"
        _write_campaign_yaml(
            cfg, experiment_configs=configs, output_dir=str(out_dir),
            execution="array", array_concurrency=2,
        )
        o = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True)
        scripts: list[str] = []
        o.slurm.submit_job = lambda script, **kw: scripts.append(script) or 1  # type: ignore[method-assign]
        o.submit_campaign(str(cfg))
        assert "#SBATCH --array=0-3%2" in scripts[0]

    def test_array_missing_config_excluded_and_recorded_failed(
        self, tmp_path: Path
    ) -> None:
        good = tmp_path / "good.yaml"
        _write_min_experiment_yaml(good)
        ghost = tmp_path / "ghost.yaml"  # never written
        cfg = tmp_path / "campaign.yaml"
        out_dir = tmp_path / "out"
        _write_campaign_yaml(
            cfg, experiment_configs=[good, ghost], output_dir=str(out_dir),
            execution="array",
        )
        o = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True)
        o.slurm.submit_job = lambda script, **kw: 7  # type: ignore[method-assign]
        state = o.submit_campaign(str(cfg))

        by_name = {e.name: e for e in state.experiments}
        # The ghost is recorded failed, NOT in the array.
        assert by_name["arm_1"].status == "failed"
        assert by_name["arm_1"].array_task_id is None
        # The good arm got the only array slot (task 0 of a 1-task array).
        assert by_name["arm_0"].status == "submitted"
        assert by_name["arm_0"].array_task_id == 0
        manifest = out_dir / "array_manifest.txt"
        lines = [ln for ln in manifest.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1  # only the existing config

    def test_array_stem_collision_raises(self, tmp_path: Path) -> None:
        # Two arms whose configs share a file stem would write to the SAME
        # <campaign_dir>/<stem> → silent overwrite. Raise (#9/#15).
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        a = d1 / "dup.yaml"
        b = d2 / "dup.yaml"
        _write_min_experiment_yaml(a)
        _write_min_experiment_yaml(b)
        cfg = tmp_path / "campaign.yaml"
        _write_campaign_yaml(
            cfg, experiment_configs=[a, b], output_dir=str(tmp_path / "out"),
            execution="array",
        )
        o = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True)
        o.slurm.submit_job = lambda script, **kw: 1  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="stem"):
            o.submit_campaign(str(cfg))

    def test_array_requires_slurm_where(self, tmp_path: Path) -> None:
        cfg, _, _ = self._three_arm_campaign(tmp_path)
        o = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True, where="docker")
        with pytest.raises(NotImplementedError, match="array"):
            o.submit_campaign(str(cfg))

    def test_array_respects_only_filter(self, tmp_path: Path) -> None:
        cfg, out_dir, _ = self._three_arm_campaign(tmp_path)
        o = CampaignOrchestrator(
            base_dir=str(tmp_path), dry_run=True, only=["arm_1"]
        )
        scripts: list[str] = []
        o.slurm.submit_job = lambda script, **kw: scripts.append(script) or 1  # type: ignore[method-assign]
        state = o.submit_campaign(str(cfg))
        # only arm_1 in the array → a 1-task array.
        assert [e.name for e in state.experiments] == ["arm_1"]
        assert "#SBATCH --array=0-0%" in scripts[0]


# ── _generate_ablation_configs ─────────────────────────────────────


class TestGenerateAblationConfigs:
    def test_axes_drive_generation(self, tmp_path: Path) -> None:
        base = tmp_path / "base.yaml"
        # AblationConfigGenerator validates axis paths against this dict.
        base.write_text(yaml.safe_dump({
            "config_version": "6.0",
            "metadata": {"name": "base"},
            # Nested per non-negotiable 1: the block decomposition moved
            # learning_rate under `optimizer`, and the axis below names the
            # canonical path. A flat `optimization.learning_rate` here is the
            # retired spelling, and _get_nested rightly raises on it.
            "optimization": {"optimizer": {"learning_rate": 1e-3}},
            "data": {"batch_size": 4},
            "training": {"output_dir": "results/base"},
        }))
        cfg_dict = {
            "name": "abl_test",
            "execution": "parallel",
            "experiments": [
                {"name": "base", "config": str(base), "role": "baseline"},
            ],
            "ablation_axes": [
                {
                    "config_path": "optimization.optimizer.learning_rate",
                    "values": [1e-4, 1e-3],
                }
            ],
        }
        cfg = CampaignConfigSchema.model_validate(cfg_dict)
        camp_dir = tmp_path / "out"
        camp_dir.mkdir()

        o = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True)
        out = o._generate_ablation_configs(cfg, camp_dir)
        # Two ablation arms produced and written to disk.
        assert len(out) == 2
        for p in out:
            assert Path(p).exists()
            assert str(camp_dir / "ablation_configs") in p


# ── check_progress — no-op branch ──────────────────────────────────


class TestCheckProgressNoOp:
    def test_short_circuits_when_all_terminal(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Hand-build a state where every arm is already in a terminal
        # state — check_progress must NOT call sacct.
        from spectramr.infrastructure.orchestration.campaign_state import (
            ExperimentStatus,
        )

        state = CampaignState(
            campaign_name="terminal",
            campaign_dir=str(tmp_path),
            experiments=[
                ExperimentStatus(
                    name="a",
                    config_path="x.yaml",
                    status="completed",
                    slurm_job_id=1,
                ),
                ExperimentStatus(
                    name="b",
                    config_path="x.yaml",
                    status="failed",
                    slurm_job_id=2,
                ),
            ],
        )
        state.save()  # writes campaign_state.json

        o = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True)
        out_state = o.check_progress(str(tmp_path))
        # Same number of experiments — no mutation.
        assert len(out_state.experiments) == 2
        # All still terminal — no SLURM polling fired.
        assert out_state.all_terminal


# ── cancel_campaign — no-active-jobs branch ────────────────────────


class TestCancelCampaignNoActive:
    def test_no_active_jobs_short_circuits(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.orchestration.campaign_state import (
            ExperimentStatus,
        )

        state = CampaignState(
            campaign_name="done",
            campaign_dir=str(tmp_path),
            experiments=[
                ExperimentStatus(
                    name="a",
                    config_path="x.yaml",
                    status="completed",
                    slurm_job_id=1,
                ),
            ],
        )
        state.save()

        o = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True)
        out = o.cancel_campaign(str(tmp_path))
        # State unchanged.
        assert out.experiments[0].status == "completed"


# ── AblationAxisSchema integration ─────────────────────────────────


class TestAxisSchemaIntegrationSanity:
    def test_axes_are_loadable_via_ablation_axis_schema(self) -> None:
        # Defensive check — keeps the orchestrator + schema in sync.
        a = AblationAxisSchema(
            config_path="optimization.optimizer.learning_rate", values=[1e-4, 1e-3]
        )
        assert a.config_path == "optimization.optimizer.learning_rate"
        assert a.values == [1e-4, 1e-3]


class TestCampaignLevelResume:
    """`campaign submit --resume` must not re-submit arms a prior run already
    finished or still has in flight. Previously ``submit_campaign`` always built
    a fresh state and re-queued every arm (duplicate jobs, clobbered outputs)."""

    @staticmethod
    def _save_prior(tmp_path: Path, statuses: dict[str, str]) -> None:
        state = CampaignState(
            campaign_name="c",
            campaign_config_path="x.yaml",
            campaign_dir=str(tmp_path),
            description="",
        )
        for name, status in statuses.items():
            state.experiments.append(
                ExperimentStatus(name=name, config_path=f"{name}.yaml", status=status)
            )
        state.save()  # → <tmp_path>/campaign_state.json

    def test_empty_when_not_resuming(self, tmp_path: Path) -> None:
        self._save_prior(tmp_path, {"a": "completed"})
        orch = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True, resume=False)
        assert orch._prior_handled(tmp_path) == {}

    def test_empty_when_no_prior_state(self, tmp_path: Path) -> None:
        orch = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True, resume=True)
        assert orch._prior_handled(tmp_path) == {}

    def test_skips_completed_and_active_only(self, tmp_path: Path) -> None:
        self._save_prior(
            tmp_path,
            {
                "done": "completed",
                "live": "running",
                "queued": "submitted",
                "boom": "failed",
                "todo": "pending",
            },
        )
        orch = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True, resume=True)
        handled = orch._prior_handled(tmp_path)
        # completed / running / submitted are carried; failed + pending are
        # re-submittable so they must NOT be skipped.
        assert set(handled) == {"done", "live", "queued"}
        assert handled["done"].status == "completed"


class TestSequentialStageGroupSafety:
    """`_submit_sequential` correctness:

    * A broken parent group (an experiment that fails to submit) must NOT let a
      dependent group launch with a silently-shortened ``afterok`` chain — the
      child runs without its prerequisite.
    * ``--resume`` must skip a parent that already ``completed`` rather than
      re-submitting the whole cascade.
    """

    @staticmethod
    def _seq_config(tmp_path: Path) -> CampaignConfigSchema:
        cfg_file = tmp_path / "exp.yaml"
        cfg_file.write_text("x: 1\n")
        return CampaignConfigSchema.model_validate(
            {
                "name": "seq",
                "description": "",
                "execution": "sequential",
                "output_dir": "out",
                "stage_groups": [
                    StageGroupSchema.model_validate(
                        {
                            "name": "parent",
                            "experiments": [
                                {"name": "p1", "config": str(cfg_file), "role": "variant"}
                            ],
                        }
                    ),
                    StageGroupSchema.model_validate(
                        {
                            "name": "child",
                            "depends_on": ["parent"],
                            "experiments": [
                                {"name": "c1", "config": str(cfg_file), "role": "variant"}
                            ],
                        }
                    ),
                ],
            }
        )

    def test_broken_parent_skips_dependent_group(self, tmp_path, monkeypatch) -> None:
        config = self._seq_config(tmp_path)
        campaign_dir = tmp_path / "out"
        campaign_dir.mkdir()
        orch = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=False)

        calls = {"n": 0}

        def fake_submit(script, dependency_job_ids=None):
            calls["n"] += 1
            if calls["n"] == 1:  # the parent — fails to submit
                raise RuntimeError("sbatch boom")
            return calls["n"]

        monkeypatch.setattr(orch.slurm, "submit_job", fake_submit)
        monkeypatch.setattr(
            SLURMBackend, "generate_job_script", staticmethod(lambda **k: "script")
        )

        state = CampaignState(campaign_name="seq", campaign_dir=str(campaign_dir))
        result = orch._submit_sequential(config, state, campaign_dir)

        # The child must NOT have been submitted (parent group is broken → its
        # afterok prerequisite can't be satisfied). Pre-fix the child launched
        # with an empty/shortened dependency chain (calls["n"] == 2).
        assert calls["n"] == 1
        assert result.get_experiment("p1").status == "failed"
        assert result.get_experiment("c1").status == "failed"

    def test_resume_skips_completed_parent(self, tmp_path, monkeypatch) -> None:
        config = self._seq_config(tmp_path)
        campaign_dir = tmp_path / "out"
        campaign_dir.mkdir()

        prior = CampaignState(campaign_name="seq", campaign_dir=str(campaign_dir))
        prior.experiments.append(
            ExperimentStatus(
                name="p1",
                config_path=str(tmp_path / "exp.yaml"),
                status="completed",
                stage_group="parent",
            )
        )
        prior.save()  # → <campaign_dir>/campaign_state.json

        orch = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=False, resume=True)

        submitted_deps = []

        def fake_submit(script, dependency_job_ids=None):
            submitted_deps.append(dependency_job_ids)
            return len(submitted_deps)

        monkeypatch.setattr(orch.slurm, "submit_job", fake_submit)
        monkeypatch.setattr(
            SLURMBackend, "generate_job_script", staticmethod(lambda **k: "script")
        )

        state = CampaignState(campaign_name="seq", campaign_dir=str(campaign_dir))
        result = orch._submit_sequential(config, state, campaign_dir)

        # Only the child is submitted; the completed parent is carried forward.
        assert len(submitted_deps) == 1
        # The child has NO afterok dependency on the already-completed parent.
        assert submitted_deps[0] in (None, [])
        assert result.get_experiment("p1").status == "completed"
        assert result.get_experiment("c1").status == "submitted"


class TestVanishedJobResolution:
    """A job that drops out of both ``sacct`` and ``squeue`` (state ``UNKNOWN``)
    must reach a terminal state, not be re-polled forever. Previously ``UNKNOWN``
    normalised to ``submitted`` → ``all_terminal`` never True → ``campaign watch``
    hung. Resolution probes for an output artifact (no false success)."""

    @staticmethod
    def _setup(tmp_path: Path, with_artifact: bool) -> Path:
        campaign_dir = tmp_path / "out"
        campaign_dir.mkdir()
        out = campaign_dir / "arm1"
        (out / "checkpoints").mkdir(parents=True)
        if with_artifact:
            (out / "checkpoints" / "best.pt").write_text("x")
        state = CampaignState(campaign_name="c", campaign_dir=str(campaign_dir))
        state.experiments.append(
            ExperimentStatus(
                name="arm1",
                config_path="x.yaml",
                slurm_job_id=42,
                status="running",
                start_time="2026-01-01T00:00:00",
                output_dir=str(out),
            )
        )
        state.save()
        return campaign_dir

    def _orch(self, tmp_path, monkeypatch):
        orch = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True)
        monkeypatch.setattr(
            orch.slurm,
            "query_batch_status",
            lambda ids: {42: JobStatus(job_id=42, state="UNKNOWN")},
        )
        monkeypatch.setattr(orch, "_discover_results", lambda exp: None)
        return orch

    def test_vanished_job_with_artifact_becomes_completed(
        self, tmp_path, monkeypatch
    ) -> None:
        campaign_dir = self._setup(tmp_path, with_artifact=True)
        orch = self._orch(tmp_path, monkeypatch)
        orch.check_progress(str(campaign_dir))  # poll 1 — transient, keep polling
        result = orch.check_progress(str(campaign_dir))  # poll 2 — terminal
        assert result.get_experiment("arm1").status == "completed"

    def test_vanished_job_without_artifact_becomes_failed(
        self, tmp_path, monkeypatch
    ) -> None:
        campaign_dir = self._setup(tmp_path, with_artifact=False)
        orch = self._orch(tmp_path, monkeypatch)
        orch.check_progress(str(campaign_dir))
        result = orch.check_progress(str(campaign_dir))
        assert result.get_experiment("arm1").status == "failed"

    def test_single_unknown_poll_keeps_polling(self, tmp_path, monkeypatch) -> None:
        """One UNKNOWN poll must NOT immediately terminate (a just-submitted job
        is briefly absent before the scheduler registers it)."""
        campaign_dir = self._setup(tmp_path, with_artifact=False)
        orch = self._orch(tmp_path, monkeypatch)
        result = orch.check_progress(str(campaign_dir))  # only one poll
        assert result.get_experiment("arm1").status == "running"


# --------------------------------------------------------------------------- #
# #1343 -- the best-metrics extraction sat after an unconditional `return None`
# --------------------------------------------------------------------------- #
class TestBestMetricsExtraction:
    """The block existed, was correct, and had never run.

    It was unreachable code inside ``_discover_metrics_csv``, so the campaign
    report's ``Best PSNR`` column was permanently empty and 4 ``F821`` sat on
    an ``exp`` name that had no binding in that scope.
    """

    @staticmethod
    def _exp(csv_path):
        from types import SimpleNamespace

        return SimpleNamespace(metrics_csv_path=str(csv_path), best_metrics={})

    def test_higher_is_better_metrics_take_the_maximum(self, tmp_path):
        from spectramr.infrastructure.orchestration.campaign_orchestrator import (
            CampaignOrchestrator,
        )

        csv = tmp_path / "validation_metrics.csv"
        csv.write_text("epoch,val_psnr,val_ssim\n1,30.0,0.80\n2,32.5,0.91\n")
        exp = self._exp(csv)
        CampaignOrchestrator._extract_best_metrics(exp)
        assert exp.best_metrics["psnr"] == 32.5
        assert exp.best_metrics["ssim"] == 0.91

    def test_lower_is_better_metrics_take_the_minimum(self, tmp_path):
        from spectramr.infrastructure.orchestration.campaign_orchestrator import (
            CampaignOrchestrator,
        )

        csv = tmp_path / "validation_metrics.csv"
        csv.write_text("epoch,val_mse,val_lpips\n1,0.05,0.30\n2,0.02,0.11\n")
        exp = self._exp(csv)
        CampaignOrchestrator._extract_best_metrics(exp)
        assert exp.best_metrics["mse"] == 0.02
        assert exp.best_metrics["lpips"] == 0.11

    def test_no_csv_path_is_a_no_op_not_a_crash(self):
        from types import SimpleNamespace

        from spectramr.infrastructure.orchestration.campaign_orchestrator import (
            CampaignOrchestrator,
        )

        exp = SimpleNamespace(metrics_csv_path=None, best_metrics={})
        CampaignOrchestrator._extract_best_metrics(exp)
        assert exp.best_metrics == {}

    def test_the_extractor_is_reachable_from_the_discovery_path(self):
        """A capability is not delivered until the production path calls it."""
        import inspect

        from spectramr.infrastructure.orchestration.campaign_orchestrator import (
            CampaignOrchestrator,
        )

        source = inspect.getsource(CampaignOrchestrator)
        assert source.count("_extract_best_metrics") >= 2, (
            "the extractor is defined but never called -- the exact defect "
            "#1343 reported"
        )



# ── --slurm overrides ───────────────────────────────────────────────


class TestSlurmOverrides:
    """The world size is a launch-time decision, not a committed constant.

    Under a sharded strategy ``num_devices`` also sets the effective batch
    (there is no world-size LR scaling in-tree), so baking one number into a
    campaign YAML would fix an optimisation decision for every later submitter
    on whatever allocation they happen to hold.
    """

    def test_overrides_win_over_campaign_defaults(self, tmp_path: Path) -> None:
        cfg = tmp_path / "c.yaml"
        _write_campaign_yaml(cfg, experiment_configs=[tmp_path / "e.yaml"])
        campaign = CampaignConfigSchema(**yaml.safe_load(cfg.read_text()))

        o = CampaignOrchestrator(dry_run=True, slurm_overrides={"gpus": "4"})
        assert o._resolve_slurm_params(campaign)["gpus"] == 4

    def test_overrides_win_over_a_per_arm_override_too(self, tmp_path: Path) -> None:
        """The CLI is last because it is the layer that knows the allocation."""
        cfg = tmp_path / "c.yaml"
        _write_campaign_yaml(cfg, experiment_configs=[tmp_path / "e.yaml"])
        campaign = CampaignConfigSchema(**yaml.safe_load(cfg.read_text()))

        o = CampaignOrchestrator(dry_run=True, slurm_overrides={"gpus": "8"})
        resolved = o._resolve_slurm_params(campaign, {"gpus": 2}, {"gpus": 3})
        assert resolved["gpus"] == 8

    def test_layers_below_the_cli_still_apply_in_order(self, tmp_path: Path) -> None:
        """Group then arm, unchanged when the CLI says nothing about that key."""
        cfg = tmp_path / "c.yaml"
        _write_campaign_yaml(cfg, experiment_configs=[tmp_path / "e.yaml"])
        campaign = CampaignConfigSchema(**yaml.safe_load(cfg.read_text()))

        o = CampaignOrchestrator(dry_run=True, slurm_overrides={"gpus": "4"})
        resolved = o._resolve_slurm_params(campaign, {"mem": "16GB"}, {"mem": "32GB"})
        assert (resolved["mem"], resolved["gpus"]) == ("32GB", 4)

    def test_no_override_leaves_the_campaign_untouched(self, tmp_path: Path) -> None:
        cfg = tmp_path / "c.yaml"
        _write_campaign_yaml(cfg, experiment_configs=[tmp_path / "e.yaml"])
        campaign = CampaignConfigSchema(**yaml.safe_load(cfg.read_text()))

        o = CampaignOrchestrator(dry_run=True)
        assert o.slurm_overrides == {}
        assert o._resolve_slurm_params(campaign) == campaign.slurm_defaults.model_dump()

    def test_a_string_value_is_coerced_to_the_schema_type(self) -> None:
        """argparse hands over ``"4"``; sbatch directives need the int."""
        o = CampaignOrchestrator(dry_run=True, slurm_overrides={"gpus": "4"})
        assert o.slurm_overrides["gpus"] == 4
        assert isinstance(o.slurm_overrides["gpus"], int)

    def test_an_unknown_key_raises_rather_than_being_dropped(self) -> None:
        """Planted: the shape the guard exists for.

        The generator takes a plain dict, so a typo'd key would vanish without
        a word and the submitter would read the silence as the override having
        applied (pitfall #15).
        """
        with pytest.raises(ValueError, match="Unknown SLURM override key"):
            CampaignOrchestrator(dry_run=True, slurm_overrides={"gpu": "4"})

    def test_an_out_of_range_value_raises_rather_than_reaching_sbatch(self) -> None:
        """The second shape: key is real, value is not.

        ``gpus`` is ``ge=0`` in the schema; without the round-trip this lands
        as ``#SBATCH --gpus=-1`` and is rejected once per arm, on the cluster,
        after the campaign has already been submitted.
        """
        with pytest.raises(ValidationError):
            CampaignOrchestrator(dry_run=True, slurm_overrides={"gpus": "-1"})

    def test_only_the_keys_given_are_returned(self) -> None:
        """Not the whole schema: an override must not silently reinstate a
        default the campaign deliberately changed."""
        o = CampaignOrchestrator(dry_run=True, slurm_overrides={"gpus": "4"})
        assert set(o.slurm_overrides) == {"gpus"}
