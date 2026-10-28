"""Orchestration tests — campaign_state, campaign_orchestrator, slurm_backend,
ablation_config_generator (TASK IV.G).

Coverage targets
----------------
- CampaignState: serialise / deserialise round-trip, status-machine properties,
  mutation helpers, summary_table.
- CampaignOrchestrator: state-machine transitions (queued → running →
  succeeded / failed) on a synthetic 2-arm manifest via dry-run.
- SLURMBackend: dry-run generates a parseable sbatch script (contains #SBATCH),
  JobStatus.normalised_state mapping, _parse_elapsed.
- AblationConfigGenerator: grid emit expected count, random sub-sampling,
  invalid path raises.

Skips
-----
- Any test that would invoke a real ``sbatch`` / ``sacct`` / ``squeue`` binary
  (skip with reason "requires live SLURM cluster").
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from spectramr.infrastructure.orchestration.ablation_config_generator import (
    AblationConfigGenerator,
    _get_nested,
    _set_nested,
)
from spectramr.infrastructure.orchestration.campaign_state import (
    CampaignState,
    ExperimentStatus,
)
from spectramr.infrastructure.orchestration.slurm_backend import (
    JobStatus,
    SLURMBackend,
)
from spectramr.config.schemas.campaign import AblationAxisSchema

# ── helpers ──────────────────────────────────────────────────────────────────

pytestmark = pytest.mark.usefixtures("slurm_account")


def _make_state(tmp_path: Path, **kwargs) -> CampaignState:
    return CampaignState(
        campaign_name=kwargs.get("campaign_name", "test_campaign"),
        campaign_dir=str(tmp_path),
        **{k: v for k, v in kwargs.items() if k != "campaign_name"},
    )


def _two_arm_state(tmp_path: Path) -> CampaignState:
    state = _make_state(tmp_path)
    state.experiments.append(
        ExperimentStatus(name="arm_a", config_path="a.yaml", status="pending")
    )
    state.experiments.append(
        ExperimentStatus(name="arm_b", config_path="b.yaml", status="pending")
    )
    return state


# ── CampaignState ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestCampaignStateRoundTrip:
    """Serialise → persist → load cycle preserves all fields."""

    def test_canary_round_trip(self, tmp_path: Path) -> None:
        """Basic round-trip: two arms, mixed statuses."""
        state = _two_arm_state(tmp_path)
        state.experiments[0].status = "running"
        state.experiments[1].status = "completed"
        state.experiments[1].checkpoint_path = "/checkpoints/best.pt"
        state.experiments[1].best_metrics = {"psnr": 32.5, "ssim": 0.91}

        saved_path = state.save()
        assert saved_path.exists()

        loaded = CampaignState.load(saved_path)
        assert loaded.campaign_name == "test_campaign"
        assert len(loaded.experiments) == 2
        assert loaded.experiments[0].status == "running"
        assert loaded.experiments[1].checkpoint_path == "/checkpoints/best.pt"
        assert loaded.experiments[1].best_metrics["psnr"] == pytest.approx(32.5)

    def test_json_is_valid_json(self, tmp_path: Path) -> None:
        """The saved file is valid JSON and contains expected keys."""
        state = _two_arm_state(tmp_path)
        saved_path = state.save()
        with open(saved_path) as f:
            data = json.load(f)
        assert data["campaign_name"] == "test_campaign"
        assert "experiments" in data
        assert isinstance(data["experiments"], list)

    def test_default_path_under_campaign_dir(self, tmp_path: Path) -> None:
        """When no explicit path is given, file lands in campaign_dir."""
        state = _two_arm_state(tmp_path)
        saved_path = state.save()
        assert saved_path == tmp_path / "campaign_state.json"

    def test_explicit_path(self, tmp_path: Path) -> None:
        """Explicit path parameter is respected."""
        state = _two_arm_state(tmp_path)
        custom = tmp_path / "sub" / "my_state.json"
        saved_path = state.save(custom)
        assert saved_path == custom
        assert custom.exists()

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        """Loading from a non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            CampaignState.load(tmp_path / "nonexistent.json")

    def test_round_trip_with_stage_group(self, tmp_path: Path) -> None:
        """stage_group and depends_on_jobs survive round-trip."""
        state = _make_state(tmp_path)
        state.experiments.append(
            ExperimentStatus(
                name="seq_arm",
                config_path="seq.yaml",
                stage_group="stage_1",
                depends_on_jobs=[111, 222],
                status="submitted",
                slurm_job_id=333,
            )
        )
        loaded = CampaignState.load(state.save())
        exp = loaded.get_experiment("seq_arm")
        assert exp is not None
        assert exp.stage_group == "stage_1"
        assert exp.depends_on_jobs == [111, 222]
        assert exp.slurm_job_id == 333


@pytest.mark.unit
class TestCampaignStateProperties:
    """Status-machine properties and convenience lookups."""

    def _make_mixed(self) -> CampaignState:
        state = CampaignState(campaign_name="t", campaign_dir="/tmp")
        for name, status, ckpt in [
            ("completed_a", "completed", "/ckpt"),
            ("completed_no_ckpt", "completed", None),
            ("failed_a", "failed", None),
            ("timeout_b", "timeout", None),
            ("running_c", "running", None),
            ("pending_d", "pending", None),
            ("submitted_e", "submitted", None),
        ]:
            state.experiments.append(
                ExperimentStatus(
                    name=name, config_path=f"{name}.yaml",
                    status=status, checkpoint_path=ckpt,
                )
            )
        return state

    def test_canary_property_counts(self) -> None:
        state = self._make_mixed()
        assert len(state.completed) == 2
        assert len(state.failed) == 2   # failed + timeout both land here
        assert len(state.running) == 1
        assert len(state.pending) == 1
        assert len(state.submitted) == 1

    def test_evaluable_requires_checkpoint(self) -> None:
        state = self._make_mixed()
        evals = state.evaluable
        assert len(evals) == 1
        assert evals[0].checkpoint_path is not None

    def test_all_terminal_false_while_running(self) -> None:
        state = self._make_mixed()
        assert not state.all_terminal

    def test_all_terminal_true_when_no_active(self) -> None:
        state = CampaignState(campaign_name="t", campaign_dir="/tmp")
        state.experiments.append(
            ExperimentStatus(name="a", config_path="a.yaml", status="completed")
        )
        state.experiments.append(
            ExperimentStatus(name="b", config_path="b.yaml", status="failed")
        )
        assert state.all_terminal

    def test_progress_fraction_zero_when_empty(self) -> None:
        state = CampaignState(campaign_name="t", campaign_dir="/tmp")
        assert state.progress_fraction == pytest.approx(0.0)

    def test_progress_fraction_partial(self) -> None:
        state = CampaignState(campaign_name="t", campaign_dir="/tmp")
        state.experiments.append(
            ExperimentStatus(name="a", config_path="a.yaml", status="completed")
        )
        state.experiments.append(
            ExperimentStatus(name="b", config_path="b.yaml", status="running")
        )
        assert state.progress_fraction == pytest.approx(0.5)

    def test_get_experiment_by_name(self) -> None:
        state = CampaignState(campaign_name="t", campaign_dir="/tmp")
        state.experiments.append(
            ExperimentStatus(name="my_arm", config_path="my.yaml", status="pending")
        )
        found = state.get_experiment("my_arm")
        assert found is not None
        assert found.name == "my_arm"

    def test_get_experiment_missing_returns_none(self) -> None:
        state = CampaignState(campaign_name="t", campaign_dir="/tmp")
        assert state.get_experiment("no_such") is None

    def test_get_by_job_id(self) -> None:
        state = CampaignState(campaign_name="t", campaign_dir="/tmp")
        state.experiments.append(
            ExperimentStatus(
                name="x", config_path="x.yaml", status="running", slurm_job_id=9999
            )
        )
        found = state.get_by_job_id(9999)
        assert found is not None
        assert found.name == "x"
        assert state.get_by_job_id(0) is None

    def test_get_by_stage_group(self) -> None:
        state = CampaignState(campaign_name="t", campaign_dir="/tmp")
        for name, group in [("a", "s1"), ("b", "s1"), ("c", "s2")]:
            state.experiments.append(
                ExperimentStatus(
                    name=name, config_path=f"{name}.yaml",
                    stage_group=group, status="pending"
                )
            )
        s1 = state.get_by_stage_group("s1")
        assert len(s1) == 2
        assert {e.name for e in s1} == {"a", "b"}


@pytest.mark.unit
class TestCampaignStateMutation:
    """update_experiment and summary_table."""

    def test_canary_update_status(self) -> None:
        state = CampaignState(campaign_name="t", campaign_dir="/tmp")
        state.experiments.append(
            ExperimentStatus(name="arm", config_path="arm.yaml", status="running")
        )
        state.update_experiment("arm", status="completed", checkpoint_path="/c.pt")
        arm = state.get_experiment("arm")
        assert arm is not None
        assert arm.status == "completed"
        assert arm.checkpoint_path == "/c.pt"

    def test_update_missing_raises(self) -> None:
        state = CampaignState(campaign_name="t", campaign_dir="/tmp")
        with pytest.raises(KeyError, match="no_such"):
            state.update_experiment("no_such", status="failed")

    def test_summary_table_contains_names_and_statuses(self) -> None:
        state = CampaignState(campaign_name="t", campaign_dir="/tmp")
        state.experiments.append(
            ExperimentStatus(name="arm_alpha", config_path="a.yaml", status="completed")
        )
        state.experiments.append(
            ExperimentStatus(name="arm_beta", config_path="b.yaml", status="running")
        )
        table = state.summary_table()
        assert "arm_alpha" in table
        assert "arm_beta" in table
        assert "completed" in table
        assert "running" in table

    def test_experiment_status_is_terminal(self) -> None:
        for terminal_status in ("completed", "failed", "cancelled", "timeout"):
            exp = ExperimentStatus(
                name="x", config_path="x.yaml", status=terminal_status
            )
            assert exp.is_terminal, f"Expected is_terminal for status={terminal_status}"

    def test_experiment_status_not_terminal(self) -> None:
        for active_status in ("pending", "submitted", "running"):
            exp = ExperimentStatus(
                name="x", config_path="x.yaml", status=active_status
            )
            assert not exp.is_terminal, f"Expected NOT is_terminal for status={active_status}"


# ── CampaignOrchestrator — state-machine transitions ──────────────────────────


@pytest.mark.unit
class TestCampaignOrchestratorStateMachine:
    """Test state-machine transitions using dry_run=True (no real SLURM calls)."""

    def _write_campaign_yaml(self, tmp_path: Path, arm_configs: list[str]) -> Path:
        """Write a minimal campaign YAML with the given arm config paths."""
        campaign_data = {
            "name": "sm_test",
            "description": "state-machine test",
            "output_dir": str(tmp_path / "results"),
            "experiments": [
                {"name": f"arm_{i}", "config": cfg, "role": "variant"}
                for i, cfg in enumerate(arm_configs)
            ],
        }
        path = tmp_path / "campaign.yaml"
        path.write_text(yaml.dump(campaign_data))
        return path

    def test_canary_dry_run_two_arm_submitted(self, tmp_path: Path) -> None:
        """dry_run=True transitions both arms to 'submitted' (job_id=0)."""
        # Write two real-looking (but non-schema) YAML stubs for the arms.
        # The orchestrator validates that the file *exists* before submitting.
        for name in ("arm_0.yaml", "arm_1.yaml"):
            (tmp_path / name).write_text("metadata:\n  name: stub\n")

        campaign_yaml = self._write_campaign_yaml(
            tmp_path,
            [str(tmp_path / "arm_0.yaml"), str(tmp_path / "arm_1.yaml")],
        )

        from spectramr.infrastructure.orchestration.campaign_orchestrator import (
            CampaignOrchestrator,
        )

        orch = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True)
        state = orch.submit_campaign(str(campaign_yaml))

        submitted = [e for e in state.experiments if e.status == "submitted"]
        assert len(submitted) == 2, f"Expected 2 submitted, got: {[(e.name, e.status) for e in state.experiments]}"

    def test_failed_arm_when_config_missing(self, tmp_path: Path) -> None:
        """An arm whose config YAML does not exist transitions to 'failed'."""
        # Only write arm_0.yaml; leave arm_1.yaml absent.
        (tmp_path / "arm_0.yaml").write_text("metadata:\n  name: stub\n")

        campaign_yaml = self._write_campaign_yaml(
            tmp_path,
            [str(tmp_path / "arm_0.yaml"), str(tmp_path / "arm_1_MISSING.yaml")],
        )

        from spectramr.infrastructure.orchestration.campaign_orchestrator import (
            CampaignOrchestrator,
        )

        orch = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True)
        state = orch.submit_campaign(str(campaign_yaml))

        statuses = {e.name: e.status for e in state.experiments}
        assert statuses["arm_0"] == "submitted"
        assert statuses["arm_1"] == "failed"

    def test_state_file_persisted_after_submit(self, tmp_path: Path) -> None:
        """submit_campaign writes campaign_state.json into the output dir."""
        (tmp_path / "arm_0.yaml").write_text("metadata:\n  name: stub\n")
        campaign_yaml = self._write_campaign_yaml(
            tmp_path, [str(tmp_path / "arm_0.yaml")]
        )

        from spectramr.infrastructure.orchestration.campaign_orchestrator import (
            CampaignOrchestrator,
        )

        orch = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True)
        state = orch.submit_campaign(str(campaign_yaml))

        state_file = Path(state.campaign_dir) / "campaign_state.json"
        assert state_file.exists()

    def test_only_filter_restricts_arms(self, tmp_path: Path) -> None:
        """--only=arm_0 submits arm_0 only; arm_1 is not included at all."""
        for name in ("arm_0.yaml", "arm_1.yaml"):
            (tmp_path / name).write_text("metadata:\n  name: stub\n")

        campaign_yaml = self._write_campaign_yaml(
            tmp_path,
            [str(tmp_path / "arm_0.yaml"), str(tmp_path / "arm_1.yaml")],
        )

        from spectramr.infrastructure.orchestration.campaign_orchestrator import (
            CampaignOrchestrator,
        )

        orch = CampaignOrchestrator(
            base_dir=str(tmp_path), dry_run=True, only=["arm_0"]
        )
        state = orch.submit_campaign(str(campaign_yaml))

        names = [e.name for e in state.experiments]
        assert "arm_0" in names
        assert "arm_1" not in names

    @pytest.mark.skip(reason="requires live SLURM cluster (sbatch binary)")
    def test_real_submit_calls_sbatch(self, tmp_path: Path) -> None:
        """Submits a real sbatch job — only meaningful on a SLURM node."""
        pass

    def test_topological_sort_linear(self) -> None:
        """_topological_sort returns a linear chain A → B → C in correct order."""
        from spectramr.infrastructure.orchestration.campaign_orchestrator import (
            CampaignOrchestrator,
        )
        from types import SimpleNamespace

        groups = [
            SimpleNamespace(name="A", depends_on=[]),
            SimpleNamespace(name="B", depends_on=["A"]),
            SimpleNamespace(name="C", depends_on=["B"]),
        ]
        ordered = CampaignOrchestrator._topological_sort(groups)
        names = [g.name for g in ordered]
        assert names.index("A") < names.index("B") < names.index("C")

    def test_topological_sort_cycle_raises(self) -> None:
        """A cycle in stage_group dependencies raises ValueError."""
        from spectramr.infrastructure.orchestration.campaign_orchestrator import (
            CampaignOrchestrator,
        )
        from types import SimpleNamespace

        groups = [
            SimpleNamespace(name="A", depends_on=["C"]),
            SimpleNamespace(name="B", depends_on=["A"]),
            SimpleNamespace(name="C", depends_on=["B"]),
        ]
        with pytest.raises(ValueError, match="[Cc]ycle"):
            CampaignOrchestrator._topological_sort(groups)

    def test_selector_matches_name(self) -> None:
        from spectramr.infrastructure.orchestration.campaign_orchestrator import (
            CampaignOrchestrator,
        )
        from types import SimpleNamespace

        exp = SimpleNamespace(name="my_exp", role="baseline", tags={})
        assert CampaignOrchestrator._selector_matches("name=my_exp", exp)
        assert not CampaignOrchestrator._selector_matches("name=other_exp", exp)

    def test_selector_matches_role(self) -> None:
        from spectramr.infrastructure.orchestration.campaign_orchestrator import (
            CampaignOrchestrator,
        )
        from types import SimpleNamespace

        exp = SimpleNamespace(name="e", role="baseline", tags={})
        assert CampaignOrchestrator._selector_matches("role=baseline", exp)
        assert not CampaignOrchestrator._selector_matches("role=variant", exp)

    def test_selector_invalid_form_raises(self) -> None:
        from spectramr.infrastructure.orchestration.campaign_orchestrator import (
            CampaignOrchestrator,
        )
        from types import SimpleNamespace

        exp = SimpleNamespace(name="e", role="baseline", tags={})
        with pytest.raises(ValueError, match="Invalid selector"):
            CampaignOrchestrator._selector_matches("no_equals_sign", exp)

    def test_parse_elapsed_hms(self) -> None:
        """_parse_elapsed handles HH:MM:SS format."""
        from spectramr.infrastructure.orchestration.campaign_orchestrator import (
            _parse_elapsed,
        )

        assert _parse_elapsed("01:30:00") == pytest.approx(5400.0)
        assert _parse_elapsed("00:02:30") == pytest.approx(150.0)

    def test_parse_elapsed_day_hms(self) -> None:
        """_parse_elapsed handles D-HH:MM:SS format."""
        from spectramr.infrastructure.orchestration.campaign_orchestrator import (
            _parse_elapsed,
        )

        assert _parse_elapsed("1-00:00:00") == pytest.approx(86400.0)
        assert _parse_elapsed("2-12:00:00") == pytest.approx(2 * 86400 + 12 * 3600)

    def test_parse_elapsed_invalid_returns_none(self) -> None:
        from spectramr.infrastructure.orchestration.campaign_orchestrator import (
            _parse_elapsed,
        )

        assert _parse_elapsed("not-a-time") is None
        assert _parse_elapsed("") is None


# ── SLURMBackend ──────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSLURMBackendScript:
    """generate_job_script and dry-run behaviour."""

    def _params(self) -> dict:
        return {
            "partition": "gpu",
            "time": "48:00:00",
            "mem": "32GB",
            "cpus_per_task": 4,
            "gpus": 1,
            "nodes": 1,
            "ntasks": 1,
            "mail_type": "FAIL",
        }

    def test_canary_script_contains_sbatch_directives(self) -> None:
        """Generated script starts with #!/bin/bash and has #SBATCH lines."""
        script = SLURMBackend.generate_job_script(
            experiment_name="smoke_test",
            config_path="/exp/config.yaml",
            output_dir="/out",
            base_dir="/repo",
            slurm_params=self._params(),
        )
        assert script.startswith("#!/bin/bash")
        assert "#SBATCH" in script
        assert "#SBATCH --job-name=smoke_test" in script

    def test_script_contains_config_path(self) -> None:
        script = SLURMBackend.generate_job_script(
            experiment_name="e",
            config_path="/my/path/experiment.yaml",
            output_dir="/out",
            base_dir="/repo",
        )
        assert "/my/path/experiment.yaml" in script

    def test_resume_flag_present_when_requested(self) -> None:
        script = SLURMBackend.generate_job_script(
            experiment_name="e",
            config_path="/cfg.yaml",
            output_dir="/out",
            base_dir="/repo",
            resume=True,
        )
        assert "--resume auto" in script

    def test_resume_flag_absent_by_default(self) -> None:
        script = SLURMBackend.generate_job_script(
            experiment_name="e",
            config_path="/cfg.yaml",
            output_dir="/out",
            base_dir="/repo",
            resume=False,
        )
        assert "--resume auto" not in script

    def test_config_overrides_injected(self) -> None:
        """config_overrides dict produces -O key=value flags in the script."""
        script = SLURMBackend.generate_job_script(
            experiment_name="e",
            config_path="/cfg.yaml",
            output_dir="/out",
            base_dir="/repo",
            config_overrides={"model.checkpoint_path": "/ckpt/best.pt"},
        )
        assert "-O model.checkpoint_path=/ckpt/best.pt" in script

    def test_output_dir_override_in_script(self) -> None:
        """training.output_dir is always overridden in the generated script."""
        script = SLURMBackend.generate_job_script(
            experiment_name="e",
            config_path="/cfg.yaml",
            output_dir="/campaign/out",
            base_dir="/repo",
        )
        assert "training.output_dir=/campaign/out" in script

    def test_custom_slurm_params_applied(self) -> None:
        script = SLURMBackend.generate_job_script(
            experiment_name="e",
            config_path="/cfg.yaml",
            output_dir="/out",
            base_dir="/repo",
            slurm_params={"partition": "bigmem", "mem": "128GB", "time": "72:00:00",
                          "cpus_per_task": 16, "gpus": 2, "nodes": 1, "ntasks": 1,
                          "mail_type": "END"},
        )
        assert "#SBATCH --partition=bigmem" in script
        assert "#SBATCH --mem=128GB" in script

    def test_dry_run_submit_returns_zero(self) -> None:
        """dry_run=True submit_job returns 0 without calling sbatch."""
        backend = SLURMBackend(dry_run=True)
        job_id = backend.submit_job("#!/bin/bash\n#SBATCH --job-name=test\n")
        assert job_id == 0

    def test_dry_run_query_returns_pending(self) -> None:
        """dry_run query_job_status returns PENDING."""
        backend = SLURMBackend(dry_run=True)
        status = backend.query_job_status(12345)
        assert status.state == "PENDING"
        assert status.job_id == 12345

    def test_dry_run_batch_query(self) -> None:
        backend = SLURMBackend(dry_run=True)
        result = backend.query_batch_status([1, 2, 3])
        assert set(result.keys()) == {1, 2, 3}
        assert all(js.state == "PENDING" for js in result.values())

    def test_dry_run_empty_batch_query(self) -> None:
        backend = SLURMBackend(dry_run=True)
        assert backend.query_batch_status([]) == {}

    @pytest.mark.skip(reason="requires live SLURM cluster (sbatch binary)")
    def test_real_submit_parses_job_id(self) -> None:
        pass

    @pytest.mark.skip(reason="requires live SLURM cluster (sacct/squeue binaries)")
    def test_real_query_returns_job_status(self) -> None:
        pass


@pytest.mark.unit
class TestJobStatusNormalisedState:
    """JobStatus.normalised_state maps SLURM state names correctly."""

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
            ("SOME_UNKNOWN_STATE", "submitted"),  # fallback
        ],
    )
    def test_canary_normalisation(self, slurm_state: str, expected: str) -> None:
        js = JobStatus(job_id=1, state=slurm_state)
        assert js.normalised_state == expected, (
            f"state={slurm_state!r}: expected {expected!r}, got {js.normalised_state!r}"
        )

    def test_is_terminal_true_for_terminal_states(self) -> None:
        for state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"):
            assert JobStatus(job_id=1, state=state).is_terminal

    def test_is_terminal_false_for_active_states(self) -> None:
        for state in ("PENDING", "RUNNING", "PREEMPTED"):
            assert not JobStatus(job_id=1, state=state).is_terminal

    def test_exit_code_optional(self) -> None:
        js = JobStatus(job_id=99, state="COMPLETED", exit_code=0)
        assert js.exit_code == 0
        js2 = JobStatus(job_id=100, state="FAILED")
        assert js2.exit_code is None


# ── AblationConfigGenerator ───────────────────────────────────────────────────


@pytest.mark.unit
class TestAblationConfigGeneratorExtended:
    """Grid/random generation; count verification; edge cases."""

    def _write_base(self, tmp_path: Path, extra: dict | None = None) -> Path:
        cfg = {
            "metadata": {"name": "base", "description": "ablation base"},
            "training": {"output_dir": "/tmp/out"},
            "model": {"model_kwargs": {"features": [32, 64]}},
            "objectives": {"reconstruction": {"lambda_l1": 1.0}},
        }
        if extra:
            cfg.update(extra)
        p = tmp_path / "base.yaml"
        p.write_text(yaml.dump(cfg))
        return p

    def test_canary_grid_2x3_yields_6_configs(self, tmp_path: Path) -> None:
        base = self._write_base(tmp_path)
        axes = [
            AblationAxisSchema(
                config_path="objectives.reconstruction.lambda_l1",
                values=[0.5, 1.0, 2.0],
            ),
            AblationAxisSchema(
                config_path="model.model_kwargs.features",
                values=[[32, 64], [64, 128]],
            ),
        ]
        out = tmp_path / "ablation"
        configs = AblationConfigGenerator.generate(str(base), axes, str(out))
        assert len(configs) == 6  # 3 × 2
        for p in configs:
            assert Path(p).exists()

    def test_grid_1x1_yields_1_config(self, tmp_path: Path) -> None:
        base = self._write_base(tmp_path)
        axes = [
            AblationAxisSchema(
                config_path="objectives.reconstruction.lambda_l1",
                values=[0.5],
            ),
        ]
        out = tmp_path / "ablation1"
        configs = AblationConfigGenerator.generate(str(base), axes, str(out))
        assert len(configs) == 1

    def test_generated_config_has_correct_lambda(self, tmp_path: Path) -> None:
        """Each generated config carries its swept value."""
        base = self._write_base(tmp_path)
        axes = [
            AblationAxisSchema(
                config_path="objectives.reconstruction.lambda_l1",
                values=[0.1, 0.9],
            ),
        ]
        configs = AblationConfigGenerator.generate(
            str(base), axes, str(tmp_path / "out")
        )
        lambdas_found = set()
        for p in configs:
            with open(p) as f:
                data = yaml.safe_load(f)
            lambdas_found.add(data["objectives"]["reconstruction"]["lambda_l1"])
        assert lambdas_found == {0.1, 0.9}

    def test_random_mode_sub_samples(self, tmp_path: Path) -> None:
        base = self._write_base(tmp_path)
        axes = [
            AblationAxisSchema(
                config_path="objectives.reconstruction.lambda_l1",
                values=[0.1, 0.5, 1.0, 2.0, 5.0],
            ),
        ]
        configs = AblationConfigGenerator.generate(
            str(base), axes, str(tmp_path / "rand"), mode="random", n_random_samples=3
        )
        assert len(configs) == 3

    def test_random_mode_is_deterministic(self, tmp_path: Path) -> None:
        """Same seed → same set of generated file stems."""
        base = self._write_base(tmp_path)
        axes = [
            AblationAxisSchema(
                config_path="objectives.reconstruction.lambda_l1",
                values=[0.1, 0.5, 1.0, 2.0, 5.0],
            ),
        ]
        kwargs = dict(mode="random", n_random_samples=2, seed=42)
        run1 = AblationConfigGenerator.generate(
            str(base), axes, str(tmp_path / "r1"), **kwargs
        )
        run2 = AblationConfigGenerator.generate(
            str(base), axes, str(tmp_path / "r2"), **kwargs
        )
        stems1 = sorted(Path(p).stem.split("_ablation_")[1] for p in run1)
        stems2 = sorted(Path(p).stem.split("_ablation_")[1] for p in run2)
        assert stems1 == stems2

    def test_missing_base_config_raises(self, tmp_path: Path) -> None:
        axes = [
            AblationAxisSchema(
                config_path="model.x",
                values=[1, 2],
            ),
        ]
        with pytest.raises(FileNotFoundError):
            AblationConfigGenerator.generate(
                str(tmp_path / "not_there.yaml"),
                axes,
                str(tmp_path / "out"),
            )

    def test_invalid_axis_path_raises(self, tmp_path: Path) -> None:
        base = self._write_base(tmp_path)
        axes = [AblationAxisSchema(config_path="no.such.key", values=[1, 2])]
        with pytest.raises(KeyError):
            AblationConfigGenerator.generate(str(base), axes, str(tmp_path / "out"))

    def test_invalid_mode_raises(self, tmp_path: Path) -> None:
        base = self._write_base(tmp_path)
        axes = [
            AblationAxisSchema(
                config_path="objectives.reconstruction.lambda_l1",
                values=[0.5],
            ),
        ]
        with pytest.raises(ValueError, match=r"(?i)unknown mode"):
            AblationConfigGenerator.generate(
                str(base), axes, str(tmp_path / "out"), mode="invalid_mode"
            )

    def test_summarise_sweep_total_count(self) -> None:
        axes = [
            AblationAxisSchema(config_path="a.b", values=[1, 2, 3]),
            AblationAxisSchema(config_path="c.d", values=[0.1, 0.2]),
        ]
        summary = AblationConfigGenerator.summarise_sweep(axes)
        assert "6" in summary  # 3 × 2

    def test_nested_helpers_get_set(self) -> None:
        """_get_nested / _set_nested work on deeply nested dicts."""
        cfg = {"a": {"b": {"c": 42}}}
        assert _get_nested(cfg, "a.b.c") == 42
        _set_nested(cfg, "a.b.c", 99)
        assert cfg["a"]["b"]["c"] == 99

    def test_get_nested_missing_raises(self) -> None:
        cfg = {"a": {"b": 1}}
        with pytest.raises(KeyError):
            _get_nested(cfg, "a.no_such.key")

    def test_set_nested_missing_parent_raises(self) -> None:
        cfg = {"a": {}}
        with pytest.raises(KeyError):
            _set_nested(cfg, "a.no_parent.leaf", 5)
