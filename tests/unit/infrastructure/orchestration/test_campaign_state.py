"""Unit tests for :mod:`spectramr.infrastructure.orchestration.campaign_state`.

Covers the persistent state objects ``ExperimentStatus`` and
``CampaignState`` end-to-end: round-trip serialisation, the property
helpers used by the orchestrator, mutation via ``update_experiment``
and the human-readable ``summary_table``.

These run on CPU with only ``pydantic`` + ``pytest`` available, so the
module can sit in the default test lane.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spectramr.infrastructure.orchestration.campaign_state import (
    CampaignState,
    ExperimentStatus,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _make_state(tmp_path: Path, *experiments: ExperimentStatus) -> CampaignState:
    return CampaignState(
        campaign_name="unit_test_campaign",
        campaign_dir=str(tmp_path),
        experiments=list(experiments),
    )


def _exp(name: str, **kwargs: object) -> ExperimentStatus:
    return ExperimentStatus(
        name=name,
        config_path=f"experiments/{name}.yaml",
        **kwargs,  # type: ignore[arg-type]
    )


# ── ExperimentStatus ────────────────────────────────────────────────


class TestExperimentStatus:
    def test_defaults_are_pending_and_non_terminal(self) -> None:
        e = _exp("a")
        assert e.status == "pending"
        assert e.is_terminal is False
        assert e.is_evaluable is False
        assert e.slurm_job_id is None
        assert e.best_metrics == {}
        assert e.tags == {}
        assert e.depends_on_jobs == []

    def test_terminal_states_are_recognised(self) -> None:
        for status in ("completed", "failed", "cancelled", "timeout"):
            e = _exp("a", status=status)  # type: ignore[arg-type]
            assert e.is_terminal, f"{status} must be terminal"

    def test_non_terminal_states(self) -> None:
        for status in ("pending", "submitted", "running"):
            e = _exp("a", status=status)  # type: ignore[arg-type]
            assert not e.is_terminal, f"{status} must not be terminal"

    def test_is_evaluable_only_when_completed_with_checkpoint(self) -> None:
        e = _exp("a", status="completed")
        # Missing checkpoint → not evaluable.
        assert not e.is_evaluable

        e2 = _exp("a", status="completed", checkpoint_path="/tmp/ckpt.pt")
        assert e2.is_evaluable

        e3 = _exp("a", status="failed", checkpoint_path="/tmp/ckpt.pt")
        # Failed with a checkpoint is still not evaluable.
        assert not e3.is_evaluable

    def test_extra_allow_preserves_unknown_fields(self) -> None:
        """``model_config={"extra": "allow"}`` keeps forward-compatible fields."""
        e = ExperimentStatus.model_validate({
            "name": "a",
            "config_path": "x.yaml",
            "future_field": 42,
        })
        # Field is preserved on the instance.
        assert getattr(e, "future_field") == 42


# ── CampaignState — properties ──────────────────────────────────────


class TestCampaignStateProperties:
    def test_total_zero_for_empty_campaign(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        assert state.total == 0
        assert state.progress_fraction == 0.0
        assert state.all_terminal is True

    def test_progress_fraction_counts_terminal(self, tmp_path: Path) -> None:
        state = _make_state(
            tmp_path,
            _exp("a", status="completed"),
            _exp("b", status="failed"),
            _exp("c", status="running"),
            _exp("d", status="pending"),
        )
        assert state.total == 4
        # 2 terminal of 4.
        assert state.progress_fraction == 0.5
        assert state.all_terminal is False

    def test_status_buckets(self, tmp_path: Path) -> None:
        state = _make_state(
            tmp_path,
            _exp("p", status="pending"),
            _exp("s", status="submitted"),
            _exp("r", status="running"),
            _exp("c", status="completed", checkpoint_path="/x.pt"),
            _exp("f", status="failed"),
            _exp("t", status="timeout"),
        )
        assert [e.name for e in state.pending] == ["p"]
        assert [e.name for e in state.submitted] == ["s"]
        assert [e.name for e in state.running] == ["r"]
        assert [e.name for e in state.completed] == ["c"]
        # Failed bucket includes timeout per orchestrator contract.
        assert {e.name for e in state.failed} == {"f", "t"}
        assert [e.name for e in state.evaluable] == ["c"]

    def test_all_terminal_true_when_every_arm_is_terminal(self, tmp_path: Path) -> None:
        state = _make_state(
            tmp_path,
            _exp("a", status="completed"),
            _exp("b", status="failed"),
            _exp("c", status="cancelled"),
        )
        assert state.all_terminal is True


# ── CampaignState — lookups ─────────────────────────────────────────


class TestCampaignStateLookups:
    def test_get_experiment_by_name(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path, _exp("a"), _exp("b"))
        assert state.get_experiment("b") is not None
        assert state.get_experiment("b").name == "b"  # type: ignore[union-attr]
        assert state.get_experiment("missing") is None

    def test_get_by_job_id(self, tmp_path: Path) -> None:
        state = _make_state(
            tmp_path,
            _exp("a", slurm_job_id=1001),
            _exp("b", slurm_job_id=1002),
        )
        assert state.get_by_job_id(1001).name == "a"  # type: ignore[union-attr]
        assert state.get_by_job_id(9999) is None

    def test_get_by_stage_group(self, tmp_path: Path) -> None:
        state = _make_state(
            tmp_path,
            _exp("a", stage_group="warm"),
            _exp("b", stage_group="warm"),
            _exp("c", stage_group="cold"),
        )
        warm = state.get_by_stage_group("warm")
        assert {e.name for e in warm} == {"a", "b"}
        assert state.get_by_stage_group("does_not_exist") == []


# ── CampaignState — mutation ────────────────────────────────────────


class TestCampaignStateMutation:
    def test_update_experiment_writes_fields_and_bumps_timestamp(
        self, tmp_path: Path
    ) -> None:
        state = _make_state(tmp_path, _exp("a"))
        prior_update = state.last_updated
        state.update_experiment(
            "a", status="completed", checkpoint_path="/tmp/x.pt", exit_code=0
        )

        e = state.get_experiment("a")
        assert e is not None
        assert e.status == "completed"
        assert e.checkpoint_path == "/tmp/x.pt"
        assert e.exit_code == 0
        # Bumped — string comparison is monotonic for isoformat with tz.
        assert state.last_updated >= prior_update

    def test_update_experiment_missing_raises(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path, _exp("a"))
        with pytest.raises(KeyError, match="missing"):
            state.update_experiment("missing", status="running")


# ── Persistence (save / load round-trip) ────────────────────────────


class TestCampaignStatePersistence:
    def test_save_returns_default_path(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path, _exp("a"))
        out = state.save()
        assert out == tmp_path / "campaign_state.json"
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["campaign_name"] == "unit_test_campaign"
        assert len(loaded["experiments"]) == 1

    def test_save_to_custom_path(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        custom = tmp_path / "sub" / "state.json"
        out = state.save(custom)
        assert out == custom
        assert custom.exists()  # parent dir created

    def test_round_trip_preserves_fields(self, tmp_path: Path) -> None:
        original = _make_state(
            tmp_path,
            _exp(
                "a",
                slurm_job_id=42,
                status="completed",
                checkpoint_path="/ckpt.pt",
                best_metrics={"psnr": 30.5, "ssim": 0.95},
                tags={"phase": "p2"},
                stage_group="warm",
                depends_on_jobs=[1, 2, 3],
            ),
        )
        original.description = "round-trip test"

        path = original.save()
        loaded = CampaignState.load(path)

        assert loaded.campaign_name == original.campaign_name
        assert loaded.description == "round-trip test"
        assert len(loaded.experiments) == 1
        e = loaded.experiments[0]
        assert e.slurm_job_id == 42
        assert e.best_metrics == {"psnr": 30.5, "ssim": 0.95}
        assert e.tags == {"phase": "p2"}
        assert e.depends_on_jobs == [1, 2, 3]
        assert e.stage_group == "warm"
        assert e.is_evaluable is True

    def test_load_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="state not found"):
            CampaignState.load(tmp_path / "missing.json")


# ── Summary table ───────────────────────────────────────────────────


class TestSummaryTable:
    def test_summary_table_contains_campaign_name_and_progress(
        self, tmp_path: Path
    ) -> None:
        state = _make_state(
            tmp_path,
            _exp("alpha", status="completed", best_metrics={"psnr": 32.10}),
            _exp("beta", status="failed"),
            _exp("gamma", status="running", slurm_job_id=12345),
        )
        out = state.summary_table()

        assert "unit_test_campaign" in out
        assert "alpha" in out
        assert "beta" in out
        assert "gamma" in out
        # PSNR rendered to two decimals.
        assert "32.10" in out
        # Job id rendered.
        assert "12345" in out
        # Progress fraction line present.
        assert "Progress" in out

    def test_summary_table_renders_each_status_icon_present(
        self, tmp_path: Path
    ) -> None:
        state = _make_state(
            tmp_path,
            _exp("p", status="pending"),
            _exp("s", status="submitted"),
            _exp("r", status="running"),
            _exp("c", status="completed"),
            _exp("f", status="failed"),
            _exp("x", status="cancelled"),
            _exp("t", status="timeout"),
        )
        out = state.summary_table()
        # The literal status names are rendered after the icon.
        for status in (
            "pending",
            "submitted",
            "running",
            "completed",
            "failed",
            "cancelled",
            "timeout",
        ):
            assert status in out


class TestAtomicSave:
    def test_save_failure_does_not_corrupt_existing_state(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A crash mid-serialisation must NOT truncate the previously-good
        ``campaign_state.json``. The old ``open(path, "w")`` truncated the target
        before writing, so a failing ``json.dump`` left an empty/partial file and
        the campaign's entire history was lost. Atomic write (temp + os.replace)
        leaves the prior file intact."""
        state = _make_state(tmp_path, _exp("a", status="completed"))
        path = state.save()
        good = path.read_text()
        assert good  # a real, complete JSON document

        def _boom(*_a, **_k):
            raise RuntimeError("disk full mid-write")

        monkeypatch.setattr(
            "spectramr.infrastructure.orchestration.campaign_state.json.dump", _boom
        )
        with pytest.raises(RuntimeError):
            state.save()

        # The original file is untouched and still loads.
        assert path.read_text() == good
        reloaded = CampaignState.load(path)
        assert reloaded.get_experiment("a").status == "completed"
        # No stray temp file is left behind.
        assert not list(tmp_path.glob("*.tmp"))


# ── array_task_id (execution: array — one job id, per-arm task index) ─────────


class TestArrayTaskId:
    def test_defaults_to_none(self) -> None:
        # A non-array arm leaves array_task_id unset.
        assert _exp("a").array_task_id is None

    def test_round_trips_through_save_load(self, tmp_path: Path) -> None:
        # In array mode every arm shares ONE array job id but has a distinct
        # task index; both must survive serialisation so `campaign status` /
        # evaluate can map a sacct `<jobid>_<idx>` element back to its arm.
        state = _make_state(
            tmp_path,
            _exp("a", slurm_job_id=9001, array_task_id=0),
            _exp("b", slurm_job_id=9001, array_task_id=1),
        )
        path = state.save()
        reloaded = CampaignState.load(path)
        assert reloaded.get_experiment("a").array_task_id == 0
        assert reloaded.get_experiment("b").array_task_id == 1
        # Same array job id shared by both arms.
        assert {e.slurm_job_id for e in reloaded.experiments} == {9001}


# ── resume_mode provenance (#2542) ──────────────────────────────────


class TestResumeMode:
    def test_round_trips_through_save_and_load(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        state.resume_mode = "fresh"
        path = state.save()
        assert CampaignState.load(path).resume_mode == "fresh"

    def test_a_pre_field_state_loads_as_empty(self, tmp_path: Path) -> None:
        path = _make_state(tmp_path).save()
        data = json.loads(Path(path).read_text())
        data.pop("resume_mode", None)
        Path(path).write_text(json.dumps(data))
        assert CampaignState.load(path).resume_mode == ""

    def test_an_unknown_mode_is_refused(self, tmp_path: Path) -> None:
        """Planted: a misspelt mode must not be recorded as provenance."""
        with pytest.raises(ValueError):
            CampaignState(campaign_name="c", campaign_dir=str(tmp_path), resume_mode="fersh")
