"""Tests for ``ReportingSettings``.

Targets ``spectramr.config.schemas.reporting``. v6.0 reporting-pipeline
configuration: enable/disable, task preset, optional figure / table /
metric / cohort overrides, fail-on-error policy.

Categories:

- Defaults: disabled, default task = ``"default"``, ``fail_on_error=False``
- ``extra="forbid"``: unknown keys rejected
- All optional override fields default to ``None``
- Custom overrides round-trip cleanly
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.enums import ReportStyle, ReportTask
from spectramr.config.schemas.reporting import ReportingSettings

# ---------------------------------------------------------------------------
# Enums (Task 2.1)
# ---------------------------------------------------------------------------


def test_report_task_values():
    assert ReportTask("reconstruction") == ReportTask.RECONSTRUCTION
    assert {e.value for e in ReportTask} >= {
        "default",
        "reconstruction",
        "synthesis",
        "super_resolution",
        "gan",
        "diffusion",
        "vae",
    }


def test_report_style_values():
    assert ReportStyle("nature") == ReportStyle.NATURE
    assert ReportStyle("ieee") == ReportStyle.IEEE


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_disabled() -> None:
    """Reporting is disabled by default."""
    cfg = ReportingSettings()
    assert cfg.enabled is False
    assert cfg.task == "default"
    assert cfg.out_subdir == "report"
    assert cfg.fail_on_error is False


def test_optional_fields_default_to_none() -> None:
    """Override fields are ``None`` until explicitly set."""
    cfg = ReportingSettings()
    assert cfg.method_name is None
    assert cfg.figures is None
    assert cfg.tables is None
    assert cfg.metrics is None
    assert cfg.cohort is None
    assert cfg.hyperparameters is None
    assert cfg.extra_runs is None


# ---------------------------------------------------------------------------
# Custom overrides round-trip
# ---------------------------------------------------------------------------


def test_full_custom_config() -> None:
    """All override fields round-trip cleanly."""
    cfg = ReportingSettings(
        enabled=True,
        task="reconstruction",
        method_name="my_method",
        out_subdir="custom_report",
        figures=["fig_1", "fig_2"],
        tables=["tab_1"],
        metrics=["psnr", "ssim", "vif"],
        cohort={"site": "A"},
        hyperparameters=[{"name": "lr", "value": 1e-4}],
        extra_runs=["/path/to/baseline"],
        fail_on_error=True,
    )
    assert cfg.enabled is True
    assert cfg.task == "reconstruction"
    assert cfg.method_name == "my_method"
    assert cfg.figures == ["fig_1", "fig_2"]
    assert cfg.tables == ["tab_1"]
    assert cfg.metrics == ["psnr", "ssim", "vif"]
    assert cfg.cohort == {"site": "A"}
    assert cfg.hyperparameters == [{"name": "lr", "value": 1e-4}]
    assert cfg.extra_runs == ["/path/to/baseline"]
    assert cfg.fail_on_error is True


# ---------------------------------------------------------------------------
# extra="forbid"
# ---------------------------------------------------------------------------


def test_unknown_keys_rejected() -> None:
    """Unknown YAML keys → ``ValidationError`` (``extra='forbid'``)."""
    with pytest.raises(ValidationError):
        ReportingSettings(unknown_key="x")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Style / format / case knobs (Task 2.2)
# ---------------------------------------------------------------------------


def test_defaults_are_nature_and_pdf_png():
    s = ReportingSettings(enabled=True)
    assert s.style == ReportStyle.NATURE
    assert s.formats == ["pdf", "png"]
    assert s.dpi == 600
    assert s.case_selection == "best_median_worst"
    assert s.n_report_cases == 6


def test_unknown_task_raises():
    with pytest.raises(ValidationError):
        ReportingSettings(task="not_a_task")


def test_unknown_format_raises():
    with pytest.raises(ValidationError):
        ReportingSettings(formats=["pdf", "bmp"])


def test_extra_key_forbidden():
    with pytest.raises(ValidationError):
        ReportingSettings(unexpected_key=1)


def test_empty_formats_raises():
    with pytest.raises(ValidationError):
        ReportingSettings(formats=[])


# ---------------------------------------------------------------------------
# tikz knob (2026-07-01 reporting upgrade)
# ---------------------------------------------------------------------------


def test_tikz_defaults_off():
    assert ReportingSettings().tikz is False


def test_tikz_round_trips():
    assert ReportingSettings(tikz=True).tikz is True


def test_tikz_rejects_non_bool():
    with pytest.raises(ValidationError):
        ReportingSettings(tikz="sometimes")


# ---------------------------------------------------------------------------
# QC report knobs
# ---------------------------------------------------------------------------


def test_qc_knobs_default_on():
    """QC figures / per-case CSV / HTML default on (fire only when enabled)."""
    s = ReportingSettings()
    assert s.qc_figures is True
    assert s.per_call_metrics is True
    assert s.html_report is True


def test_qc_knobs_round_trip():
    s = ReportingSettings(qc_figures=False, per_call_metrics=False, html_report=False)
    assert s.qc_figures is False
    assert s.per_call_metrics is False
    assert s.html_report is False


@pytest.mark.parametrize("field", ["qc_figures", "per_call_metrics", "html_report"])
def test_qc_knobs_reject_non_bool(field):
    with pytest.raises(ValidationError):
        ReportingSettings(**{field: "maybe"})


class TestPerCaseMetricsDescribesItselfHonestly:
    """The description is the only place this artifact's contract is stated.

    It used to claim "one row per validation-case observation … one point per
    case". Both clauses are false: a row is one validation CALL carrying the
    BATCH-AGGREGATE metrics, under a ``case_id`` derived from the step. That
    overpromise is what let ``qc_group_strip`` render eight copies of one scalar
    as a distribution (#503).

    Pinned because a description has no other guard — nothing executes it, so
    the next tidy-up would restore the false claim silently.
    """

    @staticmethod
    def _description() -> str:
        return ReportingSettings.model_fields["per_call_metrics"].description or ""

    def test_it_says_the_row_is_a_call_not_a_sample(self):
        desc = self._description()
        assert "CALL" in desc
        assert "BATCH-AGGREGATE" in desc

    def test_it_no_longer_promises_a_point_per_case(self):
        """The exact phrase that misled the plotter's consumers."""
        assert "one point per case" not in self._description()


# ---------------------------------------------------------------------------
# Interactive-layer knobs (2026-07 plotly upgrade)
# ---------------------------------------------------------------------------


def test_interactive_defaults_on_volumes_off():
    s = ReportingSettings()
    assert s.interactive is True
    assert s.record_volumes is False


def test_interactive_knobs_round_trip():
    s = ReportingSettings(interactive=False, record_volumes=True)
    assert s.interactive is False
    assert s.record_volumes is True


@pytest.mark.parametrize("field", ["interactive", "record_volumes"])
def test_interactive_knobs_reject_non_bool(field):
    with pytest.raises(ValidationError):
        ReportingSettings(**{field: "maybe"})


class TestPerCaseMetricsWasRenamed:
    """`per_case_metrics` -> `per_call_metrics` (2026-08-05).

    The old name promised a per-CASE distribution the artifact never carried:
    the sink writes one row per validation CALL holding that call's
    BATCH-AGGREGATE metrics, because the run computes metrics batch-wise and no
    per-sample value exists to record. `qc_group_strip` believed the name and
    rendered eight copies of one scalar as a box-and-whisker (#503).

    Renaming was the choice over emitting genuine per-sample rows, which would
    require per-sample metric computation the validation path does not do.
    """

    def test_the_honest_name_is_the_field(self):
        assert "per_call_metrics" in ReportingSettings.model_fields
        assert ReportingSettings().per_call_metrics is True

    def test_the_old_spelling_is_refused_by_name_not_just_refused(self):
        """Refusing is half the job; saying what replaced it is the other half.

        A silently-dropped knob is pitfall #15: the arm would believe it had
        disabled the artifact and the artifact would keep being written.
        ``extra="forbid"`` alone prevents that, and it is what this block had
        until 2026-08-22 -- but its message is pydantic's generic "Extra inputs
        are not permitted", which names neither the replacement nor the fixer.
        The record's guided text existed from 2026-08-05 and reached nobody,
        because nothing mounted ``reject_renamed_keys("reporting")``.

        Asserting only ``pytest.raises(ValidationError)`` cannot tell the two
        states apart -- both raise -- which is why that is not what this
        asserts. The mount is kept honest across every block by
        ``test_rename_mounts.py``.
        """
        with pytest.raises(ValidationError) as excinfo:
            ReportingSettings(per_case_metrics=False)

        message = str(excinfo.value)
        assert "per_call_metrics" in message, "name the replacement"
        assert "#503" in message, "name why the rename happened"
        assert "migrate_config_keys.py" in message, "name the fixer"

    def test_the_rename_is_registered_in_the_ssot(self):
        """RENAMES is where a reader looks up a retired spelling."""
        from spectramr.config.schemas.renames import RENAMES

        record = RENAMES["reporting.per_case_metrics"]
        assert record.canonical == "reporting.per_call_metrics"
        # `raise`, not `fold`: zero corpus arms declared it, so there is no
        # migration to stage.
        assert record.posture == "raise"

    def test_the_description_still_says_what_a_row_is(self):
        """The rename must not become the whole explanation -- a reader also
        needs to know a row is a CALL carrying BATCH-AGGREGATE values."""
        desc = ReportingSettings.model_fields["per_call_metrics"].description or ""
        assert "CALL" in desc
        assert "BATCH-AGGREGATE" in desc
        assert "per_case_metrics" in desc, "say what it was renamed from"


def test_a_retired_figure_id_raises_at_load_with_its_replacement():
    import pytest

    from spectramr.config.schemas.reporting import ReportingSettings

    with pytest.raises(ValueError, match="mri_a7_kspace_recon -> mri_recon_panel"):
        ReportingSettings(figures=["mri_a7_kspace_recon"])
    assert ReportingSettings(figures=["mri_recon_panel"]).figures == ["mri_recon_panel"]
