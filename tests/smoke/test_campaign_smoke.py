"""Smoke tests for the Campaign Orchestrator.

Designed to run on the cluster before launching real campaigns.
Validates the full lifecycle: schema → state → SLURM scripts →
evaluator → report generator.

Run with:
    pytest tests/smoke/test_campaign_smoke.py -v --tb=short
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.utils.corpus import tracked_yamls
from tests.utils.repo_scripts import skip_if_public_export

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def project_root() -> Path:
    """Return the project root (where experiments/ lives)."""
    root = Path(__file__).resolve().parents[2]
    assert (root / "experiments").is_dir(), f"Project root not found: {root}"
    return root


@pytest.fixture
def campaign_dir(tmp_path: Path) -> Path:
    """Return a temporary campaign output directory."""
    d = tmp_path / "campaign_output"
    d.mkdir()
    return d


@pytest.fixture
def synthetic_state(campaign_dir: Path):
    """Create a synthetic campaign state with per-sample metrics.

    Simulates 3 completed experiments with NPZ metric files.
    """
    from spectramr.infrastructure.orchestration.campaign_state import (
        CampaignState,
        ExperimentStatus,
    )

    rng = np.random.default_rng(42)

    state = CampaignState(
        campaign_name="smoke_test",
        campaign_dir=str(campaign_dir),
    )

    for name, role, psnr_offset in [
        ("baseline_model", "baseline", 0.0),
        ("better_model", "variant", 2.5),
        ("worse_model", "variant", -1.0),
    ]:
        exp_dir = campaign_dir / name
        exp_dir.mkdir()
        npz_path = exp_dir / "per_sample_metrics.npz"

        n_samples = 50
        np.savez(
            npz_path,
            psnr=rng.normal(30.0 + psnr_offset, 1.5, size=n_samples),
            ssim=rng.normal(0.85 + psnr_offset * 0.01, 0.03, size=n_samples).clip(0, 1),
        )

        state.experiments.append(
            ExperimentStatus(
                name=name,
                config_path=f"{name}.yaml",
                role=role,
                status="completed",
                checkpoint_path=str(exp_dir / "best.pt"),
                per_sample_metrics_path=str(npz_path),
                output_dir=str(exp_dir),
            )
        )

    state.save()
    return state


# ── 1. Schema validation ────────────────────────────────────────────


class TestCampaignSchemaValidation:
    """Validate all campaign YAML files parse correctly."""

    CAMPAIGNS = [
        "experiments/campaigns/debug_smoke_test.yaml",
        "experiments/campaigns/kspace_recon_shootout.yaml",
        "experiments/campaigns/ulf_to_hf_field_translation.yaml",
        "experiments/campaigns/reconstruction_architecture_shootout.yaml",
        "experiments/campaigns/fourier_bridge_architectures.yaml",
    ]

    @pytest.mark.parametrize("campaign_path", CAMPAIGNS)
    def test_campaign_parses(self, project_root: Path, campaign_path: str):
        """Each campaign YAML must parse without errors."""
        from spectramr.config.schemas.campaign import CampaignConfigSchema

        full_path = project_root / campaign_path
        if not full_path.exists():
            pytest.skip(f"Campaign not found: {campaign_path}")

        config = CampaignConfigSchema.from_yaml(str(full_path))

        assert config.name, "Campaign name is empty"
        assert len(config.enabled_experiments) >= 1, "No experiments"
        assert config.output_dir, "Output dir not auto-generated"

    @pytest.mark.parametrize("campaign_path", CAMPAIGNS)
    def test_experiment_configs_exist(self, project_root: Path, campaign_path: str):
        """All experiment config paths referenced in campaigns must exist."""
        from spectramr.config.schemas.campaign import CampaignConfigSchema

        full_path = project_root / campaign_path
        if not full_path.exists():
            pytest.skip(f"Campaign not found: {campaign_path}")

        config = CampaignConfigSchema.from_yaml(str(full_path))
        missing = [
            e.config
            for e in config.enabled_experiments
            if not (project_root / e.config).exists()
        ]

        assert not missing, f"Missing configs: {missing}"

    @pytest.mark.parametrize("campaign_path", CAMPAIGNS)
    def test_baseline_exists(self, project_root: Path, campaign_path: str):
        """Each campaign should have at least one baseline experiment."""
        from spectramr.config.schemas.campaign import CampaignConfigSchema

        full_path = project_root / campaign_path
        if not full_path.exists():
            pytest.skip(f"Campaign not found: {campaign_path}")

        config = CampaignConfigSchema.from_yaml(str(full_path))
        baselines = config.baseline_experiments

        assert len(baselines) >= 1, "No baseline experiment defined"


# ── 1b. Every campaign, every reference ──────────────────────────────

#: Campaign refs that point at a tree stripped from tracking
#: (``dd66771dc chore(public): strip maintainer-private folders``) and have no
#: candidate anywhere in ``experiments/``. Recorded rather than waived: a SIXTH
#: dangling ref fails, and re-adding one of these fails the staleness half, so
#: neither direction can drift silently.
KNOWN_DANGLING_CAMPAIGN_REFS: frozenset[tuple[str, str]] = frozenset(
    {
        (
            "bloch_cycle_contrast_translation.yaml",
            "experiments/training/tasks/task_contrast_translation.yaml",
        ),
        (
            "multi_contrast_advanced_shootout.yaml",
            "experiments/training/tasks/task_translation.yaml",
        ),
        (
            "multi_param_mapping_protocols.yaml",
            "experiments/training/tasks/task_parameter_mapping.yaml",
        ),
        (
            "ode_temporal_dynamics.yaml",
            "experiments/training/tasks/task_dynamic_reconstruction.yaml",
        ),
        (
            "vae_latent_representation.yaml",
            "experiments/training/tasks/task_vae_reconstruction.yaml",
        ),
    }
)


def _dangling_campaign_refs(project_root: Path) -> set[tuple[str, str]]:
    """Every ``experiments/**.yaml`` a campaign names that does not exist.

    Deliberately a text scan rather than a schema walk. ``CAMPAIGNS`` above is a
    hand-maintained list of five files, so a new campaign gets no coverage until
    someone remembers to add it, and ``test_experiment_configs_exist`` only
    inspects ``enabled_experiments`` — a DISABLED experiment with a dead path is
    a break that surfaces the moment it is enabled, which is the worst moment to
    find it. Scanning the text also spans both campaign shapes (``experiments:``
    and sequential ``stage_groups[].experiments:``) without either accessor.
    """
    found: set[tuple[str, str]] = set()
    for campaign in tracked_yamls(project_root / "experiments" / "campaigns"):
        refs = set(re.findall(r"experiments/[^\s\"']+\.yaml", campaign.read_text()))
        for ref in sorted(refs):
            if not (project_root / ref).is_file():
                found.add((campaign.name, ref))
    return found


class TestCampaignReferenceIntegrity:
    """A campaign that names a config which does not exist cannot be submitted."""

    def test_the_scan_sees_every_campaign(self, project_root: Path) -> None:
        """Guard the two tests below against passing vacuously."""
        skip_if_public_export("experiments/ does not ship; experiments/campaigns is absent here")
        campaigns = list(tracked_yamls(project_root / "experiments" / "campaigns"))
        assert len(campaigns) > len(TestCampaignSchemaValidation.CAMPAIGNS), (
            "the glob must cover more than the hand-maintained CAMPAIGNS list — "
            "that list missing a campaign is the hole this class exists to close"
        )

    def test_no_new_dangling_refs(self, project_root: Path) -> None:
        new = _dangling_campaign_refs(project_root) - KNOWN_DANGLING_CAMPAIGN_REFS
        assert not new, (
            "campaign(s) reference configs that do not exist:\n  "
            + "\n  ".join(f"{c} -> {r}" for c, r in sorted(new))
            + "\nRepoint them at the arm's real location; a campaign cannot be "
            "submitted with a dead config path."
        )

    def test_ratchet_has_no_stale_entries(self, project_root: Path) -> None:
        skip_if_public_export("experiments/campaigns does not ship, so every known-dangling ref reads as fixed")
        fixed = KNOWN_DANGLING_CAMPAIGN_REFS - _dangling_campaign_refs(project_root)
        assert not fixed, (
            "these refs resolve now — drop them from "
            f"KNOWN_DANGLING_CAMPAIGN_REFS: {sorted(fixed)}"
        )


# ── 2. State persistence ────────────────────────────────────────────


class TestStatePersistence:
    """Test state save/load round-trip."""

    def test_save_load_round_trip(self, synthetic_state, campaign_dir: Path):
        """State must survive JSON serialisation."""
        from spectramr.infrastructure.orchestration.campaign_state import CampaignState

        state_path = campaign_dir / "campaign_state.json"
        assert state_path.exists()

        loaded = CampaignState.load(state_path)
        assert loaded.campaign_name == "smoke_test"
        assert len(loaded.experiments) == 3
        assert loaded.completed[0].name == "baseline_model"

    def test_state_json_is_valid(self, synthetic_state, campaign_dir: Path):
        """State JSON must be valid and contain expected fields."""
        state_path = campaign_dir / "campaign_state.json"
        with open(state_path) as f:
            data = json.load(f)

        assert "campaign_name" in data
        assert "experiments" in data
        assert len(data["experiments"]) == 3


# ── 3. Dry-run submission ───────────────────────────────────────────


class TestDryRunSubmission:
    """Test campaign submission in dry-run mode."""

    def test_dry_run_generates_scripts(self, project_root: Path, tmp_path: Path):
        """Dry-run should create state + experiment dirs without calling sbatch."""
        from spectramr.infrastructure.orchestration.campaign_orchestrator import (
            CampaignOrchestrator,
        )

        campaign_path = project_root / "experiments/campaigns/debug_smoke_test.yaml"
        if not campaign_path.exists():
            pytest.skip("Debug campaign not found")

        orchestrator = CampaignOrchestrator(
            base_dir=str(project_root),
            dry_run=True,
        )

        state = orchestrator.submit_campaign(str(campaign_path))

        assert state.campaign_name == "debug_smoke_test"
        assert len(state.experiments) == 2, f"Expected 2, got {len(state.experiments)}"

        # Dry-run assigns job_id=0
        for exp in state.experiments:
            assert exp.slurm_job_id == 0
            assert exp.status == "submitted"

        # State file should exist
        state_path = Path(state.campaign_dir) / "campaign_state.json"
        assert state_path.exists()

    def test_slurm_script_content(self, project_root: Path):
        """Generated SLURM scripts should contain required directives."""
        from spectramr.infrastructure.orchestration.slurm_backend import SLURMBackend

        script = SLURMBackend.generate_job_script(
            experiment_name="test_exp",
            config_path="experiments/active/dummy_basic.yaml",
            output_dir="/tmp/test_out",
            base_dir=str(project_root),
            slurm_params={
                "partition": "gpu",
                "time": "01:00:00",
                "mem": "32GB",
                "gpus": 1,
            },
        )

        assert "#SBATCH --job-name=test_exp" in script
        assert "#SBATCH --partition=gpu" in script
        assert "#SBATCH --time=01:00:00" in script
        assert "#SBATCH --gpus=1" in script
        # The entry point, not a literal command line. This asserted
        # "python src/main.py train" until job 8000966 — a path that stopped
        # existing at the 2026-05 `src -> src/spectramr` refactor. `slurm_backend`
        # emits two TRAIN_CMD forms (bare python, and torchrun for multi-GPU);
        # both route through the module CLI, which is the part that must hold.
        assert "-m spectramr.cli train --config" in script
        assert "src/main.py" not in script, "retired entry point resurrected"
        assert "dummy_basic.yaml" in script
        # Auto-inference section
        assert "Automated Inference" in script


# ── 4. Ablation config generation ────────────────────────────────────


class TestAblationGeneration:
    """Test ablation config generation from real campaign configs."""

    def test_kspace_campaign_ablation(self, project_root: Path, tmp_path: Path):
        """K-space campaign ablation axis should generate valid configs."""
        from spectramr.config.schemas.campaign import CampaignConfigSchema
        from spectramr.infrastructure.orchestration.ablation_config_generator import (
            AblationConfigGenerator,
        )

        campaign_path = (
            project_root / "experiments/campaigns/kspace_recon_shootout.yaml"
        )
        if not campaign_path.exists():
            pytest.skip("Campaign not found")

        config = CampaignConfigSchema.from_yaml(str(campaign_path))
        if not config.ablation_axes:
            pytest.skip("No ablation axes")

        # Find baseline config
        baseline = config.baseline_experiments[0]
        base_config = project_root / baseline.config

        if not base_config.exists():
            pytest.skip(f"Base config not found: {base_config}")

        out_dir = tmp_path / "ablation"
        configs = AblationConfigGenerator.generate(
            base_config_path=str(base_config),
            axes=config.ablation_axes,
            output_dir=str(out_dir),
        )

        # Should generate 3 configs (values: [5, 10, 20])
        assert len(configs) == 3
        for cfg_path in configs:
            assert Path(cfg_path).exists()
            with open(cfg_path) as f:
                data = yaml.safe_load(f)
            assert data is not None


# ── 5. Evaluator pipeline ───────────────────────────────────────────


class TestEvaluatorPipeline:
    """Test the full evaluation pipeline with synthetic data."""

    def test_full_evaluation(self, synthetic_state, campaign_dir: Path):
        """Full evaluation should produce leaderboard + pairwise + significance."""
        from spectramr.config.schemas.campaign import EvaluationConfigSchema
        from spectramr.infrastructure.orchestration.campaign_evaluator import (
            CampaignEvaluator,
        )

        evaluator = CampaignEvaluator(
            EvaluationConfigSchema(
                metrics=["psnr", "ssim"],
                n_bootstrap=100,
                per_sample_inference=False,
                compute_uncertainty=False,
            )
        )
        report = evaluator.evaluate(synthetic_state)

        # Leaderboard
        assert report.leaderboard is not None
        assert not report.leaderboard.empty
        assert "experiment" in report.leaderboard.columns
        assert "mean" in report.leaderboard.columns

        # Pairwise results
        assert len(report.pairwise_results) > 0

        # Significance matrices
        assert "psnr" in report.significance_matrix
        assert "ssim" in report.significance_matrix

        # Summary
        assert "smoke_test" in report.summary
        assert "Significant" in report.summary

    def test_leaderboard_ranking_correct(self, synthetic_state):
        """Better model (higher PSNR offset) should rank first."""
        from spectramr.config.schemas.campaign import EvaluationConfigSchema
        from spectramr.infrastructure.orchestration.campaign_evaluator import (
            CampaignEvaluator,
        )

        evaluator = CampaignEvaluator(
            EvaluationConfigSchema(
                metrics=["psnr"],
                n_bootstrap=50,
                per_sample_inference=False,
                compute_uncertainty=False,
            )
        )
        report = evaluator.evaluate(synthetic_state)
        psnr_lb = report.leaderboard[report.leaderboard["metric"] == "psnr"]

        assert psnr_lb.iloc[0]["experiment"] == "better_model"
        assert psnr_lb.iloc[-1]["experiment"] == "worse_model"

    def test_significance_detected(self, synthetic_state):
        """With 2.5 dB offset and n=50, t-test should detect significance."""
        from spectramr.config.schemas.campaign import EvaluationConfigSchema
        from spectramr.infrastructure.orchestration.campaign_evaluator import (
            CampaignEvaluator,
        )

        evaluator = CampaignEvaluator(
            EvaluationConfigSchema(
                metrics=["psnr"],
                n_bootstrap=50,
                per_sample_inference=False,
                compute_uncertainty=False,
            )
        )
        report = evaluator.evaluate(synthetic_state)
        significant = [r for r in report.pairwise_results if r.is_significant]

        assert len(significant) > 0, "Expected at least one significant result"

    def test_effect_sizes_computed(self, synthetic_state):
        """Cohen's d should be computed for all pairwise comparisons."""
        from spectramr.config.schemas.campaign import EvaluationConfigSchema
        from spectramr.infrastructure.orchestration.campaign_evaluator import (
            CampaignEvaluator,
        )

        evaluator = CampaignEvaluator(
            EvaluationConfigSchema(
                metrics=["psnr"],
                n_bootstrap=50,
                per_sample_inference=False,
                compute_uncertainty=False,
            )
        )
        report = evaluator.evaluate(synthetic_state)

        for pr in report.pairwise_results:
            assert pr.effect_size.get("cohens_d") is not None
            assert pr.effect_size.get("interpretation") in (
                "negligible",
                "small",
                "medium",
                "large",
            )

    def test_fdr_correction(self, synthetic_state):
        """FDR correction should produce corrected p-values."""
        from spectramr.config.schemas.campaign import EvaluationConfigSchema
        from spectramr.infrastructure.orchestration.campaign_evaluator import (
            CampaignEvaluator,
        )

        evaluator = CampaignEvaluator(
            EvaluationConfigSchema(
                metrics=["psnr"],
                n_bootstrap=50,
                correction_method="fdr",
                per_sample_inference=False,
                compute_uncertainty=False,
            )
        )
        report = evaluator.evaluate(synthetic_state)

        assert "psnr" in report.corrected_p_values
        assert report.corrected_p_values["psnr"]["method"] == "fdr"


# ── 6. Report generation ────────────────────────────────────────────


class TestReportGeneration:
    """Test HTML dashboard and LaTeX table generation."""

    def test_html_dashboard(self, synthetic_state, campaign_dir: Path):
        """Dashboard HTML + plots should be generated."""
        from spectramr.config.schemas.campaign import EvaluationConfigSchema
        from spectramr.infrastructure.orchestration.campaign_evaluator import (
            CampaignEvaluator,
        )
        from spectramr.infrastructure.orchestration.campaign_report_generator import (
            CampaignReportGenerator,
        )

        evaluator = CampaignEvaluator(
            EvaluationConfigSchema(
                metrics=["psnr", "ssim"],
                n_bootstrap=50,
                per_sample_inference=False,
                compute_uncertainty=False,
            )
        )
        report = evaluator.evaluate(synthetic_state)

        gen = CampaignReportGenerator(output_dir=campaign_dir)
        dashboard_path = gen.generate(report)

        assert Path(dashboard_path).exists()
        html = Path(dashboard_path).read_text()

        # Check HTML structure
        assert "smoke_test" in html
        assert "Leaderboard" in html
        assert "plot-card" in html

        # Check plots exist
        plots_dir = campaign_dir / "plots"
        assert plots_dir.is_dir()
        assert (plots_dir / "leaderboard.png").exists()
        assert (plots_dir / "effect_sizes.png").exists()
        assert (plots_dir / "significance_psnr.png").exists()

    def test_latex_table(self, synthetic_state):
        """LaTeX export should produce valid table markup."""
        from spectramr.config.schemas.campaign import EvaluationConfigSchema
        from spectramr.infrastructure.orchestration.campaign_evaluator import (
            CampaignEvaluator,
        )

        evaluator = CampaignEvaluator(
            EvaluationConfigSchema(
                metrics=["psnr", "ssim"],
                n_bootstrap=50,
                per_sample_inference=False,
                compute_uncertainty=False,
            )
        )
        report = evaluator.evaluate(synthetic_state)

        latex = CampaignEvaluator.leaderboard_to_latex(
            report.leaderboard, caption="Smoke Test", label="tab:smoke"
        )

        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex
        assert "PSNR" in latex
        assert "SSIM" in latex
        assert r"\textbf{" in latex  # Best performer should be bold

    def test_report_json_serialisation(self, synthetic_state, campaign_dir: Path):
        """Report should serialise to valid JSON with all sections."""
        from spectramr.config.schemas.campaign import EvaluationConfigSchema
        from spectramr.infrastructure.orchestration.campaign_evaluator import (
            CampaignEvaluator,
        )

        evaluator = CampaignEvaluator(
            EvaluationConfigSchema(
                metrics=["psnr"],
                n_bootstrap=50,
                per_sample_inference=False,
                compute_uncertainty=False,
            )
        )
        report = evaluator.evaluate(synthetic_state)

        json_path = campaign_dir / "report.json"
        report.save(json_path)

        with open(json_path) as f:
            data = json.load(f)

        assert data["campaign_name"] == "smoke_test"
        assert "leaderboard" in data
        assert "pairwise_results" in data
        assert len(data["pairwise_results"]) > 0
        assert "significance_matrix_psnr" in data


# ── 7. Parse elapsed time ───────────────────────────────────────────


class TestParseElapsed:
    """Test SLURM elapsed time parsing."""

    @pytest.mark.parametrize(
        "elapsed, expected",
        [
            ("01:30:45", 5445.0),
            ("00:05:00", 300.0),
            ("1-12:00:00", 129600.0),
            ("05:30", 330.0),
            ("2-00:00:00", 172800.0),
        ],
    )
    def test_parse_elapsed(self, elapsed: str, expected: float):
        from spectramr.infrastructure.orchestration.campaign_orchestrator import (
            _parse_elapsed,
        )

        result = _parse_elapsed(elapsed)
        assert result == pytest.approx(expected)

    def test_parse_elapsed_invalid(self):
        from spectramr.infrastructure.orchestration.campaign_orchestrator import (
            _parse_elapsed,
        )

        assert _parse_elapsed("invalid") is None
        assert _parse_elapsed("") is None
