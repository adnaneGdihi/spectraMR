"""Unit tests for Sim2Rank scoring module.

Tests ADR, SCVR, Isotonic calibration, Jacobian sensitivity, and composite
scoring on synthetic trajectories with known ground-truth properties.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ──────────────────────────────────────────────────────────────────────
# ADR Tests
# ──────────────────────────────────────────────────────────────────────


class TestADR:
    """Test Asymptotic Discriminability Ranking."""

    def test_perfect_monotonic_trajectory(self):
        """A perfectly monotonic trajectory should have ADR ≈ 1.0."""
        pytest.importorskip("scripts.sim2rank.scoring")  # not in the public export
        from scripts.sim2rank.scoring import compute_adr

        # Higher-is-better metric that decays monotonically
        traj = np.linspace(100.0, 10.0, 20)  # 100 → 10 (decreasing)
        adr = compute_adr(traj, higher_is_better=True, epsilon=0.01)
        assert adr > 0.90, f"Perfect monotonic ADR should be > 0.90, got {adr:.4f}"

    def test_constant_trajectory_is_zero(self):
        """A constant trajectory (useless metric) should have ADR = 0."""
        pytest.importorskip("scripts.sim2rank.scoring")  # not in the public export
        from scripts.sim2rank.scoring import compute_adr

        traj = np.full(20, 42.0)
        adr = compute_adr(traj, higher_is_better=True)
        assert adr == 0.0, f"Constant trajectory should have ADR = 0, got {adr:.4f}"

    def test_saturated_trajectory_penalised(self):
        """A trajectory that saturates early should be penalised."""
        pytest.importorskip("scripts.sim2rank.scoring")  # not in the public export
        from scripts.sim2rank.scoring import compute_adr

        # Decays for first 5 steps, then saturates
        traj = np.concatenate([np.linspace(100, 50, 5), np.full(15, 50.0)])
        adr = compute_adr(traj, higher_is_better=True, epsilon=0.01)

        # Compare with a non-saturating trajectory
        traj_good = np.linspace(100.0, 10.0, 20)
        adr_good = compute_adr(traj_good, higher_is_better=True, epsilon=0.01)

        assert adr < adr_good, (
            f"Saturated ADR ({adr:.4f}) should be < non-saturated ({adr_good:.4f})"
        )

    def test_reversed_metric_polarity(self):
        """A lower-is-better metric (e.g., MSE) should work correctly."""
        pytest.importorskip("scripts.sim2rank.scoring")  # not in the public export
        from scripts.sim2rank.scoring import compute_adr

        # MSE increases with degradation
        traj = np.linspace(0.01, 1.0, 20)
        adr = compute_adr(traj, higher_is_better=False, epsilon=0.01)
        assert adr > 0.90, f"Increasing lower-is-better ADR should be > 0.90, got {adr:.4f}"

    def test_batch_computation(self):
        """Batch ADR should match individual computations."""
        pytest.importorskip("scripts.sim2rank.scoring")  # not in the public export
        from scripts.sim2rank.scoring import compute_adr, compute_adr_batch

        trajs = np.array([
            np.linspace(100, 10, 20),
            np.linspace(0.01, 1.0, 20),
            np.full(20, 42.0),
        ])
        flags = [True, False, True]

        batch_scores = compute_adr_batch(trajs, flags)
        individual = [compute_adr(trajs[i], flags[i]) for i in range(3)]

        np.testing.assert_allclose(batch_scores, individual, rtol=1e-10)


# ──────────────────────────────────────────────────────────────────────
# SCVR Tests
# ──────────────────────────────────────────────────────────────────────


class TestSCVR:
    """Test Signal-to-Content Variance Ratio."""

    def test_high_signal_low_content_variance(self):
        """A good metric should have high SCVR (signal >> content variance)."""
        pytest.importorskip("scripts.sim2rank.scoring")  # not in the public export
        from scripts.sim2rank.scoring import compute_scvr_ranking

        M, S, C, T = 3, 4, 2, 10

        # Metric 0: strong degradation signal, weak content variation
        data = torch.zeros(M, S, C, T)
        for t in range(T):
            data[0, :, :, t] = t * 0.1 + torch.randn(S, C) * 0.01  # Strong signal

        # Metric 1: weak signal, strong content variation
        for t in range(T):
            data[1, :, :, t] = 0.5 + torch.randn(S, C) * 0.5  # Mostly content noise

        # Metric 2: moderate signal, moderate content
        for t in range(T):
            data[2, :, :, t] = t * 0.05 + torch.randn(S, C) * 0.1

        scores, ranks = compute_scvr_ranking(data)

        # `ranks` is now 1-indexed MID-ranks (never an argsort index vector, which
        # resolves ties by array position, #243). Metric 0 is best -> rank 1.
        assert ranks[0] == 1.0, f"Best SCVR should hold rank 1, got {ranks[0]}"
        assert scores[0] > scores[1], "Signal-dominated should have higher SCVR"

    def test_scvr_shape(self):
        """SCVR output shapes should match input metric count."""
        pytest.importorskip("scripts.sim2rank.scoring")  # not in the public export
        from scripts.sim2rank.scoring import compute_scvr_ranking

        M, S, C, T = 5, 3, 2, 8
        data = torch.randn(M, S, C, T)
        scores, ranks = compute_scvr_ranking(data)

        assert scores.shape == (M,)
        assert ranks.shape == (M,)


# ──────────────────────────────────────────────────────────────────────
# Isotonic Calibration Tests
# ──────────────────────────────────────────────────────────────────────


class TestIsotonicCalibration:
    """Test isotonic regression calibration."""

    def test_calibrated_range(self):
        """Calibrated outputs should be in [0, 1]."""
        pytest.importorskip("scripts.sim2rank.scoring")  # not in the public export
        from scripts.sim2rank.scoring import isotonic_calibrate

        M, T = 5, 20
        raw = np.random.randn(M, T)
        alpha = np.linspace(0.05, 1.0, T)
        flags = [True, False, True, False, True]

        calibrated = isotonic_calibrate(raw, alpha, flags)

        assert calibrated.shape == (M, T)
        assert np.all(calibrated >= -0.01), f"Min calibrated: {calibrated.min()}"
        assert np.all(calibrated <= 1.01), f"Max calibrated: {calibrated.max()}"

    def test_monotonic_preservation(self):
        """A monotonic input should produce monotonic calibrated output."""
        pytest.importorskip("scripts.sim2rank.scoring")  # not in the public export
        from scripts.sim2rank.scoring import isotonic_calibrate

        traj = np.linspace(100, 10, 20).reshape(1, -1)  # Decreasing
        alpha = np.linspace(0.05, 1.0, 20)

        calibrated = isotonic_calibrate(traj, alpha, [True])

        # After calibration (with flip for higher-is-better), should be monotonic
        diffs = np.diff(calibrated[0])
        assert np.all(diffs >= -1e-6), "Calibrated trajectory should be monotonic"


# ──────────────────────────────────────────────────────────────────────
# Jacobian Sensitivity Tests
# ──────────────────────────────────────────────────────────────────────


class TestJacobian:
    """Test Jacobian sensitivity computation."""

    def test_axis_specific_metric(self):
        """A metric sensitive to one axis covers fewer axes than a uniform responder."""
        pytest.importorskip("scripts.sim2rank.scoring")  # not in the public export
        from scripts.sim2rank.scoring import compute_jacobian_sensitivity

        M, T = 3, 20

        axis_trajectories = {
            "noise": np.zeros((M, T)),
            "motion": np.zeros((M, T)),
            "undersampling": np.zeros((M, T)),
        }

        # Metric 0: only responds to noise
        axis_trajectories["noise"][0] = np.linspace(0, 1, T)
        axis_trajectories["motion"][0] = np.full(T, 0.5)
        axis_trajectories["undersampling"][0] = np.full(T, 0.5)

        # Metric 1: responds to all axes equally
        axis_trajectories["noise"][1] = np.linspace(0, 1, T)
        axis_trajectories["motion"][1] = np.linspace(0, 1, T)
        axis_trajectories["undersampling"][1] = np.linspace(0, 1, T)

        # Metric 2: responds to noise + motion, not undersampling
        axis_trajectories["noise"][2] = np.linspace(0, 1, T)
        axis_trajectories["motion"][2] = np.linspace(0, 0.5, T)
        axis_trajectories["undersampling"][2] = np.full(T, 0.3)

        from scripts.sim2rank.scoring import compute_adr_batch

        responds = np.stack(
            [compute_adr_batch(v, [False] * M) > 0 for v in axis_trajectories.values()], 1
        )
        jacobian, coverage = compute_jacobian_sensitivity(axis_trajectories, responds=responds)

        assert jacobian.shape == (M, 3)
        np.testing.assert_allclose(coverage, [1 / 3, 1.0, 2 / 3])

    def test_jacobian_non_negative(self):
        """Jacobian entries should be non-negative (absolute gradients)."""
        pytest.importorskip("scripts.sim2rank.scoring")  # not in the public export
        from scripts.sim2rank.scoring import compute_jacobian_sensitivity

        M, T = 5, 15
        axis_trajectories = {
            "noise": np.random.randn(M, T),
            "motion": np.random.randn(M, T),
        }

        jacobian, _ = compute_jacobian_sensitivity(axis_trajectories)
        assert np.all(jacobian >= 0), "Jacobian should be non-negative"


# ──────────────────────────────────────────────────────────────────────
# Composite Score Tests
# ──────────────────────────────────────────────────────────────────────


class TestMetricsList:
    """Test the complete metric specification list."""

    def test_count_matches_registry(self):
        """N_METRICS tracks the registered metric set.

        The exact count drifts as the metric registry grows (93 → 107
        in commit c919fa4c9, "extend METRIC_SPECS to cover full
        registry"). Pin to ≥ 93 to catch accidental removals without
        breaking on legitimate additions.
        """
        pytest.importorskip("scripts.sim2rank.metrics_list")  # not in the public export
        from scripts.sim2rank.metrics_list import N_METRICS
        assert N_METRICS >= 93, (
            f"Expected at least 93 metrics; got {N_METRICS}. "
            "Did METRIC_SPECS lose entries?"
        )

    def test_unique_registry_keys(self):
        """All registry keys must be unique."""
        pytest.importorskip("scripts.sim2rank.metrics_list")  # not in the public export
        from scripts.sim2rank.metrics_list import METRIC_KEYS
        assert len(METRIC_KEYS) == len(set(METRIC_KEYS))

    def test_per_image_summary_domain_partition(self):
        """Per-image + summary + domain + report-only signed should equal total."""
        pytest.importorskip("scripts.sim2rank.metrics_list")  # not in the public export
        from scripts.sim2rank.metrics_list import (
            DOMAIN_SPECS,
            METRIC_SPECS,
            PER_IMAGE_SPECS,
            SIGNED_AGREEMENT_KEYS,
            SUMMARY_SPECS,
        )
        n_split = len(PER_IMAGE_SPECS) + len(SUMMARY_SPECS) + len(DOMAIN_SPECS)
        assert n_split + len(SIGNED_AGREEMENT_KEYS) == len(METRIC_SPECS)

    def test_polarity_lookup(self):
        """HIGHER_IS_BETTER dict should cover all metrics."""
        pytest.importorskip("scripts.sim2rank.metrics_list")  # not in the public export
        from scripts.sim2rank.metrics_list import HIGHER_IS_BETTER, METRIC_KEYS
        for key in METRIC_KEYS:
            assert key in HIGHER_IS_BETTER

    def test_covers_registry(self):
        """Sim2Rank should include all registered metrics (except the
        documented exclusions).

        Excluded by design:
          - zipper_detection (raises NotImplementedError)
          - Novel-arm research metrics that need bespoke input contracts
            (geodesic q-map error, fabrication rate, etc.). These are
            evaluated through their own arm-specific harnesses, NOT the
            generic Sim2Rank sweep.
        """
        pytest.importorskip("scripts.sim2rank.metrics_list")  # not in the public export
        from scripts.sim2rank.metrics_list import METRIC_KEYS
        from spectramr.core.metrics.registry import MetricsRegistry

        all_registered = set(MetricsRegistry.list_available())
        included = set(METRIC_KEYS)
        # NotImplementedError + bespoke-contract research metrics.
        excluded_ok = {
            "zipper_detection",
            # Novel-arm metrics with non-standard inputs (q-maps, FC
            # matrices, MRF parameter maps, scanner-pair concordance).
            # Each one is evaluated in its own arm harness.
            "geodesic_qmap_error",
            "geodesic_fc_error",
            "geodesic_mrf_parameter_error",
            "feature_fidelity_index",
            "cosine_preservation_score",
            "fabrication_rate",
            "cross_scanner_t1t2_concordance",
            "kernelised_stein_discrepancy",
            # Novel VF/P-arm metrics (registered with b90e17bf7) evaluated in
            # their own arm harnesses, NOT the generic 2-D sweep:
            #   banding_region_mse      — bSSFP banding-region MSE (needs a band mask)
            #   bloch_manifold_residual — P2 DPS, residual on the Bloch manifold
            #   equivariance_defect     — P1 equivariance-conformal defect (transform pairs)
            "banding_region_mse",
            "bloch_manifold_residual",
            "equivariance_defect",
            # Categorically incompatible with 2-D image-pair scoring — each
            # is 100% NaN on the sim2rank sweep and was removed from
            # METRIC_SPECS (2026-05-25). They remain registered for their
            # native input contracts:
            #   through_plane_fwhm           — needs a 3-D volume (LSF along z)
            #   persistence_diameter         — consumes an MRF trajectory (T, d)
            #   topological_mask_certificate — consumes a sparse k-space mask
            "through_plane_fwhm",
            "persistence_diameter",
            "topological_mask_certificate",
            # Bespoke-contract eval/QA metrics (pre-existing gap, reconciled
            # 2026-06-03 alongside the NR-battery load). Each needs inputs the
            # generic 2-D (pred, target) sweep cannot provide — a calibration
            # set (conformal), subgroup membership labels (disparity), an
            # adversarial/OOD perturbation harness (robustness), or
            # probabilistic predictions + labels (calibration). They are scored
            # in their own arm harnesses, not the Sim2Rank image sweep.
            "conformal_risk_control",
            "qmap_conformal_coverage",
            "subgroup_psnr_disparity",
            "subgroup_ssim_gap",
            "adversarial_psnr_drop",
            "ood_acceleration_shift",
            "input_dependence_spread",
            "brier_score",
            "classwise_ece",
            "adaptive_ece",
            # The §8.7 learned NR-quality-index aggregator. It is a fitted,
            # research-mode meta-metric registered only on explicit opt-in
            # (register_nr_quality_index / --register-aggregator) and reads its
            # inputs from kwargs["nr_features"], not a (pred, target) pair. It is
            # never part of the generic Sim2Rank 2-D sweep.
            "nr_quality_index",
            # ── Registry reconciliation, 2026-07-13 (#265) ──
            # Each was MEASURED on a degraded (pred, target) magnitude pair
            # before being excluded; none can be scored by the generic 2-D
            # sweep. The 7 that CAN be (nrmse_l2, focal_frequency, lpips_alex,
            # brain_mask_dice, tissue_dice, tissue_hd95,
            # tissue_volume_similarity) were ADDED to PER_IMAGE_SPECS instead.
            #
            # Model properties, not image properties — a function of the
            # network's weights, so NaN on any (pred, target) pair and
            # constant under degradation by construction:
            "composed_spectral_norm_bound",
            "max_layer_spectral_norm",
            # Bespoke input contracts — a magnitude image is not the quantity:
            #   gaussian_nll             — needs a 2-channel [mean, logvar] prediction
            #   nse_hall                 — needs the null-space / forward operator
            #   volume_consistency       — needs the 14 DGM label maps
            #   task_based_detectability — CHO d'; undefined (NaN) at zero
            #                              degradation, needs a lesion+noise harness
            #   srf_bound                — DISTRIBUTIONAL (per-band feature-space W1
            #                              over a SET, like med_fid); a SUMMARY_SPECS
            #                              candidate, not a per-image one
            "gaussian_nll",
            "nse_hall",
            "volume_consistency",
            "task_based_detectability",
            "srf_bound",
            # qMRI / field / trajectory metrics. These DO return a finite number
            # for a magnitude-image pair — which is exactly the problem. Each
            # computes an MAE/RMSE on whatever tensor it is handed and labels the
            # result "mm^2/s" / "Hz" / "cycles/FOV". B0FieldRMSE's comparability
            # guard cannot stop it: it infers the physical KIND from
            # `is_complex()`, and a real Hz field and a real magnitude image are
            # the same dtype (#273). Admitting them would put a mislabelled
            # physical quantity on the leaderboard (pitfall #18).
            "adc_mae",
            "b0_field_rmse",
            "k_space_trajectory_rmse",
            "phys_residual_consistency",
            # SynthSeg metrics silently fall back to LabelDiceBackend (an
            # intensity-binning proxy) when no segmenter is injected — which is
            # what the sweep supplies. They would contribute Otsu-proxy numbers
            # to the leaderboard under a SynthSeg label (#272).
            "synthseg_dice",
            "synthseg_dice_risk",
            # FLOW (4D-flow) and PERFUSION (ASL) reference metrics, registered on
            # Regime.FLOW / Regime.PERFUSION. Like the qMRI cluster above, each computes
            # an MAE/RMSE on whatever tensor it is handed and labels it "cm/s" /
            # "mL/100g/min" / "mL/s" — but the sweep degrades a static magnitude image,
            # which carries no velocity field or CBF map. Untestable on this cohort
            # (pitfall #19); admitting them would put a mislabelled physical quantity on
            # the leaderboard (pitfall #18).
            "velocity_rmse",
            "peak_velocity_error",
            "net_flow_error",
            "cbf_rmse",
            "att_mae",
        }

        missing = all_registered - included - excluded_ok
        assert missing == set(), f"Missing from Sim2Rank: {missing}"

    def test_image_incompatible_metrics_excluded(self):
        """Metrics whose input contract is not a 2-D image pair must NOT be
        in the sim2rank sweep — they returned 100% NaN and silently
        polluted the leaderboard (3-D volume / MRF trajectory / k-space
        mask contracts). Regression guard for the 2026-05-25 removal.
        """
        pytest.importorskip("scripts.sim2rank.metrics_list")  # not in the public export
        from scripts.sim2rank.metrics_list import METRIC_KEYS

        for key in (
            "through_plane_fwhm",
            "persistence_diameter",
            "topological_mask_certificate",
        ):
            assert key not in METRIC_KEYS, (
                f"{key} is incompatible with 2-D image scoring and must "
                "stay out of METRIC_SPECS (see test_covers_registry)."
            )


# ──────────────────────────────────────────────────────────────────────
# Clustering Tests
# ──────────────────────────────────────────────────────────────────────


class TestClustering:
    """Test HFR clustering."""

    def test_cluster_count(self):
        """Should produce exactly n_clusters clusters."""
        pytest.importorskip("scripts.sim2rank.clustering")  # not in the public export
        from scripts.sim2rank.clustering import cluster_metrics
        pytest.importorskip("scripts.sim2rank.metrics_list")  # not in the public export
        from scripts.sim2rank.metrics_list import METRIC_SPECS

        M = len(METRIC_SPECS)
        T = 20
        traj = np.random.randn(M, T)
        adr = np.random.rand(M)

        result = cluster_metrics(traj, METRIC_SPECS, adr, n_clusters=4)

        assert len(result["clusters"]) == 4
        assert len(result["spanning_set"]) == 4

    def test_spanning_set_covers_all_clusters(self):
        """Spanning set should have one representative per cluster."""
        pytest.importorskip("scripts.sim2rank.clustering")  # not in the public export
        from scripts.sim2rank.clustering import cluster_metrics
        pytest.importorskip("scripts.sim2rank.metrics_list")  # not in the public export
        from scripts.sim2rank.metrics_list import METRIC_SPECS

        M = len(METRIC_SPECS)
        traj = np.random.randn(M, 20)
        adr = np.random.rand(M)

        result = cluster_metrics(traj, METRIC_SPECS, adr, n_clusters=4)

        spanning = result["spanning_set"]
        labels = result["cluster_labels"]

        # Each spanning set member should be from a different cluster
        cluster_ids = [labels[i] for i in spanning]
        assert len(set(cluster_ids)) == len(cluster_ids), (
            f"Spanning set should cover unique clusters: {cluster_ids}"
        )


class TestContextEnrichment:
    """``engine._compute_with_context`` supplies the acquisition-aware context.

    Regression for the 2026-06-03 enrichment: the measurement/acq-order/
    multi-contrast NR metrics used to return NaN or a constant on the sim2rank
    sweep because the engine only built coil_maps / mask=None / clean y_kspace.
    The engine now provides a real undersampling mask + degraded measurement,
    acq_order, an RNG generator, structural sibling contrasts, and a non-linear
    stand-in reconstructor — so these metrics produce finite, varying signal.
    """

    @staticmethod
    def _phantom() -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, 48), torch.linspace(-1, 1, 48), indexing="ij"
        )
        img = ((xx**2 + yy**2) < 0.7).float() * 0.6
        img += ((((xx + 0.2)) ** 2 + ((yy - 0.1)) ** 2) < 0.12).float() * 0.3
        return img[None, None]

    def _engine(self):
        pytest.importorskip("scripts.sim2rank.engine")  # not in the public export
        from scripts.sim2rank.engine import Sim2RankEngine
        pytest.importorskip("scripts.sim2rank.metrics_list")  # not in the public export
        from scripts.sim2rank.metrics_list import PER_IMAGE_SPECS

        return Sim2RankEngine(metric_specs=PER_IMAGE_SPECS, device="cpu")

    def _trajectory(self, eng, key: str, axis: str) -> list[float]:
        gt = self._phantom()

        def degrade(sev: float) -> torch.Tensor:
            g = torch.Generator().manual_seed(7)
            return (gt + sev * 0.4 * torch.randn(gt.shape, generator=g)).clamp(min=0)

        fn = eng.metrics[key]
        return [
            eng._compute_with_context(fn, degrade(s), gt, key, axis, s, None)
            for s in (0.0, 0.4, 0.8)
        ]

    def test_measurement_metrics_vary_on_undersampling(self) -> None:
        """prp_snr/fmrs/are/lrjs/pedu/aoqv are finite and non-constant."""
        eng = self._engine()
        for key in ("prp_snr", "fmrs", "are", "lrjs", "pedu", "aoqv"):
            if key not in eng.metrics:
                continue
            vals = self._trajectory(eng, key, "cartesian_undersamp")
            finite = [v for v in vals if v == v]
            assert len(finite) >= 2, f"{key}: too many NaN on undersampling: {vals}"
            assert max(finite) - min(finite) > 1e-9, f"{key}: constant: {vals}"

    def test_ccsa_and_fmed_vary(self) -> None:
        """ccsa (sibling contrasts) and fmed (image-domain recon) are live."""
        eng = self._engine()
        for key, axis in (("ccsa", "rigid_motion"), ("fmed", "rigid_motion")):
            if key not in eng.metrics:
                continue
            vals = self._trajectory(eng, key, axis)
            finite = [v for v in vals if v == v]
            assert len(finite) >= 2 and max(finite) - min(finite) > 1e-9, (
                f"{key}: not varying: {vals}"
            )

    def test_ndcr_baseline_unchanged(self) -> None:
        """NDCR keeps the fully-sampled normalised-image-error baseline (0 at
        severity 0, rising) — the enrichment must not regress it."""
        eng = self._engine()
        vals = self._trajectory(eng, "ndcr", "cartesian_undersamp")
        assert vals[0] == vals[0] and abs(vals[0]) < 1e-6, f"NDCR(0) != 0: {vals}"
        assert vals[-1] > vals[0], f"NDCR not rising: {vals}"


class TestSweepMetricsAreMeasurable:
    """A metric earns its place in the sweep by MEASURING, not by taxonomy.

    The 2026-05 leaderboard carried metrics that were 100% NaN (3-D volume /
    MRF-trajectory / k-space-mask contracts fed a 2-D image pair) and metrics
    that cannot vary with degradation at all. Both survived because membership
    was decided by what a metric was *called*, not by what it *returns*. This is
    the guard for the 2026-07-13 additions (#265).
    """

    @staticmethod
    def _pair(severity: float):
        import torch

        torch.manual_seed(0)
        target = torch.zeros(1, 1, 64, 64)
        target[0, 0, 16:48, 20:44] = 1.0
        target[0, 0, 24:32, 28:36] = 0.5
        target = target + 0.02 * torch.randn(1, 1, 64, 64)
        torch.manual_seed(1)
        return target + severity * torch.randn(1, 1, 64, 64), target

    @pytest.mark.parametrize(
        "key",
        [
            "nrmse_l2",
            "focal_frequency",
            "lpips_alex",
            "brain_mask_dice",
            "tissue_dice",
            "tissue_hd95",
            "tissue_volume_similarity",
        ],
    )
    def test_added_metric_is_finite_and_moves_with_severity(self, key: str) -> None:
        import math

        from spectramr.core.metrics.registry import MetricsRegistry

        if key == "lpips_alex":
            # optional dep (pyproject [eval]); a .[test]-only env should skip,
            # not error. The sweep engine likewise drops it with a logged warning.
            pytest.importorskip("lpips")

        metric = MetricsRegistry.get(key)
        scores = [float(metric(*self._pair(s))) for s in (0.0, 0.3, 0.6)]

        assert all(math.isfinite(v) for v in scores), f"{key} is not finite: {scores}"
        assert max(scores) - min(scores) > 1e-6, (
            f"{key} is CONSTANT under degradation ({scores}) — it cannot rank "
            "anything and must not be in the sweep."
        )


# ──────────────────────────────────────────────────────────────────────
# B9: SCVR's content factor is the subject, per contrast
# ──────────────────────────────────────────────────────────────────────


def _two_contrast_scvr(seed: int = 0, s: int = 4, t: int = 6) -> tuple[np.ndarray, list[str]]:
    """``(M=2, 2s, 1, t)`` contents, ``s`` subjects in each of two contrasts."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, t)
    raw = np.empty((2, 2 * s, 1, t))
    for n in range(2 * s):
        raw[:, n, 0] = 1.0 - grid + rng.normal(0, 0.05, (2, t)) + rng.normal(0, 0.2, (2, 1))
    return raw, ["T1"] * s + ["T2"] * s


class TestSCVRByContrast:
    def test_a_contrast_offset_does_not_inflate_the_nuisance(self) -> None:
        from scripts.sim2rank.scoring import compute_scvr_ranking, scvr_by_contrast

        raw, contrasts = _two_contrast_scvr()
        shifted = raw.copy()
        shifted[:, 4:] += 5.0  # T2 reads 5 units higher on every subject and level
        flat = [compute_scvr_ranking(torch.from_numpy(x))[0].numpy() for x in (raw, shifted)]
        assert not np.allclose(*flat)  # the flat content factor counts the contrast
        np.testing.assert_allclose(
            scvr_by_contrast(shifted, contrasts), scvr_by_contrast(raw, contrasts)
        )

    def test_opposite_subject_effects_per_contrast_stay_finite(self) -> None:
        # A (subject, contrast) reshape averages +e and -e away and reads zero nuisance.
        from scripts.sim2rank.scoring import scvr_by_contrast

        raw, contrasts = _two_contrast_scvr()
        raw[:, 4:] = raw[:, :4] - 2 * (raw[:, :4] - raw[:, :4].mean(axis=1, keepdims=True))
        assert np.isfinite(scvr_by_contrast(raw, contrasts)).all()

    def test_a_missing_cell_takes_its_level_mean(self) -> None:
        from scripts.sim2rank.scoring import scvr_by_contrast

        raw, contrasts = _two_contrast_scvr()
        others = [0, 2, 3]
        raw[0, 1, 0, 3] = raw[0, others, 0, 3].mean()
        full = scvr_by_contrast(raw, contrasts)
        holed = raw.copy()
        holed[0, 1, 0, 3] = np.nan
        np.testing.assert_allclose(scvr_by_contrast(holed, contrasts), full)

    def test_a_metric_needs_half_its_cells_in_a_contrast(self) -> None:
        from scripts.sim2rank.scoring import SCVR_MIN_FINITE_FRACTION, scvr_by_contrast

        raw, contrasts = _two_contrast_scvr()
        assert SCVR_MIN_FINITE_FRACTION == 0.5
        raw[1, :4, 0, :4] = np.nan  # 16 of 24 T1 cells missing: T1 no longer scores it
        out = scvr_by_contrast(raw, contrasts)
        np.testing.assert_allclose(out[1], scvr_by_contrast(raw[:, 4:], contrasts[4:])[1])

    def test_an_unmeasured_metric_is_nan_not_zero(self) -> None:
        from scripts.sim2rank.scoring import scvr_by_contrast

        raw, contrasts = _two_contrast_scvr()
        raw[1] = np.nan
        out = scvr_by_contrast(raw, contrasts)
        assert np.isfinite(out[0]) and np.isnan(out[1])

    def test_gen1_scvr_uses_the_per_contrast_rule(self) -> None:
        from scripts.sim2rank.run_all_generations import compute_gen1_sub_scores
        from scripts.sim2rank.scoring import scvr_by_contrast

        raw, contrasts = _two_contrast_scvr()
        raw[0, 1, 0, 3] = np.nan
        axes = {a: raw[:, :, 0].mean(axis=1) + 0.01 * k for k, a in enumerate(("x", "y"))}
        axes = {a: np.nan_to_num(v) for a, v in axes.items()}
        out = compute_gen1_sub_scores(
            axes, raw, [False, False], np.linspace(0, 1, 6), content_contrast=contrasts
        )
        np.testing.assert_allclose(out["Gen1_SCVR"], scvr_by_contrast(raw, contrasts))


# ──────────────────────────────────────────────────────────────────────
# B9: Jacobian coverage replaces the Gini sparsity
# ──────────────────────────────────────────────────────────────────────


class TestJacobianCoverage:
    def test_only_significant_directed_responses_count(self) -> None:
        # Pure seed jitter must not read as a response; a monotone row must.
        from scripts.sim2rank.scoring import directed_response

        rng = np.random.default_rng(0)
        t = 8
        noise = rng.normal(size=(400, t))
        assert directed_response(noise, [False] * 400).mean() <= 0.1
        ramp = np.tile(np.linspace(0, 1, t), (3, 1))
        assert directed_response(ramp, [False] * 3).all()
        assert not directed_response(-ramp, [False] * 3).any()  # wrong direction
        assert directed_response(-ramp, [True] * 3).all()

    def test_the_critical_value_is_one_sided_alpha(self) -> None:
        from scripts.sim2rank.scoring import spearman_critical

        assert spearman_critical(3) > 1.0  # 1/6 > 0.05: three levels cannot respond
        assert spearman_critical(4) == pytest.approx(1.0)  # P(rho = 1) = 1/24
        assert 0.48 < spearman_critical(12) < 0.53  # table value 0.503 at n = 12

    def test_gen1_coverage_uses_the_directed_test(self) -> None:
        from scripts.sim2rank.run_all_generations import compute_gen1_sub_scores
        from scripts.sim2rank.scoring import directed_response

        rng = np.random.default_rng(1)
        t = np.linspace(0, 1, 8)
        rates = np.array([1.0, 0.6, 0.3])[:, None]
        axes = {a: rates * t + rng.normal(0, 0.05, (3, 8)) for a in ("x", "y", "z")}
        axes["z"][2] = rng.normal(0, 0.05, 8)
        out = compute_gen1_sub_scores(axes, None, [False] * 3, t)
        want = np.stack([directed_response(v, [False] * 3) for v in axes.values()], 1).mean(1)
        np.testing.assert_allclose(out["Gen1_JacobianCoverage"], want)

    def test_nan_steps_count_against_coverage(self) -> None:
        from scripts.sim2rank.scoring import compute_jacobian_sensitivity

        t = np.linspace(0, 1, 6)
        holed = 1.0 - t
        holed[2:] = np.nan
        axes = {"a": np.stack([1.0 - t, 1.0 - t]), "b": np.stack([1.0 - t, holed])}
        _, coverage = compute_jacobian_sensitivity(axes, responds=np.ones((2, 2), bool))
        np.testing.assert_allclose(coverage, [1.0, 0.5])

    def test_without_a_response_mask_coverage_is_not_computed(self) -> None:
        from scripts.sim2rank.scoring import compute_jacobian_sensitivity

        t = np.linspace(0, 1, 6)
        _, coverage = compute_jacobian_sensitivity({"a": np.stack([t, t])})
        assert np.isnan(coverage).all()
