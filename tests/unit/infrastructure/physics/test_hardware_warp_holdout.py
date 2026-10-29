"""Unit tests for the hold-one-subject-out fold of the hardware-warp calibration.

The planted pose set is noiseless, so the fold's out-of-sample residual after removing the
fitted commutator must vanish to numerical precision while the residual before it does not;
that is the shape of evidence the real arm reports, checked here where the truth is known.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from spectramr.data.hardware_warp_artifacts import HELD_OUT_KEYS, validate_artifact
from spectramr.infrastructure.physics.hardware_warp_calibration import calibrate_hardware_warp
from spectramr.infrastructure.physics.hardware_warp_holdout import (
    held_out_report,
    predict_pair_residual,
    split_pairs_by_subject,
)
from tests.utils.hardware_warp_synthesis import synthesize_planted_pose_set

pytestmark = [pytest.mark.unit]

KNOBS: dict[str, Any] = {
    "max_degree": 3,
    "min_degree": 2,
    "ball_radius_mm": 100.0,
    "basis_points_per_axis": 11,
    "unit_id": "swoop-01",
    "software_version": "1.2.3",
}


@pytest.fixture(scope="module")
def planted() -> dict[str, Any]:
    return synthesize_planted_pose_set(n_pairs=6, n_subjects=3, **KNOBS)


class TestSplit:
    def test_held_out_subject_leaves_the_fit_and_keeps_its_pairs(self, planted) -> None:
        fit, fit_res, held, held_res = split_pairs_by_subject(
            planted["pose_set"], planted["residuals"], "sub-02"
        )
        assert [p["subject_id"] for p in fit["pairs"]] == ["sub-01", "sub-03"] * 2
        assert [p["subject_id"] for p in held] == ["sub-02", "sub-02"]
        assert len(fit_res) == 4 and len(held_res) == 2
        assert fit["unit_id"] == planted["pose_set"]["unit_id"]
        assert fit["center_mm"] == planted["pose_set"]["center_mm"]

    def test_unknown_subject_raises_naming_the_subjects(self, planted) -> None:
        with pytest.raises(ValueError, match=r"sub-99.*\['sub-01', 'sub-02', 'sub-03'\]"):
            split_pairs_by_subject(planted["pose_set"], planted["residuals"], "sub-99")

    def test_holding_out_the_only_subject_raises(self) -> None:
        single = synthesize_planted_pose_set(n_pairs=3, n_subjects=1, **KNOBS)
        with pytest.raises(ValueError, match="no pair to fit on"):
            split_pairs_by_subject(single["pose_set"], single["residuals"], "sub-01")

    def test_residual_count_mismatch_raises(self, planted) -> None:
        with pytest.raises(ValueError, match="6 pairs but 5 residuals"):
            split_pairs_by_subject(planted["pose_set"], planted["residuals"][:5], "sub-01")


class TestPrediction:
    def test_prediction_equals_the_planted_commutator(self, planted) -> None:
        artifact = calibrate_hardware_warp(planted["pose_set"], planted["residuals"], **KNOBS)
        pair = planted["pose_set"]["pairs"][0]
        points, residual = planted["residuals"][0]
        generator = (torch.tensor(pair["omega"]), torch.tensor(pair["translation"]))
        predicted = predict_pair_residual(artifact, generator, points)
        assert predicted.shape == residual.shape
        assert torch.allclose(predicted, residual, atol=1e-6 * float(residual.abs().max()))

    def test_too_few_points_raise(self, planted) -> None:
        artifact = calibrate_hardware_warp(planted["pose_set"], planted["residuals"], **KNOBS)
        generator = (torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64))
        with pytest.raises(ValueError, match="cannot carry a basis"):
            predict_pair_residual(artifact, generator, torch.zeros(2, 3, dtype=torch.float64))


class TestFold:
    def test_held_out_residual_vanishes_after_the_fit(self, planted) -> None:
        fit, fit_res, held, held_res = split_pairs_by_subject(
            planted["pose_set"], planted["residuals"], "sub-02"
        )
        artifact = calibrate_hardware_warp(fit, fit_res, **KNOBS)
        report = held_out_report(artifact, held, held_res)
        assert set(report) == HELD_OUT_KEYS
        assert report["subject_id"] == "sub-02"
        assert report["n_pairs"] == 2
        assert report["n_points"] == sum(int(r.shape[0]) for _, r in held_res)
        assert report["rms_before_mm"] > 1e-3
        assert report["rms_after_mm"] < 1e-9 * report["rms_before_mm"]
        assert report["rms_predicted_mm"] == pytest.approx(report["rms_before_mm"], rel=1e-9)
        artifact["held_out"] = report
        validate_artifact(artifact)

    def test_a_wrong_warp_does_not_explain_the_held_out_pairs(self, planted) -> None:
        fit, fit_res, held, held_res = split_pairs_by_subject(
            planted["pose_set"], planted["residuals"], "sub-02"
        )
        artifact = calibrate_hardware_warp(fit, fit_res, **KNOBS)
        artifact["coeffs"] = [-c for c in artifact["coeffs"]]
        report = held_out_report(artifact, held, held_res)
        assert report["rms_after_mm"] == pytest.approx(2.0 * report["rms_before_mm"], rel=1e-9)

    def test_mixed_subjects_and_empty_reports_raise(self, planted) -> None:
        artifact = calibrate_hardware_warp(planted["pose_set"], planted["residuals"], **KNOBS)
        pairs, residuals = planted["pose_set"]["pairs"], planted["residuals"]
        with pytest.raises(ValueError, match="one subject"):
            held_out_report(artifact, pairs[:2], residuals[:2])
        with pytest.raises(ValueError, match="at least one"):
            held_out_report(artifact, [], [])
