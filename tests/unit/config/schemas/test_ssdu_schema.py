"""Tests for the SSDU training sub-schema, its mount, and factory registration."""

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training.ssdu import SSDUTrainingConfigSchema


def test_accepts_valid_fields():
    cfg = SSDUTrainingConfigSchema(theta_fraction=0.35, num_masks=4, split_seed=7)
    assert cfg.theta_fraction == 0.35
    assert cfg.num_masks == 4
    assert cfg.split_seed == 7


def test_defaults():
    cfg = SSDUTrainingConfigSchema()
    assert cfg.theta_fraction == 0.4
    assert cfg.num_masks == 1
    assert cfg.split_seed == 0


def test_rejects_unknown_field():
    with pytest.raises(ValidationError):
        SSDUTrainingConfigSchema(theta_fractions=0.4)  # typo -> extra="forbid"


def test_rejects_out_of_range_fraction():
    with pytest.raises(ValidationError):
        SSDUTrainingConfigSchema(theta_fraction=1.5)
    with pytest.raises(ValidationError):
        SSDUTrainingConfigSchema(theta_fraction=0.0)


def test_ssdu_mounts_on_training_schema():
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema

    assert "ssdu" in TrainingStrategyConfigSchema.model_fields, (
        "training.ssdu sub-block not mounted on TrainingStrategyConfigSchema"
    )


def test_ssdu_keys_registered_and_resolvable():
    from spectramr.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    paths = TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    for key in ("self_supervised_reconstruction", "ssdu", "robust_ssdu"):
        assert key in paths
        assert paths[key].endswith("ssdu_strategy.SSDUReconstructionStrategy")


# --- Robust SSDU (Noisier2Noise) fields ------------------------------------
def test_robust_ssdu_fields_default_off():
    cfg = SSDUTrainingConfigSchema()
    assert cfg.noisier2noise_correction is False
    assert cfg.noise_std_estimate is None
    assert cfg.noise_model == "gaussian_kspace"
    assert cfg.n_coils == 1


def test_robust_ssdu_accepts_valid_fields():
    cfg = SSDUTrainingConfigSchema(
        noisier2noise_correction=True,
        noise_std_estimate=0.02,
        noise_model="gaussian_kspace",
        n_coils=4,
        alpha=0.5,
    )
    assert cfg.noisier2noise_correction is True
    assert cfg.noise_std_estimate == 0.02
    assert cfg.n_coils == 4
    assert cfg.alpha == 0.5


def test_the_correction_is_only_defined_for_gaussian_kspace_noise():
    """Eq. 29's correction rests on E[n|.] = E[n~|.] / alpha^2, a Gaussian identity."""
    with pytest.raises(ValidationError, match="complex-Gaussian"):
        SSDUTrainingConfigSchema(
            noisier2noise_correction=True, noise_std_estimate=0.02, noise_model="ncchi_magnitude"
        )


def test_alpha_is_refused_where_nothing_reads_it():
    with pytest.raises(ValidationError, match="alpha is read only"):
        SSDUTrainingConfigSchema(alpha=0.5)


@pytest.mark.parametrize("alpha", [0.0, -0.5])
def test_alpha_must_be_positive(alpha):
    with pytest.raises(ValidationError):
        SSDUTrainingConfigSchema(noisier2noise_correction=True, noise_std_estimate=0.1, alpha=alpha)


def test_robust_ssdu_has_one_corruption_per_step():
    with pytest.raises(ValidationError, match="num_masks"):
        SSDUTrainingConfigSchema(noisier2noise_correction=True, noise_std_estimate=0.1, num_masks=2)


def test_robust_ssdu_requires_noise_std():
    with pytest.raises(ValidationError, match="noise_std_estimate"):
        SSDUTrainingConfigSchema(noisier2noise_correction=True)


def test_robust_ssdu_rejects_nonpositive_noise_std():
    with pytest.raises(ValidationError):
        SSDUTrainingConfigSchema(noisier2noise_correction=True, noise_std_estimate=0.0)


def test_robust_ssdu_rejects_unknown_noise_model():
    with pytest.raises(ValidationError):
        SSDUTrainingConfigSchema(noise_model="poisson")


def _robust(**kw):
    return SSDUTrainingConfigSchema(
        noisier2noise_correction=True, noise_std_estimate=0.02, n_coils=4, **kw
    )


def test_noise_covariance_defaults_to_white_and_accepts_the_committed_shape():
    assert SSDUTrainingConfigSchema().noise_covariance == "white"
    assert _robust(noise_covariance="committed_m4raw").noise_covariance == "committed_m4raw"


def test_noise_covariance_without_the_correction_is_unread():
    with pytest.raises(ValidationError, match="noise_covariance shapes the Robust SSDU"):
        SSDUTrainingConfigSchema(noise_covariance="committed_m4raw", n_coils=4)


def test_the_committed_shape_needs_four_coils():
    with pytest.raises(ValidationError, match="4x4 M4Raw matrix"):
        SSDUTrainingConfigSchema(
            noisier2noise_correction=True,
            noise_std_estimate=0.02,
            n_coils=1,
            noise_covariance="committed_m4raw",
        )


class TestScoreStructuralZeros:
    def test_off_by_default_and_accepted_on_vanilla_ssdu(self):
        from spectramr.config.schemas.training.ssdu import SSDUTrainingConfigSchema

        assert SSDUTrainingConfigSchema().score_structural_zeros is False
        assert SSDUTrainingConfigSchema(score_structural_zeros=True).score_structural_zeros

    def test_refused_under_robust_ssdu_which_never_reads_it(self):
        from pydantic import ValidationError

        from spectramr.config.schemas.training.ssdu import SSDUTrainingConfigSchema

        with pytest.raises(ValidationError, match="never reads it"):
            SSDUTrainingConfigSchema(
                score_structural_zeros=True,
                noisier2noise_correction=True,
                noise_std_estimate=0.02,
            )
