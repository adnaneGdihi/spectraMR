"""Tests for the Equivariant Imaging sub-schema, its mount, and registration."""

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training.equivariant_imaging import (
    TrainingConfigEquivariantImaging,
)


def test_accepts_valid_fields():
    cfg = TrainingConfigEquivariantImaging(
        group="small_rotation",
        alpha_equivariance=0.5,
        n_group_samples=2,
        max_angle_deg=6.0,
        num_angles=3,
    )
    assert cfg.group == "small_rotation"
    assert cfg.alpha_equivariance == 0.5
    assert cfg.n_group_samples == 2
    assert cfg.max_angle_deg == 6.0


def test_defaults():
    cfg = TrainingConfigEquivariantImaging()
    assert cfg.group == "dihedral"
    assert cfg.alpha_equivariance == 1.0
    assert cfg.n_group_samples == 1
    assert cfg.robust_correction is False
    assert cfg.noise_model == "gaussian_kspace"
    assert cfg.n_coils == 1


def test_rejects_unknown_field():
    with pytest.raises(ValidationError):
        TrainingConfigEquivariantImaging(grouping="dihedral")  # unknown key -> extra="forbid"


def test_rejects_unknown_group():
    with pytest.raises(ValidationError):
        TrainingConfigEquivariantImaging(group="so2_full")


def test_robust_correction_requires_noise_std():
    """pitfall #15: an advertised robust correction with no σ must fail at load."""
    with pytest.raises(ValidationError, match="noise_std_estimate"):
        TrainingConfigEquivariantImaging(robust_correction=True)


def test_robust_correction_with_noise_std_ok():
    cfg = TrainingConfigEquivariantImaging(
        robust_correction=True, noise_std_estimate=0.01, noise_model="ncchi_magnitude", n_coils=4
    )
    assert cfg.robust_correction is True
    assert cfg.noise_std_estimate == 0.01
    assert cfg.noise_model == "ncchi_magnitude"


def test_rejects_nonpositive_noise_std():
    with pytest.raises(ValidationError):
        TrainingConfigEquivariantImaging(robust_correction=True, noise_std_estimate=0.0)


def test_mounts_on_training_schema():
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema

    assert "equivariant_imaging" in TrainingStrategyConfigSchema.model_fields, (
        "training.equivariant_imaging sub-block not mounted on TrainingStrategyConfigSchema"
    )


def test_ei_keys_registered_and_resolvable():
    from spectramr.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    paths = TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    for key in ("equivariant_imaging", "robust_ei"):
        assert key in paths
        assert paths[key].endswith(
            "equivariant_imaging_strategy.EquivariantImagingStrategy"
        )


def test_measurement_target_defaults_to_the_measurement():
    """Every existing EI arm keeps its anchor on the y it reconstructs from."""
    assert TrainingConfigEquivariantImaging().measurement_target == "measured"


def test_measurement_target_is_a_closed_set():
    with pytest.raises(ValidationError):
        TrainingConfigEquivariantImaging(measurement_target="the_other_one")


def test_the_gsure_anchor_refuses_an_alternate_repetition():
    """GSURE corrects the MEASURED anchor's noise; against a second repetition it
    has nothing to correct, so the pair is a contradiction, not a combination."""
    with pytest.raises(ValidationError, match="alternate repetition"):
        TrainingConfigEquivariantImaging(
            robust_correction=True,
            noise_std_estimate=0.1,
            measurement_target="alternate_repetition",
        )


class TestBandKnobs:
    """The acquired-band knobs default to today's anchor and target, and refuse the
    anchors that never read them (non-negotiable 8)."""

    def test_defaults_are_the_unchanged_objective(self):
        from spectramr.config.schemas.training.equivariant_imaging import (
            TrainingConfigEquivariantImaging,
        )

        cfg = TrainingConfigEquivariantImaging()
        assert (cfg.anchor_mask, cfg.anchor_normalization, cfg.equivariance_target) == (
            "input",
            "all_entries",
            "transformed",
        )

    def test_accepted_on_the_plain_measured_anchor(self):
        from spectramr.config.schemas.training.equivariant_imaging import (
            TrainingConfigEquivariantImaging,
        )

        cfg = TrainingConfigEquivariantImaging(
            anchor_mask="with_structural_zeros",
            anchor_normalization="selected",
            equivariance_target="band_limited",
        )
        assert cfg.anchor_mask == "with_structural_zeros"

    @pytest.mark.parametrize(
        "planted",
        [
            {"anchor_mask": "with_structural_zeros", "robust_correction": True, "noise_std_estimate": 0.1},
            {"anchor_normalization": "selected", "measurement_target": "alternate_repetition"},
        ],
        ids=["gsure-anchor", "alternate-repetition-anchor"],
    )
    def test_refused_where_the_anchor_never_reads_them(self, planted):
        from pydantic import ValidationError

        from spectramr.config.schemas.training.equivariant_imaging import (
            TrainingConfigEquivariantImaging,
        )

        with pytest.raises(ValidationError, match="never read them"):
            TrainingConfigEquivariantImaging(**planted)

    @pytest.mark.parametrize("field", ["anchor_mask", "anchor_normalization", "equivariance_target"])
    def test_an_unknown_value_is_refused(self, field):
        from pydantic import ValidationError

        from spectramr.config.schemas.training.equivariant_imaging import (
            TrainingConfigEquivariantImaging,
        )

        with pytest.raises(ValidationError):
            TrainingConfigEquivariantImaging(**{field: "bogus"})
