"""Tests for the quality-matching config block."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training.quality_matching import (
    QualityMatchingConfig,
    QualityTargetConfig,
)


def _target(**kw):
    base = {
        "source": "literal",
        "attributes": ["tenengrad_variance"],
        "override": {"tenengrad_variance": 0.01},
    }
    base.update(kw)
    return QualityTargetConfig(**base)


# ── strictness ────────────────────────────────────────────────────────


def test_rejects_an_unknown_key():
    # StrictSchema: a typo'd knob must not be silently absorbed. The PARENT
    # TrainingStrategyConfigSchema is extra='allow', so this block carrying its own
    # extra='forbid' is what stops a mistyped knob becoming inert (issue #550).
    with pytest.raises(ValidationError):
        QualityMatchingConfig(axes=["complex_gaussian"], target=_target(), typoo=1)


def test_is_frozen():
    # match_spacing off: this test is about frozen-ness, not geometry.
    cfg = QualityMatchingConfig(axes=["complex_gaussian"], target=_target(), match_spacing=False)
    with pytest.raises(ValidationError):
        cfg.max_evals = 10


# ── target validation ─────────────────────────────────────────────────


def test_rejects_an_unmeasurable_target_attribute():
    with pytest.raises(ValidationError, match="not a registered metric"):
        QualityTargetConfig(source="literal", attributes=["definitely_not_a_metric"], override={})


def test_rejects_a_full_reference_target_attribute():
    with pytest.raises(ValidationError, match="requires a reference"):
        QualityTargetConfig(source="literal", attributes=["psnr"], override={"psnr": 30.0})


def test_cohort_source_requires_a_manifest():
    with pytest.raises(ValidationError, match="cohort_manifest"):
        QualityTargetConfig(source="cohort", attributes=["tenengrad_variance"], override={})


def test_literal_source_requires_an_override_for_every_attribute():
    with pytest.raises(ValidationError, match="override"):
        QualityTargetConfig(source="literal", attributes=["tenengrad_variance"], override={})


def test_cohort_source_accepts_a_partial_override():
    # The documented ablation pattern: inherit the cohort fit, pin one attribute.
    cfg = QualityTargetConfig(
        source="cohort",
        cohort_manifest="data/manifests/lq_cohort.json",
        attributes=["tenengrad_variance", "laplacian_variance"],
        override={"tenengrad_variance": 0.02},
    )
    assert cfg.override == {"tenengrad_variance": 0.02}


def test_rejects_an_override_for_an_undeclared_attribute():
    # An override nobody reads is an unwired knob (pitfall #15).
    with pytest.raises(ValidationError, match="not a declared attribute"):
        QualityTargetConfig(
            source="cohort",
            cohort_manifest="data/manifests/lq_cohort.json",
            attributes=["tenengrad_variance"],
            override={"laplacian_variance": 0.02},
        )


# ── axis validation ───────────────────────────────────────────────────


def test_rejects_an_empty_axis_list():
    with pytest.raises(ValidationError):
        QualityMatchingConfig(axes=[], target=_target())


def test_rejects_an_unknown_axis():
    with pytest.raises(ValidationError, match="not a known degradation"):
        QualityMatchingConfig(axes=["not_an_axis"], target=_target())


def test_rejects_a_native_only_axis():
    # Delegating to DegradationChain means the schema and the runtime can never
    # disagree about which axes are legal.
    with pytest.raises(ValidationError, match="native"):
        QualityMatchingConfig(axes=["motion"], target=_target())


# ── defaults and mounting ─────────────────────────────────────────────


def test_defaults_are_the_documented_ones():
    cfg = QualityMatchingConfig(axes=["complex_gaussian"], target=_target(), match_spacing=False)
    assert cfg.max_evals == 400
    assert cfg.method == "differential_evolution"
    assert cfg.min_gap_closed == 0.5
    assert cfg.fit_seed == 0


def test_is_mounted_on_the_training_schema():
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema

    assert "quality_matching" in TrainingStrategyConfigSchema.model_fields


def test_mounted_block_parses_from_a_nested_dict():
    """The real wiring path: a YAML-shaped dict must reach a typed block.

    The parent schema is extra='allow', so an unmounted or mistyped block would be
    accepted as a raw dict and read back as None -- indistinguishable from success
    until something silently does nothing.
    """
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema

    parsed = TrainingStrategyConfigSchema(
        quality_matching={
            "axes": ["complex_gaussian"],
            "target": {
                "source": "literal",
                "attributes": ["tenengrad_variance"],
                "override": {"tenengrad_variance": 0.01},
            },
            "max_evals": 50,
            "match_spacing": False,
        }
    )
    assert parsed.quality_matching is not None
    assert isinstance(parsed.quality_matching, QualityMatchingConfig)
    assert parsed.quality_matching.axes == ["complex_gaussian"]
    assert parsed.quality_matching.max_evals == 50


# ── geometry knobs ────────────────────────────────────────────────────


def test_match_spacing_defaults_on():
    cfg = QualityMatchingConfig(
        axes=["complex_gaussian"], target=_target(spacing_mm=(5.0, 1.2, 1.2))
    )
    assert cfg.match_spacing is True


def test_literal_target_with_match_spacing_requires_explicit_spacing():
    # There is no cohort to measure a grid from, so match_spacing would be inert.
    with pytest.raises(ValidationError, match=r"target\.spacing_mm"):
        QualityMatchingConfig(axes=["complex_gaussian"], target=_target())


def test_literal_target_may_disable_spacing_matching():
    cfg = QualityMatchingConfig(axes=["complex_gaussian"], target=_target(), match_spacing=False)
    assert cfg.match_spacing is False


def test_cohort_target_needs_no_explicit_spacing():
    # Measured from the cohort's own headers -- the whole point.
    cfg = QualityMatchingConfig(
        axes=["complex_gaussian"],
        target=QualityTargetConfig(
            source="cohort",
            cohort_manifest="data/manifests/lq.json",
            attributes=["tenengrad_variance"],
            override={},
        ),
    )
    assert cfg.match_spacing is True
    assert cfg.target.spacing_mm is None


# ── the acquisition prior's auto default ──────────────────────────────


def _cohort_target(**kw):
    base = {
        "source": "cohort",
        "cohort_manifest": "data/manifests/lq.json",
        "attributes": ["tenengrad_variance"],
        "override": {},
    }
    base.update(kw)
    return QualityTargetConfig(**base)


def test_prior_auto_enables_for_a_cohort_target():
    cfg = QualityMatchingConfig(axes=["complex_gaussian"], target=_cohort_target())
    assert cfg.enable_acquisition_prior is None  # unset
    assert cfg.acquisition_prior_enabled is True  # resolved


def test_prior_auto_disables_for_a_literal_target():
    """The default must not invalidate a literal-target config.

    A new validation that rejected the library default would break every existing
    arm for a knob its author never set.
    """
    cfg = QualityMatchingConfig(axes=["complex_gaussian"], target=_target(), match_spacing=False)
    assert cfg.acquisition_prior_enabled is False


def test_explicitly_requesting_the_prior_on_a_literal_target_raises():
    # Guarding the EXPLICIT arg: asking for something impossible must be loud.
    with pytest.raises(ValidationError, match=r"requires target\.source='cohort'"):
        QualityMatchingConfig(
            axes=["complex_gaussian"],
            target=_target(),
            match_spacing=False,
            enable_acquisition_prior=True,
        )


def test_prior_can_be_explicitly_disabled_on_a_cohort_target():
    cfg = QualityMatchingConfig(
        axes=["complex_gaussian"],
        target=_cohort_target(),
        enable_acquisition_prior=False,
    )
    assert cfg.acquisition_prior_enabled is False


def test_more_axes_than_attributes_is_refused_as_unidentified():
    # Planted: 3 severities against 2 attributes is a zero-residual curve (issue #2411).
    with pytest.raises(ValidationError, match="unidentified"):
        QualityMatchingConfig(
            axes=["complex_gaussian", "resolution_snr", "rigid_motion"],
            target=_target(
                attributes=["tenengrad_variance", "brisque"],
                override={"tenengrad_variance": 0.01, "brisque": 5.0},
            ),
        )


def test_two_axes_for_two_attributes_is_accepted():
    cfg = QualityMatchingConfig(
        match_spacing=False,
        axes=["complex_gaussian", "resolution_snr"],
        target=_target(
            attributes=["tenengrad_variance", "brisque"],
            override={"tenengrad_variance": 0.01, "brisque": 5.0},
        ),
    )
    assert cfg.intensity_percentile == 99.0
    assert cfg.max_residual is None


def test_held_out_requires_paired_pairs():
    # Planted: an unpaired cohort has no same-subject reference to score against.
    with pytest.raises(ValidationError, match=r"requires target\.pairing: paired"):
        QualityMatchingConfig(axes=["complex_gaussian"], target=_target(), held_out_fraction=0.2)


def test_fit_seeds_and_residual_ceiling_must_be_positive():
    with pytest.raises(ValidationError):
        QualityMatchingConfig(axes=["complex_gaussian"], target=_target(), fit_seeds=0)
    with pytest.raises(ValidationError):
        QualityMatchingConfig(axes=["complex_gaussian"], target=_target(), max_residual=0.0)
