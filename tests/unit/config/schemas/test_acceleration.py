"""Tests for ``AccelerationConfigSchema``.

Targets ``spectramr.config.schemas.acceleration``. K-space undersampling
configuration with bounds (gt=0, ge/le on partial-Fourier and
center-fraction) and ``frozen=True`` immutability.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.acceleration import AccelerationConfigSchema
from spectramr.config.schemas.enums import AccelerationSchedule

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_construction() -> None:
    """All defaults applied at construction."""
    cfg = AccelerationConfigSchema()
    assert cfg.base_acceleration == 4.0
    assert cfg.max_acceleration == 8.0
    assert cfg.acceleration_schedule == AccelerationSchedule.LINEAR
    assert cfg.center_fraction == 0.08
    assert cfg.acceleration_type == "variable_density"
    assert cfg.mask_types == ["variable_density"]
    assert cfg.acceleration_range == [1.0, 16.0]
    assert cfg.enable_dynamic_mask is False


# ---------------------------------------------------------------------------
# Numeric constraints
# ---------------------------------------------------------------------------


def test_base_acceleration_must_be_positive() -> None:
    """``base_acceleration > 0``."""
    with pytest.raises(ValidationError):
        AccelerationConfigSchema(base_acceleration=0.0)
    with pytest.raises(ValidationError):
        AccelerationConfigSchema(base_acceleration=-1.0)


def test_max_acceleration_must_be_positive() -> None:
    """``max_acceleration > 0``."""
    with pytest.raises(ValidationError):
        AccelerationConfigSchema(max_acceleration=0.0)


def test_schedule_steps_at_least_one() -> None:
    """``schedule_steps >= 1``."""
    with pytest.raises(ValidationError):
        AccelerationConfigSchema(schedule_steps=0)


def test_density_power_non_negative() -> None:
    """``density_power >= 0``."""
    with pytest.raises(ValidationError):
        AccelerationConfigSchema(density_power=-0.1)


def test_pf_fraction_in_half_to_one() -> None:
    """``pf_fraction`` ∈ ``[0.5, 1.0]``."""
    with pytest.raises(ValidationError):
        AccelerationConfigSchema(pf_fraction=0.4)
    with pytest.raises(ValidationError):
        AccelerationConfigSchema(pf_fraction=1.1)
    cfg_low = AccelerationConfigSchema(pf_fraction=0.5)
    cfg_high = AccelerationConfigSchema(pf_fraction=1.0)
    assert cfg_low.pf_fraction == 0.5
    assert cfg_high.pf_fraction == 1.0


def test_center_fraction_in_unit_interval() -> None:
    """``center_fraction`` ∈ ``[0, 1]``."""
    with pytest.raises(ValidationError):
        AccelerationConfigSchema(center_fraction=-0.01)
    with pytest.raises(ValidationError):
        AccelerationConfigSchema(center_fraction=1.5)


def test_schedule_power_must_be_positive() -> None:
    """``schedule_power > 0``."""
    with pytest.raises(ValidationError):
        AccelerationConfigSchema(schedule_power=0.0)


# ---------------------------------------------------------------------------
# Enum coercion
# ---------------------------------------------------------------------------


def test_schedule_coerced_from_string() -> None:
    """Schedule is coerced from its string value."""
    cfg = AccelerationConfigSchema(acceleration_schedule="exponential")
    assert cfg.acceleration_schedule == AccelerationSchedule.EXPONENTIAL


def test_invalid_schedule_string_raises() -> None:
    """Unknown schedule value raises."""
    with pytest.raises(ValidationError):
        AccelerationConfigSchema(acceleration_schedule="bizarre_schedule")


# ---------------------------------------------------------------------------
# Optional / list fields
# ---------------------------------------------------------------------------


def test_optional_fields_default_to_none() -> None:
    """Optional path-like fields default to None."""
    cfg = AccelerationConfigSchema()
    assert cfg.pattern_path is None
    assert cfg.ground_truth_folder is None
    assert cfg.mask_seed is None


def test_mask_types_list_overridable() -> None:
    """``mask_types`` accepts an arbitrary list of strings."""
    cfg = AccelerationConfigSchema(mask_types=["uniform", "variable_density", "radial"])
    assert cfg.mask_types == ["uniform", "variable_density", "radial"]


# ---------------------------------------------------------------------------
# Frozen + extra='ignore'
# ---------------------------------------------------------------------------


def test_schema_is_frozen() -> None:
    """Cannot mutate after construction."""
    cfg = AccelerationConfigSchema()
    with pytest.raises(ValidationError):
        cfg.base_acceleration = 99.0


def test_extra_fields_silently_ignored() -> None:
    """Unknown fields don't raise (``extra='ignore'``)."""
    cfg = AccelerationConfigSchema(legacy_alias=42)
    assert cfg.base_acceleration == 4.0


# ---------------------------------------------------------------------------
# min_center_fraction (issue #550)
# ---------------------------------------------------------------------------


class TestMinCenterFractionSurvivesTheSchema:
    """The #534 ladder fix was landed in 47 YAMLs and dropped at load.

    ``min_center_fraction`` was never declared here, and this schema is
    ``extra="ignore"``, so ``from_yaml`` discarded it silently: the runtime saw
    a static ACS at ``center_fraction`` and every rung above ~1/center_fraction
    collapsed onto one mask. The physics fix, the process plumbing and the CI
    gate were all correct; the key just never arrived.
    """

    def test_declared_value_is_retained(self) -> None:
        cfg = AccelerationConfigSchema(center_fraction=0.08, min_center_fraction=0.02)
        assert cfg.min_center_fraction == 0.02

    def test_survives_model_dump(self) -> None:
        """``ModelFactory`` hands the generator a ``model_dump()``, not the object.

        A field that round-trips on the object but vanishes from the dump would
        reproduce the same bug one layer down.
        """
        dumped = AccelerationConfigSchema(min_center_fraction=0.02).model_dump()
        assert dumped["min_center_fraction"] == 0.02

    def test_defaults_to_none_meaning_static_acs(self) -> None:
        assert AccelerationConfigSchema().min_center_fraction is None

    def test_bounded_to_the_unit_interval(self) -> None:
        with pytest.raises(ValidationError):
            AccelerationConfigSchema(min_center_fraction=-0.01)
        with pytest.raises(ValidationError):
            AccelerationConfigSchema(min_center_fraction=1.5)

    def test_must_not_exceed_center_fraction(self) -> None:
        """A reversed pair describes a ramp that cannot run.

        The accelerator only ramps when ``min < center``; a larger minimum
        reads as configured and yields a static ACS, which is the exact #534
        failure wearing the fix's clothes. Raise instead (pitfall #9).
        """
        with pytest.raises(ValidationError, match="must be <="):
            AccelerationConfigSchema(center_fraction=0.02, min_center_fraction=0.08)

    def test_equal_is_allowed_as_explicit_static_acs(self) -> None:
        """15 arms set ``min == center`` deliberately; that is a valid opt-out."""
        cfg = AccelerationConfigSchema(center_fraction=0.08, min_center_fraction=0.08)
        assert cfg.min_center_fraction == 0.08


# ---------------------------------------------------------------------------
# Deprecated fields
# ---------------------------------------------------------------------------


def test_deprecated_fields_default_to_none() -> None:
    """Deprecated migration fields default to None."""
    cfg = AccelerationConfigSchema()
    assert cfg.mixed_precision is None
    assert cfg.gradient_accumulation_steps is None
    assert cfg.use_compile is None
    assert cfg.use_distributed is None
    assert cfg.use_gradient_checkpointing is None


# ---------------------------------------------------------------------------
# train_identity_rung (fully-sampled rung opt-in, issue #535)
# ---------------------------------------------------------------------------


class TestTrainIdentityRung:
    """The knob that makes ``R(0) == 1`` a trainable rung instead of a floor."""

    def test_defaults_off(self) -> None:
        """Every arm that does not ask for it keeps the old floor."""
        assert AccelerationConfigSchema().train_identity_rung is False

    def test_declared_true_survives_the_schema(self) -> None:
        assert AccelerationConfigSchema(train_identity_rung=True).train_identity_rung is True

    def test_declared_false_is_not_confused_with_absent(self) -> None:
        """A bool knob has nowhere to hide a falsy-``or`` bug.

        The resolver reads this with ``.get(key, default)`` rather than the
        ``or`` chaining the legacy numeric fields keep, because for a bool
        "declared False" and "absent" are the same token under ``or`` -- which
        is the whole value.
        """
        cfg = AccelerationConfigSchema(train_identity_rung=False)
        assert cfg.train_identity_rung is False
        assert "train_identity_rung" in cfg.model_dump()

    def test_description_states_the_precondition(self) -> None:
        """A reader must not expect it to do anything at ``R(0) > 1``.

        Pinned on the API-level condition, not on prose: the knob is a no-op
        wherever the floor is already 0, and a description that omits that
        sends someone hunting for an effect that cannot exist.
        """
        desc = AccelerationConfigSchema.model_fields["train_identity_rung"].description or ""
        assert "R(0) == 1" in desc
        assert "min_meaningful_timestep" in desc


def test_base_acceleration_below_one_is_refused() -> None:
    """Below 1.0 is not an acceleration; 1.0 means fully sampled (cohort review T0.3)."""
    with pytest.raises(ValidationError):
        AccelerationConfigSchema(base_acceleration=0.5)
    assert AccelerationConfigSchema(base_acceleration=1.0).base_acceleration == 1.0


class TestDeclaresNoAcceleration:
    """One predicate for "this block is the fully-sampled declaration"."""

    def test_base_and_max_at_one(self) -> None:
        assert AccelerationConfigSchema(
            base_acceleration=1.0, max_acceleration=1.0
        ).declares_no_acceleration

    def test_base_one_alone_is_a_ladder_to_the_default_max(self) -> None:
        assert not AccelerationConfigSchema(base_acceleration=1.0).declares_no_acceleration

    def test_a_declared_range_above_one_is_acceleration(self) -> None:
        cfg = AccelerationConfigSchema(
            base_acceleration=1.0, max_acceleration=1.0, acceleration_range=[1.0, 4.0]
        )
        assert not cfg.declares_no_acceleration

    def test_the_default_range_is_not_a_declaration(self) -> None:
        cfg = AccelerationConfigSchema(base_acceleration=1.0, max_acceleration=1.0)
        assert cfg.acceleration_range == [1.0, 16.0] and cfg.declares_no_acceleration

    def test_a_dynamic_mask_is_acceleration(self) -> None:
        cfg = AccelerationConfigSchema(
            base_acceleration=1.0, max_acceleration=1.0, enable_dynamic_mask=True
        )
        assert not cfg.declares_no_acceleration


@pytest.mark.parametrize("value", ["y", "x", None])
def test_retired_line_axis_is_rejected_at_load(value) -> None:
    """PLANTED (#2160): the block is extra="ignore", so a declared line_axis would
    otherwise vanish silently; the axis comes from data.acquisition."""
    with pytest.raises(ValidationError, match=r"data\.acquisition\.phase_encode_axis"):
        AccelerationConfigSchema(line_axis=value)
    assert "line_axis" not in AccelerationConfigSchema.model_fields


def test_adaptive_budget_rejects_unknown_value_and_rate_must_be_positive() -> None:
    import pydantic
    import pytest

    from spectramr.config.schemas.acceleration import AdaptiveAccelerationConfig

    assert AdaptiveAccelerationConfig().budget == "soft"
    assert AdaptiveAccelerationConfig(budget="hard_topk").budget == "hard_topk"
    with pytest.raises(pydantic.ValidationError):
        AdaptiveAccelerationConfig(budget="greedy")
    with pytest.raises(pydantic.ValidationError):
        AdaptiveAccelerationConfig(mask_learning_rate=0.0)
