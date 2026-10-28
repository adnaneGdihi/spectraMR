"""Unit tests for the ``physics:`` config block.

Scope note: this file guards the *advertised set* of physics knobs against the
implementation behind them. The schema layer cannot import ``infrastructure``
(the layer rule points inward only), so a field description naming a vocabulary
is an unenforced claim — exactly the shape of pitfall #15. A test can import
both sides, so the claim is checked here.
"""

from __future__ import annotations

from spectramr.config.schemas.physics import DigitalTwinConfig


class TestProgressiveDegradationsAdvertisedSet:
    """``DigitalTwinConfig.progressive_degradations`` documents two banks of
    names. Both must exist, and the simulator must accept exactly what the
    description promises."""

    @staticmethod
    def _description() -> str:
        return DigitalTwinConfig.model_fields["progressive_degradations"].description

    def test_every_native_axis_named_in_the_description_exists(self) -> None:
        from spectramr.infrastructure.physics.digital_twin_simulator import (
            NATIVE_DEGRADATION_AXES,
        )

        text = self._description()
        missing = sorted(a for a in NATIVE_DEGRADATION_AXES if a not in text)
        assert not missing, (
            f"native axes absent from the advertised set: {missing}. The "
            "description enumerates all 14; adding an axis must update it."
        )

    def test_the_description_makes_no_registry_size_claim(self) -> None:
        """A registry count in the docs is wrong the moment the registry grows, so
        the description names the registry without counting it."""
        import re

        text = self._description()
        assert "infrastructure.physics.degradations" in text
        claims = re.findall(r"\b\d+\s+(?:keys|entries|operators)\b", text)
        assert not claims, f"progressive_degradations description counts the registry: {claims}"

    def test_registry_examples_in_the_description_are_real_keys(self) -> None:
        from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY

        text = self._description()
        cited = [
            "rigid_motion",
            "pulsatile_motion",
            "rician",
            "spike",
            "nyquist_ghost",
            "susceptibility",
            "t2star_blur",
            "partial_fourier",
        ]
        assert all(c in text for c in cited), "test's own citation list drifted"
        unknown = [c for c in cited if c not in DEGRADATION_REGISTRY]
        assert not unknown, (
            f"the description cites degradations the registry does not have: "
            f"{unknown} — an advertised knob with no implementation"
        )

    def test_default_is_reachable(self) -> None:
        """The shipped default must itself be constructible — it was, but only
        because all four names happened to be native."""
        from spectramr.infrastructure.physics.digital_twin_simulator import (
            known_degradation_axes,
        )

        known = known_degradation_axes()
        assert set(DigitalTwinConfig().progressive_degradations) <= known
        assert set(DigitalTwinConfig().degradation_ranges) <= known


class TestDataConsistencyNoiseVocabulary:
    """The schema must advertise the SHIPPED vocabulary, not the designed one (#1525).

    ``noise_type``'s description named ``'rician'`` for a long time; no DC layer
    has ever implemented it, and two of the three silently degraded such a value
    to Gaussian instead of raising.
    """

    def test_noise_type_description_does_not_advertise_an_unimplemented_model(self) -> None:
        from spectramr.config.schemas.physics import DataConsistencyConfig
        from spectramr.infrastructure.physics.dc_settings import SUPPORTED_NOISE_TYPES

        description = DataConsistencyConfig.model_fields["noise_type"].description or ""
        for shipped in SUPPORTED_NOISE_TYPES:
            assert shipped in description
        # 'rician' may be MENTIONED, but only as unimplemented.
        if "rician" in description:
            assert "never had an implementation" in description

    def test_default_noise_type_is_one_the_layers_implement(self) -> None:
        from spectramr.config.schemas.physics import DataConsistencyConfig
        from spectramr.infrastructure.physics.dc_settings import SUPPORTED_NOISE_TYPES

        assert DataConsistencyConfig().noise_type in SUPPORTED_NOISE_TYPES

    def test_noise_level_defaults_match_the_layer_defaults(self) -> None:
        """A divergence here is a silent default substitution once the knob is wired."""
        import inspect

        from spectramr.config.schemas.physics import DataConsistencyConfig
        from spectramr.infrastructure.physics.data_consistency import HardDataConsistency

        cfg = DataConsistencyConfig()
        params = inspect.signature(HardDataConsistency.__init__).parameters
        assert cfg.train_noise_level == params["train_noise_level"].default
        assert cfg.eval_noise_level == params["eval_noise_level"].default


class TestOODAccelerationRange:
    """``physics.digital_twin.ood_acceleration_range`` (VF review 2026-09-03)."""

    def test_default_is_absent(self) -> None:
        assert DigitalTwinConfig().ood_acceleration_range is None

    def test_an_ascending_range_above_one_is_kept_as_floats(self) -> None:
        cfg = DigitalTwinConfig(
            enabled=True, enable_undersampling=True, ood_acceleration_range=[16, 32]
        )
        assert cfg.ood_acceleration_range == [16.0, 32.0]

    def test_a_range_on_a_twin_that_does_not_undersample_fails_at_load(self) -> None:
        """Planted violation: the shape 31 VF arms had as ``undersampling.out_of_distribution_range``."""
        import pytest

        with pytest.raises(ValueError, match="enable_undersampling is false"):
            DigitalTwinConfig(enabled=True, ood_acceleration_range=[16.0, 32.0])

    def test_a_range_on_a_disabled_twin_fails_at_load(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="`enabled` is false"):
            DigitalTwinConfig(enable_undersampling=True, ood_acceleration_range=[16.0])

    def test_an_empty_list_is_rejected(self) -> None:
        """Planted violation: a typo, not 'no rungs'."""
        import pytest

        with pytest.raises(ValueError, match="is empty"):
            DigitalTwinConfig(ood_acceleration_range=[])

    def test_a_rate_at_or_below_one_is_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="not above 1.0"):
            DigitalTwinConfig(ood_acceleration_range=[1.0, 16.0])

    def test_a_non_ascending_range_is_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="strictly ascending"):
            DigitalTwinConfig(ood_acceleration_range=[32.0, 16.0])
        with pytest.raises(ValueError, match="strictly ascending"):
            DigitalTwinConfig(ood_acceleration_range=[16.0, 16.0])


class TestApplyAtPredict:
    """``physics.data_consistency.apply_at_predict`` (predict-time hard DC)."""

    def test_defaults_to_off(self) -> None:
        from spectramr.config.schemas.physics import DataConsistencyConfig

        assert DataConsistencyConfig().apply_at_predict is False

    def test_loads_when_declared(self) -> None:
        from spectramr.config.schemas.physics import DataConsistencyConfig, PhysicsConfigSchema

        assert DataConsistencyConfig(apply_at_predict=True).apply_at_predict is True
        schema = PhysicsConfigSchema(data_consistency={"apply_at_predict": True})
        assert schema.data_consistency.apply_at_predict is True

    def test_an_unknown_value_is_refused(self) -> None:
        import pytest
        from pydantic import ValidationError

        from spectramr.config.schemas.physics import DataConsistencyConfig

        with pytest.raises(ValidationError):
            DataConsistencyConfig(apply_at_predict="maybe")

    def test_is_independent_of_enabled(self) -> None:
        """The knob is read by inference, not by the training-time layer."""
        from spectramr.config.schemas.physics import DataConsistencyConfig

        cfg = DataConsistencyConfig(enabled=False, apply_at_predict=True)
        assert cfg.apply_at_predict is True and cfg.enabled is False

    def test_the_description_states_the_two_contracts(self) -> None:
        from spectramr.config.schemas.physics import DataConsistencyConfig

        text = DataConsistencyConfig.model_fields["apply_at_predict"].description or ""
        assert "eval_noise_level" in text, "the noise-not-added rule must be stated"
        assert "DC_SSOT_KEYS" in text, "it is not a generator kwarg; say so"
        assert "raises" in text, "missing inputs raise, never skip"
