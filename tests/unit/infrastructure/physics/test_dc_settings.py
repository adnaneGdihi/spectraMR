"""Unit tests for :mod:`spectramr.infrastructure.physics.dc_settings`.

The module exists to be the ONE owner of "which physics.data_consistency key
feeds which DC kwarg, and which dc_method actually reads it" (#1525). These
tests pin both halves, and the planted-violation cases below assert that the
detector discriminates rather than merely returning something.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectramr.infrastructure.physics.dc_settings import (
    DC_SSOT_KEYS,
    SUPPORTED_NOISE_TYPES,
    inert_dc_knobs,
    reads_knob,
    resolve_effective_dc,
    ssot_pairs,
)


def _dc(**overrides):
    """A physics.data_consistency block at the SCHEMA's defaults, plus overrides.

    Built from the real schema so a fixture cannot restate a default the schema
    has since moved, which is how the stale noise-level defaults went unseen.
    """
    from spectramr.config.schemas.physics import DataConsistencyConfig

    base = {
        field: info.default for field, info in DataConsistencyConfig.model_fields.items()
    }
    base.update({"enabled": True, "method": "hard"})
    base.update(overrides)
    return SimpleNamespace(**base)


def _config(dc=None, model_kwargs=None, with_physics=True):
    physics = SimpleNamespace(data_consistency=dc if dc is not None else _dc())
    return SimpleNamespace(
        model=SimpleNamespace(model_kwargs=dict(model_kwargs or {})),
        physics=physics if with_physics else None,
    )


class TestSSOTKeyTable:
    def test_every_ssot_row_names_a_real_schema_field(self) -> None:
        """The table is what makes a schema field reachable; a typo silently unwires it."""
        from spectramr.config.schemas.physics import DataConsistencyConfig

        fields = set(DataConsistencyConfig.model_fields)
        for kwarg, field in DC_SSOT_KEYS:
            assert field in fields, f"{kwarg} maps to non-existent field {field!r}"

    def test_the_noise_keys_are_forwarded(self) -> None:
        """#1525: these were validated, documented fields with no consumer."""
        forwarded = {kwarg for kwarg, _ in DC_SSOT_KEYS}
        assert {"train_noise_level", "eval_noise_level", "noise_type"} <= forwarded

    def test_ssot_pairs_skips_fields_the_object_lacks(self) -> None:
        partial = SimpleNamespace(enabled=True, method="hard")
        assert dict(ssot_pairs(partial)) == {"use_dc": True, "dc_method": "hard"}


class TestReadsKnob:
    @pytest.mark.parametrize(
        "method,expected",
        [
            ("hard", False),  # HardDataConsistency takes NO weight argument
            ("soft", True),  # lambda_init
            ("noise_adjusted", True),  # maps to soft
            ("noise_adaptive", True),  # beta
            ("target_aware_fsdc", True),  # hf_lambda
            ("wiener_gain", True),  # beta
            ("projection_2d_consistency", True),  # SimpleDataConsistency fallback
        ],
    )
    def test_dc_weight_readership(self, method: str, expected: bool) -> None:
        assert reads_knob(method, "dc_weight") is expected

    @pytest.mark.parametrize(
        "method,expected",
        [
            ("hard", True),
            ("soft", False),
            ("adaptive", False),
            ("kan_adaptive", False),
            ("wiener_gain", False),
        ],
    )
    def test_noise_level_readership(self, method: str, expected: bool) -> None:
        assert reads_knob(method, "train_noise_level") is expected

    @pytest.mark.parametrize("sentinel", [None, "", "none", "off", "disabled"])
    def test_disabled_dc_reads_nothing(self, sentinel) -> None:
        assert reads_knob(sentinel, "dc_weight") is False


class TestResolveEffectiveDC:
    def test_physics_block_is_the_ssot(self) -> None:
        cfg = _config(_dc(method="soft", weight=3.0), model_kwargs={"dc_method": "hard"})
        r = resolve_effective_dc(cfg)
        assert r.source == "physics"
        assert r.method == "soft"
        assert r.values["dc_weight"] == 3.0

    def test_model_kwargs_used_only_when_physics_absent(self) -> None:
        cfg = _config(with_physics=False, model_kwargs={"dc_method": "soft", "dc_weight": 2.0})
        r = resolve_effective_dc(cfg)
        assert r.source == "model_kwargs"
        assert r.method == "soft"
        assert r.values["dc_weight"] == 2.0

    @pytest.mark.parametrize("sentinel", ["", "none", "off", "disabled"])
    def test_disable_sentinels_clear_the_method(self, sentinel: str) -> None:
        r = resolve_effective_dc(_config(_dc(method=sentinel)))
        assert r.enabled is False
        assert r.method is None

    def test_enabled_false_disables(self) -> None:
        assert resolve_effective_dc(_config(_dc(enabled=False))).enabled is False


class TestInertKnobDetection:
    """The detector, one case per SHAPE it must see, plus the negatives."""

    def test_hard_plus_non_default_weight_is_reported(self) -> None:
        """The 54-arm corpus shape."""
        found = inert_dc_knobs(_config(_dc(method="hard", weight=0.5)))
        assert [k for k, _, _ in found] == ["dc_weight"]
        assert found[0][1] == 0.5

    def test_model_kwargs_declaration_is_seen_too(self) -> None:
        """Both declaration sites exist in the wild; a physics-only detector is half blind."""
        cfg = _config(with_physics=False, model_kwargs={"dc_method": "hard", "dc_weight": 0.5})
        assert [k for k, _, _ in inert_dc_knobs(cfg)] == ["dc_weight"]

    def test_noise_level_under_a_soft_method_is_reported(self) -> None:
        """SoftDataConsistency's constructor takes no noise parameters."""
        found = inert_dc_knobs(_config(_dc(method="soft", train_noise_level=0.09)))
        assert [k for k, _, _ in found] == ["train_noise_level"]

    # --- negatives: the detector must NOT fire on these -------------------
    def test_hard_plus_default_weight_is_silent(self) -> None:
        """Default 1.0 is not a choice the author made; reporting it is noise."""
        assert inert_dc_knobs(_config(_dc(method="hard", weight=1.0))) == ()

    def test_soft_plus_non_default_weight_is_silent(self) -> None:
        """soft READS dc_weight as lambda_init -- firing here would be a false positive."""
        assert inert_dc_knobs(_config(_dc(method="soft", weight=0.5))) == ()

    def test_hard_plus_non_default_noise_is_silent(self) -> None:
        """hard DC DOES simulate noise, so these are live knobs under it."""
        assert inert_dc_knobs(_config(_dc(method="hard", train_noise_level=0.09))) == ()

    @pytest.mark.parametrize(
        "method", ["soft", "adaptive", "kan_adaptive", "noise_adaptive", "target_aware_fsdc"]
    )
    def test_undeclared_noise_levels_are_silent_under_every_method(self, method: str) -> None:
        """PLANTED: the schema-default block the detector reported on every non-hard arm."""
        found = inert_dc_knobs(_config(_dc(method=method)))
        assert [k for k, _, _ in found if "noise" in k] == []

    def test_defaults_are_read_from_the_schema(self) -> None:
        from spectramr.config.schemas.physics import DataConsistencyConfig
        from spectramr.infrastructure.physics.dc_settings import schema_defaults

        fields = DataConsistencyConfig.model_fields
        assert schema_defaults() == {kw: fields[f].default for kw, f in DC_SSOT_KEYS}

    def test_disabled_dc_reports_nothing(self) -> None:
        assert inert_dc_knobs(_config(_dc(enabled=False, method="hard", weight=0.5))) == ()

    def test_unknown_method_reports_nothing(self) -> None:
        """The SimpleDataConsistency fallback accepts every knob here."""
        cfg = _config(_dc(method="projection_2d_consistency", weight=0.5))
        assert inert_dc_knobs(cfg) == ()


class TestSupportedNoiseTypes:
    def test_only_gaussian_is_implemented(self) -> None:
        """'rician' was advertised in the schema description and never implemented."""
        assert frozenset({"gaussian"}) == SUPPORTED_NOISE_TYPES


class TestApplyAtPredictIsNotAGeneratorKwarg:
    """``apply_at_predict`` is read by the inference strategies, so it must NOT be
    forwarded to the generator through ``DC_SSOT_KEYS``. A row added here would
    hand every DC-capable generator a kwarg none of them accepts."""

    def test_the_field_exists_and_is_absent_from_the_forwarding_table(self) -> None:
        from spectramr.config.schemas.physics import DataConsistencyConfig
        from spectramr.infrastructure.physics.dc_settings import DC_SSOT_KEYS

        assert "apply_at_predict" in DataConsistencyConfig.model_fields
        assert "apply_at_predict" not in {field for _, field in DC_SSOT_KEYS}
        assert "apply_at_predict" not in {kwarg for kwarg, _ in DC_SSOT_KEYS}

    def test_it_is_not_reported_as_an_inert_knob(self) -> None:
        """Inert-knob detection is per generator method; this knob has its own reader."""
        from types import SimpleNamespace

        from spectramr.config.schemas.physics import DataConsistencyConfig
        from spectramr.infrastructure.physics.dc_settings import inert_dc_knobs

        cfg = SimpleNamespace(
            model=SimpleNamespace(model_kwargs={}),
            physics=SimpleNamespace(
                data_consistency=DataConsistencyConfig(
                    enabled=True, method="hard", apply_at_predict=True
                )
            ),
        )
        assert inert_dc_knobs(cfg) == ()
