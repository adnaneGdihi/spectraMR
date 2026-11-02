"""HybridQMapGenerator refuses test-time-optimisation knobs it cannot run (#2307)."""

from __future__ import annotations

import pytest

from spectramr.models.generators.hybrid_pinn import HybridQMapGenerator


@pytest.mark.parametrize(
    "knob", [{"enable_tto": True}, {"tto_steps": 5}, {"tto_lambda_div_free": 1.0}]
)
def test_tto_knobs_raise_instead_of_being_absorbed(knob: dict) -> None:
    with pytest.raises(ValueError, match="test-time optimisation"):
        HybridQMapGenerator(input_dim=1, feature_dim=8, **knob)


def test_without_tto_knobs_the_model_builds() -> None:
    model = HybridQMapGenerator(in_channels=2, feature_dim=8)
    assert model.input_dim == 2


def test_the_audit_rejects_enable_tto_at_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Planted: ``enable_tto: true`` on hybrid_pinn is an advertised_options error, not a build crash."""
    from types import SimpleNamespace

    import spectramr.models.registry as registry_mod
    from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker

    # The registry is empty until populated; resolve the alias to the class under test.
    monkeypatch.setattr(registry_mod, "get_model_class", lambda name: HybridQMapGenerator)

    def results(kwargs: dict) -> list:
        cfg = SimpleNamespace(model=SimpleNamespace(model_type="hybrid_pinn", model_kwargs=kwargs))
        return ConfigHealthChecker().check_advertised_options(cfg)

    errors = [r for r in results({"enable_tto": True}) if not r.passed and r.severity == "error"]
    assert errors and "enable_tto" in errors[0].message
    assert all(r.passed for r in results({"enable_tto": False}))
