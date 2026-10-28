"""Regression: three ``config_health_checker`` checks were silently broken.

2026-06-11 infrastructure audit.

* ``check_declared_losses_registered`` compared the raw-case YAML loss name
  against an all-lowercase registry set, so CamelCase registered aliases
  (``MMDLoss``, ``SSDULoss``, …) false-failed the audit at *error* severity.
* ``check_field_strength_declared`` did ``getattr(<dict>, "tags")`` on the
  dict-typed ``metadata`` field, so ULF detection was always False (false-pass).
* ``check_acceleration_present`` matched only the literal ``"kspace"``, so the
  more common k-space dataset types (``m4raw`` / ``fastmri_kspace`` / …) were
  silently exempted from the "must declare acceleration" rule (false-pass).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

from spectramr.infrastructure.validation.config_health_checker import (
    ConfigHealthChecker,
)


def _checker_with_registry(loss_names):
    checker = ConfigHealthChecker()
    # Short-circuit _lazy_load_registries (it returns early when the model
    # registry is already populated) and inject a known loss registry.
    checker._model_registry = set()
    checker._strategy_registry = set()
    checker._loss_registry = {n.lower() for n in loss_names}
    return checker


def test_declared_losses_registered_accepts_camelcase_alias():
    checker = _checker_with_registry({"mmdloss", "ssduloss"})
    cfg = SimpleNamespace(
        losses=SimpleNamespace(
            image_losses=[SimpleNamespace(name="MMDLoss", enabled=True)],
            kspace_losses=[],
            complex_losses=[],
        )
    )
    results = checker.check_declared_losses_registered(cfg)
    failures = [r for r in results if not r.passed]
    assert not failures, f"CamelCase alias false-failed: {[r.message for r in failures]}"


def test_declared_losses_registered_still_rejects_truly_unknown():
    checker = _checker_with_registry({"mmdloss"})
    cfg = SimpleNamespace(
        losses=SimpleNamespace(
            image_losses=[SimpleNamespace(name="DefinitelyNotALoss", enabled=True)],
            kspace_losses=[],
            complex_losses=[],
        )
    )
    results = checker.check_declared_losses_registered(cfg)
    assert any(not r.passed for r in results)


def test_field_strength_declared_fires_for_ulf_dict_metadata():
    checker = ConfigHealthChecker()
    cfg = SimpleNamespace(
        physics=SimpleNamespace(field_strength=None),
        metadata={"tags": ["ulf", "translation"]},
    )
    result = checker.check_field_strength_declared(cfg)
    assert result.passed is False
    assert result.severity == "error"


def test_field_strength_declared_passes_when_b0_present():
    checker = ConfigHealthChecker()
    cfg = SimpleNamespace(
        physics=SimpleNamespace(field_strength=0.3),
        metadata={"tags": ["ulf"]},
    )
    assert checker.check_field_strength_declared(cfg).passed is True


def test_acceleration_present_covers_full_kspace_family():
    checker = ConfigHealthChecker()
    for ds in ("m4raw", "fastmri_kspace", "bart_kspace", "ismrmrd_kspace"):
        cfg = SimpleNamespace(data=SimpleNamespace(dataset_type=ds))
        # No ``acceleration`` block → must NOT be silently exempted.
        result = checker.check_acceleration_present(cfg)
        assert result.passed is False, f"{ds} was wrongly exempted from acceleration check"


def test_acceleration_present_skips_non_kspace():
    checker = ConfigHealthChecker()
    cfg = SimpleNamespace(data=SimpleNamespace(dataset_type="image_pairs"))
    assert checker.check_acceleration_present(cfg).passed is True


# ── check_advertised_options — de-vacuum (2026-06-12 follow-up) ────────
#
# The pitfall-#9 guard was vacuous: no model defines OPTION_SCHEMA, so the
# check always passed with a reassuring "all in advertised set" line while
# validating nothing. Two upgrades, both pinned here:
#   1. The OPTION_SCHEMA error path works when a model DOES declare one.
#   2. NEW signature advisory: a model_kwargs key that is neither an explicit
#      __init__ parameter nor absorbable by **kwargs is certainly swallowed
#      by the factory's signature filter (pitfall #16 facade) or a TypeError
#      at build — surfaced at *info* severity (advisory; never blocks the
#      467-arm audit without a cluster-wide blast-radius pass).


class _SchemaModel:
    OPTION_SCHEMA: ClassVar[dict] = {"attention_type": ["spatial", "cross"]}

    def __init__(self, in_channels=1, attention_type="spatial"):
        pass


class _NoKwargsModel:
    def __init__(self, in_channels=1, depth=4):
        pass


class _VarKwargsModel:
    def __init__(self, in_channels=1, **kwargs):
        pass


def _cfg_with_kwargs(model_type, kwargs):
    return SimpleNamespace(
        model=SimpleNamespace(model_type=model_type, model_kwargs=kwargs)
    )


def _patched_checker(monkeypatch, cls):
    import spectramr.models.registry as registry_mod

    monkeypatch.setattr(registry_mod, "get_model_class", lambda name: cls)
    return ConfigHealthChecker()


def test_option_schema_violation_is_error(monkeypatch):
    checker = _patched_checker(monkeypatch, _SchemaModel)
    cfg = _cfg_with_kwargs("m", {"attention_type": "dual_domain"})
    results = checker.check_advertised_options(cfg)
    errors = [r for r in results if not r.passed and r.severity == "error"]
    assert errors, "OPTION_SCHEMA violation must be a hard error"
    assert "dual_domain" in errors[0].message


def test_unknown_kwarg_on_no_varkwargs_model_is_advisory(monkeypatch):
    checker = _patched_checker(monkeypatch, _NoKwargsModel)
    cfg = _cfg_with_kwargs("m", {"attention_type": "spatial"})
    results = checker.check_advertised_options(cfg)
    advisories = [r for r in results if r.severity == "info" and not r.passed]
    assert advisories, (
        "an unknown kwarg on a no-**kwargs model must surface a "
        "possibly-swallowed advisory (pitfall #16)"
    )
    assert "attention_type" in advisories[0].message


def test_explicit_param_kwarg_passes_clean(monkeypatch):
    checker = _patched_checker(monkeypatch, _NoKwargsModel)
    cfg = _cfg_with_kwargs("m", {"depth": 6})
    results = checker.check_advertised_options(cfg)
    assert all(r.passed for r in results)


class _ReadsKnobsStrategy:
    reads_model_kwargs: ClassVar[frozenset[str]] = frozenset({"gain_mode"})


def _with_strategy(monkeypatch, checker, strategy_cls):
    monkeypatch.setattr(ConfigHealthChecker, "_resolve_strategy_class", staticmethod(lambda c: strategy_cls))
    return checker


def test_a_key_the_strategy_reads_is_not_a_facade_knob(monkeypatch):
    # The q-map strategy reads gain_mode from model_kwargs; the advisory flagged it on
    # bloch_parameter_bottleneck in every q-map run (sabine job 17889904).
    checker = _with_strategy(
        monkeypatch, _patched_checker(monkeypatch, _NoKwargsModel), _ReadsKnobsStrategy
    )
    results = checker.check_advertised_options(_cfg_with_kwargs("m", {"gain_mode": "projected"}))
    assert all(r.passed for r in results)


def test_a_key_neither_the_model_nor_the_strategy_reads_is_still_flagged(monkeypatch):
    checker = _with_strategy(
        monkeypatch, _patched_checker(monkeypatch, _NoKwargsModel), _ReadsKnobsStrategy
    )
    cfg = _cfg_with_kwargs("m", {"gain_mode": "projected", "typo_knob": 1})
    flagged = [r for r in checker.check_advertised_options(cfg) if not r.passed]
    assert [r.yaml_keys for r in flagged] == [["model.model_kwargs.typo_knob"]]


def test_a_key_outside_the_declared_readers_is_an_error(monkeypatch):
    # The model's signature and the strategy's declaration name every reader, so a key in
    # neither is certainly unread: `contrast_droput: 0.3` trained silently with p = 0.
    checker = _with_strategy(
        monkeypatch, _patched_checker(monkeypatch, _NoKwargsModel), _ReadsKnobsStrategy
    )
    cfg = _cfg_with_kwargs("m", {"gain_mode": "projected", "contrast_droput": 0.3})
    flagged = [r for r in checker.check_advertised_options(cfg) if not r.passed]
    assert [r.severity for r in flagged] == ["error"]


def test_a_strategy_without_the_declaration_changes_nothing(monkeypatch):
    checker = _with_strategy(monkeypatch, _patched_checker(monkeypatch, _NoKwargsModel), object)
    results = checker.check_advertised_options(_cfg_with_kwargs("m", {"gain_mode": "projected"}))
    assert [r.severity for r in results if not r.passed] == ["info"]


def test_varkwargs_model_skips_signature_advisory(monkeypatch):
    """A **kwargs signature can absorb anything — nothing to assert statically."""
    checker = _patched_checker(monkeypatch, _VarKwargsModel)
    cfg = _cfg_with_kwargs("m", {"anything_goes": 1})
    results = checker.check_advertised_options(cfg)
    assert all(r.passed for r in results)
