"""The signal-model registry: the nonlinear half of "declares a real forward model".

These tests pin the properties that make the registry safe to hang a maturity
claim on: it resolves, it refuses to guess, and it cannot silently shadow.
"""

from __future__ import annotations

import pytest

from spectramr.config.schemas.enums import Regime
from spectramr.infrastructure.physics.signal_models.registry import (
    SignalModelRegistry,
    SignalModelSpec,
    get_signal_model,
    list_signal_models,
    register_signal_model,
)


def test_importing_the_package_fires_every_registration() -> None:
    """list_signal_models() must not depend on who imported what first.

    A bare read of the class attribute would report "not registered" purely
    because nothing had imported the physics module yet — the maturity ledger
    would then silently downgrade a regime. See
    reference_import_time_table_vs_registry.
    """
    models = list_signal_models()
    assert "extended_tofts" in models
    assert "adc_monoexp" in models


def test_extended_tofts_is_registered_for_perfusion_with_its_real_parameters() -> None:
    spec = get_signal_model("extended_tofts")
    assert spec.regime is Regime.PERFUSION
    assert spec.parameters == ("ktrans", "ve", "vp")
    assert spec.reference  # a signal model without a citation is a guess


def test_adc_monoexp_is_registered_for_diffusion() -> None:
    spec = get_signal_model("adc_monoexp")
    assert spec.regime is Regime.DIFFUSION_WEIGHTED
    assert spec.parameters == ("s0", "adc")


def test_the_decorator_returns_the_function_unchanged() -> None:
    """Registration must not wrap the physics.

    The point of decorating the pure function is that the physics stays directly
    importable and testable without touching the registry. A wrapper would make
    every physics test go through registration machinery.
    """
    from spectramr.infrastructure.physics.signal_models.perfusion_kinetics import (
        extended_tofts_forward,
    )

    assert get_signal_model("extended_tofts").fn is extended_tofts_forward


def test_unknown_name_raises_and_never_returns_a_default() -> None:
    """A missing signal model means the physics is absent.

    There is no sensible default to substitute: quietly swapping in another
    kinetic model would refit to a different answer with a perfectly healthy
    residual (pitfall #9).
    """
    with pytest.raises(KeyError, match="not registered"):
        get_signal_model("no_such_model")


def test_the_error_lists_what_is_available() -> None:
    with pytest.raises(KeyError, match="extended_tofts"):
        get_signal_model("extended_toft")  # typo


def test_registering_a_duplicate_name_raises_rather_than_shadowing() -> None:
    """Two models under one name would dispatch by import order.

    Silently keeping the last writer means a config naming that model resolves to
    whichever module happened to be imported last — the exact class of bug the
    import-order note in the registry docstring warns about.
    """

    def _fake(t_s, aif, ktrans, ve, vp):
        return t_s

    with pytest.raises(ValueError, match="already registered"):
        SignalModelRegistry.register(
            SignalModelSpec(
                name="extended_tofts",
                fn=_fake,
                regime=Regime.PERFUSION,
                parameters=("ktrans", "ve", "vp"),
                signal="concentration_time_curve",
                reference="fake",
            )
        )


def test_re_registering_the_same_function_is_idempotent() -> None:
    """A module imported twice under different paths must not explode.

    Only a genuine collision — same name, different function — is an error.
    """
    spec = get_signal_model("extended_tofts")
    SignalModelRegistry.register(spec)  # must not raise
    assert get_signal_model("extended_tofts").fn is spec.fn


def test_for_regime_partitions_the_registry() -> None:
    perfusion = SignalModelRegistry.for_regime(Regime.PERFUSION)
    assert "extended_tofts" in perfusion
    assert "adc_monoexp" not in perfusion


def test_two_models_can_share_a_regime_with_different_parameter_contracts() -> None:
    """Sharing a regime does NOT make two signal models interchangeable.

    gamma_variate is PERFUSION physics too, but its signature is
    (t_s, amplitude, t0_s, alpha, beta_s). A consumer dispatching on a config
    knob that checked only the regime would bind Tofts kinetics to a
    gamma-variate's amplitude/onset and get a plausible curve back. This is why
    SignalModelSpec.parameters exists and why the perfusion strategy checks it.
    """
    perfusion = SignalModelRegistry.for_regime(Regime.PERFUSION)
    assert {"extended_tofts", "gamma_variate"} <= set(perfusion)
    assert (
        perfusion["extended_tofts"].parameters != perfusion["gamma_variate"].parameters
    )


def test_the_registry_contract_has_no_adjoint() -> None:
    """The whole reason this registry is separate from OperatorRegistry.

    An OperatorRegistry entry promises adjoint(), and generic callers
    (OperatorProjectionDataConsistency, FISTAMBIRSolver, NullSpaceProjection) consume that
    promise assuming <Ax,y> = <x,A'y>. These maps are nonlinear and have none, so
    the contract must not offer a slot where a fake could be supplied.
    """
    assert not hasattr(SignalModelSpec, "adjoint")
    assert "adjoint" not in SignalModelSpec.__dataclass_fields__


def test_register_signal_model_decorator_records_metadata() -> None:
    @register_signal_model(
        name="_test_only_model",
        regime=Regime.SPECTROSCOPIC,
        parameters=("a",),
        signal="test",
        reference="none",
    )
    def _model(x):
        return x

    try:
        spec = get_signal_model("_test_only_model")
        assert spec.fn is _model
        assert spec.regime is Regime.SPECTROSCOPIC
    finally:
        SignalModelRegistry._models.pop("_test_only_model", None)
