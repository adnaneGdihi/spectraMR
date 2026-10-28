"""Unit tests for the registry helpers in :mod:`spectramr.models.registry`.

The integrity / contract tests in
:mod:`tests.unit.models.test_registry_integrity` exercise the full
production registry. This file instead targets the *helper* functions in
isolation:

* ``register_model`` decorator semantics (success, overwrite-rejection,
  and rejection of a re-registration that DROPS a declared capability)
* ``get_model_class`` / ``get_model_mode`` error messages
* ``model_supports`` returning False on unknown models / capabilities
* ``get_model_capabilities`` distinguishing "unannotated" (returns None)
  from "annotated as not-supported" (returns the dataclass)
* ``list_models_with_capability``

We back the registry up around each test so we don't poison the
production singleton.
"""

from __future__ import annotations

import pytest

import spectramr.models.registry as reg_mod
from spectramr.models.capabilities import ModelCapabilities
from spectramr.models.init_registry import populate_model_registry
from spectramr.models.registry import (
    MODEL_REGISTRY,
    get_model_capabilities,
    get_model_class,
    get_model_mode,
    list_models,
    list_models_with_capability,
    model_supports,
    register_model,
)

# ── Fixture: clean registry per-test ───────────────────────────────


@pytest.fixture
def clean_registry() -> dict:
    """Snapshot MODEL_REGISTRY, clear, then restore — keeps the
    production registry untouched after every test in this module.

    Triggers auto-discovery before snapshotting so the backup actually
    contains the production model set. Without this, if a test in this
    module is the first to access ``MODEL_REGISTRY`` (test ordering is
    not deterministic), the backup captures an empty dict and the
    "restore" leaves the registry empty for the rest of the session —
    poisoning every later test that depends on auto-discovered models
    (e.g. ``test_registry_contract.py``, ~1500 parametrized tests).

    The paragraph above stated the hazard correctly and then defended
    against it with the wrong mechanism for as long as it existed; see
    the comment below (#1961).
    """
    # Populate by CALLING, not by importing. ``import init_registry`` only
    # binds the module: measured in a fresh process, MODEL_REGISTRY is 0 both
    # before and after it, so the snapshot below captured ``{}`` and the
    # "restore" emptied the registry -- the exact poisoning the docstring
    # above set out to prevent.
    #
    # There is no recovery once that happens. The ``_REGISTRY_POPULATED``
    # latch makes a later ``populate_model_registry()`` a no-op, and
    # ``force=True`` does not help either, because the ``@register_model``
    # decorators fire at module *import* time and the model modules are
    # already in ``sys.modules``. Measured: 588 -> clear -> 0, populate() 0,
    # populate(force=True) 0. So the snapshot must be non-empty or the
    # session is unrecoverable, hence the assert.
    populate_model_registry()

    backup = dict(MODEL_REGISTRY)
    assert backup, (
        "MODEL_REGISTRY is empty after populate_model_registry(); this fixture "
        "would 'restore' an empty registry and poison every later test that "
        "depends on auto-discovered models, with no way back in this process "
        "(#1961)"
    )
    MODEL_REGISTRY.clear()
    try:
        yield MODEL_REGISTRY
    finally:
        MODEL_REGISTRY.clear()
        MODEL_REGISTRY.update(backup)


def _new_cls(name: str = "_FakeModel") -> type:
    """Build a unique, throwaway class for registration tests."""
    return type(name, (), {})


# ── register_model success path ────────────────────────────────────


class TestRegisterModelSuccess:
    def test_registration_stores_class_mode_and_capabilities(self, clean_registry: dict) -> None:
        cls = _new_cls()
        decorated = register_model("alpha", training_mode="gan")(cls)
        assert decorated is cls  # decorator returns the class unchanged
        assert "alpha" in clean_registry
        entry = clean_registry["alpha"]
        assert entry["class"] is cls
        assert entry["mode"] == "gan"
        # Capabilities dataclass is always present, default-empty here.
        assert isinstance(entry["capabilities"], ModelCapabilities)
        # One owner (#1916): capability flags live ONLY on the nested
        # dataclass. The entry carries no ad-hoc top-level *capability* keys,
        # which is what let two readers of one registry disagree.
        # ``role`` (#1932) is a routing key, not a capability: it says which
        # bucket ModelFactory files the class under, it is not a claim about
        # what the model can do, and nothing reads it through
        # ``model_supports``. Pinning the exact key set is what makes a
        # re-introduced ad-hoc flag fail here instead of drifting in quietly.
        assert set(entry) == {"class", "mode", "role", "capabilities"}
        assert entry["role"] == "generator"
        # Undeclared is None ("unannotated"), NOT False. False is a positive
        # claim that the model ignores the id; None means nobody said.
        assert entry["capabilities"].supports_contrast_conditioning is None
        assert entry["capabilities"].supports_vendor_conditioning is None

    def test_capability_flags_round_trip_through_register(self, clean_registry: dict) -> None:
        cls = _new_cls()
        register_model(
            "beta",
            training_mode="diffusion",
            supports_contrast_conditioning=True,
            spatial_dims=(2,),
            input_domain="kspace",
            output_domain="image",
            accepts_complex=True,
            expects_real_imag_interleaved=False,
            requires_paired_data=True,
        )(cls)
        entry = clean_registry["beta"]
        caps = entry["capabilities"]
        assert caps.supports_contrast_conditioning is True
        assert caps.spatial_dims == (2,)
        assert caps.input_domain == "kspace"
        assert caps.output_domain == "image"
        assert caps.accepts_complex is True
        assert caps.expects_real_imag_interleaved is False
        assert caps.requires_paired_data is True

    def test_same_class_re_registration_is_idempotent(self, clean_registry: dict) -> None:
        """Re-decorating the *same* class is allowed — common in test
        reloads / import cycles."""
        cls = _new_cls()
        register_model("gamma", training_mode="gan")(cls)
        # Decorate again — should not raise.
        register_model("gamma", training_mode="gan")(cls)
        assert clean_registry["gamma"]["class"] is cls


# ── register_model duplicate rejection ─────────────────────────────


class TestRegisterModelDuplicateRejection:
    def test_overwriting_with_different_class_raises(self, clean_registry: dict) -> None:
        cls_a = _new_cls("ModelA")
        cls_b = _new_cls("ModelB")
        register_model("dup", training_mode="gan")(cls_a)
        with pytest.raises(ValueError) as ei:
            register_model("dup", training_mode="diffusion")(cls_b)
        msg = str(ei.value)
        # Error names both classes and modes for debuggability.
        assert "ModelA" in msg
        assert "ModelB" in msg
        # And explicitly says it's refusing.
        assert "refusing to overwrite" in msg


# ── get_model_class / get_model_mode ───────────────────────────────


class TestLookups:
    def test_get_model_class_returns_class(self, clean_registry: dict) -> None:
        cls = _new_cls()
        register_model("alpha", training_mode="gan")(cls)
        assert get_model_class("alpha") is cls

    def test_get_model_class_unknown_lists_available(self, clean_registry: dict) -> None:
        register_model("alpha", training_mode="gan")(_new_cls())
        with pytest.raises(ValueError) as ei:
            get_model_class("does_not_exist")
        msg = str(ei.value)
        assert "does_not_exist" in msg
        assert "alpha" in msg

    def test_get_model_mode(self, clean_registry: dict) -> None:
        register_model("alpha", training_mode="reconstruction")(_new_cls())
        assert get_model_mode("alpha") == "reconstruction"

    def test_get_model_mode_unknown_raises(self, clean_registry: dict) -> None:
        with pytest.raises(ValueError):
            get_model_mode("does_not_exist")


# ── model_supports / list_models_with_capability ───────────────────


class TestCapabilityFlags:
    def test_model_supports_returns_true_for_registered_flag(self, clean_registry: dict) -> None:
        register_model(
            "supports_cc",
            training_mode="gan",
            supports_contrast_conditioning=True,
        )(_new_cls("SupportsCC"))
        assert model_supports("supports_cc", "supports_contrast_conditioning") is True

    def test_model_supports_returns_false_for_disabled_flag(self, clean_registry: dict) -> None:
        register_model(
            "no_cc",
            training_mode="gan",
            supports_contrast_conditioning=False,
        )(_new_cls("NoCC"))
        assert model_supports("no_cc", "supports_contrast_conditioning") is False

    def test_model_supports_unknown_model_returns_false(self, clean_registry: dict) -> None:
        # Unknown MODEL silently returns False — the *only* silent answer
        # this helper still gives, because the audit layer owns the loud
        # failure for a bad model_type. Note the capability name must be a
        # real one; an unknown capability raises (see below), and that check
        # runs first.
        assert model_supports("ghost", "accepts_complex") is False

    def test_model_supports_unknown_capability_raises(self, clean_registry: dict) -> None:
        """An unknown flag name raises instead of answering False (#1916).

        The old implementation read the registry entry's top-level keys with
        ``entry.get(capability, False)``, so a typo'd flag and a model that
        genuinely lacks the capability produced the same confident "no". That
        is the silent-fallback shape non-negotiable 3 forbids, and it was
        load-bearing: every nested flag name answered False for every model.
        """
        register_model("alpha", training_mode="gan")(_new_cls())
        with pytest.raises(ValueError, match="Unknown model capability"):
            model_supports("alpha", "supports_quantum_field_theory")

    def test_model_supports_rejects_non_boolean_capability(self, clean_registry: dict) -> None:
        """A non-boolean capability field is not a yes/no question.

        ``spatial_dims=(2, 3)`` is truthy, so a naive truthiness read would
        answer True to "does this model support spatial_dims?" — a confident
        answer to a question nobody asked.
        """
        register_model("alpha", training_mode="gan", spatial_dims=(2, 3))(_new_cls())
        with pytest.raises(ValueError, match="Unknown model capability"):
            model_supports("alpha", "spatial_dims")

    def test_model_supports_reads_nested_capabilities(self, clean_registry: dict) -> None:
        """The regression #1916 names: a nested-only flag was invisible.

        ``accepts_complex`` is declared exclusively through the capability
        dataclass. Before the election, ``model_supports`` read the top-level
        keys and returned False for all 18 models on ``dev`` that declare it,
        while ``get_model_capabilities`` returned the right answer — two
        owners, zero overlap, no error from either.
        """
        register_model("cplx", training_mode="gan", accepts_complex=True)(_new_cls("Cplx"))
        assert model_supports("cplx", "accepts_complex") is True
        assert list_models_with_capability("accepts_complex") == ["cplx"]

    def test_model_supports_and_list_cannot_disagree(self, clean_registry: dict) -> None:
        """The two readers answer from one owner, so they agree by construction."""
        register_model("yes", training_mode="gan", requires_paired_data=True)(_new_cls("Yes"))
        register_model("no", training_mode="gan", requires_paired_data=False)(_new_cls("No"))
        register_model("unset", training_mode="gan")(_new_cls("Unset"))
        listed = set(list_models_with_capability("requires_paired_data"))
        predicated = {n for n in clean_registry if model_supports(n, "requires_paired_data")}
        assert listed == predicated == {"yes"}

    def test_list_models_with_capability_filters(self, clean_registry: dict) -> None:
        register_model("a", training_mode="gan", supports_contrast_conditioning=True)(_new_cls("A"))
        register_model("b", training_mode="gan", supports_contrast_conditioning=False)(
            _new_cls("B")
        )
        register_model("c", training_mode="gan", supports_contrast_conditioning=True)(_new_cls("C"))

        out = list_models_with_capability("supports_contrast_conditioning")
        assert set(out) == {"a", "c"}


# ── get_model_capabilities ─────────────────────────────────────────


class TestGetModelCapabilities:
    def test_unannotated_returns_none(self, clean_registry: dict) -> None:
        """A model registered with no capability hints is treated as
        unannotated — get_model_capabilities returns None so the audit
        ladder can opt out gracefully."""
        register_model("unannotated", training_mode="gan")(_new_cls())
        assert get_model_capabilities("unannotated") is None

    def test_annotated_returns_dataclass(self, clean_registry: dict) -> None:
        register_model(
            "annotated",
            training_mode="gan",
            spatial_dims=(2,),
            input_domain="image",
            output_domain="image",
            accepts_complex=False,
        )(_new_cls())
        caps = get_model_capabilities("annotated")
        assert caps is not None
        assert caps.spatial_dims == (2,)
        assert caps.input_domain == "image"

    def test_unknown_model_returns_none(self, clean_registry: dict) -> None:
        assert get_model_capabilities("ghost") is None


# ── list_models ────────────────────────────────────────────────────


class TestListModels:
    def test_list_models_is_a_copy(self, clean_registry: dict) -> None:
        register_model("alpha", training_mode="gan")(_new_cls())
        snapshot = list_models()
        # Mutating the snapshot must not mutate the registry.
        snapshot.clear()
        assert "alpha" in MODEL_REGISTRY

    def test_list_models_returns_all_entries(self, clean_registry: dict) -> None:
        register_model("a", training_mode="gan")(_new_cls("A"))
        register_model("b", training_mode="diffusion")(_new_cls("B"))
        out = list_models()
        assert set(out) == {"a", "b"}
        assert out["a"]["mode"] == "gan"
        assert out["b"]["mode"] == "diffusion"


# ── Sanity: module-level constants ─────────────────────────────────


def test_module_exports_required_names() -> None:
    """The module's public surface includes the symbols the audit
    layer imports — keep this guard in place so refactors that rename
    a helper get caught immediately."""
    for name in (
        "MODEL_REGISTRY",
        "register_model",
        "get_model_class",
        "get_model_mode",
        "list_models",
        "model_supports",
        "get_model_capabilities",
        "list_models_with_capability",
    ):
        assert hasattr(reg_mod, name), f"Missing public name: {name}"


# ── The re-registration downgrade guard (#1916) ────────────────────


class TestReRegistrationRefusesADowngrade:
    """The guard must key on a capability being DROPPED, not on the new
    registration being empty.

    Until #1916 the condition was ``capabilities == ModelCapabilities()``
    -- "the second registration declares nothing at all". That was a
    *proxy* for a downgrade, and it was only faithful while the dataclass
    carried nothing but the three contract fields: any field set made the
    caps non-empty, and every field was one you would be sorry to lose.

    #1916 adds ``supports_contrast_conditioning`` /
    ``supports_vendor_conditioning`` to that dataclass, which breaks the
    proxy in the worst possible direction. The single most plausible
    partial re-registration -- "same model, just add the conditioning
    flag" -- became non-empty, so the old condition waved it through and
    silently reset ``spatial_dims`` / ``input_domain`` / ``output_domain``
    to None. That was measured on this branch before the fix: the
    re-registration raised on origin/dev and overwrote here.

    One planted violation per shape the rule can take: the partial
    downgrade the widening created, the total downgrade the old condition
    already caught, and -- in the other direction -- a pure addition that
    must NOT raise, since a guard that blocks enrichment would push
    authors to ``override=True`` and disable it entirely.
    """

    @staticmethod
    def _probe() -> type:
        class Probe:
            pass

        return Probe

    def test_partial_downgrade_raises(self, clean_registry):
        """Shape 1: the new registration declares a flag and drops the contract.

        This is the shape #1916's dataclass widening created. Nothing but
        the guard stands between it and a model whose declared dimension
        contract silently becomes None.
        """
        probe = self._probe()
        register_model(
            name="__probe_partial",
            training_mode="gan",
            spatial_dims=(2, 3),
            input_domain="image",
            output_domain="image",
        )(probe)

        with pytest.raises(ValueError, match=r"would DROP already-declared"):
            register_model(
                name="__probe_partial",
                training_mode="gan",
                supports_contrast_conditioning=True,
            )(probe)

        # The refusal must also be effective, not just loud: the entry
        # still carries the contract it declared first.
        caps = get_model_capabilities("__probe_partial")
        assert caps is not None
        assert caps.spatial_dims == (2, 3)
        assert caps.input_domain == "image"
        assert caps.output_domain == "image"

    def test_total_downgrade_still_raises(self, clean_registry):
        """Shape 2: a bare re-registration -- what the old condition caught.

        Kept so the rewrite is a widening and not a swap; this is the
        ``bloch_mamba_v2`` scar the guard was written for.

        NOT a detector for #1916, and it should not be counted as one.
        Run against the pre-fix guard this test does go red -- but on the
        wording (``Actual message: ... with EMPTY capabilities``), not on
        the behaviour: the old condition raised on this shape correctly.
        The one test here that failed pre-fix with ``DID NOT RAISE`` is
        ``test_partial_downgrade_raises``; that is the detector, and it
        was watched red.
        """
        probe = self._probe()
        register_model(
            name="__probe_total",
            training_mode="gan",
            spatial_dims=(2, 3),
        )(probe)

        with pytest.raises(ValueError, match=r"would DROP already-declared"):
            register_model(name="__probe_total", training_mode="gan")(probe)

    def test_pure_addition_is_allowed(self, clean_registry):
        """Shape 3, the other polarity: adding a flag while restating the
        contract drops nothing, so it must be accepted.

        Without this the guard would be un-satisfiable for anyone wanting
        to annotate an existing model, and the workaround -- passing
        ``override=True`` -- turns the guard off completely.
        """
        probe = self._probe()
        register_model(
            name="__probe_add",
            training_mode="gan",
            spatial_dims=(2, 3),
            input_domain="image",
            output_domain="image",
        )(probe)

        register_model(
            name="__probe_add",
            training_mode="gan",
            spatial_dims=(2, 3),
            input_domain="image",
            output_domain="image",
            supports_contrast_conditioning=True,
        )(probe)

        caps = get_model_capabilities("__probe_add")
        assert caps is not None
        assert caps.spatial_dims == (2, 3)
        assert caps.supports_contrast_conditioning is True

    def test_override_still_forces_the_downgrade(self, clean_registry):
        """The documented escape hatch keeps working.

        The guard names ``override=True`` in its own error message; if
        that stopped working the message would be sending authors to a
        dead end.
        """
        probe = self._probe()
        register_model(
            name="__probe_override",
            training_mode="gan",
            spatial_dims=(2, 3),
        )(probe)

        register_model(
            name="__probe_override",
            training_mode="gan",
            override=True,
        )(probe)

        assert get_model_capabilities("__probe_override") is None
