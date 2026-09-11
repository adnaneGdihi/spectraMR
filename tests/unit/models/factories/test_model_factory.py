"""Tests for the model factory's signature filter and its ledger recording.

The filter itself is deliberate and must stay: injecting ``model_kwargs`` and
physics config into a constructor that never declared those parameters is a
supported flexibility path across the corpus. What was not acceptable is that
the drop was **invisible** -- the factory computed the rejected set, logged it at
``debug`` (a level no production run enables), and discarded it, in three
separate copies of the same block.

That is how ``validation.sampler_steps`` was lost: it had a schema field, a
resolver and a call site, and still died at this seam, so the ldm cohort ran
1000 NFE instead of the declared 25 (#480), with nothing in any artifact saying
so.

Sensitivity pairs throughout: a kwarg the constructor accepts must record
nothing, and one it cannot accept must be recorded. A filter that records
everything is as useless as one that records nothing.
"""

from __future__ import annotations

import contextlib
import logging
import warnings

import pytest

from spectramr.core.execution_ledger import (
    ExecutionLedger,
    SubstitutionClass,
)
from spectramr.models.factories.model_factory import _filter_kwargs_to_signature

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _disarm():
    ExecutionLedger.reset()
    yield
    ExecutionLedger.reset()


class Narrow:
    """A constructor that declares exactly two parameters."""

    def __init__(self, in_channels: int, out_channels: int = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels


class Permissive:
    """A constructor that accepts anything via **kwargs."""

    def __init__(self, in_channels: int, **kwargs):
        self.in_channels = in_channels
        self.kwargs = kwargs


def _drops(ledger):
    return [
        s for s in ledger.substitutions if s.class_id is SubstitutionClass.DROPPED_UNCONSUMED_KWARG
    ]


def test_accepted_kwargs_pass_through_and_record_nothing():
    """CONTROL: proves the seam is not simply always recording."""
    ledger = ExecutionLedger.begin_run()
    out = _filter_kwargs_to_signature(
        {"in_channels": 2, "out_channels": 1},
        Narrow,
        logger_=logger,
        site="test",
    )
    assert out == {"in_channels": 2, "out_channels": 1}
    assert _drops(ledger) == []


def test_rejected_kwarg_is_dropped_and_recorded_with_its_consumer():
    """DEFECT: the sampler_steps class (#480)."""
    ledger = ExecutionLedger.begin_run()
    out = _filter_kwargs_to_signature(
        {"in_channels": 2, "sampler_steps": 25},
        Narrow,
        logger_=logger,
        site="model_factory:create_generator",
    )
    assert out == {"in_channels": 2}, "the filter must still filter"

    dropped = _drops(ledger)
    assert [s.path for s in dropped] == ["sampler_steps"]
    assert dropped[0].requested == 25
    assert dropped[0].consumer == "Narrow.__init__"
    assert "sampler_steps" in dropped[0].reason


def test_var_keyword_constructor_drops_nothing():
    """A **kwargs constructor consumes everything, so there is nothing to record.

    This is the case that keeps the flexibility intact: recording a "drop" here
    would be a false positive, since the value really does arrive.
    """
    ledger = ExecutionLedger.begin_run()
    out = _filter_kwargs_to_signature(
        {"in_channels": 2, "anything": True},
        Permissive,
        logger_=logger,
        site="test",
    )
    assert out == {"in_channels": 2, "anything": True}
    assert _drops(ledger) == []


def test_role_label_distinguishes_the_wrapper_seam():
    """Three call sites share this helper; the record must say which one fired."""
    ledger = ExecutionLedger.begin_run()
    _filter_kwargs_to_signature(
        {"base_channels": 64},
        Narrow,
        logger_=logger,
        site="model_factory:diffusion_wrapper",
        role="wrapper",
    )
    assert _drops(ledger)[0].consumer == "Narrow wrapper.__init__"


def test_none_target_class_is_passed_through_untouched():
    """A registry miss is the caller's error to report, not this seam's to mask."""
    ledger = ExecutionLedger.begin_run()
    out = _filter_kwargs_to_signature({"in_channels": 2}, None, logger_=logger, site="test")
    assert out == {"in_channels": 2}
    assert _drops(ledger) == []


def test_filtering_is_unchanged_when_no_ledger_is_armed():
    """Instrumentation must never alter the caller's behaviour.

    Every production path that existed before the ledger must build the same
    object whether or not anything is watching.
    """
    assert ExecutionLedger.recording() is False
    out = _filter_kwargs_to_signature(
        {"in_channels": 2, "sampler_steps": 25},
        Narrow,
        logger_=logger,
        site="test",
    )
    assert out == {"in_channels": 2}


def test_uninspectable_signature_passes_through():
    """C extensions have no introspectable signature; behaviour is unchanged."""
    ledger = ExecutionLedger.begin_run()
    out = _filter_kwargs_to_signature({"anything": 1}, type(len), logger_=logger, site="test")
    assert out == {"anything": 1}
    assert _drops(ledger) == []


class TestFilterKwargsToSignatureRecording:
    """The signature filter must leave evidence in every branch it takes.

    Before PR #718 / issue #878 the `**kwargs` branch returned early, so a key
    a `**kwargs` constructor swallowed produced no record at all -- the same
    invisibility the DROPPED_UNCONSUMED_KWARG class was introduced to break
    after `validation.sampler_steps` silently died (#480).
    """

    @staticmethod
    def _filter(mapped, cls, **kw):
        import logging

        from spectramr.models.factories.model_factory import (
            _filter_kwargs_to_signature,
        )

        return _filter_kwargs_to_signature(
            mapped, cls, logger_=logging.getLogger(__name__), site="test", **kw
        )

    def test_named_param_class_drops_and_records_the_surplus(self):
        from spectramr.core.execution_ledger import ExecutionLedger, SubstitutionClass

        class _Named:
            def __init__(self, alpha: float = 1.0):
                pass

        ledger = ExecutionLedger.begin_run(source="test")
        out = self._filter({"alpha": 2.0, "nowhere": 7}, _Named)

        assert out == {"alpha": 2.0}, "the unroutable key must be dropped"
        dropped = [
            s.path
            for s in ledger.substitutions
            if s.class_id is SubstitutionClass.DROPPED_UNCONSUMED_KWARG
        ]
        assert dropped == ["nowhere"]

    def test_var_kwargs_class_records_untyped_acceptance_instead_of_silence(self):
        """A **kwargs ctor drops nothing, so the old code recorded nothing."""
        from spectramr.core.execution_ledger import ExecutionLedger, SubstitutionClass

        class _Swallows:
            def __init__(self, alpha: float = 1.0, **kwargs):
                pass

        ledger = ExecutionLedger.begin_run(source="test")
        out = self._filter(
            {"alpha": 2.0, "nowhere": 7},
            _Swallows,
            declared=frozenset({"alpha", "nowhere"}),
        )

        assert out == {"alpha": 2.0, "nowhere": 7}, "a **kwargs ctor drops nothing"
        untyped = [
            s.path
            for s in ledger.substitutions
            if s.class_id is SubstitutionClass.EXTRA_ALLOW_UNTYPED
        ]
        assert untyped == ["nowhere"]

    def test_opportunistically_forwarded_keys_are_not_recorded(self):
        """`GeneratorBuilder` forwards every top-level `model.*` field on spec.

        Recording those would bury the author-declared `model_kwargs` entries
        that are the actual signal under one record per schema field per arm.
        """
        from spectramr.core.execution_ledger import ExecutionLedger, SubstitutionClass

        class _Swallows:
            def __init__(self, alpha: float = 1.0, **kwargs):
                pass

        ledger = ExecutionLedger.begin_run(source="test")
        self._filter(
            {"alpha": 2.0, "forwarded": 7},
            _Swallows,
            declared=frozenset({"alpha"}),  # `forwarded` was NOT author-written
        )

        assert not [
            s
            for s in ledger.substitutions
            if s.class_id is SubstitutionClass.EXTRA_ALLOW_UNTYPED
        ]

    def test_declared_none_records_nothing_in_the_var_kwargs_branch(self):
        """Every pre-existing caller passes no `declared`; they stay silent."""
        from spectramr.core.execution_ledger import ExecutionLedger, SubstitutionClass

        class _Swallows:
            def __init__(self, **kwargs):
                pass

        ledger = ExecutionLedger.begin_run(source="test")
        self._filter({"anything": 1}, _Swallows)

        assert not [
            s
            for s in ledger.substitutions
            if s.class_id is SubstitutionClass.EXTRA_ALLOW_UNTYPED
        ]

    def test_class_with_no_owned_init_passes_through_unfiltered(self):
        """nn.Module reports `(*args, **kwargs)`; the contract reports no owner.

        Both pass through -- but for different reasons, and the contract's is
        the correct one. This is the PR #718 bug class.
        """
        import torch.nn as nn

        class _InheritsInit(nn.Module):
            pass

        out = self._filter({"anything": 1}, _InheritsInit)
        assert out == {"anything": 1}


# ---------------------------------------------------------------------------
# create_model -- the config-sniffing convenience layer being retired
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings(
    "ignore::DeprecationWarning:spectramr.models.factories.model_factory"
)
class TestCreateModelIsReachableAndFullConfigOnly:
    """``create_model`` was unreachable for its entire existence.

    ``model_factory.py`` was created whole on 2026-05-18 with ``import logging``
    but **no module-level ``logger``**, while ``create_model``'s first executable
    statement is ``logger.debug(...)``. Every call, on every config, therefore
    raised ``NameError: name 'logger' is not defined`` before touching the
    config -- and ``create_model`` was the construction path for ``predict``,
    ``make`` and ``scripts/evaluation/run_test_inference.py``.

    It survived three months because **no test in the repository ever called
    it**: the file named after the factory (``test_f18_model_factory_none_config``)
    exercises ``get_model_factory()``, the constructor, and stops there. Ruff had
    been reporting all five bare-logger sites as ``F821`` the whole time.

    The tests below close both halves: the method must be *reachable*, and it
    must no longer accept a bare ``ModelConfigSchema`` -- the type-sniffed branch
    that silently dropped ``acceleration_config`` (non-negotiable 3).
    """

    def test_module_defines_the_logger_its_functions_call(self) -> None:
        """Regression guard for the NameError. Cheap, and it never sleeps."""
        import spectramr.models.factories.model_factory as mf

        assert isinstance(getattr(mf, "logger", None), logging.Logger), (
            "model_factory has bare `logger.` call sites; without a "
            "module-level logger every one of them is a NameError."
        )

    def test_create_model_rejects_a_bare_model_schema(self) -> None:
        """The retired branch B: a bare schema must now fail loud.

        Passing ``config.model`` used to select a degraded branch that resolved
        a strict subset of the builder's kwargs -- no ``acceleration_config``,
        no ``kspace_log_scaled``, no ``ModelConfigSchema`` sweep. Nothing warned;
        the model was simply built wrong.
        """
        from spectramr.config.schemas.model import ModelConfigSchema
        from spectramr.models.factories.model_factory import get_model_factory

        factory = get_model_factory()
        with pytest.raises(TypeError) as excinfo:
            factory.create_model(ModelConfigSchema(model_type="unet"))

        message = str(excinfo.value)
        assert "ModelBuilder" in message, (
            "the error must name the replacement that takes responsibility, "
            "not merely reject the input"
        )

    def test_bare_schema_rejection_is_not_a_nameerror(self) -> None:
        """Sensitivity pair: the raise must be the *intended* one.

        On unfixed code this method raises ``NameError`` for every input, so a
        bare ``pytest.raises(Exception)`` would pass against the defect.
        """
        from spectramr.config.schemas.model import ModelConfigSchema
        from spectramr.models.factories.model_factory import get_model_factory

        factory = get_model_factory()
        try:
            factory.create_model(ModelConfigSchema(model_type="unet"))
        except NameError as exc:  # pragma: no cover - fails loudly if it fires
            pytest.fail(f"create_model is still unreachable: {exc}")
        except TypeError:
            pass


class TestDeprecationSitsOnTheRetiredSurface:
    """The deprecation must fire on ``create_model`` and nothing else.

    It used to fire in ``ModelFactory.__init__``, i.e. on *every* construction
    -- including ``GeneratorBuilder.build()``, the canonical path. That forced
    the builder to wrap its own import in ``catch_warnings`` and mute it, which
    is the failure mode this class pins against: a warning the correct path has
    to silence stops being a signal. It also made enforcement caller-dependent,
    because ``pyproject.toml`` promotes ``error::DeprecationWarning:spectramr.*``
    while ``stacklevel=2`` attributes the warning to whoever called.

    Sensitivity pairs throughout: the retired surface must warn, and the
    primitives the canonical path depends on must not.
    """

    @staticmethod
    def _deprecations(fn) -> list[warnings.WarningMessage]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # the call's own outcome is not what this class measures
            with contextlib.suppress(TypeError, ValueError):
                fn()
        return [w for w in caught if issubclass(w.category, DeprecationWarning)]

    def test_constructing_a_factory_does_not_warn(self) -> None:
        """``GeneratorBuilder.build()`` constructs one on every training run."""
        from spectramr.models.factories.model_factory import ModelFactory

        assert self._deprecations(ModelFactory) == [], (
            "constructing a ModelFactory must not warn -- the canonical builder "
            "does it on every run, and a warning there is what forced the "
            "builder to mute the category wholesale"
        )

    def test_create_generator_does_not_warn(self) -> None:
        """Sensitivity pair: the primitive is not the surface being retired.

        ``GeneratorBuilder.build()`` calls exactly this, and model-internal
        composers must keep calling it -- ``check_layering.sh`` forbids
        ``models/ -> infrastructure/``, so they cannot route via a builder.
        """
        from spectramr.models.factories.model_factory import ModelFactory

        factory = ModelFactory()
        assert (
            self._deprecations(
                lambda: factory.create_generator(
                    "unet", in_channels=1, out_channels=1
                )
            )
            == []
        )

    def test_create_model_warns(self) -> None:
        """The retired surface, and the only one that may warn."""
        from spectramr.config.schemas.model import ModelConfigSchema
        from spectramr.models.factories.model_factory import ModelFactory

        factory = ModelFactory()
        found = self._deprecations(
            lambda: factory.create_model(ModelConfigSchema(model_type="unet"))
        )
        assert len(found) == 1, f"expected exactly one deprecation, got {found}"

    def test_the_warning_names_its_replacement(self) -> None:
        """A deprecation that does not say what to use instead is a dead end.

        CLAUDE.md's standing constraint on retirement is that something else
        must take responsibility; the message is where the caller learns what.
        """
        from spectramr.config.schemas.model import ModelConfigSchema
        from spectramr.models.factories.model_factory import ModelFactory

        factory = ModelFactory()
        (found,) = self._deprecations(
            lambda: factory.create_model(ModelConfigSchema(model_type="unet"))
        )
        assert "ModelBuilder" in str(found.message)


class TestCreateClassmethod:
    """``ModelFactory.create`` raised ``TypeError`` on every call (#1511).

    ``model_type`` is a positional parameter of ``create_generator`` AND a field
    of ``config.model``, so splatting the merged mapping passed it twice. The
    method has no production callers -- that is the only reason a defect this
    total survived, and it is the same shape as #1277: an entry point nothing
    exercises is an entry point nothing checks.

    These tests are also the wiring (non-negotiable 16): the fix is not "the
    code looks right", it is that a real model comes back.
    """

    @staticmethod
    def _config(**model_fields):
        from types import SimpleNamespace

        fields = {"model_type": "stub_for_create", "in_channels": 2, "out_channels": 2}
        fields.update(model_fields)
        return SimpleNamespace(model=SimpleNamespace(**fields))

    @pytest.fixture
    def registered(self):
        """A minimal generator registered under a name only these tests use.

        Registered in the GLOBAL registry, not on a factory instance:
        ``ModelFactory.create`` constructs its own factory (and with it a fresh
        ``ModelRegistry``, which syncs from the global one), so a per-instance
        registration would be invisible to the very call under test.
        """
        import torch.nn as nn

        from spectramr.models.registry import MODEL_REGISTRY

        class _Stub(nn.Module):
            def __init__(self, in_channels=1, out_channels=1, **kwargs):
                super().__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.seen = kwargs

            def forward(self, x):  # pragma: no cover - never called
                return x

        MODEL_REGISTRY["stub_for_create"] = {"class": _Stub, "training_mode": "reconstruction"}
        try:
            yield _Stub
        finally:
            MODEL_REGISTRY.pop("stub_for_create", None)

    def test_create_returns_a_model_instead_of_raising(self, registered) -> None:
        from spectramr.models.factories.model_factory import ModelFactory

        model = ModelFactory.create(self._config())
        assert isinstance(model, registered)
        assert model.in_channels == 2

    def test_model_type_is_not_passed_twice(self, registered) -> None:
        """The exact failure: ``got multiple values for argument 'model_type'``."""
        from spectramr.models.factories.model_factory import ModelFactory

        model = ModelFactory.create(self._config())
        assert "model_type" not in model.seen

    def test_explicit_model_type_kwarg_overrides_the_config(self, registered) -> None:
        """Popped, not dropped: an override is honoured rather than discarded
        silently (non-negotiable 3)."""
        from spectramr.models.factories.model_factory import ModelFactory

        config = self._config(model_type="not_registered_at_all")
        model = ModelFactory.create(config, model_type="stub_for_create")
        assert isinstance(model, registered)

    def test_extra_kwargs_reach_the_constructor(self, registered) -> None:
        """Discrimination: the pop removes ``model_type`` and nothing else."""
        from spectramr.models.factories.model_factory import ModelFactory

        model = ModelFactory.create(self._config(), base_channels=8)
        assert model.seen.get("base_channels") == 8

    def test_missing_model_type_raises_a_named_error(self, registered) -> None:
        from types import SimpleNamespace

        from spectramr.models.factories.model_factory import ModelFactory

        config = SimpleNamespace(model=SimpleNamespace(in_channels=2, out_channels=2))
        with pytest.raises(TypeError, match="needs a model type"):
            ModelFactory.create(config)


class TestRoleDeclaredNotGuessed:
    """``ModelRegistry`` buckets on the declared role (#1932).

    It used to guess with ``issubclass(cls, IDiscriminator)`` and fall through
    to ``# Default to generator for backward compatibility``. Only 10 of the
    framework's discriminators subclass that interface, so 9 more landed in the
    generator bucket and ``create_discriminator`` reported each as "not
    registered" -- a live build failure for any GAN arm naming one of them.
    """

    def test_every_discriminator_named_model_is_in_the_discriminator_bucket(self):
        """Name-heuristic oracle, independent of the mechanism it checks.

        This deliberately does NOT ask the registry how it classified anything;
        it asks whether a model *called* a discriminator can be *built* as one.
        A test written against ``role`` would pass for any consistent wiring,
        including a wiring that files everything as a generator.
        """
        from spectramr.models.factories.model_factory import ModelRegistry
        from spectramr.models.init_registry import populate_model_registry
        from spectramr.models.registry import MODEL_REGISTRY

        populate_model_registry()
        registry = ModelRegistry()
        named = {
            n
            for n in MODEL_REGISTRY
            if any(tok in n for tok in ("discriminator", "critic", "patchgan"))
        }
        assert named, "name heuristic matched nothing -- the oracle is vacuous"
        misfiled = sorted(named - set(registry._discriminators))
        assert misfiled == [], (
            f"{len(misfiled)} discriminator-named models are in the generator "
            f"bucket, so create_discriminator() reports them as not "
            f"registered: {misfiled}"
        )

    def test_the_nine_formerly_misfiled_discriminators_are_reachable(self):
        """The exact names #1932 reports, pinned so a regression names itself."""
        from spectramr.models.factories.model_factory import ModelRegistry
        from spectramr.models.init_registry import populate_model_registry

        populate_model_registry()
        registry = ModelRegistry()
        for name in (
            "domain_discriminator",
            "frequency_domain_discriminator",
            "kan_discriminator",
            "kspace_discriminator",
            "ldm_latent_discriminator",
            "ldm_multiscale_latent_discriminator",
            "ldm_patch_latent_discriminator",
            "stargan_v2_discriminator",
            "wasserstein_discriminator",
        ):
            assert registry.has_discriminator(name), (
                f"{name!r} is not in the discriminator bucket; "
                f"create_discriminator({name!r}) would raise ModelCreationError"
            )

    def test_registration_stamps_the_role(self):
        from spectramr.models.registry import MODEL_REGISTRY, register_model

        backup = dict(MODEL_REGISTRY)
        MODEL_REGISTRY.clear()
        try:
            register_model(name="g", training_mode="gan")(type("G", (), {}))
            register_model(name="d", training_mode="gan", role="discriminator")(
                type("D", (), {})
            )
            assert MODEL_REGISTRY["g"]["role"] == "generator"
            assert MODEL_REGISTRY["d"]["role"] == "discriminator"
        finally:
            MODEL_REGISTRY.clear()
            MODEL_REGISTRY.update(backup)

    def test_idiscriminator_subclass_without_role_raises_at_registration(self):
        """The default must not be able to go silently wrong.

        A class that implements the discriminator interface but is filed as a
        generator is always a mistake. Registration refuses it, where the fix
        is one keyword away -- rather than letting create_discriminator report
        the name as unregistered much later.
        """
        import torch.nn as nn

        from spectramr.models.interfaces import IDiscriminator
        from spectramr.models.registry import MODEL_REGISTRY, register_model

        backup = dict(MODEL_REGISTRY)
        MODEL_REGISTRY.clear()
        try:

            class _Critic(IDiscriminator, nn.Module):
                def forward(self, x):
                    return x

                def discriminate(self, x):
                    return x

                def get_feature_maps(self, x):
                    return [x]

            with pytest.raises(ValueError, match="role='discriminator'"):
                register_model(name="oops", training_mode="gan")(_Critic)

            # and it registers cleanly once it declares the role
            register_model(name="fine", training_mode="gan", role="discriminator")(
                _Critic
            )
            assert MODEL_REGISTRY["fine"]["role"] == "discriminator"
        finally:
            MODEL_REGISTRY.clear()
            MODEL_REGISTRY.update(backup)

    def test_undeclared_discriminator_lands_in_generators_and_cannot_be_built(self):
        """Characterization of the residue, NOT a regression detector.

        This test passes on both refs by construction and was watched doing
        so: a class that neither subclasses IDiscriminator nor declares a
        role lands in ``_generators`` before AND after #1932, because the
        ``role="generator"`` signature default is deliberate (declaring a
        role on all 602 register_model sites is a mechanical rewrite this
        change refuses to make). So a green here is not evidence #1932 is
        fixed -- the four sibling tests are, and they were watched red.

        What it pins is the exact boundary of what the fix can see. The
        detectable shape -- IDiscriminator subclass, no role -- raises at
        registration; this undetectable one is why that guard exists. The
        observable failure is not an error but a wrong bucket, and then
        ``create_discriminator`` reporting a registered name as unregistered.
        """
        from spectramr.models.factories.model_factory import ModelRegistry
        from spectramr.models.registry import MODEL_REGISTRY, register_model

        backup = dict(MODEL_REGISTRY)
        MODEL_REGISTRY.clear()
        try:
            register_model(name="silent_critic", training_mode="gan")(
                type("SilentCritic", (), {})
            )
            registry = ModelRegistry()
            assert "silent_critic" in registry._generators
            assert not registry.has_discriminator("silent_critic")
        finally:
            MODEL_REGISTRY.clear()
            MODEL_REGISTRY.update(backup)
