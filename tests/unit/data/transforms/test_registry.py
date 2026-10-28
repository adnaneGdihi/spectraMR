"""Contract tests for the config-declarable transform registry.

The registry exists because ``data.processing.transforms`` was typed
``list[dict[str, Any]]`` and its only consumer matched the single literal
``"graph_encoding"`` and stopped -- every other declaration validated at load
and was then silently discarded. These tests pin the three properties that
keep that from recurring: an unregistered name raises, every registered
transform is constructible, and the keys live strategies read have a producer.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from spectramr.data.transforms.registry import (
    TRANSFORM_REGISTRY,
    RegisteredTransform,
    build_transform,
    get_transform,
    list_transforms,
    register_transform,
    transforms_producing,
    transforms_without_effect,
)


class TestRegistryPopulation:
    def test_importing_the_package_populates_the_registry(self):
        """A decorator only runs when its module is imported.

        The registry is useless if the defining modules are never imported --
        that is precisely how a transform stays unreachable from YAML. Importing
        the package alone must be enough; ``data/transforms/__init__.py`` carries
        the force-imports that guarantee it.
        """
        import importlib

        importlib.import_module("spectramr.data.transforms")
        assert list_transforms(), "registry is empty after importing the package"

    def test_expected_transforms_are_registered(self):
        assert set(list_transforms()) >= {
            "foreground_mask",
            "graph_encoding",
            "phase_residual",
            "scout_acquisition",
        }

    def test_every_entry_is_a_registered_transform_record(self):
        for name in list_transforms():
            entry = get_transform(name)
            assert isinstance(entry, RegisteredTransform)
            assert entry.name == name
            assert isinstance(entry.cls, type)


class TestUnregisteredNamesRaise:
    def test_unknown_name_raises_and_lists_the_valid_ones(self):
        with pytest.raises(KeyError) as exc:
            get_transform("definitely_not_a_transform")
        msg = str(exc.value)
        assert "definitely_not_a_transform" in msg
        # The valid set must be in the message -- otherwise the error tells the
        # user nothing they could not already see.
        assert "phase_residual" in msg

    def test_dotted_import_path_gets_an_explicit_hint(self):
        """Four committed arms spell a dotted path; it never resolved.

        There is no dotted-path importer anywhere in the data path, so those
        entries were dropped in silence. The hint is the only thing that tells
        a user why their arm's named mechanism did nothing.
        """
        with pytest.raises(KeyError) as exc:
            get_transform("spectramr.data.transforms.slice_profile.SliceProfileTransform")
        assert "Dotted import paths are not supported" in str(exc.value)

    def test_build_transform_also_raises_on_an_unknown_name(self):
        with pytest.raises(KeyError):
            build_transform("definitely_not_a_transform")


#: Entries needing constructor args to build at all. Anything absent is expected
#: to construct on defaults.
#:
#: ``simulate_ulf_from_qmaps`` takes ``b0_source`` with no default on purpose: it
#: is the field the maps were measured at, and the operator transports T1 away
#: from it. Defaulting it to 3.0 would silently mis-transport 7 T maps by 34.5 %
#: (193.37 ms rendered as 260.12 ms at 0.064 T), which is a plausible image, not
#: an error. Declaring the arg here is what keeps the constructibility invariant
#: honest without inventing a wrong physical default.
#:
#: ``hardware_unwarp`` reads and validates its calibration artifact at construction (a wrong
#: path must fail when the pipeline is built, not at the first batch), so the table points it
#: at an identity artifact written once per session; there is no artifact-free construction.


def _identity_hardware_warp_artifact() -> str:
    from spectramr.data.hardware_warp_artifacts import identity_artifact, write_artifact

    path = Path(tempfile.mkdtemp(prefix="hardware_unwarp_")) / "identity.json"
    write_artifact(
        identity_artifact(
            min_degree=2,
            max_degree=3,
            ball_radius_mm=100.0,
            basis_points_per_axis=5,
            unit_id="registry-test",
            software_version="0",
        ),
        path,
    )
    return str(path)


_BUILD_KWARGS: dict[str, dict] = {
    "graph_encoding": {},
    "hardware_unwarp": {"artifact_path": _identity_hardware_warp_artifact()},
    "simulate_ulf_from_qmaps": {"b0_source": 3.0},
}


class TestConstruction:
    @pytest.mark.parametrize("name", list_transforms())
    def test_every_registered_transform_constructs(self, name):
        """Registration without constructibility is a facade of its own.

        Args a transform genuinely requires come from ``_BUILD_KWARGS``, the
        same table the both-chains tests below read. Two tests in one file
        asking "can this be built" through different mechanisms is a second
        owner for one invariant (non-negotiable 17); this reads the one table,
        so a required arg with no entry there still fails.
        """
        assert build_transform(name, **_BUILD_KWARGS.get(name, {})) is not None

    def test_declared_kwargs_reach_the_constructor(self):
        t = build_transform("phase_residual", kernel_size=7)
        assert t.kernel_size == 7

    def test_an_unknown_kwarg_raises_and_names_the_transform(self):
        """A silently-ignored kwarg is the same pitfall-#15 shape."""
        with pytest.raises(TypeError) as exc:
            build_transform("phase_residual", not_a_real_kwarg=1)
        assert "phase_residual" in str(exc.value)
        assert "not_a_real_kwarg" in str(exc.value)


class TestDuplicateRegistration:
    def test_two_classes_under_one_name_raises(self):
        """Import-order-dependent resolution is a defect, not a warning."""

        @register_transform("_dupe_probe")
        class _First:
            pass

        try:
            with pytest.raises(ValueError, match="already registered"):

                @register_transform("_dupe_probe")
                class _Second:
                    pass

        finally:
            TRANSFORM_REGISTRY.pop("_dupe_probe", None)

    def test_re_registering_the_same_class_is_idempotent(self):
        """Module re-import must not explode."""

        class _Same:
            pass

        try:
            register_transform("_idem_probe")(_Same)
            register_transform("_idem_probe")(_Same)
            assert get_transform("_idem_probe").cls is _Same
        finally:
            TRANSFORM_REGISTRY.pop("_idem_probe", None)


class TestProducerInvariant:
    """The anti-facade property this registry was built to make checkable.

    Each of these keys is read by a live strategy or metric whose chain was
    dead at link 0: the transform that produces it existed on disk but could
    not be constructed from any config.
    """

    @pytest.mark.parametrize(
        "key, reader",
        [
            ("phase_residual", "inverse_bloch_phase_strategy"),
            ("scout", "scas_strategy (hypernet + density penalty)"),
            ("foreground_mask", "core.metrics.context (8 no-reference metrics)"),
        ],
    )
    def test_key_read_by_a_live_consumer_has_a_registered_producer(self, key, reader):
        producers = transforms_producing(key)
        assert producers, (
            f"batch key {key!r} is read by {reader} but no registered transform "
            "produces it -- the consumer is unreachable"
        )

    def test_transforms_producing_returns_nothing_for_an_unknown_key(self):
        assert transforms_producing("no_such_key") == ()

    def test_every_registered_transform_declares_its_effect(self):
        """A transform that adds or rewrites no declared key cannot be verified to have fired."""
        assert transforms_without_effect() == ()

    def test_a_transform_declaring_no_effect_is_caught(self):
        planted = {
            "silent": RegisteredTransform(name="silent", cls=object),
            "adds": RegisteredTransform(name="adds", cls=object, produces=("k",)),
            "rewrites": RegisteredTransform(name="rewrites", cls=object, modifies=("input",)),
        }
        assert transforms_without_effect(planted) == ("silent",)

    def test_an_in_place_edit_is_not_a_producer(self):
        assert "slice_profile" not in transforms_producing("input")


#: The five modules D16's wire half attached.
_D16_WIRED = (
    "homomorphic_bias_field",
    "joint_rotation",
    "slab_to_channel",
    "select_middle_slice",
    "simulate_ulf_from_hf",
)


class TestEveryRegisteredTransformReachesBothChains:
    """The invariant, parametrised over the WHOLE registry -- not one example.

    "Registered" and "fires" are different facts. A membership test goes green
    the moment a decorator runs, while no arm constructs anything; that gap is
    the exact state the registry exists to end, so the assertion has to be made
    at the seam -- the Compose the builder returns -- and over every entry, so a
    transform added later cannot land registered-but-unreachable.
    """

    @staticmethod
    def _chain(name, kwargs, build):
        from spectramr.config.schemas.data import DataProcessingConfigSchema
        from spectramr.data.builders.torchio_transform_builder import (
            TorchIOTransformConfig,
        )
        from tests.utils.data_config_stub import DataConfigStub

        cfg = DataConfigStub(
            processing=DataProcessingConfigSchema(transforms=[{"name": name, "kwargs": kwargs}])
        )
        return build(TorchIOTransformConfig.from_training_config(cfg))

    @pytest.mark.parametrize("name", sorted(TRANSFORM_REGISTRY))
    @pytest.mark.parametrize("which", ["train", "val"])
    def test_a_declared_transform_lands_in_the_built_chain(self, name, which):
        from spectramr.data.builders.torchio_transform_builder import (
            TorchIOTransformBuilder,
        )

        build = (
            TorchIOTransformBuilder.build_train_transforms
            if which == "train"
            else TorchIOTransformBuilder.build_val_transforms
        )
        expected = TRANSFORM_REGISTRY[name].cls
        compose = self._chain(name, _BUILD_KWARGS.get(name, {}), build)
        assert any(isinstance(t, expected) for t in compose.transforms), (
            f"{name!r} is registered but does not reach the {which} chain"
        )

    def test_both_chains_get_it_not_just_train(self):
        """Applying a declared transform to train only reproduces the
        normalization split this audit already found: the model would be graded
        on data the transform never touched."""
        from spectramr.data.builders.torchio_transform_builder import (
            TorchIOTransformBuilder,
        )

        for name in sorted(TRANSFORM_REGISTRY):
            cls = TRANSFORM_REGISTRY[name].cls
            kwargs = _BUILD_KWARGS.get(name, {})
            in_train = any(
                isinstance(t, cls)
                for t in self._chain(
                    name, kwargs, TorchIOTransformBuilder.build_train_transforms
                ).transforms
            )
            in_val = any(
                isinstance(t, cls)
                for t in self._chain(
                    name, kwargs, TorchIOTransformBuilder.build_val_transforms
                ).transforms
            )
            assert in_train == in_val, f"{name!r} reaches one chain but not the other"


class TestD16WiredTransforms:
    """The five modules the wire half attached (D16).

    Each was a working ``tio.Transform`` whose only defect was having no YAML
    route. ``joint_rotation`` additionally carried a docstring asserting "there
    is no @register_transform decorator in the codebase" -- a module documenting
    its own unreachability as though it were the design.
    """

    @pytest.mark.parametrize("name", _D16_WIRED)
    def test_is_registered(self, name):
        assert name in TRANSFORM_REGISTRY

    def test_homomorphic_declares_what_it_adds(self):
        """``produces`` is the anti-facade payload: it lets an audit ask "does
        anything produce the key this consumer reads" without importing every
        strategy. A transform that adds keys and declares none is invisible."""
        assert TRANSFORM_REGISTRY["homomorphic_bias_field"].produces == (
            "log_anatomy",
            "log_bias",
        )

    def test_the_ulf_simulator_declares_it_needs_a_target(self):
        assert "target" in TRANSFORM_REGISTRY["simulate_ulf_from_hf"].requires

    def test_joint_rotation_rotates_every_key_from_one_draw(self):
        """Why this one is safe to wire where a torchvision Compose is not.

        Independent per-key draws would rotate input and target differently --
        the leak ``tests/data_integrity/test_augmentation_leak.py`` catches.
        """
        import inspect

        src = inspect.getsource(TRANSFORM_REGISTRY["joint_rotation"].cls)
        assert "include_keys" in src

    def test_the_stale_no_decorator_claim_is_gone(self):
        """It documented its own unreachability; leaving it would teach the
        next reader that the module cannot be registered."""
        import inspect

        mod = inspect.getmodule(TRANSFORM_REGISTRY["joint_rotation"].cls)
        assert "there is no ``@register_transform`` decorator" not in (mod.__doc__ or "")


class TestDeletedAugmentationPipeline:
    """The delete half. A duplicate that DISAGREES with the live path is not
    unfinished capability -- wiring it would fork, not finish."""

    def test_the_module_is_gone(self):
        import importlib

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("spectramr.data.transforms.augmentation_pipeline")

    def test_it_was_never_registered(self):
        """It must not come back through the registry either: its randomness is
        drawn per CALL, so input and target would get different geometry."""
        assert "augmentation_pipeline" not in TRANSFORM_REGISTRY
        assert not any(
            "augmentation_pipeline" in entry.cls.__module__ for entry in TRANSFORM_REGISTRY.values()
        )
