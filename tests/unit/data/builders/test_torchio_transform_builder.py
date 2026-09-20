"""Unit tests for TorchIO Transform Builder (Phase T1).

Tests the TorchIOTransformConfig dataclass and TorchIOTransformBuilder
for correct transform pipeline composition.
"""

import pytest
import torchio as tio

from spectramr.data.builders.torchio_transform_builder import (
    IMPLEMENTED_NORMALIZATION_TYPES,
    TorchIOTransformBuilder,
    TorchIOTransformConfig,
)
from tests.utils.data_config_stub import DataConfigStub


class TestTorchIOTransformConfig:
    """Tests for TorchIOTransformConfig dataclass."""

    def test_default_construction(self):
        """Test that default config is valid."""
        config = TorchIOTransformConfig()
        assert config.patch_size == (320, 320)
        assert config.standardization_mode == "smart"
        assert config.trajectory_type is None
        assert config.normalize_kspace is False
        assert config.normalize_images is False
        assert config.enable_graph_encoding is False

    def test_custom_patch_size(self):
        """Test with custom patch size."""
        config = TorchIOTransformConfig(patch_size=(512, 512, 64))
        assert config.patch_size == (512, 512, 64)

    def test_trajectory_types(self):
        """Test valid trajectory types."""
        for traj in ["cartesian", "radial", "spiral", "golden_angle", "epi"]:
            config = TorchIOTransformConfig(trajectory_type=traj)
            assert config.trajectory_type == traj

    def test_invalid_trajectory_type_raises(self):
        """An unknown trajectory type must RAISE (#1097).

        This test previously asserted the opposite — "should not raise, just warn" —
        and that warning is exactly what let `trajectory: spiralll` train on Cartesian
        physics while the YAML claimed a spiral. `data.trajectory` is typed
        `str | None` rather than a closed enum, so schema validation cannot catch the
        typo; `__post_init__` is the only place it can be caught (CLAUDE.md #9).
        """
        with pytest.raises(ValueError, match="Unknown trajectory_type"):
            TorchIOTransformConfig(trajectory_type="invalid")

    def test_the_invalid_trajectory_error_names_the_accepted_set(self):
        """A rejection that does not say what IS accepted just moves the guesswork."""
        with pytest.raises(ValueError) as exc:
            TorchIOTransformConfig(trajectory_type="spiralll")  # plausible typo
        for expected in ("cartesian", "radial", "spiral", "golden_angle", "epi"):
            assert expected in str(exc.value)

    def test_the_accepted_set_is_the_shared_constant_not_a_copy(self):
        """The whole bug was two lists that drifted. If this module ever re-inlines
        its own tuple, the accept-set can silently outgrow the routed set again."""
        from spectramr.infrastructure.physics.trajectories import TRAJECTORY_TYPES

        for traj in TRAJECTORY_TYPES:
            assert TorchIOTransformConfig(trajectory_type=traj).trajectory_type == traj

    def test_standardization_modes(self):
        """Test valid standardization modes."""
        for mode in ["smart", "strict", "none"]:
            config = TorchIOTransformConfig(standardization_mode=mode)
            assert config.standardization_mode == mode

    def test_invalid_standardization_mode(self):
        """Test that invalid standardization mode raises error."""
        with pytest.raises(ValueError, match="standardization_mode"):
            TorchIOTransformConfig(standardization_mode="invalid")

    def test_invalid_patch_size(self):
        """Test that invalid patch size raises error."""
        with pytest.raises(ValueError, match="patch_size"):
            TorchIOTransformConfig(patch_size=())

        with pytest.raises(ValueError, match="patch_size"):
            TorchIOTransformConfig(patch_size=None)

    def test_invalid_kspace_percentile(self):
        """Test that invalid kspace_percentile raises error."""
        with pytest.raises(ValueError, match="kspace_percentile"):
            TorchIOTransformConfig(kspace_percentile=0)

        with pytest.raises(ValueError, match="kspace_percentile"):
            TorchIOTransformConfig(kspace_percentile=1.5)

    def test_from_training_config_minimal(self):
        """Test from_training_config with minimal config object."""

        config = TorchIOTransformConfig.from_training_config(
            DataConfigStub(
                patch_size=(256, 256),
                acceleration=None,
            )
        )
        assert config.patch_size == (256, 256)

    def test_from_training_config_full(self):
        """Test from_training_config with full config object."""

        class AccelConfig:
            base_acceleration = 4.0
            max_acceleration = 8.0
            center_fraction = 0.1

        config = TorchIOTransformConfig.from_training_config(
            DataConfigStub(
                patch_size=(512, 512),
                trajectory="radial",
                normalize_kspace=True,
                kspace_percentile=0.95,
                normalize_images=True,
                normalization_type="standard",
                normalization_kwargs={},
                rescale_images=False,
                rescale_range=(-1.0, 1.0),
                rescale_percentiles=(0.0, 100.0),
                # Phase 11 renamed the top-level block; `from_training_config`
                # reads `config.undersampling`, so an `acceleration=` kwarg here
                # is simply never read and the schema default wins silently.
                undersampling=AccelConfig(),
                augmentation=None,
                transforms=[],
                coil_processing_mode="none",
            )
        )
        assert config.patch_size == (512, 512)
        assert config.trajectory_type == "radial"
        assert config.normalize_kspace is True
        assert config.kspace_percentile == 0.95
        assert config.acceleration == 4.0
        assert config.center_fraction == 0.1

    def test_an_omitted_patch_size_takes_the_schema_default(self):
        """Renamed from ``test_from_training_config_defaults_to_none_patch_size``.

        That test set ``patch_size=None`` and asserted the builder substituted
        ``(320, 320)``. Both halves are now obsolete: ``patch_size`` is a
        non-optional field on ``data.sampling`` with its own default, so a config
        cannot carry ``None``, and the builder no longer restates the fallback --
        it reads the declared field, which is the whole point of the schema
        owning it. Asserts the surviving behaviour: omit it, get the schema's
        value (normalised to 3-D by ``validate_patch_size``).
        """
        from spectramr.config.schemas.data import DataSamplingConfigSchema

        config = TorchIOTransformConfig.from_training_config(DataConfigStub(acceleration=None))
        assert config.patch_size == DataSamplingConfigSchema().patch_size == (320, 320, 1)

    def test_graph_encoding_config(self):
        """Test graph encoding configuration."""
        graph_config = {"k_neighbors": 16, "max_nodes": 8192}
        config = TorchIOTransformConfig(enable_graph_encoding=True, graph_config=graph_config)
        assert config.enable_graph_encoding is True
        assert config.graph_config == graph_config


class TestTorchIOTransformBuilder:
    """Tests for TorchIOTransformBuilder class."""

    def test_build_train_transforms_returns_compose(self):
        """Test that build_train_transforms returns a tio.Compose."""
        config = TorchIOTransformConfig()
        transforms = TorchIOTransformBuilder.build_train_transforms(config)
        assert isinstance(transforms, tio.Compose)

    def test_build_val_transforms_returns_compose(self):
        """Test that build_val_transforms returns a tio.Compose."""
        config = TorchIOTransformConfig()
        transforms = TorchIOTransformBuilder.build_val_transforms(config)
        assert isinstance(transforms, tio.Compose)

    def test_build_train_transforms_count_minimal(self):
        """Test that minimal config produces expected number of transforms."""
        config = TorchIOTransformConfig(
            augmentation_config=None,
            normalize_kspace=False,
            normalize_images=False,
            enable_graph_encoding=False,
        )
        transforms = TorchIOTransformBuilder.build_train_transforms(config)
        # Should have: spatial consistency (first) + physics + spatial consistency (last)
        # = 3 transforms minimum
        assert len(transforms.transforms) >= 3

    def test_geometric_standardization_is_gone(self):
        """M1: the geometry branch is deleted, and this test replaces the one
        that proved it added transforms.

        That test constructed `TorchIOTransformConfig(
        enable_geometric_standardization=True)` DIRECTLY — a state production
        could never reach, because `from_training_config` populated the flag
        with `getattr(_data_cfg, "enable_geometric_standardization", False)` on
        a name that is not a schema field. So it was green over a branch no arm
        had ever executed, while 11 arms declared the flag true in YAML and had
        the key dropped by the `extra="ignore"` block.

        It was DELETED rather than wired because the branch also appended
        `PhysicsSynchronization()`, which resolves its source key as "input"
        first — and on a k-space arm `input` IS k-space, so enabling it would
        have applied a second forward FFT on exactly those arms (A4).
        """
        import inspect

        assert not hasattr(TorchIOTransformConfig(), "enable_geometric_standardization")
        # CODE only: the deletion note left in place necessarily names the
        # class it removed, so a raw substring check reads its own explanation
        # as a survival. (It did, on the first run of this test.)
        src = inspect.getsource(TorchIOTransformBuilder.build_train_transforms)
        code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
        assert "SmartGeometricStandardization" not in code

    def test_build_train_transforms_with_normalization(self):
        """Test that normalization options add transforms."""
        config_no_norm = TorchIOTransformConfig(normalize_kspace=False, normalize_images=False)
        transforms_no_norm = TorchIOTransformBuilder.build_train_transforms(config_no_norm)

        config_with_kspace_norm = TorchIOTransformConfig(normalize_kspace=True)
        transforms_with_kspace_norm = TorchIOTransformBuilder.build_train_transforms(
            config_with_kspace_norm
        )

        # With kspace normalization should have more transforms
        assert len(transforms_with_kspace_norm.transforms) > len(transforms_no_norm.transforms)

    def test_build_val_transforms_excludes_augmentation(self):
        """Test that validation transforms do not include augmentation."""
        config = TorchIOTransformConfig(augmentation_config=None)
        train_transforms = TorchIOTransformBuilder.build_train_transforms(config)
        val_transforms = TorchIOTransformBuilder.build_val_transforms(config)

        # Both should work but not error on missing augmentation
        assert isinstance(train_transforms, tio.Compose)
        assert isinstance(val_transforms, tio.Compose)

    def test_physics_transform_cartesian(self):
        """Test physics transform for Cartesian trajectory."""
        config = TorchIOTransformConfig(trajectory_type="cartesian")
        transform = TorchIOTransformBuilder._build_physics_transform(config)
        assert isinstance(transform, tio.Transform)

    def test_physics_transform_non_cartesian(self):
        """Test physics transform for non-Cartesian trajectories."""
        for traj in ["radial", "spiral", "golden_angle", "epi"]:
            config = TorchIOTransformConfig(trajectory_type=traj)
            try:
                transform = TorchIOTransformBuilder._build_physics_transform(config)
                assert isinstance(transform, tio.Transform)
            except ImportError as e:
                if "torchkbnufft" in str(e):
                    pytest.skip(f"Skipping {traj} test: torchkbnufft not installed")
                raise

    def test_physics_transform_none_defaults_to_cartesian(self):
        """Test that None trajectory defaults to Cartesian."""
        config = TorchIOTransformConfig(trajectory_type=None)
        transform = TorchIOTransformBuilder._build_physics_transform(config)
        assert isinstance(transform, tio.Transform)

    def test_spatial_consistency_transform(self):
        """Test that spatial consistency transform is created."""
        transform = TorchIOTransformBuilder._build_spatial_consistency_transform()
        assert isinstance(transform, tio.Transform)

    def test_transform_pipeline_order_train(self):
        """Test that training transforms maintain correct order.

        Order should be:
        1. Spatial consistency (first)
        2. Geometry (if enabled)
        3. Physics sync after geometry
        4. Augmentations (if enabled)
        5. Physics sync after augmentation
        6. Physics dispatch
        7. Normalization
        8. Spatial consistency (last)
        """
        config = TorchIOTransformConfig(
            augmentation_config=None,
            normalize_kspace=False,
            normalize_images=False,
        )
        transforms = TorchIOTransformBuilder.build_train_transforms(config)

        # Should have at least: spatial consistency (first) + geometry + physics sync +
        # physics dispatch + spatial consistency (last)
        assert len(transforms.transforms) >= 5

        # First should be EnsureSpatialConsistency
        from spectramr.data.transforms.geometric import EnsureSpatialConsistency

        first_transform = transforms.transforms[0]
        assert isinstance(first_transform, EnsureSpatialConsistency)

        # The chain TERMINATES in _SyncSubjectAttributes (#1213) — a bookkeeping
        # no-op that mutates no tensor and adds no key, it only re-binds
        # ``Subject.__dict__`` so the patch sampler crops what the chain produced.
        # EnsureSpatialConsistency is still last among the transforms that touch
        # data, which is what this ordering contract is about.
        assert type(transforms.transforms[-1]).__name__ == "_SyncSubjectAttributes"
        last_transform = transforms.transforms[-2]
        assert isinstance(last_transform, EnsureSpatialConsistency)

    def test_transform_pipeline_order_val(self):
        """Test that validation transforms maintain correct order."""
        config = TorchIOTransformConfig(
            augmentation_config=None,
            normalize_kspace=False,
            normalize_images=False,
        )
        transforms = TorchIOTransformBuilder.build_val_transforms(config)

        # First and last should be spatial consistency
        from spectramr.data.transforms.geometric import EnsureSpatialConsistency

        first_transform = transforms.transforms[0]
        assert isinstance(first_transform, EnsureSpatialConsistency)

        # Terminal member is the __dict__ re-bind (#1213); see the train sibling.
        assert type(transforms.transforms[-1]).__name__ == "_SyncSubjectAttributes"
        last_transform = transforms.transforms[-2]
        assert isinstance(last_transform, EnsureSpatialConsistency)

    def test_multiple_trajectory_types(self):
        """Test building transforms for multiple trajectory types."""
        trajectories = ["cartesian", "radial", "spiral"]
        for traj in trajectories:
            config = TorchIOTransformConfig(trajectory_type=traj)
            try:
                train_transforms = TorchIOTransformBuilder.build_train_transforms(config)
                val_transforms = TorchIOTransformBuilder.build_val_transforms(config)

                assert isinstance(train_transforms, tio.Compose)
                assert isinstance(val_transforms, tio.Compose)
            except ImportError as e:
                if "torchkbnufft" in str(e):
                    pytest.skip(f"Skipping {traj} test: torchkbnufft not installed")
                raise

    def test_config_immutability_after_factory_method(self):
        """Test that config from factory method is valid and frozen."""

        config = TorchIOTransformConfig.from_training_config(
            DataConfigStub(
                patch_size=(256, 256),
                trajectory="radial",
                normalize_kspace=True,
                kspace_percentile=0.99,
                normalization_type="none",
                normalization_kwargs={},
                normalize_images=False,
                rescale_images=False,
                coil_processing_mode="none",
                rescale_range=(-1.0, 1.0),
                rescale_percentiles=(0.0, 100.0),
                acceleration=None,
            )
        )

        # Should not raise when building transforms
        try:
            train_transforms = TorchIOTransformBuilder.build_train_transforms(config)
            assert isinstance(train_transforms, tio.Compose)
        except ImportError as e:
            if "torchkbnufft" in str(e):
                pytest.skip("Skipping radial test: torchkbnufft not installed")
            raise


class TestTransformBuilderIntegration:
    """Integration tests for transform builder."""

    def test_complete_training_pipeline(self):
        """Test building a complete training pipeline."""
        config = TorchIOTransformConfig(
            patch_size=(256, 256),
            standardization_mode="smart",
            trajectory_type="cartesian",
            normalize_kspace=False,
            normalize_images=False,
        )

        transforms = TorchIOTransformBuilder.build_train_transforms(config)
        assert isinstance(transforms, tio.Compose)
        assert len(transforms.transforms) > 0

    def test_complete_validation_pipeline(self):
        """Test building a complete validation pipeline."""
        config = TorchIOTransformConfig(
            patch_size=(256, 256),
            standardization_mode="strict",
            trajectory_type="radial",
            normalize_kspace=True,
            normalize_images=True,
        )

        try:
            transforms = TorchIOTransformBuilder.build_val_transforms(config)
            assert isinstance(transforms, tio.Compose)
            assert len(transforms.transforms) > 0
        except ImportError as e:
            if "torchkbnufft" in str(e):
                pytest.skip("Skipping radial test: torchkbnufft not installed")
            raise

    def test_from_config_to_transforms_pipeline(self):
        """Test full pipeline from config object to transforms."""

        config = TorchIOTransformConfig.from_training_config(
            DataConfigStub(
                patch_size=(512, 512),
                trajectory="spiral",
                normalize_kspace=True,
                kspace_percentile=0.99,
                normalize_images=False,
                normalization_type="none",
                normalization_kwargs={},
                rescale_images=False,
                rescale_range=(-1.0, 1.0),
                rescale_percentiles=(0.0, 100.0),
                coil_processing_mode="none",
                augmentation=None,
                acceleration=None,
            )
        )
        try:
            train_transforms = TorchIOTransformBuilder.build_train_transforms(config)
            val_transforms = TorchIOTransformBuilder.build_val_transforms(config)

            assert isinstance(train_transforms, tio.Compose)
            assert isinstance(val_transforms, tio.Compose)
            assert len(train_transforms.transforms) > 0
            assert len(val_transforms.transforms) > 0
        except ImportError as e:
            if "torchkbnufft" in str(e):
                pytest.skip("Skipping spiral test: torchkbnufft not installed")
            raise


class TestImageUndersamplingBridge:
    """The image→k-space bridge (exp_c1): opt-in, appended last, zero default blast radius."""

    def test_appended_when_flag_set(self):
        cfg = TorchIOTransformConfig(acceleration=4, center_fraction=0.08, image_undersampling=True)
        for build in (
            TorchIOTransformBuilder.build_train_transforms,
            TorchIOTransformBuilder.build_val_transforms,
        ):
            names = [type(t).__name__ for t in build(cfg).transforms]
            assert "RetrospectiveImageUndersampling" in names
            # must be LAST — degrade the final magnitude image, after all spatial ops.
            # The chain then terminates in the _SyncSubjectAttributes bookkeeping
            # no-op (#1213), which touches no tensor, so the contract this test
            # guards ("nothing transforms the image after the undersampling") holds
            # at names[-2].
            assert names[-1] == "_SyncSubjectAttributes"
            assert names[-2] == "RetrospectiveImageUndersampling"

    def test_absent_by_default(self):
        cfg = TorchIOTransformConfig(acceleration=4, center_fraction=0.08)
        names = [
            type(t).__name__ for t in TorchIOTransformBuilder.build_train_transforms(cfg).transforms
        ]
        assert "RetrospectiveImageUndersampling" not in names


class TestValTransformsPicklable:
    """Regression: the val transform Compose must survive pickling.

    Python 3.14 changed the default multiprocessing start method on non-Mac
    POSIX from ``fork`` to ``forkserver``, which pickles the worker's arguments
    (including the Dataset and its transform Compose). The debug-stats probe
    used to be a *local* class (``build_val_transforms.<locals>.DebugStats``),
    whose pickle qualname contains ``<locals>`` and is therefore unpicklable —
    so val DataLoader workers crashed with ``PicklingError`` on the cluster.
    Hoisting it to module scope (``_ValDebugStats``) fixes it.
    """

    def test_val_compose_is_picklable(self):
        import pickle

        compose = TorchIOTransformBuilder.build_val_transforms(TorchIOTransformConfig())
        # Round-trips without PicklingError (the bug raised here).
        restored = pickle.loads(pickle.dumps(compose))
        assert isinstance(restored, tio.Compose)

    def test_debug_stats_is_module_level(self):
        compose = TorchIOTransformBuilder.build_val_transforms(TorchIOTransformConfig())
        debug = [t for t in compose.transforms if type(t).__name__ == "_ValDebugStats"]
        assert debug, "_ValDebugStats probe missing from val transforms"
        # A module-level class has no '<locals>' in its qualname (the tell of a
        # nested/local class that pickle cannot locate).
        assert "<locals>" not in type(debug[0]).__qualname__


# ── kspace_scale_domain reaches the transform (knob wiring, pitfall #15) ──────


class TestKSpaceScaleDomainWiring:
    """``data.kspace_scale_domain`` must reach KSpaceNormalizationTransform.

    A knob that parses but never reaches its consumer is the classic inert
    mechanism: the YAML would advertise a Parseval scale while the run silently
    used the k-space one.
    """

    @staticmethod
    def _cfg(**kw):
        from spectramr.data.builders.torchio_transform_builder import (
            TorchIOTransformConfig,
        )

        base = dict(patch_size=(16, 16, 1), normalize_kspace=True, log_scaling=True)
        base.update(kw)
        return TorchIOTransformConfig(**base)

    @staticmethod
    def _kspace_norm(pipeline):
        from spectramr.data.transforms.normalization import KSpaceNormalizationTransform

        return [t for t in pipeline.transforms if isinstance(t, KSpaceNormalizationTransform)]

    @pytest.mark.parametrize("domain", ["kspace", "image"])
    @pytest.mark.parametrize("which", ["train", "val"])
    def test_scale_domain_forwarded_to_both_pipelines(self, domain, which):
        from spectramr.data.builders.torchio_transform_builder import (
            TorchIOTransformBuilder,
        )

        cfg = self._cfg(kspace_scale_domain=domain)
        build = (
            TorchIOTransformBuilder.build_train_transforms
            if which == "train"
            else TorchIOTransformBuilder.build_val_transforms
        )
        found = self._kspace_norm(build(cfg))
        assert len(found) == 1, f"expected exactly one normalizer, got {len(found)}"
        assert found[0].scale_domain == domain

    def test_from_training_config_reads_the_data_knob(self):
        from spectramr.config.schemas.data import DataConfigSchema
        from spectramr.data.builders.torchio_transform_builder import (
            TorchIOTransformConfig,
        )

        # Canonical paths, resolved through `RENAMES`, not guessed:
        # `data.kspace_scale_domain` -> `data.processing.kspace_scale_domain`
        # is posture="raise" as of 2026-07-31, so the flat spelling this test
        # used no longer constructs at all. Its sibling
        # `data.normalize_kspace` -> `data.processing.enable_kspace_normalization`
        # is still posture="fold", but is written canonically here so the two
        # do not drift apart again.
        #
        # Non-vacuous: `kspace_scale_domain` defaults to "kspace", so asserting
        # "image" still proves the knob was read rather than defaulted.
        data = DataConfigSchema(
            processing={
                "enable_kspace_normalization": True,
                "kspace_scale_domain": "image",
            }
        )

        class _Proxy:
            def __init__(self, d):
                self._d = d
                # Mirrors the production ConfigProxy, which sets
                # `self.undersampling` (data_pipeline_director.py:341) precisely
                # so `__getattr__` does not delegate it to the data block.
                self.undersampling = None

            def __getattr__(self, name):
                return getattr(self._d, name)

        assert (
            TorchIOTransformConfig.from_training_config(_Proxy(data)).kspace_scale_domain == "image"
        )


def _builder(which: str):
    """The train / val entry point under test, selected by name."""
    return (
        TorchIOTransformBuilder.build_train_transforms
        if which == "train"
        else TorchIOTransformBuilder.build_val_transforms
    )


# ── B18: an unconstructible normalization_type must fail loud on BOTH paths ───


class TestUnknownNormalizationTypeRaises:
    """An image ``normalization_type`` the builder cannot construct must raise.

    Before this, the train path hit ``logger.warning`` and appended NO
    transform, and the val path had no unknown-type branch at all. Net effect:
    the arm trained AND was graded on un-normalised data, while only the train
    side even mentioned it -- a silent fallback (pitfall #9), which the sibling
    ``coil_processing_mode`` dispatch ~150 lines above has always rejected.
    Downstream it is worse than a no-op: a static ``data_range`` makes PSNR and
    SSIM meaningless once the images are off their assumed scale.
    """

    @staticmethod
    def _cfg(**kw):
        return TorchIOTransformConfig(**{"patch_size": (16, 16, 1), **kw})

    def test_train_path_raises(self):
        with pytest.raises(ValueError, match="Unknown normalization_type"):
            TorchIOTransformBuilder.build_train_transforms(self._cfg(normalization_type="scalar"))

    def test_val_path_raises(self):
        """The path that previously had NO unknown-type branch at all.

        The train warning at least left a trace in the log; the val chain was
        silent, so the same unimplemented value also decided what every
        validation metric was computed on, with nothing to grep for.
        """
        with pytest.raises(ValueError, match="Unknown normalization_type"):
            TorchIOTransformBuilder.build_val_transforms(self._cfg(normalization_type="scalar"))

    @pytest.mark.parametrize("which", ["train", "val"])
    def test_message_names_the_offending_value_and_the_implemented_set(self, which):
        """Both paths share one message helper, so both stay actionable."""
        build = _builder(which)
        with pytest.raises(ValueError) as excinfo:
            build(self._cfg(normalization_type="scalar"))

        message = str(excinfo.value)
        assert "'scalar'" in message
        for valid in IMPLEMENTED_NORMALIZATION_TYPES:
            assert repr(valid) in message, f"{valid!r} missing from the error text"

    @pytest.mark.parametrize("which", ["train", "val"])
    def test_kspace_normalization_logs_the_skip_instead_of_raising(self, which, caplog):
        """``normalize_kspace=True`` deliberately skips image normalization.

        Image norms (ZNormalization / RescaleIntensity) assume magnitude images
        and clamp or shift values, which destroys complex k-space, so the whole
        image-normalization block -- the raise included -- sits behind the
        k-space gate inside ``resolve_image_normalization``. The value is
        therefore never dispatched and must not raise; the run says so in the
        log instead. The line is the resolver's, so it is patched there: the
        builder no longer has a copy of it to emit.
        """
        cfg = self._cfg(normalize_kspace=True, normalization_type="scalar")
        with caplog.at_level("INFO", logger="spectramr.data.transforms.normalization"):
            pipeline = _builder(which)(cfg)  # must NOT raise

        assert isinstance(pipeline, tio.Compose)
        assert "Image normalization 'scalar' SKIPPED" in caplog.text

    @pytest.mark.parametrize("which", ["train", "val"])
    @pytest.mark.parametrize("normalization_type", IMPLEMENTED_NORMALIZATION_TYPES)
    def test_every_implemented_type_builds(self, normalization_type, which):
        """Parametrised over the constant, not a hand-copied list.

        ``IMPLEMENTED_NORMALIZATION_TYPES`` is the SSOT the raise quotes, so a
        strategy added to it without both dispatch branches fails here rather
        than at queue-build time on the cluster. Covers ``'none'`` too: it is a
        member, so neither path may raise on it.
        """
        pipeline = _builder(which)(self._cfg(normalization_type=normalization_type))
        assert isinstance(pipeline, tio.Compose)

    @pytest.mark.parametrize("which", ["train", "val"])
    def test_none_appends_no_intensity_transform(self, which):
        """``'none'`` means "no image normalization", not "quietly skipped".

        Re-pointed from ``_ComplexSafeIntensityTransform``, which no longer
        exists: ``normalization.py:1027`` records that
        :class:`ImageNormalizationTransform` *"replaced"* it, keeping the
        magnitude-first guarantee but narrowing it to the two strategies that
        need it (``ZSCORE``/``MINMAX``; ``PERCENTILE`` preserves phase by
        construction). The assertion is unchanged -- only its subject was
        renamed, so this was a stale test rather than the environment gap the
        ``ImportError`` on job 8004252 made it look like.
        """
        from spectramr.data.transforms.normalization import (
            ImageNormalizationTransform,
        )

        pipeline = _builder(which)(self._cfg(normalization_type="none"))
        assert not [t for t in pipeline.transforms if isinstance(t, ImageNormalizationTransform)]


class TestNormalizationTypeSchemaBuilderBridge:
    """The schema Literal and the builder's implemented set must stay coherent.

    They are deliberately not identical: ``'robust_percentile'`` is accepted by
    the schema and folded to ``'percentile'`` by the alias table in
    ``NormalizationStrategy.from_string``, which ``resolve_image_normalization``
    reaches when a chain is built. That fold is the ONLY licensed gap --
    without it a schema-legal arm would reach the raise, which is how
    ``'scalar'`` (a Literal member with no dispatch branch on either path)
    survived unnoticed until it was removed.
    """

    def test_robust_percentile_is_kept_verbatim_and_folded_by_the_one_owner(self):
        """``from_training_config`` used to fold the spelling itself.

        That was a second copy of an alias the enum already owns (and the
        predict path carried a third). The config now carries the declared
        spelling untouched, and the chain built from it still normalizes with
        PERCENTILE -- the fold happened in the resolver, not in the builder.
        """
        from spectramr.data.transforms.normalization import (
            ImageNormalizationTransform,
            NormalizationStrategy,
        )

        cfg = TorchIOTransformConfig.from_training_config(
            DataConfigStub(normalization_type="robust_percentile")
        )
        assert cfg.normalization_type == "robust_percentile"
        assert cfg.normalization_type not in IMPLEMENTED_NORMALIZATION_TYPES

        for build in (_builder("train"), _builder("val")):
            norms = [t for t in build(cfg).transforms if isinstance(t, ImageNormalizationTransform)]
            assert len(norms) == 1
            assert norms[0].spec.config.strategy is NormalizationStrategy.PERCENTILE

    def test_the_fold_is_the_only_gap_between_the_two_sets(self):
        from typing import get_args

        from spectramr.config.schemas.data import DataProcessingConfigSchema

        accepted = set(
            get_args(DataProcessingConfigSchema.model_fields["normalization_type"].annotation)
        )
        assert accepted - set(IMPLEMENTED_NORMALIZATION_TYPES) == {"robust_percentile"}, (
            "a schema-accepted normalization_type has no dispatch branch and no "
            "fold -- declaring it would now raise at transform-build time"
        )
        # Nothing may be implemented that the schema cannot express, either.
        assert set(IMPLEMENTED_NORMALIZATION_TYPES) <= accepted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestConfigDrivenTransformRegistry:
    """``data.processing.transforms`` resolves through the registry (D9).

    The old consumer matched the single literal ``"graph_encoding"`` and
    ``break``ed, so every other declared entry validated at load and was then
    silently discarded -- committed arms named for a transform trained without
    it and reported success.
    """

    def test_unregistered_name_raises(self):
        import pytest

        with pytest.raises(KeyError) as exc:
            TorchIOTransformConfig.from_training_config(
                DataConfigStub(transforms=[{"name": "no_such_transform"}])
            )
        assert "no_such_transform" in str(exc.value)

    def test_dotted_path_raises_with_the_hint(self):
        """Four committed arms use this spelling; it never resolved."""
        import pytest

        with pytest.raises(KeyError) as exc:
            TorchIOTransformConfig.from_training_config(
                DataConfigStub(
                    transforms=[
                        {"name": "spectramr.data.transforms.slice_profile.SliceProfileTransform"}
                    ]
                )
            )
        assert "Dotted import paths" in str(exc.value)

    def test_entry_without_a_name_raises(self):
        """One committed arm spells the key ``type:`` instead of ``name:``."""
        import pytest

        with pytest.raises(ValueError, match="no 'name'"):
            TorchIOTransformConfig.from_training_config(
                DataConfigStub(transforms=[{"type": "scout_acquisition"}])
            )

    def test_registered_name_lands_in_extra_transforms(self):
        cfg = TorchIOTransformConfig.from_training_config(
            DataConfigStub(transforms=[{"name": "phase_residual", "kwargs": {"kernel_size": 7}}])
        )
        assert cfg.extra_transforms == [("phase_residual", {"kernel_size": 7})]

    def test_flat_kwargs_are_accepted(self):
        """The committed ``graph_encoding`` arms write kwargs flat."""
        cfg = TorchIOTransformConfig.from_training_config(
            DataConfigStub(
                transforms=[{"name": "graph_encoding", "k_neighbors": 12, "max_nodes": 512}]
            )
        )
        assert cfg.enable_graph_encoding is True
        assert cfg.graph_config == {"k_neighbors": 12, "max_nodes": 512}

    def test_graph_encoding_no_longer_short_circuits_the_rest(self):
        """The old loop ``break``ed on graph_encoding, dropping later entries."""
        cfg = TorchIOTransformConfig.from_training_config(
            DataConfigStub(
                transforms=[
                    {"name": "graph_encoding"},
                    {"name": "phase_residual"},
                ]
            )
        )
        assert cfg.enable_graph_encoding is True
        assert [n for n, _ in cfg.extra_transforms] == ["phase_residual"]

    def test_transforms_are_appended_to_BOTH_chains(self):
        """Train-only application would grade the model on untransformed data."""
        cfg = TorchIOTransformConfig(extra_transforms=[("phase_residual", {"kernel_size": 5})])
        for chain in ("TRAIN", "VAL"):
            out = []
            TorchIOTransformBuilder._append_registry_transforms(out, cfg, chain)
            assert len(out) == 1
            assert type(out[0]).__name__ == "PhaseResidualTransform"

    def test_no_declarations_appends_nothing(self):
        out = []
        TorchIOTransformBuilder._append_registry_transforms(out, TorchIOTransformConfig(), "TRAIN")
        assert out == []

    def test_the_guard_reads_the_field_directly_not_through_a_default(self):
        """A defaulted read here could never fire, and would hide a rename.

        The guard used ``getattr(config, "extra_transforms", None)``. The field
        is declared ``default_factory=list``, so it is present on every
        instance and the default was unreachable -- but an unreachable default
        is indistinguishable from one silently swallowing a renamed field, and
        that spelling would disable every declared transform on every arm
        without raising. ``check_getattr_names_a_real_field`` rejects the shape
        for exactly that reason.

        Asserted behaviourally: a receiver missing the field must RAISE, which
        is the property the defaulted read destroyed.
        """
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(TorchIOTransformConfig)}
        assert "extra_transforms" in field_names, "precondition"
        assert TorchIOTransformConfig().extra_transforms == []

        class _ConfigWithoutTheField:
            pass

        with pytest.raises(AttributeError, match="extra_transforms"):
            TorchIOTransformBuilder._append_registry_transforms(
                [], _ConfigWithoutTheField(), "TRAIN"
            )


class TestPhysicsSyncCarriesTheDomainTheBuilderProved:
    """The builder resolves image-primary vs k-space, so it must SAY so.

    Reaching the non-k-space branch proves `input` holds an image. Not passing
    that on left the transform's own ambiguity guard to refuse, killing the
    DataLoader for 10 `10_paradigms` arms in cluster job 8012333 -- the caller
    established the fact, discarded it, and the callee refused for want of
    exactly that fact. Asserted behaviourally (build it, look at the instance)
    rather than by grepping the source.
    """

    @staticmethod
    def _physics_sync_in(dataset_type):
        from spectramr.config.schemas.augmentation import AugmentationConfigSchema
        from spectramr.data.transforms.physics_sync import PhysicsSynchronization

        config = TorchIOTransformConfig(
            augmentation_config=AugmentationConfigSchema(enabled=True, enable_flip=True),
            dataset_type=dataset_type,
        )
        built = TorchIOTransformBuilder.build_train_transforms(config)
        return [t for t in built.transforms if isinstance(t, PhysicsSynchronization)]

    def test_image_primary_arm_declares_input_is_image(self):
        found = self._physics_sync_in("nifti_paired")
        assert len(found) == 1, "image-primary arm lost its PhysicsSynchronization"
        assert found[0].input_is_image is True, (
            "the builder proved this arm is image-primary and then did not say so"
        )

    def test_kspace_arm_is_still_skipped_entirely(self):
        """The declaration must not become a way to reach a k-space arm."""
        assert self._physics_sync_in("m4raw") == []


class TestChainOutputSurvivesPatchExtraction:
    """The built chain's output must be what the patch sampler crops (#1213).

    ``tio.Subject`` mirrors its entries into ``self.__dict__``;
    ``Subject.__setitem__`` is not defined, so ``subject[key] = value`` diverges
    the two views silently. ``tio.Crop.apply_transform`` — the engine behind every
    :class:`torchio.data.PatchSampler` and therefore every ``tio.Queue`` — builds
    its output *solely* from ``__dict__``. A chain could run in full, mutate
    exactly what it was told to, and have the whole effect discarded at patch
    extraction: replaced images came back **pre-transform**, and newly-added
    non-image keys were **dropped**.

    Measured on the ``experiment_11_attention_none`` arm before the fix: the
    declared k-space normalization yielded ``|k|max 3.98`` on the Subject while
    the batch out of ``tio.Queue`` carried ``1974`` and neither ``kspace_scale``
    nor ``kspace_normalized`` — so the strategy's gate read "never normalized"
    and compensated, masking the loss (#1211).

    Both chains terminate in ``_SyncSubjectAttributes`` as the backstop.
    """

    @staticmethod
    def _kspace_subject() -> tio.Subject:
        """A DC-heavy k-space Subject shaped like the M4Raw loader's output."""
        import torch

        torch.manual_seed(0)
        k = torch.randn(2, 32, 32, 4) * 0.1
        k[:, 16, 16, :] = 2500.0
        return tio.Subject(input=tio.ScalarImage(tensor=k))

    @pytest.mark.parametrize(
        "build",
        [
            TorchIOTransformBuilder.build_train_transforms,
            TorchIOTransformBuilder.build_val_transforms,
        ],
        ids=["train", "val"],
    )
    def test_both_chains_terminate_in_the_attribute_sync(self, build) -> None:
        """The backstop is present and last, on both splits.

        Last matters: a sync placed mid-chain leaves every later member free to
        desync again.
        """
        names = [type(t).__name__ for t in build(TorchIOTransformConfig()).transforms]
        assert names[-1] == "_SyncSubjectAttributes"
        assert names.count("_SyncSubjectAttributes") == 1

    @pytest.mark.parametrize(
        "build",
        [
            TorchIOTransformBuilder.build_train_transforms,
            TorchIOTransformBuilder.build_val_transforms,
        ],
        ids=["train", "val"],
    )
    def test_a_patch_of_the_chain_output_is_normalized(self, build) -> None:
        """The declared normalization reaches the patch, markers and all.

        The end-to-end assertion the defect defeated: build the real chain with
        ``normalize_kspace`` declared, run it, then extract a patch the way
        ``tio.Queue`` does. A ``log1p``-compressed float32 magnitude cannot exceed
        ~44, so a patch near the input's 2500 means the chain's work was thrown
        away.
        """
        config = TorchIOTransformConfig(
            normalize_kspace=True,
            kspace_percentile=0.95,
            log_scaling=True,
            patch_size=(16, 16, 1),
        )
        chain = build(config)
        normalized = chain(self._kspace_subject())
        assert float(normalized["input"].data.abs().max()) < 44.0  # sanity: it ran

        sampler = tio.UniformSampler((16, 16, 1))
        patch = next(iter(sampler(normalized)))

        assert float(patch["input"].data.abs().max()) < 44.0
        for key in ("kspace_scale", "kspace_log_scaled", "kspace_normalized"):
            assert key in patch, f"{key!r} dropped at patch extraction (#1213)"
        assert bool(patch["kspace_normalized"]) is True

    @pytest.mark.parametrize(
        "build",
        [
            TorchIOTransformBuilder.build_train_transforms,
            TorchIOTransformBuilder.build_val_transforms,
        ],
        ids=["train", "val"],
    )
    def test_attribute_access_agrees_with_item_access(self, build) -> None:
        """``subject.input`` must return what ``subject['input']`` returns.

        The sampler-independent half of #1213, and the val chain's *live* reason to
        carry the sync. ``tio.Subject`` mirrors its entries into ``__dict__``
        precisely so attribute access works, so an unsynced chain hands
        ``subject.input`` the **pre-transform** object with no crop anywhere in the
        pipeline. Val does not patch-sample today (``build_val_queue`` has no
        production caller — its only reference is a docstring example, #1210), so
        without this consumer the val-side sync would read as dead weight and be
        removed by the next person to tidy the chain.
        """
        config = TorchIOTransformConfig(
            normalize_kspace=True,
            kspace_percentile=0.95,
            log_scaling=True,
        )
        out = build(config)(self._kspace_subject())

        assert out.input is out["input"], "attribute access reads a stale __dict__"
        assert float(out.input.data.abs().max()) < 44.0


class TestBaseAccelerationIsAppliedLiterally:
    """Cohort review 2026-09-02 (T0.3): the loader used to substitute ``max_acceleration``
    whenever ``base_acceleration <= 1.0``, so "1.0 = fully sampled" became 8x."""

    @staticmethod
    def _config(base: float, maximum: float = 8.0):
        class AccelConfig:
            base_acceleration = base
            max_acceleration = maximum
            center_fraction = 0.08

        return TorchIOTransformConfig.from_training_config(
            DataConfigStub(patch_size=(64, 64), trajectory="cartesian", undersampling=AccelConfig())
        )

    def test_base_one_means_fully_sampled_not_the_maximum(self):
        """The planted violation: base 1.0 with max 8.0 used to give 8.0."""
        assert self._config(1.0, 8.0).acceleration == 1.0

    def test_base_above_one_is_applied_as_declared(self):
        assert self._config(4.0, 32.0).acceleration == 4.0


class TestBothChainsResolveThroughTheOneOwner:
    """Neither chain decides image normalization itself any more.

    The three decisions -- k-space precedence, the ``robust_percentile`` fold
    and the spec -- used to be copied into the train chain, the val chain and
    ``pipelines/infer.py``. A copy that drifts does not raise; it produces a
    differently scaled tensor on one side. So the guard is not "the outputs
    agree today" but "there is exactly one call, and the transform appended is
    the object that call returned".
    """

    @pytest.mark.parametrize("which", ["train", "val"])
    def test_the_chain_makes_exactly_one_call_and_appends_its_answer(self, which, monkeypatch):
        from spectramr.data.transforms import normalization as norm_mod
        from spectramr.data.transforms.normalization import ImageNormalizationTransform

        calls: list[tuple[dict, object]] = []
        real = norm_mod.resolve_image_normalization

        def _recording(**kwargs):
            spec = real(**kwargs)
            calls.append((kwargs, spec))
            return spec

        monkeypatch.setattr(norm_mod, "resolve_image_normalization", _recording)
        cfg = TorchIOTransformConfig(
            patch_size=(16, 16, 1),
            normalization_type="robust_percentile",
            dataset_type="nifti",
        )
        pipeline = _builder(which)(cfg)

        assert len(calls) == 1, "one owner means one call per chain"
        kwargs, spec = calls[0]
        assert kwargs == {
            "normalization_type": "robust_percentile",
            "dataset_type": "nifti",
            "normalization_kwargs": {},
            "kspace_normalization_enabled": False,
        }, "the chain hands the resolver the declared spelling, unfolded"
        appended = [t for t in pipeline.transforms if isinstance(t, ImageNormalizationTransform)]
        assert len(appended) == 1
        assert appended[0].spec is spec, (
            "the transform must carry the resolver's spec, not a re-derivation"
        )

    @pytest.mark.parametrize("which", ["train", "val"])
    def test_a_none_answer_appends_nothing(self, which, monkeypatch):
        """The k-space precedence verdict is the resolver's; the chain obeys it."""
        from spectramr.data.transforms import normalization as norm_mod
        from spectramr.data.transforms.normalization import ImageNormalizationTransform

        monkeypatch.setattr(norm_mod, "resolve_image_normalization", lambda **kw: None)
        cfg = TorchIOTransformConfig(patch_size=(16, 16, 1), normalization_type="percentile")
        pipeline = _builder(which)(cfg)
        assert not [t for t in pipeline.transforms if isinstance(t, ImageNormalizationTransform)]

    def test_the_builder_no_longer_spells_the_fold(self):
        """Source guard for the copy that was deleted, with its shape pinned.

        The fold copy read ``if normalization_type == "robust_percentile"``.
        A re-introduced copy in any spelling of that comparison turns this red.
        """
        import inspect

        from spectramr.data.builders import torchio_transform_builder as mod

        assert not _fold_copies(inspect.getsource(mod)), (
            "the builder folds robust_percentile itself again; the alias table "
            "in NormalizationStrategy.from_string is the one owner"
        )

    def test_the_source_guard_sees_a_planted_copy(self):
        """A guard is only a guard for the shape it has been watched to catch."""
        planted = 'x = 1\nif normalization_type == "robust_percentile":\n    normalization_type = "percentile"\n'
        assert _fold_copies(planted) == [2]
        planted_single = "if nt == 'robust_percentile': nt = 'percentile'"
        assert _fold_copies(planted_single) == [1]
        assert _fold_copies("# the fold lives in from_string\n") == []


def _fold_copies(source: str) -> list[int]:
    """Line numbers of ``== "robust_percentile"`` comparisons in code (not comments)."""
    import re

    pattern = re.compile(r"""==\s*['"]robust_percentile['"]""")
    hits = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        code = line.split("#", 1)[0]
        if pattern.search(code):
            hits.append(lineno)
    return hits


# ── #2093: the coil mode the three prior-method arms need ─────────────────────


class TestCoilModeDispatchForPriorMethodArms:
    """``rss_per_channel`` mounts a combine; ``rss`` deliberately still does not.

    CDiffMR, FDB and Shen 2024 are all single-coil methods whose networks take
    two channels, so 4-coil M4Raw has to be reduced before the model sees it.
    ``rss`` cannot do that job: it has always declared "the source is already
    combined" and mounted nothing, and seven arms rely on that reading. The
    second test is what stops this change from quietly altering them.
    """

    @staticmethod
    def _cfg(mode: str):
        return TorchIOTransformConfig(patch_size=(16, 16, 1), coil_processing_mode=mode)

    @staticmethod
    def _combine_transforms(pipeline):
        from spectramr.data.transforms.kspace_coil_transforms import CoilCombineTransform

        return [t for t in pipeline.transforms if isinstance(t, CoilCombineTransform)]

    @pytest.mark.parametrize("which", ["train", "val"])
    def test_rss_per_channel_mounts_the_combine(self, which):
        mounted = self._combine_transforms(_builder(which)(self._cfg("rss_per_channel")))

        assert len(mounted) == 1
        assert mounted[0].method == "rss_per_channel"

    @pytest.mark.parametrize("which", ["train", "val"])
    def test_rss_still_mounts_nothing(self, which):
        """The planted red. Giving ``rss`` a transform would change 7 arms in place."""
        assert self._combine_transforms(_builder(which)(self._cfg("rss"))) == []

    def test_an_unknown_mode_still_raises(self):
        """Widening the dispatch must not widen it to everything (pitfall #9)."""
        with pytest.raises(ValueError, match="Unknown coil_processing_mode"):
            TorchIOTransformBuilder.build_train_transforms(self._cfg("rss_per_coil"))
