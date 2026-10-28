"""Phase 1: Comprehensive Test Suite for Leaf Builders

Tests for all Phase 1 leaf component builders:
- Model builders (Generator, Discriminator, Encoder, Decoder)
- Optimizer builders (Optimizer, Scheduler, GradScaler)
- Loss builders (Loss)
- Data builders (Dataset, DataLoader, Pipeline)
- Physics builders (FFT, Mask, DC, Physics)
"""

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from tests.utils.corpus import tracked_yamls

try:
    from spectramr.config.settings import TrainingSettings
except ImportError:
    TrainingSettings = Mock

try:
    from spectramr.infrastructure.builders.leaf import (
        DataConsistencyBuilder,
        DataLoaderBuilder,
        DatasetBuilder,
        DecoderBuilder,
        DiscriminatorBuilder,
        EncoderBuilder,
        FFTBuilder,
        GeneratorBuilder,
        MaskBuilder,
        OptimizerBuilder,
        PhysicsBuilder,
    )
except ImportError:
    # Skip tests if builders not available
    pytest.skip("Builders not available", allow_module_level=True)


@pytest.fixture
def config():
    """Fixture: Valid training configuration."""
    try:
        config_path = Path(__file__).parent.parent.parent.parent / "experiments" / "training"
        config_files = list(tracked_yamls(config_path, recursive=False))
        if config_files:
            return TrainingSettings.from_yaml(str(config_files[0]))
    except Exception:
        pass

    # Return a mock config with necessary attributes
    mock_config = Mock(spec=TrainingSettings)
    mock_config.data = Mock(dataset_type="fastmri", batch_size=32)
    # The REAL schema, not a Mock. `resolve_optimizer_spec` reads ~15 knobs
    # plus `model_fields_set` to tell a declared value from a defaulted one, and
    # a Mock auto-creates every one of them as a truthy non-None object -- so
    # `betas` looks declared, `model_fields_set` is not iterable, and each fix
    # reveals the next. A frozen schema instance gives correct defaults for free.
    from spectramr.config.schemas.optimization import OptimizationConfigSchema

    mock_config.optimization = OptimizationConfigSchema(
        optimizer={"type": "Adam", "learning_rate": 1e-3}
    )
    return mock_config


# ============================================================================
# Model Builder Tests
# ============================================================================


class TestGeneratorBuilder:
    """Tests for GeneratorBuilder."""

    def test_initialization(self, config):
        """Test builder initialization."""
        builder = GeneratorBuilder(config)
        assert builder is not None
        assert isinstance(builder, GeneratorBuilder)

    def test_fluent_api_chaining(self, config):
        """Test fluent API method chaining."""
        builder = GeneratorBuilder(config)
        result = builder.with_architecture("standard_unet")
        assert result is builder  # Should return self

    def test_with_architecture(self, config):
        """Test setting architecture."""
        builder = GeneratorBuilder(config)
        builder.with_architecture("standard_unet")
        assert builder._architecture == "standard_unet"

    def test_with_channels(self, config):
        """Test setting input/output channels."""
        builder = GeneratorBuilder(config)
        builder.with_input_channels(2).with_output_channels(2)
        assert builder._in_channels == 2
        assert builder._out_channels == 2

    def test_validate_requires_architecture(self, config):
        """Test validation requires architecture."""
        builder = GeneratorBuilder(config)
        builder.with_input_channels(2).with_output_channels(2)
        with pytest.raises(ValueError, match="Architecture not specified"):
            builder.validate()

    def test_build_returns_module(self, config):
        """Test build returns nn.Module."""
        builder = GeneratorBuilder(config)
        try:
            model = builder.with_architecture("standard_unet").build()
            assert isinstance(model, nn.Module)
        except Exception:
            # May fail if model registry not fully set up
            pytest.skip("Model registry not available")

    def test_build_does_not_leak_checkpoint_path_to_factory(self, monkeypatch):
        """REGRESSION: ``model.checkpoint_path`` must NOT reach the model factory.

        ``checkpoint_path`` is the warm-start / transfer-init knob, loaded by
        ``ModelBuilder._warm_start`` AFTER the generator is built — it is
        NOT a constructor parameter. Forwarding it to a strict model config
        (``UNetConfig``, a frozen ``@dataclass``) crashed
        ``exp_p1_b1_equivariance_conformal`` / ``eval_c5_exchangeability_test`` with
        ``Unexpected keyword argument 'checkpoint_path' for UNetConfig`` (cluster
        VF/mrixfields smoke 2026-06-16). It must be filtered by ``_SKIP`` alongside
        ``target_domain`` / ``conditioning``.
        """
        from spectramr.config.schemas.model import ModelConfigSchema
        from spectramr.models.factories import model_factory

        captured: dict = {}

        def _stub_create_generator(_self, **kwargs):
            captured.update(kwargs)
            return nn.Identity()

        monkeypatch.setattr(
            model_factory.ModelFactory, "create_generator", _stub_create_generator
        )

        cfg = Mock(spec=TrainingSettings)
        cfg.model = ModelConfigSchema(
            model_type="standard_unet",
            in_channels=2,
            out_channels=2,
            checkpoint_path="/fake/warm_start.ckpt",  # the field that used to leak
        )

        gen = (
            GeneratorBuilder(cfg, device="cpu")
            .with_architecture("standard_unet")
            .with_input_channels(2)
            .with_output_channels(2)
            .build()
        )
        assert isinstance(gen, nn.Module)
        assert "checkpoint_path" not in captured  # filtered by _SKIP (the fix)
        assert "base_channels" in captured  # non-skipped model fields still forwarded


class TestDiscriminatorBuilder:
    """Tests for DiscriminatorBuilder."""

    def test_initialization(self, config):
        """Test builder initialization."""
        builder = DiscriminatorBuilder(config)
        assert builder is not None

    def test_with_architecture(self, config):
        """Test setting architecture."""
        builder = DiscriminatorBuilder(config)
        builder.with_architecture("patch_gan")
        assert builder._architecture == "patch_gan"

    def test_validate_requires_architecture(self, config):
        """Test validation requires architecture."""
        builder = DiscriminatorBuilder(config)
        with pytest.raises(ValueError, match="Architecture not specified"):
            builder.validate()


class TestEncoderBuilder:
    """Tests for EncoderBuilder."""

    def test_initialization(self, config):
        """Test builder initialization."""
        builder = EncoderBuilder(config)
        assert builder is not None

    def test_with_latent_dim(self, config):
        """Test setting latent dimension."""
        builder = EncoderBuilder(config)
        builder.with_latent_dim(64)
        assert builder._latent_dim == 64


class TestDecoderBuilder:
    """Tests for DecoderBuilder."""

    def test_initialization(self, config):
        """Test builder initialization."""
        builder = DecoderBuilder(config)
        assert builder is not None

    def test_with_latent_dim(self, config):
        """Test setting latent dimension."""
        builder = DecoderBuilder(config)
        builder.with_latent_dim(64)
        assert builder._latent_dim == 64


# ============================================================================
# Optimizer Builder Tests
# ============================================================================


class TestOptimizerBuilder:
    """Tests for OptimizerBuilder."""

    def test_initialization(self, config):
        """Test builder initialization."""
        model = nn.Linear(10, 5)
        builder = OptimizerBuilder(config, params=model.parameters())
        assert builder is not None

    def test_fluent_api_chaining(self, config):
        """Test fluent API method chaining."""
        model = nn.Linear(10, 5)
        builder = OptimizerBuilder(config, params=model.parameters())
        result = builder.with_type("Adam")
        assert result is builder

    def test_with_learning_rate(self, config):
        """Test setting learning rate."""
        model = nn.Linear(10, 5)
        builder = OptimizerBuilder(config, params=model.parameters())
        builder.with_learning_rate(1e-3)
        assert builder._learning_rate == 1e-3

    def test_invalid_learning_rate_validation(self, config):
        """Test validation rejects invalid learning rate."""
        model = nn.Linear(10, 5)
        builder = OptimizerBuilder(config, params=model.parameters())
        builder.with_learning_rate(-1e-3)
        with pytest.raises(ValueError):
            builder.validate()

    def test_build_adam_optimizer(self, config):
        """Test building Adam optimizer."""
        model = nn.Linear(10, 5)
        builder = OptimizerBuilder(config, params=model.parameters())
        optimizer = (
            builder.with_type("Adam").with_learning_rate(1e-3).with_weight_decay(1e-4).build()
        )
        assert isinstance(optimizer, optim.Adam)

    def test_build_sgd_optimizer(self, config):
        """Test building SGD optimizer."""
        model = nn.Linear(10, 5)
        builder = OptimizerBuilder(config, params=model.parameters())
        optimizer = builder.with_type("SGD").with_learning_rate(1e-2).build()
        assert isinstance(optimizer, optim.SGD)

    def test_with_betas(self, config):
        """Test setting Adam betas."""
        model = nn.Linear(10, 5)
        builder = OptimizerBuilder(config, params=model.parameters())
        builder.with_betas(0.9, 0.999)
        assert builder._betas == (0.9, 0.999)


# ============================================================================
# Loss Builder Tests
# ============================================================================


# ============================================================================
# Data Builder Tests
# ============================================================================


class TestDatasetBuilder:
    """Tests for DatasetBuilder."""

    def test_initialization(self, config):
        """Test builder initialization."""
        builder = DatasetBuilder(config)
        assert builder is not None

    def test_with_type(self, config):
        """Test setting dataset type."""
        builder = DatasetBuilder(config)
        builder.with_type("fastmri")
        assert builder._dataset_type == "fastmri"

    def test_with_split(self, config):
        """Test setting data split."""
        builder = DatasetBuilder(config)
        builder.with_split("train")
        assert builder._split == "train"

    def test_with_motion_artifacts(self, config):
        """Test setting motion artifacts."""
        builder = DatasetBuilder(config)
        builder.with_motion_artifacts(True)
        assert builder._motion_artifacts is True

    def test_with_noise(self, config):
        """Test setting noise level."""
        builder = DatasetBuilder(config)
        builder.with_noise(0.01)
        assert builder._noise_level == 0.01


class TestDataLoaderBuilder:
    """Tests for DataLoaderBuilder."""

    def test_initialization_with_dummy_dataset(self, config):
        """Test builder initialization with dummy dataset."""
        dataset = torch.utils.data.TensorDataset(torch.randn(10, 3, 32, 32))
        builder = DataLoaderBuilder(config, dataset=dataset)
        assert builder is not None

    def test_with_batch_size(self, config):
        """Test setting batch size."""
        dataset = torch.utils.data.TensorDataset(torch.randn(10, 3, 32, 32))
        builder = DataLoaderBuilder(config, dataset=dataset)
        builder.with_batch_size(32)
        assert builder._batch_size == 32

    def test_invalid_batch_size_raises(self, config):
        """Test invalid batch size raises error."""
        dataset = torch.utils.data.TensorDataset(torch.randn(10, 3, 32, 32))
        builder = DataLoaderBuilder(config, dataset=dataset)
        with pytest.raises(ValueError):
            builder.with_batch_size(-1)

    def test_with_num_workers(self, config):
        """Test setting number of workers."""
        dataset = torch.utils.data.TensorDataset(torch.randn(10, 3, 32, 32))
        builder = DataLoaderBuilder(config, dataset=dataset)
        builder.with_num_workers(4)
        assert builder._num_workers == 4

    def test_build_dataloader(self, config):
        """Test building DataLoader."""
        dataset = torch.utils.data.TensorDataset(torch.randn(10, 3, 32, 32))
        config.data.collation.strategy = "image"
        builder = DataLoaderBuilder(config, dataset=dataset)
        loader = builder.with_batch_size(2).with_num_workers(0).build()
        assert isinstance(loader, torch.utils.data.DataLoader)
        assert loader.batch_size == 2


# ============================================================================
# Physics Builder Tests
# ============================================================================


class TestFFTBuilder:
    """Tests for FFTBuilder."""

    def test_initialization(self, config):
        """Test builder initialization."""
        builder = FFTBuilder(config)
        assert builder is not None

    def test_with_norm(self, config):
        """Test setting FFT norm."""
        builder = FFTBuilder(config)
        builder.with_norm("ortho")
        assert builder._norm == "ortho"

    def test_invalid_norm_raises(self, config):
        """Test invalid norm raises error."""
        builder = FFTBuilder(config)
        with pytest.raises(ValueError):
            builder.with_norm("invalid")

    def test_with_centering_true_is_accepted(self, config):
        """The only supported value is True, and it is stored."""
        builder = FFTBuilder(config)
        builder.with_centering(True)
        assert builder._centering is True

    # ``with_centering(False)`` raising is enforced ONCE, by the paired test
    # ``tests/unit/infrastructure/builders/leaf/test_physics_builders.py::
    # test_fftbuilder_with_centering_rejects_false`` (non-negotiable 10 puts the
    # owner beside ``src/spectramr/infrastructure/builders/leaf/physics_builders.py``;
    # this directory pairs with no source tree -- ``src/spectramr/builders/`` does not
    # exist). A second copy lived here and pinned the raise by its prose
    # (``match="always centers"``); PR #1442 reworded the message without touching
    # it and the duplicate went red while the owner stayed green (#1462). Per
    # non-negotiable 17 the loser's enforcement is deleted, not re-synced -- two
    # checkers for one invariant means neither is audited as the sole line of
    # defence. Do not re-add it here; strengthen the owner instead.

    def test_build_fft(self, config):
        """Test building FFT operator."""
        builder = FFTBuilder(config)
        fft = builder.with_norm("ortho").build()
        assert fft is not None


class TestMaskBuilder:
    """Tests for MaskBuilder."""

    def test_initialization(self, config):
        """Test builder initialization."""
        builder = MaskBuilder(config)
        assert builder is not None

    def test_with_shape(self, config):
        """Test setting k-space shape."""
        builder = MaskBuilder(config)
        builder.with_shape((320, 256))
        assert builder._shape == (320, 256)

    def test_invalid_shape_raises(self, config):
        """Test invalid shape raises error."""
        builder = MaskBuilder(config)
        with pytest.raises(ValueError):
            builder.with_shape((320,))  # Must be 2D

    def test_with_sampling_pattern_normalises_the_friendly_alias(self, config):
        """``cartesian`` is stored as the real MaskType member.

        The alias is normalised at the boundary so ``MaskType(pattern)``
        downstream never sees a value that is not a member — the assertion used
        to pin the pre-normalisation spelling, which is the state that would
        crash later rather than here.
        """
        builder = MaskBuilder(config)
        builder.with_sampling_pattern("cartesian")
        assert builder._sampling_pattern == "uniform_cartesian"

    def test_with_sampling_pattern_accepts_the_canonical_name(self, config):
        builder = MaskBuilder(config)
        builder.with_sampling_pattern("radial")
        assert builder._sampling_pattern == "radial"

    def test_with_acceleration(self, config):
        """Test setting acceleration."""
        builder = MaskBuilder(config)
        builder.with_acceleration(4)
        assert builder._acceleration == 4

    def test_with_center_fraction(self, config):
        """Test setting center fraction."""
        builder = MaskBuilder(config)
        builder.with_center_fraction(0.08)
        assert builder._center_fraction == 0.08


class TestDataConsistencyBuilder:
    """Tests for DataConsistencyBuilder."""

    def test_initialization(self, config):
        """Test builder initialization."""
        builder = DataConsistencyBuilder(config)
        assert builder is not None

    def test_with_method(self, config):
        """Test setting DC method."""
        builder = DataConsistencyBuilder(config)
        builder.with_method("hard")
        assert builder._method == "hard"

    def test_invalid_method_raises(self, config):
        """Test invalid method raises error."""
        builder = DataConsistencyBuilder(config)
        with pytest.raises(ValueError):
            builder.with_method("invalid")

    def test_with_weight(self, config):
        """Test setting DC weight."""
        builder = DataConsistencyBuilder(config)
        builder.with_weight(0.5)
        assert builder._weight == 0.5


class TestPhysicsBuilder:
    """Tests for PhysicsBuilder."""

    def test_initialization(self, config):
        """Test builder initialization."""
        builder = PhysicsBuilder(config)
        assert builder is not None

    def test_fluent_api_chaining(self, config):
        """Test fluent API method chaining."""
        builder = PhysicsBuilder(config)
        result = builder.with_device("cuda")
        assert result is builder

    def test_with_fft_norm(self, config):
        """Test setting FFT norm."""
        builder = PhysicsBuilder(config)
        builder.with_fft_norm("ortho")
        assert builder._fft_norm == "ortho"

    def test_with_mask_acceleration(self, config):
        """Test setting mask acceleration."""
        builder = PhysicsBuilder(config)
        builder.with_mask_acceleration(4)
        assert builder._mask_acceleration == 4


# ============================================================================
# Integration Tests
# ============================================================================


class TestBuilderIntegration:
    """Integration tests for builder interactions."""

    def test_model_plus_optimizer(self, config):
        """Test building model + optimizer together."""
        try:
            model_builder = GeneratorBuilder(config)
            model = model_builder.with_architecture("standard_unet").build()

            opt_builder = OptimizerBuilder(config, params=model.parameters())
            optimizer = opt_builder.with_type("Adam").with_learning_rate(1e-3).build()

            assert isinstance(model, nn.Module)
            assert isinstance(optimizer, optim.Optimizer)
        except Exception:
            pytest.skip("Model registry not fully available")

    def test_dataloader_builder_integration(self, config):
        """Test DataLoader builder with dummy dataset."""

        # Create a dataset that returns dicts, which is what the image collator expects
        class DummyDictDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.data = torch.randn(100, 3, 32, 32)
                self.target = torch.randn(100, 3, 32, 32)

            def __len__(self):
                return 100

            def __getitem__(self, idx):
                return {"image": self.data[idx], "target": self.target[idx]}

        dataset = DummyDictDataset()
        config.data.collation.strategy = "image"
        builder = DataLoaderBuilder(config, dataset=dataset)
        loader = builder.with_batch_size(16).with_num_workers(0).build()

        assert len(loader) > 0
        batch = next(iter(loader))
        assert "image" in batch
        assert "target" in batch


class TestGeneratorBuilderMetadataKwargFilter:
    """Smoke audit 2026-06-03 — the kwarg sweep must not leak domain-inference
    metadata fields into the model constructor.

    ``model.{target_domain,model_domain,output_type,input_type}`` are consumed
    by ``infer_output_domain`` / data-model-compatibility, NOT by the model
    ``__init__``. The sweep forwards every ``model`` field not in
    ``_SKIP_MODEL_FIELDS`` to the factory; strict configs (e.g. ``UNetConfig``)
    raise on unexpected kwargs, so leaking these crashed
    ``eval_c5_exchangeability_test`` with ``Unexpected keyword argument
    'target_domain' for UNetConfig``. These four fields must stay filtered.

    Asserted against the live set and the live sweep rather than against
    ``inspect.getsource``: the previous source-text pin broke on a pure
    relocation of the set (behaviour unchanged), which is a false positive,
    and would have kept passing if the set were still named ``_SKIP`` while
    the sweep stopped consulting it, which is a false negative.
    """

    REQUIRED_SKIP_FIELDS = (
        "target_domain",
        "model_domain",
        "output_type",
        "input_type",
    )

    def test_skip_set_filters_domain_metadata_fields(self) -> None:
        from spectramr.infrastructure.builders.generator_kwargs import (
            _SKIP_MODEL_FIELDS,
        )

        for field in self.REQUIRED_SKIP_FIELDS:
            assert field in _SKIP_MODEL_FIELDS, (
                f"_SKIP_MODEL_FIELDS must filter the metadata field '{field}' "
                "before forwarding model kwargs to the factory (smoke audit "
                "2026-06-03 — eval_c5 UNetConfig kwarg leak)."
            )

    def test_sweep_does_not_forward_metadata_fields(self) -> None:
        """The behavioural half: the sweep actually drops them."""
        import pydantic

        from spectramr.infrastructure.builders.generator_kwargs import (
            apply_model_field_sweep,
        )

        class _ModelStub(pydantic.BaseModel):
            target_domain: str = "image"
            model_domain: str = "image"
            output_type: str = "magnitude"
            input_type: str = "magnitude"
            base_channels: int = 32

        class _ConfigStub(pydantic.BaseModel):
            model: _ModelStub = _ModelStub()

        resolved = apply_model_field_sweep({}, _ConfigStub())

        for field in self.REQUIRED_SKIP_FIELDS:
            assert field not in resolved.kwargs, (
                f"the sweep forwarded metadata field '{field}' to the "
                "constructor — this is the eval_c5 UNetConfig kwarg leak."
            )
        # Negative control: a genuine constructor field IS forwarded, so a
        # sweep that forwarded nothing at all could not pass this test.
        assert resolved.kwargs.get("base_channels") == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestCanonicalPathDoesNotTripTheDeprecation:
    """The leaf builders must build without muting any warning category.

    Both ``GeneratorBuilder.build`` and ``DiscriminatorBuilder.build`` used to
    open ``warnings.catch_warnings()`` and filter ``DeprecationWarning`` away
    just to import ``ModelFactory``, because the factory warned on *every*
    construction -- the canonical path included. Two problems with that: a
    blanket category filter hides any *other* deprecation raised inside the
    ``with`` block, and it meant the promote-to-error in ``pyproject.toml``
    could never gate this line.

    The warning now sits on ``ModelFactory.create_model``, which these builders
    do not call, so the suppression is gone. These tests fail if either comes
    back.
    """

    def test_generator_builder_emits_no_deprecation(self) -> None:
        import warnings

        from spectramr.config.settings import TrainingSettings

        config = TrainingSettings(
            model={"model_type": "standard_unet", "in_channels": 1, "out_channels": 1},
            data={"sampling": {"patch_size": [16, 16]}, "loader": {"batch_size": 1}},
            optimization={},
            logging={},
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = (
                GeneratorBuilder(config, device="cpu")
                .with_architecture("standard_unet")
                .with_input_channels(1)
                .with_output_channels(1)
                .validate()
                .build()
            )
        assert isinstance(model, nn.Module)
        deprecations = [
            str(w.message)
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "ModelFactory" in str(w.message)
        ]
        assert deprecations == [], (
            "the canonical builder tripped the ModelFactory deprecation: "
            f"{deprecations}. It must call the create_generator primitive, not "
            "create_model -- and it must not mute the category to hide this."
        )

    def test_neither_leaf_builder_suppresses_a_warning_category(self) -> None:
        """Structural companion: no blanket filter may return to these builds.

        Deliberately narrow -- it asserts only that the *suppression* is absent,
        which is a property of the source, while the test above asserts the
        behaviour that makes the suppression unnecessary. Neither alone is
        enough: the behavioural test would still pass if someone "fixed" a
        future regression by re-muting the category.
        """
        import inspect

        from spectramr.infrastructure.builders.leaf import model_builders

        source = inspect.getsource(model_builders)
        assert "catch_warnings" not in source, (
            "a warning suppression reappeared in the leaf model builders; if a "
            "DeprecationWarning fires on the canonical path again, move the "
            "warning off that path rather than muting it here"
        )
