from unittest.mock import MagicMock

import pytest

from spectramr.config.schemas.acceleration import AccelerationConfigSchema
from spectramr.data.builders.torchio_transform_builder import (
    TorchIOTransformBuilder,
    TorchIOTransformConfig,
)
from tests.utils.data_config_stub import DataConfigStub


def test_transform_builder_reads_root_acceleration():
    """Verify builder reads acceleration from root config (v5.0 schema)."""

    # `from_training_config` reads `config.sampling.patch_size` (a DATA-level
    # path) AND `config.undersampling` (a ROOT-level one) off the SAME object.
    # No real schema has both -- DataConfigSchema has `sampling` and not
    # `undersampling`; TrainingSettings the reverse -- so the only shape it
    # accepts is a data config with the root block attached. That is exactly the
    # `_Proxy` in test_resample_configurable, built here from the real schema.
    accel_config = AccelerationConfigSchema(
        base_acceleration=4.0, max_acceleration=8.0, center_fraction=0.08
    )
    # Phase 11 folded the top-level `acceleration:` block to `undersampling:`;
    # torchio_transform_builder.py:718 reads `config.undersampling`.
    # 2. Setup Data Config. These knobs are read off `data.<sub-block>.*`, not
    # off the root -- they were set flat on the mock, so the reader never saw
    # them and a bare MagicMock answered every one with a Mock. The old
    # "[FIX] set required scalars to avoid MagicMock comparison errors" lines
    # were patching that symptom at the root; routing through the stub removes
    # the cause. No `del` is needed either: after phase 11 `undersampling` lives
    # at the ROOT and is genuinely not a DataConfigSchema field.
    mock_config = DataConfigStub(
        patch_size=(320, 320, 1),
        coil_processing_mode="none",
        kspace_percentile=0.99,
        normalize_kspace=False,
        normalize_images=False,
        rescale_images=False,
        enable_graph_encoding=False,
    )
    mock_config.enable_geometric_standardization = True
    mock_config.undersampling = accel_config

    # Execute
    transform_config = TorchIOTransformConfig.from_training_config(mock_config)

    # Assert
    assert (
        transform_config.acceleration == 4.0
    ), "Should use base_acceleration from root config"
    assert transform_config.center_fraction == 0.08


def test_transform_builder_creates_masking_transform():
    """Verify validation pipeline includes masking when acceleration > 1."""

    config = TorchIOTransformConfig(
        acceleration=4.0, center_fraction=0.08, trajectory_type="cartesian"
    )

    transforms = TorchIOTransformBuilder.build_val_transforms(config)

    # Inspect transforms
    # We expect PhysicsInformedMasking
    from spectramr.data.transforms.tio_physics import PhysicsInformedMasking

    has_masking = any(
        isinstance(t, PhysicsInformedMasking) for t in transforms.transforms
    )
    assert (
        has_masking
    ), "Validation pipeline MUST include PhysicsInformedMasking when acceleration=4"


def test_transform_builder_no_acceleration_fallback():
    """Verify fallback to 1.0 if no root acceleration config."""

    mock_config = DataConfigStub(
        patch_size=(320, 320, 1),
        coil_processing_mode="none",
        kspace_percentile=0.99,
        normalize_kspace=False,
        normalize_images=False,
        rescale_images=False,
        enable_graph_encoding=False,
    )
    mock_config.enable_geometric_standardization = True
    # No root undersampling block (phase 11 renamed `acceleration:` to it).
    mock_config.undersampling = None

    # [FIX] Set trajectory_type to "cartesian" so it doesn't skip k-space generation
    mock_config.trajectory = None
    mock_config.physics = None

    # ``MagicMock.resample_enabled`` returns a truthy MagicMock by default,
    # which triggers ``build_resample_transform`` and then rejects
    # the MagicMock strategy value. Same for ``crop_or_pad_enabled``.
    # Disable both explicitly to keep the test focused on the
    # acceleration-fallback behaviour.
    mock_config.resample_enabled = False
    mock_config.resample = None
    mock_config.crop_or_pad_enabled = False
    mock_config.crop_or_pad = None

    transform_config = TorchIOTransformConfig.from_training_config(mock_config)

    assert transform_config.acceleration == 1.0

    # Set trajectory_type for the transform config
    transform_config.trajectory_type = "cartesian"

    # Verify pipeline uses KSpaceToInput (rename only)
    transforms = TorchIOTransformBuilder.build_val_transforms(transform_config)
    from spectramr.data.builders.torchio_transform_builder import _KSpaceToInputTransform

    has_rename = any(
        isinstance(t, _KSpaceToInputTransform) for t in transforms.transforms
    )
    assert has_rename, "Should use KSpaceToInputTransform when acceleration=1"


def test_transform_builder_injects_digital_twin_when_opted_in():
    """physics.digital_twin.apply_as_transform → DigitalTwinDegradation in pipeline."""
    from spectramr.config.schemas.physics import DigitalTwinConfig
    from spectramr.data.transforms.tio_physics import DigitalTwinDegradation

    config = TorchIOTransformConfig(
        patch_size=(32, 32, 1),
        dataset_type="kspace",
        digital_twin_apply=True,
        digital_twin_config=DigitalTwinConfig(motion_severity=2.0),
        digital_twin_degradation_only=True,
    )
    transforms = TorchIOTransformBuilder.build_val_transforms(config)
    assert any(
        isinstance(t, DigitalTwinDegradation) for t in transforms.transforms
    ), "Pipeline MUST include DigitalTwinDegradation when apply_as_transform is set"


def test_transform_route_builds_the_acquisition_profile_at_the_config_field():
    from spectramr.config.schemas.physics import DigitalTwinConfig
    from spectramr.data.transforms.tio_physics import DigitalTwinDegradation

    config = TorchIOTransformConfig(
        patch_size=(32, 32, 1),
        dataset_type="kspace",
        digital_twin_apply=True,
        digital_twin_config=DigitalTwinConfig(acquisition={"scanner_profile": "m4raw_0.3T"}),
        digital_twin_field_strength_t=0.3,
    )
    twin = next(
        t
        for t in TorchIOTransformBuilder.build_val_transforms(config).transforms
        if isinstance(t, DigitalTwinDegradation)
    )
    profile = twin.simulator.scanner_profile
    assert (profile.name, profile.b0_T) == ("m4raw_0.3T", 0.3)
    record = twin.applied_record()
    assert record["scanner_profile"] == "m4raw_0.3T"
    assert record["scanner_profile_sha256"] == profile.meta_sha256()


@pytest.mark.parametrize(
    ("declared", "pe_axis", "source"),
    [(-1, "H", "data.acquisition.phase_encode_axis"), (None, "W", "undeclared")],
)
def test_the_twin_reads_the_dataset_phase_encode_axis(declared, pe_axis, source):
    from spectramr.config.schemas.physics import DigitalTwinConfig
    from spectramr.data.transforms.tio_physics import DigitalTwinDegradation

    config = TorchIOTransformConfig(
        patch_size=(32, 32, 1),
        dataset_type="kspace",
        digital_twin_apply=True,
        digital_twin_config=DigitalTwinConfig(),
        phase_encode_axis=declared,
    )
    (twin,) = (
        t
        for t in TorchIOTransformBuilder.build_val_transforms(config).transforms
        if isinstance(t, DigitalTwinDegradation)
    )
    record = twin.applied_record()
    assert (record["pe_axis"], record["pe_axis_source"]) == (pe_axis, source)


def test_from_training_config_carries_physics_field_strength_to_the_twin():
    from spectramr.config.schemas.physics import PhysicsConfigSchema

    data = DataConfigStub(patch_size=(32, 32, 1), coil_processing_mode="none")
    data.undersampling = AccelerationConfigSchema(base_acceleration=1.0)
    data.physics = PhysicsConfigSchema(
        field_strength=0.3,
        digital_twin={"apply_as_transform": True, "acquisition": {"scanner_profile": "m4raw_0.3T"}},
    )
    config = TorchIOTransformConfig.from_training_config(data)
    assert config.digital_twin_field_strength_t == 0.3


def test_transform_builder_no_digital_twin_by_default():
    from spectramr.data.transforms.tio_physics import DigitalTwinDegradation

    config = TorchIOTransformConfig(patch_size=(32, 32, 1), dataset_type="kspace")
    transforms = TorchIOTransformBuilder.build_val_transforms(config)
    assert not any(
        isinstance(t, DigitalTwinDegradation) for t in transforms.transforms
    ), "Digital twin must be off by default (no double-corruption for VF arms)"
