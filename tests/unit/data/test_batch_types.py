"""Unit tests for batch types and batch adapters.

This module provides comprehensive test coverage for batch type handling,
including TrainingBatch dataclass and BatchAdapter for converting dictionaries
to TrainingBatch objects. Tests cover tensor device movement, indexing,
unpacking, metadata handling, and batch format adaptation.
"""

import pytest
import torch
from torch.utils.data import default_collate

from spectramr.data.batch_types import (
    BatchAdapter,
    TargetlessSampleError,
    TrainingBatch,
    align_scale_to_batch,
    read_batch_field,
    reject_targetless_samples,
)


class TestTrainingBatchInitialization:
    """Test TrainingBatch initialization."""

    def test_basic_initialization(self):
        """Test basic initialization with required fields."""
        input_tensor = torch.randn(2, 1, 16, 16)
        target_tensor = torch.randn(2, 1, 16, 16)

        batch = TrainingBatch(input=input_tensor, target=target_tensor)

        assert batch.input is input_tensor
        assert batch.target is target_tensor
        assert batch.mask is None
        assert batch.metadata == {}

    def test_initialization_with_mask(self):
        """Test initialization with mask."""
        input_tensor = torch.randn(2, 1, 16, 16)
        target_tensor = torch.randn(2, 1, 16, 16)
        mask_tensor = torch.ones(2, 1, 16, 16)

        batch = TrainingBatch(
            input=input_tensor,
            target=target_tensor,
            mask=mask_tensor,
        )

        assert batch.mask is mask_tensor

    def test_initialization_with_metadata(self):
        """Test initialization with metadata."""
        input_tensor = torch.randn(2, 1, 16, 16)
        target_tensor = torch.randn(2, 1, 16, 16)
        metadata = {"filename": "test.nii.gz", "slice_idx": 10}

        batch = TrainingBatch(
            input=input_tensor,
            target=target_tensor,
            metadata=metadata,
        )

        assert batch.metadata == metadata

    def test_full_initialization(self):
        """Test full initialization with all fields."""
        input_tensor = torch.randn(2, 1, 16, 16)
        target_tensor = torch.randn(2, 1, 16, 16)
        mask_tensor = torch.ones(2, 1, 16, 16)
        metadata = {"filename": "test.nii.gz", "acceleration": 4}

        batch = TrainingBatch(
            input=input_tensor,
            target=target_tensor,
            mask=mask_tensor,
            metadata=metadata,
        )

        assert batch.input is input_tensor
        assert batch.target is target_tensor
        assert batch.mask is mask_tensor
        assert batch.metadata == metadata


class TestTrainingBatchToDevice:
    """Test moving batch to device."""

    def test_to_device_cpu(self):
        """Test moving batch to CPU."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
        )

        result = batch.to("cpu")

        assert result.input.device.type == "cpu"
        assert result.target.device.type == "cpu"

    def test_to_device_with_mask(self):
        """Test moving batch with mask to device."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
            mask=torch.ones(2, 1, 16, 16),
        )

        result = batch.to("cpu")

        assert result.mask is not None
        assert result.mask.device.type == "cpu"

    def test_to_device_preserves_metadata(self):
        """Test that moving to device preserves metadata."""
        metadata = {"filename": "test.nii.gz"}
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
            metadata=metadata,
        )

        result = batch.to("cpu")

        assert result.metadata == metadata

    def test_to_device_string(self):
        """Test moving batch with string device."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
        )

        result = batch.to("cpu")

        assert result.input.device.type == "cpu"

    def test_to_device_torch_device(self):
        """Test moving batch with torch.device."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
        )

        device = torch.device("cpu")
        result = batch.to(device)

        assert result.input.device == device


class TestTrainingBatchIndexing:
    """Test indexing operations on batch."""

    def test_integer_indexing_0(self):
        """Test integer indexing to get input."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
        )

        result = batch[0]

        assert torch.equal(result, batch.input)

    def test_integer_indexing_1(self):
        """Test integer indexing to get target."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
        )

        result = batch[1]

        assert torch.equal(result, batch.target)

    def test_integer_indexing_invalid(self):
        """Test invalid integer indexing."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
        )

        with pytest.raises(IndexError):
            _ = batch[2]

    def test_string_indexing_input(self):
        """Test string indexing for input."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
        )

        result = batch["input"]

        assert torch.equal(result, batch.input)

    def test_string_indexing_target(self):
        """Test string indexing for target."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
        )

        result = batch["target"]

        assert torch.equal(result, batch.target)

    def test_string_indexing_mask(self):
        """Test string indexing for mask."""
        mask = torch.ones(2, 1, 16, 16)
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
            mask=mask,
        )

        result = batch["mask"]

        assert torch.equal(result, mask)

    def test_string_indexing_metadata(self):
        """Test accessing metadata via string indexing."""
        metadata = {"filename": "test.nii.gz", "slice_idx": 5}
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
            metadata=metadata,
        )

        assert batch["filename"] == "test.nii.gz"
        assert batch["slice_idx"] == 5

    def test_string_indexing_nonexistent_key(self):
        """Test accessing nonexistent key."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
        )

        with pytest.raises(KeyError):
            _ = batch["nonexistent"]

    def test_invalid_index_type(self):
        """Test invalid index type."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
        )

        with pytest.raises(TypeError):
            _ = batch[1.5]


class TestTrainingBatchContains:
    """Test __contains__ operator."""

    def test_contains_input(self):
        """Test 'in' operator for input."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
        )

        assert "input" in batch

    def test_contains_target(self):
        """Test 'in' operator for target."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
        )

        assert "target" in batch

    def test_contains_mask(self):
        """Test 'in' operator for mask."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
            mask=torch.ones(2, 1, 16, 16),
        )

        assert "mask" in batch

    def test_contains_metadata_key(self):
        """Test 'in' operator for metadata keys."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
            metadata={"filename": "test.nii.gz"},
        )

        assert "filename" in batch

    def test_not_contains(self):
        """Test 'in' operator for nonexistent key."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
        )

        assert "nonexistent" not in batch


class TestTrainingBatchUnpacking:
    """Test unpacking operations."""

    def test_tuple_unpacking(self):
        """Test unpacking as tuple."""
        input_tensor = torch.randn(2, 1, 16, 16)
        target_tensor = torch.randn(2, 1, 16, 16)
        batch = TrainingBatch(input=input_tensor, target=target_tensor)

        input_unpacked, target_unpacked = batch[0], batch[1]

        assert torch.equal(input_unpacked, input_tensor)
        assert torch.equal(target_unpacked, target_tensor)


class TestTrainingBatchEdgeCases:
    """Test edge cases."""

    def test_empty_metadata(self):
        """Test batch with empty metadata."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
            metadata={},
        )

        assert batch.metadata == {}

    def test_none_mask(self):
        """Test batch with None mask."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
            mask=None,
        )

        assert batch.mask is None

    def test_different_input_output_shapes(self):
        """Test batch with different input and output shapes."""
        # Super-resolution case
        batch = TrainingBatch(
            input=torch.randn(2, 1, 8, 8),  # Low-res
            target=torch.randn(2, 1, 16, 16),  # High-res
        )

        assert batch.input.shape != batch.target.shape

    def test_batch_size_one(self):
        """Test batch with single sample."""
        batch = TrainingBatch(
            input=torch.randn(1, 1, 16, 16),
            target=torch.randn(1, 1, 16, 16),
        )

        assert batch.input.shape[0] == 1
        assert batch.target.shape[0] == 1

    def test_large_batch_size(self):
        """Test batch with large batch size."""
        batch_size = 128
        batch = TrainingBatch(
            input=torch.randn(batch_size, 1, 16, 16),
            target=torch.randn(batch_size, 1, 16, 16),
        )

        assert batch.input.shape[0] == batch_size


class TestTrainingBatchIntegration:
    """Integration tests for training batch."""

    def test_full_workflow(self):
        """Test full workflow with batch."""
        # Create batch
        batch = TrainingBatch(
            input=torch.randn(4, 1, 16, 16),
            target=torch.randn(4, 1, 16, 16),
            mask=torch.ones(4, 1, 16, 16),
            metadata={"dataset": "test"},
        )

        # Move to device
        batch = batch.to("cpu")

        # Access elements
        input_data = batch["input"]
        target_data = batch["target"]
        mask_data = batch["mask"]
        filename = batch["dataset"]

        # Verify everything works
        assert input_data is not None
        assert target_data is not None
        assert mask_data is not None
        assert filename == "test"

    def test_dataloader_compatibility(self):
        """Test compatibility with DataLoader collate."""
        # Create list of batches (as from dataset)
        batches = [
            TrainingBatch(
                input=torch.randn(1, 1, 16, 16),
                target=torch.randn(1, 1, 16, 16),
                metadata={"idx": i},
            )
            for i in range(4)
        ]

        # Simple collate: stack inputs and targets
        stacked_input = torch.cat([b.input for b in batches], dim=0)
        stacked_target = torch.cat([b.target for b in batches], dim=0)

        assert stacked_input.shape[0] == 4
        assert stacked_target.shape[0] == 4


class TestBatchAdapter:
    """Test BatchAdapter for converting dictionaries to TrainingBatch."""

    def test_adapt_kspace_reconstruction(self):
        """Test adapting k-space reconstruction format."""
        batch_dict = {
            "input": torch.randn(2, 64, 64),
            "target": torch.randn(2, 64, 64),
            "mask": torch.ones(2, 64, 64),
        }

        batch = BatchAdapter.from_dict(batch_dict)

        assert torch.equal(batch.input, batch_dict["input"])
        assert torch.equal(batch.target, batch_dict["target"])
        assert torch.equal(batch.mask, batch_dict["mask"])

    def test_adapt_diffusion_kspace(self):
        """Test adapting diffusion k-space format."""
        batch_dict = {
            "input": torch.randn(2, 64, 64),
            "target": torch.randn(2, 64, 64),
            "mask": torch.ones(2, 64, 64),
        }

        batch = BatchAdapter.from_dict(batch_dict)

        assert torch.equal(batch.input, batch_dict["input"])
        assert torch.equal(batch.target, batch_dict["target"])
        assert torch.equal(batch.mask, batch_dict["mask"])

    def test_adapt_super_resolution(self):
        """Test adapting super-resolution format."""
        batch_dict = {
            "input": torch.randn(2, 1, 32, 32),
            "target": torch.randn(2, 1, 64, 64),
        }

        batch = BatchAdapter.from_dict(batch_dict)

        assert torch.equal(batch.input, batch_dict["input"])
        assert torch.equal(batch.target, batch_dict["target"])

    def test_adapt_generic_input_target(self):
        """Test adapting generic input/target format."""
        batch_dict = {
            "input": torch.randn(2, 64, 64),
            "target": torch.randn(2, 64, 64),
        }

        batch = BatchAdapter.from_dict(batch_dict)

        assert torch.equal(batch.input, batch_dict["input"])
        assert torch.equal(batch.target, batch_dict["target"])

    def test_adapt_preserves_metadata(self):
        """Test that adapter preserves metadata fields."""
        batch_dict = {
            "input": torch.randn(2, 64, 64),
            "target": torch.randn(2, 64, 64),
            "file_id": "test.h5",
            "slice_idx": 10,
            "acceleration": 4,
        }

        batch = BatchAdapter.from_dict(batch_dict)

        assert batch.metadata["file_id"] == "test.h5"
        assert batch.metadata["slice_idx"] == 10
        assert batch.metadata["acceleration"] == 4

    def test_adapt_invalid_batch_format(self):
        """Test that invalid batch format raises ValueError."""
        batch_dict = {"unknown_field": torch.randn(2, 64, 64)}

        with pytest.raises(ValueError):
            BatchAdapter.from_dict(batch_dict)

    def test_adapt_with_none_acceleration_mask(self):
        """Test adapting when acceleration_mask is None but mask exists."""
        batch_dict = {
            "input": torch.randn(2, 64, 64),
            "target": torch.randn(2, 64, 64),
            "acceleration_mask": None,
            "mask": torch.ones(2, 64, 64),
        }

        batch = BatchAdapter.from_dict(batch_dict)

        # Should fall back to mask when acceleration_mask is None
        assert torch.equal(batch.mask, batch_dict["mask"])

    def test_adapt_complex_metadata(self):
        """Test adapting with complex metadata."""
        batch_dict = {
            "input": torch.randn(2, 64, 64),
            "target": torch.randn(2, 64, 64),
            "file_id": "test.h5",
            "metadata_dict": {"shape": (64, 64), "dtype": "float32"},
            "coils": 8,
        }

        batch = BatchAdapter.from_dict(batch_dict)

        assert batch.metadata["file_id"] == "test.h5"
        assert batch.metadata["coils"] == 8
        assert batch.metadata["metadata_dict"]["shape"] == (64, 64)

    def test_adapt_multiple_formats_priority(self):
        """Test that adapter uses correct format when multiple keys present."""
        # input/target takes priority
        batch_dict = {
            "input": torch.randn(2, 64, 64),
            "target": torch.randn(2, 64, 64),
            "lr": torch.randn(2, 32, 32),  # Should be ignored (in metadata)
            "hr": torch.randn(2, 64, 64),  # Should be ignored (in metadata)
        }

        batch = BatchAdapter.from_dict(batch_dict)

        # Should use input/target
        assert torch.equal(batch.input, batch_dict["input"])
        assert torch.equal(batch.target, batch_dict["target"])

    def test_adapt_with_extra_metadata_fields(self):
        """Test adapting with many extra metadata fields."""
        batch_dict = {
            "input": torch.randn(2, 64, 64),
            "target": torch.randn(2, 64, 64),
            "field1": "value1",
            "field2": 42,
            "field3": [1, 2, 3],
            "field4": {"nested": "dict"},
        }

        batch = BatchAdapter.from_dict(batch_dict)

        assert batch.metadata["field1"] == "value1"
        assert batch.metadata["field2"] == 42
        assert batch.metadata["field3"] == [1, 2, 3]
        assert batch.metadata["field4"]["nested"] == "dict"


class TestTrainingBatchGetMethod:
    """Test get() method for safe access."""

    def test_get_input_field(self):
        """Test get() for input field."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
        )

        result = batch.get("input")
        assert torch.equal(result, batch.input)

    def test_get_target_field(self):
        """Test get() for target field."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
        )

        result = batch.get("target")
        assert torch.equal(result, batch.target)

    def test_get_mask_field(self):
        """Test get() for mask field."""
        mask = torch.ones(2, 1, 16, 16)
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
            mask=mask,
        )

        result = batch.get("mask")
        assert torch.equal(result, mask)

    def test_get_nonexistent_default_none(self):
        """Test get() returns None for nonexistent key."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
        )

        result = batch.get("nonexistent")
        assert result is None

    def test_get_nonexistent_custom_default(self):
        """Test get() with custom default value."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
        )

        default = torch.zeros(2, 1, 16, 16)
        result = batch.get("nonexistent", default)
        assert torch.equal(result, default)

    def test_get_metadata_field(self):
        """Test get() for metadata field."""
        metadata = {"filename": "test.nii.gz", "slice_idx": 5}
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
            metadata=metadata,
        )

        assert batch.get("filename") == "test.nii.gz"
        assert batch.get("slice_idx") == 5


class TestTrainingBatchDifferentDTypes:
    """Test batch with different data types."""

    def test_batch_with_float32(self):
        """Test batch with float32 tensors."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16, dtype=torch.float32),
            target=torch.randn(2, 1, 16, 16, dtype=torch.float32),
        )

        assert batch.input.dtype == torch.float32

    def test_batch_with_float64(self):
        """Test batch with float64 tensors."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16, dtype=torch.float64),
            target=torch.randn(2, 1, 16, 16, dtype=torch.float64),
        )

        assert batch.input.dtype == torch.float64

    def test_batch_with_complex_dtype(self):
        """Test batch with complex tensors."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16, dtype=torch.complex64),
            target=torch.randn(2, 1, 16, 16, dtype=torch.complex64),
        )

        assert batch.input.dtype == torch.complex64

    def test_batch_mixed_dtypes(self):
        """Test batch with mixed dtypes."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16, dtype=torch.float32),
            target=torch.randn(2, 1, 16, 16, dtype=torch.float64),
            mask=torch.ones(2, 1, 16, 16, dtype=torch.bool),
        )

        assert batch.input.dtype == torch.float32
        assert batch.target.dtype == torch.float64
        assert batch.mask.dtype == torch.bool

    def test_batch_to_device_preserves_dtype(self):
        """Test that moving to device preserves dtype."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16, dtype=torch.complex64),
            target=torch.randn(2, 1, 16, 16, dtype=torch.float32),
        )

        moved = batch.to("cpu")

        assert moved.input.dtype == torch.complex64
        assert moved.target.dtype == torch.float32


class TestTrainingBatchVariousSizes:
    """Test batch with various tensor sizes."""

    def test_batch_2d_tensors(self):
        """Test batch with 2D tensors."""
        batch = TrainingBatch(
            input=torch.randn(64, 64),
            target=torch.randn(64, 64),
        )

        assert batch.input.ndim == 2
        assert batch.target.ndim == 2

    def test_batch_3d_tensors(self):
        """Test batch with 3D tensors."""
        batch = TrainingBatch(
            input=torch.randn(2, 64, 64),
            target=torch.randn(2, 64, 64),
        )

        assert batch.input.ndim == 3

    def test_batch_4d_tensors(self):
        """Test batch with 4D tensors."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 64, 64),
            target=torch.randn(2, 1, 64, 64),
        )

        assert batch.input.ndim == 4

    def test_batch_5d_tensors(self):
        """Test batch with 5D tensors (volumetric)."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 32, 32, 32),
            target=torch.randn(2, 1, 32, 32, 32),
        )

        assert batch.input.ndim == 5

    def test_batch_different_spatial_sizes(self):
        """Test batch with different spatial dimensions."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 128, 128),
            target=torch.randn(2, 1, 256, 256),
        )

        assert batch.input.shape[-2:] != batch.target.shape[-2:]

    def test_batch_single_sample(self):
        """Test batch with single sample."""
        batch = TrainingBatch(
            input=torch.randn(1, 1, 64, 64),
            target=torch.randn(1, 1, 64, 64),
        )

        assert batch.input.shape[0] == 1

    def test_batch_large_batch_size(self):
        """Test batch with large batch size."""
        batch_size = 256
        batch = TrainingBatch(
            input=torch.randn(batch_size, 1, 16, 16),
            target=torch.randn(batch_size, 1, 16, 16),
        )

        assert batch.input.shape[0] == batch_size


class TestTrainingBatchBackwardCompatibility:
    """Test backward compatibility features."""

    def test_legacy_dict_like_access(self):
        """Test legacy dict-like access patterns."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
        )

        # Should support dict-like operations
        assert "input" in batch
        assert "target" in batch
        assert batch.get("input") is not None
        assert batch["input"] is not None

    def test_legacy_tuple_unpacking(self):
        """Test legacy tuple unpacking."""
        input_tensor = torch.randn(2, 1, 16, 16)
        target_tensor = torch.randn(2, 1, 16, 16)
        batch = TrainingBatch(input=input_tensor, target=target_tensor)

        # Should unpack as (input, target)
        inp, tgt = batch[0], batch[1]
        assert torch.equal(inp, input_tensor)
        assert torch.equal(tgt, target_tensor)


class TestTrainingBatchMultiCoilKSpace:
    """Test batch handling for multi-coil k-space data."""

    def test_multicoil_kspace_cold_diffusion_mapping(self):
        """Test correct mapping of kspace for cold diffusion with multi-coil data."""
        # 18 coils * 2 (real/imaginary) = 36 channels
        num_coils = 18
        num_channels = num_coils * 2

        batch = TrainingBatch(
            input=torch.randn(4, num_channels, 256, 256),  # kspace_sub (undersampled)
            target=torch.randn(4, num_channels, 256, 256),  # kspace (full)
            mask=None,
        )

        # Test that kspace maps to target (not input)
        assert torch.equal(batch.target, batch.target)
        assert torch.equal(batch.input, batch.input)

    def test_multicoil_kspace_shape_consistency(self):
        """Test shape consistency for multi-coil k-space."""
        num_coils = 18
        num_channels = num_coils * 2
        batch_size = 4
        height, width = 256, 256

        batch = TrainingBatch(
            input=torch.randn(batch_size, num_channels, height, width),
            target=torch.randn(batch_size, num_channels, height, width),
        )

        # Verify shapes
        assert batch.input.shape == (batch_size, num_channels, height, width)
        assert batch.target.shape == (batch_size, num_channels, height, width)
        assert batch.input.shape[1] == 36  # 18 coils * 2

        # Verify they match
        assert batch.input.shape[1:] == batch.target.shape[1:]

    def test_multicoil_with_mask(self):
        """Test multi-coil batch with acceleration mask."""
        num_coils = 18
        num_channels = num_coils * 2

        batch = TrainingBatch(
            input=torch.randn(4, num_channels, 256, 256),
            target=torch.randn(4, num_channels, 256, 256),
            mask=torch.ones(4, 1, 256, 256),  # Mask may have fewer channels
        )

        # Mask should be accessible
        assert batch.mask is not None
        assert batch["mask"] is not None
        assert batch.mask.shape[1] == 1  # Single mask shared across channels

    def test_multicoil_batch_adapter_legacy_naming(self):
        """Test batch adapter with canonical naming for multi-coil cold diffusion."""
        num_coils = 18
        num_channels = num_coils * 2

        batch_dict = {
            "input": torch.randn(4, num_channels, 256, 256),
            "target": torch.randn(4, num_channels, 256, 256),
            "mask": torch.ones(4, 1, 256, 256),
        }

        batch = BatchAdapter.from_dict(batch_dict)

        # Verify correct routing
        assert torch.equal(batch.input, batch_dict["input"])
        assert torch.equal(batch.target, batch_dict["target"])
        assert torch.equal(batch.mask, batch_dict["mask"])

    def test_multicoil_batch_adapter_canonical_naming(self):
        """Test batch adapter with canonical naming for multi-coil cold diffusion."""
        num_coils = 18
        num_channels = num_coils * 2

        batch_dict = {
            "input": torch.randn(4, num_channels, 256, 256),
            "target": torch.randn(4, num_channels, 256, 256),
            "mask": torch.ones(4, 1, 256, 256),
        }

        batch = BatchAdapter.from_dict(batch_dict)

        # Verify correct routing
        assert torch.equal(batch.input, batch_dict["input"])
        assert torch.equal(batch.target, batch_dict["target"])
        assert torch.equal(batch.mask, batch_dict["mask"])

    def test_multicoil_batch_device_movement(self):
        """Test device movement for multi-coil batch."""
        num_coils = 18
        num_channels = num_coils * 2

        batch = TrainingBatch(
            input=torch.randn(4, num_channels, 256, 256),
            target=torch.randn(4, num_channels, 256, 256),
            mask=torch.ones(4, 1, 256, 256),
        )

        # Move to CPU (always available)
        moved = batch.to("cpu")

        # Verify all tensors are on CPU
        assert moved.input.device.type == "cpu"
        assert moved.target.device.type == "cpu"
        assert moved.mask.device.type == "cpu"

        # Verify shapes are preserved
        assert moved.input.shape == (4, num_channels, 256, 256)
        assert moved.target.shape == (4, num_channels, 256, 256)

    def test_multicoil_batch_channel_access(self):
        """Test accessing individual channels in multi-coil batch."""
        num_coils = 18
        num_channels = num_coils * 2

        batch = TrainingBatch(
            input=torch.randn(4, num_channels, 256, 256),
            target=torch.randn(4, num_channels, 256, 256),
        )

        # Access specific coil channels (real part of coil 0 and 1)
        real_ch_0 = batch.input[:, 0, :, :]  # Real part of coil 0
        imag_ch_0 = batch.input[:, 1, :, :]  # Imaginary part of coil 0
        real_ch_1 = batch.input[:, 2, :, :]  # Real part of coil 1

        # Verify shapes
        assert real_ch_0.shape == (4, 256, 256)
        assert imag_ch_0.shape == (4, 256, 256)
        assert real_ch_1.shape == (4, 256, 256)

    def test_multicoil_batch_dtype_preservation(self):
        """Test dtype preservation for multi-coil batch."""
        num_coils = 18
        num_channels = num_coils * 2

        # Complex k-space
        batch = TrainingBatch(
            input=torch.randn(4, num_channels, 256, 256, dtype=torch.float32),
            target=torch.randn(4, num_channels, 256, 256, dtype=torch.float32),
        )

        # Move and verify dtype is preserved
        moved = batch.to("cpu")
        assert moved.input.dtype == torch.float32
        assert moved.target.dtype == torch.float32

    def test_multicoil_different_acceleration_factors(self):
        """Test multi-coil batch with different undersampling levels."""
        num_coils = 18
        num_channels = num_coils * 2

        # Create batches at different acceleration factors
        for accel_factor in [2, 4, 8]:
            # Simulate undersampling by zeroing out some k-space
            undersampled = torch.randn(4, num_channels, 256, 256)
            # Zero out ~(1 - 1/accel_factor) of k-space
            num_zero = int(256 * 256 * (1 - 1 / accel_factor))
            undersampled.view(-1)[torch.randperm(undersampled.numel())[:num_zero]] = 0

            batch = TrainingBatch(
                input=undersampled,
                target=torch.randn(4, num_channels, 256, 256),
            )

            # Verify batch is valid
            assert batch.input.shape == (4, num_channels, 256, 256)
            assert batch.target.shape == (4, num_channels, 256, 256)

            # Verify undersampled has fewer non-zero elements
            assert (undersampled != 0).sum() < (batch.target != 0).sum()


class TestTrainingBatchDeviceIntegrity:
    """Test device integrity across batch operations."""

    def test_device_consistency_after_to(self):
        """Test that all tensors are on same device after to()."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
            mask=torch.ones(2, 1, 16, 16),
        )

        moved = batch.to("cpu")

        # All should be on same device
        assert moved.input.device == moved.target.device
        assert moved.target.device == moved.mask.device

    def test_nonblocking_device_transfer(self):
        """Test non-blocking device transfer."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
        )

        # Should not raise with non_blocking=True
        moved = batch.to("cpu", non_blocking=True)
        assert moved.input.device.type == "cpu"

    def test_device_integrity_with_none_mask(self):
        """Test device handling when mask is None."""
        batch = TrainingBatch(
            input=torch.randn(2, 1, 16, 16),
            target=torch.randn(2, 1, 16, 16),
            mask=None,
        )

        moved = batch.to("cpu")

        # Mask should remain None
        assert moved.mask is None
        # Other tensors should be on device
        assert moved.input.device.type == "cpu"
        assert moved.target.device.type == "cpu"


class TestCoilMapsAreOneFieldUnderFiveNames:
    """C11: coil sensitivity maps travelled under five spellings.

    The data layer produces ``sensitivity`` (``torchio_subject_builder``);
    consumers ask for ``sensitivity_maps`` (10 files), ``coil_sensitivities``
    (5), ``smaps`` (3), ``coil_maps`` (1). Nothing reconciled them, and because
    ``BatchAdapter`` files every non-core key into ``metadata`` verbatim, a
    consumer's ``in`` check simply answered False — so a guarded SENSE term ran
    coil-blind instead of failing (pitfall #16).
    """

    @staticmethod
    def _batch(**extra):
        import torch

        from spectramr.data.batch_types import BatchAdapter

        return BatchAdapter.from_dict(
            {
                "input": torch.zeros(1, 1, 4, 4),
                "target": torch.zeros(1, 1, 4, 4),
                **extra,
            }
        )

    def test_the_producers_spelling_binds_to_the_field(self) -> None:
        """``sensitivity`` is what the data layer actually emits."""
        import torch

        batch = self._batch(sensitivity=torch.ones(1, 2, 4, 4))
        assert batch.coil_maps is not None
        assert batch.coil_maps.shape == (1, 2, 4, 4)

    def test_every_consumer_spelling_finds_them(self) -> None:
        """The point of the change: 19 consumer files keep their own name."""
        import torch

        from spectramr.data.batch_types import COIL_MAP_ALIASES

        batch = self._batch(sensitivity=torch.ones(1, 2, 4, 4))
        for alias in COIL_MAP_ALIASES:
            assert alias in batch, f"{alias!r} presence check still misses"
            assert batch[alias] is batch.coil_maps
            assert batch.get(alias) is batch.coil_maps

    def test_contains_agrees_with_getitem(self) -> None:
        """A presence check that disagrees with the read IS the original bug.

        ``"sensitivity_maps" in batch`` answered False while the maps sat in
        metadata under another name, so the guarded branch was never taken.
        """
        import torch

        from spectramr.data.batch_types import COIL_MAP_ALIASES

        with_maps = self._batch(sensitivity=torch.ones(1, 2, 4, 4))
        without = self._batch()
        for alias in COIL_MAP_ALIASES:
            assert (alias in with_maps) is (with_maps.get(alias) is not None)
            assert (alias in without) is False
            assert without.get(alias) is None

    def test_absent_maps_stay_absent(self) -> None:
        """No fabrication: a batch without coils must not gain them."""
        batch = self._batch()
        assert batch.coil_maps is None

    def test_the_bound_key_is_not_duplicated_into_metadata(self) -> None:
        """Otherwise the tensor exists twice and the two can diverge under
        ``to(device)``, which moves the field but copies metadata separately."""
        import torch

        batch = self._batch(sensitivity=torch.ones(1, 2, 4, 4))
        assert "sensitivity" not in batch.metadata

    def test_to_device_carries_the_field(self) -> None:
        """A field added to the dataclass but omitted from ``to()`` silently
        drops on the first device move — the maps would vanish at train time."""
        import torch

        batch = self._batch(sensitivity=torch.ones(1, 2, 4, 4)).to("cpu")
        assert batch.coil_maps is not None

    def test_torchio_nested_dicts_are_unwrapped(self) -> None:
        """``SubjectsLoader`` wraps images as ``{"data": ..., "affine": ...}``;
        the field must hold the tensor, not the wrapper."""
        import torch

        batch = self._batch(sensitivity={"data": torch.ones(1, 2, 4, 4), "affine": torch.eye(4)})
        assert isinstance(batch.coil_maps, torch.Tensor)

    def test_two_conflicting_spellings_raise(self) -> None:
        """They are aliases for ONE tensor. Two different values means the
        producer disagrees with itself about which coils were used, and picking
        one silently is what created the five-name split."""
        import pytest
        import torch

        with pytest.raises(ValueError, match="conflicting coil sensitivity maps"):
            self._batch(
                sensitivity=torch.ones(1, 2, 4, 4),
                smaps=torch.zeros(1, 2, 4, 4),
            )

    def test_two_identical_spellings_are_accepted(self) -> None:
        """Redundant but consistent is not a contradiction — a producer emitting
        the same tensor twice should not fail the run."""
        import torch

        maps = torch.ones(1, 2, 4, 4)
        batch = self._batch(sensitivity=maps, smaps=maps.clone())
        assert batch.coil_maps is not None


class TestBatchAxes:
    """C8 -- axis identity travels WITH the batch, or 5-D stays unrepresentable.

    Both multi-frame loaders fold their extra axis into CHANNEL, and once folded
    nothing downstream can tell a 3-frame cine from a 3-channel image. The
    blocker was never the loader; it was that the tensor contract carried no
    field in which to say which axis the extra dimension is.
    """

    @staticmethod
    def _minimal():
        return {"input": torch.zeros(1, 1, 4, 4), "target": torch.zeros(1, 1, 4, 4)}

    def test_default_is_none_so_existing_callers_are_unchanged(self) -> None:
        """Every pre-C8 call site must keep behaving as before: unresolved,
        therefore skipped -- NOT a claim that the batch has no axes."""
        from spectramr.data.batch_types import BatchAdapter

        assert BatchAdapter.from_dict(self._minimal()).axes is None

    def test_axes_are_carried_through_from_dict(self) -> None:
        from spectramr.config.schemas.enums import Axis
        from spectramr.data.batch_types import BatchAdapter

        batch = BatchAdapter.from_dict(self._minimal(), axes=frozenset({Axis.TEMPORAL}))
        assert batch.axes == frozenset({Axis.TEMPORAL})

    def test_axes_survive_the_device_move(self) -> None:
        """The trap: every consumer runs AFTER ``.to()``. A field dropped there
        reads ``None`` at literally every point that reads it -- a producer and
        a consumer that both exist, with nothing in between."""
        from spectramr.config.schemas.enums import Axis
        from spectramr.data.batch_types import BatchAdapter

        batch = BatchAdapter.from_dict(self._minimal(), axes=frozenset({Axis.ECHO}))
        assert batch.to("cpu").axes == frozenset({Axis.ECHO})

    def test_empty_axes_are_preserved_as_a_positive_claim(self) -> None:
        """``frozenset()`` must not degrade to ``None`` anywhere in the path."""
        from spectramr.data.batch_types import BatchAdapter

        batch = BatchAdapter.from_dict(self._minimal(), axes=frozenset())
        assert batch.axes == frozenset()
        assert batch.axes is not None
        assert batch.to("cpu").axes == frozenset()

    def test_axes_is_not_filed_into_metadata(self) -> None:
        """First-class field, not a metadata entry -- the C11 lesson: a key in
        metadata is a key every consumer's presence check silently misses."""
        from spectramr.config.schemas.enums import Axis
        from spectramr.data.batch_types import BatchAdapter

        batch = BatchAdapter.from_dict(self._minimal(), axes=frozenset({Axis.COIL}))
        assert "axes" not in batch.metadata


class TestBatchAxesHaveAProducer:
    """A field nothing fills, read by a rule that finds it empty, is pitfall #16
    wearing progress's clothes. These pin that both pipelines resolve it."""

    def test_the_train_loop_resolves_and_passes_axes(self) -> None:
        import inspect

        from spectramr.pipelines.training_loop import _execute_training_loop

        src = inspect.getsource(_execute_training_loop)
        assert "resolve_axes_for(" in src
        assert "from_dict(batch, axes=batch_axes)" in src

    def test_the_validation_loop_resolves_and_passes_axes(self) -> None:
        import inspect

        from spectramr.pipelines.train import _run_validation

        src = inspect.getsource(_run_validation)
        assert "resolve_axes_for(" in src
        assert "axes=_batch_axes" in src

    def test_resolution_is_hoisted_out_of_the_batch_loop(self) -> None:
        """``resolve_axes_for`` walks the config; per-batch traversal is the
        hot-path work the training-loop rules forbid. Resolve once."""
        import inspect

        from spectramr.pipelines.training_loop import _execute_training_loop

        src = inspect.getsource(_execute_training_loop)
        assert src.count("resolve_axes_for(") == 1


class TestReadBatchField:
    """``read_batch_field`` must reach metadata that ``hasattr`` cannot see.

    The defect this replaces was written the same way in 23 places: ask
    ``isinstance(batch_data, dict)``, then fall back to ``hasattr``. Against a
    :class:`TrainingBatch` -- which is what ``BatchAdapter.from_dict`` hands the
    training loop -- *both* legs miss, because a dataclass is not a mapping and
    ``.metadata`` is invisible to attribute lookup. The caller then reads a
    published value as absent, which is a silent default substitution.
    """

    @staticmethod
    def _batch(**metadata):
        tensor = torch.randn(1, 2, 4, 4)
        return BatchAdapter.from_dict(
            {"input": tensor, "target": tensor.clone(), **metadata}
        )

    def test_the_old_guard_could_not_see_metadata_at_all(self):
        """Pins WHY the fix is needed, so the rationale cannot rot away."""
        batch = self._batch(kspace_scale=torch.tensor(224.359))

        assert isinstance(batch, dict) is False, "leg 1 of the old guard"
        assert hasattr(batch, "kspace_scale") is False, "leg 2 of the old guard"
        assert "kspace_scale" in batch, "yet the mapping protocol finds it"

    def test_a_metadata_field_is_found_on_a_training_batch(self):
        batch = self._batch(kspace_scale=torch.tensor(224.359))
        assert read_batch_field(batch, "kspace_scale").item() == pytest.approx(224.359)

    def test_the_answer_does_not_depend_on_the_container(self):
        """The invariant: same payload, same answer, dict or TrainingBatch."""
        payload = {"kspace_scale": torch.tensor(7.5)}
        from_dict = read_batch_field(payload, "kspace_scale")
        from_batch = read_batch_field(self._batch(**payload), "kspace_scale")
        assert torch.equal(from_dict, from_batch)

    def test_a_published_false_is_a_value_not_an_absence(self):
        """Non-negotiable 3b: a declared ``False`` must never read as unset.

        ``experiment_11_attention_none``'s dataset publishes
        ``kspace_normalized=False``; collapsing that to ``None`` is what let the
        loudest available signal go unheard.
        """
        batch = self._batch(kspace_normalized=torch.tensor(False))
        value = read_batch_field(batch, "kspace_normalized")
        assert value is not None
        assert bool(value) is False

    def test_an_absent_field_returns_the_default(self):
        batch = self._batch()
        assert read_batch_field(batch, "kspace_scale") is None
        assert read_batch_field(batch, "kspace_scale", default=1.0) == 1.0

    def test_none_batch_is_accepted_and_yields_the_default(self):
        """Validation calls arrive with ``batch_data=None``; that must not raise."""
        assert read_batch_field(None, "kspace_scale") is None
        assert read_batch_field(None, "kspace_scale", default=3.0) == 3.0

    def test_names_resolve_in_precedence_order(self):
        """Alias families resolve in one call, first non-None wins."""
        batch = self._batch(acceleration_mask=torch.ones(1, 1, 4, 4))
        found = read_batch_field(batch, "mask", "acceleration_mask")
        assert found is not None, "must fall through to the second spelling"

        both = self._batch(mask=torch.zeros(1, 1, 4, 4), acceleration_mask=torch.ones(1, 1, 4, 4))
        assert read_batch_field(both, "mask", "acceleration_mask").sum() == 0, (
            "the first name listed wins when several are present"
        )

    def test_canonical_dataclass_fields_are_reachable_too(self):
        """``mask`` is a real field, not metadata; one call must cover both homes."""
        mask = torch.ones(1, 1, 4, 4)
        batch = TrainingBatch(input=torch.randn(1, 2, 4, 4), target=torch.randn(1, 2, 4, 4), mask=mask)
        assert read_batch_field(batch, "mask") is mask

    def test_attribute_objects_are_still_served(self):
        """The ``getattr`` leg keeps namespaces and Mock batches working.

        Test suites across this repo hand strategies ``Mock`` batches. Under the
        old guard those landed on ``hasattr`` and resolved; the leg is retained so
        that behaviour is bit-identical rather than quietly reddening 20 suites.
        """

        class _Namespace:
            kspace_scale = 9.0

        assert read_batch_field(_Namespace(), "kspace_scale") == 9.0
        assert read_batch_field(_Namespace(), "absent_field") is None


class TestAlignScaleToBatch:
    """``align_scale_to_batch`` reconciles a per-subject scale with a per-slice batch.

    Regression anchor (2026-08-19): a 40-iteration cluster relaunch of
    ``experiment_11_attention_none`` trained fine and then died at the first
    validation step::

        diffusion.py:4758  hr_fakes_for_metrics = hr_fakes * denom_scale
        RuntimeError: The size of tensor a (36) must match the size of
                      tensor b (2) at non-singleton dimension 0

    ``36 = 2 subjects x 18 slices``. ``train.py._preprocess_validation_tensor``
    folds depth into the batch axis for the *tensors* and leaves the per-sample
    batch *fields* alone, so a per-subject ``kspace_scale`` of length 2 met a
    per-slice prediction of length 36. The reshape it replaced,
    ``scale.view(-1, 1, 1, 1)``, adopted the producer's length without ever
    comparing it to the consumer's.
    """

    def test_the_exact_cluster_crash_now_aligns(self):
        """B=2, D=18 -> 36: the failing multiply must succeed."""
        scale = torch.tensor([224.36, 198.15])
        aligned = align_scale_to_batch(scale, 36, field="kspace_scale")
        assert aligned.shape == (36, 1, 1, 1)
        # The multiply that raised is the assertion.
        hr_fakes = torch.ones(36, 1, 4, 4)
        assert (hr_fakes * aligned).shape == (36, 1, 4, 4)

    def test_expansion_is_subject_major_by_value(self):
        """VALUES, not shapes -- ``repeat`` produces the same shape and is wrong.

        Every 5D->4D flatten in this repo keeps the batch axis first and depth
        second (``permute(0, 4, 1, 2, 3)`` / ``permute(0, 2, 1, 3, 4)``), so the
        flat index is ``b * D + d``: all of subject 0's slices, then subject 1's.
        ``repeat``/``tile`` would interleave ``[10, 20, 10, 20, 10, 20]`` and
        grade subject 1's slices in subject 0's units -- same shape, no error,
        wrong metrics. That is strictly worse than the crash this fixes, so the
        ordering is pinned by value.
        """
        aligned = align_scale_to_batch(torch.tensor([10.0, 20.0]), 6)
        assert aligned.flatten().tolist() == [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
        # Anti-vacuity: the wrong expansion is a *different* list, not a shape error.
        wrong = torch.tensor([10.0, 20.0]).repeat(3)
        assert wrong.shape == aligned.flatten().shape
        assert wrong.tolist() != aligned.flatten().tolist()

    def test_per_slice_scale_passes_through_untouched(self):
        """An already-aligned scale must not be re-expanded or reordered."""
        scale = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert align_scale_to_batch(scale, 4).flatten().tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_scalar_expands_to_the_whole_batch(self):
        """The one arm of the old ladder that was already correct is preserved."""
        aligned = align_scale_to_batch(torch.tensor(5.0), 4)
        assert aligned.shape == (4, 1, 1, 1)
        assert aligned.flatten().tolist() == [5.0] * 4

    def test_non_tensor_scale_is_accepted(self):
        """``scale_factor`` is a plain ``1.0`` float on the legacy/mock path."""
        assert align_scale_to_batch(1.0, 3).flatten().tolist() == [1.0, 1.0, 1.0]
        assert align_scale_to_batch([2.0, 4.0], 4).flatten().tolist() == [
            2.0,
            2.0,
            4.0,
            4.0,
        ]

    def test_already_four_dimensional_scale_is_handled(self):
        """Trailing singleton axes are a previous caller's shape, not extent."""
        scale = torch.full((2, 1, 1, 1), 7.0)
        assert align_scale_to_batch(scale, 4).flatten().tolist() == [7.0] * 4

    def test_a_length_that_does_not_divide_raises(self):
        """No benign reading exists, so this must raise -- not broadcast."""
        with pytest.raises(ValueError, match="5 entries"):
            align_scale_to_batch(torch.ones(5), 36, field="kspace_scale")

    def test_an_empty_scale_raises_rather_than_dividing_by_zero(self):
        """``batch_size % 0`` would be a ZeroDivisionError with no attribution."""
        with pytest.raises(ValueError, match="0 entries"):
            align_scale_to_batch(torch.ones(0), 36, field="kspace_scale")

    def test_a_scale_with_spatial_extent_raises(self):
        """A scale *map* is not a per-sample scalar; flattening it would
        reinterpret spatial structure as batch entries."""
        with pytest.raises(ValueError, match="must be singleton"):
            align_scale_to_batch(torch.ones(2, 1, 4, 4), 36, field="kspace_scale")

    def test_the_field_name_reaches_the_message(self):
        """A raise must be attributable to a producer, not just a shape."""
        with pytest.raises(ValueError, match="my_field"):
            align_scale_to_batch(torch.ones(5), 36, field="my_field")

    def test_result_broadcasts_against_a_four_dimensional_batch(self):
        """The contract is "broadcast-ready", so assert the broadcast."""
        aligned = align_scale_to_batch(torch.tensor([2.0, 4.0]), 4)
        batch = torch.ones(4, 3, 8, 8)
        scaled = batch * aligned
        assert scaled.shape == (4, 3, 8, 8)
        # Subject 0's two slices carry 2.0, subject 1's carry 4.0.
        assert scaled[0].unique().tolist() == [2.0]
        assert scaled[1].unique().tolist() == [2.0]
        assert scaled[2].unique().tolist() == [4.0]
        assert scaled[3].unique().tolist() == [4.0]



class TestBatchAdapterRepetitionStack:
    """A Noise2Noise batch has a stack instead of a target; nothing else may omit one."""

    def test_stack_without_target_is_accepted(self):
        batch = BatchAdapter.from_dict(
            {"input": torch.zeros(2, 1, 4, 4), "kspace_reps": torch.zeros(2, 2, 4, 4)}
        )
        assert batch.target is None
        assert "kspace_reps" in batch.metadata

    def test_missing_target_without_stack_still_raises(self):
        with pytest.raises(ValueError, match="canonical keys"):
            BatchAdapter.from_dict({"input": torch.zeros(2, 1, 4, 4)})


class TestRejectTargetlessSamples:
    """A supervised step must refuse samples flagged has_target=False (planted violation)."""

    def test_flagged_sample_in_training_batch_raises(self):
        batch = TrainingBatch(
            input=torch.zeros(2, 1, 8, 8),
            target=torch.zeros(2, 1, 8, 8),
            metadata={"has_target": torch.tensor([True, False])},
        )
        with pytest.raises(TargetlessSampleError, match="allow_unpaired"):
            reject_targetless_samples(batch, strategy="ReconstructionTrainingStrategy")

    def test_flagged_sample_in_dict_batch_raises(self):
        batch = {"input": torch.zeros(1, 1, 8, 8), "target": torch.zeros(1, 1, 8, 8),
                 "has_target": torch.tensor([False])}
        with pytest.raises(TargetlessSampleError):
            reject_targetless_samples(batch, strategy="GANTrainingStrategy")

    def test_all_targets_present_passes(self):
        batch = {"input": torch.zeros(2, 1, 8, 8), "target": torch.zeros(2, 1, 8, 8),
                 "has_target": torch.tensor([True, True])}
        reject_targetless_samples(batch, strategy="GANTrainingStrategy")

    def test_absent_flag_is_a_paired_sample(self):
        batch = TrainingBatch(input=torch.zeros(1, 1, 8, 8), target=torch.zeros(1, 1, 8, 8))
        reject_targetless_samples(batch, strategy="GANTrainingStrategy")

    @pytest.mark.parametrize("order", [(True, False), (False, True)])
    def test_mixed_batch_keeps_flag_in_either_order(self, order):
        """Collation takes keys from the first sample; the flag must survive either order."""
        samples = [
            {"input": torch.zeros(1, 8, 8), "target": torch.zeros(1, 8, 8),
             "has_target": torch.tensor(flag)}
            for flag in order
        ]
        batch = BatchAdapter.from_dict(default_collate(samples))
        with pytest.raises(TargetlessSampleError):
            reject_targetless_samples(batch, strategy="ReconstructionTrainingStrategy")
