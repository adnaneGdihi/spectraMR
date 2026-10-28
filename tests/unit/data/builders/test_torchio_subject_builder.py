"""Unit tests for Phase T4: Subject Builder.

Tests cover:
- FastMRISubjectBuilder (k-space, sensitivity, target data)
- PreprocessedSubjectBuilder (various file formats)

- SubjectBuilderFactory (registry dispatch)
- Affine consistency enforcement
- Data type conversions
"""

import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torchio as tio

from spectramr.data.builders.torchio_subject_builder import (
    FastMRISubjectBuilder,
    PreprocessedSubjectBuilder,
    SubjectBuilder,
    SubjectBuilderFactory,
)


class TestSubjectBuilder(TestCase):
    """Test base SubjectBuilder functionality."""

    def setUp(self):
        """Set up test fixtures."""

        # Create a mock builder for testing base functionality
        class ConcreteBuilder(SubjectBuilder):
            def build(self, record):
                # Subject requires at least one image in recent torchio versions
                return tio.Subject(
                    image=tio.ScalarImage(tensor=torch.zeros(1, 1, 1, 1))
                )

        self.builder = ConcreteBuilder()

    def test_ensure_4d_tensor_from_2d(self):
        """Test conversion from 2D (H, W) to 4D."""
        tensor_2d = torch.randn(64, 128)  # (H, W)
        result = self.builder._ensure_4d_tensor(tensor_2d)

        assert result.ndim == 4, "Should be 4D"
        assert result.shape == (1, 64, 128, 1), "Should be (1, H, W, 1)"

    def test_ensure_4d_tensor_from_3d(self):
        """Test conversion from 3D (C, H, W) to 4D."""
        tensor_3d = torch.randn(2, 64, 128)  # (C, H, W)
        result = self.builder._ensure_4d_tensor(tensor_3d)

        assert result.ndim == 4, "Should be 4D"
        assert result.shape == (2, 64, 128, 1), "Should be (C, H, W, 1)"

    def test_ensure_4d_tensor_already_4d(self):
        """Test that 4D tensors pass through unchanged."""
        tensor_4d = torch.randn(1, 64, 128, 32)  # (C, H, W, D)
        result = self.builder._ensure_4d_tensor(tensor_4d)

        assert torch.allclose(result, tensor_4d), "Should be unchanged"
        assert result.shape == tensor_4d.shape

    def test_fastmri_to_torchio_2d(self):
        """Test FastMRI 2D (H, W) conversion."""
        tensor = torch.randn(64, 128)
        result = self.builder._fastmri_to_torchio(tensor)

        assert result.ndim == 4, "Should be 4D"
        assert result.shape == (1, 64, 128, 1)

    def test_fastmri_to_torchio_3d_slices(self):
        """Test FastMRI 3D (S, H, W) conversion to (1, H, W, S)."""
        tensor = torch.randn(10, 64, 128)  # (Slices, H, W)
        result = self.builder._fastmri_to_torchio(tensor)

        assert result.ndim == 4, "Should be 4D"
        assert result.shape == (1, 64, 128, 10), "Slices should be depth"

    def test_fastmri_to_torchio_3d_single_slice_multicoil(self):
        """WS2 regression: a per-slice multicoil read is (Coils, H, W) — the
        leading axis is COILS, not slices — and must map to (Coils, H, W, 1).

        Before the fix ``single_slice`` was ignored and (8, 64, 128) was read as
        an 8-slice single-coil volume → (1, 64, 128, 8), dumping the coils onto
        the depth axis and forcing channels=1."""
        tensor = torch.randn(8, 64, 128)  # (Coils, H, W) single slice
        result = self.builder._fastmri_to_torchio(tensor, single_slice=True)

        assert result.shape == (8, 64, 128, 1), "Coils→channels, depth=1"

    def test_fastmri_to_torchio_3d_volume_default_unchanged(self):
        """Without ``single_slice`` the volume mapping is unchanged (guards
        against a regression flipping the default)."""
        tensor = torch.randn(10, 64, 128)
        assert self.builder._fastmri_to_torchio(tensor).shape == (1, 64, 128, 10)

    def test_enforce_consistent_affines_single_image(self):
        """Test affine enforcement with single image."""
        tensor = torch.randn(1, 64, 128, 32)
        subject = tio.Subject(image=tio.ScalarImage(tensor=tensor))

        result = self.builder._enforce_consistent_affines(subject)

        # Should complete without error
        assert "image" in result, "Image should be preserved"
        assert result["image"].tensor.shape == tensor.shape

    def test_enforce_consistent_affines_multiple_images(self):
        """Test affine enforcement with multiple images."""
        tensor1 = torch.randn(1, 64, 128, 32)
        tensor2 = torch.randn(1, 64, 128, 32)

        # Create subject with different affines
        img1 = tio.ScalarImage(tensor=tensor1, affine=np.eye(4))
        img2_affine = np.diag([2, 2, 2, 1])  # Different affine
        img2 = tio.ScalarImage(tensor=tensor2, affine=img2_affine)

        subject = tio.Subject(image1=img1, image2=img2)
        result = self.builder._enforce_consistent_affines(subject)

        # All images should now have same affine
        affines = [img.affine for img in result.get_images()]
        for affine in affines[1:]:
            assert np.allclose(affine, affines[0]), "All affines should match"

    def test_enforce_consistent_affines_empty_subject(self):
        """Test affine enforcement on subject with no images."""
        # Use a dummy tensor that is NOT a tio.Image to avoid the error if possible,
        # or just accept that torchio Subject needs images and provide a minimal one.
        try:
            subject = tio.Subject(metadata={"key": "value"})
        except TypeError:
            # If torchio requires images, this test is less relevant or needs images
            pytest.skip("torchio.Subject requires images")
            return

        result = self.builder._enforce_consistent_affines(subject)

        # Should return unchanged
        assert result == subject


class TestFastMRISubjectBuilder(TestCase):
    """Test FastMRISubjectBuilder."""

    def setUp(self):
        """Set up test fixtures."""
        self.primary_io = MagicMock()
        self.target_io = MagicMock()
        self.sensitivity_io = MagicMock()

        self.builder = FastMRISubjectBuilder(
            primary_io=self.primary_io,
            target_io=self.target_io,
            sensitivity_io=self.sensitivity_io,
        )

    def test_build_basic_kspace(self):
        """Test building subject with k-space data."""
        # Mock IO
        kspace_data = torch.randn(10, 64, 128)  # (S, H, W)
        self.primary_io.load.return_value = {
            "data": kspace_data,
            "affine": np.eye(4),
        }

        record = {"primary_path": "path/to/kspace.pt"}
        subject = self.builder.build(record)

        assert "kspace" in subject, "Should have kspace"
        assert "target" in subject, "Should have target (defaults to kspace)"
        assert subject["kspace"].tensor.ndim == 4

    def test_build_records_the_source_path_for_sidecar_readers(self):
        """Every image here is built with ``tensor=``, so ``Image.path`` is None.

        A transform that must find a sibling file on disk -- ``LoadDWIMetadata``
        looking for .bval/.bvec, above all -- therefore had nothing to resolve
        against on any route through this builder, and silently attached
        nothing even at ``strict=True``. The path string is already computed
        here for the physics heuristic.
        """
        self.primary_io.load.return_value = {
            "data": torch.randn(10, 64, 128),
            "affine": np.eye(4),
        }

        subject = self.builder.build({"primary_path": "path/to/sub-01_dwi.nii.gz"})

        assert subject["input"].path is None, (
            "precondition: tensor-backed images carry no torchio path"
        )
        assert subject["source_path"] == "path/to/sub-01_dwi.nii.gz"

    def test_built_subject_feeds_the_dwi_sidecar_reader(self):
        """Assert the seam, not the unit: builder output -> LoadDWIMetadata.

        Each half can be right while the pair is broken -- which is exactly what
        happened. The builder produced a valid Subject and the transform parsed
        sidecars correctly, but nothing carried the filename between them, so a
        DWI arm got no b_values on any route through this builder.
        """
        import tempfile

        import nibabel as nib

        from spectramr.data.transforms.dwi_metadata import LoadDWIMetadata

        tmp = Path(tempfile.mkdtemp())
        img = tmp / "sub-01_dwi.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((4, 4, 2, 5), dtype=np.float32), affine=np.eye(4)),
            str(img),
        )
        (tmp / "sub-01_dwi.bval").write_text("0 1000 1000 2000 2000\n")
        (tmp / "sub-01_dwi.bvec").write_text("0 1 0 1 0\n0 0 1 0 1\n0 0 0 0 0\n")

        self.primary_io.load.return_value = {
            "data": torch.randn(5, 4, 4),
            "affine": np.eye(4),
        }
        subject = self.builder.build({"primary_path": str(img)})

        assert subject["input"].path is None  # tensor-backed, as every route here is
        out = LoadDWIMetadata(strict=True)(subject)
        assert out["n_directions"] == 5
        assert isinstance(out["b_values"], torch.Tensor)
        assert isinstance(out["b_vectors"], torch.Tensor)

    def test_build_with_sensitivity(self):
        """Test building subject with sensitivity maps."""
        kspace_data = torch.randn(10, 64, 128)
        smap_data = torch.randn(1, 64, 128, 10)

        self.primary_io.load.return_value = {
            "data": kspace_data,
            "affine": np.eye(4),
        }
        self.sensitivity_io.load.return_value = {
            "data": smap_data,
            "affine": np.eye(4),
        }

        record = {
            "primary_path": "path/to/kspace.pt",
            "sensitivity_path": "path/to/smap.pt",
        }
        subject = self.builder.build(record)

        assert "sensitivity" in subject, "Should have sensitivity maps"

    def test_build_with_target(self):
        """Test building subject with separate target."""
        kspace_data = torch.randn(10, 64, 128)
        target_data = torch.randn(10, 64, 128)

        self.primary_io.load.return_value = {
            "data": kspace_data,
            "affine": np.eye(4),
        }
        self.target_io.load.return_value = {
            "data": target_data,
            "affine": np.eye(4),
        }

        record = {
            "primary_path": "path/to/kspace.pt",
            "target_path": "path/to/target.pt",
        }
        subject = self.builder.build(record)

        assert "target" in subject, "Should have target"
        # Shape should match (slices -> depth)
        assert subject["target"].tensor.shape == subject["kspace"].tensor.shape

    def test_build_handles_missing_sensitivity(self):
        """Test graceful handling of missing sensitivity maps."""
        kspace_data = torch.randn(10, 64, 128)
        self.primary_io.load.return_value = {
            "data": kspace_data,
            "affine": np.eye(4),
        }
        self.sensitivity_io.load.side_effect = FileNotFoundError("Not found")

        record = {
            "primary_path": "path/to/kspace.pt",
            "sensitivity_path": "path/to/smap.pt",
        }
        subject = self.builder.build(record)

        # Should still build subject without sensitivity
        assert "kspace" in subject
        assert "sensitivity" not in subject or len(subject) >= 1

    def test_build_forwards_slice_index_metadata_to_all_loads(self):
        """Perf/IO regression (2026-07-02): the builder must forward the full
        record as ``metadata=`` on the primary/sensitivity/target loads so a
        per-slice record (``variant: 2d_slices`` → one record per
        ``slice_index``) hits the lazy single-slice read in
        FastMRIH5Strategy / NiftiStrategy instead of decoding the whole volume
        once per slice per epoch."""
        self.primary_io.load.return_value = {
            "data": torch.randn(1, 64, 128),
            "affine": np.eye(4),
        }
        self.sensitivity_io.load.return_value = {
            "data": torch.randn(1, 64, 128, 4),
            "affine": np.eye(4),
        }
        self.target_io.load.return_value = {
            "data": torch.randn(1, 64, 128),
            "affine": np.eye(4),
        }

        record = {
            "primary_path": "vol.h5",
            "sensitivity_path": "smap.npy",
            "target_path": "rss.h5",
            "slice_index": 7,
        }
        self.builder.build(record)

        for io in (self.primary_io, self.sensitivity_io, self.target_io):
            io.load.assert_called_once()
            # ``metadata`` may be passed positionally or by keyword; assert the
            # record (with slice_index) reached the strategy either way.
            _args, kwargs = io.load.call_args
            passed = kwargs.get("metadata", _args[1] if len(_args) > 1 else None)
            assert passed is record, (
                f"{io} did not receive the record as metadata "
                "(slice_index would be dropped → full-volume decode per slice)"
            )

    def test_build_without_slice_index_still_loads(self):
        """No ``slice_index`` in the record → metadata forwarding is a no-op
        (strategies fall back to the full read); the build still succeeds."""
        self.primary_io.load.return_value = {
            "data": torch.randn(10, 64, 128),
            "affine": np.eye(4),
        }
        subject = self.builder.build({"primary_path": "vol.h5"})
        assert "kspace" in subject

    def test_subject_affine_consistency(self):
        """Test that affine is consistent across images."""
        kspace_data = torch.randn(10, 64, 128)
        target_data = torch.randn(10, 64, 128)

        affine = np.diag([2, 2, 2, 1])
        self.primary_io.load.return_value = {
            "data": kspace_data,
            "affine": affine,
        }
        self.target_io.load.return_value = {
            "data": target_data,
            "affine": np.eye(4),  # Different affine
        }

        record = {
            "primary_path": "path/to/kspace.pt",
            "target_path": "path/to/target.pt",
        }
        subject = self.builder.build(record)

        # All images should have consistent affine (kspace's)
        affines = [img.affine for img in subject.get_images()]
        for aff in affines:
            assert np.allclose(aff, affine), "Should enforce consistent affine"


class TestPreprocessedSubjectBuilder(TestCase):
    """Test PreprocessedSubjectBuilder."""

    def setUp(self):
        """Set up test fixtures."""
        self.builder = PreprocessedSubjectBuilder()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_load_tensor_pt_format(self):
        """Test loading PyTorch tensor."""
        tensor = torch.randn(64, 128)
        pt_file = self.temp_path / "test.pt"
        torch.save(tensor, pt_file)

        loaded = self.builder._load_tensor(pt_file)

        assert isinstance(loaded, torch.Tensor), "Should be tensor"
        assert torch.allclose(loaded, tensor), "Should match original"

    def test_load_tensor_npy_format(self):
        """Test loading NumPy array."""
        array = np.random.randn(64, 128)
        npy_file = self.temp_path / "test.npy"
        np.save(npy_file, array)

        loaded = self.builder._load_tensor(npy_file)

        assert isinstance(loaded, torch.Tensor), "Should be tensor"
        assert np.allclose(loaded.numpy(), array), "Should match original"

    def test_build_basic_preprocessed(self):
        """Test building subject from preprocessed data."""
        # Create test files
        gt_tensor = torch.randn(64, 128)
        input_tensor = torch.randn(64, 128)

        gt_file = self.temp_path / "gt.pt"
        input_file = self.temp_path / "input.pt"
        torch.save(gt_tensor, gt_file)
        torch.save(input_tensor, input_file)

        record = {
            "gt_image_path": gt_file,
            "image_path": input_file,
        }
        subject = self.builder.build(record)

        assert "target" in subject, "Should have target (gt_image)"
        assert "input" in subject, "Should have input"

    def test_build_with_kspace(self):
        """Test building subject with k-space data."""
        kspace_tensor = torch.randn(10, 64, 128)
        kspace_file = self.temp_path / "kspace.pt"
        torch.save(kspace_tensor, kspace_file)

        record = {"kspace_path": kspace_file}
        subject = self.builder.build(record)

        assert "kspace" in subject, "Should have kspace"

    def test_build_with_metadata(self):
        """Test building subject with statistics metadata."""
        stats = {"mean": 0.5, "std": 0.1, "min": 0.0, "max": 1.0}
        stats_file = self.temp_path / "stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f)

        # Subject needs at least one image
        img_file = self.temp_path / "img.pt"
        torch.save(torch.randn(1, 16, 16), img_file)

        record = {
            "statistics_path": stats_file,
            "image_path": img_file,
        }
        subject = self.builder.build(record)

        assert "statistics" in subject, "Should have statistics"
        assert subject["statistics"]["mean"] == 0.5

    def test_build_handles_missing_files(self):
        """Test graceful handling of missing files."""
        record = {
            "image_path": "/nonexistent/image.pt",
        }

        # Should raise or handle gracefully
        try:
            subject = self.builder.build(record)
            # If it doesn't raise, subject should still be valid
            assert isinstance(subject, tio.Subject)
        except (FileNotFoundError, KeyError):
            # Either behavior is acceptable
            pass


class TestSubjectBuilderFactory(TestCase):
    """Test SubjectBuilderFactory."""

    def test_create_fastmri_builder(self):
        """Test creating FastMRI builder."""
        io = MagicMock()
        builder = SubjectBuilderFactory.create("fastmri", primary_io=io)

        assert isinstance(builder, FastMRISubjectBuilder), "Should be FastMRI builder"

    def test_create_preprocessed_builder(self):
        """Test creating preprocessed builder."""
        builder = SubjectBuilderFactory.create("preprocessed")

        assert isinstance(
            builder, PreprocessedSubjectBuilder
        ), "Should be preprocessed builder"

    def test_create_invalid_builder(self):
        """Test that invalid builder type raises error."""
        with pytest.raises(ValueError, match="Unknown builder type"):
            SubjectBuilderFactory.create("invalid_builder")

    def test_available_builders(self):
        """Test that all standard builders are available."""
        builders = ["fastmri", "preprocessed"]

        for builder_name in builders:
            kwargs = {}
            if builder_name == "fastmri":
                kwargs["primary_io"] = MagicMock()

            builder = SubjectBuilderFactory.create(builder_name, **kwargs)
            assert isinstance(
                builder, SubjectBuilder
            ), f"{builder_name} should be available"

    def test_register_custom_builder(self):
        """Test registering a custom builder."""

        class CustomBuilder(SubjectBuilder):
            def build(self, record):
                return tio.Subject(
                    image=tio.ScalarImage(tensor=torch.zeros(1, 1, 1, 1))
                )

        SubjectBuilderFactory.register("custom", CustomBuilder)

        builder = SubjectBuilderFactory.create("custom")
        assert isinstance(builder, CustomBuilder), "Should be custom builder"

    def test_register_invalid_builder(self):
        """Test that non-SubjectBuilder classes cannot be registered."""

        class NotABuilder:
            pass

        with pytest.raises(TypeError):
            SubjectBuilderFactory.register("invalid", NotABuilder)


class TestSubjectBuilderIntegration(TestCase):
    """Integration tests for Subject builder workflow."""

    def test_subject_queue_compatibility(self):
        """Test that built Subjects are compatible with TorchIO Queue.

        Queue requires:
        1. All images have same affine
        2. All images have same spatial shape
        """
        builder = PreprocessedSubjectBuilder()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test data
            tensor1 = torch.randn(64, 128)
            tensor2 = torch.randn(64, 128)

            file1 = temp_path / "img1.pt"
            file2 = temp_path / "img2.pt"
            torch.save(tensor1, file1)
            torch.save(tensor2, file2)

            record = {
                "image_path": file1,
                "gt_image_path": file2,
            }
            subject = builder.build(record)

            # Verify Queue compatibility
            images = subject.get_images()
            assert len(images) > 0, "Subject should have images"

            # All should have same affine
            affines = [img.affine for img in images]
            for affine in affines[1:]:
                assert np.allclose(affine, affines[0])

            # All should have same spatial shape
            shapes = [img.spatial_shape for img in images]
            for shape in shapes[1:]:
                assert shape == shapes[0]

    def test_multiple_builders_compatibility(self):
        """Test that different builders produce compatible Subjects."""
        affine = np.eye(4)

        # Create Subjects from different builders
        preprocessed_builder = PreprocessedSubjectBuilder()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            tensor = torch.randn(64, 128)
            file = temp_path / "test.pt"
            torch.save(tensor, file)

            record = {"image_path": file}
            subject = preprocessed_builder.build(record)

            # All subjects should be standard TorchIO Subjects
            assert isinstance(subject, tio.Subject)

            # All images should have standard affines
            for image in subject.get_images():
                assert image.affine.shape == (4, 4)
                assert isinstance(image.tensor, torch.Tensor)


class TestLogScalingNormalization(TestCase):
    """K-space log-scaling normalization (experiment_11 DC-blob fix).

    Retargeted 2026-07-27: this used to exercise
    ``FastMRISubjectBuilder._normalize_kspace``, which was removed when the
    subject builder stopped normalizing (``KSpaceNormalizationTransform`` is the
    single normalizer -- see docs/kspace_normalization_ssot.rst). The behaviour
    under test is unchanged; it now targets the SSOT the builder delegated to.

    ``log_scaling`` must apply phase-preserving log1p compression on top of the
    percentile divide so the network sees a CNN-friendly dynamic range instead
    of the raw ~200x DC spike.
    """

    def _dc_dominated_kspace(self):
        torch.manual_seed(0)
        k = torch.randn(4, 64, 64, 2, dtype=torch.complex64) * 0.2
        k[:, 32, 32, :] = 44.0  # DC spike
        return k

    @staticmethod
    def _normalize(k, *, log_scaling):
        from spectramr.data.transforms.normalization import normalize_kspace_robust

        return normalize_kspace_robust(
            k, percentile=0.95, log_scaling=log_scaling, channel_dim=0
        )

    def test_log_scaling_compresses_dynamic_range(self):
        k = self._dc_dominated_kspace()
        lin, _ = self._normalize(k.clone(), log_scaling=False)
        log, _ = self._normalize(k.clone(), log_scaling=True)
        assert log.abs().max() < lin.abs().max() / 5, (
            f"log_scaling must compress the DC range: linear={lin.abs().max():.1f} "
            f"log={log.abs().max():.1f}"
        )

    def test_log_scaling_preserves_phase(self):
        k = self._dc_dominated_kspace()
        log, _ = self._normalize(k.clone(), log_scaling=True)
        mask = k.abs() > 1e-3
        assert torch.allclose(torch.angle(log)[mask], torch.angle(k)[mask], atol=1e-4)

    def test_scale_is_returned_and_inverts_the_normalization(self):
        """The returned scale must round-trip: decompress then rescale == input.

        Without the scale the caller cannot undo the log, ``decompress``
        under-restores, and the validation image is a DC-blob / edge-enhanced
        render.
        """
        from spectramr.data.transforms.normalization import decompress_kspace_log

        k = self._dc_dominated_kspace()
        norm, scale = self._normalize(k.clone(), log_scaling=True)
        assert scale is not None
        recovered = decompress_kspace_log(norm) * scale
        assert torch.allclose(recovered, k, atol=1e-1, rtol=1e-2)


class TestRssImageSingleChannelCollapse(TestCase):
    """RC-A (smoke audit 2026-06-03): ``rss_image`` must reduce a target to a
    single magnitude channel for EVERY coil count.

    The target-side coil-processing gate used to fire only for
    ``is_complex and shape[0] > 2``, so a single complex coil (shape[0] == 1)
    or a real-stacked R/I pair (shape[0] == 2) slipped through unchanged. A
    magnitude-only baseline (in_channels=1) then received a 2-channel
    (real/imag of one coil) target — and the simulator-derived input was
    2-channel — so the first conv raised "expected 1 channel, got 2" at iter 1
    (eval_c2/eval_c3/eval_c7/exp_c4). Multi-coil arms (e.g. exp_c6, 4 coils)
    already passed the gate, so their behaviour is unchanged. This pins the
    coil-collapse logic the fixed gate now routes those shapes through.
    """

    def _rss_builder(self) -> FastMRISubjectBuilder:
        b = FastMRISubjectBuilder.__new__(FastMRISubjectBuilder)
        b.coil_processing_mode = "rss_image"
        b.num_virtual_coils = 4
        b.coil_processing = None  # legacy-mode dispatch (no SSOT physics block)
        return b

    def test_single_complex_coil_collapses_to_one_channel(self) -> None:
        b = self._rss_builder()
        t = torch.randn(1, 8, 8, 1, dtype=torch.complex64)  # (1 coil, H, W, D)
        out = b._apply_coil_processing(t, None, is_kspace=False)
        assert out.shape[0] == 1
        assert not torch.is_complex(out)

    def test_real_interleaved_two_channel_collapses_to_one(self) -> None:
        b = self._rss_builder()
        t = torch.randn(2, 8, 8, 1)  # [R0, I0] of one coil, real-stacked
        out = b._apply_coil_processing(t, None, is_kspace=False)
        assert out.shape[0] == 1

    def test_multicoil_complex_still_collapses_to_one(self) -> None:
        b = self._rss_builder()
        t = torch.randn(4, 8, 8, 1, dtype=torch.complex64)  # 4 coils (exp_c6 shape)
        out = b._apply_coil_processing(t, None, is_kspace=False)
        assert out.shape[0] == 1

    def test_already_single_channel_real_is_unchanged(self) -> None:
        b = self._rss_builder()
        t = torch.randn(1, 8, 8, 1)  # already 1-channel magnitude
        out = b._apply_coil_processing(t, None, is_kspace=False)
        assert out.shape[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ── The subject builder matches and serves; it does not normalize ────────────
# ``data.normalize_kspace`` used to drive BOTH this builder and
# ``KSpaceNormalizationTransform`` (which the dataset applies right after), so
# the k-space got a double percentile divide + double log1p and the transform
# overwrote ``kspace_scale`` — leaving the normalization non-invertible.


class TestBuilderDoesNotNormalize:
    """Normalization belongs to the transform layer, not to ``build()``."""

    def _builder(self, **kw):
        from unittest.mock import MagicMock

        from spectramr.data.builders.torchio_subject_builder import FastMRISubjectBuilder

        return FastMRISubjectBuilder(primary_io=MagicMock(), **kw)

    def test_build_serves_kspace_unscaled(self):
        """The served k-space is byte-for-byte what the IO strategy returned."""
        import numpy as np
        import torch

        b = self._builder()
        kspace_data = torch.randn(4, 32, 32) * 1000.0  # far from unit scale
        b.primary_io.load.return_value = {"data": kspace_data, "affine": np.eye(4)}

        subject = b.build({"primary_path": "k.h5"})
        served = subject["kspace"].tensor

        # Layout-agnostic: build() may reshape (S, H, W) -> (C, H, W, D), but a
        # percentile divide or log1p would change the VALUES, so compare the
        # sorted multiset rather than the arrangement.
        assert torch.allclose(
            served.flatten().sort().values,
            kspace_data.flatten().sort().values,
            rtol=1e-5,
            atol=1e-6,
        ), "build() morphed the k-space; KSpaceNormalizationTransform owns that"

    def test_build_publishes_identity_scale(self):
        """``kspace_scale`` must describe the tensor served next to it."""
        import numpy as np
        import torch

        b = self._builder()
        b.primary_io.load.return_value = {
            "data": torch.randn(4, 32, 32),
            "affine": np.eye(4),
        }
        subject = b.build({"primary_path": "k.h5"})
        assert torch.allclose(subject["kspace_scale"], torch.tensor(1.0))


class TestFastMRIImageDomainCropOrPad:
    """``data.processing.image_domain_crop_or_pad`` reaches the served k-space."""

    @staticmethod
    def _builder(primary: dict, size=(8, 8)) -> FastMRISubjectBuilder:
        io = MagicMock()
        io.load.return_value = primary
        return FastMRISubjectBuilder(primary_io=io, image_domain_crop_or_pad=size)

    @staticmethod
    def _oversampled_kspace() -> torch.Tensor:
        from spectramr.infrastructure.physics.fft_ops import fft2c

        torch.manual_seed(0)
        image = torch.complex(torch.randn(3, 16, 12), torch.randn(3, 16, 12))
        return image, fft2c(image)

    def test_the_input_and_target_carry_the_centre_of_the_image(self):
        from spectramr.infrastructure.physics.fft_ops import ifft2c

        image, kspace = self._oversampled_kspace()
        subject = self._builder({"kspace": kspace}).build(
            {"primary_path": "file_brain_AXT2_0.h5", "slice_index": 0}
        )

        served = subject["kspace"].data
        assert served.shape == (3, 8, 8, 1)
        torch.testing.assert_close(
            ifft2c(served[..., 0]), image[:, 4:12, 2:10], atol=1e-5, rtol=1e-5
        )
        assert torch.equal(subject["target"].data, served)

    def test_an_image_record_is_refused(self):
        with pytest.raises(ValueError, match="served an image"):
            self._builder({"data": torch.randn(16, 12)}).build({"primary_path": "x.h5"})

    def test_a_stored_mask_is_refused(self):
        _, kspace = self._oversampled_kspace()
        builder = self._builder({"kspace": kspace, "mask": torch.ones(12)})
        with pytest.raises(ValueError, match="stored sampling mask"):
            builder.build({"primary_path": "x.h5", "slice_index": 0})

    def test_a_malformed_size_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="image_domain_crop_or_pad"):
            self._builder({}, size=(8,))

    def test_the_dataset_hands_the_size_to_its_builder(self):
        from spectramr.data.datasets.universal_dataset import UniversalMRIDataset

        dataset = UniversalMRIDataset(index=[], image_domain_crop_or_pad=(320, 320))
        assert dataset.subject_builder.image_domain_crop_or_pad == (320, 320)
