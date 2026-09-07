"""Tests for contrast filtering in data loaders.

This module tests the contrast and target_contrast filtering functionality
added to parse_fastmri_index and IndexBuilder methods.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from types import SimpleNamespace

from tests.utils.data_config_stub import DataConfigStub

import pytest

from spectramr.data.metadata.index_builder import IndexBuilder


def MockDataConfig(  # noqa: N802 - kept as a constructor-shaped name for call sites
    data_root="/tmp",
    contrasts=None,
    target_contrasts=None,
    dataset_type="nifti",
    data_layout="flat",
    validation_split=0.1,
    datasets=None,
    holdout_site=None,
    train_sites=None,
    manifest_roles=None,
):
    """Mock data config for testing, built from the SHARED stub.

    This was a hand-rolled class that had been migrated for `data.split.*` and
    not for the rest, so it set `contrasts` flat while the reader walks
    `data.pairing.contrasts` and `data.source.root`. DataConfigStub routes every
    flat legacy kwarg through RENAMES, which is the same table the loader folds
    YAML with -- so the stub cannot disagree with production about where a
    reader looks, and needs no edit when the next leaf moves.
    """
    return DataConfigStub(
        data_root=data_root,
        contrasts=contrasts,
        target_contrasts=target_contrasts,
        dataset_type=dataset_type,
        data_layout=data_layout,
        validation_split=validation_split,
        datasets=datasets or [MagicMock(variant="3d_full")],
        holdout_site=holdout_site,
        train_sites=train_sites,
        manifest_roles=manifest_roles or MagicMock(),
    )


class TestParseNiftiFiltering:
    """Tests for NIfTI contrast filtering in IndexBuilder."""

    @pytest.fixture
    def nifti_files(self, tmp_path):
        """Create mock NIfTI files with different contrasts."""
        # build_nifti_index looks for split subdirectory (train/val) by default
        nifti_dir = tmp_path / "train"
        nifti_dir.mkdir()

        # Create dummy NIfTI files with contrast names in filenames
        contrasts = ["T1w", "T2w", "FLAIR", "PD"]
        files = {}
        for contrast in contrasts:
            file_path = nifti_dir / f"subject_01_{contrast}.nii.gz"
            file_path.touch()
            files[contrast] = file_path

        return tmp_path, nifti_dir, files

    def test_no_contrast_filter_returns_all(self, nifti_files):
        """Test that no contrast filter returns all files."""
        root_path, nifti_dir, files = nifti_files
        config = MockDataConfig(data_root=str(root_path), contrasts=None)

        with patch(
            "spectramr.data.metadata.index_builder.PathResolver.resolve"
        ) as mock_resolve:
            mock_resolve.side_effect = lambda x: str(x)
            index = IndexBuilder.build_nifti_index(config, split="train")

        # Should return all 4 files
        assert len(index) == 4
        file_ids = [rec["file_id"] for rec in index]
        assert any("T1w" in fid for fid in file_ids)
        assert any("T2w" in fid for fid in file_ids)
        assert any("FLAIR" in fid for fid in file_ids)

    def test_single_contrast_filter(self, nifti_files):
        """Test filtering by single contrast."""
        root_path, nifti_dir, files = nifti_files
        config = MockDataConfig(data_root=str(root_path), contrasts=["T1w"])

        with patch(
            "spectramr.data.metadata.index_builder.PathResolver.resolve"
        ) as mock_resolve:
            mock_resolve.side_effect = lambda x: str(x)
            index = IndexBuilder.build_nifti_index(config, split="train")

        # Should return only T1w file
        assert len(index) == 1
        assert "T1w" in index[0]["file_id"]

    def test_multiple_contrast_filter(self, nifti_files):
        """Test filtering by multiple contrasts."""
        root_path, nifti_dir, files = nifti_files
        config = MockDataConfig(data_root=str(root_path), contrasts=["T1w", "T2w"])

        with patch(
            "spectramr.data.metadata.index_builder.PathResolver.resolve"
        ) as mock_resolve:
            mock_resolve.side_effect = lambda x: str(x)
            index = IndexBuilder.build_nifti_index(config, split="train")

        # Should return T1w and T2w files
        assert len(index) == 2
        file_ids = [rec["file_id"] for rec in index]
        assert any("T1w" in fid for fid in file_ids)
        assert any("T2w" in fid for fid in file_ids)

    def test_case_insensitive_contrast_filter(self, nifti_files):
        """Test that contrast filtering is case-insensitive."""
        root_path, nifti_dir, files = nifti_files
        # Request 't1w' (lowercase) should match 'T1w'
        config = MockDataConfig(data_root=str(root_path), contrasts=["t1w"])

        with patch(
            "spectramr.data.metadata.index_builder.PathResolver.resolve"
        ) as mock_resolve:
            mock_resolve.side_effect = lambda x: str(x)
            index = IndexBuilder.build_nifti_index(config, split="train")

        # Should match case-insensitively
        assert len(index) == 1
        assert "T1w" in index[0]["file_id"]

    def test_target_contrast_filter(self, nifti_files):
        """Test filtering by target contrasts only."""
        root_path, nifti_dir, files = nifti_files
        config = MockDataConfig(
            data_root=str(root_path),
            contrasts=None,
            target_contrasts=["FLAIR"],
        )

        with patch(
            "spectramr.data.metadata.index_builder.PathResolver.resolve"
        ) as mock_resolve:
            mock_resolve.side_effect = lambda x: str(x)
            index = IndexBuilder.build_nifti_index(config, split="train")

        # Should return only FLAIR file
        assert len(index) == 1
        assert "FLAIR" in index[0]["file_id"]

    def test_input_and_target_contrast_filter(self, nifti_files):
        """Test filtering by both input and target contrasts (OR logic)."""
        root_path, nifti_dir, files = nifti_files
        config = MockDataConfig(
            data_root=str(root_path),
            contrasts=["T1w"],
            target_contrasts=["T2w"],
        )

        with patch(
            "spectramr.data.metadata.index_builder.PathResolver.resolve"
        ) as mock_resolve:
            mock_resolve.side_effect = lambda x: str(x)
            # With OR logic: files matching EITHER contrasts OR target_contrasts are included
            index = IndexBuilder.build_nifti_index(config, split="train")

        # Should have T1w (matches contrasts) and T2w (matches target_contrasts)
        assert len(index) == 2
        file_ids = [rec["file_id"] for rec in index]
        assert any("T1w" in fid for fid in file_ids)
        assert any("T2w" in fid for fid in file_ids)

    def test_no_matching_contrasts_returns_empty(self, nifti_files):
        """Test that non-existent contrast returns empty index."""
        root_path, nifti_dir, files = nifti_files
        config = MockDataConfig(data_root=str(root_path), contrasts=["NonExistent"])

        with patch(
            "spectramr.data.metadata.index_builder.PathResolver.resolve"
        ) as mock_resolve:
            mock_resolve.side_effect = lambda x: str(x)
            index = IndexBuilder.build_nifti_index(config, split="train")

        # Should return empty index
        assert len(index) == 0


class TestPairedNiftiFiltering:
    """Tests for contrast filtering in paired NIfTI datasets."""

    @pytest.fixture
    def paired_nifti_files(self, tmp_path):
        """Create mock paired NIfTI files."""
        root = tmp_path / "paired_data"
        root.mkdir()

        source_dir = root / "source"
        target_dir = root / "target"
        source_dir.mkdir()
        target_dir.mkdir()

        # Create paired files
        contrasts = ["T1w", "T2w"]
        for contrast in contrasts:
            (source_dir / f"subject_01_{contrast}.nii.gz").touch()
            (target_dir / f"subject_01_{contrast}.nii.gz").touch()

        return root, source_dir, target_dir

    def test_paired_no_contrast_filter(self, paired_nifti_files):
        """Test paired dataset without contrast filter."""
        root, source_dir, target_dir = paired_nifti_files
        config = MockDataConfig(
            data_root=str(root),
            dataset_type="nifti_paired",
            contrasts=None,
        )

        with patch(
            "spectramr.data.metadata.index_builder.PathResolver.resolve"
        ) as mock_resolve:
            mock_resolve.side_effect = lambda x: str(x)
            index = IndexBuilder._build_paired_nifti_index(
                config, Path(str(root)), split="train", is_split_dir=True
            )

        # Should return both T1w and T2w pairs
        assert len(index) == 2

    def test_paired_with_contrast_filter(self, paired_nifti_files):
        """Test paired dataset with contrast filter."""
        root, source_dir, target_dir = paired_nifti_files
        config = MockDataConfig(
            data_root=str(root),
            dataset_type="nifti_paired",
            contrasts=["T1w"],
        )

        with patch(
            "spectramr.data.metadata.index_builder.PathResolver.resolve"
        ) as mock_resolve:
            mock_resolve.side_effect = lambda x: str(x)
            index = IndexBuilder._build_paired_nifti_index(
                config, Path(str(root)), split="train", is_split_dir=True
            )

        # Should return only T1w pair
        assert len(index) == 1
        assert "T1w" in index[0]["file_id"]


class TestFastMRIManifestFiltering:
    """Tests for contrast filtering in FastMRI manifest loading."""

    def test_parse_fastmri_index_with_contrasts(self, tmp_path):
        """Test parse_fastmri_index with contrast filtering."""
        import pickle

        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()

        # Create mock manifest with v2 format
        manifest_data = {
            "version": 2,
            "data_root": str(tmp_path / "data"),
            "use_relative": True,
            "files": [
                {
                    "path": "file_T1w_001.h5",
                    "file_id": "file_T1w_001",
                },
                {
                    "path": "file_T2w_001.h5",
                    "file_id": "file_T2w_001",
                },
                {
                    "path": "file_FLAIR_001.h5",
                    "file_id": "file_FLAIR_001",
                },
            ],
        }

        manifest_path = manifest_dir / "test_manifest.pkl"
        with open(manifest_path, "wb") as f:
            pickle.dump(manifest_data, f)

        # Create data directory structure
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        from spectramr.data.datasets.universal_dataset import parse_fastmri_index

        # Test without contrast filter
        index = parse_fastmri_index(str(manifest_path), data_root=str(data_dir))
        assert len(index) == 3

        # Test with single contrast
        index = parse_fastmri_index(
            str(manifest_path), data_root=str(data_dir), contrasts=["T1w"]
        )
        assert len(index) == 1
        assert "T1w" in index[0]["file_id"]

        # Test with multiple contrasts
        index = parse_fastmri_index(
            str(manifest_path),
            data_root=str(data_dir),
            contrasts=["T1w", "T2w"],
        )
        assert len(index) == 2

        # Test with non-existent contrast
        index = parse_fastmri_index(
            str(manifest_path),
            data_root=str(data_dir),
            contrasts=["NonExistent"],
        )
        assert len(index) == 0

    def test_m4raw_filename_tokens_are_T1_T2_FLAIR_not_T1w(self, tmp_path):
        """M4Raw's tokens are ``T1``/``T2``/``FLAIR`` -- never the NIfTI ``T1w``.

        The filter is an uppercased SUBSTRING match on ``file_id``, and M4Raw
        ships ``<subject>_T101.h5`` / ``<subject>_FLAIR01.h5`` (the trailing two
        digits are the repetition index). So:

        * ``T1`` matches ``2022061203_T101``; ``T1W`` does not.
        * The corpus convention is the other way round -- all 51 arms declaring
          ``pairing.contrasts`` are ``nifti_paired`` with ``T1w``/``T2w``, and the
          schema field's own description advertises ``['T1w', 'FLAIR']``, which
          matches NOTHING on M4Raw.

        This matters because the n2n cohort excludes FLAIR by contrast filter:
        FLAIR ships 2 repetitions and the leave-one-out NEX reference needs 3, so
        a wrong token silently changes which contrasts the cohort trains on. The
        fixture is M4Raw-shaped rather than borrowed from the NIfTI convention
        (a fixture must come from the producer).
        """
        import pickle

        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Real M4Raw basenames: subject id, contrast token, 2-digit repetition.
        files = [
            "2022061203_T101",
            "2022061203_T201",
            "2022061401_FLAIR01",
        ]
        manifest_path = manifest_dir / "m4raw_manifest.pkl"
        with open(manifest_path, "wb") as f:
            pickle.dump(
                {
                    "version": 2,
                    "data_root": str(data_dir),
                    "use_relative": True,
                    "files": [{"path": f"{n}.h5", "file_id": n} for n in files],
                },
                f,
            )

        from spectramr.data.datasets.universal_dataset import parse_fastmri_index

        unfiltered = parse_fastmri_index(str(manifest_path), data_root=str(data_dir))
        assert len(unfiltered) == 3  # anti-vacuity: the fixture is not empty

        # The n2n cohort's declaration: keep the 3-repetition contrasts, drop FLAIR.
        kept = parse_fastmri_index(
            str(manifest_path), data_root=str(data_dir), contrasts=["T1", "T2"]
        )
        assert {r["file_id"] for r in kept} == {"2022061203_T101", "2022061203_T201"}
        assert not any("FLAIR" in r["file_id"] for r in kept)

        # The NIfTI token the schema description advertises matches nothing here.
        assert parse_fastmri_index(
            str(manifest_path), data_root=str(data_dir), contrasts=["T1w"]
        ) == []

        # ...and it fails LOUDLY downstream rather than training on an empty set:
        # `load_fastmri_splits` raises "Train split is empty" on this result.

    def test_parse_fastmri_index_target_contrasts(self, tmp_path):
        """Test parse_fastmri_index with target contrast filtering."""
        import pickle

        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()

        # Create mock manifest
        manifest_data = {
            "version": 2,
            "data_root": str(tmp_path / "data"),
            "use_relative": True,
            "files": [
                {
                    "path": "file_T1w_to_FLAIR_001.h5",
                    "file_id": "file_T1w_to_FLAIR_001",
                },
                {
                    "path": "file_T2w_to_PD_001.h5",
                    "file_id": "file_T2w_to_PD_001",
                },
            ],
        }

        manifest_path = manifest_dir / "test_manifest.pkl"
        with open(manifest_path, "wb") as f:
            pickle.dump(manifest_data, f)

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        from spectramr.data.datasets.universal_dataset import parse_fastmri_index

        # Test with target contrast filter
        index = parse_fastmri_index(
            str(manifest_path),
            data_root=str(data_dir),
            target_contrasts=["FLAIR"],
        )
        assert len(index) == 1
        assert "FLAIR" in index[0]["file_id"]


class TestIntegrationContrastFiltering:
    """Integration tests for contrast filtering across pipeline."""

    def test_contrast_filter_propagation(self, tmp_path):
        """Test that contrast filters propagate through the pipeline."""
        # This test verifies that when a DataConfig has contrasts set,
        # they are correctly passed through consolidated_dataset_factory
        # to the underlying index builders and parse_fastmri_index

        nifti_dir = tmp_path / "nifti_data"
        nifti_dir.mkdir()

        # Create test files
        for contrast in ["T1w", "T2w", "FLAIR"]:
            (nifti_dir / f"{contrast}_001.nii.gz").touch()

        config = MockDataConfig(
            data_root=str(nifti_dir),
            contrasts=["T1w"],
            dataset_type="nifti",
        )

        with patch(
            "spectramr.data.metadata.index_builder.PathResolver.resolve"
        ) as mock_resolve:
            mock_resolve.side_effect = lambda x: str(x)
            index = IndexBuilder.build_nifti_index(config, split="train")

        # Should only have T1w
        assert len(index) == 1
        assert "T1w" in index[0]["file_id"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_parse_fastmri_index_preserves_shape(tmp_path):
    """F-OOM 2026-06-08: parse_fastmri_index must carry the manifest 'shape'
    into each record. Without it, TorchIOQueueBuilder's patch-compat filter
    can't check spatial extent from metadata and instead loads EVERY volume at
    queue-build (1024 x 38MB ~= 39GB -> host OOM, independent of num_workers)."""
    import json

    from spectramr.data.datasets.universal_dataset import parse_fastmri_index

    manifest = {
        "manifest_version": "3.0",
        "data_root": str(tmp_path / "data"),
        "file_type": "h5",
        "records": [
            {"relative_path": "a.h5", "file_id": "a", "shape": [18, 4, 256, 256]},
            {"relative_path": "b.h5", "file_id": "b", "shape": [18, 4, 256, 256]},
        ],
    }
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(manifest))
    (tmp_path / "data").mkdir()

    idx = parse_fastmri_index(str(mp), data_root=str(tmp_path / "data"))
    assert len(idx) == 2
    assert all(r.get("shape") == [18, 4, 256, 256] for r in idx), (
        f"manifest 'shape' must survive parsing; got keys {sorted(idx[0].keys())}"
    )
