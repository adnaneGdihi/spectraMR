"""Unit tests for IO strategies for MRI data loading.

Tests cover:
- BaseIOStrategy abstract class
- FastMRIH5Strategy for H5 files
- NiftiStrategy for NIfTI files
- DicomStrategy for DICOM files
- Format-agnostic data loading
- Error handling
- Sensitivity map loading
"""

from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from spectramr.data.io_strategies import (
    AutoDetectStrategy,
    BartCflStrategy,
    BaseIOStrategy,
    DicomStrategy,
    FastMRIH5Strategy,
    IOStrategyFactory,
    IsmrmrdStrategy,
    MatStrategy,
    NiftiStrategy,
    TwixStrategy,
    load_tensor_from_file,
)


class TestBaseIOStrategy:
    """Test BaseIOStrategy abstract class."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that BaseIOStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseIOStrategy()  # type: ignore

    def test_load_method_required(self) -> None:
        """Test that load method is abstract."""

        class IncompleteStrategy(BaseIOStrategy):
            pass

        with pytest.raises(TypeError):
            IncompleteStrategy()  # type: ignore

    def test_load_sensitivity_default_pt(self) -> None:
        """Test loading sensitivity map from .pt file."""

        class ConcreteStrategy(BaseIOStrategy):
            def load(self, path, metadata=None):
                return {"data": torch.tensor([1, 2, 3])}

        strategy = ConcreteStrategy()

        with patch("torch.load") as mock_load:
            mock_load.return_value = torch.tensor([1, 2, 3])
            result = strategy.load_sensitivity("/path/to/sens.pt")
            assert result is not None

    def test_load_sensitivity_default_npy(self) -> None:
        """Test loading sensitivity map from .npy file."""

        class ConcreteStrategy(BaseIOStrategy):
            def load(self, path, metadata=None):
                return {"data": torch.tensor([1, 2, 3])}

        strategy = ConcreteStrategy()

        with patch("numpy.load") as mock_load:
            mock_load.return_value = np.array([1, 2, 3])
            result = strategy.load_sensitivity("/path/to/sens.npy")
            assert result is not None

    def test_load_sensitivity_unsupported_format(self) -> None:
        """Test that unsupported format raises ValueError."""

        class ConcreteStrategy(BaseIOStrategy):
            def load(self, path, metadata=None):
                return {"data": torch.tensor([1, 2, 3])}

        strategy = ConcreteStrategy()

        with pytest.raises(ValueError):
            strategy.load_sensitivity("/path/to/sens.txt")

    def test_load_sensitivity_npy_float64_downcast_to_float32(self) -> None:
        """dtype guard (2026-07-02): a double-stored real ``.npy`` must load as
        float32, not silently upcast the (float32) model downstream."""

        class ConcreteStrategy(BaseIOStrategy):
            def load(self, path, metadata=None):
                return {"data": torch.tensor([1.0])}

        strategy = ConcreteStrategy()
        with patch("numpy.load") as mock_load:
            mock_load.return_value = np.ones((2, 4), dtype=np.float64)
            result = strategy.load_sensitivity("/path/to/sens.npy")
            assert result.dtype == torch.float32

    def test_load_sensitivity_npy_complex_preserved(self) -> None:
        """Complex sensitivity maps must NOT be coerced to real by the guard."""

        class ConcreteStrategy(BaseIOStrategy):
            def load(self, path, metadata=None):
                return {"data": torch.tensor([1.0])}

        strategy = ConcreteStrategy()
        with patch("numpy.load") as mock_load:
            mock_load.return_value = np.ones((2, 4), dtype=np.complex128)
            result = strategy.load_sensitivity("/path/to/sens.npy")
            assert torch.is_complex(result)


class TestFastMRIH5Strategy:
    """Test FastMRIH5Strategy for H5 file loading."""

    def test_load_with_kspace_and_rss(self) -> None:
        """Test loading H5 file with k-space and RSS reconstruction."""
        strategy = FastMRIH5Strategy()

        mock_kspace = torch.randn(10, 128, 128)
        mock_rss = torch.randn(10, 128, 128)

        with patch("h5py.File") as mock_h5:
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_h5.return_value = mock_file

            mock_file.__getitem__ = MagicMock(
                side_effect=lambda key: {
                    "kspace": MagicMock(__getitem__=MagicMock(return_value=mock_kspace.numpy())),
                    "reconstruction_rss": MagicMock(
                        __getitem__=MagicMock(return_value=mock_rss.numpy())
                    ),
                }.get(key)
            )
            mock_file.__contains__ = MagicMock(
                side_effect=lambda key: key in ["kspace", "reconstruction_rss"]
            )
            mock_file.attrs = {}

            result = strategy.load("/path/to/file.h5")

            assert result is not None
            assert "kspace" in result
            assert "data" in result
            assert "affine" in result

    def test_load_with_kspace_complex(self) -> None:
        """Test loading complex k-space data."""
        strategy = FastMRIH5Strategy()

        # Create complex tensor and view as real
        complex_kspace = torch.randn(10, 128, 128, dtype=torch.complex64)
        real_kspace = torch.view_as_real(complex_kspace)

        with patch("h5py.File") as mock_h5:
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_h5.return_value = mock_file

            mock_file.__getitem__ = MagicMock(
                side_effect=lambda key: {
                    "kspace": MagicMock(__getitem__=MagicMock(return_value=real_kspace.numpy())),
                }.get(key)
            )
            mock_file.__contains__ = MagicMock(side_effect=lambda key: key == "kspace")
            mock_file.attrs = {}

            result = strategy.load("/path/to/file.h5")

            assert result["kspace"] is not None

    def test_load_returns_identity_affine(self) -> None:
        """Test that H5 loading returns identity affine."""
        strategy = FastMRIH5Strategy()

        with patch("h5py.File") as mock_h5:
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_h5.return_value = mock_file

            mock_file.__getitem__ = MagicMock(return_value=None)
            mock_file.__contains__ = MagicMock(return_value=False)
            mock_file.attrs = {}

            result = strategy.load("/path/to/file.h5")

            assert np.allclose(result["affine"], np.eye(4))

    def test_load_multicoil_kspace(self) -> None:
        """Test loading multi-coil k-space."""
        strategy = FastMRIH5Strategy()

        mock_kspace = torch.randn(10, 8, 128, 128)  # (Slices, Coils, H, W)

        with patch("h5py.File") as mock_h5:
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_h5.return_value = mock_file

            mock_file.__getitem__ = MagicMock(
                side_effect=lambda key: {
                    "kspace": MagicMock(__getitem__=MagicMock(return_value=mock_kspace.numpy())),
                }.get(key)
            )
            mock_file.__contains__ = MagicMock(side_effect=lambda key: key == "kspace")
            mock_file.attrs = {}

            result = strategy.load("/path/to/file.h5")

            assert result["kspace"].shape[0] == 10  # Slices preserved

    def test_fastmri_h5_strategy_rejects_nifti_path(self) -> None:
        """IOM-003: a NIfTI path into FastMRIH5Strategy must RAISE, not silently
        delegate to NiftiStrategy (NN#3 / pitfall #9 — no silent fallbacks)."""
        strategy = FastMRIH5Strategy()
        with pytest.raises(ValueError, match="NIfTI path"):
            strategy.load("/tmp/sub-01_T1w.nii.gz")
        with pytest.raises(ValueError, match="NIfTI path"):
            strategy.load("/tmp/sub-01_T1w.nii")


class TestNiftiStrategy:
    """Test NiftiStrategy for NIfTI file loading."""

    def test_load_3d_nifti(self) -> None:
        """Test loading 3D NIfTI file."""
        strategy = NiftiStrategy()

        mock_data = np.random.randn(128, 128, 64)

        with patch("nibabel.load") as mock_load:
            mock_img = MagicMock()
            mock_img.get_fdata.return_value = mock_data
            mock_img.affine = np.eye(4)
            mock_load.return_value = mock_img

            result = strategy.load("/path/to/file.nii.gz")

            assert result["data"] is not None
            assert result["data"].ndim == 4  # Should add channel dimension

    def test_load_4d_nifti(self) -> None:
        """Test loading 4D NIfTI file."""
        strategy = NiftiStrategy()

        mock_data = np.random.randn(128, 128, 64, 3)

        with patch("nibabel.load") as mock_load:
            mock_img = MagicMock()
            mock_img.get_fdata.return_value = mock_data
            mock_img.affine = np.eye(4)
            mock_load.return_value = mock_img

            result = strategy.load("/path/to/file.nii.gz")

            assert result["data"].shape[0] == 3  # Channels

    def test_load_nifti_preserves_affine(self) -> None:
        """Test that affine matrix is preserved."""
        strategy = NiftiStrategy()

        custom_affine = np.array(
            [
                [2, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 1],
            ]
        )

        with patch("nibabel.load") as mock_load:
            mock_img = MagicMock()
            mock_img.get_fdata.return_value = np.random.randn(64, 64, 32)
            mock_img.affine = custom_affine
            mock_load.return_value = mock_img

            result = strategy.load("/path/to/file.nii.gz")

            assert np.allclose(result["affine"], custom_affine)


class TestDicomStrategy:
    """Test DicomStrategy for DICOM loading."""

    def test_load_single_dicom_file(self) -> None:
        """Test loading single DICOM file."""
        strategy = DicomStrategy()

        with patch("pathlib.Path.is_file", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=False):
                with patch.object(strategy, "_load_single_file") as mock_load:
                    mock_load.return_value = {
                        "data": torch.randn(512, 512),
                        "affine": np.eye(4),
                    }

                    result = strategy.load("/path/to/file.dcm")

                    assert result is not None

    def test_load_dicom_directory(self) -> None:
        """Test loading DICOM series from directory."""
        strategy = DicomStrategy()

        with patch("pathlib.Path.is_file", return_value=False):
            with patch("pathlib.Path.is_dir", return_value=True):
                # This will depend on the actual implementation
                # Just test it doesn't crash
                pass

    def test_load_missing_pydicom(self) -> None:
        """Test that missing pydicom raises ImportError."""
        strategy = DicomStrategy()

        with patch("builtins.__import__", side_effect=ImportError):
            with pytest.raises(ImportError):
                strategy.load("/path/to/file.dcm")

    def test_autodetect_routes_ima_to_dicom(self) -> None:
        """Siemens ``.IMA`` (uppercase DICOM export) routes to DicomStrategy.

        The strategy table advertises DicomStrategy for ``.dcm``/``.IMA``, but the
        AutoDetect factory matched only ``.dcm``/``.dicom`` — a ``.IMA`` path silently
        fell through to the NIfTI default (pitfall #16: advertised but unwired). Patch
        DicomStrategy.load so no real DICOM is needed; the routing is what's asserted.
        """
        sentinel = {"reader": "dicom-routed"}
        with patch.object(DicomStrategy, "load", return_value=sentinel) as routed:
            out = AutoDetectStrategy().load("/data/STJUDE_Siemens_SKYRA_3T_08132015.IMA")
        routed.assert_called_once()
        assert out is sentinel


class TestIOStrategyFactory:
    """Test IO strategy factory or selection."""

    def test_fastmri_strategy_selection(self) -> None:
        """Test selecting FastMRI strategy."""
        # The factory pattern varies by implementation
        # This assumes IOStrategyFactory.get() or similar
        try:
            from spectramr.data.io_strategies import IOStrategyFactory

            strategy = IOStrategyFactory.get("fastmri_h5")
            assert isinstance(strategy, FastMRIH5Strategy)
        except ImportError:
            # Factory may not be implemented
            pass


class TestIOStrategyIntegration:
    """Integration tests for IO strategies."""

    def test_load_and_normalize_h5(self) -> None:
        """Test loading and normalizing H5 file."""
        strategy = FastMRIH5Strategy()

        mock_kspace = torch.randn(10, 128, 128)
        mock_rss = torch.randn(10, 128, 128)

        with patch("h5py.File") as mock_h5:
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_h5.return_value = mock_file

            mock_file.__getitem__ = MagicMock(
                side_effect=lambda key: {
                    "kspace": MagicMock(__getitem__=MagicMock(return_value=mock_kspace.numpy())),
                    "reconstruction_rss": MagicMock(
                        __getitem__=MagicMock(return_value=mock_rss.numpy())
                    ),
                }.get(key)
            )
            mock_file.__contains__ = MagicMock(
                side_effect=lambda key: key in ["kspace", "reconstruction_rss"]
            )
            mock_file.attrs = {"shape": (10, 128, 128)}

            result = strategy.load("/path/to/file.h5")

            # Should return dict with proper structure
            assert "data" in result or "kspace" in result

    def test_multiple_strategies_consistency(self) -> None:
        """Test consistency across different strategies."""
        # All strategies should return dict with certain keys
        expected_keys = {"data", "affine"}

        # Test FastMRI
        h5_strategy = FastMRIH5Strategy()

        with patch("h5py.File") as mock_h5:
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_h5.return_value = mock_file
            mock_file.__getitem__ = MagicMock(return_value=None)
            mock_file.__contains__ = MagicMock(return_value=False)
            mock_file.attrs = {}

            h5_result = h5_strategy.load("/path/to/file.h5")

            # Should contain expected keys (at minimum)
            assert "affine" in h5_result

    def test_error_handling_missing_file(self) -> None:
        """Test error handling for missing file."""
        strategy = NiftiStrategy()

        with patch("nibabel.load", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                strategy.load("/path/to/nonexistent.nii.gz")

    def test_error_handling_corrupted_file(self) -> None:
        """Test error handling for corrupted file."""
        strategy = FastMRIH5Strategy()

        with patch("h5py.File", side_effect=OSError("Corrupted file")):
            with pytest.raises(OSError):
                strategy.load("/path/to/corrupted.h5")

    def test_autodetect_unknown_extension_raises_not_nifti(self) -> None:
        """An unrecognised extension must raise, not silently default to NIfTI.

        Regression for the warn-and-degrade fallback (pitfalls #9/#10): an
        extensionless DICOM or a typo'd path was handed to nibabel instead of
        failing loud.
        """
        with pytest.raises(ValueError, match=r"(?i)unrecognised|unknown"):
            AutoDetectStrategy().load("/data/scan.weirdext")


def _write_bart(directory, name: str, arr: np.ndarray):
    """Write a synthetic BART ``.cfl``/``.hdr`` pair (column-major complex64).

    Mirrors BART's on-disk convention: the ``.hdr`` carries a ``# Dimensions``
    comment followed by the space-separated dim vector; the ``.cfl`` is raw
    little-endian complex64 stored in Fortran (column-major) order. Returns the
    ``.cfl`` path.
    """
    dims = list(arr.shape)
    (directory / f"{name}.hdr").write_text("# Dimensions\n" + " ".join(str(d) for d in dims) + "\n")
    arr.astype(np.complex64).flatten(order="F").tofile(directory / f"{name}.cfl")
    return directory / f"{name}.cfl"


class TestBartCflStrategy:
    """BART ``.cfl``/``.hdr`` reader (E2 — non-Cartesian k-space)."""

    def test_reads_complex_array_with_bart_dims(self, tmp_path) -> None:
        """A ``.cfl``/``.hdr`` pair round-trips to a complex tensor + dim vector."""
        rng = np.random.default_rng(0)
        arr = (rng.standard_normal((2, 3, 1, 2)) + 1j * rng.standard_normal((2, 3, 1, 2))).astype(
            np.complex64
        )
        cfl = _write_bart(tmp_path, "k", arr)

        out = BartCflStrategy().load(str(cfl))

        assert out["header"]["bart_dims"] == [2, 3, 1, 2]
        data = out["data"]
        assert torch.is_complex(data)
        assert tuple(data.shape) == (2, 3, 1, 2)
        np.testing.assert_allclose(data.numpy(), arr, rtol=1e-5, atol=1e-5)

    def test_accepts_basename_without_cfl_suffix(self, tmp_path) -> None:
        """BART addresses arrays by basename; the reader accepts that too."""
        arr = np.ones((2, 2), dtype=np.complex64)
        _write_bart(tmp_path, "vol", arr)

        out = BartCflStrategy().load(str(tmp_path / "vol"))
        assert tuple(out["data"].shape) == (2, 2)

    def test_metadata_stamps_format(self, tmp_path) -> None:
        """Provenance: the resolved format is recorded (pitfall #15)."""
        arr = np.zeros((1, 1), dtype=np.complex64)
        cfl = _write_bart(tmp_path, "z", arr)
        out = BartCflStrategy().load(str(cfl))
        assert out["metadata"]["format"] == "bart_cfl"

    def test_missing_hdr_raises_not_silent(self, tmp_path) -> None:
        """No silent fallback (pitfall #9): a ``.cfl`` without its ``.hdr`` raises."""
        (tmp_path / "k.cfl").write_bytes(b"\x00" * 8)
        with pytest.raises((FileNotFoundError, ValueError)):
            BartCflStrategy().load(str(tmp_path / "k.cfl"))

    def test_element_count_mismatch_raises(self, tmp_path) -> None:
        """Header dims must match the ``.cfl`` byte count, else raise."""
        (tmp_path / "k.hdr").write_text("# Dimensions\n2 3 1 2\n")  # expects 12 complex
        (tmp_path / "k.cfl").write_bytes(b"\x00" * 8)  # one complex64 only
        with pytest.raises(ValueError):
            BartCflStrategy().load(str(tmp_path / "k.cfl"))

    def test_factory_resolves_bart(self) -> None:
        """The reader is registered under both ``bart`` and ``bart_cfl``."""
        assert isinstance(IOStrategyFactory.get("bart"), BartCflStrategy)
        assert isinstance(IOStrategyFactory.get("bart_cfl"), BartCflStrategy)

    def test_autodetect_routes_cfl_to_bart(self, tmp_path) -> None:
        """``.cfl`` suffix dispatches to the BART reader, not the NIfTI default."""
        arr = np.ones((2, 2), dtype=np.complex64)
        cfl = _write_bart(tmp_path, "k", arr)
        out = AutoDetectStrategy().load(str(cfl))
        assert torch.is_complex(out["data"])
        assert out["header"]["bart_dims"] == [2, 2]


def _write_mrd(path, n_acq=4, n_coil=2, n_samp=8, traj_dim=2):
    """Write a synthetic ISMRMRD ``.mrd`` HDF5 container.

    Mirrors the real layout: a ``dataset`` group with a structured ``data`` array
    (per-acquisition ``head`` + varlen ``traj`` + varlen interleaved-complex
    ``data``) and an ``xml`` ISMRMRD header. The ``head`` carries only the fields
    the reader consumes (it reads by name, so a real full header works too).
    """
    import h5py

    head_dt = np.dtype(
        [
            ("number_of_samples", "<u2"),
            ("active_channels", "<u2"),
            ("trajectory_dimensions", "<u2"),
        ]
    )
    vlen = h5py.vlen_dtype(np.float32)
    acq_dt = np.dtype([("head", head_dt), ("traj", vlen), ("data", vlen)])
    arr = np.empty(n_acq, dtype=acq_dt)
    for i in range(n_acq):
        arr["head"][i] = (n_samp, n_coil, traj_dim)
        arr["data"][i] = np.arange(2 * n_coil * n_samp, dtype=np.float32) + i
        arr["traj"][i] = np.linspace(-np.pi, np.pi, max(traj_dim * n_samp, 0), dtype=np.float32)
    xml = (
        '<?xml version="1.0"?>'
        '<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">'
        "<encoding><trajectory>radial</trajectory>"
        "<encodedSpace><matrixSize><x>8</x><y>8</y><z>1</z></matrixSize>"
        "</encodedSpace></encoding></ismrmrdHeader>"
    )
    with h5py.File(path, "w") as f:
        g = f.create_group("dataset")
        g.create_dataset("data", data=arr)
        g.create_dataset("xml", data=np.array([xml], dtype=object), dtype=h5py.string_dtype())


class TestIsmrmrdStrategy:
    """ISMRMRD ``.mrd`` raw-acquisition reader (spec E1)."""

    def test_reads_kspace_and_trajectory(self, tmp_path) -> None:
        p = tmp_path / "scan.mrd"
        _write_mrd(p, n_acq=4, n_coil=2, n_samp=8, traj_dim=2)
        out = IsmrmrdStrategy().load(str(p))
        assert torch.is_complex(out["kspace"])
        assert tuple(out["kspace"].shape) == (4, 2, 8)  # [n_acq, n_coil, n_samp]
        assert tuple(out["trajectory"].shape) == (4, 8, 2)  # [n_acq, n_samp, ndim]
        # interleave check: data[i]=arange+i → sample(0,0)=re i, im 1+i; i=0 → 0+1j
        assert out["kspace"][0, 0, 0] == 0 + 1j

    def test_parses_xml_encoding_header(self, tmp_path) -> None:
        p = tmp_path / "scan.mrd"
        _write_mrd(p)
        hdr = IsmrmrdStrategy().load(str(p))["header"]
        assert hdr["trajectory"] == "radial"
        assert hdr["encoded_matrix"] == [8, 8, 1]

    def test_no_trajectory_key_when_zero_dims(self, tmp_path) -> None:
        p = tmp_path / "cart.mrd"
        _write_mrd(p, traj_dim=0)
        out = IsmrmrdStrategy().load(str(p))
        assert "trajectory" not in out

    def test_metadata_stamps_format(self, tmp_path) -> None:
        p = tmp_path / "scan.mrd"
        _write_mrd(p)
        out = IsmrmrdStrategy().load(str(p))
        assert out["metadata"]["format"] == "ismrmrd"

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            IsmrmrdStrategy().load(str(tmp_path / "nope.mrd"))

    def test_non_ismrmrd_hdf5_raises(self, tmp_path) -> None:
        import h5py

        p = tmp_path / "bad.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("foo", data=np.zeros(3))
        with pytest.raises(ValueError, match="No ISMRMRD group"):
            IsmrmrdStrategy().load(str(p))

    def test_factory_and_autodetect(self, tmp_path) -> None:
        assert isinstance(IOStrategyFactory.get("ismrmrd"), IsmrmrdStrategy)
        p = tmp_path / "s.mrd"
        _write_mrd(p)
        out = AutoDetectStrategy().load(str(p))
        assert torch.is_complex(out["kspace"])

    def test_autodetect_routes_ismrmrd_in_h5_to_ismrmrd(self, tmp_path) -> None:
        """An ISMRMRD container saved as .h5 (kasper monitored spiral) must route
        to IsmrmrdStrategy, not FastMRIH5Strategy — sniffed by structure."""
        p = tmp_path / "SPIFI_monitored.h5"
        _write_mrd(p)  # writes the dataset/data + dataset/xml ISMRMRD structure
        assert IsmrmrdStrategy.is_ismrmrd_h5(str(p)) is True
        out = AutoDetectStrategy().load(str(p))
        assert torch.is_complex(out["kspace"])  # the ISMRMRD path fired

    def test_plain_fastmri_h5_is_not_sniffed_as_ismrmrd(self, tmp_path) -> None:
        import h5py

        p = tmp_path / "fastmri.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("kspace", data=np.zeros((2, 2, 4, 4), dtype=np.complex64))
        assert IsmrmrdStrategy.is_ismrmrd_h5(str(p)) is False


def _write_mat_v73(path):
    """Write a synthetic MATLAB-v7.3 (HDF5) ``.mat``: a complex-compound array, a
    real array, and a nested struct group — the osu_4dflow_cine shape in miniature."""
    import h5py

    comp = np.zeros((2, 3), dtype=[("real", "<f4"), ("imag", "<f4")])
    comp["real"] = 1.0
    comp["imag"] = 2.0
    with h5py.File(path, "w") as f:
        f.create_dataset("kdata", data=comp)  # complex compound
        f.create_dataset("scale", data=np.array([[5.0]]))
        g = f.create_group("D")  # MATLAB struct → HDF5 group
        g.create_dataset("TE", data=np.array([[3.0]]))  # codespell:ignore TE


class TestMatStrategy:
    """MATLAB v7.3 (and v5) ``.mat`` reader (spec — 4D-flow / OCMR)."""

    def test_reads_v73_complex_and_recurses_structs(self, tmp_path) -> None:
        p = tmp_path / "flow.mat"
        _write_mat_v73(p)
        out = MatStrategy().load(str(p))
        variables = out["variables"]
        assert torch.is_complex(variables["kdata"])
        assert variables["kdata"][0, 0] == 1 + 2j
        assert "D" in variables and "TE" in variables["D"]  # nested struct recursed  # codespell:ignore TE
        assert out["metadata"]["format"] == "mat_v73"

    def test_mat_key_selects_nested_variable(self, tmp_path) -> None:
        p = tmp_path / "flow.mat"
        _write_mat_v73(p)
        out = MatStrategy().load(str(p), metadata={"mat_key": "D.TE"})  # codespell:ignore TE
        assert float(out["data"].reshape(-1)[0]) == 3.0

    def test_mat_key_unknown_raises(self, tmp_path) -> None:
        p = tmp_path / "flow.mat"
        _write_mat_v73(p)
        with pytest.raises(KeyError):
            MatStrategy().load(str(p), metadata={"mat_key": "D.NOPE"})

    def test_skips_refs_table(self, tmp_path) -> None:
        import h5py

        p = tmp_path / "r.mat"
        with h5py.File(p, "w") as f:
            f.create_dataset("x", data=np.array([[1.0]]))
            f.create_group("#refs#")  # MATLAB internal reference table
        out = MatStrategy().load(str(p))
        assert "#refs#" not in out["variables"]
        assert "x" in out["variables"]

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            MatStrategy().load(str(tmp_path / "nope.mat"))

    def test_factory_and_autodetect(self, tmp_path) -> None:
        assert isinstance(IOStrategyFactory.get("mat"), MatStrategy)
        p = tmp_path / "a.mat"
        _write_mat_v73(p)
        out = AutoDetectStrategy().load(str(p))
        assert "kdata" in out["variables"]


def _write_twix_vd(path, *, field=2.89362, coils=4, n_samp=32, n_acq=2) -> np.ndarray:
    """Write a minimal single-measurement VD TWIX ``.dat`` and return its k-space."""
    asc = (
        f'<ParamLong."iMaxNoOfRxChannels">  {{ {coils}  }}\n'
        "### ASCCONV BEGIN object=MrProtDataImpl version=1 ###\n"
        f"sProtConsistencyInfo.flNominalB0\t = \t{field}\n"
        'tSequenceFileName\t = \t"%SiemensSeq%\\trufi"\n'
        f"sKSpace.lBaseResolution\t = \t{n_samp}\n"
        f"sKSpace.lPhaseEncodingLines\t = \t{n_acq}\n"
        "### ASCCONV END ###\n"
    ).encode("latin-1")

    kspace = (
        np.arange(n_acq * coils * n_samp).reshape(n_acq, coils, n_samp)
        + 1j * np.arange(n_acq * coils * n_samp).reshape(n_acq, coils, n_samp)
    ).astype(np.complex64)

    def _scan(data, eval0=0):  # data: [coils, n_samp]
        head = bytearray(192)
        struct.pack_into("<I", head, 40, eval0)  # eval mask
        struct.pack_into("<H", head, 48, data.shape[1])  # samples
        struct.pack_into("<H", head, 50, data.shape[0])  # channels
        parts = [bytes(head)]
        for c in range(data.shape[0]):
            ch = bytearray(32)
            struct.pack_into("<H", ch, 24, c)
            parts.append(bytes(ch) + data[c].astype(np.complex64).tobytes())
        return b"".join(parts)

    scans = b"".join(_scan(kspace[i]) for i in range(n_acq))
    scans += _scan(np.zeros((1, 16), np.complex64), eval0=0x1)  # ACQEND terminator
    hdr_len = 4 + len(asc)
    body = struct.pack("<I", hdr_len) + asc + scans
    offset = 8 + 152
    entry = struct.pack("<IIQQ64s64s", 1, 1, offset, len(body), b"pat", b"prot")
    path.write_bytes(struct.pack("<II", 0, 1) + entry + body)
    return kspace


class TestTwixStrategy:
    """Siemens TWIX ``.dat`` reader — delegates to ``spectramr.data.twix.read_twix``."""

    def test_reads_kspace_as_complex_torch_tensor(self, tmp_path) -> None:
        p = tmp_path / "meas.dat"
        expected = _write_twix_vd(p, coils=4, n_samp=32, n_acq=2)
        out = TwixStrategy().load(str(p))
        assert torch.is_tensor(out["kspace"]) and torch.is_complex(out["kspace"])
        assert tuple(out["kspace"].shape) == (2, 4, 32)
        np.testing.assert_allclose(out["kspace"].numpy(), expected)
        # the standard IO contract: 'data' mirrors 'kspace' for a raw-k-space reader
        assert out["data"] is out["kspace"] or torch.equal(out["data"], out["kspace"])

    def test_metadata_and_header_stamped(self, tmp_path) -> None:
        p = tmp_path / "meas.dat"
        _write_twix_vd(p, field=0.55, coils=4)
        out = TwixStrategy().load(str(p))
        assert out["metadata"]["format"] == "twix"
        assert out["header"]["field_strength_t"] == pytest.approx(0.55)
        assert out["header"]["n_coils"] == 4
        assert out["header"]["sequence"] == "trufi"

    def test_factory_resolves_twix(self) -> None:
        assert isinstance(IOStrategyFactory.get("twix"), TwixStrategy)
        assert isinstance(IOStrategyFactory.get("dat"), TwixStrategy)

    def test_autodetect_routes_dat_to_twix(self, tmp_path) -> None:
        p = tmp_path / "meas.dat"
        _write_twix_vd(p)
        out = AutoDetectStrategy().load(str(p))
        assert out["metadata"]["format"] == "twix"

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises((FileNotFoundError, OSError)):
            TwixStrategy().load(str(tmp_path / "nope.dat"))


class TestLoadTensorFromFileDtype:
    """dtype guard for the module-level ``load_tensor_from_file`` (2026-07-02)."""

    def test_npy_real_float64_downcast(self, tmp_path) -> None:
        from spectramr.data.io_strategies import load_tensor_from_file

        p = tmp_path / "x.npy"
        np.save(p, np.ones((3, 3), dtype=np.float64))
        t = load_tensor_from_file(str(p))
        assert t.dtype == torch.float32

    def test_npy_complex_preserved(self, tmp_path) -> None:
        from spectramr.data.io_strategies import load_tensor_from_file

        p = tmp_path / "xc.npy"
        np.save(p, np.ones((3, 3), dtype=np.complex128))
        t = load_tensor_from_file(str(p))
        assert torch.is_complex(t)

    def test_npy_float32_unchanged(self, tmp_path) -> None:
        from spectramr.data.io_strategies import load_tensor_from_file

        p = tmp_path / "x32.npy"
        np.save(p, np.ones((3, 3), dtype=np.float32))
        t = load_tensor_from_file(str(p))
        assert t.dtype == torch.float32


class TestFloat64DowncastWS6:
    """WS6: extend the real-float64→float32 guard beyond ``.npy`` to the ``.pt``
    and HDF5 read paths (load_tensor_from_file, load_sensitivity ``.pt``), which
    previously let a double-stored recon target silently upcast the model.
    Complex stores are still preserved."""

    def test_load_tensor_pt_float64_downcast(self, tmp_path):
        p = tmp_path / "t.pt"
        torch.save(torch.ones(2, 3, dtype=torch.float64), p)
        assert load_tensor_from_file(p).dtype == torch.float32

    def test_load_tensor_h5_float64_downcast(self, tmp_path):
        import h5py

        p = tmp_path / "t.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("data", data=np.ones((2, 3), dtype=np.float64))
        assert load_tensor_from_file(p, h5_keys=("data",)).dtype == torch.float32

    def test_load_tensor_npy_complex_preserved(self, tmp_path):
        p = tmp_path / "t.npy"
        np.save(p, np.ones((2, 3), dtype=np.complex128))
        assert torch.is_complex(load_tensor_from_file(p))

    def test_load_tensor_float32_unchanged(self, tmp_path):
        p = tmp_path / "t.pt"
        torch.save(torch.ones(2, 3, dtype=torch.float32), p)
        assert load_tensor_from_file(p).dtype == torch.float32

    def test_load_sensitivity_pt_float64_downcast(self, tmp_path):
        class ConcreteStrategy(BaseIOStrategy):
            def load(self, path, metadata=None):
                return {"data": torch.tensor([1.0])}

        p = tmp_path / "sens.pt"
        torch.save(torch.ones(2, 4, dtype=torch.float64), p)
        assert ConcreteStrategy().load_sensitivity(str(p)).dtype == torch.float32


class TestLoadReferenceVolumeSkipsKspace:
    """``load()`` reads every dataset in the file. A fastMRI brain multicoil volume is
    ~500 MB of k-space wrapping a ~13 MB reconstruction -- reading it to export the
    reference is a ~40x read amplification across the annotated cohort, for bytes that
    are immediately discarded."""

    def _h5(self, tmp_path, *, rss=True, kspace=True, header=True):
        import h5py

        p = tmp_path / "file_brain_AXT2_200_2000019.h5"
        with h5py.File(p, "w") as f:
            if rss:
                f.create_dataset(
                    "reconstruction_rss",
                    data=np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4),
                )
            if kspace:
                f.create_dataset("kspace", data=np.ones((2, 2, 3, 4), dtype=np.complex64))
            if header:
                f.create_dataset("ismrmrd_header", data=np.bytes_(b"<ismrmrdHeader/>"))
        return p

    def test_it_returns_the_reference_and_the_header(self, tmp_path) -> None:
        out = FastMRIH5Strategy.load_reference_volume(self._h5(tmp_path))
        assert tuple(out["data"].shape) == (2, 3, 4)
        assert out["data"].dtype is torch.float32
        assert b"ismrmrdHeader" in bytes(out["header"])

    def test_it_does_not_read_kspace(self, tmp_path) -> None:
        """The point of the method: the k-space is never touched, so it is not in the
        result at all."""
        out = FastMRIH5Strategy.load_reference_volume(self._h5(tmp_path))
        assert "kspace" not in out

    def test_it_works_on_a_file_with_no_kspace(self, tmp_path) -> None:
        out = FastMRIH5Strategy.load_reference_volume(self._h5(tmp_path, kspace=False))
        assert tuple(out["data"].shape) == (2, 3, 4)

    def test_a_file_with_no_reference_raises(self, tmp_path) -> None:
        """No reconstruction_rss means nothing to segment AND nothing to grade against --
        a silent skip would shrink the cohort without saying so."""
        with pytest.raises(KeyError, match="no clean reference"):
            FastMRIH5Strategy.load_reference_volume(self._h5(tmp_path, rss=False))

    def test_the_reference_keeps_slices_on_axis_zero(self, tmp_path) -> None:
        """reconstruction_rss is [S, H, W] on disk and load({"slice_index": i}) indexes
        axis 0. spectramr.data.nifti_export writes its NIfTIs on that contract."""
        p = self._h5(tmp_path)
        vol = FastMRIH5Strategy.load_reference_volume(p)["data"]
        one = FastMRIH5Strategy().load(str(p), {"slice_index": 1})["data"]
        assert torch.equal(vol[1], one)


class TestReadH5Kspace:
    """``read_h5_kspace`` is the one HDF5 k-space read both routes call (#1757):
    ``FastMRIH5Strategy.load`` and the M4Raw loader."""

    @staticmethod
    def _file(tmp_path, arr):
        import h5py

        p = tmp_path / "k.h5"
        with h5py.File(str(p), "w") as f:
            f.create_dataset("kspace", data=arr)
        return p

    @staticmethod
    def _arr():
        rng = np.random.default_rng(0)
        return (rng.standard_normal((4, 2, 8, 8)) + 1j * rng.standard_normal((4, 2, 8, 8))).astype(
            np.complex64
        )

    def test_whole_read_is_the_array(self, tmp_path) -> None:
        import h5py

        from spectramr.data.io_strategies import read_h5_kspace

        arr = self._arr()
        with h5py.File(str(self._file(tmp_path, arr)), "r") as f:
            assert np.array_equal(read_h5_kspace(f["kspace"]), arr)

    def test_slice_read_is_that_slice_with_the_axis_dropped(self, tmp_path) -> None:
        import h5py

        from spectramr.data.io_strategies import read_h5_kspace

        arr = self._arr()
        with h5py.File(str(self._file(tmp_path, arr)), "r") as f:
            out = read_h5_kspace(f["kspace"], 2)
        assert out.shape == (2, 8, 8)
        assert np.array_equal(out, arr[2])

    def test_out_of_range_raises_naming_the_count(self, tmp_path) -> None:
        """Planted violation: h5py's own error names no slice count."""
        import h5py

        from spectramr.data.io_strategies import read_h5_kspace

        with (
            h5py.File(str(self._file(tmp_path, self._arr())), "r") as f,
            pytest.raises(IndexError, match=r"slice_index=4.*4 slice"),
        ):
            read_h5_kspace(f["kspace"], 4)

    def test_negative_index_is_refused(self, tmp_path) -> None:
        """A negative index would wrap to the last slice and read as success."""
        import h5py

        from spectramr.data.io_strategies import read_h5_kspace

        with (
            h5py.File(str(self._file(tmp_path, self._arr())), "r") as f,
            pytest.raises(IndexError, match="slice_index=-1"),
        ):
            read_h5_kspace(f["kspace"], -1)

    def test_non_integer_index_is_refused(self, tmp_path) -> None:
        import h5py

        from spectramr.data.io_strategies import read_h5_kspace

        with (
            h5py.File(str(self._file(tmp_path, self._arr())), "r") as f,
            pytest.raises(TypeError),
        ):
            read_h5_kspace(f["kspace"], 1.5)  # type: ignore[arg-type]

    def test_numpy_integer_index_is_accepted(self, tmp_path) -> None:
        import h5py

        from spectramr.data.io_strategies import read_h5_kspace

        arr = self._arr()
        with h5py.File(str(self._file(tmp_path, arr)), "r") as f:
            assert np.array_equal(read_h5_kspace(f["kspace"], np.int64(1)), arr[1])

    def test_fastmri_strategy_reads_one_slice_of_a_real_file(self, tmp_path) -> None:
        """The slice-aware branch of ``FastMRIH5Strategy.load`` on disk, not a
        mock: the slice, the dropped axis, and the stamped ``slice_index``."""
        arr = self._arr()
        path = self._file(tmp_path, arr)
        whole = FastMRIH5Strategy().load(str(path))
        assert tuple(whole["kspace"].shape) == (4, 2, 8, 8)
        one = FastMRIH5Strategy().load(str(path), metadata={"slice_index": 3})
        assert tuple(one["kspace"].shape) == (2, 8, 8)
        assert np.array_equal(one["kspace"].numpy(), arr[3])
        assert one["metadata"]["slice_index"] == 3

    def test_fastmri_strategy_out_of_range_slice_raises(self, tmp_path) -> None:
        path = self._file(tmp_path, self._arr())
        with pytest.raises(IndexError, match="4 slice"):
            FastMRIH5Strategy().load(str(path), metadata={"slice_index": 7})


class TestLoadH5DatasetIfPresent:
    """A strict named read: the dataset asked for, or ``None``, never another one."""

    def test_returns_the_named_dataset(self, tmp_path) -> None:
        h5py = pytest.importorskip("h5py")
        from spectramr.data.io_strategies import load_h5_dataset_if_present

        p = tmp_path / "case.h5"
        mask = np.zeros((8, 8), dtype=bool)
        mask[:, ::2] = True
        with h5py.File(p, "w") as f:
            f.create_dataset("kspace", data=np.ones((2, 8, 8), dtype=np.float32))
            f.create_dataset("mask", data=mask)
        out = load_h5_dataset_if_present(p, "mask")
        assert out is not None and out.dtype == torch.bool
        assert torch.equal(out, torch.from_numpy(mask))

    def test_an_absent_key_is_none_even_when_other_keys_exist(self, tmp_path) -> None:
        """The hazard ``load_tensor_from_file(h5_keys=...)`` has: it would return 'kspace'."""
        h5py = pytest.importorskip("h5py")
        from spectramr.data.io_strategies import load_h5_dataset_if_present, load_tensor_from_file

        p = tmp_path / "case.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("kspace", data=np.ones((2, 8, 8), dtype=np.float32))
        assert load_h5_dataset_if_present(p, "mask") is None
        # The preference reader falls back to the first key; that is why it is
        # the wrong tool for a sidecar.
        assert load_tensor_from_file(p, h5_keys=("mask",)).shape == (2, 8, 8)

    def test_a_non_hdf5_path_is_none_without_touching_the_file(self, tmp_path) -> None:
        from spectramr.data.io_strategies import load_h5_dataset_if_present

        assert load_h5_dataset_if_present(tmp_path / "does_not_exist.npy", "mask") is None
        assert load_h5_dataset_if_present(tmp_path / "vol.nii.gz", "mask") is None

    def test_float64_is_downcast_like_every_other_reader(self, tmp_path) -> None:
        h5py = pytest.importorskip("h5py")
        from spectramr.data.io_strategies import load_h5_dataset_if_present

        p = tmp_path / "case.hdf5"
        with h5py.File(p, "w") as f:
            f.create_dataset("mask", data=np.ones((4, 4), dtype=np.float64))
        out = load_h5_dataset_if_present(p, "mask")
        assert out is not None and out.dtype == torch.float32


def test_save_nifti_round_trips_through_lazy_load_nifti(tmp_path):
    import numpy as np

    from spectramr.data.io_strategies import lazy_load_nifti, save_nifti

    vol = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    save_nifti(tmp_path / "v.nii.gz", vol, affine)
    img = lazy_load_nifti(tmp_path / "v.nii.gz")
    assert np.array_equal(np.asarray(img.dataobj), vol)
    assert np.array_equal(img.affine, affine)
