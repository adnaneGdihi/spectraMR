"""Tests for the EDA sample loader (directory-scan-first, format-robust)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spectramr.data.eda.catalog import discover, discover_physical
from spectramr.data.eda.loader import DatasetSample, sample


def _entry(manifest_path: Path, tmp_path: Path, dataset_id: str):
    return {e.dataset_id: e for e in discover(manifest_path.parent, tmp_path)}[dataset_id]


def test_sample_present_kspace(present_kspace_manifest, tmp_path):
    entry = _entry(present_kspace_manifest, tmp_path, "tiny_kspace")
    s = sample(entry, budget=2)
    assert isinstance(s, DatasetSample)
    assert len(s.images) >= 1 and s.images[0].ndim == 2
    assert len(s.kspace) >= 1 and s.kspace[0].is_complex()


def test_sample_absent_returns_empty(absent_external_manifest, tmp_path):
    entry = _entry(absent_external_manifest, tmp_path, "calgary_campinas")
    s = sample(entry, budget=2)
    assert s.images == [] and s.kspace == []
    assert any("absent" in n.lower() or "no scan dir" in n.lower() for n in s.notes)


def test_dry_run_budget_zero_skips_load(present_kspace_manifest, tmp_path):
    entry = _entry(present_kspace_manifest, tmp_path, "tiny_kspace")
    s = sample(entry, budget=0)
    assert s.images == [] and any("dry-run" in n for n in s.notes)


def test_directory_scan_fallback_walks_to_ancestor(tmp_path):
    """data_root nested below where files actually sit → loader walks up to find them."""
    import h5py
    real = tmp_path / "databases" / "m4raw" / "data" / "multicoil_train"
    real.mkdir(parents=True)
    rng = np.random.default_rng(1)
    img = rng.standard_normal((2, 16, 16)) + 1j * rng.standard_normal((2, 16, 16))
    ksp = np.fft.fftshift(np.fft.fft2(img, axes=(-2, -1)), axes=(-2, -1)).astype(np.complex64)
    with h5py.File(real / "scan0.h5", "w") as f:
        f.create_dataset("kspace", data=ksp)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    mpath = manifests / "m4raw_multicoil_train_kspace.json"
    mpath.write_text(json.dumps({
        "dataset_name": "m4raw_multicoil_train_kspace",
        "data_root": str(real / "multicoil_train_image" / "kspace"),  # missing nested dir
        "file_type": "h5",
        "records": [{"relative_path": "x.pt"}],  # bogus record → only ancestor scan saves it
    }))
    entry = _entry(mpath, tmp_path, "m4raw_multicoil_train_kspace")
    s = sample(entry, budget=2)
    assert s.scan_dir is not None and s.scan_dir.name == "multicoil_train"
    assert len(s.kspace) >= 1


def test_h5_mid_slice_index_is_header_only(tmp_path):
    """The OOM guard: we compute the middle slice from the header so the strategy reads one
    slice lazily instead of the full multi-slice volume."""
    import h5py

    from spectramr.data.eda.loader import _h5_mid_slice

    p = tmp_path / "vol.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("kspace", data=np.zeros((5, 2, 16, 16), np.complex64))
    assert _h5_mid_slice(p) == 2  # middle of 5 slices

    # A 2-D (H, W) single image has no slice axis → None (load whole, never slice a row).
    p2 = tmp_path / "img.h5"
    with h5py.File(p2, "w") as f:
        f.create_dataset("kspace", data=np.zeros((16, 16), np.complex64))
    assert _h5_mid_slice(p2) is None


def test_large_h5_with_slice_axis_is_lazily_loaded(tmp_path, monkeypatch):
    """A multi-GB h5 with a slice axis is read one slice at a time (lazy h5py indexing), so the
    size guard must NOT skip it — otherwise large sliceable datasets (kasper, mridata_org)
    produce only a card and no figures."""
    import h5py

    from spectramr.data.eda import loader

    p = tmp_path / "big.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("kspace", data=np.zeros((4, 2, 16, 16), np.complex64))
    monkeypatch.setattr(loader, "_MAX_VOXEL_BYTES", 1)  # treat as oversized without writing GBs
    s = DatasetSample()
    loader._load_one(p, s)
    assert len(s.kspace) >= 1
    assert not any("skipped voxel load" in n for n in s.notes)


def test_large_non_sliceable_file_is_skipped(tmp_path, monkeypatch):
    """A format we must read wholesale (npy) stays size-gated — the guard still protects RAM."""
    from spectramr.data.eda import loader

    p = tmp_path / "big.npy"
    np.save(p, np.zeros((16, 16), np.float32))
    monkeypatch.setattr(loader, "_MAX_VOXEL_BYTES", 1)
    s = DatasetSample()
    loader._load_one(p, s)
    assert s.images == [] and any("skipped voxel load" in n for n in s.notes)


def test_load_one_noncartesian_skips_cartesian_recon(tmp_path):
    """Radial/spiral k-space must NOT be ifft2c'd into a fake image — Cartesian recon of
    non-Cartesian samples produces the misleading 'strip' montages. The raw k-space is kept
    (for the logmag view); no recon image is emitted."""
    import h5py

    from spectramr.data.eda import loader

    p = tmp_path / "spokes.h5"
    rng = np.random.default_rng(0)
    ksp = (rng.standard_normal((40, 89)) + 1j * rng.standard_normal((40, 89))).astype(np.complex64)
    with h5py.File(p, "w") as f:
        f.create_dataset("kspace", data=ksp)
    s = DatasetSample()
    loader._load_one(p, s, cartesian=False)
    assert len(s.kspace) >= 1
    assert s.images == [] and s.complex_images == []


def test_load_one_cartesian_still_recons(tmp_path):
    """Cartesian k-space keeps the ifft2c recon (default behavior is unchanged)."""
    import h5py

    from spectramr.data.eda import loader

    p = tmp_path / "cart.h5"
    rng = np.random.default_rng(0)
    ksp = (rng.standard_normal((2, 40, 40)) + 1j * rng.standard_normal((2, 40, 40))).astype(np.complex64)
    with h5py.File(p, "w") as f:
        f.create_dataset("kspace", data=ksp)
    s = DatasetSample()
    loader._load_one(p, s, cartesian=True)
    assert len(s.kspace) >= 1 and len(s.images) >= 1


def _write_cfl(base: Path, arr: np.ndarray) -> None:
    """Write a BART .cfl/.hdr pair (column-major payload + dim header)."""
    np.asarray(arr, np.complex64).ravel(order="F").tofile(base.with_suffix(".cfl"))
    base.with_suffix(".hdr").write_text("# Dimensions\n" + " ".join(map(str, arr.shape)) + "\n")


def test_cfl_central_slice_picks_two_largest_dims(tmp_path):
    from spectramr.data.eda.loader import _cfl_central_slice

    base = tmp_path / "vol0001_vis1"
    rng = np.random.default_rng(0)
    arr = (rng.standard_normal((32, 24, 3, 4)) + 1j * rng.standard_normal((32, 24, 3, 4))).astype(np.complex64)
    _write_cfl(base, arr)
    sl = _cfl_central_slice(base.with_suffix(".cfl"))
    assert sl is not None and sl.is_complex()
    assert tuple(sl.shape) == (32, 24)  # two largest BART dims; coils/slices reduced to centre


def test_large_cfl_is_lazily_loaded_as_kspace(tmp_path, monkeypatch):
    """A multi-GB BART .cfl is memmap-sliced (its column-major central plane is contiguous),
    so the size guard must NOT skip it — radial datasets (cardiac_radial_realtime_bssfp,
    multiecho_radial_b0_r2star) otherwise produced only a card despite data on disk."""
    from spectramr.data.eda import loader

    base = tmp_path / "vol0002_vis1"
    rng = np.random.default_rng(1)
    arr = (rng.standard_normal((40, 30, 2, 4)) + 1j * rng.standard_normal((40, 30, 2, 4))).astype(np.complex64)
    _write_cfl(base, arr)
    monkeypatch.setattr(loader, "_MAX_VOXEL_BYTES", 1)  # treat as oversized without writing GBs
    s = DatasetSample()
    loader._load_one(base.with_suffix(".cfl"), s, cartesian=False)
    assert len(s.kspace) >= 1 and s.kspace[0].is_complex()   # raw k-space kept (logmag view)
    # non-Cartesian .cfl now ALSO gets a best-effort adjoint-NUFFT recon (golden-angle assumed),
    # labeled in the notes — no longer a meaningless ifft2c 'strip', but a real gridded image.
    assert len(s.images) >= 1
    assert any("nufft" in n.lower() for n in s.notes)
    assert not any("skipped voxel load" in nn for nn in s.notes)


def test_png_loading(tmp_path):
    from PIL import Image
    root = tmp_path / "databases" / "imgset"
    root.mkdir(parents=True)
    Image.fromarray((np.random.default_rng(2).random((24, 24)) * 255).astype(np.uint8)).save(root / "a.png")
    physical = discover_physical(tmp_path / "databases", covered=[])
    entry = {e.dataset_id: e for e in physical}["disk_imgset"]
    s = sample(entry, budget=2)
    assert len(s.images) >= 1 and s.images[0].shape == (24, 24)


def test_discover_physical_finds_uncovered_dirs(tmp_path):
    import h5py
    db = tmp_path / "databases"
    # paired NIfTI-less stand-in: source/target with png
    from PIL import Image
    for sub in ("source", "target"):
        (db / "mock_paired" / sub).mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(db / "mock_paired" / sub / "s0.png")
    (db / "kasper").mkdir(parents=True)
    with h5py.File(db / "kasper" / "spiral0.h5", "w") as f:
        f.create_dataset("kspace", data=np.zeros((1, 8, 8), np.complex64))
    entries = {e.dataset_id: e for e in discover_physical(db, covered=[])}
    assert "disk_mock_paired" in entries and entries["disk_mock_paired"].modality == "paired"
    assert "disk_kasper" in entries and entries["disk_kasper"].modality == "noncartesian"


def test_load_one_skips_degenerate_1d_read_without_crashing(tmp_path):
    """A reader that returns a 1-D vector (the osi2one 'wrong key / flattened' bug) must degrade
    to a note — never feed a (N,) array to render_montage / np.gradient downstream. The guard
    keeps the EDA producing a card instead of crashing the figure suite."""
    import h5py

    from spectramr.data.eda import loader

    p = tmp_path / "osi2one.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("kspace", data=np.zeros((150,), np.complex64))  # 1-D — not a plane
    s = DatasetSample()
    loader._load_one(p, s)
    assert s.kspace == [] and s.images == []
    assert any("malformed" in n.lower() or "shape" in n.lower() for n in s.notes)


def test_sample_scans_extracted_sibling_when_data_root_is_raw(tmp_path):
    """End-to-end: a stub-style entry whose data_root points at an empty '<x>/raw' but whose data
    sits in the sibling '<x>/extracted' renders via the shared resolve_scan_dir (data_relocated)."""
    import h5py

    base = tmp_path / "databases" / "external" / "ds_x"
    (base / "raw").mkdir(parents=True)
    rng = np.random.default_rng(0)
    ksp = (rng.standard_normal((2, 16, 16)) + 1j * rng.standard_normal((2, 16, 16))).astype(np.complex64)
    (base / "extracted").mkdir()
    with h5py.File(base / "extracted" / "scan0.h5", "w") as f:
        f.create_dataset("kspace", data=ksp)
    manifests = tmp_path / "manifests" / "external"
    manifests.mkdir(parents=True)
    mpath = manifests / "ds_x.json"
    mpath.write_text(json.dumps({
        "dataset_name": "ds_x",
        "data_root": "databases/external/ds_x/raw",
        "file_type": "h5",
        "records": [],
        "status": "not_downloaded",
        "source": {"role": "RAW", "raw_kspace": True},
    }))
    entry = _entry(mpath, tmp_path, "ds_x")
    s = sample(entry, budget=2)
    assert s.scan_dir is not None and s.scan_dir.name == "extracted"
    assert len(s.kspace) >= 1


def test_load_one_squeezes_singleton_coil_dim_to_a_plane(tmp_path, monkeypatch):
    """ISMRMRD raw acquisitions read back **whole** as (n_acq, 1, n_samples) — no slice axis is
    recognised, so the strategy returns the full array. The singleton coil dim must be squeezed so
    the (n_acq, n_samples) acquired-samples matrix renders as a k-space plane (osi2one
    (2994,1,150) / (3840,1,48)) instead of being dropped as 'malformed'. We force the whole-read
    path (``_h5_mid_slice`` → None) to model that ISMRMRD case."""
    import h5py

    from spectramr.data.eda import loader

    monkeypatch.setattr(loader, "_h5_mid_slice", lambda _p: None)  # no slice axis → read whole
    p = tmp_path / "raw_acq.h5"
    ksp = (np.random.default_rng(0).standard_normal((24, 1, 8))
           + 1j * np.random.default_rng(1).standard_normal((24, 1, 8))).astype(np.complex64)
    with h5py.File(p, "w") as f:
        f.create_dataset("kspace", data=ksp)
    s = DatasetSample()
    loader._load_one(p, s, cartesian=False)  # raw acquisition → non-Cartesian, no ifft2c
    assert len(s.kspace) == 1
    assert s.kspace[0].shape == (24, 8)          # singleton coil squeezed away
    assert s.images == [] and not any("malformed" in n.lower() for n in s.notes)


def test_load_one_guards_1d_npy_image_from_montage_crash(tmp_path):
    """A 1-D .npy (the mrf_100mt_optimum fingerprint-dictionary vector, shape (1423170,)) must
    be dropped with a note — never appended as an 'image' where render_montage(imshow) crashes
    with 'Invalid shape (N,)'. The .pt/.npy/.png image-append paths need the same _is_plane guard
    as the k-space path."""
    from spectramr.data.eda import loader

    p = tmp_path / "fingerprint.npy"
    np.save(p, np.zeros((4096,), np.float32))  # 1-D dictionary vector, not an image
    s = DatasetSample()
    loader._load_one(p, s)
    assert s.images == []
    assert any("malformed" in n.lower() or "shape" in n.lower() for n in s.notes)


def test_load_one_extracts_main_array_from_mat_variables_dict(tmp_path):
    """A .mat is a *container of named variables*: MatStrategy returns ``data`` as the whole
    variables dict (no mat_key passed), so the EDA's ``torch.is_tensor(data)`` check missed it and
    every .mat dataset showed 'no readable files' (dualstage_bssfp_cine 51 files, osu_4dflow). The
    loader must pull the largest ≥2-D array out of the variables dict and render it."""
    import h5py

    from spectramr.data.eda import loader

    p = tmp_path / "CropSax1.mat"  # v7.3 .mat is HDF5 → MatStrategy reads via h5py
    with h5py.File(p, "w") as f:
        f.create_dataset("scalar", data=np.array([3.0], np.float32))      # noise variable
        f.create_dataset("cine", data=np.zeros((12, 20, 20), np.float32))  # the main array
    s = DatasetSample()
    loader._load_one(p, s)
    assert len(s.images) == 1 and s.images[0].shape == (20, 20)  # a 2-D slice of the cine volume
    assert not any("no readable" in n.lower() or "malformed" in n.lower() for n in s.notes)


def test_largest_array_picks_biggest_2d_in_nested_mat_dict():
    """_largest_array walks a nested MAT variables dict and returns the largest >=2-D tensor,
    skipping scalars/1-D metadata (the .mat container's 'main array')."""
    import torch

    from spectramr.data.eda.loader import _largest_array

    blob = {"meta": {"scale": torch.tensor([2.0])},        # 1-D → skipped
            "struct": {"img": torch.zeros(8, 8), "vol": torch.zeros(4, 16, 16)}}
    out = _largest_array(blob)
    assert out is not None and tuple(out.shape) == (4, 16, 16)


def test_mat_v73_large_file_is_lazily_mid_sliced(tmp_path, monkeypatch):
    """A multi-GB v7.3 (HDF5) .mat must be read one plane at a time (lazy h5py mid-slice) instead
    of being size-skipped to a card — bssfp_qsm_7t vol*.mat, osu_4dflow 4D cine .mat."""
    import h5py

    from spectramr.data.eda import loader

    p = tmp_path / "vol1.mat"
    with h5py.File(p, "w") as f:
        f.create_dataset("img", data=np.zeros((8, 20, 24), np.float32))  # (slices, H, W)
    monkeypatch.setattr(loader, "_MAX_VOXEL_BYTES", 1)  # force the "oversized" path
    s = DatasetSample()
    loader._load_one(p, s)
    assert len(s.images) == 1 and s.images[0].shape == (20, 24)   # one lazily-read plane
    assert not any("skipped voxel load" in n for n in s.notes)


def test_to_time_frames_samples_along_the_time_axis(tmp_path):
    """_to_time_frames reduces an N-D cine volume to up to k 2-D frames sampled along the largest
    non-spatial (time) axis — the two largest axes are spatial."""
    import torch

    from spectramr.data.eda.loader import _to_time_frames

    vol = torch.arange(10 * 16 * 20, dtype=torch.float32).reshape(10, 16, 20)  # (T, H, W)
    frames = _to_time_frames(vol, k=4)
    assert len(frames) == 4
    assert all(tuple(f.shape) == (16, 20) for f in frames)
    assert not torch.equal(frames[0], frames[-1])  # different time points
    # too few dims → no time axis
    assert _to_time_frames(torch.zeros(16, 20)) == []


def _write_min_dicom(path: Path, arr: np.ndarray) -> None:
    """A minimal but valid MR DICOM (osi2one ships 176 reconstructed .dcm)."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = MRImageStorage
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = fm
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.Rows, ds.Columns = arr.shape
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = arr.astype(np.uint16).tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def test_sample_surfaces_dicom_recon_when_kspace_fills_budget(tmp_path):
    """osi2one_47mt_recon ships 176 reconstructed .dcm next to its ISMRMRD .h5 — but the .h5 fill
    the sample budget, so only 'kspace mag' rendered. When the primary read yields *only* k-space,
    the loader scans for reconstructed-image files (.dcm/.nii/.png) so a real image montage shows."""
    import h5py

    root = tmp_path / "db" / "osi2one"
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(3):  # raw k-space files that fill the sample budget
        ksp = (rng.standard_normal((40, 80)) + 1j * rng.standard_normal((40, 80))).astype(np.complex64)
        with h5py.File(root / f"raw{i}.h5", "w") as f:
            f.create_dataset("kspace", data=ksp)
    _write_min_dicom(root / "recon0.dcm", rng.standard_normal((24, 24)) * 100 + 500)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "osi2one_47mt_recon.json").write_text(json.dumps({
        "dataset_name": "osi2one_47mt_recon", "data_root": str(root), "file_type": "ismrmrd",
        "records": [{"relative_path": "raw0.h5"}],
        "source": {"role": "RAW", "raw_kspace": True},
    }))
    entry = _entry(manifests / "osi2one_47mt_recon.json", tmp_path, "osi2one_47mt_recon")
    assert entry.modality == "noncartesian"
    s = sample(entry, budget=2)               # budget filled by raw k-space .h5
    assert len(s.kspace) >= 1                  # raw k-space still shown (logmag)
    assert len(s.images) >= 1                  # AND the reconstructed DICOM is surfaced


def test_sample_does_not_double_scan_recon_for_cartesian(tmp_path):
    """Cartesian k-space already reconstructs an image (ifft2c) — the recon-image fallback must
    NOT fire (no wasted second scan) when s.images is already populated."""
    import h5py

    root = tmp_path / "db" / "cartset"
    root.mkdir(parents=True)
    rng = np.random.default_rng(1)
    ksp = (rng.standard_normal((2, 32, 32)) + 1j * rng.standard_normal((2, 32, 32))).astype(np.complex64)
    with h5py.File(root / "k.h5", "w") as f:
        f.create_dataset("kspace", data=ksp)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "m4raw_train_kspace.json").write_text(json.dumps({
        "dataset_name": "m4raw_train_kspace", "data_root": str(root), "file_type": "h5",
        "records": [{"relative_path": "k.h5"}], "source": {"role": "RAW", "raw_kspace": True},
    }))
    entry = _entry(manifests / "m4raw_train_kspace.json", tmp_path, "m4raw_train_kspace")
    assert entry.modality == "kspace"
    s = sample(entry, budget=2)
    assert len(s.images) >= 1 and len(s.kspace) >= 1   # recon from ifft2c, no extra scan needed


def test_sample_retries_parent_dir_when_scan_dir_unreadable(tmp_path):
    """kasper_7t_spiral_fmri: its data_root resolves to raw/, which holds only huge
    field-monitoring .h5 (size-skipped / no recognised key → nothing readable), while the actual
    images are .nii in a sibling extracted/. When the scan dir yields nothing displayable, the
    loader retries the parent dataset dir for reconstructed images so a montage renders."""
    import h5py
    import nibabel as nib

    base = tmp_path / "databases" / "external" / "kasper_set"
    raw, extracted = base / "raw", base / "extracted"
    raw.mkdir(parents=True)
    extracted.mkdir()
    with h5py.File(raw / "field.h5", "w") as f:            # field-monitoring data, no image key
        f.create_dataset("FieldDynamics", data=np.zeros((8, 8), np.float32))
    nib.save(nib.Nifti1Image(np.random.default_rng(0).standard_normal((20, 22, 4)).astype(np.float32),
                             np.eye(4)), str(extracted / "recon.nii.gz"))  # the real images
    manifests = tmp_path / "manifests" / "external"
    manifests.mkdir(parents=True)
    (manifests / "kasper_set.json").write_text(json.dumps({
        "dataset_name": "kasper_set", "data_root": "databases/external/kasper_set/raw",
        "file_type": "h5", "records": [{"relative_path": "field.h5"}],
        "source": {"role": "RAW", "raw_kspace": True},
    }))
    entry = _entry(manifests / "kasper_set.json", tmp_path, "kasper_set")
    s = sample(entry, budget=2)
    assert len(s.images) >= 1                  # recovered from the sibling extracted/ .nii
    assert not any("no readable files" in n for n in s.notes)


def test_archive_image_reads_dicom_from_zip(tmp_path):
    """oracle_bssfp ships its reconstructed images inside *_DICOM.zip archives — _archive_image
    must read a slice straight out of the zip (no extraction to disk)."""
    import zipfile

    from spectramr.data.eda.loader import _archive_image

    dcm = tmp_path / "slice.dcm"
    _write_min_dicom(dcm, np.random.default_rng(0).standard_normal((20, 18)) * 80 + 400)
    zp = tmp_path / "I_DEGRE_DICOM.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.write(dcm, arcname="DEGRE/0001.dcm")
    img = _archive_image(zp)
    assert img is not None and img.ndim == 2 and tuple(img.shape) == (20, 18)


def test_sample_surfaces_dicom_from_zip_when_no_loose_images(tmp_path):
    """oracle_bssfp: extracted/ holds only TWIX .dat (unreadable) + *_DICOM.zip. With no loose
    recon images, the loader peeks into the DICOM zips so a real montage renders."""
    import zipfile

    base = tmp_path / "databases" / "external" / "oracle_set" / "extracted" / "InVivo"
    base.mkdir(parents=True)
    (base / "meas_raw.dat").write_bytes(b"\x00" * 32)  # TWIX raw — not a readable EDA format
    dcm = tmp_path / "s.dcm"
    _write_min_dicom(dcm, np.random.default_rng(1).standard_normal((24, 24)) * 50 + 300)
    with zipfile.ZipFile(base / "I_MESE_DICOM.zip", "w") as z:
        z.write(dcm, arcname="MESE/img0.dcm")
    manifests = tmp_path / "manifests" / "external"
    manifests.mkdir(parents=True)
    (manifests / "oracle_set.json").write_text(json.dumps({
        "dataset_name": "oracle_set", "data_root": "databases/external/oracle_set/extracted",
        "file_type": "twix", "records": [{"relative_path": "InVivo/meas_raw.dat"}],
        "source": {"role": "RAW", "raw_kspace": True},
    }))
    entry = _entry(manifests / "oracle_set.json", tmp_path, "oracle_set")
    s = sample(entry, budget=2)
    assert len(s.images) >= 1   # recovered from the DICOM zip


def _tar_gz_with_nifti(tar_path: Path, n: int = 2) -> None:
    """A .tar.gz holding a few .nii.gz (mirrors kirby MMRR-*.tar.gz)."""
    import tarfile

    import nibabel as nib

    tmp = tar_path.parent / "_staging"
    tmp.mkdir(exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as t:
        for i in range(n):
            p = tmp / f"sub{i}_MPRAGE.nii.gz"
            nib.save(nib.Nifti1Image(np.random.default_rng(i).standard_normal((16, 18, 4)).astype(np.float32),
                                     np.eye(4)), str(p))
            t.add(p, arcname=f"MMRR/3T/sub{i}_MPRAGE.nii.gz")


def test_archive_image_reads_nifti_from_tar_gz(tmp_path):
    """kirby_kki_multimodal ships its images inside MMRR-*.tar.gz — read a slice straight out of
    the archive (no full extraction to disk)."""
    from spectramr.data.eda.loader import _archive_image

    tarp = tmp_path / "MMRR-3T7T-2-1_multimodal.tar.gz"
    _tar_gz_with_nifti(tarp)
    img = _archive_image(tarp)
    assert img is not None and img.ndim == 2


def test_sample_renders_dataset_with_only_archives(tmp_path):
    """A downloaded-but-not-unpacked dataset (kirby: raw/ holds only *.tar.gz) must render — the
    catalog counts archives as data (tier=present) and the loader reads an image from the archive.
    """
    raw = tmp_path / "databases" / "external" / "kirby_kki_multimodal" / "raw"
    raw.mkdir(parents=True)
    _tar_gz_with_nifti(raw / "MMRR-3T7T-2-1_multimodal.tar.gz")
    manifests = tmp_path / "manifests" / "external"
    manifests.mkdir(parents=True)
    (manifests / "kirby_kki_multimodal.json").write_text(json.dumps({
        "dataset_name": "kirby_kki_multimodal",
        "data_root": "databases/external/kirby_kki_multimodal/raw",
        "file_type": "nifti", "records": [], "status": "not_downloaded",
        "source": {"role": "img"},
    }))
    entry = _entry(manifests / "kirby_kki_multimodal.json", tmp_path, "kirby_kki_multimodal")
    assert entry.tier == "present"          # archives count as data
    s = sample(entry, budget=2)
    assert len(s.images) >= 1               # image read out of the .tar.gz


def test_sample_does_not_pull_sibling_images_via_shared_parent(tmp_path):
    """whole_heart_5d regression: its scan dir resolves to the dataset root external/<x> (TWIX
    .dat, no readable image). The recon/parent-retry fallback must NOT climb to the shared
    external/ container and pull a *sibling* brain dataset's images (cross-dataset contamination)."""
    import zipfile

    import h5py

    ext = tmp_path / "databases" / "external"
    wh = ext / "whole_heart_5d"
    wh.mkdir(parents=True)
    with zipfile.ZipFile(wh / "inVivo.zip", "w") as z:  # TWIX .dat in a zip → no decodable image
        z.writestr("V01/meas.dat", b"\x00" * 16)
    sib = ext / "some_brain_set" / "extracted"          # a sibling dataset's data
    sib.mkdir(parents=True)
    with h5py.File(sib / "brain.h5", "w") as f:
        f.create_dataset("kspace", data=np.zeros((2, 8, 8), np.complex64))
    manifests = ext.parent.parent / "manifests" / "external"
    manifests.mkdir(parents=True)
    (manifests / "whole_heart_5d.json").write_text(json.dumps({
        "dataset_name": "whole_heart_5d", "data_root": "databases/external/whole_heart_5d",
        "file_type": "twix", "records": [], "status": "downloaded", "source": {"role": "RAW"},
    }))
    entry = _entry(manifests / "whole_heart_5d.json", tmp_path, "whole_heart_5d")
    s = sample(entry, budget=2)
    assert s.images == []   # must NOT show the sibling brain set's images


def test_radial_nufft_recon_round_trips_a_disk(tmp_path):
    """Adjoint-NUFFT recon of radial k-space recovers a centred disk (math check). Forward-NUFFT a
    disk phantom onto a golden-angle trajectory, then _radial_nufft_recon must bring back an image
    whose centre is brighter than its corners."""
    import torchkbnufft as tkbn

    from spectramr.data.eda.loader import _radial_nufft_recon
    from spectramr.infrastructure.physics.trajectories import TrajectoryFactory

    n, spokes = 48, 64
    yy, xx = np.mgrid[0:n, 0:n]
    disk = ((yy - n / 2) ** 2 + (xx - n / 2) ** 2 < (n / 5) ** 2).astype(np.float32)
    traj, _ = TrajectoryFactory.get_radial_trajectory((n, n), num_spokes=spokes,
                                                       samples_per_spoke=n, golden_angle=True)
    import torch
    fwd = tkbn.KbNufft(im_size=(n, n))
    ksp = fwd(torch.from_numpy(disk).to(torch.complex64)[None, None], traj[None])  # (1,1,N)
    recon = _radial_nufft_recon(ksp.reshape(spokes, n))    # (spokes, samples) plane
    assert recon is not None and recon.ndim == 2
    c = recon[n // 2 - 3:n // 2 + 3, n // 2 - 3:n // 2 + 3].mean()
    corner = recon[:4, :4].mean()
    assert c > corner * 2   # disk centre reconstructed brighter than background


def test_sample_radial_cfl_renders_nufft_recon(tmp_path):
    """A radial .cfl dataset (noncartesian, no trajectory file) renders a best-effort NUFFT recon
    image (golden-angle assumed) in addition to the k-space logmag — not just card+logmag."""
    rng = np.random.default_rng(0)
    arr = (rng.standard_normal((64, 40)) + 1j * rng.standard_normal((64, 40))).astype(np.complex64)
    root = tmp_path / "databases" / "external" / "cardiac_radial_x" / "raw"
    root.mkdir(parents=True)
    _write_cfl(root / "vol0001_vis1", arr)
    manifests = tmp_path / "manifests" / "external"
    manifests.mkdir(parents=True)
    (manifests / "cardiac_radial_x.json").write_text(json.dumps({
        "dataset_name": "cardiac_radial_x", "data_root": "databases/external/cardiac_radial_x/raw",
        "file_type": "bart", "records": [], "status": "downloaded",
        "source": {"role": "RAW", "raw_kspace": True},
    }))
    entry = _entry(manifests / "cardiac_radial_x.json", tmp_path, "cardiac_radial_x")
    assert entry.modality == "noncartesian"
    s = sample(entry, budget=2)
    assert len(s.kspace) >= 1                              # raw k-space logmag kept
    assert len(s.images) >= 1                              # AND a NUFFT recon image
    assert any("nufft" in n.lower() for n in s.notes)      # labeled (golden-angle assumed)


def test_a_scaled_int16_nifti_decodes_with_slope_and_intercept_applied():
    """np.asarray(dataobj, dtype=float32) replaced np.array(..., copy=...) for numpy 2 (#2674);
    it must still return scaled values, as get_fdata does, not the stored integers."""
    import nibabel as nib

    from spectramr.data.eda.loader import _decode_member_image

    stored = (np.arange(64).reshape(8, 8, 1) * 7).astype(np.int16)
    img = nib.Nifti1Image(stored, np.eye(4))
    img.header.set_data_dtype(np.int16)
    img.header.set_slope_inter(2.5, 10.0)
    decoded = _decode_member_image("scan.nii", img.to_bytes())
    assert decoded is not None
    assert np.allclose(decoded.numpy().ravel(), np.sort((stored * 2.5 + 10.0).ravel()))
