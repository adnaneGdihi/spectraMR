"""The affine and the slice axis: the two things a green SynthSeg run cannot check.

A wrong affine does not crash SynthSeg -- it resamples with it, segments a brain that was
never acquired, and returns a confident label map. A wrong slice axis does not crash the
cache either: it silently hands the metric a mask from a different plane. Neither is
visible downstream, so both are pinned here.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest
import torch

from spectramr.data.nifti_export import (
    PROVENANCE_FILENAME,
    SLICE_AXIS,
    ExportVerificationError,
    GeometryUnavailableError,
    GridMismatchError,
    VoxelGeometry,
    check_label_grid,
    check_label_spacing,
    load_export_grids,
    load_export_spacings,
    parse_ismrmrd_geometry,
    verify_slice_first_nifti,
    verify_slice_last_nifti,
    write_slice_first_nifti,
    write_slice_last_nifti,
    zooms_from_affine,
)

nib = pytest.importorskip("nibabel")


def ismrmrd_xml(
    *,
    recon_x: int = 320,
    recon_y: int = 320,
    recon_z: int = 1,
    fov_x: float = 220.0,
    fov_y: float = 220.0,
    fov_z: float = 5.0,
) -> bytes:
    """A minimal ISMRMRD header in the real namespace, shaped like fastMRI's."""
    return f"""<?xml version="1.0" encoding="utf-8"?>
<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
  <encoding>
    <encodedSpace>
      <matrixSize><x>640</x><y>320</y><z>1</z></matrixSize>
      <fieldOfView_mm><x>440</x><y>220</y><z>5</z></fieldOfView_mm>
    </encodedSpace>
    <reconSpace>
      <matrixSize><x>{recon_x}</x><y>{recon_y}</y><z>{recon_z}</z></matrixSize>
      <fieldOfView_mm>
        <x>{fov_x}</x><y>{fov_y}</y><z>{fov_z}</z>
      </fieldOfView_mm>
    </reconSpace>
  </encoding>
</ismrmrdHeader>""".encode()


class TestSpacingComesFromTheHeaderOrNotAtAll:
    def test_square_recon_matrix_resolves_in_plane_spacing(self) -> None:
        geom = parse_ismrmrd_geometry(ismrmrd_xml(), n_slices=16, rows=320, cols=320)
        assert geom.col_mm == pytest.approx(220.0 / 320.0)
        assert geom.row_mm == pytest.approx(220.0 / 320.0)
        assert geom.slice_mm == pytest.approx(5.0)

    def test_a_2d_multislice_header_states_thickness_not_spacing(self) -> None:
        """matrixSize.z == 1: reconSpace describes ONE slice, so its z-FOV is the slice
        THICKNESS. The inter-slice GAP is nowhere in a fastMRI header, so contiguous
        slices are assumed -- and the assumption is recorded, which is what makes it
        falsifiable instead of invisible."""
        geom = parse_ismrmrd_geometry(
            ismrmrd_xml(recon_z=1), n_slices=16, rows=320, cols=320
        )
        assert geom.slice_gap_assumed_zero is True
        assert geom.as_provenance()["slice_gap_assumed_zero"] is True

    def test_a_3d_header_gives_true_spacing_and_assumes_nothing(self) -> None:
        geom = parse_ismrmrd_geometry(
            ismrmrd_xml(recon_z=16, fov_z=80.0), n_slices=16, rows=320, cols=320
        )
        assert geom.slice_mm == pytest.approx(5.0)
        assert geom.slice_gap_assumed_zero is False

    def test_rows_and_columns_are_not_swapped(self) -> None:
        """On a square matrix a row/column swap is invisible. Here they differ, and the
        header's x is the readout (columns), y the phase-encode (rows)."""
        geom = parse_ismrmrd_geometry(
            ismrmrd_xml(recon_x=320, recon_y=256, fov_x=320.0, fov_y=128.0),
            n_slices=16,
            rows=256,
            cols=320,
        )
        assert geom.col_mm == pytest.approx(1.0)
        assert geom.row_mm == pytest.approx(0.5)

    def test_an_in_plane_mapping_that_cannot_be_resolved_raises(self) -> None:
        """A 320x320 matrix against a 256x192 array.

        The verdict is unchanged -- an unresolvable readout/phase-encode mapping must
        stop the export -- but the reason moved when spacing became crop-aware. It is no
        longer "matches neither orientation" (a 320 matrix admits both a 256 and a 192
        axis once a centre crop is allowed); it is "admits both, matches neither
        exactly", which is the same guess wearing different clothes. The literal
        matches-neither case is now
        ``TestTheStoredArrayIsACropOfWhatTheHeaderDescribes
        ::test_an_array_larger_than_the_recon_matrix_raises``.
        """
        with pytest.raises(GeometryUnavailableError, match="BOTH orientations"):
            parse_ismrmrd_geometry(ismrmrd_xml(), n_slices=16, rows=256, cols=192)

    def test_no_header_raises_rather_than_defaulting_to_1mm(self) -> None:
        """The whole point. A 1 mm default is a fabrication that SynthSeg will act on."""
        with pytest.raises(GeometryUnavailableError, match="fabrication"):
            parse_ismrmrd_geometry(None, n_slices=16, rows=320, cols=320)

    def test_an_unparseable_header_raises(self) -> None:
        with pytest.raises(GeometryUnavailableError, match="not valid XML"):
            parse_ismrmrd_geometry(b"<ismrmrdHeader", n_slices=16, rows=320, cols=320)

    def test_a_header_in_metres_is_rejected_not_silently_used(self) -> None:
        """FOV 0.22 (m) instead of 220 (mm) -> 0.7 micron voxels. SynthSeg would still
        return a segmentation."""
        with pytest.raises(GeometryUnavailableError, match="plausible band"):
            parse_ismrmrd_geometry(
                ismrmrd_xml(fov_x=0.22, fov_y=0.22), n_slices=16, rows=320, cols=320
            )


class TestTheAffineDeclaresTheSliceAxis:
    def geom(self) -> VoxelGeometry:
        return VoxelGeometry(
            slice_mm=5.0,
            row_mm=0.7,
            col_mm=0.5,
            n_slices=16,
            rows=256,
            cols=320,
            slice_gap_assumed_zero=True,
            source="test",
        )

    def test_voxel_sizes_come_back_in_slice_row_col_order(self) -> None:
        sizes = nib.affines.voxel_sizes(self.geom().affine())
        assert tuple(sizes) == pytest.approx((5.0, 0.7, 0.5))

    def test_stepping_the_slice_index_moves_only_through_plane(self) -> None:
        """Axis 0 must be the through-plane direction. If it were an in-plane axis, the
        volume SynthSeg sees is 16 mm wide and 320 mm deep, and it resamples accordingly.
        """
        aff = self.geom().affine()
        origin = aff @ np.array([0, 0, 0, 1.0])
        stepped = aff @ np.array([1, 0, 0, 1.0])
        assert (stepped - origin)[:3] == pytest.approx([0.0, 0.0, 5.0])

    def test_the_affine_is_right_handed(self) -> None:
        """A negative determinant reads as a flipped volume to SynthSeg's RAS alignment."""
        assert np.linalg.det(self.geom().affine()[:3, :3]) > 0


class TestTheWrittenFileIsTheVolumeThatWasWritten:
    def marked_volume(self, s: int = 6, h: int = 8, w: int = 10) -> torch.Tensor:
        """Slice i is filled with the value i -- so a transposition is not merely a shape
        change, it is a value change, and both are caught."""
        vol = torch.zeros(s, h, w, dtype=torch.float32)
        for i in range(s):
            vol[i] = float(i)
        return vol

    def geom_for(self, vol: torch.Tensor) -> VoxelGeometry:
        s, h, w = (int(x) for x in vol.shape)
        return VoxelGeometry(
            slice_mm=5.0,
            row_mm=0.7,
            col_mm=0.5,
            n_slices=s,
            rows=h,
            cols=w,
            slice_gap_assumed_zero=True,
            source="test",
        )

    def test_slice_i_reads_back_as_slice_i_on_axis_zero(self, tmp_path) -> None:
        """THE round-trip. RegionMaskCache indexes vol[slice_index] and the graded image
        is reconstruction_rss[i]; if the exported volume put slices anywhere but axis 0,
        every tissue mask would be a cut through the wrong plane -- silently."""
        from spectramr.data.io_strategies import NiftiStrategy

        vol = self.marked_volume()
        out = tmp_path / "v.nii.gz"
        write_slice_first_nifti(out, vol, self.geom_for(vol))

        read = NiftiStrategy().load(str(out))["data"].squeeze()
        assert tuple(read.shape) == (6, 8, 10)
        for i in range(6):
            assert torch.all(
                read[i] == float(i)
            ), f"slice {i} did not survive the export"
        assert SLICE_AXIS == 0

    def test_verify_catches_a_transposed_file(self, tmp_path) -> None:
        vol = self.marked_volume()
        out = tmp_path / "v.nii.gz"
        # What a slice-last export would have produced.
        transposed = vol.permute(1, 2, 0).numpy()
        nib.save(nib.Nifti1Image(transposed, np.eye(4)), str(out))
        with pytest.raises(ExportVerificationError, match="slice axis moved"):
            verify_slice_first_nifti(out, vol)

    def test_an_uncompressed_nii_is_refused(self, tmp_path) -> None:
        """SynthSeg keys its QC csv on basename.replace('.nii.gz',''), and names its
        output p.replace('.nii', '_synthseg.nii'). For 'x.nii' the QC subject stays
        'x.nii' while the volume id becomes 'x_synthseg' -- nothing matches, every volume
        drops as no_qc_row, and the cache builder calls the tissue axis unsound AFTER the
        GPU run."""
        vol = self.marked_volume()
        with pytest.raises(ValueError, match=re.escape("nii.gz")):
            write_slice_first_nifti(tmp_path / "v.nii", vol, self.geom_for(vol))

    def test_a_non_finite_volume_is_refused(self, tmp_path) -> None:
        vol = self.marked_volume()
        vol[2, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            write_slice_first_nifti(tmp_path / "v.nii.gz", vol, self.geom_for(vol))

    def test_a_geometry_that_does_not_match_the_volume_is_refused(
        self, tmp_path
    ) -> None:
        vol = self.marked_volume()
        wrong = self.geom_for(self.marked_volume(s=99))
        with pytest.raises(ValueError, match="does not match the geometry"):
            write_slice_first_nifti(tmp_path / "v.nii.gz", vol, wrong)


class TestTheLabelsMustComeBackOnTheImagesGrid:
    """The segmenter is a black box between the export and the cache.

    SynthSeg resamples to 1 mm internally and resamples *back* to the input grid before
    saving. If it ever does not (``--resample``, ``--crop``, or a segmentation directory
    from a different export), the cache would still happily index the label volume with
    the h5's slice numbers -- masking slice ``i`` of the labels against slice ``i`` of a
    volume that has different slices. That is not a degraded mask, it is a mask of
    somewhere else, and it is invisible: a resampled label map is a perfectly valid label
    map. The export provenance is the receipt that makes the check possible.
    """

    @staticmethod
    def provenance(tmp_path, volumes):
        d = tmp_path / "nii"
        d.mkdir(parents=True, exist_ok=True)
        (d / PROVENANCE_FILENAME).write_text(
            json.dumps(
                {
                    "volumes": [
                        {"file_id": f, "geometry": {"shape_shw": list(s)}}
                        for f, s in volumes.items()
                    ]
                }
            )
        )
        return d

    def test_grids_are_read_from_the_export_receipt(self, tmp_path) -> None:
        d = self.provenance(tmp_path, {"a.h5": (16, 320, 320), "b": (12, 256, 256)})

        grids = load_export_grids(d)

        assert grids == {"a": (16, 320, 320), "b": (12, 256, 256)}
        assert load_export_grids(d / PROVENANCE_FILENAME) == grids

    def test_a_missing_receipt_raises_rather_than_skipping_the_check(
        self, tmp_path
    ) -> None:
        with pytest.raises(FileNotFoundError, match=re.escape(PROVENANCE_FILENAME)):
            load_export_grids(tmp_path)

    def test_a_segmentation_on_the_image_grid_passes(self, tmp_path) -> None:
        grids = load_export_grids(self.provenance(tmp_path, {"a": (16, 320, 320)}))
        check_label_grid("a", (16, 320, 320), grids)

    def test_a_resampled_segmentation_is_caught(self, tmp_path) -> None:
        """SynthSeg's internal grid is 1 mm isotropic: a 16x320x320 5 mm stack becomes
        ~80 slices there. Cached as-is, ``labels[7]`` is a slice the image never had."""
        grids = load_export_grids(self.provenance(tmp_path, {"a": (16, 320, 320)}))
        with pytest.raises(GridMismatchError, match="not on the image's grid"):
            check_label_grid("a", (80, 320, 320), grids)

    def test_a_transposed_segmentation_is_caught(self, tmp_path) -> None:
        grids = load_export_grids(self.provenance(tmp_path, {"a": (16, 320, 320)}))
        with pytest.raises(GridMismatchError, match="not on the image's grid"):
            check_label_grid("a", (320, 320, 16), grids)

    def test_a_segmentation_from_an_unknown_export_is_caught(self, tmp_path) -> None:
        """Its grid is not wrong -- it is unknown, which is worse: nothing ties it to the
        images being graded."""
        grids = load_export_grids(self.provenance(tmp_path, {"a": (16, 320, 320)}))
        with pytest.raises(GridMismatchError, match="not in the export provenance"):
            check_label_grid("stranger", (16, 320, 320), grids)


class TestTheStoredArrayIsACropOfWhatTheHeaderDescribes:
    """``reconstruction_rss`` is fastMRI's centre crop; ``reconSpace`` is uncropped.

    Requiring ``matrixSize == array size`` conflated two different things: it dropped
    every genuinely-cropped volume, and it silently accepted ``FOV / array_size`` as the
    spacing whenever the matrix happened to equal the crop. Spacing comes from the
    header's own ``(fieldOfView_mm, matrixSize)`` pair, because a crop changes the field
    of view, not the voxel size.
    """

    def test_spacing_divides_the_fov_by_the_recon_matrix_not_the_cropped_array(
        self,
    ) -> None:
        """The regression. Pre-fix this raised; a naive relax-the-check fix would return
        220/320 = 0.6875. Only dividing by the true recon matrix gives 220/512."""
        geometry = parse_ismrmrd_geometry(
            ismrmrd_xml(recon_x=512, recon_y=512, fov_x=220.0, fov_y=220.0),
            n_slices=16,
            rows=320,
            cols=320,
        )
        assert geometry.col_mm == pytest.approx(220.0 / 512)
        assert geometry.row_mm == pytest.approx(220.0 / 512)
        # The wrong answer this test exists to exclude.
        assert geometry.col_mm != pytest.approx(220.0 / 320)

    def test_the_crop_is_recorded_so_the_spacing_can_be_re_derived(self) -> None:
        geometry = parse_ismrmrd_geometry(
            ismrmrd_xml(recon_x=512, recon_y=512, fov_x=220.0, fov_y=220.0),
            n_slices=16,
            rows=320,
            cols=320,
        )
        assert (geometry.recon_rows, geometry.recon_cols) == (512, 512)
        assert (geometry.crop_rows, geometry.crop_cols) == (192, 192)
        prov = geometry.as_provenance()
        assert prov["recon_matrix_rc"] == [512, 512]
        assert prov["center_crop_rc"] == [192, 192]
        # The receipt has to be sufficient to reconstruct the field of view.
        assert prov["col_mm"] * 512 == pytest.approx(220.0)

    def test_an_uncropped_volume_is_unchanged(self) -> None:
        """The pre-existing exact-match path must keep its exact previous answer."""
        geometry = parse_ismrmrd_geometry(
            ismrmrd_xml(recon_x=320, recon_y=320, fov_x=220.0, fov_y=220.0),
            n_slices=16,
            rows=320,
            cols=320,
        )
        assert geometry.col_mm == pytest.approx(220.0 / 320)
        assert (geometry.crop_rows, geometry.crop_cols) == (0, 0)

    def test_an_array_larger_than_the_recon_matrix_raises(self) -> None:
        """Not a crop in any orientation, so the header is not describing this array."""
        with pytest.raises(GeometryUnavailableError, match="smaller than the array"):
            parse_ismrmrd_geometry(
                ismrmrd_xml(recon_x=256, recon_y=256),
                n_slices=16,
                rows=320,
                cols=320,
            )

    def test_an_asymmetric_crop_raises(self) -> None:
        """The affine centres the volume, which is only the acquired centre when equal
        amounts came off both sides. An odd difference means it does not."""
        with pytest.raises(
            GeometryUnavailableError, match=re.escape("not a symmetric centre crop")
        ):
            parse_ismrmrd_geometry(
                ismrmrd_xml(recon_x=321, recon_y=321),
                n_slices=16,
                rows=320,
                cols=320,
            )

    def test_a_non_square_array_admitted_by_both_orientations_raises(self) -> None:
        """Relaxing exact equality re-opened the guess the exact rule prevented: with a
        crop, a non-square array can fit the matrix either way round."""
        with pytest.raises(GeometryUnavailableError, match="BOTH"):
            parse_ismrmrd_geometry(
                ismrmrd_xml(recon_x=512, recon_y=512),
                n_slices=16,
                rows=256,
                cols=320,
            )

    def test_a_non_square_exact_match_still_resolves(self) -> None:
        """An exact hit disambiguates, so the non-square path must not be lost."""
        geometry = parse_ismrmrd_geometry(
            ismrmrd_xml(recon_x=320, recon_y=256, fov_x=220.0, fov_y=176.0),
            n_slices=16,
            rows=256,
            cols=320,
        )
        assert geometry.col_mm == pytest.approx(220.0 / 320)
        assert geometry.row_mm == pytest.approx(176.0 / 256)


class TestTheLabelsMustComeBackAtTheImagesScale:
    """The gap every other check in the chain leaves open.

    ``write_slice_first_nifti`` compares shapes, ``verify_slice_first_nifti`` compares
    values, ``check_label_grid`` compares shapes. None compares the affine -- and the
    affine is the only thing SynthSeg actually resamples with.
    """

    @staticmethod
    def provenance(tmp_path, spacings: dict[str, tuple[float, float, float]]):
        doc = {
            "volumes": [
                {
                    "file_id": f"{vid}.h5",
                    "geometry": {
                        "shape_shw": [16, 320, 320],
                        "slice_mm": s[0],
                        "row_mm": s[1],
                        "col_mm": s[2],
                    },
                }
                for vid, s in spacings.items()
            ]
        }
        (tmp_path / PROVENANCE_FILENAME).write_text(json.dumps(doc))
        return tmp_path

    def test_matching_spacing_passes(self, tmp_path) -> None:
        sp = load_export_spacings(
            self.provenance(tmp_path, {"a": (5.0, 0.6875, 0.6875)})
        )
        check_label_spacing("a", (5.0, 0.6875, 0.6875), sp)

    def test_the_crop_ratio_error_is_caught(self, tmp_path) -> None:
        """Exactly the failure the crop-aware spacing prevents, arriving from the other
        direction: labels produced at FOV/320 against an image exported at FOV/512."""
        sp = load_export_spacings(
            self.provenance(tmp_path, {"a": (5.0, 220.0 / 512, 220.0 / 512)})
        )
        with pytest.raises(GridMismatchError, match="row spacing"):
            check_label_spacing("a", (5.0, 220.0 / 320, 220.0 / 512), sp)

    def test_a_wrong_slice_spacing_is_caught(self, tmp_path) -> None:
        """The 1 mm-isotropic resample, which leaves the shape alone when the label map
        is resampled back but the header is not."""
        sp = load_export_spacings(
            self.provenance(tmp_path, {"a": (5.0, 0.6875, 0.6875)})
        )
        with pytest.raises(GridMismatchError, match="slice spacing"):
            check_label_spacing("a", (1.0, 0.6875, 0.6875), sp)

    def test_an_unrecorded_volume_raises(self, tmp_path) -> None:
        sp = load_export_spacings(
            self.provenance(tmp_path, {"a": (5.0, 0.6875, 0.6875)})
        )
        with pytest.raises(GridMismatchError, match="no recorded spacing"):
            check_label_spacing("stranger", (5.0, 0.6875, 0.6875), sp)

    def test_a_missing_receipt_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match=re.escape(PROVENANCE_FILENAME)):
            load_export_spacings(tmp_path)

    def test_zooms_from_affine_inverts_the_geometrys_own_affine(self) -> None:
        """The two must agree, or the check compares a spacing against a different
        spacing and silently never fires."""
        geometry = VoxelGeometry(
            slice_mm=5.0,
            row_mm=0.6875,
            col_mm=0.4297,
            n_slices=16,
            rows=320,
            cols=320,
            slice_gap_assumed_zero=True,
            source="test",
            recon_rows=512,
            recon_cols=512,
        )
        assert zooms_from_affine(geometry.affine()) == pytest.approx(geometry.zooms())

    def test_a_non_affine_input_raises(self) -> None:
        with pytest.raises(GridMismatchError, match="4x4"):
            zooms_from_affine(np.eye(3))


def _geom_shw(s: int, h: int, w: int) -> VoxelGeometry:
    return VoxelGeometry(
        slice_mm=3.0,
        row_mm=0.5,
        col_mm=0.75,
        n_slices=s,
        rows=h,
        cols=w,
        slice_gap_assumed_zero=True,
        source="test",
    )


def test_slice_last_write_is_sliced_on_depth_by_the_training_loader(tmp_path):
    # The paired-NIfTI loader takes the LAST on-disk axis as depth; a slice-first file
    # would hand it (S, H) cuts through the volume.
    from spectramr.data.io_strategies import NiftiStrategy

    vol = torch.arange(3 * 5 * 7, dtype=torch.float32).reshape(3, 5, 7)
    path = write_slice_last_nifti(tmp_path / "v.nii.gz", vol, _geom_shw(3, 5, 7))
    loaded = NiftiStrategy().load(str(path))["data"]
    assert tuple(loaded.shape) == (1, 5, 7, 3)
    assert torch.equal(loaded[0, :, :, 1], vol[1])
    ref = NiftiStrategy.load_reference_volume(path)
    assert torch.equal(ref["data"], vol)
    assert ref["spacing_mm"] == pytest.approx((3.0, 0.5, 0.75))
    verify_slice_last_nifti(path, vol, (3.0, 0.5, 0.75))


def test_slice_last_verify_rejects_a_slice_first_file(tmp_path):
    vol = torch.rand(3, 5, 7)
    path = write_slice_first_nifti(tmp_path / "v.nii.gz", vol, _geom_shw(3, 5, 7))
    with pytest.raises(ExportVerificationError, match="slice axis moved"):
        verify_slice_last_nifti(path, vol, (3.0, 0.5, 0.75))
