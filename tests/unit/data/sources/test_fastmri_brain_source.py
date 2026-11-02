"""The fastMRI brain source: the QC gate, and the AXT1/AXT1POST prefix trap."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from spectramr.data.annotations.fastmri_plus import ESTABLISHED_Y_ORIGIN  # noqa: E402
from spectramr.data.annotations.manifest_qc import (  # noqa: E402
    ApprovalDigestMismatchError,
    approval_digest,
)
from spectramr.data.sources.fastmri_brain_source import (  # noqa: E402
    ALLOWED_ACQUISITIONS,
    MIN_LESION_BRAIN_COVERAGE,
    CohortDropError,
    ContrastMismatchError,
    FastMRIBrainSource,
    QCNotApprovedError,
    acquisition_from_filename,
)


def _manifest(
    tmp_path,
    *,
    approved: bool = True,
    records=None,
    y_origin: str = ESTABLISHED_Y_ORIGIN.value,
    digest: str | None = "auto",
) -> str:
    """A manifest as the builder would write it.

    ``digest="auto"`` mints the digest the approval is a statement about, exactly as
    ``--approve-qc`` does. Pass ``None`` to simulate an approval from the pre-correction
    generator, or a literal to simulate tampering.
    """
    path = tmp_path / "fastmri_plus.json"
    manifest = {
        "manifest_version": "3.3",
        "data_root": str(tmp_path / "data"),
        "annotation": {"csv_path": "brain.csv", "boxes_parsed": 1},
        "qc": {
            "y_origin": y_origin,
            "overlay_dir": str(tmp_path / "qc"),
            "overlays": [str(tmp_path / "qc" / "file_brain_AXT2_201_6002717_sl004.png")],
            "approved": approved,
        },
        "total_records": 1,
        "records": (
            records
            if records is not None
            else [
                {
                    "file_id": "file_brain_AXT2_201_6002717.h5",
                    "slice_index": 4,
                    "boxes": [
                        {
                            "x": 10,
                            "y_raw": 20,
                            "width": 30,
                            "height": 30,
                            "label": "Edema",
                            "group": "edema",
                        }
                    ],
                }
            ]
        ),
    }
    if approved and digest is not None:
        manifest["qc"]["approved_digest"] = (
            approval_digest(manifest) if digest == "auto" else digest
        )
    path.write_text(json.dumps(manifest))
    return str(path)


class TestTheQCGateIsWiredEndToEnd:
    def test_an_unapproved_manifest_is_refused(self, tmp_path) -> None:
        """The y-origin QC flag is not decoration. A wrong origin flips every box
        into mirror tissue and yields a confident null -- so the source will not
        run until a human has looked at the overlays."""
        with pytest.raises(QCNotApprovedError, match=re.escape("qc.approved = false")):
            FastMRIBrainSource(_manifest(tmp_path, approved=False))

    def test_the_refusal_points_at_the_overlays(self, tmp_path) -> None:
        with pytest.raises(QCNotApprovedError, match="approve-qc"):
            FastMRIBrainSource(_manifest(tmp_path, approved=False))

    def test_an_approved_manifest_loads(self, tmp_path) -> None:
        src = FastMRIBrainSource(_manifest(tmp_path, approved=True))
        assert src.name == "fastmri_brain"
        assert len(src) == 1


class TestApprovalCannotBeTransplantedOntoADifferentYOrigin:
    """An approval is a statement ABOUT a y-origin.

    Nothing used to stop someone approving a manifest and then editing ``qc.y_origin``
    -- which mirrors every lesion box onto the contralateral hemisphere while the green
    stamp rides along untouched. The digest binds the approval to what it was about.
    """

    def test_editing_y_origin_after_approval_raises(self, tmp_path) -> None:
        path = Path(_manifest(tmp_path, approved=True))
        manifest = json.loads(path.read_text())
        assert manifest["qc"]["y_origin"] == ESTABLISHED_Y_ORIGIN.value
        manifest["qc"]["y_origin"] = "top_left"  # the flip, smuggled in post-approval
        path.write_text(json.dumps(manifest))

        with pytest.raises(ApprovalDigestMismatchError, match="edited after approval"):
            FastMRIBrainSource(str(path))

    def test_an_approval_with_no_digest_is_refused(self, tmp_path) -> None:
        """Every manifest approved before 2026-07-13 was approved under the INVERTED
        y-origin. Those approvals are precisely the ones that must not be honoured."""
        with pytest.raises(ApprovalDigestMismatchError, match="y-origin was CORRECTED"):
            FastMRIBrainSource(_manifest(tmp_path, approved=True, digest=None))

    def test_a_forged_digest_is_refused(self, tmp_path) -> None:
        with pytest.raises(ApprovalDigestMismatchError, match="digests to"):
            FastMRIBrainSource(_manifest(tmp_path, approved=True, digest="deadbeef"))


class TestTheAXT1PrefixTrap:
    """'AXT1' is a prefix of 'AXT1PRE' and 'AXT1POST'.

    A substring test would match all three, silently merging three DIFFERENT
    contrasts into one cell of the anatomy axis -- the axis whose entire purpose is
    to separate them. Nothing would crash.
    """

    def test_axt1post_is_not_read_as_axt1(self) -> None:
        assert acquisition_from_filename("file_brain_AXT1POST_201_6002717.h5") == "AXT1POST"

    def test_axt1pre_is_not_read_as_axt1(self) -> None:
        assert acquisition_from_filename("file_brain_AXT1PRE_201_6002717.h5") == "AXT1PRE"

    def test_plain_axt1_still_resolves(self) -> None:
        assert acquisition_from_filename("file_brain_AXT1_201_6002717.h5") == "AXT1"

    def test_a_naive_substring_test_would_have_been_wrong(self) -> None:
        """Pins the bug this guards against, so nobody 'simplifies' it back."""
        name = "file_brain_AXT1POST_201.h5"
        assert "AXT1" in name  # the tempting check...
        assert acquisition_from_filename(name) != "AXT1"  # ...and why it is wrong

    @pytest.mark.parametrize("acq", ALLOWED_ACQUISITIONS)
    def test_every_allowed_acquisition_round_trips(self, acq: str) -> None:
        assert acquisition_from_filename(f"file_brain_{acq}_201_6002717.h5") == acq

    def test_a_filename_with_no_acquisition_token_raises(self) -> None:
        with pytest.raises(ContrastMismatchError, match="0 recognised"):
            acquisition_from_filename("file_brain_201_6002717.h5")

    def test_a_filename_with_two_tokens_raises(self) -> None:
        with pytest.raises(ContrastMismatchError, match="2 recognised"):
            acquisition_from_filename("file_brain_AXT1_AXT2_201.h5")

    def test_a_token_outside_the_allow_list_raises(self) -> None:
        with pytest.raises(ContrastMismatchError, match="not one of"):
            acquisition_from_filename("file_brain_AXT2FLAIR_201_6002717.h5")

    def test_the_error_type_is_owned_by_the_contrast_grid(self) -> None:
        from spectramr.data.datasets import contrast_grid

        assert ContrastMismatchError is contrast_grid.ContrastMismatchError


def _rec(file_id: str, slice_index: int = 4, boxes=None) -> dict:
    return {
        "file_id": file_id,
        "slice_index": slice_index,
        "boxes": (
            boxes
            if boxes is not None
            else [
                {
                    "x": 10,
                    "y_raw": 20,
                    "width": 30,
                    "height": 30,
                    "label": "Edema",
                    "group": "edema",
                }
            ]
        ),
    }


class TestUnreadableContrastsAreCountedNotEaten:
    """A record used to vanish from the cohort on a bare `continue`.

    No count, no log. "This cohort has N studies" then became indistinguishable from "it
    has N plus the ones the source ate" -- and the dropped files are not a random sample,
    they are the ones with unusual names. Pitfall #9, in the module otherwise most
    carefully armoured against it.
    """

    def test_a_dropped_record_is_counted_and_reported(self, tmp_path) -> None:
        records = [_rec("file_brain_AXT2_201.h5")] * 9 + [_rec("file_brain_201.h5")]
        src = FastMRIBrainSource(_manifest(tmp_path, records=records))

        assert len(src) == 9
        assert src.n_contrast_dropped == 1
        assert src.contrast_drop_rate == pytest.approx(0.1)

        report = src.drop_report()
        assert report["n_records"] == 10
        assert report["n_eligible"] == 9
        assert "file_brain_201.h5" in report["contrast_dropped"]
        assert "recognised acquisition" in report["contrast_dropped"]["file_brain_201.h5"]

    def test_too_many_drops_raises_rather_than_yielding_a_selection(self, tmp_path) -> None:
        """Past the ceiling the survivors are a selection, not the manifest. Reporting
        them as the cohort would be a lie of omission."""
        records = [_rec("file_brain_AXT2_201.h5")] * 5 + [_rec("file_brain_201.h5")] * 5
        with pytest.raises(CohortDropError, match=re.escape("(50.0%)")) as exc:
            FastMRIBrainSource(_manifest(tmp_path, records=records))
        assert "over the 20% ceiling" in str(exc.value)
        assert "not a random sample" in str(exc.value)

    def test_a_clean_manifest_reports_zero_drops(self, tmp_path) -> None:
        src = FastMRIBrainSource(_manifest(tmp_path))
        assert src.n_contrast_dropped == 0
        assert src.contrast_drop_rate == 0.0

    def test_drops_are_counted_per_record_not_per_file(self, tmp_path) -> None:
        """The manifest holds one record per annotated SLICE, so one unreadable filename
        drops every slice of that volume. Counting distinct files would report '1 dropped'
        where 40 slices actually left the cohort -- and the drop-rate ceiling would then
        never fire. (This was a real bug in the first cut of this counter.)"""
        records = [_rec("file_brain_AXT2_201.h5", slice_index=i) for i in range(6)] + [
            _rec("file_brain_201.h5", slice_index=i) for i in range(4)
        ]
        with pytest.raises(CohortDropError, match=re.escape("(40.0%)")):
            FastMRIBrainSource(_manifest(tmp_path, records=records))


def _box(label: str, group: str, *, x: int = 10, y: int = 10, w: int = 20, h: int = 20) -> dict:
    return {"x": x, "y_raw": y, "width": w, "height": h, "label": label, "group": group}


class _FakeCache:
    """A SynthSeg cache whose brain mask is the left half of the image."""

    def __init__(self, h: int = 64, w: int = 64) -> None:
        self._brain = torch.zeros(h, w, dtype=torch.bool)
        self._brain[:, : w // 2] = True  # brain = columns [0, 32)

    def has(self, file_id: str) -> bool:
        return True

    def brain(self, file_id: str, slice_index: int):
        return self._brain


class TestNotEveryBoxIsALesion:
    """``path:lesion_any`` used to pool EVERY box, whatever it was labelled.

    The taxonomy said in prose that some groups must never be pooled; the pooling code
    ignored it. So a "Possible artifact" (505 boxes in the shipped CSV) and a "Normal
    variant" (73) went straight into the lesion region -- and sim2rank's whole method is to
    INJECT artifacts and rank metrics on them, so the lesion arm carried pre-existing
    artifacts and confounded the axis with itself.
    """

    def _regions_for(self, tmp_path, boxes, cache=None):
        src = FastMRIBrainSource(
            _manifest(tmp_path, records=[_rec("file_brain_AXT2_201.h5", boxes=boxes)]),
            region_cache=cache,
        )
        rec = src._eligible[0]
        return src._regions(rec, 64, 64)

    def test_an_artifact_box_never_enters_the_lesion_region(self, tmp_path) -> None:
        masks, prov = self._regions_for(tmp_path, [_box("Possible artifact", "artifact")])
        assert "path:lesion_any" not in masks
        assert prov["path:lesion_any"]["dropped"] == "all_boxes_excluded"
        assert "non_parenchymal_group" in prov["path:lesion_any"]["excluded"]

    def test_a_normal_variant_box_never_enters_the_lesion_region(self, tmp_path) -> None:
        masks, _ = self._regions_for(tmp_path, [_box("Normal variant", "normal")])
        assert "path:lesion_any" not in masks

    def test_a_real_lesion_still_enters(self, tmp_path) -> None:
        masks, prov = self._regions_for(tmp_path, [_box("Edema", "edema")])
        assert "path:lesion_any" in masks
        assert prov["path:lesion_any"]["n_boxes"] == 1
        assert prov["path:lesion_any"]["n_boxes_excluded"] == 0

    def test_the_excluded_boxes_are_counted_not_silently_dropped(self, tmp_path) -> None:
        masks, prov = self._regions_for(
            tmp_path,
            [_box("Edema", "edema"), _box("Possible artifact", "artifact", x=40)],
        )
        assert "path:lesion_any" in masks  # the real lesion survives
        p = prov["path:lesion_any"]
        assert p["n_boxes"] == 1
        assert p["n_boxes_excluded"] == 1
        assert p["excluded"]["non_parenchymal_group"] == ["Possible artifact (artifact)"]


class TestTheOffBrainGateIsMeasuredNotGuessed:
    """POSTSURGICAL is mixed: a craniotomy sits on the skull flap, a resection cavity in
    parenchyma. Same label, opposite answers about whether a mirror control is meaningful.

    A taxonomy cannot call that. Brain coverage can -- and it also catches any box the
    labels simply got wrong, which no group-level rule ever would.
    """

    def _regions_for(self, tmp_path, boxes):
        src = FastMRIBrainSource(
            _manifest(tmp_path, records=[_rec("file_brain_AXT2_201.h5", boxes=boxes)]),
            region_cache=_FakeCache(),
        )
        return src._regions(src._eligible[0], 64, 64)

    def test_a_postsurgical_box_on_brain_is_kept(self, tmp_path) -> None:
        """A resection cavity: fully inside the brain half. Its mirror IS a valid control."""
        masks, prov = self._regions_for(
            tmp_path, [_box("Resection cavity", "postsurgical", x=5, w=20)]
        )
        assert "path:lesion_any" in masks
        assert prov["path:lesion_any"]["n_boxes_excluded"] == 0

    def test_a_postsurgical_box_off_brain_is_excluded(self, tmp_path) -> None:
        """A craniotomy on the skull flap: entirely in the non-brain half. Its mirror is
        intact skull, so pooling it would make lesion_vs_control meaningless."""
        masks, prov = self._regions_for(tmp_path, [_box("Craniotomy", "postsurgical", x=40, w=20)])
        assert "path:lesion_any" not in masks
        assert "off_brain" in prov["path:lesion_any"]["excluded"]

    def test_the_same_label_gives_opposite_verdicts_by_position(self, tmp_path) -> None:
        """The point of measuring rather than guessing: one label, two answers."""
        on_brain, _ = self._regions_for(tmp_path, [_box("Craniotomy", "postsurgical", x=5, w=20)])
        off_brain, _ = self._regions_for(tmp_path, [_box("Craniotomy", "postsurgical", x=40, w=20)])
        assert "path:lesion_any" in on_brain
        assert "path:lesion_any" not in off_brain

    def test_the_threshold_is_the_declared_constant(self, tmp_path) -> None:
        """Brain is columns [0,32) of 64. A box at x=22..42 is 10/20 = 50% brain, exactly
        at the floor -- and the gate is `< floor`, so it is kept."""
        assert MIN_LESION_BRAIN_COVERAGE == 0.5
        masks, _ = self._regions_for(tmp_path, [_box("Edema", "edema", x=22, w=20)])
        assert "path:lesion_any" in masks


class TestHeaderVsFilename:
    def test_agreement_resolves_the_contrast(self, tmp_path) -> None:
        src = FastMRIBrainSource(_manifest(tmp_path))
        got = src._resolve_contrast("file_brain_AXT2_201.h5", {"acquisition": "AXT2"})
        assert got == "AXT2"

    def test_a_bytes_attr_is_decoded(self, tmp_path) -> None:
        src = FastMRIBrainSource(_manifest(tmp_path))
        assert src._resolve_contrast("file_brain_AXT2_1.h5", {"acquisition": b"AXT2"}) == "AXT2"

    def test_disagreement_raises_instead_of_picking_one(self, tmp_path) -> None:
        """A file whose header and name disagree has broken provenance. Picking one
        to believe is a silent fallback (pitfall #9)."""
        src = FastMRIBrainSource(_manifest(tmp_path))
        with pytest.raises(ContrastMismatchError, match="broken provenance"):
            src._resolve_contrast("file_brain_AXT2_201.h5", {"acquisition": "AXFLAIR"})

    def test_a_missing_attr_falls_back_to_the_exact_filename_token(self, tmp_path) -> None:
        """Not a guess: the token match is exact. There is simply nothing to
        cross-check it against."""
        src = FastMRIBrainSource(_manifest(tmp_path))
        assert src._resolve_contrast("file_brain_AXFLAIR_1.h5", {}) == "AXFLAIR"


class TestContrastFiltering:
    def test_an_unknown_contrast_filter_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="unknown acquisition"):
            FastMRIBrainSource(_manifest(tmp_path), contrasts=["AXT9"])

    def test_filtering_excludes_other_contrasts(self, tmp_path) -> None:
        src = FastMRIBrainSource(_manifest(tmp_path), contrasts=["AXFLAIR"])
        assert len(src) == 0  # the only record is AXT2


class TestTissueRegionsAreGated:
    def test_requiring_tissue_regions_without_a_cache_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="no region_cache"):
            FastMRIBrainSource(_manifest(tmp_path), require_tissue_regions=True)


class TestCachedMasksMustBeOnTheImageGrid:
    """The read-side twin of ``nifti_export.check_label_grid``.

    ``check_label_grid`` compares a segmentation against the NIfTI it came from,
    at cache-BUILD time. Nothing checked the other end: this source reconstructs
    from the **h5** at its native matrix size, so a cache built from a re-gridded
    export hands back a mask of a different sampling of space.

    The failure is silent exactly where it costs most. ``_brain_coverage``
    divides two areas and returns a plausible float, so a lesion would be
    accepted or rejected against a brain mask describing somewhere else -- no
    IndexError, no shape error, just a wrong number.
    """

    def test_a_regridded_mask_raises_rather_than_masking_the_wrong_pixels(
        self,
    ) -> None:
        import torch

        from spectramr.data.sources.fastmri_brain_source import (
            MaskGridMismatchError,
            _check_mask_grid,
        )

        # Cache built from a 320x320 export; h5 reconstruction is 384x384.
        mask = torch.zeros(320, 320, dtype=torch.bool)
        with pytest.raises(MaskGridMismatchError) as exc:
            _check_mask_grid("file_brain_AXT2_200_2000080", mask.shape, (384, 384))

        message = str(exc.value)
        assert "(320, 320)" in message and "(384, 384)" in message
        # Names the cause and the correct remedy, not just the mismatch.
        assert "--expect-inplane 0" in message
        assert "per-volume" in message

    def test_a_matching_grid_passes(self) -> None:
        import torch

        from spectramr.data.sources.fastmri_brain_source import _check_mask_grid

        _check_mask_grid("f", torch.zeros(384, 384, dtype=torch.bool).shape, (384, 384))

    def test_a_leading_axis_is_ignored(self) -> None:
        """Masks arrive as [H, W]; a [1, H, W] must not read as a mismatch."""
        import torch

        from spectramr.data.sources.fastmri_brain_source import _check_mask_grid

        _check_mask_grid("f", torch.zeros(1, 320, 320).shape, (320, 320))

    def test_mixed_grids_across_volumes_are_not_the_failure(self) -> None:
        """A heterogeneous COHORT is fine; only a re-gridded EXPORT is not.

        The cache is one .npz per file_id with its own shape, so 320x320 and
        384x384 volumes each pass against their own image.
        """
        import torch

        from spectramr.data.sources.fastmri_brain_source import _check_mask_grid

        _check_mask_grid("a", torch.zeros(320, 320).shape, (320, 320))
        _check_mask_grid("b", torch.zeros(384, 384).shape, (384, 384))
