"""fastMRI brain volume source: manifest-driven, QC-gated, region-attached.

Five gates, all of which have to pass before a slice reaches the sweep:

1. **The fastMRI+ manifest's QC must be approved**, and the approval must still describe
   the manifest it sits in. A wrong y-origin flips every lesion box into mirror-image
   normal tissue and yields a confident, meaningless null; the matched control mirrors
   with it, so nothing downstream can detect the flip.
2. **The acquisition attr must agree with the filename token**, and a record whose
   filename yields no single token is **counted** as it is dropped -- never silently
   skipped. Past :data:`MAX_CONTRAST_DROP_RATE` the survivors are a selection rather than
   the cohort, and the source refuses.
3. **A box must be a lesion to enter the lesion region.** ``"Possible artifact"`` (505
   boxes) and ``"Normal variant"`` (73) are not pathology, and ``"Paranasal sinus
   opacification"`` (40) is not on brain. See
   :data:`~spectramr.data.annotations.fastmri_plus_classes.NON_PARENCHYMAL_GROUPS`.
4. **A box must be ON brain to enter the lesion region** -- measured against SynthSeg, at
   :data:`MIN_LESION_BRAIN_COVERAGE`, because the label alone cannot tell a craniotomy on
   the skull flap from a resection cavity in parenchyma.
5. **The tissue axis must be sound** before any ``tissue:*`` region is attached -- but a
   SynthSeg failure does not block the pathology axis.

Every exclusion is counted and stamped into the slice's provenance. A box or a record
that quietly vanished would be a cohort change with no record of itself.

The AXT1 / AXT1POST prefix trap
--------------------------------
fastMRI brain acquisitions are ``AXT1``, ``AXT1PRE``, ``AXT1POST``, ``AXT2``,
``AXFLAIR``. **"AXT1" is a prefix of both "AXT1PRE" and "AXT1POST"**, so a
substring test (``"AXT1" in filename``) matches all three and silently labels
every post-contrast scan as pre-contrast.

That would not crash. It would merge three *different* contrasts into one cell of
the leaderboard's anatomy axis -- the axis whose entire purpose is to separate
them. So the filename is split into underscore tokens and matched **exactly**.

``io_strategies.FastMRIH5Strategy`` already carries ``f.attrs`` (which holds
``acquisition``) into ``result["metadata"]``, and until now **nothing read it**.
This source reads it, cross-checks it against the filename token, and **raises on
disagreement** -- a file whose header and name disagree about its contrast is one
whose provenance is broken, and guessing which to believe is exactly the silent
fallback the repo forbids (pitfall #9).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import torch

from spectramr.data.annotations.fastmri_plus import LesionBox, YOrigin, pooled_lesion_mask
from spectramr.data.annotations.fastmri_plus_classes import LesionGroup, is_poolable
from spectramr.data.annotations.manifest_qc import verify_approval
from spectramr.data.annotations.matched_control import sample_matched_controls
from spectramr.data.datasets.contrast_grid import (
    ContrastMismatchError,
    brain_filename_contrast,
    resolve_brain_contrast,
)
from spectramr.data.io_strategies import FastMRIH5Strategy
from spectramr.data.regions.mask_cache import RegionMaskCache
from spectramr.data.regions.region_registry import REGION_SPECS, RegionTier, region_spec
from spectramr.data.sources.sim2rank_source import Sim2RankSlice

__all__ = [
    "ALLOWED_ACQUISITIONS",
    "MAX_CONTRAST_DROP_RATE",
    "MIN_LESION_BRAIN_COVERAGE",
    "CohortDropError",
    "ContrastMismatchError",
    "FastMRIBrainSource",
    "QCNotApprovedError",
    "acquisition_from_filename",
]

logger = logging.getLogger(__name__)

ALLOWED_ACQUISITIONS: tuple[str, ...] = (
    "AXT1",
    "AXT1PRE",
    "AXT1POST",
    "AXT2",
    "AXFLAIR",
)

MIN_LESION_BRAIN_COVERAGE = 0.5
"""A pooled lesion box must sit at least this much on brain (SynthSeg), or it is out.

The pooled region only means anything if its contralateral mirror is a valid control:
same size, same anatomy, no lesion. ``matched_control._accept`` already demands the
**control** be >=95% brain. Accepting a *lesion* box at any coverage while holding the
control to 95% makes the two arms incomparable in the one property the control exists to
hold fixed -- so the lesion is gated too.

Measured, not assumed, because the taxonomy genuinely cannot call it: ``POSTSURGICAL``
holds both craniotomy (on the skull flap -- mirror is intact skull, useless) and
resection cavity (in parenchyma -- mirror is normal brain, exactly right). One label,
two answers. Brain coverage separates them; a group name cannot. See
:data:`~spectramr.data.annotations.fastmri_plus_classes.NON_PARENCHYMAL_GROUPS`.

Only applies when a SynthSeg cache is present. Without one there is no brain mask, hence
no control, hence no ``lesion_vs_control`` table for the gate to protect.
"""

MAX_CONTRAST_DROP_RATE = 0.20
"""Above this share of records dropped for an unreadable contrast, the cohort is unsound.

The dropped files are not a random sample -- they are the ones with unusual names. Past
that share, "the cohort" is a selection, and reporting it as the cohort is a lie of
omission.
"""


class QCNotApprovedError(RuntimeError):
    """The fastMRI+ manifest's y-origin has not been confirmed by a human."""


class CohortDropError(RuntimeError):
    """Too many records were dropped for the surviving cohort to be representative."""


class MaskGridMismatchError(RuntimeError):
    """A cached region mask is not on the grid of the image it would mask."""


def _check_mask_grid(file_id: str, mask_shape: tuple[int, ...], image_hw: tuple[int, int]) -> None:
    """The read-side twin of ``nifti_export.check_label_grid``.

    ``check_label_grid`` runs when the cache is BUILT and compares a segmentation
    against the NIfTI it came from. Nothing checked the other end: this source
    reconstructs from the **h5** at its native matrix size, so a cache built from a
    re-gridded export hands back a mask of a different sampling of space. The
    failure is silent where it matters most -- ``_brain_coverage`` divides two
    areas and returns a plausible float, so a lesion would be accepted or rejected
    against a brain mask describing somewhere else.

    Grids diverge only when an export was resampled. Do not resample: the cache is
    per volume, so a mixed-grid cohort is already correct.
    """
    if tuple(mask_shape[-2:]) != image_hw:
        raise MaskGridMismatchError(
            f"{file_id}: cached region mask is {tuple(mask_shape[-2:])} but the h5 "
            f"reconstruction is {image_hw}. The mask describes a different sampling "
            "of space than the image it would multiply, so every region measurement "
            "on this volume would be of the wrong pixels. This means the NIfTI export "
            "was re-gridded relative to the h5 -- re-export with --expect-inplane 0 "
            "(the region cache is per-volume and handles mixed grids) and rebuild the "
            "cache."
        )


def acquisition_from_filename(file_id: str) -> str:
    """Exact underscore-token match -- never a substring test -- restricted to the allow-list.

    ``"AXT1" in "file_brain_AXT1POST_201.h5"`` is True, which would label a
    post-contrast scan as pre-contrast and quietly merge two contrasts into one
    cell of the anatomy axis. The token parse is owned by ``contrast_grid``.
    """
    found = brain_filename_contrast(file_id)
    if found not in ALLOWED_ACQUISITIONS:
        raise ContrastMismatchError(
            f"{file_id!r}: acquisition {found!r} is not one of {list(ALLOWED_ACQUISITIONS)}."
        )
    return found


class FastMRIBrainSource:
    """Manifest-driven fastMRI brain slices with regions attached.

    Never globs the data tree: the manifest is the SSOT, produced by
    ``scripts/data/build_fastmri_plus_manifest.py``. Globbing would silently
    include whatever happens to be on disk, which is how a cohort quietly changes
    size between runs.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        region_cache: RegionMaskCache | None = None,
        global_seed: int = 0,
        contrasts: Sequence[str] | None = None,
        require_tissue_regions: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text())

        qc = manifest.get("qc", {})
        # `is not True`, not truthiness: qc["approved"] = "false" (a string) is TRUTHY, and
        # this is the gate the entire y-origin safety argument rests on.
        if qc.get("approved") is not True:
            raise QCNotApprovedError(
                f"{self.manifest_path} has qc.approved = false. The fastMRI+ "
                "y-origin has not been confirmed by a human. A wrong origin flips "
                "every lesion box into mirror-image normal tissue: nothing crashes, "
                "the boxes still look like brain, and the sweep reports a confident "
                "null result that means nothing. Inspect "
                f"{qc.get('overlay_dir')!r} and run build_fastmri_plus_manifest.py "
                "--approve-qc."
            )

        # An approval is a statement ABOUT a y-origin. Nothing stopped someone from
        # approving a manifest and then editing qc.y_origin, which mirrors every box
        # onto the contralateral hemisphere while the stamp stays green. The digest
        # binds the approval to the contents it was made about; drift raises.
        verify_approval(manifest, source=str(self.manifest_path))

        self._y_origin = YOrigin(qc["y_origin"])
        self.data_root = Path(manifest["data_root"])
        self._records = manifest["records"]
        self._region_cache = region_cache
        self._global_seed = global_seed
        self._strategy = FastMRIH5Strategy()

        # The tissue axis is opt-in AND gated. A SynthSeg failure must not take the
        # pathology axis down with it -- fastMRI+ boxes do not use SynthSeg.
        self._tissue_regions_enabled = False
        if require_tissue_regions:
            if region_cache is None:
                raise ValueError(
                    "require_tissue_regions=True but no region_cache was given; "
                    "tissue regions come from the SynthSeg cache."
                )
            region_cache.report.raise_if_tissue_axis_unsound()
            self._tissue_regions_enabled = True

        allowed = set(contrasts) if contrasts else set(ALLOWED_ACQUISITIONS)
        unknown = allowed - set(ALLOWED_ACQUISITIONS)
        if unknown:
            raise ValueError(
                f"unknown acquisition(s) {sorted(unknown)}; allowed: {list(ALLOWED_ACQUISITIONS)}"
            )
        self._allowed = allowed

        # Partition ONCE, here, rather than re-deriving it inside a generator that both
        # __len__ and __iter__ call. Doing it lazily is what let the unreadable-contrast
        # drop stay invisible: it was a bare `continue`, so a record could vanish from
        # the cohort with no count and no log, and "this cohort has N studies" became
        # indistinguishable from "it has N + the ones we ate". Pitfall #9.
        self._eligible: list[dict] = []
        # RECORDS dropped, and the distinct FILES they came from. Both, deliberately: the
        # manifest holds one record per annotated *slice*, so a single unreadable filename
        # drops every slice of that volume. Counting files would report "1 dropped" where
        # 40 slices left the cohort.
        self._n_contrast_dropped = 0
        self._contrast_drops: dict[str, str] = {}
        for rec in self._records:
            try:
                acq = acquisition_from_filename(rec["file_id"])
            except ContrastMismatchError as exc:
                self._n_contrast_dropped += 1
                self._contrast_drops[rec["file_id"]] = str(exc)
                continue
            if acq in self._allowed:
                self._eligible.append(rec)

        if self._contrast_drops:
            logger.warning(
                "%d of %d fastMRI+ record(s) dropped across %d file(s): the filename "
                "carries no single recognised acquisition token (%.1f%% of the "
                "manifest). Files: %s",
                self.n_contrast_dropped,
                len(self._records),
                len(self._contrast_drops),
                100.0 * self.contrast_drop_rate,
                sorted(self._contrast_drops)[:5],
            )
        if self.contrast_drop_rate > MAX_CONTRAST_DROP_RATE:
            raise CohortDropError(
                f"{self.n_contrast_dropped} of {len(self._records)} records "
                f"({self.contrast_drop_rate:.1%}) have no readable acquisition token, "
                f"over the {MAX_CONTRAST_DROP_RATE:.0%} ceiling. The survivors are not a "
                "random sample -- they are the files with conventional names -- so the "
                "cohort this source would yield is a selection, not the manifest. Fix "
                "the filenames or narrow the manifest deliberately. Dropped: "
                f"{sorted(self._contrast_drops)[:10]}"
            )

    @property
    def name(self) -> str:
        return "fastmri_brain"

    @property
    def contrasts(self) -> Sequence[str]:
        return tuple(sorted(self._allowed))

    @property
    def n_contrast_dropped(self) -> int:
        """RECORDS dropped because the filename had no single acquisition token."""
        return self._n_contrast_dropped

    @property
    def contrast_drop_rate(self) -> float:
        if not self._records:
            return 0.0
        return self.n_contrast_dropped / len(self._records)

    def drop_report(self) -> dict[str, object]:
        """Every record this source refused to yield, and why. For run provenance."""
        return {
            "n_records": len(self._records),
            "n_eligible": len(self._eligible),
            "n_contrast_dropped": self.n_contrast_dropped,
            "n_contrast_dropped_files": len(self._contrast_drops),
            "contrast_drop_rate": round(self.contrast_drop_rate, 4),
            "contrast_dropped": dict(sorted(self._contrast_drops.items())),
        }

    def __len__(self) -> int:
        return len(self._eligible)

    def _eligible_records(self) -> Iterator[dict]:
        return iter(self._eligible)

    def _resolve_contrast(self, file_id: str, metadata: Mapping[str, object]) -> str:
        """Header vs filename through the one resolver; disagreement raises."""
        raw = metadata.get("acquisition")
        return resolve_brain_contrast(file_id, raw if raw is None or isinstance(raw, bytes) else str(raw))

    def _boxes(self, rec: dict) -> list[LesionBox]:
        return [
            LesionBox(
                file_id=rec["file_id"],
                slice_index=rec["slice_index"],
                x=b["x"],
                y_raw=b["y_raw"],
                width=b["width"],
                height=b["height"],
                label=b["label"],
                group=LesionGroup(b["group"]),
                y_origin=self._y_origin,
            )
            for b in rec["boxes"]
        ]

    @staticmethod
    def _brain_coverage(box: LesionBox, brain: torch.Tensor) -> float:
        """Fraction of a box's pixels that SynthSeg calls brain."""
        h, w = int(brain.shape[-2]), int(brain.shape[-1])
        m = box.to_bool_mask(h, w)
        return float((brain & m).sum()) / float(m.sum())

    def _regions(
        self, rec: dict, h: int, w: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        file_id, slice_index = rec["file_id"], rec["slice_index"]
        masks: dict[str, torch.Tensor] = {"full": torch.ones(h, w, dtype=torch.bool)}
        provenance: dict[str, object] = {"full": {"source": "identity"}}

        # Gate 1 (label): a group whose contralateral mirror is not brain can never
        # enter the pool. A sinus opacification's "matched control in normal-appearing
        # tissue" is the OTHER SINUS, and the paired test still returns a number.
        all_boxes = self._boxes(rec)
        boxes = [b for b in all_boxes if is_poolable(b.group)]
        excluded: dict[str, list[str]] = {}
        if len(boxes) != len(all_boxes):
            excluded["non_parenchymal_group"] = sorted(
                f"{b.label} ({b.group.value})" for b in all_boxes if not is_poolable(b.group)
            )

        has_cache = self._region_cache is not None and self._region_cache.has(file_id)
        brain: torch.Tensor | None = None
        if has_cache:
            assert self._region_cache is not None
            brain = self._region_cache.brain(file_id, slice_index)
            _check_mask_grid(file_id, brain.shape, (h, w))
            masks["brain"] = brain
            provenance["brain"] = {"source": "synthseg", "qc": "passed"}

            # Gate 2 (measurement): the taxonomy cannot tell a craniotomy (on the skull
            # flap) from a resection cavity (in parenchyma) -- same POSTSURGICAL label,
            # opposite answers. Brain coverage can. This also catches any box the labels
            # got wrong, which no group-level rule ever would.
            off_brain = [
                b for b in boxes if self._brain_coverage(b, brain) < MIN_LESION_BRAIN_COVERAGE
            ]
            if off_brain:
                boxes = [b for b in boxes if b not in off_brain]
                excluded["off_brain"] = sorted(
                    f"{b.label} ({self._brain_coverage(b, brain):.0%} brain)" for b in off_brain
                )

        if boxes:
            masks["path:lesion_any"] = pooled_lesion_mask(boxes, h, w)
            provenance["path:lesion_any"] = {
                "source": "fastmri_plus",
                "y_origin": self._y_origin.value,
                "n_boxes": len(boxes),
                "n_boxes_excluded": len(all_boxes) - len(boxes),
                "excluded": excluded,
                "labels": sorted({b.label for b in boxes}),
            }
        elif all_boxes:
            # Every box on this slice was excluded. Say so -- a slice that silently
            # stops carrying a pathology region is a cohort change with no record.
            provenance["path:lesion_any"] = {
                "source": "fastmri_plus",
                "dropped": "all_boxes_excluded",
                "excluded": excluded,
            }

        if has_cache and boxes:
            assert brain is not None
            # The matched control needs the brain mask, so it only exists when SynthSeg
            # is available for this volume.
            report = sample_matched_controls(boxes, brain=brain, global_seed=self._global_seed)
            if report.n_dropped:
                # An unpaired lesion re-introduces the size confound the control exists
                # to remove, so the LESION goes too.
                masks.pop("path:lesion_any", None)
                provenance["path:lesion_any"] = {
                    "source": "fastmri_plus",
                    "dropped": "unpaired_lesion",
                    "excluded": excluded,
                    "detail": report.to_dict()["dropped_detail"],
                }
            elif report.controls:
                control = torch.zeros(h, w, dtype=torch.bool)
                for c in report.controls:
                    control |= c.to_bool_mask(h, w)
                masks["control:lesion_any"] = control
                provenance["control:lesion_any"] = {
                    "source": "matched_control",
                    "controls": [c.provenance for c in report.controls],
                }

        if has_cache and self._tissue_regions_enabled:
            assert self._region_cache is not None
            for rid, spec in REGION_SPECS.items():
                if spec.tier is not RegionTier.TISSUE:
                    continue
                m = self._region_cache.region(file_id, slice_index, spec.tissues)
                _check_mask_grid(file_id, m.shape, (h, w))
                if bool(m.any()):  # an empty region is not a region to score
                    masks[rid] = m
                    provenance[rid] = {
                        "source": "synthseg",
                        "tissues": [t.value for t in spec.tissues],
                    }

        return masks, provenance

    def __iter__(self) -> Iterator[Sim2RankSlice]:
        for rec in self._eligible_records():
            file_id, slice_index = rec["file_id"], rec["slice_index"]
            path = self.data_root / (file_id if file_id.endswith(".h5") else f"{file_id}.h5")
            loaded = self._strategy.load(str(path), {"slice_index": slice_index})

            target = loaded.get("data")
            if target is None:
                raise ValueError(
                    f"{path} has no reconstruction_rss/_esc -- there is no clean "
                    "reference to degrade or to score against."
                )
            clean = target.squeeze().abs().float()[None]  # [1, H, W]
            h, w = clean.shape[-2:]

            contrast = self._resolve_contrast(file_id, loaded.get("metadata", {}))
            masks, provenance = self._regions(rec, h, w)

            yield Sim2RankSlice(
                content_id=f"{file_id}#{slice_index}",
                clean=clean,
                contrast=contrast,
                file_id=file_id,
                slice_index=slice_index,
                # fastMRI de-identifies patients; one file is the finest subject unit known.
                subject_id=file_id,
                kspace=loaded.get("kspace"),
                region_masks=masks,
                provenance={
                    "source": self.name,
                    "manifest": str(self.manifest_path),
                    "regions": provenance,
                },
            )


def _assert_regions_declared() -> None:
    """Import-time guard: every region this source can emit is in the registry."""
    for rid in ("full", "brain", "path:lesion_any", "control:lesion_any"):
        region_spec(rid)


_assert_regions_declared()
