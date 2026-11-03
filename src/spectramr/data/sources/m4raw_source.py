"""M4Raw volume source -- the control cohort for the region axis.

M4Raw is where the region machinery gets validated **first**, before a byte of
fastMRI is downloaded: it is already on the cluster, its NEX pseudo-GT is already
implemented, and its anatomical tiers are the *control condition* for the fastMRI+
pathology tiers anyway. If the leaderboard does not move between ``full`` and
``brain`` on M4Raw, the region axis has a problem that has nothing to do with
lesions.

M4Raw carries no pathology annotations, so it emits ``full`` and (when a SynthSeg
cache exists) the tissue tiers -- never ``path:*`` or ``control:*``.

The reference is the NEX pseudo-GT
----------------------------------
Multi-repetition complex k-space averaging (Macovski 1996) gives the full sqrt(N)
SNR boost, and :func:`~spectramr.data.datasets.pseudo_gt_reference.synthesize_pseudo_gt`
is the SSOT for it. This source wraps it rather than re-deriving a reference --
sim2rank grades every metric against this image, so a second, subtly different
reference implementation would silently change what "correct" means.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import torch

from spectramr.data.datasets.contrast_grid import contrast_label
from spectramr.data.datasets.m4raw_repetition_groups import discover_repetition_groups
from spectramr.data.regions.mask_cache import RegionMaskCache
from spectramr.data.regions.region_registry import REGION_SPECS, RegionTier
from spectramr.data.sources.sim2rank_source import Sim2RankSlice
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.data.datasets.pseudo_gt_reference import synthesize_pseudo_gt

__all__ = ["M4RawSource"]


class M4RawSource:
    """Slices from M4Raw repetition groups, with the NEX pseudo-GT as reference."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        slices_per_volume: int = 5,
        region_cache: RegionMaskCache | None = None,
        max_volumes: int | None = None,
        max_subjects: int = 5,
        max_contrasts: int = 3,
        require_tissue_regions: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.slices_per_volume = slices_per_volume
        self._region_cache = region_cache

        groups = discover_repetition_groups(
            self.data_root, max_subjects=max_subjects, max_contrasts=max_contrasts
        )

        # discover_repetition_groups() falls back to SINGLETON groups when it finds
        # no multi-rep ones. A 1-rep "pseudo-GT" is just the noisy acquisition
        # itself: there is no sqrt(N) averaging, so the reference carries the same
        # noise as the input. Every metric would then be graded on how well it
        # tracks degradation away from a noisy image -- which is not the question,
        # and it would silently depress every full-reference metric at once. Refuse.
        singletons = [g for g in groups if len(g) < 2]
        self._groups: list[list[Path]] = [g for g in groups if len(g) >= 2]
        self.dropped_singletons = len(singletons)

        if not self._groups:
            raise FileNotFoundError(
                f"no multi-repetition M4Raw groups under {self.data_root} "
                f"({len(singletons)} singleton group(s) found). The NEX pseudo-GT "
                "needs >= 2 repetitions: with one rep there is no sqrt(N) SNR boost, "
                "so the 'clean' reference is just the noisy acquisition and every "
                "full-reference metric would be graded against noise."
            )
        if max_volumes is not None:
            self._groups = self._groups[:max_volumes]

        self._tissue_regions_enabled = False
        if require_tissue_regions:
            if region_cache is None:
                raise ValueError("require_tissue_regions=True but no region_cache was given.")
            region_cache.report.raise_if_tissue_axis_unsound()
            self._tissue_regions_enabled = True

    @property
    def name(self) -> str:
        return "m4raw"

    @property
    def contrasts(self) -> Sequence[str]:
        return tuple(sorted({contrast_label(g[0]) for g in self._groups}))

    @staticmethod
    def _group_id(rep_paths: list[Path]) -> str:
        """Stable id for a repetition group: the stem minus the rep number.

        Must not depend on list order or on which rep happens to be first --
        ``content_id`` is the ranker's covariate C, so an id that changed between
        runs would look like a different subject.
        """
        return rep_paths[0].name.split(".")[0][:-2]

    @staticmethod
    def _subject_id(group_id: str) -> str:
        """``YYYYMMDDNN_<contrast>`` -> ``YYYYMMDDNN``: every contrast of a scan is one subject."""
        return group_id.split("_")[0]

    def __len__(self) -> int:
        return len(self._groups) * self.slices_per_volume

    def _regions(
        self, file_id: str, slice_index: int, h: int, w: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        masks = {"full": torch.ones(h, w, dtype=torch.bool)}
        provenance: dict[str, object] = {"full": {"source": "identity"}}

        cache = self._region_cache
        if cache is None or not cache.has(file_id):
            return masks, provenance

        masks["brain"] = cache.brain(file_id, slice_index)
        provenance["brain"] = {"source": "synthseg", "qc": "passed"}

        if self._tissue_regions_enabled:
            for rid, spec in REGION_SPECS.items():
                if spec.tier is not RegionTier.TISSUE:
                    continue
                m = cache.region(file_id, slice_index, spec.tissues)
                if bool(m.any()):
                    masks[rid] = m
                    provenance[rid] = {
                        "source": "synthseg",
                        "tissues": [t.value for t in spec.tissues],
                    }

        return masks, provenance

    def __iter__(self) -> Iterator[Sim2RankSlice]:
        for rep_paths in self._groups:
            paths = [Path(p) for p in rep_paths]
            group_id = self._group_id(paths)
            contrast = contrast_label(paths[0])

            for slice_index in self._slice_indices(self._volume_depth(paths[0])):
                # synthesize_pseudo_gt returns (coil_images, smaps, x_gt_mag, p99).
                # This used to unpack it as (clean_c, kspace, smaps, _), so EVERY field
                # was off by one: `clean` got the coil images, `kspace` got the coil
                # maps, `coil_maps` got the magnitude reference. M4Raw is 4-coil, so
                # `coil_images.squeeze()[None]` came out (1, 4, H, W) and Sim2RankSlice's
                # own [1, H, W] check rejected it -- the M4Raw source could not yield a
                # single slice. No test caught it because no test calls __iter__.
                coil_images, smaps, x_gt_mag, p99 = synthesize_pseudo_gt(
                    paths, target_slice=slice_index
                )
                # x_gt_mag is (1, 1, H, W), already p99-normalised -> [1, H, W].
                clean = x_gt_mag[0].abs().float()
                # coil_images is IMAGE domain (m4raw_pseudo_gt returns it that way to
                # avoid a double FFT). Sim2RankSlice.kspace is the *measurement*, so the
                # physics metrics need the transform -- fastMRI yields true k-space, and
                # handing them image data for one cohort and k-space for the other would
                # silently compare different domains.
                kspace = fft2c(coil_images)
                h, w = clean.shape[-2:]
                masks, provenance = self._regions(group_id, slice_index, h, w)

                yield Sim2RankSlice(
                    content_id=f"{group_id}#{slice_index}",
                    clean=clean,
                    contrast=contrast,
                    file_id=group_id,
                    slice_index=slice_index,
                    subject_id=self._subject_id(group_id),
                    kspace=kspace,
                    coil_maps=smaps,
                    region_masks=masks,
                    provenance={
                        "source": self.name,
                        "reference": "nex_pseudo_gt",
                        "n_reps": len(paths),
                        "normalisation": {"mode": "p99", "value": float(p99)},
                        "regions": provenance,
                    },
                )

    @staticmethod
    def _volume_depth(path: Path) -> int:
        """Slice count, without reading a single voxel.

        A shape probe, not a load. ``h5py`` is legal here because this IS the data
        layer -- the SSOT rule forbids h5py in the layers *above* it.
        """
        import h5py

        with h5py.File(path, "r", swmr=True) as f:
            key = "kspace" if "kspace" in f else next(iter(f))
            node = f[key]
            if not isinstance(node, h5py.Dataset):
                raise TypeError(f"{path}:{key} is a {type(node).__name__}, not a Dataset")
            return int(node.shape[0])

    def _slice_indices(self, depth: int) -> list[int]:
        """``slices_per_volume`` absolute indices centred on the volume.

        Centred, and absolute -- an offset like ``-2`` would index from the END of
        the volume, quietly sampling the outermost slices instead of the middle.
        Those are mostly air, on which every metric agrees, so they would dilute
        every leaderboard cell with a region that has nothing in it.
        """
        n = min(self.slices_per_volume, depth)
        start = max(0, (depth - n) // 2)
        return list(range(start, start + n))
