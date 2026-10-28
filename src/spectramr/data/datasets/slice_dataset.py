"""NPY Slice Dataset for ULF → HF Translation.

Simple dataset for pre-paired .npy slice files with shape [2, H, W]
where channel 0 is ULF input and channel 1 is HF target.

Extracts MRI contrast (T1w, T2w, FLAIR, DWI) from filenames and combines
with field strength for physics-informed training.

Filename format: sub-{subject_id}_{contrast}_slice{slice_num}.npy
Example: sub-0015_FLAIR_slice084.npy
"""

import glob
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset, get_worker_info

logger = logging.getLogger(__name__)


# =============================================================================
# Known Contrasts and Field Strengths
# =============================================================================

KNOWN_CONTRASTS = {"T1w", "T1", "T2w", "T2", "FLAIR", "DWI", "PD"}
CONTRAST_PATTERN = re.compile(r"_(T1w|T2w|T1|T2|FLAIR|DWI|PD)_", re.IGNORECASE)
CANONICAL_CONTRASTS = ["T1w", "T2w", "FLAIR", "DWI", "PD", "unknown"]
CONTRAST_TO_IDX = {c: i for i, c in enumerate(CANONICAL_CONTRASTS)}

# Upper-cased matched token -> canonical label. Kept in lockstep with
# CONTRAST_PATTERN: returning the raw ``.upper()`` token ("T1W"/"T2W") misses the
# mixed-case CONTRAST_TO_IDX / PHYSICS_REGISTRY keys and silently relabels the
# scan as "unknown" (CLAUDE.md pitfall #16 facade / #9 silent mislabel).
_TOKEN_TO_CANONICAL = {
    "T1W": "T1w",
    "T1": "T1w",
    "T2W": "T2w",
    "T2": "T2w",
    "FLAIR": "FLAIR",
    "DWI": "DWI",
    "PD": "PD",
}


def extract_contrast_from_filename(filename: str) -> str:
    """Extract MRI contrast from filename.

    Args:
        filename: Filename like 'sub-0015_FLAIR_slice084.npy'

    Returns:
        Contrast string (T1w, T2w, FLAIR, DWI, PD) or 'unknown'
    """
    match = CONTRAST_PATTERN.search(filename)
    if not match:
        return "unknown"
    # Canonicalize to the CONTRAST_TO_IDX / PHYSICS_REGISTRY case (T1/T2 aliases
    # fold to T1w/T2w). Direct index (not .get) so a pattern token missing from
    # the map fails loud instead of silently mislabelling (pitfall #9).
    return _TOKEN_TO_CANONICAL[match.group(1).upper()]


def get_physics_for_contrast(contrast: str, field_strength: float) -> Any | None:
    """Get physics parameters for a contrast at a specific field strength.

    Combines contrast-specific timing (TR, TE, TI) with field strength B0.

    Args:
        contrast: MRI contrast (T1w, T2w, FLAIR, DWI)
        field_strength: B0 field strength in Tesla (0.064 for ULF, 3.0 for HF)

    Returns:
        PhysicsParameters instance
    """
    try:
        from spectramr.infrastructure.physics.physics_registry import (
            PHYSICS_REGISTRY,
            PhysicsParameters,
        )

        # Get base parameters from registry
        if contrast in PHYSICS_REGISTRY:
            base = PHYSICS_REGISTRY[contrast]
        elif contrast == "T1w" and "T1w" in PHYSICS_REGISTRY:
            base = PHYSICS_REGISTRY["T1w"]
        elif contrast == "T2w" and "T2w" in PHYSICS_REGISTRY:
            base = PHYSICS_REGISTRY["T2w"]
        else:
            # Fallback to T1w
            base = PHYSICS_REGISTRY.get("T1w", PHYSICS_REGISTRY["T1"])

        # Create new params with correct field strength
        return PhysicsParameters(
            TR=base.TR,
            TE=base.TE,
            TI=base.TI,
            B0=field_strength,
            contrast_type=f"{contrast}@{field_strength}T",
        )

    except ImportError:
        # Fallback if physics registry not available
        return None


# =============================================================================
# Slice Dataset with Contrast Extraction
# =============================================================================


class SliceDataset(Dataset):
    """Dataset for paired ULF/HF slices stored as .npy files.

    Each .npy file should have shape [2, H, W] where:
        - Channel 0: ULF (0.064T) slice
        - Channel 1: HF (3T) slice

    Extracts contrast from filename (T1w, T2w, FLAIR, DWI) and combines
    with field strength for physics parameters.

    Filename format: sub-{subject_id}_{contrast}_slice{slice_num}.npy

    Returns samples with:
        - input: ULF tensor [1, H, W]
        - target: HF tensor [1, H, W]
        - contrast: Extracted contrast (T1w, T2w, FLAIR, DWI)
        - subject_id: Subject ID from filename
        - slice_num: Slice number from filename
    """

    # Field strengths
    ULF_FIELD_STRENGTH = 0.064  # 64 mT
    HF_FIELD_STRENGTH = 3.0  # 3 Tesla

    def __init__(
        self,
        data_dir: str | None = None,
        transform: Any | None = None,
        normalize: bool = True,
        out_range: tuple = (-1.0, 1.0),
        normalization: Literal["percentile", "zscore", "minmax", "none"] = "percentile",
        percentile: float = 99.5,
        file_list: list[str] | None = None,
    ):
        """Initialize SliceDataset.

        Args:
            data_dir: Directory containing .npy files (ignored when
                ``file_list`` is provided).
            transform: Optional TorchIO-compatible transform.
            normalize: Whether to normalize tensors.
            out_range: Output range for normalization.
            normalization: Normalization strategy.
            percentile: Percentile for robust normalization.
            file_list: Explicit list of .npy file paths.  When supplied,
                ``data_dir`` is not scanned.  Use this to produce two
                ``SliceDataset`` splits without ``torch.utils.data.Subset``
                (which lacks ``dry_iter`` and breaks ``tio.Queue``).
        """
        self.data_dir = Path(data_dir) if data_dir is not None else Path(".")
        self.transform = transform
        self.normalize = normalize
        self.out_range = out_range
        self.normalization = normalization
        self.percentile = percentile

        if file_list is not None:
            self.files = list(file_list)
        else:
            if data_dir is None:
                raise ValueError("Either data_dir or file_list must be provided.")
            self.files = sorted(glob.glob(f"{data_dir}/*.npy"))

        if len(self.files) == 0:
            raise RuntimeError(
                "No .npy files found. "
                "Expected files with shape [2, H, W] and naming: "
                "sub-{id}_{contrast}_slice{num}.npy"
            )

        # Log contrast distribution
        contrasts = [extract_contrast_from_filename(f) for f in self.files]
        contrast_counts: dict[str, int] = {}
        for c in contrasts:
            contrast_counts[c] = contrast_counts.get(c, 0) + 1

        logger.info(
            f"[SliceDataset] Found {len(self.files)} slices\n"
            f"  Contrast distribution: {contrast_counts}"
        )

    def __len__(self) -> int:
        """__len__.

        Returns:
            int: Description.
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> tio.Subject:
        """Load a paired ULF/HF slice with extracted contrast.

        Returns:
            ``tio.Subject`` with:
                - ``input``  : ``tio.ScalarImage`` — ULF slice [C, H, W, 1]
                - ``target`` : ``tio.ScalarImage`` — HF slice  [C, H, W, 1]
                - ``contrast``, ``contrast_idx``, ``subject_id``,
                  ``slice_num``, ``file_id`` as non-image metadata fields.
        """
        filepath = self.files[idx]
        filename = Path(filepath).stem

        # --- Metadata extraction -------------------------------------------------
        contrast = extract_contrast_from_filename(filename)
        parts = filename.split("_")
        subject_id = parts[0] if parts else "unknown"
        slice_num = -1
        for p in parts:
            if p.startswith("slice"):
                try:
                    slice_num = int(p[5:])
                except ValueError as _exc:
                    logger.debug("Suppressed exception: %s", _exc)

        # --- Load -----------------------------------------------------------------
        data = np.load(filepath)
        if data.ndim != 3 or data.shape[0] != 2:
            raise ValueError(f"Expected shape [2, H, W], got {data.shape} for {filepath}")

        ulf = torch.from_numpy(data[0]).float().unsqueeze(0)  # [1, H, W]
        hf = torch.from_numpy(data[1]).float().unsqueeze(0)  # [1, H, W]

        # --- Normalise -----------------------------------------------------------
        if self.normalize:
            ulf = self._normalize_tensor(ulf)
            hf = self._normalize_tensor(hf)

        # --- Build tio.Subject ---------------------------------------------------
        # TorchIO ScalarImage expects (C, H, W, D); add trailing depth dim = 1.
        subject_kwargs: dict = {
            "input": tio.ScalarImage(tensor=ulf.unsqueeze(-1), affine=np.eye(4)),
            "target": tio.ScalarImage(tensor=hf.unsqueeze(-1), affine=np.eye(4)),
            "contrast": contrast,
            "contrast_idx": torch.tensor(
                CONTRAST_TO_IDX.get(contrast, CONTRAST_TO_IDX["unknown"]),
                dtype=torch.long,
            ),
            "subject_id": subject_id,
            "slice_num": slice_num,
            "file_id": filename,
        }

        # `input_physics` / `target_physics` were emitted here and never read
        # (audit D21: 9 writes, 0 reads tree-wide), so the two
        # `get_physics_for_contrast` lookups per sample bought nothing.

        subject = tio.Subject(**subject_kwargs)

        # --- Apply transforms (no wrap/unwrap needed) ----------------------------
        # A failing transform must surface: returning the untransformed subject
        # silently trains on unnormalized / unaugmented input (CLAUDE.md pitfall
        # #16 facade / #9 no silent fallback).
        if self.transform is not None:
            try:
                subject = self.transform(subject)
            except Exception as e:
                raise RuntimeError(
                    f"[SliceDataset] Transform failed for {filename!r}: {e}. "
                    "Refusing to return untransformed data."
                ) from e

        return subject

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply normalization to tensor.

        Delegates to :func:`spectramr.data.transforms.normalization.normalize_tensor`
        (SSOT).
        """
        from spectramr.data.transforms.normalization import (
            NormalizationConfig,
            NormalizationStrategy,
            normalize_tensor,
        )

        pct = self.percentile / 100.0 if self.percentile > 1.0 else self.percentile
        config = NormalizationConfig(
            strategy=NormalizationStrategy.from_string(self.normalization),
            percentile=pct,
            out_range=self.out_range,
            clamp=True,
        )
        return normalize_tensor(tensor, config)


# =============================================================================
# Per-slice wrapper for volumetric (3D-NIfTI) datasets
# =============================================================================


def _canonical_depth_index(spatial: tuple[int, ...]) -> int:
    """Index (within ``spatial``) of the slice/depth axis.

    For anisotropic brain volumes the in-plane is a square pair (``H == W``)
    and the slice axis is the odd one out. Given the three spatial sizes in
    stored order, return the index of the axis whose size differs from an
    otherwise-equal pair. When the pattern is ambiguous — isotropic (all
    equal) or all-distinct — return the last index, i.e. assume the volume is
    already canonical ``[..., D]`` and leave it untouched.

    This is provably identity for every correctly-oriented ``[C, H, W, D]``
    volume and only rewrites a genuinely transposed one (e.g. the sub-0011
    T1w stored ``[C, D, H, W]``, manifest shape ``[1, 150, 448, 448]``).
    """
    s = list(spatial)
    if len(s) != 3:
        return len(s) - 1
    a, b, c = s
    if a == b and c != a:
        return 2
    if a == c and b != a:
        return 1
    if b == c and a != b:
        return 0
    return 2  # isotropic or all-distinct → assume already depth-last


class SliceVolumeDataset(Dataset):
    """Expose each axial slice (or thin slab) of a volumetric dataset as one
    lower-depth sample.

    A model fed a whole ``[C, H, W, D]`` volume either crashes (a 2D
    ``AutoencoderKL`` with ``spatial_dims: 2`` → "Expected 3D/4D input, got 5D"
    in ``conv2d``) or OOMs (a 3D slab VAE running a ``D``-deep conv over all 156
    slices) — the stage-1 ULF→HF LDM VAEs (2026-06-21). Naively flattening
    ``[1, C, H, W, D] -> [D, C, H, W]`` in the strategy trades the crash for a
    ``D``-slice gradient-bearing forward → OOM.

    This wrapper indexes ``(base_idx, start)`` over a *volumetric* base dataset
    and returns a **depth-``slab_depth``** ``[C, H, W, K]`` Subject per item
    (non-overlapping windows of ``K = slab_depth`` consecutive slices), so
    ``batch_size: 1`` = one slab (gradient-light, no OOM) and a ``D``-deep
    volume expands to ``D // K`` training samples (a trailing ``D % K`` partial
    slab is dropped — a fixed-depth model can't consume it).

    - ``slab_depth=1`` → depth-1 ``[C, H, W, 1]`` slices. For a **2D** model the
      collation depth-squeeze (``squeeze_depth_dim``, auto from ``patch_size``
      depth 1) collapses the singleton back to ``[B, C, H, W]``. For a **3D**
      model set ``data.collation.squeeze_depth_dim: false`` to KEEP the depth.
    - ``slab_depth=K>1`` → ``[C, H, W, K]`` slabs for a 3D / 2.5D slab model;
      the depth is never squeezed (``patch_size`` depth ``!= 1``).

    The base subject (**post-transform**) is loaded once per volume and cached
    in a per-worker LRU (``cache_size``), then sliced in-memory. Each cache MISS
    re-decodes AND re-runs the base transform on the whole volume, so under a
    shuffled loader (window order randomized, per-worker cache) most windows
    miss and pay the full-volume cost — set ``data.slice_cache_size`` >= the
    volume count for a few-volumes / many-slices corpus to hold every volume
    resident.

    A resident volume carries the augmentation drawn when it was loaded, so with
    ``redraw=True`` (train split, augmentation on) a cached volume is evicted once it
    has served one epoch's share of its windows in this worker. The next access
    re-runs the base transform and draws new augmentation.

    Note on why the transform is not pushed down: the base's intensity
    normalization (TorchIO ``ZNormalization`` / ``RescaleIntensity``) is
    **volume-level** — it reduces over the whole ``[C,H,W,D]`` subject. Reading
    a single slice lazily through ``slice_index`` (which ``io_strategies``
    honours) and transforming that slice would change the normalization
    statistics from per-volume to per-slab, silently altering numerics. A true
    lazy read is therefore only safe when every configured transform is
    slice-invariant, and is deliberately deferred rather than made the default.

    Args:
        base: the volumetric dataset (e.g. ``UniversalMRIDataset``). Must be
            indexable and expose ``.index`` (the record list) so per-volume
            depth can be resolved from record ``shape`` metadata or the NIfTI
            header without materialising every volume up front.
        slice_axis: spatial depth axis of the ``[C, H, W, D]`` Subject tensors
            (default ``-1`` = the last axis, ``D``).
        slab_depth: number of consecutive slices per window (``1`` = single 2D
            slice; ``>1`` = a thin slab for a 3D model). Must be ``>= 1``.
        cache_size: number of recently-loaded volumes to retain (LRU).
        redraw: re-run the base transform after a cached volume has served
            ``ceil(n_windows / num_workers)`` windows, so random augmentation is
            redrawn about once per epoch instead of once per run.
    """

    def __init__(
        self,
        base: Any,
        slice_axis: int = -1,
        slab_depth: int = 1,
        cache_size: int = 2,
        canonicalize_axes: bool = True,
        redraw: bool = False,
    ) -> None:
        from collections import OrderedDict

        if slab_depth < 1:
            raise ValueError(f"slab_depth must be >= 1, got {slab_depth}.")

        self.base = base
        self.slice_axis = slice_axis
        self.slab_depth = int(slab_depth)
        # Move a permuted slice axis to ``slice_axis`` so every same-contrast
        # volume shares axis order (fixes the transposed sub-0011 T1w, which
        # otherwise feeds the VAE 150-tall slices that crash on the ÷8 decoder).
        self.canonicalize_axes = bool(canonicalize_axes)
        self._cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, Any] = OrderedDict()
        self.redraw = bool(redraw)
        self._served: dict[int, int] = {}
        self._windows_per_volume: dict[int, int] = {}

        index = getattr(base, "index", None)
        if index is None:
            raise AttributeError(
                "SliceVolumeDataset requires the base dataset to expose `.index` "
                "(the record list) so per-volume depth can be resolved."
            )
        self._index_map: list[tuple[int, int]] = []
        dropped = 0
        for base_idx in range(len(index)):
            depth = self._depth_of(base_idx)
            if depth < self.slab_depth:
                raise ValueError(
                    f"SliceVolumeDataset: record {base_idx} has depth {depth} < "
                    f"slab_depth {self.slab_depth}; cannot form a window."
                )
            # Non-overlapping windows; drop a trailing partial slab.
            n_windows = depth // self.slab_depth
            dropped += depth - n_windows * self.slab_depth
            for w in range(n_windows):
                self._index_map.append((base_idx, w * self.slab_depth))
        self._windows_per_volume = dict(Counter(base_idx for base_idx, _ in self._index_map))
        logger.info(
            "[DATASET] SliceVolumeDataset: %d volumes -> %d windows "
            "(slab_depth=%d, axis=%d, dropped %d trailing slices)",
            len(index),
            len(self._index_map),
            self.slab_depth,
            slice_axis,
            dropped,
        )

    def _depth_of(self, base_idx: int) -> int:
        """Resolve a volume's slice count cheaply (record shape → header → load)."""
        rec = self.base.index[base_idx]
        shape = rec.get("shape") if isinstance(rec, dict) else None
        if shape is not None and len(shape) >= 3:
            if self.canonicalize_axes:
                spatial = tuple(int(x) for x in shape[-3:])
                return spatial[_canonical_depth_index(spatial)]
            return int(shape[self.slice_axis])
        path = None
        if isinstance(rec, dict):
            path = rec.get("primary_path") or rec.get("path")
        if path is not None:
            try:
                # SSOT lazy NIfTI handle (no voxel load) — reads only the header
                # shape, keeping the nibabel dependency inside io_strategies.
                from spectramr.data.io_strategies import lazy_load_nifti

                return int(lazy_load_nifti(str(path)).shape[self.slice_axis])
            except Exception as exc:  # fall back to a real load
                logger.debug("Header read failed for %s (%s); loading volume", path, exc)
        subject = self._load_volume(base_idx)
        return int(self._first_tensor(subject).shape[self.slice_axis])

    @staticmethod
    def _first_tensor(subject: Any) -> torch.Tensor:
        images = subject.get_images_dict(intensity_only=False)
        return next(iter(images.values())).tensor

    def _redraw_after(self, base_idx: int) -> int:
        info = get_worker_info()
        workers = info.num_workers if info is not None else 1
        return max(1, math.ceil(self._windows_per_volume.get(base_idx, 1) / workers))

    def _load_volume(self, base_idx: int) -> Any:
        if base_idx in self._cache and not (
            self.redraw and self._served[base_idx] >= self._redraw_after(base_idx)
        ):
            self._cache.move_to_end(base_idx)
            self._served[base_idx] += 1
            return self._cache[base_idx]
        self._cache.pop(base_idx, None)
        subject = self.base[base_idx]
        if self.canonicalize_axes:
            subject = self._canonicalize_subject(subject)
        self._cache[base_idx] = subject
        self._served[base_idx] = 1
        while len(self._cache) > self._cache_size:
            evicted, _ = self._cache.popitem(last=False)
            self._served.pop(evicted, None)
        return subject

    def _canonicalize_subject(self, subject: Any) -> Any:
        """Move every image's slice/depth axis to ``self.slice_axis``.

        Identity for correctly-oriented volumes; only rewrites a transposed
        one. Rebuilds the Subject (TorchIO images are not mutated in place).
        """
        target = self.slice_axis
        new_images: dict[str, Any] = {}
        moved = False
        for key, image in subject.get_images_dict(intensity_only=False).items():
            tensor = image.tensor
            spatial = tuple(int(x) for x in tensor.shape[-3:])
            depth_axis = tensor.ndim - 3 + _canonical_depth_index(spatial)
            tgt = target if target >= 0 else tensor.ndim + target
            if depth_axis != tgt:
                tensor = torch.movedim(tensor, depth_axis, tgt).contiguous()
                moved = True
            new_images[key] = type(image)(tensor=tensor, affine=image.affine)
        if not moved:
            return subject
        logger.warning(
            "[DATASET] SliceVolumeDataset: canonicalised a transposed volume to "
            "depth-last (the slice axis was permuted into an in-plane position — "
            "e.g. the sub-0011 T1w stored [C, D, H, W])."
        )
        new_subject = tio.Subject(**new_images)
        for key, value in subject.items():
            if key not in new_images and not isinstance(value, tio.Image):
                new_subject[key] = value
        return new_subject

    def dry_iter(self) -> list[tio.Subject]:
        """Per-window Subject shells for TorchIO ``Queue`` compatibility.

        ``Queue`` calls ``dry_iter()`` to size ``iterations_per_epoch`` without
        loading voxels; it needs an indexable ``Sequence[Subject]``, one per
        delivered sample. Each item of this dataset is one window, so emit
        ``len(self)`` stubs carrying the originating volume's path metadata.
        Mirrors ``UniversalMRIDataset.dry_iter`` (regression: the slat_slab
        ``'SliceVolumeDataset' object has no attribute 'dry_iter'`` crash).
        """
        _stub = torch.zeros(1, 1, 1, 1)
        subjects: list[tio.Subject] = []
        for base_idx, start in self._index_map:
            rec = self.base.index[base_idx]
            if isinstance(rec, dict):
                path = str(rec.get("primary_path", rec.get("path", "unknown")))
                file_id = rec.get("file_id", Path(path).stem)
            else:
                path, file_id = "unknown", "unknown"
            subjects.append(
                tio.Subject(
                    input=tio.ScalarImage(tensor=_stub),
                    path=path,
                    file_id=file_id,
                    slice_index=start,
                )
            )
        return subjects

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> tio.Subject:
        base_idx, start = self._index_map[idx]
        subject = self._load_volume(base_idx)

        window = torch.arange(start, start + self.slab_depth)
        sliced: dict[str, Any] = {}
        for key, image in subject.get_images_dict(intensity_only=False).items():
            tensor = image.tensor
            # Take the `slab_depth`-deep window along the depth axis (the depth
            # is kept as a singleton when slab_depth == 1, for the squeeze).
            sl = tensor.index_select(self.slice_axis, window.to(tensor.device)).clone()
            sliced[key] = type(image)(tensor=sl, affine=image.affine)

        new_subject = tio.Subject(**sliced)
        # Carry forward non-image metadata (path, file_id, …).
        image_keys = set(sliced)
        for key, value in subject.items():
            if key not in image_keys and not isinstance(value, tio.Image):
                new_subject[key] = value
        new_subject["slice_index"] = start
        return new_subject
