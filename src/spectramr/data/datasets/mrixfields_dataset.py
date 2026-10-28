"""Multi-field, multi-contrast paired dataset for MICCAI MRIxFields2026.

Each travelling-volunteer subject is scanned at five field strengths
``{0.1, 1.5, 3, 5, 7}`` T with three contrasts (T1w / T2w / T2-FLAIR);
magnitude-only, MNI 0.5 mm, values in ``[0, 1]``. This dataset forms ordered
``(source @ b_s, target @ b_t)`` translation pairs *within* a ``pairing_group``
(default ``"{subject_id}|{contrast}"``) and emits BOTH field values so the
downstream strategy can condition the renderer on the target field (B-3.8 / B-1.9)
or integrate a velocity field along ``tau = log10(B0)`` (B-3.1, field-flow).

Data SSOT (CLAUDE.md #7 / #11): this module is the only place that turns a file
into samples; higher layers consume it via ``DataPipelineDirector``. The default
NIfTI loader is imported lazily so torch-less test collection still works and so
an injected ``image_loader`` (tests) bypasses disk entirely.
"""

from __future__ import annotations

import functools
import logging
from collections import OrderedDict
from collections.abc import Callable
from itertools import permutations
from typing import Any, ClassVar

import torch

# Canonical contrast-id mapping (the three challenge contrasts; tolerant aliases).
logger = logging.getLogger(__name__)

_CONTRAST_INDEX = {
    "T1w": 0,
    "T2w": 1,
    "T2FLAIR": 2,
    "FLAIR": 2,
    "T2-FLAIR": 2,
    "T2_FLAIR": 2,
}
_POLICIES = (
    "fixed_target",
    "all_pairs",
    "ulf_source",
    "multi_source",
    "multi_contrast",
    "multi_field",
    "prior",
)
# Policies whose containers live in ``_tuples`` rather than ``_pairs``. Enumerated
# ONCE: the predicate was previously spelled out at the volume-mode refusal, the
# build dispatch, ``_containers`` and ``__getitem__``, and a policy added to three
# of the four reads as a complete change while silently mis-serving the fourth.
_TUPLE_POLICIES = ("multi_source", "multi_contrast", "multi_field")
_ULF_FIELD = 0.1
# Sample keys carrying image tensors — wrapped as ``tio.ScalarImage`` before the
# injected ``tio.Compose`` runs (the rest are scalars / ids and ride through).
_IMAGE_KEYS = ("input", "target", "sources", "heldout")

# Slice-serving modes for ``mrixfields_slice_mode`` (see the dataset docstring).
_SLICE_MODES = ("central", "all_slices", "volume")

# Foreground slice filter (``all_slices`` mode). A depth slice counts as foreground
# when the fraction of voxels above ``_FOREGROUND_EPS`` (measured on the GLOBALLY
# [0,1]-normalised volume, not the per-slice renorm) reaches ``_FOREGROUND_MIN_FRAC``.
# The global scale is load-bearing: ``_extract_chw_unit`` renorms every SERVED slice to
# [0,1], which amplifies a faint-noise air slice to full scale (the #171 pathology —
# an unshuffled val loader over per-slice data would otherwise serve top-of-head air as
# a full-contrast "brain"). Judging foreground on the whole-volume scale — where air is
# ~0 — is precisely what keeps those air slices OUT of the loader; the per-slice renorm
# is cosmetic once the air is gone. Threshold picked so MNI skull-stripped air (≈0% of
# voxels lit) is dropped while a brain slice (tens of % lit) is kept.
_FOREGROUND_EPS = 1e-3
_FOREGROUND_MIN_FRAC = 0.02  # >= 2% of the slice above eps -> foreground


# The derived slice is [1, H, W] float32 (~0.6 MB at 364x436) and the manifests
# hold 45 train + 45 val volumes, so the whole working set caches in <100 MB per
# worker.
_SLICE_CACHE_MAXSIZE = 128

# Default resident-volume budget for ``all_slices`` / ``volume`` mode. A 364x436x364
# volume is ~0.62 MB/slice x 364 ~= 226 MB in float32, so holding ALL ~90 volumes
# (~20 GB, x the DataLoader workers) is an OOM — refused. This is the number of volumes a
# worker keeps decoded at once; ~226 MB each, so the default 6 is ~1.4 GB/worker.
#
# It is a BUDGET, not a hit-rate tuning knob. What makes ``all_slices`` affordable is
# pairing it with ``VolumeBlockedSliceSampler``, which orders the epoch so that the
# containers sharing a resident volume set are consumed together — the same
# load-a-few-volumes / emit-many-slices / shuffle-the-buffer algorithm ``tio.Queue``
# uses, expressed as a torch Sampler because this dataset emits plain dicts rather than
# ``tio.Subject`` (see the sampler's docstring for why the Queue itself cannot be used).
# Under that order one decode is amortised over every foreground slice of every container
# the volume participates in: a 5-field ``all_pairs`` group is 20 containers drawn from 5
# volumes, so 5 decodes serve ~5,000 samples.
#
# Without the sampler a shuffled loader visits each (container, slice) in random order and
# NO attainable budget gives intra-epoch reuse over the ~45-volume / ~10 GB working set —
# which is what made an unordered ``all_slices`` a starvation trap.
#
# The floor is the per-``__getitem__`` working set: ``_getitem_multi_source`` touches N
# sources PLUS the target, so the corpus' four source fields need 5 resident at once. A
# budget of 4 against that is a cyclic access of period 5 over capacity 4 — a hit rate of
# exactly ZERO. ``test_volume_budget_exceeds_worst_case_working_set`` pins the
# relationship so a fifth source field cannot silently re-break it. The value is a config
# knob (``data.mrixfields_max_resident_volumes``); this is only the default.
_VOLUME_CACHE_MAXSIZE = 6

#: Per-worker RAM sanity bound for the decoded-volume cache (#198). The count
#: above is the real knob; this only catches the case where that count, at THIS
#: corpus' volume size, cannot fit. 4 GiB leaves ~3x headroom over the corpus
#: figure the schema quotes (6 x ~226 MB = ~1.4 GB) — generous enough not to
#: fire on a working setup, tight enough to catch a volume-size surprise before
#: `num_workers` multiplies it into a host oom_kill.
_VOLUME_CACHE_MAX_BYTES = 4 * 2**30


def _nifti_depth(path: str) -> int | None:
    """Depth of a 3-D NIfTI, read from the header alone (no voxel IO).

    ``None`` for any other rank, which routes the caller back to the
    full-volume load that ``_to_chw_unit`` already knows how to collapse.
    """
    import nibabel as nib

    shape = nib.load(path).shape
    return int(shape[-1]) if len(shape) == 3 else None


@functools.lru_cache(maxsize=_SLICE_CACHE_MAXSIZE)
def _load_central_slice(path: str) -> torch.Tensor:
    """Decode only the central depth slice of ``path``; memoised per worker.

    ``_to_chw_unit`` keeps a single central slice, so decoding the full volume
    was ~99.7% waste: these are 364x436x364 ``.nii.gz`` and the metadata-less
    ``NiftiStrategy.load`` took the ``get_fdata()`` branch, materialising them
    as float64 (~462 MB) on EVERY ``__getitem__``. Four prefetching workers
    doing that starved the GPU to ~0% util and OOM-killed the DataLoader.
    ``NiftiStrategy.load`` already supports a lazy ``dataobj[..., i]`` read —
    this asks for it, and memoises the result because the same volume is
    referenced by many pairs and revisited every epoch.

    ``NiftiStrategy.load`` returns a dict ``{"data": tensor, ...}``, NOT a bare
    array; passing the dict to ``torch.as_tensor`` raises ``RuntimeError: Could
    not infer dtype of dict`` (the 2026-06-20 DataLoader crash).
    """
    from spectramr.data.io_strategies import IOStrategyFactory

    depth = _nifti_depth(path)
    metadata = {"slice_index": depth // 2} if depth is not None else None

    loaded = IOStrategyFactory.get("nifti").load(path, metadata=metadata)
    if isinstance(loaded, dict):
        arr = loaded.get("image", loaded.get("data"))
        if arr is None:
            raise ValueError(
                f"nifti loader returned a dict with no 'image'/'data' key for "
                f"{path!r}; keys present: {sorted(loaded)}"
            )
    else:
        arr = loaded
    return arr if isinstance(arr, torch.Tensor) else torch.as_tensor(arr)


def _default_nifti_loader(path: str) -> torch.Tensor:
    """Cached central-slice load. Clones so transforms cannot poison the cache."""
    return _load_central_slice(path).clone()


def _load_full_volume(path: str) -> torch.Tensor:
    """Decode the WHOLE ``[C, H, W, D]`` volume (``all_slices`` / ``volume`` modes).

    ``all_slices`` needs every depth slice (both to scan for foreground once at
    construction and to serve arbitrary slice indices), and ``volume`` needs the
    whole 3-D volume, so unlike ``_load_central_slice`` this cannot take the lazy
    single-slice ``dataobj[..., i]`` path — it asks ``NiftiStrategy.load`` for the
    full volume (the ``get_fdata()`` branch). Cast to float32 so the cache holds
    ~226 MB per volume rather than ~452 MB (float64); see ``_VOLUME_CACHE_MAXSIZE``
    for the LRU sizing arithmetic. ``NiftiStrategy.load`` returns a dict
    ``{"data": tensor, ...}``, not a bare array (same contract as the central load).
    """
    from spectramr.data.io_strategies import IOStrategyFactory

    loaded = IOStrategyFactory.get("nifti").load(path)  # no slice metadata -> full volume
    if isinstance(loaded, dict):
        arr = loaded.get("image", loaded.get("data"))
        if arr is None:
            raise ValueError(
                f"nifti loader returned a dict with no 'image'/'data' key for "
                f"{path!r}; keys present: {sorted(loaded)}"
            )
    else:
        arr = loaded
    tensor = arr if isinstance(arr, torch.Tensor) else torch.as_tensor(arr)
    return tensor.float()


def _default_nifti_volume_loader(path: str) -> torch.Tensor:
    """Uncached whole-volume decode. Callers cache via :class:`_VolumeLRU`.

    The cache moved from a module-level ``functools.lru_cache`` onto the dataset INSTANCE
    for two reasons. First, ``lru_cache``'s maxsize is fixed at decoration, so
    ``data.mrixfields_max_resident_volumes`` could not have been a real knob — it would
    have been an advertised setting nothing reads (pitfall #15). Second, a module-level
    cache is shared by the train and val datasets in one process while the budget is meant
    to be per-worker; an instance cache is per-worker by construction, because each
    DataLoader worker gets its own copy of the dataset.

    Returned WITHOUT a defensive copy. The central-slice loader clones because its
    ~0.6 MB payload makes the copy free; cloning here copied the whole ~226 MB volume to
    extract one ~0.6 MB slice — 44 ms per call against 0.14 ms for slice-then-copy, i.e.
    ~300x waste on every load, cache HIT included. Safe because no caller can alias a
    cached tensor into a transform: every coercion materialises a fresh tensor before
    returning (``_extract_chw_unit`` / ``_to_chw_unit`` / ``_volume_to_cdhw_unit`` all end
    in ``clamp``, which never returns a view) and ``_foreground_slices`` only reads.
    ``test_volume_cache_cannot_be_poisoned_by_a_transform`` pins the invariant.
    """
    return _load_full_volume(path)


class _VolumeLRU:
    """Least-recently-used cache of decoded volumes, sized at construction.

    Deliberately not ``functools.lru_cache``: the budget is a config knob, and
    ``lru_cache`` fixes its maxsize at decoration time. Small enough to be obvious — an
    ``OrderedDict`` used as an LRU — because the interesting behaviour is the ACCESS ORDER
    that fills it (``VolumeBlockedSliceSampler``), not the eviction policy.
    """

    def __init__(
        self,
        maxsize: int,
        loader: Callable[[str], torch.Tensor],
        max_bytes: int | None = _VOLUME_CACHE_MAX_BYTES,
    ) -> None:
        if maxsize < 1:
            raise ValueError(f"_VolumeLRU maxsize must be >= 1; got {maxsize}.")
        self._maxsize = int(maxsize)
        self._loader = loader
        self._max_bytes = max_bytes
        self._checked_footprint = False
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _check_footprint(self, volume: torch.Tensor) -> None:
        """Raise if this budget, at this volume size, cannot fit in RAM.

        The budget stays denominated in VOLUMES, deliberately. It is a
        working-set count, not a memory knob: ``_getitem_multi_source`` touches
        N sources plus the target, so a budget below the widest container's
        volume count makes every sample evict the volume the next one needs — a
        cyclic access of period 5 over capacity 4 is a hit rate of exactly zero.
        Re-denominating to bytes would let a byte budget silently pick a count
        under that floor.

        So the byte bound is a SANITY CHECK layered on top, and it can only run
        after the first decode: nothing knows a volume's size at construction.
        ~226 MB per volume is the corpus figure the schema quotes, but a
        higher-resolution or multi-echo corpus scales it, and the product is
        per-worker — multiplied again by ``num_workers``. Without this the
        symptom is a host oom_kill with no attribution, hours in.
        """
        if self._checked_footprint or self._max_bytes is None:
            return
        self._checked_footprint = True

        per_volume = volume.element_size() * volume.nelement()
        projected = per_volume * self._maxsize
        if projected <= self._max_bytes:
            return
        raise ValueError(
            f"mrixfields volume cache would hold {self._maxsize} x "
            f"{per_volume / 2**20:.0f} MB = {projected / 2**30:.2f} GB per "
            f"DataLoader worker, over the {self._max_bytes / 2**30:.2f} GB "
            "sanity bound — and that is multiplied again by data.loader."
            "num_workers. Lower data.mrixfields.max_resident_volumes, but NOT "
            "below the widest container's volume count (multi_source needs N "
            "sources + the target resident at once, so the four-source corpus "
            "needs 5; multi_field needs len(fields) + len(heldout_fields)); "
            "under that floor every sample evicts the volume the next "
            "one needs. Otherwise reduce num_workers, or raise the bound "
            "deliberately if the node really has the RAM."
        )

    def __call__(self, path: str) -> torch.Tensor:
        if path in self._store:
            self._store.move_to_end(path)
            self.hits += 1
            return self._store[path]
        self.misses += 1
        volume = self._loader(path)
        self._check_footprint(volume)
        self._store[path] = volume
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)
        return volume

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    @property
    def maxsize(self) -> int:
        return self._maxsize


def canonical_contrasts(index: list[dict[str, Any]]) -> list[str]:
    """Distinct contrasts in ``index``, in canonical ``_CONTRAST_INDEX`` order (deduped
    across aliases, e.g. T2FLAIR/FLAIR -> one class).

    Module-level so the instantiator can pin the SAME set from the FULL manifest and
    hand it to both the train and val datasets (uniform ``multi_contrast`` stack arity).
    """
    by_idx: dict[int, str] = {}
    for rec in index:
        c = str(rec.get("contrast", "T1w"))
        by_idx.setdefault(_CONTRAST_INDEX.get(c, 99), c)
    return [by_idx[k] for k in sorted(by_idx)]


#: Named so the audit can point a stale-manifest failure at the SSOT generator,
#: mirroring the runtime raise in ``MRIxFieldsPairedDataset._build_pairs``.
_REGEN_HINT = (
    "every pairing group holds a SINGLE field -- an unpaired-by-design prospective "
    "manifest built BEFORE ordinal pairing. REGENERATE it: "
    "`python scripts/data/build_mrixfields2026_manifest.py` (it re-keys the "
    "Validating/Testing_prospective splits so same-rank volumes pair across fields). "
    "The manifest is a derived, gitignored artifact; the committed generator is the SSOT."
)


def mrixfields_pairing_feasibility(
    index: list[dict[str, Any]],
    *,
    policy: str,
    target_field: float | None,
) -> tuple[bool, str | None]:
    """Static *necessary-condition* test for whether ``policy`` can form >=1 pair.

    Returns ``(feasible, reason_if_infeasible)``. A ``False`` is a **guarantee**
    of 0 pairs/tuples on ``index`` — the exact pre-flight signal the audit needs
    to reject the 2026-07-07 stale-manifest cluster failure at Tier 1 instead of
    at dataloader-build time. A ``True`` is **not** a promise of pairs: the full
    sufficient conditions live in :meth:`MRIxFieldsPairedDataset._build_pairs`
    (and ``_build_multi_source`` / ``_build_multi_contrast`` / ``_build_multi_field``)
    and are kept there
    to avoid the audit drifting from the loader. The audit only ever rejects the
    provably-empty case, so this never false-positives.

    ``prior`` is always feasible (it emits identity pairs for unpaired
    singletons, so it can never yield 0). The field-pinned policies all require a
    ``pairing_group`` holding >= 2 records (the singleton signature is the stale
    manifest) plus the pinned field(s) being present.
    """
    if policy == "prior":
        return True, None

    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in index:
        key = rec.get("pairing_group") or f"{rec['subject_id']}|{rec['contrast']}"
        groups.setdefault(key, []).append(rec)
    max_group = max((len(v) for v in groups.values()), default=0)
    fields = {float(r["field_strength"]) for r in index}

    def present(value: float) -> bool:
        return any(abs(f - value) <= 1e-6 for f in fields)

    def _singleton_reason() -> str:
        present_sorted = sorted(fields)
        return f"{_REGEN_HINT} (fields present across the manifest = {present_sorted})."

    if policy == "all_pairs":
        if max_group < 2:
            return False, _singleton_reason()
        return True, None

    if policy == "fixed_target":
        if target_field is None:
            return False, "fixed_target requires mrixfields_target_field (Tesla)."
        if not present(float(target_field)):
            return False, (
                f"target field {target_field} T is absent; fields present = {sorted(fields)}."
            )
        if max_group < 2:
            return False, _singleton_reason()
        return True, None

    if policy == "ulf_source":
        if not present(_ULF_FIELD):
            return False, (
                f"ulf_source needs a {_ULF_FIELD} T source; fields present = {sorted(fields)}."
            )
        if not any(f > _ULF_FIELD + 1e-6 for f in fields):
            return False, "ulf_source needs a field above 0.1 T to translate to."
        if max_group < 2:
            return False, _singleton_reason()
        return True, None

    if policy == "multi_source":
        if target_field is None:
            return False, "multi_source requires mrixfields_target_field (Tesla)."
        tgt = float(target_field)
        if not present(tgt):
            return False, (f"target field {tgt} T is absent; fields present = {sorted(fields)}.")
        sources = {f for f in fields if f < tgt - 1e-6}
        if len(sources) < 2:
            return False, (
                f"multi_source needs >=2 distinct source fields below {tgt} T; "
                f"found {sorted(sources)}."
            )
        if max_group < 2:
            return False, _singleton_reason()
        return True, None

    if policy == "multi_contrast":
        if target_field is None:
            return False, "multi_contrast requires mrixfields_target_field (Tesla)."
        if not present(float(target_field)):
            return False, (
                f"target field {target_field} T is absent; fields present = {sorted(fields)}."
            )
        return True, None

    # Unknown policy — defer to the dataset __init__ raise (no silent verdict).
    return True, None


class MRIxFieldsPairedDataset(torch.utils.data.Dataset):
    """Travelling-volunteer cross-field translation pairs.

    Args:
        index: Parsed manifest records; each is one field-snapshot with keys
            ``primary_path``, ``field_strength`` (Tesla), ``contrast``,
            ``subject_id`` and (optionally) ``pairing_group``.
        pairing_policy: ``"all_pairs"`` (any ordered pair; any-to-any),
            ``"fixed_target"`` (target pinned to ``target_field``),
            ``"ulf_source"`` (source pinned to 0.1 T), ``"multi_source"``
            (B-1.1 travelling-volunteer TUPLE: one item per subject-contrast group
            carrying ALL source fields ``< target_field`` stacked as ``sources``
            ``[N,1,H,W]`` + ``field_strengths`` ``[N]`` against the shared ``target``;
            ``input`` is the first source so unpack/validation stay compatible), or
            ``"prior"`` (adaptive: ULF→HF pairs for a group that has them — so a paired
            travelling-volunteer manifest validates real recon — and ``(rec, rec)``
            *identity* pairs for unpaired singletons — so an unpaired pool, e.g. the
            retrospective field cohort, trains a field-conditioned prior on each image
            as its own ``target``. Used by the prior-based DPS/MAP arms to train the
            prior on the large unpaired pool while validating on the small paired set).
        target_field: required when ``pairing_policy in {'fixed_target','multi_source'}``.
        slice_mode: which slice(s) each volume serves. ``"all_slices"`` (default, the
            correct one) expands each pair/tuple to EVERY foreground slice (an
            ``_index_map`` of ``(container_idx, slice_idx)``; ``len`` grows to the total
            foreground-slice count) with the air-slice filter below dropping empty MNI
            top/bottom slices — the same ``slice_idx`` is served for source and target
            (co-registered MNI 0.5 mm). The volumes are 3-D and every foreground slice is
            training signal; serving one slice wastes ~363/364 of each volume.
            ``"central"`` keeps ONLY the middle 2-D slice via ``_to_chw_unit`` — a
            debug/smoke shortcut, not a training mode. ``"volume"`` emits the whole
            ``[C,H,W,D]`` volume (``spatial_dims: 3`` arms); it is NOT supported for the
            stacking policies -- ``_TUPLE_POLICIES``, which owns that set -- because the
            stack axis on top of a volume would emit a 5-D tensor no strategy consumes,
            and it raises for those.
        transform: optional ``tio.Compose`` (or any tio transform). Applied via
            ``_apply_transform``, which wraps the image keys in a ``tio.Subject``
            first — a raw dict makes TorchIO raise (see that method's docstring).
        image_loader: injectable loader ``path -> Tensor`` (tests); defaults to
            the real NIfTI IO strategy. For ``central`` the default reads only the
            central slice; for ``all_slices`` / ``volume`` it reads the whole volume
            (an injected loader is used unchanged for every mode — it returns the whole
            (small) test volume, from which the requested slice(s) are selected).
    """

    def __init__(
        self,
        index: list[dict[str, Any]],
        *,
        pairing_policy: str = "all_pairs",
        target_field: float | None = None,
        expose_target_field: bool = True,
        source_fields: list[float] | None = None,
        fields: list[float] | None = None,
        heldout_fields: list[float] | None = None,
        output_contrast: str | None = None,
        contrasts: list[str] | None = None,
        slice_mode: str = "all_slices",
        max_resident_volumes: int = _VOLUME_CACHE_MAXSIZE,
        rescale_per_image: bool = False,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        image_loader: Callable[[str], torch.Tensor] | None = None,
    ) -> None:
        if pairing_policy not in _POLICIES:
            raise ValueError(
                f"Unknown mrixfields pairing_policy {pairing_policy!r}; "
                f"expected one of {_POLICIES}."
            )
        if slice_mode not in _SLICE_MODES:
            # No silent fallback to 'central' (#9): an unknown mode is a config error.
            raise ValueError(
                f"Unknown mrixfields slice_mode {slice_mode!r}; expected one of {_SLICE_MODES}."
            )
        if slice_mode == "volume" and pairing_policy in _TUPLE_POLICIES:
            # Honest refusal, not a facade: 'volume' with a tuple policy would have to
            # emit a 5-D [N,C,H,W,D] 'sources'/'input' stack that no strategy consumes.
            raise ValueError(
                f"slice_mode='volume' is not supported with pairing_policy="
                f"{pairing_policy!r} (it would emit a 5-D sources stack no strategy "
                "consumes); use 'central' or 'all_slices'."
            )
        if (
            pairing_policy in ("fixed_target", "multi_source", "multi_contrast")
            and target_field is None
        ):
            raise ValueError(f"pairing_policy={pairing_policy!r} requires target_field (Tesla).")
        if pairing_policy == "multi_field":
            if not fields:
                raise ValueError(
                    "pairing_policy='multi_field' requires a non-empty "
                    "data.mrixfields.fields list (the stack, in declared order)."
                )
            for name, seq in (("fields", fields), ("heldout_fields", heldout_fields or [])):
                if len(seq) != len({float(f) for f in seq}):
                    raise ValueError(
                        f"data.mrixfields.{name} has duplicate field strengths {list(seq)}; "
                        "each field may appear at most once."
                    )
            overlap = {float(f) for f in fields} & {float(f) for f in (heldout_fields or [])}
            if overlap:
                # The whole point of the policy is that the held-out field is NOT fitted.
                # An overlap would put it in both halves and the extrapolation test would
                # measure nothing while still producing a number.
                raise ValueError(
                    f"data.mrixfields.fields and heldout_fields overlap on {sorted(overlap)}; "
                    "a held-out field must not also be in the fitted stack."
                )
        elif fields or heldout_fields:
            # Never accept a knob this policy does not read (non-negotiable 8).
            raise ValueError(
                f"data.mrixfields.fields/heldout_fields are only read by "
                f"pairing_policy='multi_field'; got {pairing_policy!r}."
            )
        self._stack_fields = [float(f) for f in (fields or [])]
        self._heldout_fields = [float(f) for f in (heldout_fields or [])]
        self._policy = pairing_policy
        self._slice_mode = slice_mode
        self._target_field = target_field
        self._expose_target_field = expose_target_field
        # multi_contrast: which contrast the target image is (default: first source
        # contrast in canonical order).
        self._output_contrast = output_contrast
        # multi_contrast: the canonical source-contrast set, PINNED by the instantiator
        # from the full manifest so train and val stack the SAME channel count (and it
        # matches model.in_channels). Deriving it per-split would drift the arity when a
        # split happens to lack a contrast, crashing the encoder's first conv.
        self._pinned_contrasts = list(contrasts) if contrasts is not None else None
        # An explicit source-field set (passed by the instantiator from the FULL manifest)
        # pins N identically across the train/val splits — recomputing per-split would give
        # train and val different consensus arity under heterogeneous field coverage.
        self._fixed_source_fields = (
            sorted(float(f) for f in source_fields) if source_fields is not None else None
        )
        self._transform = transform
        # The resident-volume budget is per-WORKER (each DataLoader worker holds its own
        # dataset copy), and it must be at least the widest container's volume count or
        # every item evicts what the next one needs — validated here rather than left to
        # surface as a mysterious throughput collapse.
        self._max_resident_volumes = int(max_resident_volumes)
        if self._max_resident_volumes < 2:
            raise ValueError(
                f"max_resident_volumes must be >= 2 (a pair reads a source AND a "
                f"target); got {self._max_resident_volumes}."
            )
        self._volume_cache: _VolumeLRU | None = None
        # An injected loader is used verbatim for every mode (it returns the whole small
        # test volume). The default loader depends on the mode: 'central' keeps the cheap
        # lazy central-slice read; 'all_slices'/'volume' decode the whole [C,H,W,D] volume
        # through a per-instance LRU so the budget knob is real and per-worker (see
        # _default_nifti_volume_loader for why it is not a functools.lru_cache).
        if image_loader is not None:
            self._load = image_loader
        elif slice_mode == "central":
            self._load = _default_nifti_loader
        else:
            self._volume_cache = _VolumeLRU(
                self._max_resident_volumes, _default_nifti_volume_loader
            )
            self._load = self._volume_cache
        # Per-sample image coercion when a WHOLE slice/volume is served (slice_idx is None):
        # 'central' collapses to the central [1,H,W] slice; 'volume' normalises the whole
        # [C,H,W,D]. 'all_slices' bypasses this and calls _extract_chw_unit(vol, slice_idx).
        # Per-image min-max renorm is OPT-IN: it erases the cross-field intensity
        # relationship this dataset exists to teach (see _extract_chw_unit).
        self._rescale_per_image = bool(rescale_per_image)
        self._coerce = functools.partial(
            self._volume_to_cdhw_unit if slice_mode == "volume" else self._to_chw_unit,
            rescale=self._rescale_per_image,
        )
        # multi_source emits tuples (list[src], tgt); the pair policies emit (src, tgt).
        # Always defined: only the tuple builders assign it, and a reader that has to
        # guard with getattr cannot tell "no groups skipped" from "policy never counted".
        self._skipped_groups = 0
        self._tuples: list[Any] = []
        if pairing_policy == "multi_source":
            self._tuples = self._build_multi_source(index)
            self._pairs = []
        elif pairing_policy == "multi_field":
            self._tuples = self._build_multi_field(index)
            self._pairs = []
        elif pairing_policy == "multi_contrast":
            self._tuples = self._build_multi_contrast(index)
            self._pairs = []
        else:
            self._pairs = self._build_pairs(index)
        if self._skipped_groups:
            # Written by the tuple builders and, until now, read only by a test -- so a
            # corpus quietly losing incomplete groups looked identical to a complete one.
            logger.warning(
                "[DATASET] mrixfields pairing_policy=%s skipped %d incomplete group(s); "
                "%d container(s) built. A group is skipped when it lacks a declared "
                "field/source, so a large count means the manifest is not the "
                "complete-by-design travelling-volunteer set.",
                pairing_policy,
                self._skipped_groups,
                len(self._tuples),
            )
        # all_slices: expand each pair/tuple to its target volume's FOREGROUND slices.
        if slice_mode == "all_slices":
            self._fg_cache: dict[str, list[int]] = {}
            self._index_map = self._build_index_map()

    def _build_pairs(
        self, index: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for rec in index:
            key = rec.get("pairing_group") or f"{rec['subject_id']}|{rec['contrast']}"
            groups.setdefault(key, []).append(rec)
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for recs in groups.values():
            if self._policy == "prior":
                # Adaptive prior policy. A group that carries a 0.1 T source AND a higher
                # field is a real travelling-volunteer pairing (the prospective val set) →
                # emit ULF→HF pairs so validation grades the actual recon. An unpaired
                # singleton (the retrospective cohort: one field per volunteer) has no
                # partner → emit an IDENTITY pair so the field-conditioned prior trains on
                # the image as its own target (compute_ulf_dps_loss reads batch['target']).
                ulf = [r for r in recs if abs(float(r["field_strength"]) - _ULF_FIELD) <= 1e-6]
                higher = [r for r in recs if abs(float(r["field_strength"]) - _ULF_FIELD) > 1e-6]
                if ulf and higher:
                    pairs.extend((u, tgt) for u in ulf for tgt in higher)
                else:
                    pairs.extend((r, r) for r in recs)
                continue
            for src, tgt in permutations(recs, 2):
                if self._policy == "fixed_target":
                    assert self._target_field is not None  # enforced in __init__
                    if abs(float(tgt["field_strength"]) - float(self._target_field)) > 1e-6:
                        continue
                if self._policy == "ulf_source" and (
                    abs(float(src["field_strength"]) - _ULF_FIELD) > 1e-6
                ):
                    continue
                pairs.append((src, tgt))
        if not pairs and self._policy in ("fixed_target", "ulf_source", "all_pairs"):
            # Fail-fast (symmetry with multi_source): a pinned policy that matches nothing
            # would otherwise yield a SILENTLY empty dataset (e.g. fixed_target=7.0 on a proxy
            # manifest with no 7T column) — a zero-batch loader, not a clear error (#9/#15).
            # `all_pairs` belongs here too: it needs >= 2 fields per pairing group, and a
            # stale / singleton-group manifest (the recurring cluster failure every other
            # branch of this method guards against) silently yields len(dataset) == 0 for
            # all 27 task-3 arms. The Tier-1 `mrixfields_pairing_feasibility` hook catches
            # it pre-flight only when the manifest is READABLE — and data/manifests/ is
            # gitignored, so the audit skips it on any host that lacks the file.
            present = sorted({float(r["field_strength"]) for r in index})
            pin = self._target_field if self._policy == "fixed_target" else _ULF_FIELD
            max_group = max((len(v) for v in groups.values()), default=0)
            if max_group <= 1:
                # Every pairing group holds a SINGLE field -> an unpaired-by-design
                # Validating/Testing_prospective manifest that was built BEFORE ordinal
                # pairing (each subject scanned at one field, renumbered per field). The
                # generator re-keys same-rank == same anatomy so groups span all fields;
                # a stale manifest never got that pass. This is the common cluster failure
                # (the generator was updated but the gitignored manifest was not rebuilt).
                hint = (
                    "every pairing group holds a SINGLE field, so this is an unpaired-by-design "
                    "prospective manifest built BEFORE ordinal pairing -- REGENERATE it: "
                    "`python scripts/data/build_mrixfields2026_manifest.py` (it re-keys the "
                    "Validating/Testing_prospective splits so same-rank volumes pair across "
                    "fields). The manifest is a derived, gitignored artifact -- the committed "
                    "generator is the SSOT and must be re-run wherever it is stale."
                )
            else:
                hint = (
                    "the train/val split stranded the pinned field in the OTHER split (a flat "
                    "record slice on a field-sorted manifest) -- the builder must re-split "
                    "GROUP-AWARE for field-pinned policies (see "
                    "DatasetInstantiator._create_mrixfields). If the manifest genuinely lacks "
                    "the pinned field, point at the full travelling-volunteer corpus."
                )
            if self._policy == "all_pairs":
                # Unpinned: all_pairs needs >= 2 distinct fields inside a pairing group,
                # so quote the group size rather than a pinned field that does not exist.
                raise ValueError(
                    f"pairing_policy='all_pairs' produced 0 pairs: it needs at least 2 "
                    f"records per pairing group (subject|contrast) to form an ordered "
                    f"(source, target) pair, but the largest group holds {max_group}; "
                    f"fields present = {present}. This usually means {hint}"
                )
            raise ValueError(
                f"pairing_policy={self._policy!r} produced 0 pairs: no group matches the pinned "
                f"field {pin} T; fields present = {present}. This usually means {hint}"
            )
        return pairs

    def _build_multi_source(
        self, index: list[dict[str, Any]]
    ) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
        """One TUPLE per subject-contrast group: all source fields (< target) + the target.

        The source-field set is the sorted distinct non-target fields across the WHOLE index,
        so every emitted tuple has the SAME ``N`` (uniform stacking for the default collate).
        Groups missing any source field or the target are skipped (the travelling-volunteer
        cohort has all fields by design; incomplete groups cannot form a consensus tuple).
        """
        assert self._target_field is not None  # enforced in __init__
        tgt_f = float(self._target_field)
        # Prefer an explicit (cross-split-shared) source set; else derive from this index.
        if self._fixed_source_fields is not None:
            source_fields = [f for f in self._fixed_source_fields if f < tgt_f - 1e-6]
        else:
            source_fields = sorted(
                {
                    float(r["field_strength"])
                    for r in index
                    if float(r["field_strength"]) < tgt_f - 1e-6
                }
            )
        if len(source_fields) < 2:
            raise ValueError(
                "pairing_policy='multi_source' needs >=2 distinct source fields below "
                f"target_field={tgt_f}; found {source_fields}. A consensus needs >=2 sources."
            )
        groups: dict[str, list[dict[str, Any]]] = {}
        for rec in index:
            key = rec.get("pairing_group") or f"{rec['subject_id']}|{rec['contrast']}"
            groups.setdefault(key, []).append(rec)
        tuples: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        skipped = 0
        for key, recs in groups.items():
            fields = [float(r["field_strength"]) for r in recs]
            if len(fields) != len(set(fields)):
                # The travelling-volunteer cohort is one volume per (subject,contrast,field);
                # a duplicate field is a malformed manifest — fail fast (a silent dict-collapse
                # would discard data non-deterministically).
                raise ValueError(
                    f"multi_source group {key!r} has duplicate field strengths {sorted(fields)}; "
                    "expected one volume per field (travelling-volunteer invariant)."
                )
            by_field = {float(r["field_strength"]): r for r in recs}
            tgt = by_field.get(tgt_f)
            srcs = [by_field.get(f) for f in source_fields]
            if tgt is None or any(s is None for s in srcs):
                skipped += 1
                continue
            tuples.append(([s for s in srcs if s is not None], tgt))
        if not tuples:
            raise ValueError(
                "pairing_policy='multi_source' produced 0 tuples: no subject-contrast group "
                f"has all source fields {source_fields} AND target {tgt_f} ({skipped} skipped)."
            )
        self._source_fields = source_fields
        self._skipped_groups = skipped
        return tuples

    def _build_multi_field(
        self, index: list[dict[str, Any]]
    ) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
        """One TUPLE per subject-contrast group: the declared field stack + any held-out fields.

        The container is ``(all_records, judge_record)``, matching the shape the two
        generic consumers already destructure -- ``_build_index_map`` (``container[1]``
        supplies the slice indices, ``container[0]`` is depth-checked against it) and
        ``container_volume_paths`` (``container[0]`` plus ``container[1]`` is the
        working set). ``all_records`` is the stack in DECLARED order followed by the
        held-out fields in declared order, so the getter recovers the split by
        ``len(self._stack_fields)`` -- the order is this method's own guarantee.

        ``judge_record`` is the HIGHEST field of the stack. Some volume has to decide
        which depth slices count as foreground, there is no target to inherit that from,
        and the volumes are co-registered onto one grid so any member gives nearly the
        same mask; the highest field has the best SNR, so it gives the least noisy one.
        It is deliberately never a held-out field: on the extrapolation test the held-out
        acquisition is the 0.1 T one, whose mask would be the worst available.

        Groups missing any declared field are skipped and counted, as ``multi_source``
        does -- an incomplete group cannot form a stack of the declared arity.
        """
        wanted = self._stack_fields + self._heldout_fields
        present = sorted({float(r["field_strength"]) for r in index})
        missing = [f for f in wanted if not any(abs(f - q) <= 1e-6 for q in present)]
        if missing:
            # Without this the every-group-skips path below raises "0 tuples", which is
            # loud but names the wrong cause -- it reads as a broken manifest rather than
            # a field the corpus never had.
            raise ValueError(
                f"pairing_policy='multi_field' declares field(s) {missing} that no record "
                f"in this split carries; fields present={present}. Check "
                "data.mrixfields.fields / heldout_fields against the manifest."
            )
        groups: dict[str, list[dict[str, Any]]] = {}
        for rec in index:
            key = rec.get("pairing_group") or f"{rec['subject_id']}|{rec['contrast']}"
            groups.setdefault(key, []).append(rec)
        tuples: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        skipped = 0
        for key, recs in groups.items():
            fields = [float(r["field_strength"]) for r in recs]
            if len(fields) != len(set(fields)):
                # Same travelling-volunteer invariant multi_source enforces: one volume
                # per (subject, contrast, field). A duplicate would collapse in the dict
                # below and discard data non-deterministically.
                raise ValueError(
                    f"multi_field group {key!r} has duplicate field strengths "
                    f"{sorted(fields)}; expected one volume per field "
                    "(travelling-volunteer invariant)."
                )
            by_field = {float(r["field_strength"]): r for r in recs}
            picked = [by_field.get(f) for f in wanted]
            if any(r is None for r in picked):
                skipped += 1
                continue
            all_recs = [r for r in picked if r is not None]
            judge = all_recs[self._stack_fields.index(max(self._stack_fields))]
            tuples.append((all_recs, judge))
        if not tuples:
            raise ValueError(
                f"pairing_policy='multi_field' produced 0 tuples: no subject-contrast "
                f"group carries every declared field {wanted} ({skipped} skipped)."
            )
        self._skipped_groups = skipped
        return tuples

    def _canonical_contrasts(self, index: list[dict[str, Any]]) -> list[str]:
        """The pinned contrast set (SSOT across splits) if given, else derived from
        ``index`` in canonical ``_CONTRAST_INDEX`` order."""
        if self._pinned_contrasts is not None:
            return list(self._pinned_contrasts)
        return canonical_contrasts(index)

    def _build_multi_contrast(
        self, index: list[dict[str, Any]]
    ) -> list[tuple[list[dict[str, Any]], dict[str, Any], float]]:
        """One tuple per (subject, source field): ALL contrasts at that field stacked as
        the encoder input + the target-contrast image at ``target_field``.

        The relaxometry inversion (idea 2.1) needs J>=3 independent source contrasts to
        be over-determined (Proposition 3), so ``input`` is the full canonical contrast
        stack ``[C, H, W]``. Subject-fields missing any contrast (or the target) are
        skipped; a uniform ``C`` across tuples keeps the default collate valid.
        """
        assert self._target_field is not None  # enforced in __init__
        tgt_f = float(self._target_field)
        canon = self._canonical_contrasts(index)
        if not canon:
            raise ValueError("multi_contrast: no contrasts found in the index.")
        out_c = self._output_contrast or canon[0]
        if out_c not in _CONTRAST_INDEX:
            # Silently defaulting an unknown output contrast to T1w (index 0) would
            # train/validate against the wrong target with no signal (#9). Fail loud.
            raise ValueError(
                f"multi_contrast output_contrast={out_c!r} is not a known contrast "
                f"{sorted(set(_CONTRAST_INDEX))}; set data.mrixfields_output_contrast "
                "to one of them."
            )
        out_idx = _CONTRAST_INDEX[out_c]
        groups: dict[str, list[dict[str, Any]]] = {}
        for rec in index:
            groups.setdefault(rec["subject_id"], []).append(rec)
        tuples: list[tuple[list[dict[str, Any]], dict[str, Any], float]] = []
        skipped = 0
        for recs in groups.values():
            by_fc: dict[tuple[float, int], dict[str, Any]] = {}
            for r in recs:
                key = (
                    float(r["field_strength"]),
                    _CONTRAST_INDEX.get(str(r.get("contrast", "T1w")), 99),
                )
                by_fc[key] = r
            tgt = by_fc.get((tgt_f, out_idx))
            if tgt is None:
                skipped += 1
                continue
            src_fields = sorted({f for (f, _ci) in by_fc if f < tgt_f - 1e-6})
            for sf in src_fields:
                input_recs = [by_fc.get((sf, _CONTRAST_INDEX.get(c, 99))) for c in canon]
                if any(ir is None for ir in input_recs):
                    continue
                tuples.append(([ir for ir in input_recs if ir is not None], tgt, sf))
        if not tuples:
            raise ValueError(
                "pairing_policy='multi_contrast' produced 0 tuples "
                f"(skipped {skipped}): need a subject with ALL contrasts {canon} at a "
                f"source field AND {out_c}@{tgt_f}T. Point at the full "
                "travelling-volunteer corpus (or check the split kept subjects whole)."
            )
        self._n_contrasts = len(canon)
        return tuples

    def __len__(self) -> int:
        if self._slice_mode == "all_slices":
            return len(self._index_map)  # one sample per (pair/tuple, foreground slice)
        if self._policy in _TUPLE_POLICIES:
            return len(self._tuples)
        return len(self._pairs)

    @staticmethod
    def _to_chw_unit(img: torch.Tensor, *, rescale: bool = True) -> torch.Tensor:
        """Coerce to ``[1, H, W]`` float in ``[0, 1]`` (magnitude-only, 2-D slice).

        ``rescale=False`` keeps the corpus's own global ``[0, 1]`` scale — see
        :meth:`_extract_chw_unit` for why that matters to field translation.

        ``NiftiStrategy.load`` returns a 4-D channel-first volume ``[C, H, W, D]``
        (the medical/TorchIO convention). These arms are 2-D (``spatial_dims: 2``),
        so collapse a volume to a single magnitude slice — first channel, central
        depth. Without this a 4-D load fell through every branch unchanged and
        ``_apply_transform``'s ``unsqueeze(-1)`` produced a 5-D tensor that
        ``tio.ScalarImage`` rejects (``Input tensor must be 4D, but it is 5D``) —
        the 2026-06-20 crash that took out every NIfTI-backed mrixfields2026 arm at
        the first batch. NOTE: only the central slice is ingested per volume; if the
        source volumes are multi-slice, per-slice expansion is a viability follow-up.
        """
        x = img.float()
        if x.ndim == 4:  # [C, H, W, D] volume -> first channel, central depth slice
            x = x[0, :, :, x.shape[-1] // 2].unsqueeze(0)  # -> [1, H, W]
        if x.ndim == 3 and x.shape[-1] == 1:  # H, W, 1 -> 1, H, W
            x = x.squeeze(-1).unsqueeze(0)
        elif x.ndim == 2:  # H, W -> 1, H, W
            x = x.unsqueeze(0)
        elif x.ndim == 3 and x.shape[0] != 1:  # already C,H,W with C>1: keep first
            x = x[:1]
        if rescale:
            lo, hi = x.amin(), x.amax()
            if hi > lo:
                x = (x - lo) / (hi - lo)
        return x.clamp(0.0, 1.0)

    @staticmethod
    def _extract_chw_unit(
        img: torch.Tensor, slice_idx: int, *, rescale: bool = True
    ) -> torch.Tensor:
        """Coerce depth slice ``slice_idx`` of a volume to ``[1, H, W]`` in ``[0, 1]``.

        The ``all_slices`` counterpart of ``_to_chw_unit``: instead of the CENTRAL
        depth it takes ``slice_idx`` (first channel), then applies the SAME per-slice
        min-max renorm. That renorm alone would amplify a faint-noise air slice to full
        scale — the foreground filter (``_foreground_slices``), not this renorm, is what
        keeps air slices out of the loader, so only slices that already passed the
        whole-volume-scale foreground test ever reach here.

        ``rescale=False`` (the ``mrixfields_rescale_per_image`` default) SKIPS that
        renorm and keeps the corpus's own global ``[0, 1]`` scale. Field translation
        learns how intensity maps from one B0 to another, and renormalising the source
        AND the target each to exactly ``[0, 1]`` erases precisely that relationship:
        the model is handed a source pre-scaled to the target's range, so the objective
        degenerates toward structure matching. The module docstring already records that
        this corpus is ``[0, 1]`` on disk, so the renorm converts a MEANINGFUL global
        scale into an arbitrary per-image one rather than establishing a scale.

        The cost is larger under ``all_slices`` than it was under ``central``: with one
        slice per volume the gain was at least comparable across samples, but stretching
        each of ~364 depth slices independently amplifies a low-dynamic-range slice near
        the top of the head to the same ``[0, 1]`` as mid-brain — a per-slice gain the
        network cannot invert because nothing in the batch records it. The ``clamp``
        stays either way: on an already-``[0, 1]`` corpus it is a no-op safety net.
        """
        x = img.float()
        if x.ndim == 4:  # [C, H, W, D] -> first channel, slice_idx
            x = x[0, :, :, slice_idx].unsqueeze(0)  # -> [1, H, W]
        elif x.ndim == 3:  # [H, W, D] -> slice_idx
            x = x[:, :, slice_idx].unsqueeze(0)  # -> [1, H, W]
        else:
            raise ValueError(
                f"all_slices expects a 3-/4-D volume to slice; got ndim={x.ndim} "
                f"shape={tuple(x.shape)}."
            )
        if rescale:
            lo, hi = x.amin(), x.amax()
            if hi > lo:
                x = (x - lo) / (hi - lo)
        return x.clamp(0.0, 1.0)

    @staticmethod
    def _volume_to_cdhw_unit(img: torch.Tensor, *, rescale: bool = True) -> torch.Tensor:
        """Coerce a whole volume to ``[C, H, W, D]`` float, normalised to ``[0, 1]``.

        The ``volume`` slice-mode payload (``spatial_dims: 3`` arms). Unlike the 2-D
        coercions this keeps ALL depth and normalises over the WHOLE volume (one min-max
        over every voxel), not per-slice. ``rescale=False`` keeps the corpus's global
        scale; the per-volume min-max still rescales SOURCE and TARGET independently,
        so it erases the cross-field intensity relationship the same way the 2-D path
        does — just at volume granularity.
        """
        x = img.float()
        if x.ndim == 3:  # [H, W, D] -> [1, H, W, D]
            x = x.unsqueeze(0)
        elif x.ndim != 4:
            raise ValueError(
                f"volume slice_mode expects a 3-/4-D volume; got ndim={x.ndim} "
                f"shape={tuple(x.shape)}."
            )
        if rescale:
            lo, hi = x.amin(), x.amax()
            if hi > lo:
                x = (x - lo) / (hi - lo)
        return x.clamp(0.0, 1.0)  # [C, H, W, D]

    def _img(self, path: str, slice_idx: int | None) -> torch.Tensor:
        """Load ``path`` and coerce to the served payload for the active slice_mode.

        ``slice_idx is None`` -> whole-slice/whole-volume path (``central`` central
        slice via ``_coerce=_to_chw_unit``, or ``volume`` via ``_coerce=
        _volume_to_cdhw_unit``). An int -> ``all_slices`` serves that depth slice.
        """
        vol = self._load(path)
        if slice_idx is None:
            return self._coerce(vol)
        return self._extract_chw_unit(vol, slice_idx, rescale=self._rescale_per_image)

    def _foreground_slices(self, path: str) -> list[int]:
        """Depth indices of ``path`` whose slices pass the foreground filter.

        Scanned ONCE per unique volume at construction (cached in ``self._fg_cache``),
        never per ``__getitem__``. Foreground is judged on the GLOBALLY [0,1]-normalised
        volume so an air slice — near-0 on that scale — is dropped even though per-slice
        serving would later renorm it to full scale (see ``_FOREGROUND_EPS`` /
        ``_FOREGROUND_MIN_FRAC``).
        """
        cached = self._fg_cache.get(path)
        if cached is not None:
            return cached
        vol = self._load(path).float()
        if vol.ndim == 4:  # [C, H, W, D] -> channel 0 (magnitude), like _to_chw_unit
            chan = vol[0]
        elif vol.ndim == 3:  # [H, W, D]
            chan = vol
        else:
            raise ValueError(
                f"all_slices foreground scan expects a 3-/4-D volume for {path!r}; "
                f"got ndim={vol.ndim} shape={tuple(vol.shape)}."
            )
        lo, hi = chan.amin(), chan.amax()
        norm = (chan - lo) / (hi - lo) if hi > lo else torch.zeros_like(chan)
        # Fraction of voxels lit per depth slice -> [D]; vectorised (no per-slice loop).
        frac = (norm > _FOREGROUND_EPS).float().mean(dim=(0, 1))  # [D]
        fg = torch.nonzero(frac >= _FOREGROUND_MIN_FRAC).flatten().tolist()
        self._fg_cache[path] = fg
        return fg

    @staticmethod
    def _record_depth(rec: dict[str, Any]) -> int | None:
        """Depth of ``rec``'s volume from the manifest ``shape``, or ``None``.

        Header-free: ``build_mrixfields2026_manifest.py`` already writes ``shape`` per
        volume, so the co-registration guard costs no IO. ``None`` (absent or
        malformed ``shape``) means "cannot tell" and the guard skips that record
        rather than inventing a depth — an absent annotation is a SKIP, never a
        rejection, the same polarity the workflow axis guards use.
        """
        shape = rec.get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) < 3:
            return None
        try:
            return int(shape[-1])
        except (TypeError, ValueError):
            return None

    def _assert_source_depths_match(self, container: Any) -> None:
        """Fail at CONSTRUCTION if a source volume is shallower than its target.

        ``_build_index_map`` derives foreground indices from the TARGET volume only and
        applies the same ``slice_idx`` to every source, which is sound exactly as long as
        source and target share the co-registered MNI grid. Nothing asserted it. A
        shallower source raised ``IndexError`` at an arbitrary point mid-epoch — hours
        into a cluster run, with no message naming the offending pair.

        Compared depth-to-depth (source vs target) rather than depth-to-slice-index,
        because the question is whether the two volumes agree, not whether the manifest
        agrees with the voxels: an injected test loader legitimately returns a volume
        whose depth differs from the manifest ``shape``, and grading against the slice
        index would reject that self-consistent case.

        This does not (and cannot, without voxel IO) verify that equal depths are the
        SAME grid; it closes the crash, and the mismatched-anatomy case stays a manifest
        contract. ``scripts/data/show_mrixfields_dimensions.py`` exists to report
        heterogeneous shapes, so the corpus was already known not to be guaranteed
        uniform.
        """
        target = container[1]
        tgt_depth = self._record_depth(target)
        if tgt_depth is None:
            return
        sources = container[0]
        for rec in sources if isinstance(sources, list) else [sources]:
            depth = self._record_depth(rec)
            if depth is not None and depth < tgt_depth:
                raise ValueError(
                    f"mrixfields slice_mode='all_slices': source volume "
                    f"{rec.get('primary_path')!r} has depth {depth}, shallower than its "
                    f"target {target.get('primary_path')!r} (depth {tgt_depth}). Slice "
                    "indices are taken from the target and applied to the source, so "
                    "this would raise IndexError mid-epoch. Source and target must "
                    "share the co-registered MNI grid; re-check the manifest pairing."
                )

    def _containers(self) -> list[Any]:
        """The policy's containers: ``_tuples`` for the tuple policies, else ``_pairs``."""
        return self._tuples if self._policy in _TUPLE_POLICIES else self._pairs

    def container_volume_paths(self) -> list[frozenset[str]]:
        """Volume paths each container reads, indexed by container.

        The input :class:`~spectramr.data.samplers.VolumeBlockedSliceSampler` needs to order
        an epoch so that containers sharing volumes are consumed while those volumes are
        resident. Exposed as a method rather than inlined in the sampler because only the
        dataset knows the per-policy container shape — a pair reads ``{src, tgt}``, a
        ``multi_source`` tuple reads every source plus the target.

        Empty list for ``central`` / ``volume`` mode, where there is no slice expansion to
        order and the sampler does not apply.
        """
        if self._slice_mode != "all_slices":
            return []
        out: list[frozenset[str]] = []
        for container in self._containers():
            sources = container[0]
            recs = sources if isinstance(sources, list) else [sources]
            paths = {str(r["primary_path"]) for r in recs}
            paths.add(str(container[1]["primary_path"]))
            out.append(frozenset(paths))
        return out

    def __getstate__(self) -> dict[str, Any]:
        """Drop the volume cache when pickled to a DataLoader worker.

        Under ``spawn`` the dataset is pickled per worker; shipping a cache holding up to
        ``max_resident_volumes`` x ~226 MB would serialise gigabytes and hand every worker
        an identical, immediately-stale cache. Workers rebuild it lazily, which is also
        what makes the budget per-worker rather than shared.
        """
        state = dict(self.__dict__)
        if state.get("_volume_cache") is not None:
            state["_volume_cache"] = None
            state["_load"] = None  # bound to the dropped cache; rebuilt in __setstate__
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if self._slice_mode != "central" and self.__dict__.get("_load") is None:
            self._volume_cache = _VolumeLRU(
                self._max_resident_volumes, _default_nifti_volume_loader
            )
            self._load = self._volume_cache

    def _build_index_map(self) -> list[tuple[int, int]]:
        """``all_slices`` map: ``(container_idx, slice_idx)`` per foreground slice.

        Wraps whatever the policy built — ``_pairs`` for the pair policies or ``_tuples``
        for the stacking policies (``_TUPLE_POLICIES``) — so all policies gain per-slice
        expansion. Foreground is taken from each container's TARGET volume (element ``[1]``
        of both the ``(src, tgt)`` pair and the ``(recs, tgt[, sf])`` tuple); source and
        target are co-registered MNI 0.5 mm, so the same ``slice_idx`` is anatomy-aligned.
        """
        containers = self._containers()
        index_map: list[tuple[int, int]] = []
        for c_idx, container in enumerate(containers):
            tgt = container[1]  # (src,tgt) | (recs,tgt) | (recs,tgt,sf) -> tgt
            self._assert_source_depths_match(container)
            fg = self._foreground_slices(tgt["primary_path"])
            for s in fg:
                index_map.append((c_idx, s))
        if not index_map:
            raise ValueError(
                "slice_mode='all_slices' produced 0 foreground slices across all "
                f"{len(containers)} pairs/tuples (every target volume is pure air at "
                f"eps={_FOREGROUND_EPS}, min_frac={_FOREGROUND_MIN_FRAC}); check the "
                "target manifest / foreground threshold."
            )
        return index_map

    #: The loader for each pairing policy that is not the plain source-target pair.
    _GETITEM_BY_POLICY: ClassVar[dict[str, str]] = {
        "multi_source": "_getitem_multi_source",
        "multi_contrast": "_getitem_multi_contrast",
        "multi_field": "_getitem_multi_field",
    }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._slice_mode == "all_slices":
            container_idx, slice_idx = self._index_map[idx]
        else:
            container_idx, slice_idx = idx, None
        getter = self._GETITEM_BY_POLICY.get(self._policy)
        if getter is not None:
            return getattr(self, getter)(container_idx, slice_idx)
        src, tgt = self._pairs[container_idx]
        sample: dict[str, Any] = {
            "input": self._img(src["primary_path"], slice_idx),
            "target": self._img(tgt["primary_path"], slice_idx),
            "field_strength": torch.tensor(float(src["field_strength"])),
            "contrast_id": torch.tensor(_CONTRAST_INDEX.get(str(src.get("contrast", "T1w")), 0)),
            "subject_id": src["subject_id"],
        }
        if self._expose_target_field:
            sample["field_strength_target"] = torch.tensor(float(tgt["field_strength"]))
        return self._apply_transform(sample)

    def _getitem_multi_source(self, idx: int, slice_idx: int | None = None) -> dict[str, Any]:
        src_recs, tgt = self._tuples[idx]
        sources = torch.stack(
            [self._img(r["primary_path"], slice_idx) for r in src_recs]
        )  # [N, 1, H, W]
        field_strengths = torch.tensor([float(r["field_strength"]) for r in src_recs])
        sample: dict[str, Any] = {
            # input/field_strength = the FIRST source so unpack + validation stay compatible.
            "input": sources[0],
            "field_strength": field_strengths[0],
            "target": self._img(tgt["primary_path"], slice_idx),
            # multi-source TUPLE for the consensus loss (B-1.1). The renderer is
            # field-invariant in encode() and renders at the target field, so per-source
            # field values are intentionally NOT emitted (they would be inert).
            "sources": sources,
            "contrast_id": torch.tensor(
                _CONTRAST_INDEX.get(str(src_recs[0].get("contrast", "T1w")), 0)
            ),
            "subject_id": src_recs[0]["subject_id"],
        }
        if self._expose_target_field:  # honour the knob, like the pair policies
            sample["field_strength_target"] = torch.tensor(float(tgt["field_strength"]))
        return self._apply_transform(sample)

    def _getitem_multi_contrast(self, idx: int, slice_idx: int | None = None) -> dict[str, Any]:
        input_recs, tgt, sf = self._tuples[idx]
        # Stack the canonical contrast set at the source field as [C, H, W]. NOTE
        # (honest scope): each contrast is normalised INDEPENDENTLY to [0,1] by
        # `_to_chw_unit`, matching the dataset-wide magnitude convention. This attenuates
        # the absolute cross-contrast intensity ratios the Proposition-3 relaxometry
        # inversion would ideally exploit, so the encoder's maps are best read as a
        # physics-structured latent rather than calibrated qMRI (the same caveat the
        # RelaxometryEncoder docstring states). Joint cross-contrast scaling is a
        # deliberate follow-up, not wired here (it can zero-out a low-signal contrast on
        # magnitude data).
        stack = torch.cat(
            [self._img(r["primary_path"], slice_idx) for r in input_recs],
            dim=0,
        )
        sample: dict[str, Any] = {
            "input": stack,  # [C, H, W] multi-contrast source (encoder in_channels = C)
            "field_strength": torch.tensor(float(sf)),
            "target": self._img(tgt["primary_path"], slice_idx),
            "contrast_id": torch.tensor(_CONTRAST_INDEX.get(str(tgt.get("contrast", "T1w")), 0)),
            "subject_id": input_recs[0]["subject_id"],
        }
        if self._expose_target_field:
            sample["field_strength_target"] = torch.tensor(float(tgt["field_strength"]))
        return self._apply_transform(sample)

    def _getitem_multi_field(self, idx: int, slice_idx: int | None = None) -> dict[str, Any]:
        """The M-field stack as ``input`` ``[M, H, W]``, plus any held-out field separately.

        Emits NO ``target`` key, deliberately. ``DispersionBlochAEStrategy`` reads its
        reconstruction target as ``batch.get("target", x)``
        (``dispersion_bloch_ae_strategy.py:70``), so an absent key makes the autoencoder
        reconstruct its own input -- which is the objective. A ``target`` here would
        silently become that objective instead: a single-field image against an
        M-channel prediction (a shape error), or the held-out image (no error at all,
        and the extrapolation test would then be fitting the field it claims to predict).

        NOTE (honest scope, the same caveat ``_getitem_multi_contrast`` carries for
        contrasts, and sharper here): each field is a separate volume through
        ``_to_chw_unit``, so every channel of the stack is normalised INDEPENDENTLY to
        [0,1]. The cross-field intensity ratio is exactly what an SPGR render predicts,
        so this attenuates the signal the fit would ideally use. Magnitude MR carries no
        absolute scale anyway, so the per-field gain is a nuisance parameter the model
        has to absorb -- which is why the data-consistency term must be scale-invariant
        per channel rather than a raw L2 (Phase 1 of the unified-operator plan).
        """
        all_recs, _judge = self._tuples[idx]
        n_stack = len(self._stack_fields)
        stack_recs, heldout_recs = all_recs[:n_stack], all_recs[n_stack:]
        stack = torch.cat(
            [self._img(r["primary_path"], slice_idx) for r in stack_recs],
            dim=0,
        )
        sample: dict[str, Any] = {
            "input": stack,  # [M, H, W], channel m == self._stack_fields[m]
            "field_strengths": torch.tensor([float(r["field_strength"]) for r in stack_recs]),
            # Scalar too: ~15 strategies and the generic unpack/validation path index
            # batch["field_strength"] directly, the same compatibility reason
            # _getitem_multi_source:1091 gives for emitting the first source's field.
            "field_strength": torch.tensor(float(stack_recs[0]["field_strength"])),
            "contrast_id": torch.tensor(
                _CONTRAST_INDEX.get(str(stack_recs[0].get("contrast", "T1w")), 0)
            ),
            "subject_id": stack_recs[0]["subject_id"],
        }
        if heldout_recs:
            # Wrapped in the SAME tio.Subject as the stack ("heldout" is in _IMAGE_KEYS),
            # so a random spatial transform samples one warp for both. Scored against a
            # render at heldout_field_strengths; never fitted.
            sample["heldout"] = torch.cat(
                [self._img(r["primary_path"], slice_idx) for r in heldout_recs],
                dim=0,
            )
            sample["heldout_field_strengths"] = torch.tensor(
                [float(r["field_strength"]) for r in heldout_recs]
            )
        return self._apply_transform(sample)

    def _apply_transform(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply the injected ``tio.Compose`` via canonical Subject wrapping.

        The pipeline injects a ``tio.Compose`` (built for a ``tio.Subject``), but
        this dataset emits a plain dict. Handing a raw dict to a TorchIO transform
        raises ``RuntimeError: ... a value for "include" must be specified`` — the
        2026-06-20 crash that took out every transform-bearing mrixfields2026 arm.
        Mirror ``contrast_aware`` / ``oracle_bssfp``: wrap the image-bearing keys
        as ``tio.ScalarImage`` (4-D ``[C, H, W, 1]``; a paired Subject so any
        spatial transform stays consistent across input/target/sources), run the
        transform, then unwrap back to the plain dict the strategy consumes.
        """
        if self._transform is None:
            return sample
        import torchio as tio

        images: dict[str, Any] = {}
        kinds: dict[str, str] = {}
        for key in _IMAGE_KEYS:
            t = sample.get(key)
            if not isinstance(t, torch.Tensor):
                continue
            if key == "sources":  # [N, 1, H, W] -> [N, H, W, 1] (N as channels)
                images[key] = tio.ScalarImage(tensor=t.squeeze(1).unsqueeze(-1))
                kinds[key] = "sources"
            elif t.ndim == 4:  # volume mode: [C, H, W, D] is already tio-shaped (4-D)
                images[key] = tio.ScalarImage(tensor=t)
                kinds[key] = "volume"
            else:  # [C, H, W] -> [C, H, W, 1] (C == 1 for a pair, M for a multi_field stack)
                images[key] = tio.ScalarImage(tensor=t.unsqueeze(-1))
                kinds[key] = "chw"
        transformed = self._transform(tio.Subject(**images))
        out = dict(sample)
        for key, kind in kinds.items():
            data = transformed[key].data
            if kind == "sources":  # [N, H, W, 1] -> [N, 1, H, W]
                out[key] = data.squeeze(-1).unsqueeze(1)
            elif kind == "volume":  # [C, H, W, D] as-is
                out[key] = data
            else:  # [1, H, W, 1] -> [1, H, W]
                out[key] = data.squeeze(-1)
        return out
