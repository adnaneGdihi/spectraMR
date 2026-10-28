"""Data Pipeline Director

Orchestrates data loading components using fluent builders.
Replaces the deprecated ConsolidatedDatasetFactory.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from torch.utils.data import DataLoader

from spectramr.core.topology import resolve_run_topology
from spectramr.core.worker_policy import clamp_worker_count
from spectramr.infrastructure.builders.context import (
    BuilderContext,
    accepts_builder_context,
)

if TYPE_CHECKING:
    import torch

    from spectramr.config.schemas.physics import CoilProcessingConfig
    from spectramr.config.settings import TrainingSettings  # noqa: F401  (docstring ref)

    # ``Balancer`` is imported lazily inside ``build_multi_domain_dataloaders``
    # to avoid an eager import; declare it here so its return annotation
    # resolves at module scope (F821) — safe under ``from __future__ import
    # annotations`` (the annotation is never evaluated at runtime).
    from spectramr.data.builders.site_balancer import Balancer

logger = logging.getLogger(__name__)


def _self_indexed_dataset_types() -> frozenset[str]:
    """Dataset types that build their OWN index and must skip the pre-split.

    Derived from the dataset registry (``DatasetCreator.indexed``) rather than
    restated. The hand-written frozenset this replaced named 5 of the 12
    self-indexed types, and its own comment asked the next editor to "keep this
    in sync with ``DatasetInstantiator``" -- a manual sync that never happened.

    The 7 missing types were not merely running redundant work. A type absent
    here falls through to ``ManifestLoader.load_fastmri_splits``, which globs for
    ``*.h5``; on an arm carrying no ``index_path`` that raises before the arm's
    own creator is ever called, so the type is unusable. That is what the
    ``oracle_bssfp`` note recorded -- it raised at ``_build_on_the_fly_index``
    (F4 smoke 2026-06-16), was added to the list, and the other 7 were left.

    A function, not a module constant: ``dataset_instantiator`` populates the
    registry at import and this module imports it lazily inside
    ``build_dataloaders``, so a module-level comprehension would evaluate
    against an empty dict and silently restore the drift it exists to remove.
    """
    from spectramr.data.builders.dataset_instantiator import self_indexed_dataset_types

    return self_indexed_dataset_types()


#: fMRI/SFC ``data.expose_*`` keys that activate SFCConformalFMRIKeysWrapper.
_SFC_EXPOSE_KEYS = (
    "expose_conformal_jacobian",
    "expose_cortex_flatten_grid",
    "expose_glm_design_matrix",
    "expose_scanner_id",
    "expose_site_id",
    "expose_field_strength",
)


def _container_volumes_or_none(dataset) -> list | None:
    """The per-container volume sets, or ``None`` when the concept does not apply.

    ``None`` for the ``central`` / ``volume`` modes, for ``npy_slice`` (samples are
    already independent 2-D files), and for any dataset not exposing the
    container/volume mapping.

    Returns the LIST rather than a bool because ``container_volume_paths`` is not
    cached — the mrixfields implementation rebuilds it by walking every container
    on each call. Splitting "does it apply?" from "give me the value" would make
    the sampler path pay for that walk twice.
    """
    volumes_fn = getattr(dataset, "container_volume_paths", None)
    index_map = getattr(dataset, "_index_map", None)
    if not callable(volumes_fn) or not index_map:
        return None
    volumes: list = list(volumes_fn())
    return volumes or None


def _shares_expensive_volume_decode(dataset) -> bool:
    """Whether many of ``dataset``'s samples come from ONE expensive volume decode.

    True exactly for the slice-expanded volume-container case: the dataset maps
    sample indices onto containers (``_index_map``) and can name the volume behind
    each (``container_volume_paths``).

    Two readers depend on this one predicate, and they want opposite things from
    the same fact, which is why it is named rather than inlined twice:
    :func:`_build_slice_sampler` uses it to decide the train order is worth
    blocking by volume, and the validation loader uses it to decide that worker
    fan-out would multiply a whole-volume decode across processes. The validation
    reader sits behind a short-circuiting ``and`` on a nonzero worker count, so at
    the default 0 it costs nothing on any arm in the corpus.
    """
    return _container_volumes_or_none(dataset) is not None


def _build_slice_sampler(dataset) -> object | None:
    """A :class:`~spectramr.data.samplers.VolumeBlockedSliceSampler` for a slice-expanded
    dataset, else ``None``.

    ``None`` means "no special order needed" and the caller falls back to ordinary
    shuffling — the ``central`` / ``volume`` modes, ``npy_slice`` (whose samples are
    already independent 2-D files with no volume to amortise), and any dataset that does
    not expose the container/volume mapping. That is a genuine not-applicable, not a
    silent degradation: the expensive order only exists for datasets whose samples share
    an expensive decode, and the sampler raises rather than skips when its own
    preconditions are violated.
    """
    container_volumes = _container_volumes_or_none(dataset)
    if container_volumes is None:
        return None
    index_map = dataset._index_map

    from spectramr.data.samplers import VolumeBlockedSliceSampler

    # Read the budget off the dataset, which already resolved it from config. Re-reading
    # a module default here would let the sampler block for a budget the dataset's cache
    # does not actually have — the two must agree or the blocking buys nothing.
    return VolumeBlockedSliceSampler(
        index_map,
        container_volumes,
        max_resident_volumes=dataset._max_resident_volumes,
        shuffle=True,
    )


def resolve_sfc_expose_flags(data_config) -> dict[str, bool]:
    """Resolve the ``expose_*`` keys that gate ``SFCConformalFMRIKeysWrapper``.

    The mrixfields cross-field dataset emits per-sample ``field_strength`` itself
    (unconditionally, in ``__getitem__``), so its ``expose_field_strength`` must NOT
    activate the fMRI/SFC keys wrapper — otherwise every cross-field/flow/bridge
    dataset gets a spurious (no-op) fMRI wrapper stacked around it (#9). Extracted
    as a pure helper so this exclusion is unit-testable.

    Reads ``data.expose.<leaf>``, not the flat ``data.expose_<leaf>`` these keys
    are named for. Phase 9a folded them into an ``expose:`` sub-block; because
    the read is by STRING, the stale spelling returned ``False`` instead of
    raising, so every flag was permanently off and the wrapper was never
    constructed — an inert mechanism, not a crash (pitfall #16). The flat
    spelling is still accepted so an unmigrated caller keeps working, and the
    returned dict keeps the legacy key names because they are the wrapper's
    kwarg contract.
    """
    expose = getattr(data_config, "expose", None)

    def _declared(legacy_key: str) -> bool:
        canonical = getattr(expose, legacy_key.removeprefix("expose_"), None)
        if canonical is not None:
            return bool(canonical)
        return bool(getattr(data_config, legacy_key, False))

    flags = {k: _declared(k) for k in _SFC_EXPOSE_KEYS}
    if getattr(data_config, "dataset_type", None) == "mrixfields":
        flags["expose_field_strength"] = False
    return flags


def strided_validation_subset(val_ds, validation_cfg, val_batch_size):
    """Spread a CAPPED validation subsample across the dataset (#171).

    ``validation.num_validation_batches`` / ``num_samples`` cap the val loop by
    stopping after the first N *contiguous* batches (``pipelines/train.py``). On
    an unshuffled loader that grades the first N adjacent slices of one volume —
    usually background at the top of an axial stack — so the tissue-segmentation
    metrics warn that every target brain mask is (almost) empty. Wrapping
    ``val_ds`` in a deterministic strided ``Subset`` keeps the SAME compute budget
    (only the kept samples load) while the kept slices span the whole set
    (multiple volumes / slice positions), so background-only batches stop
    dominating the average.

    Returns ``val_ds`` unchanged — byte-identical full-val behavior — when
    validation is uncapped, the dataset has no ``__len__`` (iterable), or the cap
    would keep everything. Extracted as a pure helper so it is unit-testable
    without building a real pipeline.
    """
    if validation_cfg is None:
        return val_ds

    batch = max(1, int(val_batch_size or 1))
    n_batches = validation_cfg.loader.num_batches if validation_cfg else None
    if n_batches is not None:
        target = max(1, int(n_batches)) * batch
    else:
        n_samples = validation_cfg.loader.num_samples if validation_cfg else None
        target = int(n_samples) if n_samples is not None else None
    if target is None or target < 1:
        return val_ds

    try:
        total = len(val_ds)
    except TypeError:
        # Iterable-style dataset with no length → cannot stride deterministically.
        return val_ds
    if total <= target:
        return val_ds  # the cap keeps everything; nothing to spread.

    from torch.utils.data import Subset

    # Endpoint-inclusive spread, NOT a fixed stride truncated to length (#171
    # residual). `range(0, total, total // target)[:target]` starts correctly but
    # the integer stride rounds DOWN, so it exhausts `target` samples before
    # reaching the end and the tail is unreachable -- deterministically, every
    # epoch, on every arm. At total=1000, target=300 the stride is 3, so the last
    # kept index is 897 and the final 10.2% of the validation set is never graded.
    # Smaller ratios are worse (total=10, target=3 loses 30%).
    #
    # The endpoint form pins the first and last sample and distributes the rest
    # evenly between them. `set` before `sorted` because rounding can collide when
    # target approaches total; that makes the realised count a MAXIMUM rather than
    # an exact figure, which is why it is logged rather than assumed.
    if target <= 1:
        indices = [0]
    else:
        indices = sorted({round(i * (total - 1) / (target - 1)) for i in range(target)})

    logger.info(
        "[DATASET] Strided validation subset: %d of %d samples (first=%d, last=%d, requested=%d)",
        len(indices),
        total,
        indices[0],
        indices[-1],
        target,
    )
    return Subset(val_ds, indices)


def apply_coil_processing(
    kspace: torch.Tensor, cfg: CoilProcessingConfig
) -> dict[str, torch.Tensor | None]:
    """Run the config-driven coil pipeline: compress → estimate → combine.

    Single source of truth for the ``physics.coil_processing`` block. Delegates
    each stage to the physics primitives (``fit_svd_basis`` /
    ``apply_svd_compression``, ``estimate_smaps``, ``coil_combine``) so the math
    lives in one place (``infrastructure/physics/``).

    Args:
        kspace: Complex multi-coil k-space ``(B, C, H, W)``.
        cfg: A ``CoilProcessingConfig`` (``physics.coil_processing``).

    Returns:
        ``{"kspace", "smaps", "target"}`` where ``kspace`` is the (possibly
        compressed) k-space, ``smaps`` is ``(B, C, H, W)`` complex or ``None``
        when estimation is off, and ``target`` is the coil-combined magnitude
        image ``(B, 1, H, W)``.

    Raises:
        NotImplementedError: ``compression.method == "gcc"`` (reserved, not yet
            implemented — no silent fallback, pitfall #9).
        ValueError: ``combine.method == "sense"`` without estimation enabled
            (SENSE needs sensitivity maps).
    """
    from spectramr.infrastructure.physics.coil_compression import (
        apply_svd_compression,
        fit_svd_basis,
    )
    from spectramr.infrastructure.physics.coil_sensitivity import estimate_smaps
    from spectramr.infrastructure.physics.fft_ops import coil_combine, ifft2c

    ks = kspace
    cm = cfg.compression
    if cm.method == "svd":
        basis = fit_svd_basis(ks, cm.num_virtual_coils, cm.calibration_lines)
        ks = apply_svd_compression(ks, basis)
    elif cm.method == "gcc":
        raise NotImplementedError(
            "physics.coil_processing.compression.method='gcc' is reserved but "
            "not implemented; use 'svd' or 'none'."
        )

    es = cfg.estimation
    smaps: torch.Tensor | None = None
    if es.enabled and es.method != "none":
        smaps = estimate_smaps(
            ks,
            method=es.method,
            kernel_size=es.kernel_size,
            acs_size=es.acs_size,
            eigen_threshold=es.eigen_threshold,
            maps_path=es.maps_path,
        )

    if cfg.combine.method == "sense" and smaps is None:
        raise ValueError(
            "physics.coil_processing.combine.method='sense' requires estimation "
            "enabled (smaps); set estimation.enabled=true with a non-'none' "
            "method."
        )

    coil_imgs = ifft2c(ks)
    if cfg.combine.method == "none":
        # No combine — keep the (compressed) coil images; there is no single-image
        # reconstruction target.
        target = coil_imgs
    else:
        target = coil_combine(coil_imgs, method=cfg.combine.method, smaps=smaps)
    return {"kspace": ks, "smaps": smaps, "target": target}


# Where each ``data.multi_domain.domains[].<field>`` override lands in the data
# config. The entry keeps the short spellings (it names a DOMAIN's location, not
# a `data:` path); the destinations are the canonical post-decomposition paths
# from RENAMES -- `data.data_root` -> `data.source.root`, `data.index_path` ->
# `data.source.index_path`. `dataset_type` stays at the top level: RENAMES marks
# it deliberately un-folded because it is a dispatch key, not a location.
_DOMAIN_OVERRIDE_DESTINATIONS: dict[str, tuple[str, ...]] = {
    "data_root": ("source", "root"),
    "index_path": ("source", "index_path"),
    "dataset_type": ("dataset_type",),
}


def _apply_domain_overrides(config, domain):
    """Return a copy of ``config`` with one domain's location overrides applied.

    Every hop is gated on ``model_fields`` so a destination that MOVES raises
    here instead of writing an attribute nothing reads. That is not defensive
    padding: the previous implementation used ``object.__setattr__`` to set
    ``data_root``/``index_path`` directly on the data config, and those names
    moved into ``data.source`` on 2026-07-31. The write still "succeeded" --
    it created a shadow attribute that read back correctly while the dataset
    builder went on reading ``data.source.root`` -- so every domain silently
    loaded the SAME corpus under a different tag (pitfall #16).

    Args:
        config: The frozen parent settings (or a bare data config).
        domain: One ``DomainConfigSchema`` entry.

    Returns:
        A new settings object; the parent is never mutated.

    Raises:
        AttributeError: If an override's destination path no longer exists.
    """
    has_parent = hasattr(config, "data")
    data_cfg = config.data if has_parent else config

    for field, path in _DOMAIN_OVERRIDE_DESTINATIONS.items():
        value = getattr(domain, field, None)
        if value is None:
            continue

        # Walk to the owner of the leaf, checking each hop declares the next.
        owner = data_cfg
        for hop in path[:-1]:
            if hop not in type(owner).model_fields:
                raise AttributeError(
                    f"multi_domain domain '{domain.name}' overrides '{field}', "
                    f"whose destination '{'.'.join(path)}' is stale: "
                    f"{type(owner).__name__} declares no '{hop}' block. "
                    "Update _DOMAIN_OVERRIDE_DESTINATIONS to the current "
                    "canonical path (see config/schemas/renames.py)."
                )
            owner = getattr(owner, hop)

        leaf = path[-1]
        if leaf not in type(owner).model_fields:
            raise AttributeError(
                f"multi_domain domain '{domain.name}' overrides '{field}', "
                f"whose destination '{'.'.join(path)}' is stale: "
                f"{type(owner).__name__} declares no '{leaf}' field. "
                "Update _DOMAIN_OVERRIDE_DESTINATIONS to the current "
                "canonical path (see config/schemas/renames.py)."
            )

        # Rebuild bottom-up. `model_copy` on a frozen model returns a new
        # instance; the membership gate above is what makes a moved name loud.
        patched = owner.model_copy(update={leaf: value})
        for depth in range(len(path) - 2, -1, -1):
            parent = data_cfg
            for hop in path[:depth]:
                parent = getattr(parent, hop)
            patched = parent.model_copy(update={path[depth]: patched})
        data_cfg = patched

    if not has_parent:
        return data_cfg
    return config.model_copy(update={"data": data_cfg})



def _slice_records_fill_patch(data_config, train_ds, queue_config) -> bool:
    """Whether a ``tio.Queue`` over slice-level M4Raw records would only re-serve each record.

    One uniform patch per depth-1 record, as large as the slice, is the record itself, so the
    queue adds nothing but a synchronous refill every ``queue_length`` samples that leaves
    the GPU idle for the whole decode (measured 130-270 s per 100 slices on sabine).
    """
    if (
        getattr(data_config, "dataset_type", None) != "m4raw"
        or not getattr(data_config, "slice_level_records", False)
        or queue_config.sampler_type != "uniform"
        or queue_config.samples_per_volume != 1
    ):
        return False
    index = getattr(train_ds, "index", None)
    patch = tuple(queue_config.patch_size[:2])
    return (
        isinstance(index, list)
        and bool(index)
        and all(
            isinstance(r, dict) and r.get("shape") and tuple(r["shape"][-2:]) == patch
            for r in index
        )
    )


def _slice_records_val_inherits_workers(data_config, val_cfg) -> bool:
    """Whether validation loads slice-level M4Raw records with the training worker count.

    Only when the arm leaves ``validation.loader.num_workers`` undeclared: a declared
    value, including 0, is the user's.
    """
    return (
        val_cfg is not None
        and getattr(data_config, "dataset_type", None) == "m4raw"
        and bool(getattr(data_config, "slice_level_records", False))
        and "num_workers" not in val_cfg.loader.model_fields_set
    )


class DataPipelineDirector:
    """Orchestrates data builders (DatasetInstantiator, DataLoaderBuilder, TransformBuilder)
    to create training and validation data loaders.
    """

    @accepts_builder_context
    def __init__(self, ctx: BuilderContext) -> None:
        config = ctx.config
        self._config = config
        #: The applied train/val/test split of the last paired build (for provenance).
        self.split_provenance: dict = {}

    def coil_process(self, kspace: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """Apply the configured ``physics.coil_processing`` pipeline to k-space.

        Thin SSOT seam over the module-level :func:`apply_coil_processing`,
        reading the resolved config from the frozen ``TrainingSettings`` so
        callers never re-derive the coil block. Returns ``{kspace, smaps, target}``.
        """
        return apply_coil_processing(kspace, self._config.physics.coil_processing)

    def build_dataloaders(
        self,
        num_workers: int | None = None,
        pin_memory: bool | None = None,
        device=None,
        val_shuffle: bool = False,
        val_batch_size: int | None = None,
        persistent_workers: bool | None = None,
    ) -> tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders.

        ``persistent_workers=None`` reads ``data.loader.persistent_workers``; a
        caller that walks the loaders once (export, dataset dumps) passes
        ``False`` so no worker pool outlives the walk.
        """
        config = self._config
        data_config = config.data if hasattr(config, "data") else config

        # Resolve worker count from config when the caller doesn't override it.
        # The old ``num_workers=0`` default was a latent single-process trap:
        # any caller that omitted it got a serial loader despite
        # ``data.num_workers`` (default 4). ``None`` → read the config knob.
        if num_workers is None:
            num_workers = getattr(data_config.loader, "num_workers", 0)
        num_workers = int(num_workers or 0)

        # CHOKE POINT for every training loader built from this director. The
        # declared count is a CEILING: it is clamped down to this rank's share of
        # the node's cores and never raised, so an arm that already fits is
        # untouched. Before this, ``num_workers`` had no topology term at all, so
        # a 4-rank node spawned 4x the declared decoders and thrashed -- which is
        # why a multi-GPU run could come out SLOWER than a single-GPU one.
        # It has to live here rather than in the leaf builder: line ~691 below
        # unconditionally overwrites ``queue_config.num_workers`` with this value,
        # so a clamp applied downstream in ``torchio_queue_builder`` is dead code.
        _topology = resolve_run_topology()
        _reserved = int(data_config.loader.reserved_cores)
        _train_workers = clamp_worker_count(
            num_workers, _topology, role="train", reserved_cores=_reserved
        )
        num_workers = _train_workers.workers

        # Same treatment for pin_memory: ``None`` → read ``data.pin_memory``
        # (default True). Callers used to hardcode ``pin_memory=True``, so an
        # explicit ``data.pin_memory: false`` was silently ignored on the CPU /
        # shared-host path where pinning wastes memory (pitfall #15).
        if pin_memory is None:
            pin_memory = bool(getattr(data_config.loader, "pin_memory", True))

        # Shape-invariant loader knobs, honored on every leaf chain below (the
        # leaf ``build()`` already no-ops both when ``num_workers == 0``).
        # ``persistent_workers`` avoids respawning workers every epoch; wiring
        # it here fixes the val + npy_slice/mrixfields/full-sampler paths that
        # previously ignored the knob (pitfall #15).
        _persistent = (
            bool(data_config.loader.persistent_workers)
            if persistent_workers is None
            else bool(persistent_workers)
        )
        _prefetch = getattr(data_config.loader, "prefetch_factor", 2)

        # Validation loader worker policy (fixes the cohort-wide cgroup host-RAM
        # OOM at the first validation, 2026-07). ``all_slices`` validation walks
        # every slice of every val volume in volume-major order, and each sample
        # is a fresh ~226 MB volume decode. Building the val loader with the same
        # ``num_workers`` as training makes each val worker decode a *separate*
        # volume in parallel (prefetch fan-out) at first validation — ON TOP of
        # the still-resident persistent TRAIN workers — so peak host RAM jumps by
        # ``val_workers`` whole volumes and the process is oom_killed before any
        # metric is written. ``validation.loader.num_batches`` caps the loop, not
        # this spawn-time fan-out, which is why capping it did not help.
        # Validation is infrequent and capped, and volume-major order already
        # gives perfect locality with a single reader, so the DEFAULT loads it in
        # the main process. Training keeps its full ``num_workers`` fan-out.
        #
        # It is a knob rather than a constant because the OOM mechanism is a
        # property of the DATASET, not of validation: it needs samples that share
        # an expensive decode. ``npy_slice`` validation has none — its samples are
        # already independent 2-D files — so 0 there is pure serialisation with
        # nothing bought. The guard below is what keeps the knob from re-opening
        # the bug: it raises on the volume-backed sets instead of degrading (#9),
        # so a nonzero value cannot silently reproduce the oom_kill hours later.
        _val_cfg = getattr(self._config, "validation", None)
        _val_num_workers = (
            int(getattr(_val_cfg.loader, "num_workers", 0)) if _val_cfg is not None else 0
        )
        if _slice_records_val_inherits_workers(data_config, _val_cfg):
            # Independent one-slice reads, so no whole-volume decode fans out; the
            # main-process default read ~2 s per slice with the GPU idle (sabine).
            _val_num_workers = int(data_config.loader.num_workers)
        # Same ceiling for validation. A declared 0 -- the default, and what the
        # guard below insists on for volume-backed sets -- passes through
        # untouched, so this cannot resurrect a nonzero count where 0 was chosen.
        _val_workers = clamp_worker_count(
            _val_num_workers, _topology, role="val", reserved_cores=_reserved
        )
        _val_num_workers = _val_workers.workers

        # Kept for provenance and tests: what was asked for vs what will run.
        # torch raises on persistent_workers with 0 workers, and the val loader
        # defaults to 0, so the flag is decided per loader from its clamped count.
        _persistent_train = _persistent and num_workers > 0
        _persistent_val = _persistent and _val_num_workers > 0
        self.worker_decisions = {
            "train": replace(_train_workers, persistent=_persistent_train).to_dict(),
            "val": replace(_val_workers, persistent=_persistent_val).to_dict(),
        }

        from spectramr.data.builders.dataset_instantiator import DatasetInstantiator
        from spectramr.data.builders.manifest_loader import ManifestLoader
        from spectramr.data.builders.torchio_queue_builder import (
            TorchIOQueueBuilder,
            TorchIOQueueConfig,
        )
        from spectramr.data.builders.torchio_transform_builder import (
            TorchIOTransformBuilder,
            TorchIOTransformConfig,
        )
        from spectramr.infrastructure.builders.leaf.data_builders import DataLoaderBuilder

        # --- 1. Transforms ---
        torchio_config = TorchIOTransformConfig.from_settings(config)
        train_transforms = TorchIOTransformBuilder.build_train_transforms(torchio_config)
        val_transforms = TorchIOTransformBuilder.build_val_transforms(torchio_config)

        # --- 2. Manifest Loading & Split ---
        _has_manifest_roles = (
            any(r.get("manifest", "") for r in data_config.manifest_roles.inputs)
            if hasattr(data_config, "manifest_roles") and data_config.manifest_roles
            else False
        )
        _is_image_folder = data_config.dataset_type == "image" and not data_config.source.index_path

        train_index, val_index = [], []
        # Self-indexed datasets build their own index in DatasetInstantiator and
        # must skip the ManifestLoader pre-split (see _self_indexed_dataset_types).
        # Their creators take no index argument, and nothing downstream reads
        # train_index/val_index -- they are passed to create_datasets and never
        # referenced again -- so for these types the pre-split can only waste a
        # manifest parse or raise.
        if (
            data_config.dataset_type not in _self_indexed_dataset_types()
            and not _has_manifest_roles
            and not _is_image_folder
        ):
            if data_config.dataset_type in (
                "contrast_aware_paired",
                "nifti_paired",
                "paired_nifti",
                "paired_mri",
            ):
                train_index, val_index = ManifestLoader.load_paired_nifti_splits(
                    data_config, partition_stamp=self.split_provenance
                )
            elif data_config.dataset_type in ("nifti", "dicom"):
                train_index, val_index = ManifestLoader.load_nifti_splits(data_config)
            else:
                train_index, val_index = ManifestLoader.load_fastmri_splits(data_config)

        # --- 3. Datasets ---
        train_ds, val_ds = DatasetInstantiator.create_datasets(
            data_config,
            train_index,
            val_index,
            train_transforms,
            val_transforms,
            coil_processing=getattr(
                getattr(self._config, "physics", None), "coil_processing", None
            ),
        )

        # Guard the validation worker knob HERE, on the raw dataset, because this
        # is the last point at which the question can be answered. Everything
        # below rebinds ``val_ds`` to something that does not forward the
        # attributes the predicate reads: neither optional wrapper defines
        # ``__getattr__``, and ``strided_validation_subset`` returns a plain
        # ``torch.utils.data.Subset`` whenever validation is capped. Asking after
        # the rebind would answer "no shared decode" for every wrapped or capped
        # arm — that is, the guard would go quiet on exactly the configurations
        # most likely to OOM. Wrapping does not change whether the samples
        # underneath share a volume decode, so the raw answer is the true one.
        if _val_num_workers > 0 and _shares_expensive_volume_decode(val_ds):
            raise ValueError(
                f"validation.loader.num_workers={_val_num_workers} on a "
                "validation set whose samples share a whole-volume decode "
                f"(dataset_type={data_config.dataset_type!r}, "
                f"slice_mode={getattr(data_config.sampling, 'slice_mode', '?')!r}"
                "). Each worker would decode a SEPARATE ~226 MB volume in "
                "parallel at spawn time, on top of the still-resident persistent "
                "train workers — the 2026-07 cohort-wide cgroup oom_kill at the "
                "first validation, which capping validation.loader.num_batches "
                "does not prevent because the cap bounds the loop, not the "
                "spawn-time fan-out. Set validation.loader.num_workers: 0 (the "
                "default; volume-major order already gives perfect locality with "
                "a single reader), or validate on a dataset whose samples do not "
                "share a decode (e.g. npy_slice)."
            )

        # --- 3a. Optional wrappers (Phase 4f Amendments H + J) ---
        train_ds, val_ds = self._apply_optional_wrappers(
            train_ds,
            val_ds,
            data_config,
            device,
            field_strength_default=getattr(
                getattr(self._config, "physics", None), "field_strength", None
            ),
        )

        # --- 4. Data Loaders ---
        _val_batch_size = (
            val_batch_size if val_batch_size is not None else data_config.loader.batch_size
        )

        # When validation is capped (num_validation_batches / num_samples), spread
        # the kept samples across the whole set so the val loop stops grading the
        # first N adjacent (background) slices of one volume — the empty-target-mask
        # warning in the tissue-seg metrics (#171). Single choke point: applies to
        # every branch-specific val loader built below. No-op when uncapped.
        val_ds = strided_validation_subset(
            val_ds, getattr(self._config, "validation", None), _val_batch_size
        )

        # ``npy_slice`` and ``mrixfields`` emit plain-dict samples (not tio.Subject), so
        # they must NOT go through the tio.Queue patch path — the queue's
        # patch-compatibility filter calls ``subj.get_images_names()`` and crashes on a
        # dict. Route both through the leaf DataLoaderBuilder (its robust collate handles
        # dict samples, incl. the B-1.1 ``sources`` [B,N,1,H,W] tuple), same as the val
        # loader already does. Without this every mrixfields arm crashes at train start.
        if data_config.dataset_type in ("npy_slice", "mrixfields"):
            # slice_mode='all_slices' expands each container to one sample per foreground
            # depth slice, which is the right sampling but makes a plain shuffled order
            # pathological: every sample becomes a fresh ~226 MB volume decode because the
            # working set (~45 volumes) cannot be held resident. VolumeBlockedSliceSampler
            # restores tio.Queue's load-few / emit-many / shuffle-the-buffer behaviour for
            # this plain-dict dataset, so one decode is amortised over every slice of every
            # container sharing that volume. Returns None (-> ordinary shuffle) for central
            # / volume mode and for npy_slice, which has no volume to amortise. Validation
            # keeps the natural volume-major order (shuffle=False already gives the same
            # locality, deterministically).
            train_sampler = _build_slice_sampler(train_ds)
            _train_builder = (
                DataLoaderBuilder(config, dataset=train_ds)
                .with_batch_size(data_config.loader.batch_size)
                .with_num_workers(num_workers)
                .with_pin_memory(pin_memory)
                .with_persistent_workers(_persistent_train)
                .with_prefetch_factor(_prefetch)
            )
            train_loader = (
                _train_builder.with_sampler(train_sampler).build()
                if train_sampler is not None
                else _train_builder.with_shuffle(True).build()
            )

            val_loader = (
                DataLoaderBuilder(config, dataset=val_ds)
                .with_batch_size(_val_batch_size)
                .with_num_workers(_val_num_workers)
                .with_pin_memory(pin_memory)
                .with_persistent_workers(_persistent_val)
                .with_prefetch_factor(_prefetch)
                .with_shuffle(val_shuffle)
                .build()
            )
            return train_loader, val_loader

        # 3D Queue Builder handling patch-based volumetric datasets
        queue_config = TorchIOQueueConfig.from_training_config(data_config)

        if queue_config.sampler_type == "full" or _slice_records_fill_patch(
            data_config, train_ds, queue_config
        ):
            # Full-slice training (whole-volume sampler, no patching): route TRAINING
            # through the same no-Queue DataLoaderBuilder path validation uses, so every
            # sample is a whole slice (not a random patch) and the queue's patch-compat
            # filter cannot drop subjects whose extent is below patch_size. Mirrors the
            # npy_slice/mrixfields bypass above. Without this, sampler.type='full' on the
            # training queue silently falls back to a UniformSampler (CLAUDE.md #9) — the
            # root of the ULF 'patches of a slice' symptom.
            train_loader = (
                DataLoaderBuilder(config, dataset=train_ds)
                .with_batch_size(data_config.loader.batch_size)
                .with_num_workers(num_workers)
                .with_pin_memory(pin_memory)
                .with_persistent_workers(_persistent_train)
                .with_prefetch_factor(_prefetch)
                .with_shuffle(True)
                .build()
            )
        else:
            # Honor the caller's loader knobs on the training path too. Previously
            # ``num_workers``/``pin_memory`` reached only the val loader while the
            # train queue silently re-read ``data.num_workers``/``data.pin_memory``,
            # so an explicit override was a no-op for training. ``pin_memory`` stays
            # gated on real CUDA availability (the queue config did the same).
            import torch

            queue_config.num_workers = num_workers
            queue_config.pin_memory = pin_memory and torch.cuda.is_available()
            _train_queue, train_loader = TorchIOQueueBuilder.build_train_queue(
                train_ds, queue_config
            )

        val_loader = (
            DataLoaderBuilder(config, dataset=val_ds)
            .with_batch_size(_val_batch_size)
            .with_num_workers(_val_num_workers)
            .with_pin_memory(pin_memory)
            .with_persistent_workers(_persistent_val)
            .with_prefetch_factor(_prefetch)
            .with_shuffle(val_shuffle)
            .build()
        )

        return train_loader, val_loader

    @staticmethod
    def _apply_optional_wrappers(
        train_ds, val_ds, data_config, device=None, field_strength_default=None
    ):
        """Stack optional dataset wrappers in canonical order (Phase 4f).

        Currently applied (in order — outer wrappers first, since they
        run last when ``__getitem__`` is called):

        1. ``MetaLearningDataset`` (Amendment J) — wraps the inner
           base dataset to sample meta-learning tasks. Activated when
           ``data.meta_learning.enabled=True``. Only wraps train, not val.
        2. ``LazyEncodeWrapper`` (Amendment H) — encodes each sample
           through a frozen stage-1 checkpoint on access. Activated when
           ``data.latent_diffusion.enabled=True``. Wraps both train + val
           (val must use the same encoder for parity).

        No-op when neither block is enabled — preserves pre-Phase-4f
        behavior for ~200 existing YAMLs.
        """
        # Meta-learning wrapper (train only) — must wrap BEFORE lazy-encode
        # so the meta-task sampler sees raw samples.
        meta_cfg = getattr(data_config, "meta_learning", None)
        if meta_cfg is not None and meta_cfg.enabled:
            from spectramr.data.datasets.meta_learning_dataset import MetaLearningDataset

            train_ds = MetaLearningDataset(base_dataset=train_ds, config=meta_cfg)
            logger.info(
                "[DataPipelineDirector] Wrapped train dataset in "
                f"MetaLearningDataset (support={meta_cfg.support_size}, "
                f"query={meta_cfg.query_size}, "
                f"tasks_per_epoch={meta_cfg.tasks_per_epoch})"
            )

        # Lazy-encode wrapper (both train + val) — runs in outer position
        # so the encoder sees the (optional) meta-task sampled batches.
        ld_cfg = getattr(data_config, "latent_diffusion", None)
        if ld_cfg is not None and ld_cfg.enabled:
            from spectramr.data.builders.lazy_encode import LazyEncodeWrapper

            train_ds = LazyEncodeWrapper(inner=train_ds, config=ld_cfg, device=device)
            val_ds = LazyEncodeWrapper(inner=val_ds, config=ld_cfg, device=device)
            logger.info(
                "[DataPipelineDirector] Wrapped datasets in "
                f"LazyEncodeWrapper (stage1={ld_cfg.stage1_checkpoint}, "
                f"latent_key={ld_cfg.latent_key})"
            )

        # SFC / conformal + fMRI/MRF 2026 key-population wrapper.
        # Activated when any of the ``data.expose_*`` flags is set.
        _sfc_flags = resolve_sfc_expose_flags(data_config)
        expose_jac = _sfc_flags["expose_conformal_jacobian"]
        expose_grid = _sfc_flags["expose_cortex_flatten_grid"]
        expose_design = _sfc_flags["expose_glm_design_matrix"]
        expose_scanner = _sfc_flags["expose_scanner_id"]
        expose_site = _sfc_flags["expose_site_id"]
        expose_field = _sfc_flags["expose_field_strength"]
        if any(_sfc_flags.values()):
            from spectramr.data.builders.sfc_conformal_fmri_keys_wrapper import (
                SFCConformalFMRIKeysWrapper,
            )

            _wrap_kw = dict(
                **_sfc_flags,
                field_strength_default=field_strength_default,
            )
            train_ds = SFCConformalFMRIKeysWrapper(train_ds, **_wrap_kw)
            val_ds = SFCConformalFMRIKeysWrapper(val_ds, **_wrap_kw)
            logger.info(
                "[DataPipelineDirector] Wrapped datasets in "
                "SFCConformalFMRIKeysWrapper (jac=%s grid=%s design=%s scanner=%s "
                "site=%s field_strength=%s)",
                expose_jac,
                expose_grid,
                expose_design,
                expose_scanner,
                expose_site,
                expose_field,
            )

        return train_ds, val_ds

    # ── Phase 4e of TODO/audit/data_layer_unification_plan.md ───────────────

    def build_multi_domain_dataloaders(
        self,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> Balancer:
        """Build per-domain loaders + a balancer wrapper (Phase 4e).

        Required schema: ``data.multi_domain.enabled=true`` with at
        least two entries in ``data.multi_domain.domains``. Each entry
        may override ``data_root`` / ``index_path`` / ``dataset_type``;
        unset fields inherit from the parent data config.

        Returns:
            A :class:`Balancer` (from :py:mod:`spectramr.data.builders.site_balancer`)
            that yields domain-tagged batches per the declared balancing
            strategy. The training-strategy code iterates the balancer
            directly — it never sees the underlying DataLoaders.

        Raises:
            ValueError: When ``multi_domain.enabled`` is False (caller
                should use ``build_dataloaders`` instead).
        """
        from spectramr.data.builders.site_balancer import build_balancer

        config = self._config
        data_config = config.data if hasattr(config, "data") else config
        md = getattr(data_config, "multi_domain", None)
        if md is None or not md.enabled:
            raise ValueError(
                "build_multi_domain_dataloaders requires "
                "data.multi_domain.enabled=true. "
                "For single-domain training, use build_dataloaders()."
            )
        # No arity guard here: MultiDomainConfigSchema already refuses
        # enabled=true with fewer than two domains at construction, so a
        # second check could never fire.
        per_domain_loaders: dict[str, DataLoader] = {}
        weights: dict[str, float] = {}
        for domain in md.domains:
            domain_settings = _apply_domain_overrides(config, domain)

            sub_director = DataPipelineDirector(domain_settings)
            train_loader, _val_loader = sub_director.build_dataloaders(
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            per_domain_loaders[domain.name] = train_loader
            weights[domain.name] = domain.weight

        return build_balancer(
            loaders=per_domain_loaders,
            strategy=md.balancing,
            weights=weights,
        )
