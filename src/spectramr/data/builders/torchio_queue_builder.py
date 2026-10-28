"""TorchIO Queue Builder - Phase T2 Refactoring.

Extracts queue creation logic from ConsolidatedDatasetFactory.

Provides reusable, testable builders for constructing TorchIO Queue pipelines
for training and validation, with support for:
- Queue length configuration
- Samples per volume tuning
- Patch-based sampling with uniform distribution
- Validation strategies (queue-based vs direct)
- Worker configuration for parallel data loading

Usage:
    config = TorchIOQueueConfig(
        queue_length=256,
        samples_per_volume=16,
        patch_size=(256, 256),
    )
    val_queue, val_loader = TorchIOQueueBuilder.build_val_queue(
        val_dataset, config, slice_aware=False
    )

    .. mermaid::

        graph TD
            A[Subject Dataset] -->|Input| B(QueueBuilder)
            C[Config] -->|Params| B
            B -->|UniformSampler| D[Patches]
            D -->|Enqueue| E[TorchIO Queue]
            E -->|Dequeue| F[DataLoader]

            subgraph Parallel Workers
                E
            end
"""

import logging
from dataclasses import dataclass

import torch
import torchio as tio

from spectramr.config.schemas.data import SUBJECT_PREFETCH_AUTO, validate_subject_prefetch
from spectramr.core.worker_seeding import seed_worker
from spectramr.data.builders.prefetching_queue import PrefetchingQueue, resolve_subject_prefetch

logger = logging.getLogger(__name__)


def _require_dry_iter(dataset: object, role: str) -> None:
    """Fail loud at build time if a queued dataset lacks ``dry_iter``.

    ``tio.Queue.iterations_per_epoch`` calls ``subjects_dataset.dry_iter()``; a
    SubjectsDataset that omits it crashes at the FIRST training step with a
    cryptic ``'<Dataset>' object has no attribute 'dry_iter'`` (the 2026-06
    oracle_bssfp / exp_vf_29 incident). Converting that into a clear pre-flight
    error (CLAUDE.md #9 fail-loud) means a new dataset can never silently reach
    the queue without the method again — the fix is to add ``dry_iter`` (see
    any dataset in ``spectramr.data.datasets`` for the one-line stub pattern).
    """
    if not callable(getattr(dataset, "dry_iter", None)):
        raise TypeError(
            f"{role} dataset {type(dataset).__name__} cannot be wrapped in a "
            f"tio.Queue: it has no dry_iter() method, which "
            f"tio.Queue.iterations_per_epoch requires. Add a dry_iter() returning "
            f"a length-correct list of stub tio.Subject shells (mirror "
            f"OracleBssfpDataset.dry_iter)."
        )


@dataclass
class TorchIOQueueConfig:
    """Configuration for TorchIO Queue pipeline.

    Attributes:
        queue_length: Maximum number of patches in queue (memory limit)
            Higher = more memory used but less I/O blocking
            Typical: 128-512 depending on patch size and RAM

        samples_per_volume: Number of patches to extract per volume per epoch
            Higher = more training steps per epoch but longer epoch time
            Typical: 4-32 depending on volume size and desired augmentation

        patch_size: Target patch size for sampling
            Should match transform pipeline patch_size
            Example: (256, 256) for 2D, (64, 64, 64) for 3D

        batch_size: Number of patches per training batch
            Typical: 8-32 depending on model and VRAM

        num_workers: Number of parallel workers for data loading
            0 = main process only (debug/testing)
            > 0 = parallel dataloaders (production)

        use_queue_for_validation: Whether validation uses Queue (True) or Direct (False)
            True = queue-based (controlled sampling, slower)
            False = direct access (full volume, faster)

        shuffle_subjects_train: Whether to shuffle train subjects between epochs
        shuffle_patches_train: Whether to shuffle patches within queue
        shuffle_subjects_val: Whether to shuffle validation subjects
        shuffle_patches_val: Whether to shuffle validation patches

        subject_prefetch: ``"auto"`` or an int >= 1 -- the train queue's subject
            loader prefetch depth, resolved at build time against the clamped
            worker count (``data.sampling.subject_prefetch``, #2673)
    """

    # Queue parameters
    queue_length: int = 256
    samples_per_volume: int = 16
    subject_prefetch: int | str = SUBJECT_PREFETCH_AUTO
    dataset_type: str = "fastmri_kspace"

    # Sampling parameters
    patch_size: tuple[int, ...] = (256, 256)

    # DataLoader parameters
    batch_size: int = 8
    num_workers: int = 0

    # Validation strategy
    use_queue_for_validation: bool = False

    # Shuffling
    shuffle_subjects_train: bool = True
    shuffle_patches_train: bool = True
    shuffle_subjects_val: bool = False
    shuffle_patches_val: bool = False

    # Collation
    collation: dict | None = None
    enable_slab_mode: bool = False

    # ── Phase 5: sampler variant info ────────────────────────────────
    # When populated from a modes-aware config, the train sampler kind
    # routes through spectramr.data.builders.sampler_factory. None → legacy
    # tio.UniformSampler path.
    sampler_type: str = "uniform"
    sampler_label_name: str | None = None
    sampler_label_probabilities: dict | None = None
    sampler_probability_map: str | None = None

    # Optimization
    pin_memory: bool = False  # Set dynamically based on CUDA availability
    prefetch_factor: int = 2
    persistent_workers: bool = False  # Keep workers alive between epochs (requires num_workers > 0)

    def __post_init__(self):
        """Validate configuration."""
        if self.queue_length <= 0:
            raise ValueError(f"queue_length must be positive, got {self.queue_length}")

        if self.samples_per_volume <= 0:
            raise ValueError(f"samples_per_volume must be positive, got {self.samples_per_volume}")

        validate_subject_prefetch(self.subject_prefetch)

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")

        if not self.patch_size or any(p <= 0 for p in self.patch_size):
            raise ValueError(f"patch_size must contain positive values, got {self.patch_size}")

    @staticmethod
    def from_training_config(config) -> "TorchIOQueueConfig":
        """Create from TrainingSettings config object.

        Extracts relevant queue/sampling parameters from config.

        Args:
            config: TrainingSettings (or dict-like with getattr support)

        Returns:
            TorchIOQueueConfig instance
        """
        dataset_type = config.dataset_type if hasattr(config, "dataset_type") else "fastmri_kspace"

        # Phase 3 (data-layer unification plan): prefer per-mode sampler
        # overrides from ``config.modes.{train,val}.*`` when present, falling
        # back to legacy flat fields for older fixtures / mocked configs that
        # don't carry the ``modes`` block.
        modes = getattr(config, "modes", None)
        train_mode = getattr(modes, "train", None) if modes is not None else None
        val_mode = getattr(modes, "val", None) if modes is not None else None

        if train_mode is not None:
            samples_per_volume = train_mode.sampler.samples_per_volume
            queue_length = train_mode.sampler.queue_length
        else:
            samples_per_volume = config.sampling.samples_per_volume
            queue_length = config.sampling.queue_length

        patch_size = config.sampling.patch_size
        subject_prefetch = config.sampling.subject_prefetch
        batch_size = config.loader.batch_size
        num_workers = config.loader.num_workers

        # A non-positive queue_length is a config error and RAISES (in
        # ``TorchIOQueueConfig.__post_init__`` below) — never auto-corrects.
        # The silent rewrite this replaced guarded the 7 ``experiment_11*``
        # arms that set ``queue_length: 0``; all have since been migrated and
        # both schema paths (v6.1 sampler + legacy flat field) enforce ge=1,
        # so a warning-and-continue here is pitfall #9/#10: training would
        # sample from a queue the user never configured.

        # Validation strategy
        # Phase 3: ``modes.val.sampler.type == "uniform"`` is the
        # mode-aware equivalent of the legacy ``use_queue_for_validation``
        # boolean. Prefer the mode-aware signal when present.
        if val_mode is not None:
            use_queue_for_validation = val_mode.sampler.type == "uniform"
        else:
            use_queue_for_validation = (
                config.use_queue_for_validation
                if hasattr(config, "use_queue_for_validation")
                else False
            )
        enable_slab_mode = (
            config.sampling.enable_slab_mode if config.sampling.enable_slab_mode else False
        )
        collation = config.collation if hasattr(config, "collation") else None
        pin_memory = config.loader.pin_memory and torch.cuda.is_available()
        # ``prefetch_factor`` is the authoritative knob; the deprecated
        # ``max_prefetch`` is folded into it at schema-load (_fold_legacy_prefetch).
        prefetch_factor = getattr(config.loader, "prefetch_factor", config.loader.prefetch_factor)
        persistent_workers = config.loader.persistent_workers

        # Or check for legacy "slice_aware" name
        if not use_queue_for_validation and hasattr(config, "slice_aware"):
            use_queue_for_validation = config.slice_aware

        # Phase 5: thread sampler variant fields when train_mode is set.
        # When train_mode is None (legacy / mocked config), the queue
        # builder falls back to tio.UniformSampler — preserves all
        # existing behavior.
        if train_mode is not None:
            sampler_type = train_mode.sampler.type
            sampler_label_name = train_mode.sampler.label_name
            sampler_label_probabilities = train_mode.sampler.label_probabilities
            sampler_probability_map = train_mode.sampler.probability_map
        else:
            sampler_type = "uniform"
            sampler_label_name = None
            sampler_label_probabilities = None
            sampler_probability_map = None

        return TorchIOQueueConfig(
            dataset_type=dataset_type,
            queue_length=queue_length,
            samples_per_volume=samples_per_volume,
            subject_prefetch=subject_prefetch,
            patch_size=patch_size,
            batch_size=batch_size,
            num_workers=num_workers,
            use_queue_for_validation=use_queue_for_validation,
            enable_slab_mode=enable_slab_mode,
            collation=collation,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            # ── Phase 5: sampler variants ─────────────────────────────
            sampler_type=sampler_type,
            sampler_label_name=sampler_label_name,
            sampler_label_probabilities=sampler_label_probabilities,
            sampler_probability_map=sampler_probability_map,
        )


def config_to_sampler_view(config):
    """Adapter: expose ``TorchIOQueueConfig.sampler_*`` fields as a
    ``ModeSamplerSchema``-shaped object the Phase-5 factory consumes.

    The factory uses ``getattr(sampler_config, "type")`` etc., so a
    SimpleNamespace works fine — no Pydantic round-trip needed.

    Args:
        config: a :class:`TorchIOQueueConfig` carrying the
            ``sampler_type`` / ``sampler_label_name`` / etc. fields.

    Returns:
        A namespace with the four sampler-config attributes the
        factory reads.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        type=config.sampler_type,
        label_name=config.sampler_label_name,
        label_probabilities=config.sampler_label_probabilities,
        probability_map=config.sampler_probability_map,
    )



def _record_hwd(record: dict) -> tuple[int, int, int]:
    """(H, W, depth) of an index record, whose ``shape`` layout depends on the file kind.

    NIfTI manifests store ``[C, H, W, D]``; raw k-space indexes store ``[S, C, H, W]``
    (slice-leading), where depth is the leading axis.
    """
    shape = record["shape"]
    path = str(record.get("primary_path", ""))
    if len(shape) == 4 and path.endswith((".nii", ".nii.gz")):
        return shape[1], shape[2], shape[3]
    return shape[-2], shape[-1], shape[0]


def _subject_count(dataset) -> int | None:
    """``len(dataset)``, or None for a streaming dataset whose length is not knowable."""
    try:
        return len(dataset)
    except TypeError:
        return None


def _require_a_subject_fits(role: str, before: int | None, dataset, patch_size) -> None:
    """Raise when the patch filter left ``role`` with no subject, naming the patch and the extents.

    Without it the empty dataset reaches the sampler as ``num_samples=0``, which names neither
    the patch nor the volumes that were too small for it.
    """
    if not before or _subject_count(dataset) != 0:
        return
    raise ValueError(
        f"All {before} {role} subject(s) are smaller than patch_size={tuple(patch_size)} on at "
        "least one axis, so none can be sampled. Declare a patch_size every volume covers "
        "(see the [QUEUE] drop warning above for the per-run count), or drop the axis that "
        "exceeds the data (a 1 on the depth axis makes the patch 2-D)."
    )


class TorchIOQueueBuilder:
    """Builder for composing TorchIO Queue-based data loaders.

    Handles:
    - Queue creation with configurable length and sampling
    - Patch-based sampling with UniformSampler
    - Worker configuration for parallel loading
    - Validation strategy dispatch (queue vs direct)
    - DataLoader wrapping with proper batch composition

    Key principle: Queue parameters affect training dynamics:
    - queue_length: Memory vs I/O tradeoff
    - samples_per_volume: Augmentation strength (more=more variety)
    - num_workers: Throughput vs resource usage
    """

    @staticmethod
    def _filter_patch_compatible_subjects(dataset, patch_size: tuple):
        """Drop subjects whose spatial extent is smaller than ``patch_size``.

        F10/E35 (2026-05-21 smoke audit): when a volume in the manifest
        has fewer Z-slices than the configured ``patch_size[2]``,
        ``tio.UniformSampler`` silently produced an empty patch
        (``shape=(..., 128, 128, 0)``) which then crashed downstream
        in ``diffusion.py::_prepare_diffusion_inputs`` with
        ``5D target_batch has a zero-sized dim``. The error message
        suggested manually filtering the manifest; this implements that
        filter generically at the queue-build layer.

        Subjects with sufficient extent are kept unchanged; subjects
        too small are skipped with a single info log. Returns a list-
        like that tio.Queue accepts (the same SubjectsDataset class
        when no filtering was needed, or a plain list otherwise).
        """
        # TorchIO always stores data as ``(C, H, W, D)`` 4-D, even for
        # 2-D patches (D=1). A 2-element ``patch_size`` means
        # "2-D in the H-W plane"; pad with 1 on the depth axis so the
        # extent check matches the TorchIO sampler's own convention.
        patch_size_3d = tuple(patch_size) + (1,) * (3 - len(patch_size))
        ph, pw, pd = patch_size_3d

        # ── FAST PATH (no voxel load): filter from per-record shape metadata. ──
        # The legacy probe below iterates ``dataset`` and touches ``.data`` on
        # every subject. For a LAZY dataset (e.g. ``UniversalMRIDataset``, whose
        # ``__getitem__`` loads + coil-processes the volume on access) that
        # materialises the ENTIRE corpus here at queue-build time — 1024 M4Raw
        # volumes x 38 MB ~= 39 GB, a host OOM that fires independent of
        # ``num_workers`` and before the first training step (F-OOM 2026-06-08).
        # The manifest already records each volume's ``shape``, so the spatial
        # extent check needs no pixels. ``index`` records are the raw-file shape
        # ([S, C, H, W] for fastmri/M4Raw k-space): H/W are the trailing two axes
        # (universal for image + k-space layouts) and the depth/slice axis is the
        # leading dim, which only constrains 3-D patches (``pd > 1``).
        index = getattr(dataset, "index", None)
        if (
            isinstance(index, list)
            and len(index) == len(dataset)
            and all(isinstance(r, dict) and r.get("shape") and len(r["shape"]) >= 2 for r in index)
        ):
            extents = [_record_hwd(r) for r in index]
            keep_idx = [
                i
                for i, (h, w, d) in enumerate(extents)
                if h >= ph and w >= pw and (pd <= 1 or d >= pd)
            ]
            dropped = len(index) - len(keep_idx)
            if dropped:
                logger.warning(
                    f"[QUEUE] F10/E35: dropped {dropped} subject(s) with spatial "
                    f"extent < patch_size={patch_size} (from manifest shape, no "
                    f"voxel load)."
                )
                # Reduce the dataset's own index in place — keeps it lazy (no
                # subjects materialised) and preserves its dry_iter/transform.
                dataset.index = [index[i] for i in keep_idx]
            return dataset

        # ── FALLBACK: no shape metadata (e.g. a pre-built ``tio.SubjectsDataset``
        # whose subjects are already resident in memory, so probing them adds no
        # materialisation). Inspect the real ``(C, H, W, D)`` tensor extent. ──
        keep = []
        dropped = 0
        for subj in dataset:
            ok = True
            for img_name in subj.get_images_names():
                img = subj[img_name]
                data = getattr(img, "data", None)
                if data is None:
                    continue
                # Inspect the trailing 3 spatial dims (H, W, D).
                spatial = tuple(data.shape[-3:])
                for s, p in zip(spatial, patch_size_3d):
                    if s < p:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                keep.append(subj)
            else:
                dropped += 1
        if dropped:
            logger.warning(
                f"[QUEUE] F10/E35: dropped {dropped} subject(s) with "
                f"spatial extent < patch_size={patch_size}. Configure a "
                f"smaller patch_size or filter the manifest upstream to "
                f"silence this warning."
            )
        # If nothing was dropped, return the original dataset unchanged
        # (preserves SubjectsDataset transform behavior). Otherwise
        # wrap in tio.SubjectsDataset re-using the transform.
        if dropped == 0:
            return dataset
        transform = getattr(dataset, "transform", None) or getattr(dataset, "_transform", None)
        return tio.SubjectsDataset(keep, transform=transform)

    @staticmethod
    def build_train_queue(
        train_dataset,
        config: TorchIOQueueConfig,
    ) -> tuple[tio.Queue, tio.SubjectsLoader]:
        """Build training queue and loader.

        Args:
            train_dataset: TorchIO Subject dataset for training
            config: TorchIOQueueConfig

        Returns:
            (train_queue, train_loader) tuple
        """
        logger.debug(
            f"[QUEUE] Building training queue: "
            f"length={config.queue_length}, "
            f"samples_per_volume={config.samples_per_volume}"
        )

        # F10/E35: pre-filter subjects whose spatial extent is too small
        # for the configured patch_size. Prevents the downstream
        # zero-depth ``5D target_batch has a zero-sized dim`` crash.
        n_train = _subject_count(train_dataset)
        train_dataset = TorchIOQueueBuilder._filter_patch_compatible_subjects(
            train_dataset,
            config.patch_size,
        )
        _require_a_subject_fits("train", n_train, train_dataset, config.patch_size)

        # =====================================================================
        # 1. Create Queue with patch sampling
        # =====================================================================
        # The M4RawRepetitionDataset stores underlying file paths as raw strings internally,
        # perfectly alleviating `pathlib.Path` multiprocessing pickling deadlocks inside PyTorch DataLoaders.
        effective_workers = config.num_workers

        # Phase 5: dispatch sampler variant via factory. Legacy
        # configs (sampler_type='uniform') get the same tio.UniformSampler
        # behavior; new configs can request 'label' (class-balanced) or
        # 'weighted' (probability-map driven) via
        # data.modes.train.sampler.type.
        from spectramr.data.builders.sampler_factory import (
            PATCH_SAMPLER_TYPES,
            build_patch_sampler,
        )

        if config.sampler_type in PATCH_SAMPLER_TYPES:
            sampler = build_patch_sampler(config_to_sampler_view(config), config.patch_size)
        elif config.sampler_type in ("temporal_uniform", "temporal_grid"):
            # Phase 4c samplers are Subject-level, not TorchIO patch-level —
            # they can't slot into ``tio.Queue``. Fail-loud per CLAUDE.md #9
            # rather than silently fall through to UniformSampler.
            raise ValueError(
                f"sampler.type='{config.sampler_type}' (Phase 4c) is not a "
                "tio.PatchSampler subclass and cannot be passed to tio.Queue. "
                "Use spectramr.data.builders.temporal_sampler.TemporalUniformSampler "
                "or TemporalGridSampler directly in a Subject-level iteration "
                "loop. See docs/data_pipeline_reference.rst section '4D Cine MRI'."
            )
        elif config.sampler_type in ("full", "grid"):
            # 'full' (whole-volume) is routed through DataPipelineDirector's no-Queue
            # DataLoaderBuilder bypass (full-slice training); 'grid' is inference-only
            # (build_grid_inference_handle). Reaching the training queue with either is a
            # wiring bug — raise rather than silently fall back to a UniformSampler
            # patch queue (CLAUDE.md #9), which previously degraded full-slice training
            # to random patches and hid the ULF 'patches of a slice' regression.
            raise ValueError(
                f"sampler.type='{config.sampler_type}' is a whole-volume / sliding-window "
                "sampler and must not reach the training tio.Queue. Full-slice training is "
                "routed through DataPipelineDirector's no-Queue DataLoaderBuilder bypass "
                "(it skips the queue when the resolved train sampler is 'full'); 'grid' is "
                "inference-only. Reaching here means that bypass did not fire."
            )
        else:
            raise ValueError(
                f"Unknown sampler.type='{config.sampler_type}'. "
                "Must be one of: uniform, label, weighted, full, grid, "
                "temporal_uniform, temporal_grid."
            )

        _require_dry_iter(train_dataset, "train")
        # Resolved here, not in from_training_config: the director overwrites
        # num_workers with the topology-clamped count after that call.
        subject_prefetch = resolve_subject_prefetch(
            config.subject_prefetch,
            queue_length=config.queue_length,
            samples_per_volume=config.samples_per_volume,
            num_workers=effective_workers,
        )
        logger.info(
            f"[QUEUE] Train queue subject prefetch: declared={config.subject_prefetch!r}, "
            f"resolved={subject_prefetch} per worker x {effective_workers} worker(s) "
            f"(queue_length={config.queue_length}, "
            f"samples_per_volume={config.samples_per_volume}; None = main-process loading)"
        )
        train_queue = PrefetchingQueue(
            subjects_dataset=train_dataset,
            max_length=config.queue_length,
            samples_per_volume=config.samples_per_volume,
            sampler=sampler,
            num_workers=effective_workers,
            shuffle_subjects=config.shuffle_subjects_train,
            shuffle_patches=config.shuffle_patches_train,
            subject_prefetch=subject_prefetch,
        )

        logger.debug(f"[QUEUE] Created training queue with {config.queue_length} max patches")

        # An epoch over this queue yields exactly ``n_subjects *
        # samples_per_volume`` patches, and the loader below sets
        # ``drop_last=True``. When that product is smaller than one batch the
        # loader yields ZERO batches: the training loop completes the epoch
        # without a single optimizer step and the run reports success. Fail
        # loud instead — a run that trains on nothing must not look green
        # (CLAUDE.md pitfall #10).
        try:
            _n_subjects = len(train_dataset)
        except TypeError:
            _n_subjects = None
        if _n_subjects is not None:
            _patches_per_epoch = _n_subjects * config.samples_per_volume
            if _patches_per_epoch < config.batch_size:
                raise ValueError(
                    f"[QUEUE] Training epoch would yield ZERO batches: "
                    f"{_n_subjects} subject(s) x samples_per_volume="
                    f"{config.samples_per_volume} = {_patches_per_epoch} patch(es), "
                    f"which is fewer than batch_size={config.batch_size} and the "
                    "training loader drops the trailing partial batch. Lower "
                    "data.loader.batch_size, raise "
                    "data.modes.train.sampler.samples_per_volume, or supply more "
                    "training subjects."
                )

        # =====================================================================
        # 2. Wrap in DataLoader
        # =====================================================================
        # [PHASE 2.3] Use CollationStrategySelector for centralized strategy selection
        # Select collation strategy using centralized selector
        from spectramr.config.schemas.collation import CollationConfigSchema
        from spectramr.data.collation.strategies import CollateStrategyFactory
        from spectramr.data.collation.strategy_selector import CollationStrategySelector

        collation_config = config.collation
        if collation_config is None:
            collation_config = CollationConfigSchema()
        elif isinstance(collation_config, dict):
            collation_config = CollationConfigSchema(**collation_config)

        strategy_name, strategy_kwargs = CollationStrategySelector.select_strategy(
            config=collation_config,
            dataset_type=config.dataset_type,
            enable_slab_mode=config.enable_slab_mode,
            patch_size=config.patch_size,
        )

        logger.info(f"[QUEUE] Train queue using '{strategy_name}' collation strategy")
        collate_strategy = CollateStrategyFactory.create(strategy_name, **strategy_kwargs)

        # [TORCHIO] Use SubjectsLoader instead of standard DataLoader
        # [PERF B7] pin_memory=True pins the assembled batch in page-locked memory
        # for faster H2D transfer via non_blocking=True in the training loop.
        train_loader = tio.SubjectsLoader(
            train_queue,
            batch_size=config.batch_size,
            num_workers=0,  # Queue handles multiprocessing internally
            collate_fn=collate_strategy.collate,
            pin_memory=config.pin_memory,
            # Drop the trailing partial batch: patch sampling rarely yields a
            # multiple of batch_size, and a size-1 batch breaks BatchNorm /
            # skews running stats on the training path.
            drop_last=True,
        )

        logger.debug(f"[QUEUE] Created training loader with batch_size={config.batch_size}")

        return train_queue, train_loader

    @staticmethod
    def build_val_queue(
        val_dataset,
        config: TorchIOQueueConfig,
    ) -> tuple[tio.Queue | None, tio.SubjectsLoader]:
        """Build validation queue and loader.

        Supports two strategies:
        1. Queue-based (slice_aware): Sample patches from validation volumes
        2. Direct: Access full volumes without sampling

        Args:
            val_dataset: TorchIO Subject dataset for validation
            config: TorchIOQueueConfig

        Returns:
            (val_queue, val_loader) tuple
            val_queue is None if using direct strategy
        """
        if config.use_queue_for_validation:
            # F10/E35: pre-filter subjects too small for patch_size — but ONLY
            # for the queue-based (patch-sampling) strategy, where an
            # undersized subject crashes the UniformSampler. The direct
            # strategy below loads FULL volumes (no sampler), so the same
            # subjects are valid there; filtering them dropped legitimate
            # validation volumes (and crashed with 'Subjects list is empty'
            # when every val volume was smaller than the train patch_size).
            n_val = _subject_count(val_dataset)
            val_dataset = TorchIOQueueBuilder._filter_patch_compatible_subjects(
                val_dataset,
                config.patch_size,
            )
            _require_a_subject_fits("validation", n_val, val_dataset, config.patch_size)

            logger.debug(
                f"[QUEUE] Building validation queue (queue-based strategy): "
                f"length={config.queue_length // 2}"
            )

            # =====================================================================
            # Queue-Based Validation (Patch Sampling)
            # =====================================================================
            effective_workers = config.num_workers

            _require_dry_iter(val_dataset, "validation")
            val_queue = tio.Queue(
                subjects_dataset=val_dataset,
                max_length=config.queue_length // 2,  # Smaller queue for val
                samples_per_volume=config.samples_per_volume,
                sampler=tio.data.UniformSampler(patch_size=config.patch_size),
                num_workers=effective_workers,
                shuffle_subjects=config.shuffle_subjects_val,
                shuffle_patches=config.shuffle_patches_val,
            )

            # [PHASE 2.3] Use CollationStrategySelector for centralized strategy selection
            # Select collation strategy using centralized selector (same as train)
            from spectramr.config.schemas.collation import CollationConfigSchema
            from spectramr.data.collation.strategies import CollateStrategyFactory
            from spectramr.data.collation.strategy_selector import (
                CollationStrategySelector,
            )

            collation_config = config.collation
            if collation_config is None:
                collation_config = CollationConfigSchema()
            elif isinstance(collation_config, dict):
                collation_config = CollationConfigSchema(**collation_config)

            strategy_name, strategy_kwargs = CollationStrategySelector.select_strategy(
                config=collation_config,
                dataset_type=config.dataset_type,
                enable_slab_mode=config.enable_slab_mode,
                patch_size=config.patch_size,
            )

            logger.info(f"[QUEUE] Val queue using '{strategy_name}' collation strategy")
            collate_strategy = CollateStrategyFactory.create(strategy_name, **strategy_kwargs)

            # [TORCHIO] Use SubjectsLoader instead of standard DataLoader
            val_loader = tio.SubjectsLoader(
                val_queue,
                batch_size=config.batch_size,
                num_workers=0,
                collate_fn=collate_strategy.collate,
                pin_memory=config.pin_memory,
            )

            logger.debug(
                f"[QUEUE] Created queue-based validation loader (batch_size={config.batch_size})"
            )

            return val_queue, val_loader
        else:
            logger.debug("[QUEUE] Building validation loader (direct strategy): full volumes")

            # =====================================================================
            # Direct Validation (Full Volumes)
            # =====================================================================
            # [TORCHIO] Direct Validation (Full Volumes) - Use SubjectsLoader
            # [PERF] persistent_workers=True keeps worker processes alive between epochs,
            # reducing spawn overhead (~10ms per epoch with 4 workers)
            effective_workers = config.num_workers

            val_loader = tio.SubjectsLoader(
                val_dataset,
                batch_size=1,  # One volume per batch
                num_workers=effective_workers,
                prefetch_factor=(config.prefetch_factor if effective_workers > 0 else None),
                pin_memory=config.pin_memory,
                persistent_workers=(config.persistent_workers and effective_workers > 0),
                # Seed NumPy/``random`` per worker so numpy-based transforms in
                # the direct-val workers don't duplicate draws across workers
                # (this loader has real workers; no-op at num_workers=0).
                worker_init_fn=seed_worker,
            )

            logger.debug(
                "[QUEUE] Created direct validation loader (batch_size=1, one full volume per batch)"
            )

            return None, val_loader

    @staticmethod
    def get_queue_stats(config: TorchIOQueueConfig) -> dict:
        """Compute queue statistics for debugging/profiling.

        Args:
            config: TorchIOQueueConfig

        Returns:
            Dictionary with memory and throughput estimates
        """
        patch_elements = 1
        for dim in config.patch_size:
            patch_elements *= dim

        # Assume 2 channels (complex) or 1 channel, float32
        bytes_per_element = 4  # float32
        bytes_per_patch = patch_elements * bytes_per_element * 2  # Conservative: 2 channels

        queue_memory_mb = (config.queue_length * bytes_per_patch) / (1024**2)
        patches_per_epoch = config.samples_per_volume  # Per volume
        batches_per_epoch = patches_per_epoch // config.batch_size

        return {
            "patch_size": config.patch_size,
            "patch_elements": patch_elements,
            "queue_length": config.queue_length,
            "estimated_memory_mb": queue_memory_mb,
            "samples_per_volume": config.samples_per_volume,
            "batch_size": config.batch_size,
            "batches_per_volume": batches_per_epoch,
            "num_workers": config.num_workers,
        }


__all__ = [
    "TorchIOQueueBuilder",
    "TorchIOQueueConfig",
]
