"""Dataset Instantiator Builder.

Responsible for selecting the correct Dataset class ( Universal, M4RawRepetition,
ContrastAwarePair, etc.) and instantiating it with the resolved indices and transforms.
"""

import logging
from pathlib import Path
from typing import Any

import torchio as tio
from torch.utils.data import Dataset

from spectramr.config.data_config import DataConfig
from spectramr.config.schemas.data import MRIxFieldsDataConfigSchema
from spectramr.data.datasets.universal_dataset import UniversalMRIDataset
from spectramr.data.split_utils import split_index

logger = logging.getLogger(__name__)


from spectramr.data.datasets.registry import (  # noqa: E402
    get_dataset_creator,
    register_dataset,
)


class DatasetInstantiator:
    """Builder class for instantiating specialized MRI datasets."""

    @staticmethod
    def _has_valid_manifest_roles(config: DataConfig) -> bool:
        """Check if manifest_roles has any actual (non-empty) manifest paths.

        ManifestRoleConfigSchema always creates default empty manifests, making
        config.manifest_roles always truthy. This helper checks if there are
        actual manifests to load (non-empty manifest paths).

        Returns:
            bool: True if manifest_roles has at least one non-empty manifest path
        """
        if not hasattr(config, "manifest_roles") or not config.manifest_roles:
            return False

        roles = config.manifest_roles
        for role in ["inputs", "targets", "auxiliary"]:
            items = getattr(roles, role, []) or []
            for item in items:
                if isinstance(item, dict) and item.get("manifest"):
                    return True

        return False

    #: Registration is lazy-but-once: the creators are ``classmethod``/
    #: ``staticmethod`` attributes, so they cannot be decorated at definition
    #: time without binding gymnastics. Calling this from ``create_datasets``
    #: keeps the registry populated no matter which entry point runs first,
    #: and the idempotence check in ``register_dataset`` makes repeat calls
    #: free.
    _REGISTERED = False

    @classmethod
    def _ensure_registered(cls) -> None:
        if cls._REGISTERED:
            return
        # (name, creator, indexed, serves)
        routes = [
            (
                "m4raw",
                cls._create_m4raw_repetition,
                True,
                "M4Raw multi-coil multi-repetition k-space (HDF5)",
            ),
            (
                "contrast_aware_paired",
                cls._create_contrast_aware,
                True,
                "paired NIfTI with per-contrast normalisation",
            ),
            ("nifti", cls._create_nifti_universal, True, "NIfTI volumes"),
            (
                "nifti_paired",
                cls._create_nifti_universal,
                True,
                "paired NIfTI for translation",
            ),
            ("dicom", cls._create_nifti_universal, True, "DICOM series"),
            (
                "mrixfields",
                cls._create_mrixfields,
                True,
                "MRIxFields2026 multi-field paired translation",
            ),
            (
                "preprocessed",
                cls._create_preprocessed_dataset,
                False,
                "preprocessed pipeline outputs",
            ),
            (
                "synthetic",
                cls._create_synthetic_dataset,
                False,
                "synthetic smoke-test data",
            ),
            (
                "pde_synthetic",
                cls._create_pde_synthetic_dataset,
                False,
                "PDE benchmarks (Burgers, Darcy)",
            ),
            (
                "quantitative",
                cls._create_quantitative_dataset,
                False,
                "quantitative parameter maps (T1/T2/PD)",
            ),
            ("cine", cls._create_cine_dataset, False, "4-D cine MRI"),
            (
                "bart_kspace",
                cls._create_bart_dataset,
                False,
                "BART .cfl/.hdr non-Cartesian / multi-coil k-space",
            ),
            (
                "bids_paired",
                cls._create_bids_paired_dataset,
                False,
                "BIDS low-field/high-field paired NIfTI",
            ),
            (
                "png_paired",
                cls._create_png_paired_dataset,
                False,
                "paired-PNG super-resolution",
            ),
            (
                "field_ref",
                cls._create_field_ref_dataset,
                False,
                "NIfTI real-B0/B1 field reference",
            ),
            (
                "ismrmrd_kspace",
                cls._create_ismrmrd_dataset,
                False,
                "ISMRMRD measured-trajectory k-space",
            ),
            (
                "oracle_bssfp",
                cls._create_oracle_bssfp_dataset,
                False,
                "phase-cycled bSSFP + analytical Hz B0",
            ),
            (
                "npy_slice",
                cls._create_npy_slice_dataset,
                False,
                "pre-paired .npy slices",
            ),
            (
                "fmri",
                cls._create_fmri_dataset,
                False,
                "4-D BOLD series with a legible time axis",
            ),
        ]
        for name, fn, indexed, serves in routes:
            register_dataset(name, fn, indexed=indexed, serves=serves)

        # Out-of-tree datasets: the ONLY seam by which a ``register_dataset``
        # call made OUTSIDE the spectramr tree joins DATASET_REGISTRY. Mirrors
        # ``populate_model_registry``'s ``discover_plugins("spectramr.models")``.
        #
        # Adding "spectramr.datasets" to ``ENTRY_POINT_GROUPS`` alone would have
        # been the registered-but-unwired shape non-negotiable 16 is about: the
        # group would exist, a plugin could declare it, and nothing would ever
        # read it. This call is what makes the group real.
        from spectramr.plugins import discover_plugins

        discover_plugins("spectramr.datasets")

        cls._REGISTERED = True

    @classmethod
    def create_datasets(
        cls,
        config: DataConfig,
        train_index: list[dict[str, Any]],
        val_index: list[dict[str, Any]],
        train_transforms: tio.Compose,
        val_transforms: tio.Compose,
        coil_processing: Any = None,
    ) -> tuple[Dataset, Dataset]:
        """Factory method to create train and validation datasets based on config.

        ``coil_processing`` is the resolved ``physics.coil_processing`` block (the
        4-axis SSOT) threaded down to the subject builder so the data-load reads it
        directly. ``None`` keeps the pure legacy ``data.coil_processing_mode`` path.
        """

        cls._coil_processing = coil_processing
        # ``fastmri_image`` is neither canonical nor an alias, so it can never
        # arrive -- the second disjunct was permanently false.
        load_sensitivity = config.dataset_type != "image"

        num_virtual_coils = config.coils.num_virtual_coils

        cls._ensure_registered()

        # --- O(1) registry dispatch (CLAUDE.md non-negotiable #6) ------------
        # This replaced a 21-branch if/elif chain. Because the branch labels
        # were hand-written and the schema folds aliases BEFORE dispatch, ten
        # labels were unreachable and one canonical type (``graph_mri``) had no
        # branch at all. Membership in a registry is checkable; a chain is not.
        creator = get_dataset_creator(config.dataset_type)
        if creator is not None:
            if creator.indexed:
                return creator.fn(config, train_index, val_index, train_transforms, val_transforms)
            return creator.fn(config, train_transforms, val_transforms)

        # --- residual routes: NOT keyed on dataset_type alone ----------------
        # Deliberately left as explicit conditions rather than folded into the
        # registry, because neither is a name lookup and hiding them behind one
        # would trade a visible chain for an invisible one. Both are reachable
        # only for the residual ``kspace`` / ``image`` types, which is exactly
        # where the old chain evaluated them.

        # (a) manifest_roles is a PREDICATE on the config, not a type.
        if cls._has_valid_manifest_roles(config):
            return cls._create_manifest_role_dataset(config, train_transforms, val_transforms)

        # (b) ``image`` is a type PLUS a condition: with an index path the same
        #     type routes to the FastMRI/universal loader below.
        if config.dataset_type == "image" and not config.source.index_path:
            return cls._create_image_folder_dataset(config, train_transforms, val_transforms)

        if config.dataset_type not in ("kspace", "image"):
            from spectramr.config.schemas.data import CANONICAL_DATASET_TYPES

            # Derived from the SSOT, never restated. The hand-written list this
            # replaces OMITTED mrixfields and oracle_bssfp -- 87 arms' worth of
            # real, servable types -- while advertising ten alias spellings that
            # could never reach here.
            raise ValueError(
                f"dataset_type={config.dataset_type!r} is not recognised. "
                f"Known types: {', '.join(CANONICAL_DATASET_TYPES)}."
            )
        return cls._create_fastmri_universal(
            config,
            train_index,
            val_index,
            train_transforms,
            val_transforms,
            load_sensitivity,
            num_virtual_coils,
        )

    @staticmethod
    def _create_m4raw_repetition(
        config, train_index, val_index, train_tfm, val_tfm, num_virtual_coils=4
    ):
        """Create M4Raw repetition dataset (cross-contrast or single-contrast).

        Args:
            config: DataConfigSchema instance.
            train_index: Training file index records.
            val_index: Validation file index records.
            train_tfm: Training transforms.
            val_tfm: Validation transforms.
            num_virtual_coils: Number of virtual coils for SVD.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        from spectramr.data.datasets.m4raw_dataset import M4RawRepetitionDataset

        kspace_pct = config.processing.kspace_percentile
        normalize_ks = config.processing.enable_kspace_normalization
        single_contrast = config.pairing.single_contrast

        train_h5 = [Path(r["primary_path"]) for r in train_index]
        val_h5 = [Path(r["primary_path"]) for r in val_index]

        if not val_h5 and config.split.validation_fraction > 0:
            raise ValueError(
                "M4Raw validation split is empty while validation_split="
                f"{config.split.validation_fraction!r} > 0. Reusing the training files "
                "for validation would leak validation into training; provide a "
                "separate validation manifest (>= 2 files total), or set "
                "validation_split: 0 for an explicit train-only run."
            )

        # Read the NEX knobs as plain attributes, NOT `getattr(config, name,
        # <default>)`. Both are declared fields on DataConfigSchema, so on a real
        # config the fallback is unreachable -- but it is exactly the shape that
        # silently disabled `rule_spatial_rank` and the SFC wrapper when their
        # leaves moved into sub-blocks (PR #644). If a later phase decomposes
        # these into, say, `data.nex.*`, a defaulting getattr would hand back
        # `complex_mean` -- the mode `.claude/rules/data.md` calls destructive for
        # M4Raw, because complex-averaging phase-incoherent reps cancels signal
        # rather than averaging it. A bare attribute read fails loud instead.
        target_mode = config.target_mode
        # The VALIDATION split may serve a different NEX target than training
        # (#695 follow-up). It exists for r2r, whose target is redrawn every
        # __getitem__: validating against it would score the noise draw, and
        # `track_best_metric` would checkpoint the luckiest epoch. None means
        # "same as training". Bare attribute read, for the reason above.
        val_target_mode = config.val_target_mode or target_mode
        # Read unconditionally so the knob is never silently defaulted: the
        # schema already refuses r2r_alpha under any other target_mode, and a
        # wrong alpha does not raise anywhere downstream -- it just shifts noise
        # between the two R2R halves. Bare attribute read, for the reason above.
        r2r_alpha = config.r2r_alpha
        nex_exclude_input = config.nex_target_exclude_input
        nex_fallback = config.nex_fallback
        # ``use_repetitions`` defaults to None in the schema: the ROUTE decides.
        # On this route the default is True -- the NEX average IS the M4Raw
        # target, and the 46 arms that never declare the knob have always
        # trained against it. An explicit false is honoured (rep 0 for both
        # input and target). Bare attribute read, for the reason above.
        declared_reps = config.use_repetitions
        use_reps = True if declared_reps is None else bool(declared_reps)
        # Slice-level index (#1757): one record per (group, slice), one slice
        # read per repetition per item. Bare read, for the reason above.
        slice_level_records = config.slice_level_records

        train_ds = M4RawRepetitionDataset(
            train_h5,
            normalize_kspace=normalize_ks,
            kspace_percentile=kspace_pct,
            transform=train_tfm,
            use_repetitions=use_reps,
            coil_processing_mode=config.coils.processing_mode,
            num_virtual_coils=num_virtual_coils,
            single_contrast=single_contrast,
            log_scaling=config.processing.enable_log_scaling,
            target_mode=target_mode,
            r2r_alpha=r2r_alpha,
            nex_target_exclude_input=nex_exclude_input,
            nex_fallback=nex_fallback,
            slice_level_records=slice_level_records,
        )
        val_ds = M4RawRepetitionDataset(
            val_h5,
            normalize_kspace=normalize_ks,
            kspace_percentile=kspace_pct,
            transform=val_tfm,
            use_repetitions=use_reps,
            coil_processing_mode=config.coils.processing_mode,
            num_virtual_coils=num_virtual_coils,
            single_contrast=single_contrast,
            log_scaling=config.processing.enable_log_scaling,
            target_mode=val_target_mode,
            r2r_alpha=r2r_alpha,
            nex_target_exclude_input=nex_exclude_input,
            nex_fallback=nex_fallback,
            slice_level_records=slice_level_records,
        )
        unit = "slice records" if slice_level_records else "groups"
        logger.info(
            f"[DATASET] M4RawRepetitionDataset with {len(train_ds)} train {unit}, "
            f"{len(val_ds)} val {unit} (single_contrast={single_contrast}, "
            f"use_repetitions={use_reps}, slice_level_records={slice_level_records})"
        )
        return train_ds, val_ds

    @staticmethod
    def _create_contrast_aware(config, train_index, val_index, train_tfm, val_tfm):
        """_create_contrast_aware.

        Args:
            config (Any): Description.
            train_index (Any): Description.
            val_index (Any): Description.
            train_tfm (Any): Description.
            val_tfm (Any): Description.
        Returns:
            Any: Description.
        """
        from spectramr.data.datasets.contrast_aware import (
            ContrastAwarePairedDataset,
            ContrastConfig,
        )

        if not config.input_contrast or not config.target_contrast:
            raise ValueError(
                "contrast_aware_paired requires 'input_contrast' and 'target_contrast'"
            )

        # Pydantic v2: use .model_dump() (v1 .dict() raises AttributeError)
        in_c = ContrastConfig(**config.input_contrast.model_dump())
        tgt_c = ContrastConfig(**config.target_contrast.model_dump())

        bidirectional_mode = config.pairing.bidirectional_mode
        allow_unpaired = config.pairing.allow_unpaired
        hf_resolution = config.pairing.hf_resolution

        train_ds = ContrastAwarePairedDataset(
            index=train_index,
            input_contrast=in_c,
            target_contrast=tgt_c,
            io_strategy="nifti",
            transform=train_tfm,
            verify_contrast=True,
            skip_nan_samples=True,
            max_skip_attempts=10,
            bidirectional_mode=bidirectional_mode,
            allow_unpaired=False,  # training always requires paired samples
        )
        val_ds = ContrastAwarePairedDataset(
            index=val_index,
            input_contrast=in_c,
            target_contrast=tgt_c,
            io_strategy="nifti",
            transform=val_tfm,
            verify_contrast=True,
            skip_nan_samples=True,
            max_skip_attempts=10,
            bidirectional_mode=bidirectional_mode,
            allow_unpaired=allow_unpaired,  # val may include unpaired ULF subjects
        )
        logger.info(
            f"[DATASET] ContrastAwarePairedDataset: input={in_c.name}, target={tgt_c.name}, "
            f"bidirectional_mode={bidirectional_mode}, allow_unpaired={allow_unpaired}, "
            f"hf_resolution={hf_resolution!r}"
        )
        return train_ds, val_ds

    @staticmethod
    def _swap_paired_arms(index: list[dict]) -> list[dict]:
        """Swap ``primary_path``↔``target_path`` (+ paired field/contrast
        labels) per record — the producer half of ``bidirectional_mode``.

        The paired ULF/HF manifest stores ``primary_path``=ULF (the default
        model INPUT) and ``target_path``=HF. ``hf_to_ulf`` routes the **HF**
        arm into the input and the **ULF** arm into the target — a genuine
        HF→ULF *translation* direction (for bidirectional testing), NOT an
        autoencoder (use ``hf_to_hf`` for that; see ``_autoencode_field``).
        Raises if a record carries no ``target_path`` to swap against, rather
        than silently no-op'ing the knob (CLAUDE.md #9/#15).
        """
        swapped: list[dict] = []
        for rec in index:
            if not rec.get("target_path"):
                raise ValueError(
                    "bidirectional_mode='hf_to_ulf' requires every paired record to "
                    "carry a 'target_path' to swap into the input arm; record "
                    f"{rec.get('file_id', rec.get('primary_path', '?'))} has none."
                )
            r = dict(rec)
            r["primary_path"], r["target_path"] = (
                rec["target_path"],
                rec["primary_path"],
            )
            for a, b in (
                ("input_field", "target_field"),
                ("input_contrast", "target_contrast"),
            ):
                if a in r and b in r:
                    r[a], r[b] = rec.get(b), rec.get(a)
            swapped.append(r)
        return swapped

    @staticmethod
    def _autoencode_field(index: list[dict], *, arm: str) -> list[dict]:
        """Rewrite each record to autoencode ONE field — the consumer of the
        ``hf_to_hf`` / ``ulf_to_ulf`` modes.

        Drops the opposite arm by clearing ``target_path`` to ``None`` so the
        subject builder takes its self-supervised branch (``target =
        primary_tensor``) → ``input ≡ target`` by construction. This is the real
        guard that a stage-1 VAE reconstructs a single field cleanly; a
        translation direction would instead train ``Dec(Enc(x)) → y`` — a
        degradation network — whenever the two arms share a spatial shape.

        ``arm='target'`` (``hf_to_hf``): promote ``target_path`` (HF) into
        ``primary_path`` so the model INPUT is HF; **raises** if a record has no
        ``target_path`` (an HF autoencoder must be fed HF; a ULF-only record
        cannot supply it, and silently skipping is #9/#16). ``arm='primary'``
        (``ulf_to_ulf``): keep ``primary_path`` (ULF). No raise.
        """
        if arm not in ("primary", "target"):
            raise ValueError(f"_autoencode_field: arm must be 'primary'|'target', got {arm!r}.")
        out: list[dict] = []
        for rec in index:
            r = dict(rec)
            if arm == "target":
                if not rec.get("target_path"):
                    raise ValueError(
                        "bidirectional_mode='hf_to_hf' requires every record to "
                        "carry a 'target_path' (the HF field to autoencode); record "
                        f"{rec.get('file_id', rec.get('primary_path', '?'))} has "
                        "none. Feed a split whose records are HF-paired."
                    )
                r["primary_path"] = rec["target_path"]
                for a, b in (
                    ("input_field", "target_field"),
                    ("input_contrast", "target_contrast"),
                ):
                    if b in r:
                        r[a] = rec.get(b)
            r["target_path"] = None
            for k in ("target_field", "target_contrast"):
                r.pop(k, None)
            out.append(r)
        return out

    @staticmethod
    def _create_nifti_universal(config, train_index, val_index, train_tfm, val_tfm):
        """_create_nifti_universal.

        Args:
            config (Any): Description.
            train_index (Any): Description.
            val_index (Any): Description.
            train_tfm (Any): Description.
            val_tfm (Any): Description.
        Returns:
            Any: Description.
        """
        paired_strategy = (
            "nifti"
            if config.dataset_type in ("nifti_paired", "paired_nifti", "paired_mri")
            else None
        )

        # Honor ``bidirectional_mode`` for paired NIfTI (#15). Previously ONLY
        # the ``contrast_aware_paired`` path read this knob, so paired-NIfTI
        # arms that set ``hf_to_ulf`` (the stage-1 LDM VAEs — to autoencode the
        # HF arm) silently kept ULF as the input and trained ULF→HF instead.
        # Swap the arms here so the advertised knob is not inert.
        if paired_strategy is not None:
            from spectramr.data.datasets.contrast_aware import (
                _VALID_BIDIRECTIONAL_MODES,
            )

            mode = config.pairing.bidirectional_mode
            if mode not in _VALID_BIDIRECTIONAL_MODES:
                raise ValueError(
                    f"bidirectional_mode={mode!r} is not recognised "
                    f"(allowed: {sorted(_VALID_BIDIRECTIONAL_MODES)})."
                )
            if mode == "hf_to_ulf":
                train_index = DatasetInstantiator._swap_paired_arms(train_index)
                val_index = DatasetInstantiator._swap_paired_arms(val_index)
                logger.info(
                    "[DATASET] nifti_paired bidirectional_mode=hf_to_ulf → "
                    "swapped primary↔target so the model INPUT is the HF arm"
                )
            elif mode == "hf_to_hf":
                train_index = DatasetInstantiator._autoencode_field(train_index, arm="target")
                val_index = DatasetInstantiator._autoencode_field(val_index, arm="target")
                logger.info(
                    "[DATASET] nifti_paired bidirectional_mode=hf_to_hf → "
                    "autoencode HF (input≡target=HF; ULF arm dropped)"
                )
            elif mode == "ulf_to_ulf":
                train_index = DatasetInstantiator._autoencode_field(train_index, arm="primary")
                val_index = DatasetInstantiator._autoencode_field(val_index, arm="primary")
                logger.info(
                    "[DATASET] nifti_paired bidirectional_mode=ulf_to_ulf → "
                    "autoencode ULF (input≡target=ULF; HF arm dropped)"
                )

        # Contrast conditioning (#15): thread the per-arm contrast_map so the
        # dataset stamps a ``contrast_idx`` on each sample and FiLM /
        # contrast-guidance actually fires. Gated on ``multi_contrast.enabled``
        # so contrast-agnostic arms (ulf_physics, unpaired_ulf) stay untouched —
        # emitting the schema-default {T1,T2,FLAIR,PD} map on them would RAISE on
        # the manifest's T1w/T2w/ADC tags.
        contrast_map = None
        mc = getattr(config, "multi_contrast", None)
        if mc is not None and getattr(mc, "enabled", False):
            contrast_map = getattr(mc, "contrast_map", None) or None

        # Field conditioning (#15): thread the CONSTANT target field (Tesla) so the
        # dataset stamps ``field_strength_target`` on each sample and the
        # field-conditioned ULF→HF restoration strategies (ulf_dps, monotone_field,
        # field_conditioned_inr, generative_refiner, ulf_redegrad_tta) run on the
        # paired-ULF data instead of raising. Gated on ``expose_field_strength_target``
        # (default True) AND a set ``mrixfields_target_field`` — field-agnostic arms
        # leave the latter None, so the value gate keeps them untouched. The paired
        # ULF task is a fixed 64mT→3T translation, so a config constant (not a
        # per-record field like the multi-field mrixfields dataset) is the honest
        # source.
        target_field_strength = None
        if getattr(config.expose, "field_strength_target", True):
            target_field_strength = getattr(config.mrixfields, "target_field", None)

        train_ds = UniversalMRIDataset(
            index=train_index,
            io_strategy="nifti",
            paired_io_strategy=paired_strategy,
            transform=train_tfm,
            load_sensitivity=False,
            contrast_map=contrast_map,
            target_field_strength=target_field_strength,
        )
        val_ds = UniversalMRIDataset(
            index=val_index,
            io_strategy="nifti",
            paired_io_strategy=paired_strategy,
            transform=val_tfm,
            load_sensitivity=False,
            contrast_map=contrast_map,
            target_field_strength=target_field_strength,
        )
        logger.info(f"[DATASET] NIFTI UniversalMRIDataset (paired={paired_strategy is not None})")

        # Per-slice / thin-slab sampling (#15 knob): a model fed a whole
        # [C,H,W,D] volume crashes (2D conv2d) or OOMs (3D conv over all D
        # slices). When data.slice_2d is set, wrap each volumetric dataset so a
        # sample is a depth-K window (K = patch_size depth: 1 = pure 2D slice
        # for a spatial_dims=2 model, >1 = a thin slab for a 3D slab model).
        # batch_size=1 = one window (gradient-light → no OOM). The stage-1 LDM
        # VAEs (autoencode 3D HF volumes) need this.
        if config.sampling.enable_slice_2d:
            from spectramr.data.datasets.slice_dataset import SliceVolumeDataset

            patch = config.sampling.patch_size
            slab_depth = int(patch[2]) if patch and len(patch) >= 3 and int(patch[2]) >= 1 else 1
            cache_size = int(getattr(config, "slice_cache_size", 2) or 2)
            train_ds = SliceVolumeDataset(train_ds, slab_depth=slab_depth, cache_size=cache_size)
            val_ds = SliceVolumeDataset(val_ds, slab_depth=slab_depth, cache_size=cache_size)
            logger.info(
                "[DATASET] slice_2d=True → wrapped in SliceVolumeDataset "
                "(slab_depth=%d from patch_size; depth-%d [C,H,W,%d] windows)",
                slab_depth,
                slab_depth,
                slab_depth,
            )

        return train_ds, val_ds

    @staticmethod
    @staticmethod
    def _regroup_mrixfields_multi_source(
        train_index,
        val_index,
        target,
        val_split,
        *,
        group_by_subject=False,
        explicit_val=False,
    ):
        """Group-aware re-split for the field-pinned mrixfields policies.

        Used by ``multi_source``, ``ulf_source``, ``prior``, ``fixed_target``,
        ``multi_contrast`` and ``multi_field`` (every policy that needs specific field
        strengths / contrasts co-resident in each split). Returns ``(train_records, val_records,
        source_fields)``. Splits on whole groups so each side holds COMPLETE
        travelling-volunteer groups and no subject leaks across the split. The group key
        is normally ``pairing_group`` (``subject_id|contrast``), which
        ``group_by_subject=True`` coarsens to ``subject_id`` alone. Two policies ask for
        that, for DIFFERENT reasons. ``multi_contrast`` needs ALL contrasts of a subject
        co-resident because it stacks them per source field, so the fine key would both
        leak the subject and prevent any complete stack. ``multi_field`` stacks over
        FIELDS within one ``subject_id|contrast`` group, so the fine key already forms
        every stack it needs — the coarsening buys leakage control alone: without it a
        subject's T1w trains while its T2w is scored, and that is the same anatomy on
        both sides of the split.
        ``source_fields`` (the shared consensus set) is derived from the FULL index only
        when ``target`` is set (multi_source); it is ``None`` otherwise and the caller
        ignores it for the non-consensus policies.

        ``explicit_val``: when True (the caller set ``data.validation_index_path``), the
        train/val split is authoritative and this method HONORS it instead of merging
        the two and re-slicing by ``val_split`` — otherwise the disjoint validation
        subjects (e.g. the ordinal-paired ``mrixfields2026_val.json``) would leak into
        training and val would collapse to a ``val_split`` fraction. The merge+group-slice
        below then runs only for the internal random-split case (val came from
        ``split_index`` on a single field-sorted manifest, where a flat slice would
        otherwise strand an entire field in one split).
        """
        full = list(train_index) + list(val_index)
        tgt_f = float(target) if target is not None else None
        source_fields = (
            sorted(
                {
                    float(r["field_strength"])
                    for r in full
                    if "field_strength" in r and float(r["field_strength"]) < (tgt_f or 0.0) - 1e-6
                }
            )
            if tgt_f is not None
            else None
        )
        if explicit_val and val_index:
            # Authoritative explicit validation manifest: honor the caller's train/val
            # split verbatim (no merge + re-slice), preventing validation-subject leakage
            # into training. source_fields is still union-derived above so the consensus
            # arity N matches across the two splits.
            return list(train_index), list(val_index), source_fields
        groups: dict[str, list] = {}
        for rec in full:
            if group_by_subject:
                key = str(rec.get("subject_id"))
            else:
                key = rec.get("pairing_group") or f"{rec.get('subject_id')}|{rec.get('contrast')}"
            groups.setdefault(key, []).append(rec)
        keys = list(groups.keys())  # manifest order (deterministic; no shuffle)
        # Split the GROUP KEYS, not the records — that is what keeps every record
        # of a subject on one side of the boundary. The unit differs from the
        # flat case, but the arithmetic does not, so it goes through the SSOT
        # rather than being re-derived: the local form here lacked the clamp to
        # ``n - 1``, so ``validation_fraction: 1.0`` held out every group and left
        # TRAINING EMPTY, and a single-group corpus silently became train-only
        # instead of raising (the SSOT's explicit escape hatch for that is
        # ``validation_fraction: 0``).
        _train_keys, _val_keys = split_index(keys, val_split)
        val_keys = set(_val_keys)
        train_recs = [r for k in keys if k not in val_keys for r in groups[k]]
        val_recs = [r for k in keys if k in val_keys for r in groups[k]]
        return train_recs, val_recs, source_fields

    @staticmethod
    def _create_mrixfields(config, train_index, val_index, train_tfm, val_tfm):
        """MRIxFields2026 travelling-volunteer multi-field paired translation.

        Args:
            config: Resolved data config (reads mrixfields_pairing_policy /
                mrixfields_target_field / mrixfields_slice_mode).
            train_index, val_index: Parsed manifest records (one per field-snapshot).
            train_tfm, val_tfm: Per-sample transforms.

        Returns:
            (train_dataset, val_dataset) of MRIxFieldsPairedDataset.
        """
        from spectramr.data.datasets.mrixfields_dataset import MRIxFieldsPairedDataset

        policy = getattr(config.mrixfields, "pairing_policy", "all_pairs")
        target = getattr(config.mrixfields, "target_field", None)
        expose_tgt = getattr(config.expose, "field_strength_target", True)
        output_contrast = getattr(config.mrixfields, "output_contrast", None)
        # multi_field only; the dataset raises if either is set under another policy.
        mf_fields = getattr(config.mrixfields, "fields", None)
        mf_heldout_fields = getattr(config.mrixfields, "heldout_fields", None)
        # central (default) = the middle slice via a memoised lazy read; all_slices =
        # every foreground slice (better sampling, opt-in on IO grounds — see the field
        # description); volume = whole [C,H,W,D]. The dataset validates the value.
        # The fallback is read FROM the schema rather than restated, so a test double
        # without the attribute cannot disagree with what a real config would resolve to
        # (the same copy-rot that made the accepted-version set three literals).
        slice_mode = getattr(
            config.mrixfields,
            "slice_mode",
            MRIxFieldsDataConfigSchema.model_fields["slice_mode"].default,
        )
        # Per-worker resident-volume budget for all_slices. Read from the schema default
        # rather than restated, same reason as slice_mode above.
        max_resident_volumes = getattr(
            config.mrixfields,
            "max_resident_volumes",
            MRIxFieldsDataConfigSchema.model_fields["max_resident_volumes"].default,
        )
        # Opt-in per-image renorm (default off — the corpus is already [0,1] and the
        # renorm erases the cross-field intensity signal). Schema-sourced default for
        # the same anti-copy-rot reason as above.
        rescale_per_image = getattr(
            config.mrixfields,
            "rescale_per_image",
            MRIxFieldsDataConfigSchema.model_fields["rescale_per_image"].default,
        )
        source_fields = None
        # Field-pinned policies need each split to hold COMPLETE pairing groups (every
        # field co-resident). The upstream split is a flat RECORD slice; on a field-sorted
        # manifest (mrixfields2026_train.json is ordered 0.1 -> 7 T) it strands an entire
        # field in one split -- e.g. ulf_source's 0.1 T source ends up only in train, so
        # validation has no 0.1 T and pairing raises "0 pairs ... fields present=[5,7]".
        # Re-split GROUP-AWARE (whole pairing groups on one side; also prevents subject
        # leakage) so every field is present in both splits. multi_source additionally
        # needs ONE shared source-field set across train/val for uniform consensus arity N.
        if policy in (
            "multi_source",
            "multi_contrast",
            "multi_field",
            "ulf_source",
            "prior",
            "fixed_target",
        ):
            train_index, val_index, regrouped_source_fields = (
                DatasetInstantiator._regroup_mrixfields_multi_source(
                    train_index,
                    val_index,
                    target,
                    float(config.split.validation_fraction or 0.1),
                    # multi_contrast stacks ALL contrasts of a subject -> the split must
                    # keep subjects whole (subject_id key), not subject|contrast.
                    # multi_contrast stacks every contrast of a subject and
                    # multi_field every field of one, so both need subjects kept
                    # whole. multi_field could group by subject|contrast and still
                    # form complete stacks, but that would put a subject's T1 in
                    # train and its T2 in val -- leakage the coarser key prevents.
                    group_by_subject=(policy in ("multi_contrast", "multi_field")),
                    # An explicit validation manifest (data.validation_index_path) is
                    # authoritative -> honor its train/val boundary instead of
                    # merging+re-splitting (which would leak val subjects into train).
                    explicit_val=bool(config.source.validation_index_path),
                )
            )
            if policy == "multi_source":
                source_fields = regrouped_source_fields
        # multi_contrast: pin the canonical contrast set from the FULL (post-regroup)
        # index so train and val stack the SAME channel count (== model.in_channels);
        # deriving it per split drifts the arity and crashes the encoder's first conv.
        pinned_contrasts = None
        if policy == "multi_contrast":
            from spectramr.data.datasets.mrixfields_dataset import canonical_contrasts

            pinned_contrasts = canonical_contrasts(list(train_index) + list(val_index))
        train_ds = MRIxFieldsPairedDataset(
            train_index,
            pairing_policy=policy,
            fields=mf_fields,
            heldout_fields=mf_heldout_fields,
            target_field=target,
            expose_target_field=expose_tgt,
            source_fields=source_fields,
            output_contrast=output_contrast,
            contrasts=pinned_contrasts,
            slice_mode=slice_mode,
            max_resident_volumes=max_resident_volumes,
            rescale_per_image=rescale_per_image,
            transform=train_tfm,
        )
        val_ds = MRIxFieldsPairedDataset(
            val_index,
            pairing_policy=policy,
            fields=mf_fields,
            heldout_fields=mf_heldout_fields,
            target_field=target,
            expose_target_field=expose_tgt,
            source_fields=source_fields,
            output_contrast=output_contrast,
            contrasts=pinned_contrasts,
            slice_mode=slice_mode,
            max_resident_volumes=max_resident_volumes,
            rescale_per_image=rescale_per_image,
            transform=val_tfm,
        )
        logger.info(
            "[DATASET] MRIxFieldsPairedDataset (policy=%s, target=%s, slice_mode=%s, "
            "max_resident_volumes=%d, rescale_per_image=%s): %d train, %d val",
            policy,
            target,
            slice_mode,
            max_resident_volumes,
            rescale_per_image,
            len(train_ds),
            len(val_ds),
        )
        return train_ds, val_ds

    @staticmethod
    def _create_fastmri_universal(
        config,
        train_idx,
        val_idx,
        train_tfm,
        val_tfm,
        load_sens,
        num_virtual_coils=4,
    ):
        """_create_fastmri_universal.

        Args:
            config (Any): Description.
            train_idx (Any): Description.
            val_idx (Any): Description.
            train_tfm (Any): Description.
            val_tfm (Any): Description.
            load_sens (Any): Description.
            num_virtual_coils (int, optional): Number of virtual coils.
        Returns:
            Any: Description.
        """
        svd_cal = getattr(config.coils, "svd_calibration_lines", None)
        coil_proc = getattr(DatasetInstantiator, "_coil_processing", None)
        train_ds = UniversalMRIDataset(
            index=train_idx,
            io_strategy="fastmri_h5",
            transform=train_tfm,
            load_sensitivity=load_sens,
            coil_processing_mode=config.coils.processing_mode,
            num_virtual_coils=num_virtual_coils,
            svd_calibration_lines=svd_cal,
            coil_processing=coil_proc,
            normalize_kspace=config.processing.enable_kspace_normalization,
            kspace_percentile=config.processing.kspace_percentile,
            log_scaling=config.processing.enable_log_scaling,
        )
        val_ds = UniversalMRIDataset(
            index=val_idx,
            io_strategy="fastmri_h5",
            transform=val_tfm,
            load_sensitivity=load_sens,
            coil_processing_mode=config.coils.processing_mode,
            num_virtual_coils=num_virtual_coils,
            svd_calibration_lines=svd_cal,
            coil_processing=coil_proc,
            normalize_kspace=config.processing.enable_kspace_normalization,
            kspace_percentile=config.processing.kspace_percentile,
            log_scaling=config.processing.enable_log_scaling,
        )
        logger.info("[DATASET] FastMRI UniversalMRIDataset")
        return train_ds, val_ds

    @staticmethod
    def _create_preprocessed_dataset(config, train_tfm, val_tfm):
        """_create_preprocessed_dataset.

        Args:
            config (Any): Description.
            train_tfm (Any): Description.
            val_tfm (Any): Description.
        Returns:
            Any: Description.
        """
        from spectramr.data.datasets.preprocessed_dataset import PreprocessedMRIDataset
        from spectramr.data.metadata.path_resolver import PathResolver

        preprocessing_dir = config.source.preprocessing_dir or config.source.root
        if not preprocessing_dir:
            raise ValueError(
                "dataset_type='preprocessed' requires 'preprocessing_dir' or 'data_root' "
                "pointing to a *_image/ preprocessing output directory."
            )
        preprocessing_dir = PathResolver.resolve(preprocessing_dir)

        kwargs = {
            "output_dir": preprocessing_dir,
            # task_type is not part of DataConfigSchema — read defensively
            # so the preprocessed-dataset path also works with the leaner
            # data schema used by tests and minimal configs.
            "task_type": getattr(config, "task_type", "reconstruction"),
            "output_domain": config.domain.output,
            "graph_type": config.domain.graph_type,
            "load_coil_sensitivity": True,
            "load_statistics": True,
            "contrasts": config.pairing.contrasts,
            "sessions": config.pairing.sessions,
            "input_artifact": config.domain.input_artifact,
            "target_artifact": config.domain.target_artifact,
            "target_contrasts": config.pairing.target_contrasts,
            "target_sessions": config.pairing.target_sessions,
        }

        train_ds = PreprocessedMRIDataset(transform=train_tfm, split="train", **kwargs)
        val_ds = PreprocessedMRIDataset(transform=val_tfm, split="val", **kwargs)

        logger.info(f"[DATASET] Preprocessed: {len(train_ds)} train, {len(val_ds)} val")
        return train_ds, val_ds

    @staticmethod
    def _create_manifest_role_dataset(config, train_tfm, val_tfm):
        """_create_manifest_role_dataset.

        Args:
            config (Any): Description.
            train_tfm (Any): Description.
            val_tfm (Any): Description.
        Returns:
            Any: Description.
        """
        from spectramr.data.metadata.index_builder import IndexBuilder

        train_subjects, val_subjects = IndexBuilder.load_from_manifest_roles(
            config, train_tfm, val_tfm
        )
        train_ds = tio.SubjectsDataset(train_subjects, transform=train_tfm)
        val_ds = tio.SubjectsDataset(val_subjects, transform=val_tfm)
        return train_ds, val_ds

    @staticmethod
    def _create_synthetic_dataset(config, train_tfm, val_tfm):
        """_create_synthetic_dataset.

        Args:
            config (Any): Description.
            train_tfm (Any): Description.
            val_tfm (Any): Description.
        Returns:
            Any: Description.
        """
        import torch

        logger.warning("[DATASET] ⚠️ SYNTHETIC DATASET SELECTED - FOR TESTING ONLY!")

        num_samples = config.sampling.num_synthetic_samples
        patch_size = config.sampling.patch_size
        if patch_size and len(patch_size) == 3:
            patch_h, patch_w, patch_d = patch_size
        else:
            patch_h, patch_w = patch_size[:2] if patch_size else (64, 64)
            patch_d = 1

        import numpy as np

        identity_affine = np.eye(4)

        synthetic_subjects = []
        for i in range(num_samples):
            input_data = torch.randn(1, patch_h, patch_w, patch_d) * 0.5 + 0.5
            target_data = input_data.clone() + torch.randn_like(input_data) * 0.1
            physics_vec = torch.rand(1, 4)

            subject = tio.Subject(
                input=tio.ScalarImage(tensor=input_data, affine=identity_affine),
                target=tio.ScalarImage(tensor=target_data, affine=identity_affine),
                mri=tio.ScalarImage(tensor=target_data, affine=identity_affine),
                physics=physics_vec,
                file_id=f"synthetic_{i:04d}",
            )
            synthetic_subjects.append(subject)

        split_idx = max(1, int(0.9 * len(synthetic_subjects)))
        train_subjects = synthetic_subjects[:split_idx]
        val_subjects = synthetic_subjects[split_idx:] or synthetic_subjects[:1]

        train_ds = tio.SubjectsDataset(train_subjects, transform=train_tfm)
        val_ds = tio.SubjectsDataset(val_subjects, transform=val_tfm)
        logger.info(f"[DATASET] Synthetic: {len(train_subjects)} train, {len(val_subjects)} val")
        return train_ds, val_ds

    @staticmethod
    def _create_pde_synthetic_dataset(config, train_tfm, val_tfm):
        """Build train/val PDE benchmark datasets (Burgers / Darcy).

        The PDE problem is selected by ``config.pde_problem`` which
        the v6.0 schema accepts as a free-form string in
        ``data.extra``-style fields. Defaults: ``problem='burgers_1d'``,
        ``resolution`` derived from ``patch_size[0]``, ``n_samples=1000``.
        """
        import torchio as tio

        from spectramr.data.datasets.pde_synthetic import make_pde_dataset

        # Pull problem-specific config out of the standard data block.
        # For PDE data we hijack `num_synthetic_samples` for sample count
        # and `patch_size[0]` for resolution — same convention as the
        # MRI synthetic dataset so we don't need a new config field.
        problem = getattr(config, "pde_problem", None) or "burgers_1d"
        n_train = config.sampling.num_synthetic_samples or 1000
        n_val = max(1, n_train // 10)
        if config.sampling.patch_size and len(config.sampling.patch_size) >= 1:
            resolution = int(config.sampling.patch_size[0])
        else:
            resolution = 64

        logger.info(
            f"[DATASET] PDE synthetic — problem={problem}, "
            f"resolution={resolution}, n_train={n_train}, n_val={n_val}"
        )
        train_pde = make_pde_dataset(
            problem=problem,
            n_samples=n_train,
            resolution=resolution,
            seed_offset=0,
        )
        val_pde = make_pde_dataset(
            problem=problem,
            n_samples=n_val,
            resolution=resolution,
            seed_offset=1_000_000,  # disjoint from train seeds
        )

        # Eagerly materialise the subjects so TorchIO's SubjectsDataset
        # can apply the standard transform pipeline. For small benchmark
        # sizes this is fast; for large datasets we'd switch to
        # lazy-loading via a wrapper Subject.
        train_subjects = [train_pde[i] for i in range(len(train_pde))]
        val_subjects = [val_pde[i] for i in range(len(val_pde))]
        train_ds = tio.SubjectsDataset(train_subjects, transform=train_tfm)
        val_ds = tio.SubjectsDataset(val_subjects, transform=val_tfm)
        return train_ds, val_ds

    @staticmethod
    def _create_quantitative_dataset(config, train_tfm, val_tfm):
        """Build train/val quantitative parameter-map datasets (Phase 4b).

        Activated by ``dataset_type='quantitative'`` AND
        ``data.quantitative.enabled=true``. Splits a BIDS-style directory
        into train/val by the standard ``validation_split`` fraction.

        The :class:`QuantitativeMapDataset` yields a TorchIO Subject with
        an ``input`` ScalarImage stacking all declared input contrasts plus
        one ScalarImage per declared target map.
        """
        from spectramr.data.datasets.quantitative_dataset import (
            QuantitativeMapDataset,
            build_quantitative_index,
        )

        quant_cfg = getattr(config, "quantitative", None)
        if quant_cfg is None or not quant_cfg.enabled:
            raise ValueError(
                "dataset_type='quantitative' requires data.quantitative.enabled=true "
                "and a non-empty data.quantitative.target_maps."
            )

        target_maps = list(quant_cfg.target_maps)
        # Non-BIDS corpora (e.g. cluster NIST-MRF) provide a committed-generator
        # manifest via data.index_path; the loader consumes it instead of the
        # BIDS globber. BIDS trees leave index_path unset and glob data_root.
        manifest_path = config.source.index_path
        index = build_quantitative_index(
            data_root=config.source.root,
            target_maps=target_maps,
            manifest_path=manifest_path,
        )
        if not index:
            raise FileNotFoundError(
                f"Quantitative index empty under data_root={config.source.root}. "
                f"Expected BIDS-style sub-*/anat/sub-*_T1w.nii.gz layout with "
                f"sibling parameter maps named *_T1map.nii.gz, *_T2map.nii.gz, etc."
            )

        train_index, val_index = split_index(index, config.split.validation_fraction)

        logger.info(
            "[DATASET] Quantitative — maps=%s n_train=%d n_val=%d",
            target_maps,
            len(train_index),
            len(val_index),
        )

        train_ds = QuantitativeMapDataset(
            index=train_index,
            quantitative_config=quant_cfg,
            transform=train_tfm,
        )
        val_ds = QuantitativeMapDataset(
            index=val_index,
            quantitative_config=quant_cfg,
            transform=val_tfm,
        )
        return train_ds, val_ds

    @staticmethod
    def _create_cine_dataset(config, train_tfm, val_tfm):
        """Build train/val cine 4D datasets (Phase 4c).

        Activated by ``dataset_type='cine'`` AND
        ``data.temporal.enabled=true``. Splits by ``validation_split``.
        """
        from spectramr.data.datasets.cine_dataset import (
            CineMRIDataset,
            build_cine_index,
        )

        temporal_cfg = getattr(config, "temporal", None)
        if temporal_cfg is None or not temporal_cfg.enabled:
            raise ValueError("dataset_type='cine' requires data.temporal.enabled=true.")

        # Frames occupy the TorchIO channel slot, so volumes that disagree on
        # frame count cannot be stacked -- torchio's collate raises
        # "stack expects each tensor to be equal size" from inside the loader,
        # naming neither cine nor the knob. Say it here instead, while the
        # config that caused it is still in hand.
        batch_size = config.loader.batch_size
        if batch_size > 1 and temporal_cfg.total_frames is None:
            raise ValueError(
                f"dataset_type='cine' with loader.batch_size={batch_size} "
                "requires temporal.total_frames. Cine folds frames into the "
                "channel axis, so volumes with different frame counts cannot "
                "be stacked. This is a declaration requirement, not a claim "
                "your cohort is heterogeneous: the builder cannot know without "
                "opening every volume, so it asks you to assert uniformity and "
                "then enforces it (the dataset raises on the first volume that "
                "disagrees with the declared count). Set temporal.total_frames, "
                "or set loader.batch_size: 1, which serves a mixed cohort fine. "
                "Windowing every draw to frames_per_window would lift the "
                "restriction entirely -- that is temporal_sampler, which is not "
                "wired into the queue yet."
            )

        index = build_cine_index(
            data_root=config.source.root,
            glob_pattern=temporal_cfg.glob_pattern,
            target_suffix=temporal_cfg.target_suffix,
        )
        if not index:
            raise FileNotFoundError(
                f"Cine index empty under data_root={config.source.root} for "
                f"temporal.glob_pattern={temporal_cfg.glob_pattern!r}. "
                "Widen the pattern or check the root."
            )

        train_index, val_index = split_index(index, config.split.validation_fraction)

        logger.info(
            "[DATASET] Cine 4D — target_source=%s target_suffix=%s "
            "glob_pattern=%s frames_per_window=%d (declarative: sampler "
            "unwired) frame_axis=%d n_train=%d n_val=%d",
            temporal_cfg.target_source,
            temporal_cfg.target_suffix,
            temporal_cfg.glob_pattern,
            temporal_cfg.frames_per_window,
            temporal_cfg.frame_axis,
            len(train_index),
            len(val_index),
        )

        train_ds = CineMRIDataset(
            index=train_index, temporal_config=temporal_cfg, transform=train_tfm
        )
        val_ds = CineMRIDataset(index=val_index, temporal_config=temporal_cfg, transform=val_tfm)
        return train_ds, val_ds

    @staticmethod
    def _create_bart_dataset(config, train_tfm, val_tfm):
        """Build train/val BART k-space datasets (spec E2).

        Activated by ``dataset_type='bart_kspace'`` AND ``data.bart.enabled=true``.
        Reads the validated ``data.bart`` block (closing pitfall #15 — the knob is
        consumed here) and splits by ``validation_split``.
        """
        from spectramr.data.builders.manifest_index import build_index_from_manifest
        from spectramr.data.datasets.bart_dataset import (
            BartKspaceDataset,
            build_bart_index,
        )

        bart_cfg = getattr(config, "bart", None)
        if bart_cfg is None or not bart_cfg.enabled:
            raise ValueError(
                "dataset_type='bart_kspace' requires data.bart.enabled=true "
                "and a validated data.bart.bart_dim_map."
            )

        # Mechanism A: when an experiment selects the dataset via a populated
        # external manifest (data.index_path), build the index from its records[]
        # — index_path is then a read knob (pitfall #15). Otherwise glob data_root.
        if config.source.index_path:
            # F3: honour data.bart.file_pattern on the manifest branch too — a
            # mixed-acquisition manifest otherwise loads radial siblings into a
            # Cartesian arm and crashes the canonicalizer (pitfall #15).
            index = build_index_from_manifest(
                config.source.index_path, "bart", file_pattern=bart_cfg.file_pattern
            )
        else:
            index = build_bart_index(
                data_root=config.source.root, file_pattern=bart_cfg.file_pattern
            )
        if not index:
            raise FileNotFoundError(
                f"BART index empty under data_root={config.source.root} "
                f"(file_pattern={bart_cfg.file_pattern!r}). "
                "Expected BART .cfl/.hdr pairs (excluding *_traj.cfl)."
            )

        train_index, val_index = split_index(index, config.split.validation_fraction)

        logger.info(
            "[DATASET] BART k-space — sampling=%s n_roles=%d n_train=%d n_val=%d",
            bart_cfg.sampling,
            len(bart_cfg.bart_dim_map),
            len(train_index),
            len(val_index),
        )

        train_ds = BartKspaceDataset(index=train_index, bart_config=bart_cfg, transform=train_tfm)
        val_ds = BartKspaceDataset(index=val_index, bart_config=bart_cfg, transform=val_tfm)
        return train_ds, val_ds

    @staticmethod
    def _create_bids_paired_dataset(config, train_tfm, val_tfm):
        """Build train/val BIDS paired-field datasets (spec E5).

        Activated by ``dataset_type='bids_paired'`` AND
        ``data.bids_paired.enabled=true``. Reads the validated ``data.bids_paired``
        block (closing pitfall #15) and splits by ``validation_split``.
        """
        from spectramr.data.datasets.bids_paired_dataset import (
            BidsPairedDataset,
            build_bids_paired_index,
        )

        bids_cfg = getattr(config, "bids_paired", None)
        if bids_cfg is None or not bids_cfg.enabled:
            raise ValueError("dataset_type='bids_paired' requires data.bids_paired.enabled=true.")

        index = build_bids_paired_index(
            data_root=config.source.root,
            low_field_dir=bids_cfg.low_field_dir,
            high_field_dir=bids_cfg.high_field_dir,
            contrasts=tuple(bids_cfg.contrasts),
        )
        if not index:
            raise FileNotFoundError(
                f"BIDS paired index empty under data_root={config.source.root} "
                f"(low='{bids_cfg.low_field_dir}', high='{bids_cfg.high_field_dir}', "
                f"contrasts={bids_cfg.contrasts}). No (subject, contrast) pairs found."
            )

        train_index, val_index = split_index(index, config.split.validation_fraction)

        logger.info(
            "[DATASET] BIDS paired — low=%s high=%s n_pairs=%d n_train=%d n_val=%d",
            bids_cfg.low_field_dir,
            bids_cfg.high_field_dir,
            len(index),
            len(train_index),
            len(val_index),
        )

        train_ds = BidsPairedDataset(index=train_index, transform=train_tfm)
        val_ds = BidsPairedDataset(index=val_index, transform=val_tfm)
        return train_ds, val_ds

    @staticmethod
    def _create_png_paired_dataset(config, train_tfm, val_tfm):
        """Build train/val paired-PNG super-resolution datasets (brats_sr).

        Activated by ``dataset_type='png_paired'`` AND
        ``data.png_paired.enabled=true``. Reads the validated ``data.png_paired``
        block (closing pitfall #15) and splits by ``validation_split``.
        """
        from spectramr.data.datasets.png_paired_dataset import (
            PngPairedDataset,
            build_png_paired_index,
        )

        png_cfg = getattr(config, "png_paired", None)
        if png_cfg is None or not png_cfg.enabled:
            raise ValueError("dataset_type='png_paired' requires data.png_paired.enabled=true.")

        index = build_png_paired_index(
            data_root=config.source.root,
            lr_dir=png_cfg.lr_dir,
            hr_dir=png_cfg.hr_dir,
            lesion_dir=png_cfg.lesion_dir,
        )
        if not index:
            raise FileNotFoundError(
                f"PNG-paired index empty under data_root={config.source.root} "
                f"(lr='{png_cfg.lr_dir}', hr='{png_cfg.hr_dir}'). No common "
                "LR/HR filenames found."
            )

        train_index, val_index = split_index(index, config.split.validation_fraction)

        logger.info(
            "[DATASET] PNG paired SR — lr=%s hr=%s n_pairs=%d n_train=%d n_val=%d",
            png_cfg.lr_dir,
            png_cfg.hr_dir,
            len(index),
            len(train_index),
            len(val_index),
        )

        train_ds = PngPairedDataset(index=train_index, transform=train_tfm)
        val_ds = PngPairedDataset(index=val_index, transform=val_tfm)
        return train_ds, val_ds

    @staticmethod
    def _create_field_ref_dataset(config, train_tfm, val_tfm):
        """Build train/val NIfTI field-reference datasets (kasper / traveling_heads).

        Activated by ``dataset_type='field_ref'`` AND ``data.field_ref.enabled=true``.
        Reads the validated ``data.field_ref`` block (closing pitfall #15) and splits
        by ``validation_split``.
        """
        from spectramr.data.datasets.field_ref_dataset import (
            FieldRefDataset,
            build_field_ref_index,
        )

        fr_cfg = getattr(config, "field_ref", None)
        if fr_cfg is None or not fr_cfg.enabled:
            raise ValueError("dataset_type='field_ref' requires data.field_ref.enabled=true.")

        index = build_field_ref_index(
            data_root=config.source.root,
            anatomy_glob=fr_cfg.anatomy_glob,
            b0_map=fr_cfg.b0_map,
            b1_map=fr_cfg.b1_map,
        )
        train_index, val_index = split_index(index, config.split.validation_fraction)

        logger.info(
            "[DATASET] NIfTI field-ref — n_anatomy=%d b0=%s b1=%s n_train=%d n_val=%d",
            len(index),
            fr_cfg.b0_map,
            fr_cfg.b1_map,
            len(train_index),
            len(val_index),
        )

        train_ds = FieldRefDataset(index=train_index, transform=train_tfm)
        val_ds = FieldRefDataset(index=val_index, transform=val_tfm)
        return train_ds, val_ds

    @staticmethod
    def _create_fmri_dataset(config, train_tfm, val_tfm):
        """Build train/val 4-D BOLD datasets (the temporal route).

        Activated by ``dataset_type='fmri'`` AND ``data.fmri.enabled=true``.
        Reads the validated ``data.fmri`` block (pitfall #15 -- ``tr_seconds``
        and ``phase_encode_axis`` were previously constructor defaults with no
        config route at all) and splits by ``validation_split``.
        """
        from spectramr.data.datasets.fmri_dataset import (
            FMRIBoldSeriesDataset,
            build_fmri_index,
        )

        fmri_cfg = getattr(config, "fmri", None)
        if fmri_cfg is None or not fmri_cfg.enabled:
            raise ValueError("dataset_type='fmri' requires data.fmri.enabled=true.")

        index = build_fmri_index(data_root=config.source.root, glob_pattern=fmri_cfg.volume_glob)
        train_index, val_index = split_index(index, config.split.validation_fraction)

        logger.info(
            "[DATASET] 4-D BOLD — n_volumes=%d tr=%.3fs pe_axis=%d n_train=%d n_val=%d",
            len(index),
            fmri_cfg.tr_seconds,
            fmri_cfg.phase_encode_axis,
            len(train_index),
            len(val_index),
        )

        common = {
            "tr_seconds": fmri_cfg.tr_seconds,
            "phase_encode_axis": fmri_cfg.phase_encode_axis,
            # Forwarded, not defaulted: the dataset REFUSES an undeclared pairing
            # rather than cloning input into target (#739 / the fMRI backlog #2).
            "target_source": fmri_cfg.target_source,
            "target_suffix": fmri_cfg.target_suffix,
        }
        train_ds = FMRIBoldSeriesDataset(train_index, transform=train_tfm, **common)
        val_ds = FMRIBoldSeriesDataset(val_index, transform=val_tfm, **common)
        return train_ds, val_ds

    @staticmethod
    def _create_oracle_bssfp_dataset(config, train_tfm, val_tfm):
        """Build train/val oracle_bssfp datasets (phase-cycled bSSFP + real Hz B0).

        Activated by ``dataset_type='oracle_bssfp'`` AND
        ``data.oracle_bssfp.enabled=true``. Reads the validated ``data.oracle_bssfp``
        block (pitfall #15) and splits by ``validation_split``.
        """
        from spectramr.data.builders.manifest_index import build_index_from_manifest
        from spectramr.data.datasets.oracle_bssfp_dataset import (
            OracleBssfpDataset,
            build_oracle_bssfp_index,
        )

        ob_cfg = getattr(config, "oracle_bssfp", None)
        if ob_cfg is None or not ob_cfg.enabled:
            raise ValueError("dataset_type='oracle_bssfp' requires data.oracle_bssfp.enabled=true.")

        # Mechanism A — manifest records pair each stack with the shared b0_map
        # (from the sub-block); otherwise glob data_root.
        if config.source.index_path:
            index = build_index_from_manifest(
                config.source.index_path, "oracle_bssfp", oracle_b0_map=ob_cfg.b0_map
            )
        else:
            index = build_oracle_bssfp_index(
                data_root=config.source.root,
                stack_glob=ob_cfg.stack_glob,
                b0_map=ob_cfg.b0_map,
            )
        train_index, val_index = split_index(index, config.split.validation_fraction)

        logger.info(
            "[DATASET] oracle_bssfp — n_stacks=%d b0=%s n_train=%d n_val=%d",
            len(index),
            ob_cfg.b0_map,
            len(train_index),
            len(val_index),
        )

        train_ds = OracleBssfpDataset(index=train_index, transform=train_tfm)
        val_ds = OracleBssfpDataset(index=val_index, transform=val_tfm)
        return train_ds, val_ds

    @staticmethod
    def _create_ismrmrd_dataset(config, train_tfm, val_tfm):
        """Build train/val ISMRMRD measured-trajectory datasets (kasper spiral).

        Activated by ``dataset_type='ismrmrd_kspace'`` AND
        ``data.ismrmrd.enabled=true``. Reads the validated ``data.ismrmrd`` block
        (closing pitfall #15) and splits by ``validation_split``.
        """
        import fnmatch

        from spectramr.data.builders.manifest_index import build_index_from_manifest
        from spectramr.data.datasets.ismrmrd_dataset import (
            IsmrmrdKspaceDataset,
            build_ismrmrd_index,
        )

        ism_cfg = getattr(config, "ismrmrd", None)
        if ism_cfg is None or not ism_cfg.enabled:
            raise ValueError("dataset_type='ismrmrd_kspace' requires data.ismrmrd.enabled=true.")

        exclude_glob = ism_cfg.nominal_file_glob if ism_cfg.emit_paired_trajectory else None
        # Mechanism A — manifest records, with the same nominal-file exclusion the
        # glob path applies (auxiliary nominal files are resolved per measured file,
        # never reconstructed as standalone samples); otherwise glob data_root.
        if config.source.index_path:
            index = build_index_from_manifest(config.source.index_path, "ismrmrd")
            if exclude_glob:
                index = [p for p in index if not fnmatch.fnmatch(Path(p).name, exclude_glob)]
        else:
            index = build_ismrmrd_index(data_root=config.source.root, exclude_glob=exclude_glob)
        if not index:
            raise FileNotFoundError(
                f"ISMRMRD index empty under data_root={config.source.root} "
                "(expected .mrd / .h5 files)."
            )

        train_index, val_index = split_index(index, config.split.validation_fraction)

        logger.info(
            "[DATASET] ISMRMRD measured-traj — n_files=%d dcf=%s n_train=%d n_val=%d",
            len(index),
            ism_cfg.density_compensation,
            len(train_index),
            len(val_index),
        )

        train_ds = IsmrmrdKspaceDataset(
            index=train_index, ismrmrd_config=ism_cfg, transform=train_tfm
        )
        val_ds = IsmrmrdKspaceDataset(index=val_index, ismrmrd_config=ism_cfg, transform=val_tfm)
        return train_ds, val_ds

    @staticmethod
    def _create_image_folder_dataset(config, train_tfm, val_tfm):
        """_create_image_folder_dataset.

        Args:
            config (Any): Description.
            train_tfm (Any): Description.
            val_tfm (Any): Description.
        Returns:
            Any: Description.
        """
        logger.info("[DATASET] Loading from folder structure")
        root = Path(config.source.root)
        train_dir = root / "train"
        val_dir = root / "val"

        def load_folder(d: Path):
            """load_folder.

            Args:
                d (Path): Description.
            Returns:
                Any: Description.
            """
            if not d.exists():
                return []
            subjects = []
            for p in sorted(d.glob("*")):
                if p.is_file() and p.suffix.lower() in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".nii",
                    ".nii.gz",
                ]:
                    subjects.append(
                        tio.Subject(
                            input=tio.ScalarImage(p),
                            target=tio.ScalarImage(p),
                            file_id=p.stem,
                        )
                    )
            return subjects

        train_subjects = load_folder(train_dir)
        if not train_subjects and root.exists():
            train_subjects = load_folder(root)

        val_subjects = load_folder(val_dir)
        if not val_subjects and len(train_subjects) > 1:
            split = int(0.9 * len(train_subjects))
            val_subjects = train_subjects[split:]
            train_subjects = train_subjects[:split]

        train_ds = tio.SubjectsDataset(train_subjects, transform=train_tfm)
        val_ds = tio.SubjectsDataset(val_subjects, transform=val_tfm)
        return train_ds, val_ds

    @staticmethod
    def _create_npy_slice_dataset(config, train_tfm, val_tfm):
        """Create NPY slice datasets for ULF→HF translation.

        Returns ``(train_dataset, val_dataset)`` — plain datasets, NOT
        DataLoaders.  Loader creation is handled by
        :class:`ConsolidatedDatasetFactory` via :class:`DataLoaderBuilder`,
        which now uses ``tio.SubjectsLoader`` for all dataset types.
        """

        from spectramr.data.datasets.slice_dataset import SliceDataset

        logger.info("[DATASET] Loading NPY slice dataset for ULF→HF translation")
        root = Path(config.source.root)
        train_dir = root / "train" if (root / "train").exists() else root
        val_dir = root / "val" if (root / "val").exists() else None

        out_range = (-1.0, 1.0)
        norm_kwargs = config.processing.normalization_kwargs
        if norm_kwargs:
            out_range = tuple(norm_kwargs.get("out_range", [-1.0, 1.0]))

        train_ds = SliceDataset(
            data_dir=str(train_dir),
            transform=train_tfm,
            normalize=False,
            out_range=out_range,
        )

        if val_dir and val_dir.exists():
            val_ds = SliceDataset(
                data_dir=str(val_dir),
                transform=val_tfm,
                normalize=False,
                out_range=out_range,
            )
        else:
            # Split the file list deterministically instead of using
            # torch.utils.data.random_split, which returns a Subset that
            # lacks the dry_iter() method required by tio.Queue.
            import random

            all_files = list(train_ds.files)  # already sorted in SliceDataset
            rng = random.Random(42)  # fixed seed for reproducibility
            rng.shuffle(all_files)

            train_files, val_files = split_index(all_files, config.split.validation_fraction)

            # Rebuild train dataset with the reduced file list
            train_ds = SliceDataset(
                file_list=train_files,
                transform=train_tfm,
                normalize=False,
                out_range=out_range,
            )
            val_ds = SliceDataset(
                file_list=val_files,
                transform=val_tfm,
                normalize=False,
                out_range=out_range,
            )

        logger.info(f"[DATASET] NPY slice: {len(train_ds)} train, {len(val_ds)} val samples")
        return train_ds, val_ds


# Populate the registry at IMPORT, not lazily on the first ``create_datasets``
# call. A registry that fills only when the pipeline runs is empty to anything
# that merely inspects it -- an audit check, a CLI that lists the servable
# types, a test parametrised over its contents. (That last one bit immediately:
# the first cut of tests/unit/data/datasets/test_registry.py silently produced
# an EMPTY parameter set and reported a green skip, which is the
# covers-nothing-but-passes shape this repo has been bitten by before.)
# ``register_dataset`` is idempotent for the same callable, so this is safe to
# re-enter.
DatasetInstantiator._ensure_registered()
