"""Manifest Loader Builder for Data Pipelines.

Responsible for resolving paths, finding metadata, and parsing dataset manifests
or building on-the-fly indexes from raw directories.
"""

import logging
import re
from pathlib import Path
from typing import Any

from spectramr.config.data_config import DataConfig
from spectramr.data.datasets.universal_dataset import parse_fastmri_index
from spectramr.data.metadata.index_builder import IndexBuilder
from spectramr.data.metadata.path_resolver import PathResolver
from spectramr.data.split_utils import split_index

logger = logging.getLogger(__name__)


#: Manifests are gitignored and never rsynced -- the committed generator is their
#: SSOT -- so "not found" is almost always "not generated yet on this host", and
#: naming the path alone costs a cluster round trip to work that out.
_MANIFEST_GENERATOR = "scripts/data/regenerate_cluster_manifests.py"


def _missing_manifest_message(path: str) -> str:
    """Name the path AND the command that produces it."""
    stem = Path(path).stem
    lines = [
        f"Index file not found at {path}",
        "",
        "Manifests are gitignored and generated on the host that holds the data:",
        f"    python {_MANIFEST_GENERATOR} --data-base <databases> \\",
        "        --datasets m4raw_multicoil_train m4raw_multicoil_val m4raw_multicoil_test",
    ]
    match = re.search(r"_nex(\d+)$", stem)
    if match:
        lines += [
            "",
            f"This is a repetition-filtered manifest ({stem}), so it also needs the",
            f"filter that produces it -- the same command with --min-reps {match.group(1)}.",
            "It keeps only NEX groups with that many repetitions, which a leave-one-out",
            "target requires, and writes to this '_nex' name rather than the shared one.",
        ]
    lines += ["", "Run with --dry-run first to see what would be indexed."]
    return "\n".join(lines)



class ManifestLoader:
    """Builder class for loading dataset manifests and splitting train/val."""

    @staticmethod
    def _resolve_data_root(config: DataConfig) -> Path:
        """Resolve the effective common data root."""
        # Root Inference: Use dataset path if data_root is default
        if hasattr(config, "datasets") and config.datasets:
            ds_def = config.datasets[0]
            if ds_path := ds_def.path:
                return Path(PathResolver.resolve(ds_path))

        # Fallback to config data_root if specified
        if config.source.root:
            return Path(PathResolver.resolve(config.source.root))

        return Path(".")

    @staticmethod
    def _extract_sensitivity_params(
        config: DataConfig, default_suffix: str = "_coil_maps.pt"
    ) -> tuple[str | None, str]:
        """Extract sensitivity map root and suffix from config."""
        smap_root = None
        smap_suffix = default_suffix

        if hasattr(config, "extra_params") and config.extra_params:
            smap_root = config.extra_params.get("sensitivity_root")
            smap_suffix = config.extra_params.get("sensitivity_suffix", smap_suffix)

        if hasattr(config, "datasets") and config.datasets:
            ds_def = config.datasets[0]
            if hasattr(ds_def, "extra_params") and ds_def.extra_params:
                root_cand = ds_def.extra_params.get("sensitivity_root")
                if root_cand:
                    smap_root = root_cand
                    smap_suffix = ds_def.extra_params.get("sensitivity_suffix", smap_suffix)

        return smap_root, smap_suffix

    @staticmethod
    def _build_on_the_fly_index(
        config: DataConfig,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Build index by scanning directories for .h5 files."""
        logger.info(
            "[DATASET] No index_path provided - building index on-the-fly from datasets list"
        )

        if not hasattr(config, "datasets") or not config.datasets:
            raise ValueError(
                f"dataset_type='{config.dataset_type}' requires either 'index_path' "
                "or 'datasets' list with data paths in config."
            )

        train_index, val_index = [], []
        # `both` is only honoured by the caller's random-split fallback, which is
        # skipped whenever an explicit val source exists. Track the two so the
        # combination can be refused rather than silently half-applied.
        both_sources: list[str] = []
        declared_val_sources = False
        for ds_def in config.datasets:
            data_source_path = ds_def.path or ds_def.data_path
            if not data_source_path:
                logger.warning(
                    f"Dataset {getattr(ds_def, 'name', 'unknown')} has no path - skipping"
                )
                continue

            data_dir = Path(PathResolver.resolve(data_source_path))
            if not data_dir.exists():
                logger.warning(f"⚠️ Data path not found: {data_dir} - skipping")
                continue

            h5_files = sorted(data_dir.glob("**/*.h5"))
            if not h5_files:
                logger.warning(f"No .h5 files found in {data_dir}")
                continue

            logger.info(
                f"[DATASET] Found {len(h5_files)} H5 files in {data_dir} (split={ds_def.split})"
            )

            # `split` is Literal["train", "val", "both"], so an unrecognised value
            # cannot reach here -- the schema rejects it at load. What CAN reach
            # here is "both", which the old `else` swallowed into train as an
            # unnamed catch-all (#9). Name it, and raise on anything else so a
            # future member added to the Literal cannot silently mean "train".
            if ds_def.split == "val":
                declared_val_sources = True
            elif ds_def.split == "both":
                both_sources.append(getattr(ds_def, "name", "unknown"))
            elif ds_def.split != "train":
                raise ValueError(
                    f"Dataset {getattr(ds_def, 'name', 'unknown')!r} declares "
                    f"split={ds_def.split!r}, which _build_on_the_fly_index does "
                    "not route. Add a branch here in the same change that adds "
                    "the member to DatasetSourceSchema.split -- defaulting it to "
                    "train would put held-out data in the training set silently."
                )

            for h5_path in h5_files:
                record = {"primary_path": str(h5_path), "file_id": h5_path.stem}
                if ds_def.split == "val":
                    val_index.append(record)
                else:
                    # "train" and "both" alike. "both" lands in train here and is
                    # ALSO in full_index, so when no source declares an explicit
                    # val split the caller's random-split fallback draws its
                    # validation set from it -- which is what "both" means. The
                    # guard below covers the case where that fallback is skipped.
                    train_index.append(record)

        if both_sources and declared_val_sources:
            raise ValueError(
                f"Dataset source(s) {both_sources} declare split='both' while "
                "another source declares split='val'. 'both' is only honoured by "
                "the random-split fallback, and that fallback is skipped whenever "
                "an explicit val source is present -- so these sources would "
                "contribute to TRAINING ONLY while the config says otherwise, and "
                "nothing downstream would report it. Give every source an explicit "
                "'train'/'val', or drop the explicit val source and let "
                "data.split.validation_fraction divide the 'both' pool."
            )

        full_index = train_index + val_index
        if not full_index:
            raise ValueError("No H5 files found in any dataset paths.")

        logger.info(f"[DATASET] Built on-the-fly index with {len(full_index)} samples")
        return full_index, train_index, val_index

    @classmethod
    def load_fastmri_splits(
        cls, config: DataConfig
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Load and split FastMRI format datasets (H5)."""
        effective_data_root = cls._resolve_data_root(config)
        resolved_index_path = (
            PathResolver.resolve(config.source.index_path) if config.source.index_path else None
        )
        resolved_val_index_path = (
            PathResolver.resolve(config.source.validation_index_path)
            if config.source.validation_index_path
            else None
        )

        smap_root, smap_suffix = cls._extract_sensitivity_params(config)

        # 1. Load Primary Manifest
        if resolved_index_path:
            if not Path(resolved_index_path).exists():
                raise ValueError(_missing_manifest_message(resolved_index_path))

            # BUG FIX: v2 manifests have their own embedded data_root which should always be used
            # NEVER pass config.data_root to parse_fastmri_index for v2 manifests
            # The data_root field validator converts "./data" to absolute path,
            # so string comparison doesn't work. Always pass None for v2 manifests.
            full_index = parse_fastmri_index(
                resolved_index_path,
                sensitivity_root=smap_root,
                sensitivity_suffix=smap_suffix,
                data_root=None,  # Always None for v2 manifests - use embedded data_root
                contrasts=config.pairing.contrasts,
                target_contrasts=config.pairing.target_contrasts,
            )
            train_index, val_index = [], []  # Will be split below
        else:
            full_index, train_index, val_index = cls._build_on_the_fly_index(config)

        # 2. Split Logic
        if resolved_val_index_path:
            logger.info(f"[DATASET] Using explicit validation manifest: {resolved_val_index_path}")
            # BUG FIX: Same as above - always use manifest's embedded data_root for v2 manifests
            val_index = parse_fastmri_index(
                resolved_val_index_path,
                sensitivity_root=smap_root,
                sensitivity_suffix=smap_suffix,
                data_root=None,  # Always None for v2 manifests - use embedded data_root
                contrasts=config.pairing.contrasts,
                target_contrasts=config.pairing.target_contrasts,
            )
            train_index = full_index

        elif val_index:  # Pre-populated from on-the-fly split
            logger.info(f"[DATASET] Using explicit validation folders ({len(val_index)} samples)")

        elif config.split.holdout_site:
            holdout = str(config.split.holdout_site).lower()
            logger.info(f"[DATASET] Using Leave-One-Site-Out validation (holdout={holdout})")
            train_index, val_index = [], []
            for record in full_index:
                meta = record.get("metadata", {})
                sites = [
                    str(s).lower()
                    for s in [
                        meta.get("site"),
                        meta.get("institution"),
                        meta.get("institutionName"),
                        meta.get("systemVendor"),
                    ]
                    if s
                ]
                if any(holdout in s for s in sites):
                    val_index.append(record)
                else:
                    train_index.append(record)
            if not val_index:
                # Symmetric with the empty-TRAIN raise below: a typo'd holdout
                # used to warn and continue, so every "best" checkpoint was
                # selected on an empty validation set (cohort review 2026-09-02).
                raise RuntimeError(
                    f"[DATASET] LOSO holdout '{holdout}' matched no samples: the "
                    "validation split is empty, so no metric could select a "
                    "checkpoint. Check data.split.holdout_site against the "
                    "manifest's site/institution/vendor metadata."
                )

        else:
            val_split = config.split.validation_fraction
            logger.info(f"[DATASET] Using random split (validation_split={val_split})")
            train_index, val_index = split_index(full_index, val_split)

        if not train_index:
            raise RuntimeError("Train split is empty. Check data paths/manifests.")

        train_index, val_index = cls._apply_subject_cap(config, train_index, val_index)
        cls._log_split_stats(full_index, train_index, val_index)
        return train_index, val_index

    @classmethod
    def load_nifti_splits(
        cls, config: DataConfig
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Load NIFTI format datasets using IndexBuilder."""
        train_index = IndexBuilder.build_nifti_index(config, split="train")
        val_index = IndexBuilder.build_nifti_index(config, split="val")
        return cls._apply_subject_cap(config, train_index, val_index)

    @classmethod
    def load_paired_nifti_splits(
        cls, config: DataConfig
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Load paired NIfTI splits, dispatching to v4 JSON manifest when available.

        When ``config.paired_manifest_path`` is set the v4 JSON manifest is
        used (filters by split_hint, pairing_status, contrasts, and
        bidirectional_mode).  Otherwise falls back to the standard
        :meth:`load_nifti_splits` behaviour (BIDS directory crawl).

        Parameters
        ----------
        config :
            Data configuration.  Expected to expose:
            - ``paired_manifest_path`` (str | None)
            - ``allow_unpaired`` (bool)
            - ``bidirectional_mode`` (str)
            - ``contrasts`` (list[str] | None)

        Returns
        -------
        tuple[list[dict], list[dict]]
            ``(train_index, val_index)`` suitable for
            :class:`~spectramr.data.datasets.contrast_aware.ContrastAwarePairedDataset`.
        """
        paired_manifest_path = config.source.paired_manifest_path

        if paired_manifest_path:
            resolved = PathResolver.resolve(paired_manifest_path)
            logger.info(f"[MANIFEST] Using v4 paired manifest: {resolved}")
            train_index = IndexBuilder.load_paired_bids_manifest(
                manifest_path=resolved,
                split="train",
                config=config,
            )
            val_index = IndexBuilder.load_paired_bids_manifest(
                manifest_path=resolved,
                split="val",
                config=config,
            )

            # Fallback: when val is empty (e.g. all val subjects are
            # unpaired and allow_unpaired=False), carve a fraction of
            # train records for validation so the val DataLoader doesn't
            # get num_samples=0.
            #
            # Through `split_index`, not a local slice. The hand-rolled carve
            # this replaces disagreed with the SSOT on the two cases that
            # matter: it truncated with `int()` where the SSOT rounds (1 val
            # record instead of 2 on a 10-file corpus at 0.15), and on a
            # SINGLE train record it produced `max(1, 0) == 1`, handing the only
            # file to validation and leaving TRAINING EMPTY -- one of the three
            # drift behaviours split_utils' docstring names as its reason to
            # exist. The SSOT raises there instead.
            if not val_index and train_index:
                val_frac = config.split.validation_fraction
                train_index, val_index = split_index(train_index, val_frac)
                logger.info(
                    f"[MANIFEST] Val split was empty — carved {len(val_index)} "
                    f"records from train for validation "
                    f"(train={len(train_index)}, val={len(val_index)})"
                )

            train_index, val_index = cls._apply_subject_cap(config, train_index, val_index)
            cls._log_split_stats(train_index + val_index, train_index, val_index)
            return train_index, val_index

        # Fallback: standard BIDS directory crawl
        logger.info("[MANIFEST] No paired_manifest_path set — falling back to BIDS directory crawl")
        return cls.load_nifti_splits(config)

    @staticmethod
    def _apply_subject_cap(
        config: DataConfig,
        train_index: list[dict[str, Any]],
        val_index: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Truncate each split to ``config.max_{train,val}_subjects`` if set.

        The TorchIO ``SubjectsDataset`` builder is eager (every record is
        loaded + coil-processed at build time), so build cost scales with
        split size. Smoke/debug runs set these caps to collapse the build
        from minutes to seconds. ``None`` (production default) is a no-op.
        Idempotent — capping an already-capped list is harmless.
        """
        max_train = config.split.max_train_subjects
        max_val = config.split.max_val_subjects
        if max_train is not None and len(train_index) > max_train:
            logger.info(
                f"[SUBSAMPLE] Capping train split {len(train_index)} → {max_train} "
                f"(data.split.max_train_subjects)"
            )
            train_index = train_index[:max_train]
        if max_val is not None and len(val_index) > max_val:
            logger.info(
                f"[SUBSAMPLE] Capping val split {len(val_index)} → {max_val} "
                f"(data.split.max_val_subjects)"
            )
            val_index = val_index[:max_val]
        return train_index, val_index

    @staticmethod
    def _log_split_stats(full_idx: list, train_idx: list, val_idx: list):
        """_log_split_stats.

        Args:
            full_idx (list): Description.
            train_idx (list): Description.
            val_idx (list): Description.
        Returns:
            Any: Description.
        """
        logger.info(f"\n{'=' * 60}")
        logger.info("[DATA SPLIT DEBUG]")
        logger.info(f"  Total samples: {len(full_idx)}")
        logger.info(
            f"  Train samples: {len(train_idx)} ({len(train_idx) / max(1, len(full_idx)) * 100:.1f}%)"
        )
        logger.info(
            f"  Val samples:   {len(val_idx)} ({len(val_idx) / max(1, len(full_idx)) * 100:.1f}%)"
        )
        logger.info(f"{'=' * 60}\n")
