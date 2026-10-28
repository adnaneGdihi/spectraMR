"""Dataset Factory Implementation.

This module provides the actual implementation for dataset creation,
bridging the domain-level DatasetRegistry with the data-layer datasets.
"""

import logging
from typing import Any

from torch.utils.data import Dataset

from spectramr.config.settings import TrainingSettings
from spectramr.data.split_utils import record_split_key, split_index_grouped
from spectramr.domain.entities.data.dataset_registry import DatasetRegistry

logger = logging.getLogger(__name__)


def create_dataset(name: str, config: TrainingSettings, **kwargs: Any) -> Dataset:
    """Implementation of dataset creation.

    Delegates to specialized dataset classes or the DataPipelineDirector
    logic for Phase 3 compatibility.

    TODO: [FUTURE REFACTOR] This factory is LEGACY and should be replaced by
    spectramr.infrastructure.builders.directors.data_pipeline_director.DataPipelineDirector
    for all pipelines.
    Currently maintained for backward compatibility with older DI configurations.
    """
    if name != "synthetic":
        logger.warning(
            f"[DEPRECATION WARNING] Using legacy dataset factory for '{name}'. "
            "This path is deprecated. Route data loading through DataPipelineDirector "
            "(spectramr.infrastructure.builders.directors.data_pipeline_director)."
        )

    logger.info(f"Creating dataset '{name}' with config and kwargs: {kwargs}")

    # Extract common parameters
    split = kwargs.pop("split", "train")
    transform = kwargs.pop("transform", None)
    if split not in ("train", "val"):
        # Validated here, before any manifest is opened: the manifest try/except
        # below re-raises as RuntimeError and would otherwise swallow the type.
        raise ValueError(f"Unknown split {split!r}. Choose one of ['train', 'val'].")

    # Case 1: Synthetic dataset (common for testing)
    if name == "synthetic":
        from spectramr.data.datasets.synthetic_mri_dataset import SyntheticMRIDataset

        num_samples = kwargs.pop("num_samples", 100)
        noise_level = kwargs.pop("noise_level", 0.0)

        return SyntheticMRIDataset(
            num_samples=num_samples,
            noise_level=noise_level,
            external_augmentations=transform,
            **kwargs,
        )

    # Case 2: Universal MRI Dataset (FastMRI, etc.)
    # For Phase 3, we largely use UniversalMRIDataset
    if name in ("fastmri", "kspace", "image", "nifti"):
        from spectramr.data.datasets.universal_dataset import (
            UniversalMRIDataset,
            parse_fastmri_index,
        )

        # Load index from manifest if specified in config, otherwise use provided index
        index = kwargs.pop("index", None)

        if index is None:
            # Try to load manifest from config.data.index_path

            # --- VALIDATION HIERARCHY FOR FACTORY ---
            # Priority 1: Explicit Validation Manifest (if split='val')
            if split == "val" and config.data.source.validation_index_path:
                manifest_path = config.data.source.validation_index_path
                logger.info(f"Loading explicit validation manifest from {manifest_path}")
                try:
                    index = parse_fastmri_index(
                        manifest_path=manifest_path,
                        data_root=config.data.source.root,
                        contrasts=config.data.pairing.contrasts,
                        target_contrasts=config.data.pairing.target_contrasts,
                    )
                except Exception as e:
                    logger.warning(f"Failed to load validation manifest: {e}. using empty index.")
                    index = []

            # Priority 2: Standard Index Path (for train or fallback)
            elif config.data.source.index_path:
                manifest_path = config.data.source.index_path
                data_root = config.data.source.root

                logger.info(f"Loading manifest from {manifest_path}")
                try:
                    full_index = parse_fastmri_index(
                        manifest_path=manifest_path,
                        data_root=data_root,
                        contrasts=config.data.pairing.contrasts,
                        target_contrasts=config.data.pairing.target_contrasts,
                    )

                    # Apply splitting to full_index
                    if len(full_index) > 0:
                        # Priority 3: Leave-One-Site-Out (LOSO)
                        if config.data.split.holdout_site and config.data.split.holdout_site:
                            holdout = config.data.split.holdout_site
                            train_index = []
                            val_index = []

                            for record in full_index:
                                metadata = record.get("metadata", {})
                                sites_found = [
                                    metadata.get("site"),
                                    metadata.get("institution"),
                                    metadata.get("institutionName"),
                                    metadata.get("systemVendor"),
                                ]
                                sites_found = [str(s).lower() for s in sites_found if s]
                                is_holdout = any(holdout.lower() in s for s in sites_found)

                                if is_holdout:
                                    val_index.append(record)
                                else:
                                    train_index.append(record)

                            if split == "train":
                                index = train_index
                                logger.info(
                                    f"Using LOSO training split (holdout={holdout}): {len(index)} samples"
                                )
                            elif split == "val":
                                index = val_index
                                logger.info(
                                    f"Using LOSO validation split (holdout={holdout}): {len(index)} samples"
                                )
                            else:
                                index = full_index

                        # Priority 4: Random Split
                        else:
                            # Check if we should split at all.
                            # If explicit validation_index_path is provided, we use the FULL index_path for training.
                            has_val_index = bool(config.data.source.validation_index_path)

                            if has_val_index:
                                if split == "train":
                                    index = full_index
                                    logger.info(
                                        f"Using FULL training index (split avoided as validation_index_path is provided): {len(index)} samples"
                                    )
                                else:
                                    # This case (split='val') should have been caught by Priority 1,
                                    # but just in case, we fallback to full index if it reaches here.
                                    index = full_index
                                    logger.warning(
                                        "Validation split requested on main index, but validation_index_path is provided. Using full main index for safety."
                                    )
                            else:
                                # One split owner (non-negotiable 17): this used
                                # to compute its own boundary, int((1-f)*n),
                                # which disagreed with the SSOT's round(n*f)
                                # (n=25, f=0.1: 2 vs 3 validation items).
                                val_split = config.data.split.validation_fraction
                                train_part, val_part = split_index_grouped(
                                    full_index, val_split, key=record_split_key
                                )
                                by_split = {"train": train_part, "val": val_part}
                                if split not in by_split:
                                    raise ValueError(
                                        f"Unknown split {split!r}. Choose one of "
                                        f"{sorted(by_split)}."
                                    )
                                index = by_split[split]
                                logger.info(
                                    f"Using deterministic {split} split "
                                    f"(validation_split={val_split}): {len(index)} samples"
                                )

                    else:
                        index = []

                except Exception as e:
                    # No empty-index fallback: a dataset built on [] trains on
                    # nothing and reports success (non-negotiable 3).
                    raise RuntimeError(f"Failed to load manifest for split={split!r}: {e}") from e
            else:
                # No manifest specified, use empty index
                index = []

        # Determine IO Strategy based on name if not provided
        io_strategy = kwargs.pop("io_strategy", None)
        if io_strategy is None:
            if name in ("fastmri", "kspace"):
                io_strategy = "fastmri_h5"
            else:
                io_strategy = "auto"

        # Extract k-space specific parameters from config.data (SSOT)
        # Handle cases where config.data might not have all fields (e.g., minimalist configs)
        coil_processing_mode = config.data.coils.processing_mode
        normalize_kspace = config.data.processing.enable_kspace_normalization

        # ``kspace_percentile`` is the k-space knob; ``normalization_kwargs``
        # belongs to ``normalization_type`` (IMAGE normalization) and reading it
        # here made the dataset disagree with the transform (issue #572).
        kspace_percentile = config.data.processing.kspace_percentile

        return UniversalMRIDataset(
            index=index,
            io_strategy=io_strategy,
            transform=transform,
            coil_processing_mode=coil_processing_mode,
            normalize_kspace=normalize_kspace,
            kspace_percentile=kspace_percentile,
            log_scaling=config.data.processing.enable_log_scaling,
            **kwargs,
        )

    # Fallback to a generic error if name is unknown
    raise ValueError(f"Unknown dataset type: {name}")


def initialize_dataset_registry():
    """Register the data-layer implementation with the domain-level registry."""

    # We use a wrapper to handle the 'self' argument when called as an instance method
    def create_dataset_wrapper(*args, **kwargs):
        # Determine if first arg is 'self' (DatasetRegistry or instance)
        """create_dataset_wrapper.

        Returns:
            Any: Description.
        """
        if (args and isinstance(args[0], DatasetRegistry)) or (
            args and hasattr(args[0], "register") and hasattr(args[0], "get")
        ):
            return create_dataset(*args[1:], **kwargs)
        return create_dataset(*args, **kwargs)

    DatasetRegistry.create_dataset = create_dataset_wrapper
    logger.info("DatasetRegistry.create_dataset implemented by data layer (with wrapper)")
