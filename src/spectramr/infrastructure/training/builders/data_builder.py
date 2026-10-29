"""Data Builder

Creates training, validation, inference, and ablation data loaders via
``DataPipelineDirector``, which orchestrates the leaf builders
(DatasetBuilder → TorchIOTransformBuilder → DataLoaderBuilder).

Supports all pipeline modes (training, validation, inference, ablation, HPO)
with consistent data loading from DataConfigSchema.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

from spectramr.config.schemas.validation import validation_declared_off
from spectramr.core.topology import resolve_run_topology
from spectramr.core.worker_policy import clamp_worker_count
from spectramr.data.builders import TorchIOTransformBuilder, TorchIOTransformConfig
from spectramr.data.collation.strategies import CollateStrategyFactory
from spectramr.data.datasets.contrast_aware import (
    ContrastAwarePairedDataset,
    ContrastConfig,
)
from spectramr.data.datasets.universal_dataset import UniversalMRIDataset
from spectramr.infrastructure.builders.context import (
    BuilderContext,
    accepts_builder_context,
)

from .base import Builder

logger = logging.getLogger(__name__)


class DataBuilder(Builder):
    """Builds data loaders from configuration.

    Delegates to ``DataPipelineDirector`` which orchestrates the leaf builders.
    Returns a dictionary with keys like "train", "val", "test", "inference",
    and "ablation_*" when available.

    Supports:
    - Training/Validation: `build_train_val_loaders()`
    - Testing: `build_test_loader()`
    - Inference: `build_inference_loader(input_path)`
    - Ablation: `build_ablation_subsets(subset_sizes)`
    """

    @accepts_builder_context
    def __init__(self, ctx: BuilderContext) -> None:
        """__init__.

        Args:
            config (TrainingSettings): Description.
        """
        config = ctx.config
        self._config = config
        self._loaders: dict[str, Any] = {}
        #: The partition stamp of the train/val build, copied from its director.
        self.split_provenance: dict[str, Any] = {}

    def build_train_val_loaders(self) -> DataBuilder:
        """Create train and validation data loaders.

        Returns:
            self: Enables fluent chaining
        """
        data_config = self._config.data if hasattr(self._config, "data") else None

        if data_config is None:
            raise ValueError("TrainingSettings.data is required to build loaders")

        try:
            # One field now, so one read. This used to be two: the short
            # ``val_batch_size`` via ``effective_val_batch_size``, then the long
            # ``validation_batch_size`` as a fallback -- which the property had
            # already consulted, so the fallback could never fire.
            val_cfg = getattr(self._config, "validation", None)
            _val_bs = val_cfg.loader.batch_size if val_cfg is not None else None
            # ``validation.enabled: false`` is honoured here, the one place a val loader is
            # attached: without one the loop has no validation event to run and no
            # reachability gate to fail (#673). The val dataset is still instantiated with
            # the train one; only the loader is withheld.
            validation_off = validation_declared_off(val_cfg)

            from spectramr.infrastructure.builders.directors.data_pipeline_director import (
                DataPipelineDirector,
            )

            director = DataPipelineDirector(self._config)

            num_workers = getattr(data_config.loader, "num_workers", 4)
            val_shuffle = val_cfg.loader.shuffle if val_cfg else False

            # pin_memory is a no-op (and a warning) on CPU-only runs and
            # leaks pinned host memory if mistakenly enabled, so gate on actual
            # CUDA availability (findings booklet 2026-05-05 T-6). But also
            # HONOR an explicit ``data.pin_memory: false`` — the old
            # unconditional ``torch.cuda.is_available()`` made that knob a
            # silent no-op on CUDA boxes (pitfall #15).
            pin_memory = (
                bool(getattr(data_config.loader, "pin_memory", True)) and torch.cuda.is_available()
            )

            # Multi-domain (domain adaptation) routes to the balancer instead
            # of the single-corpus loader. `data.multi_domain` is the declared
            # switch (schemas/data.py names this method in its description);
            # until now nothing called it, so `enabled: true` built an ordinary
            # single-domain loader and the run reported success (pitfall #16).
            # The balancer is loader-shaped (`__iter__`/`__len__`) and emits the
            # domain-tagged batches `strategies/domain_adaptation.py` unpacks.
            multi_domain = getattr(data_config, "multi_domain", None)
            if multi_domain is not None and multi_domain.enabled:
                logger.info(
                    "Multi-domain enabled: building %d domain loaders (balancing=%s)",
                    len(multi_domain.domains),
                    multi_domain.balancing,
                )
                train_loader = director.build_multi_domain_dataloaders(
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
                # Validation stays single-corpus: the balancer has no val half,
                # and a domain-tagged val set would make metrics incomparable
                # across arms. Built from the parent config unchanged.
                _, val_loader = director.build_dataloaders(
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    val_shuffle=val_shuffle,
                    val_batch_size=_val_bs,
                )
            else:
                train_loader, val_loader = director.build_dataloaders(
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    val_shuffle=val_shuffle,
                    val_batch_size=_val_bs,
                )

            self.split_provenance = dict(director.split_provenance)
            if train_loader is not None:
                self._loaders["train"] = train_loader
            if val_loader is not None and validation_off:
                logger.info(
                    "validation.enabled=false: the val loader is not attached, so no "
                    "validation event will run and no checkpoint_best.pt will be written"
                )
                val_loader = None
            if val_loader is not None:
                self._loaders["val"] = val_loader
            logger.info(
                "Data loaders created: train=%s, val=%s",
                train_loader is not None,
                val_loader is not None,
            )
        except Exception as exc:
            raise ValueError(f"Failed to create data loaders: {exc}") from exc

        return self

    def build_test_loader(self) -> DataBuilder:
        """Create the held-out test loader when ``data.source.test_index_path`` is declared.

        Returns:
            self: Enables fluent chaining
        """
        from spectramr.infrastructure.builders.directors.test_split import (
            build_test_loader,
            declares_test_set,
        )

        if not declares_test_set(self._config):
            return self
        self._loaders["test"] = build_test_loader(
            self._config,
            num_workers=int(getattr(self._config.data.loader, "num_workers", 0) or 0),
            persistent_workers=False,
        )
        return self

    def build_inference_loader(
        self,
        input_path: Path,
        batch_size: int | None = None,
    ) -> DataBuilder:
        """Create inference data loader from input directory.

        Uses the same preprocessing/transforms as validation (no augmentation).

        Args:
            input_path: Path to inference input directory (NIfTI files)
            batch_size: Override batch size (uses config default if None)

        Returns:
            self: Enables fluent chaining
        """
        data_config = self._config.data if hasattr(self._config, "data") else None

        if data_config is None:
            raise ValueError("TrainingSettings.data is required for inference loader")

        # Scan for input files
        input_path = Path(input_path)
        nifti_files = sorted(input_path.glob("**/*.nii.gz")) + sorted(input_path.glob("**/*.nii"))

        if len(nifti_files) == 0:
            raise ValueError(f"No NIfTI files found in {input_path}")

        logger.info(f"Found {len(nifti_files)} files for inference in {input_path}")

        # Build index entries
        inference_index = [
            {"primary_path": str(nii_path), "file_id": nii_path.stem} for nii_path in nifti_files
        ]

        # Create transforms (validation mode = no augmentation)
        torchio_config = TorchIOTransformConfig.from_settings(self._config)
        if torchio_config.digital_twin_apply:
            raise ValueError(
                "physics.digital_twin.apply_as_transform puts a synthetic degradation in "
                "the val chain; inference serves real acquisitions, so it would degrade "
                "them. Turn the flag off for inference."
            )
        val_transforms = TorchIOTransformBuilder.build_val_transforms(torchio_config)

        # Create dataset based on dataset_type
        dataset_type = (
            data_config.dataset_type if hasattr(data_config, "dataset_type") else "universal"
        )

        if dataset_type == "contrast_aware_paired":
            input_contrast = ContrastConfig(
                name=data_config.input_contrast.name,
                normalization=data_config.input_contrast.normalization,
                percentile=data_config.input_contrast.percentile,
                out_range=data_config.input_contrast.out_range,
                clamp=data_config.input_contrast.clamp,
                keywords=data_config.input_contrast.keywords,
            )
            target_contrast = ContrastConfig(
                name=data_config.target_contrast.name,
                normalization=data_config.target_contrast.normalization,
                percentile=data_config.target_contrast.percentile,
                out_range=data_config.target_contrast.out_range,
                clamp=data_config.target_contrast.clamp,
                keywords=data_config.target_contrast.keywords,
            )

            inference_ds = ContrastAwarePairedDataset(
                index=inference_index,
                input_contrast=input_contrast,
                target_contrast=target_contrast,
                io_strategy="nifti",
                transform=val_transforms,
                verify_contrast=False,
                skip_nan_samples=True,
                allow_unpaired=True,  # inference records carry no target_path
            )
        else:
            inference_ds = UniversalMRIDataset(
                index=inference_index,
                io_strategy="nifti",
                transform=val_transforms,
                load_sensitivity=False,
            )

        # Create DataLoader. Imported inside the method, matching how
        # DataPipelineDirector reaches the same class: `infrastructure.builders`
        # pulls a wide leaf/director surface at package import, and this module
        # is itself imported from that surface.
        from spectramr.infrastructure.builders.leaf.data_builders import (
            DataLoaderBuilder,
        )

        collate_strategy = CollateStrategyFactory.create("image")
        # No hasattr guard on the legacy leaf: both are declared fields, so the
        # guard only ever answered "yes" -- until phase 9a moved them under
        # `loader`, at which point it answered "no" for every arm and silently
        # pinned training to batch_size=1 / num_workers=4.
        effective_batch_size = batch_size or data_config.loader.batch_size
        num_workers = data_config.loader.num_workers
        # The ONE loader that bypasses DataPipelineDirector, so it needs its own
        # call to the shared clamp rather than inheriting the director's. Same
        # ceiling semantics: never raised, a declared 0 passes through.
        num_workers = clamp_worker_count(
            num_workers,
            resolve_run_topology(),
            role="inference",
            reserved_cores=data_config.loader.reserved_cores,
        ).workers
        # Constructed through the leaf DataLoaderBuilder, not `DataLoader(...)`
        # here: one construction site (data.md SSOT, enforced by
        # scripts/ci/check_dataloader_construction_ssot.py), and the leaf also
        # supplies the per-worker NumPy/`random` seeding this path never had.
        #
        # The "image" collate is passed EXPLICITLY. The leaf derives one from
        # `data.collation` only when none is given, so omitting it here would
        # silently swap the pinned strategy for a config-derived one on
        # slab-mode / non-image arms -- a change to inference OUTPUT, disguised
        # as a refactor.
        #
        # prefetch_factor / persistent_workers were hardcoded 2 / True (#194).
        # `persistent_workers=True` was the worse of the two: it holds a worker
        # pool open across a single-pass walk that never repeats. It stays
        # False here even though `data.loader.persistent_workers` now defaults
        # to True for training.
        inference_loader = (
            DataLoaderBuilder(self._config, dataset=inference_ds)
            .with_batch_size(effective_batch_size)
            .with_num_workers(num_workers)
            # pin_memory only useful with CUDA; mirrors the gating in the
            # train/val loader path. See findings booklet 2026-05-05 T-6.
            .with_pin_memory(bool(torch.cuda.is_available()))
            .with_shuffle(False)
            .with_collate_fn(collate_strategy.collate)
            .with_prefetch_factor(data_config.loader.prefetch_factor)
            .with_persistent_workers(False)
            .build()
        )

        self._loaders["inference"] = inference_loader
        logger.info(
            f"Inference loader created: {len(inference_loader)} batches, "
            f"batch_size={effective_batch_size}"
        )

        return self

    def build_ablation_subsets(
        self,
        subset_fractions: list[float],
        seed: int = 42,
    ) -> DataBuilder:
        """Create stratified subset loaders for ablation studies.

        Creates multiple train loaders at different data fractions,
        useful for studying data efficiency.

        Args:
            subset_fractions: List of fractions (e.g., [0.1, 0.25, 0.5, 1.0])
            seed: Random seed for reproducible subset selection

        Returns:
            self: Enables fluent chaining
        """
        # First build full train/val loaders
        if "train" not in self._loaders:
            self.build_train_val_loaders()

        train_loader = self._loaders.get("train")
        if train_loader is None:
            raise ValueError("Cannot create ablation subsets without train loader")

        from spectramr.infrastructure.builders.leaf.data_builders import (
            DataLoaderBuilder,
        )

        dataset = train_loader.dataset
        # For patch-sampled training the loader wraps a ``tio.Queue`` whose
        # ``__len__`` is the patch-buffer capacity (``queue_length``), NOT the
        # number of training volumes. Sizing a "data fraction" against that
        # buffer — and ``Subset``-ing the queue's transient patch indices —
        # produces a meaningless ablation that does not correspond to any fixed
        # fraction of the corpus (review 2026-07-01). A correct data-efficiency
        # ablation must subset SUBJECTS and rebuild the queue over the subset;
        # until that exists, fail loud rather than silently mis-size. Non-patch
        # datasets (full / npy_slice), whose ``len`` IS the corpus, proceed.
        # (Class-name check, not ``hasattr(subjects_dataset)``: a MagicMock in a
        # test auto-vivifies any attribute, so ``hasattr`` would false-positive.)
        if type(dataset).__name__ == "Queue":
            raise NotImplementedError(
                "build_ablation_subsets: the train loader wraps a tio.Queue "
                "(patch sampling). len(queue) is the patch-buffer capacity, not "
                "the training-corpus size, so a data-fraction subset here is "
                "meaningless. Subset the underlying subjects "
                "(dataset.subjects_dataset) and rebuild the queue over the "
                "subset, or run data-efficiency ablations on a non-patch "
                "(full / npy_slice) dataset."
            )
        total_samples = len(dataset)
        random.seed(seed)
        all_indices = list(range(total_samples))
        random.shuffle(all_indices)

        for fraction in subset_fractions:
            if not 0 < fraction <= 1.0:
                logger.warning(f"Invalid fraction {fraction}, skipping")
                continue

            subset_size = int(total_samples * fraction)
            subset_indices = all_indices[:subset_size]

            # Create subset
            subset_ds = Subset(dataset, subset_indices)

            # Create loader with same settings as train, through the leaf
            # DataLoaderBuilder (data.md SSOT). Beyond removing a construction
            # site, this gives the ablation loaders the per-worker seeding the
            # hand-rolled call omitted -- an ablation compares runs, so
            # unseeded NumPy draws across workers were noise on the axis being
            # measured.
            #
            # `collate_fn` is forwarded rather than left to the leaf's
            # config-derived default: torch always populates `loader.collate_fn`
            # (default_collate when the caller passes None), so a falsy value
            # here means a stand-in rather than a real absence, and the explicit
            # default keeps that case byte-identical instead of silently
            # promoting it to a `data.collation`-derived strategy.
            subset_loader = (
                DataLoaderBuilder(self._config, dataset=subset_ds)
                .with_batch_size(train_loader.batch_size)
                .with_num_workers(train_loader.num_workers)
                .with_pin_memory(getattr(train_loader, "pin_memory", False))
                .with_shuffle(True)
                .with_collate_fn(
                    train_loader.collate_fn or torch.utils.data.dataloader.default_collate
                )
                .build()
            )

            key = f"ablation_{int(fraction * 100)}pct"
            self._loaders[key] = subset_loader
            logger.info(
                f"Ablation loader '{key}' created: {len(subset_loader)} batches, "
                f"{subset_size}/{total_samples} samples"
            )

        return self

    def validate(self) -> DataBuilder:
        """Validate that at least one loader exists."""
        if not self._loaders:
            raise ValueError(
                "No dataloaders created. Call build_train_val_loaders() or "
                "build_inference_loader() first."
            )
        return self

    def build(self) -> dict[str, Any]:
        """build.

        Returns:
            dict[str, Any]: Description.
        """
        if not self._loaders:
            raise ValueError("No dataloaders created. Call a build_* method first.")
        return dict(self._loaders)
