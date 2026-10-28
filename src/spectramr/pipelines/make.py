"""Model/Dataset Creation Pipeline - Config-Driven Factory Interface.

This pipeline provides a unified interface for creating models, optimizers,
and datasets directly from TrainingSettings configuration.

Single Source of Truth: TrainingSettings (loaded from YAML via ``from_yaml``,
or supplied in-memory by a scripting-API caller via the ``config=`` argument).
"""

import logging
from pathlib import Path
from typing import Any

import torch

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.builders.leaf.optimizer_builders import OptimizerBuilder
from spectramr.infrastructure.checkpointing.restore import weights

logger = logging.getLogger(__name__)


def _resolve_settings(
    config_path: Path | None, config: TrainingSettings | None
) -> TrainingSettings:
    """Resolve a :class:`TrainingSettings` from an in-memory object or a path.

    An explicit in-memory ``config`` wins; otherwise the YAML at
    ``config_path`` is loaded via the SSOT :meth:`TrainingSettings.from_yaml`.
    Passing neither raises (no silent default — pitfall #9): the scripting-API
    in-memory path and the YAML path are the only two sources.

    Args:
        config_path: Path to a training-config YAML, or ``None``.
        config: An already-built in-memory ``TrainingSettings``, or ``None``.

    Returns:
        The resolved frozen ``TrainingSettings``.
    """
    if config is not None:
        return config
    if config_path is None:
        raise ValueError("make_*: provide either config_path or config")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    return TrainingSettings.from_yaml(str(config_path))


def _config_path_label(config_path: Path | None) -> str:
    """Provenance label for the config source (path or ``<in-memory>``)."""
    return str(config_path) if config_path else "<in-memory>"


#: Split name -> the loader the director actually produces for it.
#:
#: The director yields exactly ``(train_loader, val_loader)``; the held-out test loader of an
#: arm that declares ``data.source.test_index_path`` is built by the training environment and
#: scored by ``pipelines/test_pass.py``, not through here. So ``test`` resolves to the
#: VALIDATION loader, and the metadata says so via ``split_resolved`` rather than letting a
#: test-set number pass as something it is not.
_SPLIT_TO_LOADER: dict[str, str] = {
    "train": "train",
    "val": "val",
    "validation": "val",
    "test": "val",
}


def _refuse_substitution_for_a_declared_test_set(config: TrainingSettings, split: str) -> None:
    """A config that declares a held-out test set must not get validation as ``test``.

    The ``test -> val`` mapping below is honest only while no test set exists:
    the metadata says ``split_resolved: val`` and nobody can mistake the number.
    Once ``data.source.test_index_path`` (or ``data.enable_test_split``) is
    declared, the same substitution would report a checkpoint-selection number
    as a held-out one -- so it raises until the pipeline builds the test loader.
    """
    if split != "test":
        return
    data = config.data
    source = getattr(data, "source", None)
    declared = bool(getattr(source, "test_index_path", None)) or bool(
        getattr(data, "enable_test_split", False)
    )
    if declared:
        raise RuntimeError(
            "This config declares a held-out test set (data.source.test_index_path / "
            "data.enable_test_split), but the data pipeline builds no test loader yet. "
            "Refusing to hand back the VALIDATION loader as 'test': that would report a "
            "checkpoint-selection number as a held-out one. Ask for split='val' "
            "explicitly, or build the test loader first."
        )


def _resolve_split(split: str) -> str:
    """Map a requested split to the director's loader key, or RAISE.

    One owner for the mapping (non-negotiable 17). Both ``make_dataset`` and
    ``make_dataloader`` used to carry their own ``if/elif`` whose ``else`` was the
    TRAIN loader, so an unrecognised split silently returned training data while
    the returned metadata echoed the bogus name back -- an artifact that looks
    authoritative while reporting the wrong split (non-negotiable 3). Two copies
    of the chain is how the same bug got written twice.
    """
    try:
        return _SPLIT_TO_LOADER[split]
    except KeyError:
        raise ValueError(
            f"Unknown split {split!r}. Choose one of {sorted(_SPLIT_TO_LOADER)}. "
            "Note 'test' currently resolves to the validation loader -- the "
            "pipeline builds no held-out test set yet (issue #665); the returned "
            "metadata records that under 'split_resolved'."
        ) from None


def make_model(
    config_path: Path | None = None,
    device: str = "cuda",
    checkpoint_path: Path | None = None,
    *,
    config: TrainingSettings | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Create a model from configuration.

    Args:
        config_path: Path to training config YAML (or omit and pass ``config``).
        device: Device to create model on ('cuda' or 'cpu').
        checkpoint_path: Optional checkpoint to load weights from.
        config: An in-memory ``TrainingSettings`` (scripting-API path); takes
            precedence over ``config_path``.

    Returns:
        Tuple of (model, metadata).
    """
    device_obj = torch.device(device)
    config = _resolve_settings(config_path, config)
    logger.info(f"Config loaded: model={config.model.model_type}")

    # Create model through the canonical builder -- the same one training uses.
    # This module advertises "Single Source of Truth: TrainingSettings", but the
    # `ModelFactory.create_model(config.model)` it called was handed a bare
    # ModelConfigSchema and so ran the branch that drops `acceleration_config`
    # and `kspace_log_scaled`, quietly building a different model than the same
    # config trains. ModelBuilder's injections are contract-gated, so nothing
    # reaches a constructor that does not declare it.
    logger.info(f"Creating model: {config.model.model_type}")
    from spectramr.infrastructure.training.builders.model_builder import ModelBuilder

    # Without a checkpoint of its own, the model is warm-started as training would be
    # (stage-1 VAE, model.checkpoint_path): scripted callers hand it straight to fit.
    model = (
        ModelBuilder(config, device_obj, warm_start=not checkpoint_path)
        .build_generator()
        .validate()
        .build()["generator"]
    )
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created: {num_params} parameters")

    if checkpoint_path:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        weights(checkpoint_path, "generator", model, policy="exact")
        logger.info("Weights loaded successfully")

    metadata = {
        "config_path": _config_path_label(config_path),
        "model_type": config.model.model_type,
        "device": device,
        "num_parameters": num_params,
        "checkpoint_loaded": checkpoint_path is not None,
    }

    return model, metadata


def make_optimizer(
    config_path: Path | None = None,
    model: torch.nn.Module | None = None,
    *,
    config: TrainingSettings | None = None,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    """Create an optimizer from configuration.

    Args:
        config_path: Path to training config YAML (or omit and pass ``config``).
        model: Model to optimize (required).
        config: An in-memory ``TrainingSettings`` (scripting-API path); takes
            precedence over ``config_path``.

    Returns:
        Tuple of (optimizer, metadata).
    """
    if model is None:
        raise ValueError("make_optimizer: 'model' is required")
    config = _resolve_settings(config_path, config)
    logger.info(f"Config loaded: optimizer={config.optimization.optimizer.type}")

    # Create optimizer
    logger.info(f"Creating optimizer: {config.optimization.optimizer.type}")
    builder = OptimizerBuilder(config, params=model.parameters())
    optimizer = builder.validate().build()
    logger.info(f"Optimizer created with lr={config.optimization.optimizer.learning_rate}")

    metadata = {
        "config_path": _config_path_label(config_path),
        "optimizer_type": config.optimization.optimizer.type,
        "learning_rate": config.optimization.optimizer.learning_rate,
    }

    return optimizer, metadata


def make_dataset(
    config_path: Path | None = None,
    split: str = "train",
    device: str = "cpu",
    *,
    config: TrainingSettings | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Create a dataset from configuration.

    Args:
        config_path: Path to training config YAML (or omit and pass ``config``).
        split: Dataset split — ``"train"``, ``"val"``/``"validation"``, or
            ``"test"``. Anything else raises (see :func:`_resolve_split`).
        device: Device for data operations. Forwarded to the director, which
            threads it to ``LazyEncodeWrapper`` so latent-diffusion lazy encoding
            runs where the caller asked. It used to be accepted and dropped.
        config: An in-memory ``TrainingSettings`` (scripting-API path); takes
            precedence over ``config_path``.

    Returns:
        Tuple of (dataset, metadata).
    """
    config = _resolve_settings(config_path, config)
    _refuse_substitution_for_a_declared_test_set(config, split)
    loader_key = _resolve_split(split)
    logger.info(f"Config loaded: dataset={config.data.dataset_type}, split={split}")

    from spectramr.infrastructure.builders.directors.data_pipeline_director import (
        DataPipelineDirector,
    )

    director = DataPipelineDirector(config)
    train_loader, val_loader = director.build_dataloaders(num_workers=0, device=device)

    loader = train_loader if loader_key == "train" else val_loader
    if loader is None:
        raise RuntimeError(
            f"Director produced no {loader_key!r} loader for split={split!r}. "
            "Refusing to substitute another split's data."
        )
    dataset = loader.dataset

    logger.info(f"Dataset created: {len(dataset)} samples")  # type: ignore[arg-type]

    metadata = {
        "config_path": _config_path_label(config_path),
        "dataset_type": config.data.dataset_type,
        "split": split,
        # What was ASKED for vs what was actually read. They differ for 'test',
        # which has no held-out loader yet (#665) — a divergence the artifact must
        # state rather than hide (cf. the debug-snapshot provenance contract).
        "split_resolved": loader_key,
        "device": device,
        "num_samples": len(dataset),  # type: ignore[arg-type]
        "batch_size": config.data.loader.batch_size,
    }

    return dataset, metadata


def make_dataloader(
    config_path: Path | None = None,
    split: str = "train",
    num_workers: int | None = None,
    *,
    device: str = "cpu",
    config: TrainingSettings | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Create a dataloader from configuration.

    Delegates to :class:`~spectramr.infrastructure.builders.directors.data_pipeline_director.DataPipelineDirector`
    instead of instantiating ``torch.utils.data.DataLoader`` here. This
    keeps the pipelines layer free of direct ``DataLoader`` construction
    (CLAUDE.md pitfall #11 — data SSOT lives in
    :mod:`spectramr.data` + the director). Phase 4 of
    ``TODO/backlog_ssot_and_layering_cleanup.md``.

    Args:
        config_path: Path to training config YAML (or omit and pass ``config``).
        split: Dataset split — ``"train"``, ``"val"``/``"validation"``, or
            ``"test"``. Anything else raises (see :func:`_resolve_split`).
        num_workers: Override number of workers (passed to the director).
        device: Device for data operations, forwarded to the director.
        config: An in-memory ``TrainingSettings`` (scripting-API path); takes
            precedence over ``config_path``.

    Returns:
        Tuple of (dataloader, metadata).

    .. note::

       There is deliberately **no** ``batch_size`` parameter. It used to exist,
       documented as "currently unused ... kept for backward-compat", and a
       caller passing ``batch_size=999`` silently got the config's value. Batch
       size is not this builder's to own, and an override here would not be a
       harmless convenience:

       * ``config.data.loader.batch_size`` is the SSOT, and the director exposes
         only a **val-side** override (``val_batch_size``) — there is no
         train-side target to bind to at all;
       * ``strided_validation_subset`` derives the validation **stride** from the
         val batch size, so changing it changes *which records* the val set
         contains, not merely how they are grouped;
       * ``dataset_type='cine'`` raises unless ``loader.batch_size == 1``, a guard
         an override would route around.

       Set ``data.loader.batch_size`` in the config instead.
    """
    config = _resolve_settings(config_path, config)
    _refuse_substitution_for_a_declared_test_set(config, split)
    loader_key = _resolve_split(split)

    # Build via the canonical data-pipeline director (SSOT entry point).
    from spectramr.infrastructure.builders.directors.data_pipeline_director import (
        DataPipelineDirector,
    )

    director = DataPipelineDirector(config)
    if num_workers is None:
        num_workers = config.data.loader.num_workers
    train_loader, val_loader = director.build_dataloaders(
        num_workers=num_workers, device=device, persistent_workers=False
    )

    dataloader = train_loader if loader_key == "train" else val_loader
    if dataloader is None:
        raise RuntimeError(
            f"Director produced no {loader_key!r} loader for split={split!r}. "
            "Refusing to substitute another split's data."
        )

    logger.info(
        f"DataLoader resolved: {len(dataloader)} batches (num_workers={num_workers}, split={split})"
    )

    metadata = {
        "config_path": _config_path_label(config_path),
        "dataset_type": config.data.dataset_type,
        "split": split,
        "split_resolved": loader_key,
        "device": device,
        "num_batches": len(dataloader),
        "batch_size": getattr(dataloader, "batch_size", config.data.loader.batch_size),
        "num_workers": num_workers,
        # Read off the CONSTRUCTED loader, not inferred from the split name.
        # ``split == "train"`` was a guess that happened to be right for the
        # default path and wrong for any arm that turns train shuffling off.
        "shuffle": _loader_shuffles(dataloader),
    }

    return dataloader, metadata


def _loader_shuffles(dataloader: Any) -> bool | None:
    """Whether ``dataloader`` actually randomises order, or ``None`` if unknown.

    Read off the CONSTRUCTED loader, replacing a ``split == "train"`` guess.

    **The queue must be checked first.** On the TorchIO path the loader's sampler
    is a ``SequentialSampler`` *by design* -- ``tio.Queue`` requires it, because
    the Queue is what randomises (``shuffle_subjects`` / ``shuffle_patches``,
    both defaulting True for train). Asking the sampler alone therefore reports
    ``False`` for a training loader that does shuffle: a new wrong answer in
    place of the old one, and a more convincing one for being "measured".

    ``None`` is a legitimate result. An order this cannot characterise is
    reported as unknown rather than guessed (non-negotiable 3).
    """
    dataset = getattr(dataloader, "dataset", None)
    q_subjects = getattr(dataset, "shuffle_subjects", None)
    q_patches = getattr(dataset, "shuffle_patches", None)
    if isinstance(q_subjects, bool) or isinstance(q_patches, bool):
        return bool(q_subjects) or bool(q_patches)

    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None:
        from torch.utils.data import RandomSampler, SequentialSampler

        if isinstance(sampler, RandomSampler):
            return True
        if isinstance(sampler, SequentialSampler):
            return False

    shuffle_attr = getattr(dataloader, "shuffle", None)
    return bool(shuffle_attr) if isinstance(shuffle_attr, bool) else None
