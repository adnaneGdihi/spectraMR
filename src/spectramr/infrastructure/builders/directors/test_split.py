"""The held-out test loader: the validation pipeline pointed at ``data.source.test_index_path``.

A test pass must see the data exactly as validation does (same transforms, target mode and
scoring maps) over different subjects, so the loader is the validation loader of a config
whose validation manifest is the test manifest rather than a second dataset path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from spectramr.data.metadata.paired_partition import declares_test_set_in


#: A test pass is never capped.
_TEST_LOADER = {"num_batches": None, "num_samples": None}
#: One slice per batch, so every metric row is a single slice rather than a batch mean.
TEST_BATCH_SIZE = 1


def declares_test_set(config: Any) -> bool:
    """Whether the arm names a held-out test set: a test manifest or a config-owned paired test."""
    return declares_test_set_in(getattr(config, "data", None))


def config_for_test_split(config: Any) -> Any:
    """A copy of ``config`` whose validation split IS the test manifest, uncapped.

    ``validation.loader.num_batches`` / ``num_samples`` cap a validation pass for speed; a
    reported test number covers every record, so both are cleared. The director always builds a
    training half too; it is pointed at the same small manifest so the pass does not index the
    full training corpus.

    Raises:
        ValueError: The arm declares no ``data.source.test_index_path``.
    """
    if not declares_test_set(config):
        raise ValueError(
            "No held-out test set: declare data.source.test_index_path. Refusing to build a "
            "'test' loader from the validation or training manifest."
        )
    source = config.data.source
    if not source.test_index_path:
        return _config_for_paired_test(config)
    data = config.data.model_copy(
        update={
            "source": source.model_copy(
                update={
                    "index_path": source.test_index_path,
                    "validation_index_path": source.test_index_path,
                    "test_index_path": None,
                }
            )
        }
    )
    update: dict[str, Any] = {"data": data}
    validation = getattr(config, "validation", None)
    if validation is not None:
        loader = validation.loader.model_copy(update=_TEST_LOADER)
        update["validation"] = validation.model_copy(update={"loader": loader})
    return config.model_copy(update=update)


def _config_for_paired_test(config: Any) -> Any:
    """Paired route: the test subjects, pinned, become the validation subjects of a copy.

    The ids come from the same partition that trained the arm, so the test loader cannot
    re-carve a different test set. Training is capped to one record: the director builds a
    training half too, and it must not index the corpus for a pass that never trains.
    """
    from spectramr.data.metadata.paired_partition import resolve_paired_test_ids

    data = config.data
    test_ids = resolve_paired_test_ids(data, Path(data.source.paired_manifest_path))
    if not test_ids:
        raise ValueError("The declared paired test partition resolved to no subjects.")
    split = data.split.model_copy(
        update={
            "type": "subjects",
            "val_subjects": test_ids,
            "test_subjects": None,
            "max_train_subjects": 1,
            "max_val_subjects": None,
        }
    )
    data = data.model_copy(update={"split": split, "enable_test_split": False})
    update: dict[str, Any] = {"data": data}
    validation = getattr(config, "validation", None)
    if validation is not None:
        loader = validation.loader.model_copy(update=_TEST_LOADER)
        update["validation"] = validation.model_copy(update={"loader": loader})
    return config.model_copy(update=update)


def build_test_loader(config: Any, **loader_kwargs: Any) -> DataLoader:
    """The test loader for ``config``; ``loader_kwargs`` are ``build_dataloaders``' arguments.

    The batch size is not one of them: the test loader always batches ``TEST_BATCH_SIZE``.
    """
    from spectramr.infrastructure.builders.directors.data_pipeline_director import (
        DataPipelineDirector,
    )

    _, test_loader = DataPipelineDirector(config_for_test_split(config)).build_dataloaders(
        **loader_kwargs, val_batch_size=TEST_BATCH_SIZE
    )
    return test_loader


__all__ = ["TEST_BATCH_SIZE", "build_test_loader", "config_for_test_split", "declares_test_set"]
