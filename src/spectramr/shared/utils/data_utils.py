"""Standardized utility for fetching a representative data batch.

Replaces hardcoded ``dummy_input = torch.randn(...)`` patterns throughout
the codebase with a validated, config-driven data sample.  This ensures
ONNX/TorchScript tracing uses real tensor shapes from the configured
data pipeline.
"""

import logging
from typing import Any

import torch

from spectramr.config.settings import TrainingSettings

logger = logging.getLogger(__name__)


def get_sample_batch(
    config: TrainingSettings,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Fetch a single-sample tensor from the configured data pipeline.

    Uses :class:`DataPipelineDirector` to build the training dataloader,
    pulls the first batch, and slices it to ``batch_size=1`` for tracing.

    Args:
        config: Immutable training configuration (frozen Pydantic model).
        device: Target device for the returned tensor.

    Returns:
        A ``[1, C, ...]`` tensor suitable for ``torch.onnx.export`` or
        ``torch.jit.trace``.

    Raises:
        RuntimeError: When the data pipeline cannot produce a valid batch.
    """
    from spectramr.infrastructure.builders.directors.data_pipeline_director import (
        DataPipelineDirector,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        director = DataPipelineDirector(config)
        # build_dataloaders returns tuple[DataLoader, DataLoader]
        train_loader, _val_loader = director.build_dataloaders(persistent_workers=False)

        # Fetch first batch
        batch = next(iter(train_loader))

        # Extract the primary input tensor
        input_tensor = _extract_input_tensor(batch)

        # Slice to a single sample for tracing (ONNX dynamic_axes handles rest)
        if input_tensor.shape[0] > 1:
            input_tensor = input_tensor[:1]

        return input_tensor.to(device)

    except Exception as e:
        logger.error("Failed to fetch dynamic sample batch: %s", e)
        raise RuntimeError(
            f"Could not fetch dynamic sample batch from configured data pipeline: {e}"
        ) from e


def _extract_input_tensor(batch: Any) -> torch.Tensor:
    """Pull the primary input tensor out of a batch.

    Handles the common batch formats used across the codebase:

    * ``dict`` with keys ``source``, ``image``, ``kspace``, or ``data``
    * ``tuple | list`` where ``batch[0]`` is the input
    * bare ``torch.Tensor``
    """
    # Dict-style batches (TorchIO subjects, custom collators)
    if isinstance(batch, dict):
        for key in ("source", "image", "kspace", "data"):
            if key in batch:
                candidate = batch[key]
                if isinstance(candidate, dict) and "data" in candidate:
                    # TorchIO ScalarImage dict: {"data": tensor, "affine": ...}
                    candidate = candidate["data"]
                if isinstance(candidate, torch.Tensor):
                    return candidate

        # Fallback: first tensor value in the dict
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 3:
                return v

        raise ValueError(
            f"No suitable input tensor found in batch dict (keys: {list(batch.keys())})"
        )

    # Tuple/list-style batches (x, y) or (x, y, mask)
    if isinstance(batch, (list, tuple)):
        if len(batch) == 0:
            raise ValueError("Empty batch tuple/list")
        candidate = batch[0]
        if isinstance(candidate, torch.Tensor):
            return candidate
        raise ValueError(f"First element of batch is {type(candidate)}, expected Tensor")

    # Bare tensor
    if isinstance(batch, torch.Tensor):
        return batch

    raise ValueError(f"Unsupported batch type: {type(batch)}")
