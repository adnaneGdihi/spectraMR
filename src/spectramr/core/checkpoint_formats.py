"""The checkpoint formats a run may declare, and its default: one owner for both.

``pt`` and ``pth`` are the same ``torch.save`` container; only the extension differs.
"""

from __future__ import annotations

from typing import Literal, get_args

CheckpointFormat = Literal["safetensors", "pt", "pth"]
CHECKPOINT_FORMATS: tuple[str, ...] = get_args(CheckpointFormat)
#: What a run writes when its YAML declares no ``checkpoint.format``; the schema, the
#: converter and the path drain all read this one value.
DEFAULT_CHECKPOINT_FORMAT: CheckpointFormat = "safetensors"

__all__ = ["CHECKPOINT_FORMATS", "DEFAULT_CHECKPOINT_FORMAT", "CheckpointFormat"]
