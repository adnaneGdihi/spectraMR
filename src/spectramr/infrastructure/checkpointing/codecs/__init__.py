"""The only code that writes or reads a checkpoint file (the data-IO gate's exemption).

The codec is chosen by the file's extension, and the RunState is validated on both
sides, so a file this build cannot interpret is refused rather than half-loaded.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spectramr.infrastructure.checkpointing.codecs import (  # noqa: F401  (registers)
    safetensors_codec,
    torch_codec,
)
from spectramr.infrastructure.checkpointing.codecs._registry import codec_for
from spectramr.infrastructure.checkpointing.layout import validate_format
from spectramr.infrastructure.checkpointing.payload import RunState, validate

_SCALARS = (bool, int, float, complex, str, type(None))
_NUMERIC_KINDS = frozenset("biufc")


def _check_storable(obj: Any, where: str) -> None:
    """Refuse, by path, any value no codec can read back weights-only.

    Checked before any byte is written, so a file that is published can always be read;
    a value outside this set (an object, a structured or text array, a namedtuple) would
    otherwise land on disk and fail only at resume or predict.
    """
    if isinstance(obj, (torch.Tensor, *_SCALARS)):
        return
    if isinstance(obj, (np.ndarray, np.generic)):
        if obj.dtype.kind not in _NUMERIC_KINDS:
            raise TypeError(f"{where}: a {obj.dtype} numpy value cannot be stored in a checkpoint")
        return
    if type(obj) in (list, tuple):
        for i, value in enumerate(obj):
            _check_storable(value, f"{where}[{i}]")
        return
    if type(obj) in (dict, OrderedDict):
        for key, value in obj.items():
            _check_storable(key, f"{where}<key>")
            _check_storable(value, f"{where}.{key}")
        return
    raise TypeError(f"{where}: {type(obj).__name__} cannot be stored in a checkpoint")


def _format_of(path: Path) -> str:
    return validate_format(path.suffix.removeprefix("."))


def write(path: str | Path, state: RunState) -> None:
    """Validate ``state`` and publish it atomically at ``path`` in the extension's format."""
    path = Path(path)
    fmt = _format_of(path)
    checked = validate(state, source=str(path))
    if checked["meta"]["format"] != fmt:
        raise ValueError(
            f"{path}: meta.format is {checked['meta']['format']!r} but the file "
            f"extension says {fmt!r}"
        )
    for part, value in checked.items():
        _check_storable(value, part)
    codec_for(fmt).write(path, checked)


def read(path: str | Path) -> RunState:
    """Read and validate the RunState at ``path``."""
    path = Path(path)
    fmt = _format_of(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint {path} does not exist")
    return validate(codec_for(fmt).read(path), source=str(path))


__all__ = ["read", "write"]
