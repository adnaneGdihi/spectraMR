"""``pt`` and ``pth``: one ``torch.save`` container, read weights-only."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spectramr.core.rng_state import RNG_STATE_SAFE_GLOBALS
from spectramr.infrastructure.checkpointing.codecs._registry import register_codec
from spectramr.infrastructure.checkpointing.payload import CONVERTER, DOCS, SchemaVersionError
from spectramr.shared.utils.safe_io import atomic_publish

#: Data-only reconstructors for every numeric numpy scalar and array the write check admits.
_NUMPY_SAFE_GLOBALS: list[Any] = [
    np._core.multiarray.scalar,
    *sorted(
        {
            type(np.dtype(code))
            for code in np.typecodes["AllInteger"] + np.typecodes["AllFloat"] + "?"
        },
        key=lambda cls: cls.__name__,
    ),
]


@register_codec("pt", "pth")
class TorchCodec:
    def write(self, path: Path, state: Any) -> None:
        atomic_publish(path, lambda temp: torch.save(dict(state), temp))

    def read(self, path: Path) -> Any:
        # NumPy values need data-only reconstructors beyond torch's allowlist; anything
        # else a weights-only load refuses was not written by this codec.
        try:
            with torch.serialization.safe_globals(RNG_STATE_SAFE_GLOBALS + _NUMPY_SAFE_GLOBALS):
                return torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as exc:
            raise SchemaVersionError(
                f"checkpoint {path} holds objects a weights-only load refuses, so it was not "
                f"written by this build's checkpoint codec. A pre-1.0 spectramr run is "
                f"converted with: python {CONVERTER} <run_dir>; external weights are out "
                f"of scope (see {DOCS})."
            ) from exc
