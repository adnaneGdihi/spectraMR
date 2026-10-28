"""``safetensors``: every tensor stored by name, everything else as one JSON header.

safetensors holds only tensors plus string metadata, yet a resume needs optimizer
hyperparameters, int-keyed optimizer state, RNG tuples and early-stopping records. The
header below encodes those exactly (tuples stay tuples, int keys stay ints) and reads
back without pickle, so the format is safe to share.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from spectramr.infrastructure.checkpointing.codecs._registry import register_codec
from spectramr.infrastructure.checkpointing.payload import (
    CONVERTER,
    SchemaVersionError,
    refuse_without_meta,
)
from spectramr.shared.utils.safe_io import atomic_publish

STATE_KEY = "spectramr.state"

_TENSOR, _NDARRAY, _TUPLE, _PAIRS = "__tensor__", "__ndarray__", "__tuple__", "__pairs__"
_SENTINELS = frozenset({_TENSOR, _NDARRAY, _TUPLE, _PAIRS})
_SCALARS = (bool, int, float, str, type(None))
#: Tensor dtypes safetensors has no encoding for; ``pt`` stores them.
_UNSUPPORTED_DTYPES = frozenset({torch.complex32, torch.complex128})


class _Encoder:
    """Replace tensors and arrays by references; collapse identical views (ties)."""

    def __init__(self) -> None:
        self.tensors: dict[str, torch.Tensor] = {}
        self._by_view: dict[tuple[Any, ...], str] = {}

    def encode(self, obj: Any, where: str) -> Any:
        if isinstance(obj, torch.Tensor):
            return {_TENSOR: self._tensor(obj, where)}
        if isinstance(obj, np.ndarray):
            raw = np.ascontiguousarray(obj).reshape(-1).view(np.uint8).copy()
            name = self._tensor(torch.from_numpy(raw), where)
            return {_NDARRAY: name, "dtype": obj.dtype.str, "shape": list(obj.shape)}
        if isinstance(obj, np.generic):
            return self.encode(obj.item(), where)
        if isinstance(obj, _SCALARS):
            return obj
        if isinstance(obj, tuple):
            return {_TUPLE: [self.encode(v, f"{where}[{i}]") for i, v in enumerate(obj)]}
        if isinstance(obj, list):
            return [self.encode(v, f"{where}[{i}]") for i, v in enumerate(obj)]
        if isinstance(obj, Mapping):
            if all(isinstance(k, str) and k not in _SENTINELS for k in obj):
                return {k: self.encode(v, f"{where}.{k}") for k, v in obj.items()}
            return {
                _PAIRS: [
                    [self.encode(k, f"{where}<key>"), self.encode(v, f"{where}[{k!r}]")]
                    for k, v in obj.items()
                ]
            }
        raise TypeError(
            f"{where}: {type(obj).__name__} cannot be stored in a safetensors checkpoint; "
            "set checkpoint.format to pt"
        )

    def _tensor(self, tensor: torch.Tensor, where: str) -> str:
        if tensor.dtype in _UNSUPPORTED_DTYPES or tensor.is_quantized:
            raise TypeError(
                f"{where}: {tensor.dtype} tensors cannot be stored in a safetensors "
                "checkpoint; set checkpoint.format to pt"
            )
        tensor = tensor.detach()
        view = None
        if tensor.numel() > 0:
            view = (
                tensor.untyped_storage().data_ptr(),
                tensor.storage_offset(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.dtype,
                tensor.device,
            )
            if view in self._by_view:
                return self._by_view[view]
        name = f"{len(self.tensors)}:{where.lstrip('.')}"
        self.tensors[name] = tensor
        if view is not None:
            self._by_view[view] = name
        return name

    def finalize(self) -> dict[str, torch.Tensor]:
        """CPU, contiguous, and no two entries sharing storage (safetensors refuses it)."""
        owners: dict[int, int] = {}
        for tensor in self.tensors.values():
            if tensor.numel() > 0:
                ptr = tensor.untyped_storage().data_ptr()
                owners[ptr] = owners.get(ptr, 0) + 1
        out: dict[str, torch.Tensor] = {}
        for name, tensor in self.tensors.items():
            shared = tensor.numel() > 0 and owners[tensor.untyped_storage().data_ptr()] > 1
            out[name] = (tensor.clone() if shared else tensor).to("cpu").contiguous()
        return out


def _decode(obj: Any, tensors: Mapping[str, torch.Tensor]) -> Any:
    if isinstance(obj, list):
        return [_decode(v, tensors) for v in obj]
    if not isinstance(obj, dict):
        return obj
    if set(obj) == {_TENSOR}:
        return tensors[obj[_TENSOR]]
    if _NDARRAY in obj:
        raw = tensors[obj[_NDARRAY]].numpy().tobytes()
        return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()
    if set(obj) == {_TUPLE}:
        return tuple(_decode(v, tensors) for v in obj[_TUPLE])
    if set(obj) == {_PAIRS}:
        return {_decode(k, tensors): _decode(v, tensors) for k, v in obj[_PAIRS]}
    return {k: _decode(v, tensors) for k, v in obj.items()}


@register_codec("safetensors")
class SafetensorsCodec:
    def write(self, path: Path, state: Any) -> None:
        encoder = _Encoder()
        header = json.dumps(encoder.encode(dict(state), ""))
        tensors = encoder.finalize()
        atomic_publish(path, lambda temp: save_file(tensors, temp, metadata={STATE_KEY: header}))

    def read(self, path: Path) -> Any:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            # A safe_open handle is not a mapping; .keys() is its only iterator.
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118
        if STATE_KEY not in metadata:
            if (refusal := refuse_without_meta(str(path), tensors)) is not None:
                raise refusal
            raise SchemaVersionError(
                f"checkpoint {path} is a safetensors file without a spectramr state header "
                f"(written before schema 1). Convert the run first: python {CONVERTER} <run_dir>"
            )
        return _decode(json.loads(metadata[STATE_KEY]), tensors)
