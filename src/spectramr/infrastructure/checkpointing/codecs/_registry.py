"""Format -> codec. One codec per format, refused twice, refused when unknown."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from spectramr.infrastructure.checkpointing.layout import validate_format


class Codec(Protocol):
    def write(self, path: Path, state: Any) -> None: ...

    def read(self, path: Path) -> Any: ...


_CODECS: dict[str, Codec] = {}


def register_codec(*formats: str) -> Callable[[type], type]:
    """Register the decorated class's instance as the codec for each of ``formats``."""

    def decorate(cls: type) -> type:
        instance = cls()
        for fmt in formats:
            validate_format(fmt)
            if fmt in _CODECS:
                raise ValueError(f"checkpoint format {fmt!r} already has a codec")
            _CODECS[fmt] = instance
        return cls

    return decorate


def codec_for(fmt: str) -> Codec:
    validate_format(fmt)
    try:
        return _CODECS[fmt]
    except KeyError:  # pragma: no cover - test_every_format_has_a_codec pins this
        raise ValueError(f"checkpoint format {fmt!r} has no codec registered") from None


def registered_formats() -> frozenset[str]:
    return frozenset(_CODECS)
