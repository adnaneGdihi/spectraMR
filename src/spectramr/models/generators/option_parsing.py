"""Parse a string-valued model option into its enum, or raise.

A model_kwargs option that names one of a fixed set must never degrade to a default when the value
is not in the set: the run would train a different network than the YAML declares and stamp the
declared value into provenance (non-negotiable 3).
"""

from __future__ import annotations

from enum import Enum

__all__ = ["parse_enum_option"]


def parse_enum_option[E: Enum](enum_cls: type[E], raw: object, key: str) -> E:
    """``enum_cls(raw.lower())``, raising ``ValueError`` that names the key and the valid set."""
    valid = sorted(member.value for member in enum_cls)
    if not isinstance(raw, str):
        raise ValueError(f"{key} must be a string in {valid}, got {raw!r}.")
    try:
        return enum_cls(raw.lower())
    except ValueError:
        raise ValueError(f"{key} '{raw}' is not valid. Valid options: {valid}") from None
