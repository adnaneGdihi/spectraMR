"""Per-record B0 stamps for the input and target domains of a NIfTI pair.

Paired and unpaired ULF records carry ``input_field`` and ``target_field``
(Tesla), the same keys the bidirectional swap exchanges, or the manifest's
``metadata.field_strength_input`` / ``field_strength_target`` labels (``"64mT"``). The stamp is opt-in via
``data.expose.field_strength_pair``; with it on, a record without its field
raises rather than emitting a guessed value.
"""

from __future__ import annotations

import re
from typing import Any

import torch
import torchio as tio

#: The ulf_paired manifest writer (scripts/preprocessing/preprocess_ulf_paired.py) states
#: each side's field as a label under ``metadata``; the key it uses for each record field.
_METADATA_KEYS = {"input_field": "field_strength_input", "target_field": "field_strength_target"}
_LABEL = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(mT|T)\s*$")


def field_label_to_tesla(label: str) -> float:
    """``"64mT"`` -> 0.064, ``"3T"`` -> 3.0; anything else raises."""
    match = _LABEL.match(str(label))
    if match is None:
        raise ValueError(f"field label {label!r} is not '<number>mT' or '<number>T'")
    value = float(match.group(1))
    return value / 1000.0 if match.group(2) == "mT" else value


def _field(record: dict[str, Any], key: str) -> float:
    value = record.get(key)
    if value is not None:
        return float(value)
    label = (record.get("metadata") or {}).get(_METADATA_KEYS[key])
    if label is None:
        raise ValueError(
            f"data.expose.field_strength_pair needs record['{key}'] (Tesla) or "
            f"record['metadata']['{_METADATA_KEYS[key]}']; the record "
            f"{record.get('file_id', record.get('primary_path', '?'))} has neither."
        )
    return field_label_to_tesla(label)


def stamp_field_strength_pair(
    subject: tio.Subject,
    input_record: dict[str, Any],
    target_record: dict[str, Any],
    *,
    swap: bool = False,
) -> None:
    """Stamp ``field_strength_source_t`` (input domain) and ``field_strength_t`` (target).

    ``input_record`` supplies the input-domain field and ``target_record`` the target's.
    ``swap`` exchanges them when the subject was built with ``bidirectional_mode`` swap.
    """
    source_t = _field(input_record, "input_field")
    target_t = _field(target_record, "target_field")
    if swap:
        source_t, target_t = target_t, source_t
    subject["field_strength_source_t"] = torch.tensor([source_t], dtype=torch.float32)
    subject["field_strength_t"] = torch.tensor([target_t], dtype=torch.float32)


__all__ = ["field_label_to_tesla", "stamp_field_strength_pair"]
