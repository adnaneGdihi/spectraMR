"""Per-record B0 stamps for the input and target domains of a NIfTI pair.

Paired and unpaired ULF records carry ``input_field`` and ``target_field``
(Tesla), the same keys the bidirectional swap exchanges. The stamp is opt-in via
``data.expose.field_strength_pair``; with it on, a record without its field
raises rather than emitting a guessed value.
"""

from __future__ import annotations

from typing import Any

import torch
import torchio as tio


def _field(record: dict[str, Any], key: str) -> float:
    value = record.get(key)
    if value is None:
        raise ValueError(
            f"data.expose.field_strength_pair needs record['{key}'] (Tesla); the record "
            f"{record.get('file_id', record.get('primary_path', '?'))} has none."
        )
    return float(value)


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


__all__ = ["stamp_field_strength_pair"]
