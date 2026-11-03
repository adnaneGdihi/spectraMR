"""Stamp a per-sample ``vendor_id`` from the manifest's ``vendor`` field.

``data.multi_contrast.vendor_map`` names each vendor / protocol and its id; every
manifest record must carry a ``vendor`` the map knows, or the build raises (a
default id would silently merge two scanners). The id reaches the batch beside
``contrast_idx`` for a vendor-conditioned generator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

__all__ = ["VendorTaggedDataset", "vendor_ids_by_path"]


def vendor_ids_by_path(
    records: Sequence[Mapping[str, Any]],
    paths: Sequence[Path],
    vendor_map: Mapping[str, int],
    n_vendors: int,
) -> dict[str, int]:
    """``{path: vendor_id}`` for the record paths and their staged copies (same order)."""
    if len(records) != len(paths):
        raise ValueError(f"{len(records)} records but {len(paths)} paths")
    bad = {k: v for k, v in vendor_map.items() if not 0 <= int(v) < n_vendors}
    if bad:
        raise ValueError(f"vendor_map ids {bad} fall outside [0, n_vendors={n_vendors})")
    out: dict[str, int] = {}
    for record, path in zip(records, paths, strict=True):
        vendor = record.get("vendor")
        if vendor not in vendor_map:
            raise ValueError(
                f"record {record.get('primary_path', record)!r} has vendor {vendor!r}, "
                f"which data.multi_contrast.vendor_map {sorted(vendor_map)} does not name"
            )
        vid = int(vendor_map[vendor])
        out[str(record["primary_path"])] = vid
        out[str(path)] = vid
    return out


class VendorTaggedDataset(Dataset):
    """Adds ``vendor_id`` to each item of ``dataset`` from its ``source_path``."""

    def __init__(self, dataset: Dataset, vendor_by_path: Mapping[str, int]) -> None:
        self.dataset, self.vendor_by_path = dataset, dict(vendor_by_path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> Any:
        item = self.dataset[i]
        path = str(item["source_path"])
        if path not in self.vendor_by_path:
            raise KeyError(f"no vendor for {path}; it is not in the split's manifest")
        item["vendor_id"] = torch.tensor(self.vendor_by_path[path], dtype=torch.long)
        return item

    def __getattr__(self, name: str) -> Any:
        if name in ("dataset", "vendor_by_path"):
            raise AttributeError(name)
        return getattr(self.dataset, name)
