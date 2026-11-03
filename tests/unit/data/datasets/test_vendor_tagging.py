"""vendor_id comes from each manifest record's vendor; an unknown vendor raises."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from spectramr.data.datasets.vendor_tagging import VendorTaggedDataset, vendor_ids_by_path

_RECORDS = [
    {"primary_path": "/raw/a.h5", "vendor": "siemens"},
    {"primary_path": "/raw/b.h5", "vendor": "ge"},
]
_MAP = {"siemens": 0, "ge": 1}


def test_ids_are_keyed_by_record_and_staged_path() -> None:
    ids = vendor_ids_by_path(_RECORDS, [Path("/node/a.h5"), Path("/node/b.h5")], _MAP, 2)
    assert ids == {"/raw/a.h5": 0, "/node/a.h5": 0, "/raw/b.h5": 1, "/node/b.h5": 1}


def test_unknown_vendor_and_out_of_range_ids_raise() -> None:
    with pytest.raises(ValueError, match="philips"):
        vendor_ids_by_path(
            [{"primary_path": "/x.h5", "vendor": "philips"}], [Path("/x.h5")], _MAP, 2
        )
    with pytest.raises(ValueError, match="outside"):
        vendor_ids_by_path(_RECORDS, [Path("/a"), Path("/b")], {"siemens": 0, "ge": 2}, 2)


def test_the_wrapper_stamps_each_items_vendor() -> None:
    items = [{"source_path": "/node/a.h5"}, {"source_path": "/node/b.h5"}, {"source_path": "/z.h5"}]
    ds = VendorTaggedDataset(items, {"/node/a.h5": 0, "/node/b.h5": 1})
    assert ds[1]["vendor_id"].dtype == torch.long and int(ds[1]["vendor_id"]) == 1
    with pytest.raises(KeyError, match="no vendor"):
        ds[2]
    assert len(ds) == 3
