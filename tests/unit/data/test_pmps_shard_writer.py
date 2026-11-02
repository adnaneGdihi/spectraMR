"""Unit tests for the PMPS Stage-C shard and manifest writer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spectramr.data import pmps_shard_writer as writer

pytestmark = [pytest.mark.unit]


def _arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "ulf": rng.random((1, 4, 4)),
        "hf": rng.random((1, 4, 4)),
        "theta": rng.random((3, 4, 4)),
    }


@pytest.mark.parametrize("fmt", ["npz", "h5", "nifti"])
def test_writes_under_verdict_and_shard(tmp_path: Path, fmt: str) -> None:
    if fmt == "h5" and not writer.HAVE_H5PY:
        pytest.skip("h5py absent")
    if fmt == "nifti" and not writer.HAVE_NIBABEL:
        pytest.skip("nibabel absent")
    files = writer.write_pair(tmp_path, shard=2, index=7, accepted=False, arrays=_arrays(), fmt=fmt)
    assert files and all(f.startswith("rejected/shard_002/pair_000007") for f in files)
    assert all((tmp_path / f).is_file() for f in files)


def test_npz_round_trips_the_arrays(tmp_path: Path) -> None:
    arrays = _arrays()
    (rel,) = writer.write_pair(tmp_path, shard=0, index=0, accepted=True, arrays=arrays, fmt="npz")
    loaded = np.load(tmp_path / rel)
    for key in writer.PAIR_KEYS:
        np.testing.assert_allclose(loaded[key], arrays[key].astype(np.float32))


def test_unknown_format_and_missing_key_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_format"):
        writer.write_pair(tmp_path, shard=0, index=0, accepted=True, arrays=_arrays(), fmt="png")
    arrays = _arrays()
    del arrays["theta"]
    with pytest.raises(ValueError, match="theta"):
        writer.write_pair(tmp_path, shard=0, index=0, accepted=True, arrays=arrays, fmt="npz")


def test_manifest_counts_accepted(tmp_path: Path) -> None:
    records = [{"index": 0, "accepted": True}, {"index": 1, "accepted": False}]
    path = writer.write_manifest(tmp_path, records, {"seed": 1})
    payload = json.loads(path.read_text())
    assert (payload["n_pairs"], payload["n_accepted"]) == (2, 1)
    assert payload["provenance"] == {"seed": 1}
