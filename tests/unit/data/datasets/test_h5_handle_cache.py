"""The per-process HDF5 handle cache reuses a file only while it is the same file."""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from spectramr.data.datasets import h5_handle_cache as cache


@pytest.fixture(autouse=True)
def _empty_cache():
    cache.close_all()
    yield
    cache.close_all()


def _write(path, value: float) -> None:
    with h5py.File(path, "w") as f:
        f["kspace"] = np.full((2, 4), value, dtype=np.float32)


def test_a_second_read_reuses_the_open_handle(tmp_path):
    path = tmp_path / "a.h5"
    _write(path, 1.0)
    assert cache.open_h5(path) is cache.open_h5(path)


def test_a_replaced_file_is_reopened_not_served_stale(tmp_path):
    path = tmp_path / "a.h5"
    _write(path, 1.0)
    assert cache.open_h5(path)["kspace"][0, 0] == 1.0
    replacement = tmp_path / "b.h5"
    _write(replacement, 2.0)
    os.replace(replacement, path)
    assert cache.open_h5(path)["kspace"][0, 0] == 2.0


def test_a_forked_process_does_not_reuse_the_parents_handle(tmp_path, monkeypatch):
    path = tmp_path / "a.h5"
    _write(path, 1.0)
    parent = cache.open_h5(path)
    monkeypatch.setattr(cache.os, "getpid", lambda: -1)
    assert cache.open_h5(path) is not parent


def test_the_least_recently_used_handle_is_closed_beyond_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "MAX_OPEN_FILES", 2)
    handles = []
    for name in ("a", "b", "c"):
        _write(tmp_path / f"{name}.h5", 0.0)
        handles.append(cache.open_h5(tmp_path / f"{name}.h5"))
    assert not handles[0].id.valid
    assert handles[1].id.valid and handles[2].id.valid


def test_the_cap_follows_the_file_descriptor_limit(monkeypatch):
    monkeypatch.setattr(cache.resource, "getrlimit", lambda _: (1024, 4096))
    assert cache._open_files_cap() == 1024 // cache._FD_SHARE
    monkeypatch.setattr(cache.resource, "getrlimit", lambda _: (10**6, 10**6))
    assert cache._open_files_cap() == cache.OPEN_FILES_CEILING
    inf = cache.resource.RLIM_INFINITY
    monkeypatch.setattr(cache.resource, "getrlimit", lambda _: (inf, inf))
    assert cache._open_files_cap() == cache.OPEN_FILES_CEILING


def test_a_cached_handle_has_no_chunk_cache(tmp_path):
    path = tmp_path / "a.h5"
    _write(path, 1.0)
    nbytes = cache.open_h5(path).id.get_access_plist().get_cache()[2]
    assert nbytes == 0
