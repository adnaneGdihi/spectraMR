"""The precomputed NEX-shift store: format, key, header, and the two missing-group policies (#2977)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spectramr.data.datasets.m4raw_nex_shift_store import (
    NexShiftMismatchError,
    NexShiftMissingError,
    NexShiftStore,
    read_shift_groups,
    shift_key,
    write_shift_groups,
)

_PATHS = [Path("/a/2022_T101.h5"), Path("/b/2022_T102.h5"), Path("/c/2022_T103.h5")]
_SHIFTS = [(0.0, 0.0), (0.5, -0.25), (1.0, 0.75)]


def test_the_key_is_the_ordered_file_names_not_the_directories() -> None:
    assert shift_key(_PATHS) == "2022_T101.h5|2022_T102.h5|2022_T103.h5"
    assert shift_key(reversed(list(_PATHS))) != shift_key(_PATHS)


def test_what_is_written_is_what_is_read(tmp_path) -> None:
    write_shift_groups(tmp_path, {shift_key(_PATHS): _SHIFTS})
    assert read_shift_groups(tmp_path) == {shift_key(_PATHS): _SHIFTS}


def test_an_empty_directory_reads_as_no_groups(tmp_path) -> None:
    assert read_shift_groups(tmp_path) == {}


@pytest.mark.parametrize("field", ["registration_slices", "upsample", "format"])
def test_a_store_made_with_other_settings_is_refused(tmp_path, field) -> None:
    path = write_shift_groups(tmp_path, {shift_key(_PATHS): _SHIFTS})
    payload = json.loads(path.read_text())
    payload[field] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(NexShiftMismatchError, match=field):
        read_shift_groups(tmp_path)


def test_a_declared_store_with_nothing_in_it_is_an_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match=r"nex_shifts\.json"):
        NexShiftStore.declared(tmp_path, "raise")


def test_a_group_absent_from_the_store_raises_or_defers_by_policy() -> None:
    other = [Path("x_T101.h5"), Path("x_T102.h5")]
    with pytest.raises(NexShiftMissingError, match=r"x_T101\.h5"):
        NexShiftStore({shift_key(_PATHS): _SHIFTS}, "raise").get(other)
    assert NexShiftStore({shift_key(_PATHS): _SHIFTS}, "estimate").get(other) is None


def test_an_entry_of_the_wrong_length_is_refused() -> None:
    store = NexShiftStore({shift_key(_PATHS[:2]): _SHIFTS}, "raise")
    with pytest.raises(NexShiftMismatchError, match="3 shift"):
        store.get(_PATHS[:2])


def test_an_unknown_policy_is_refused() -> None:
    with pytest.raises(ValueError, match="nex_shifts_missing"):
        NexShiftStore({}, "skip")
