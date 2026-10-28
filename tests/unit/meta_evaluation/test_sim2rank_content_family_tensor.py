"""Differential tests for the batched ``_content_family_tensor`` (#1532).

The helper feeds the Sim2Rank composite ranker, so the batched rewrite has to be
*indistinguishable* from the per-cell loop it replaced, not merely close. The
pre-#1532 implementation is kept verbatim below as the oracle and every case is
compared at the **bit** level: ``torch.equal`` accepts ``-0.0`` for ``0.0`` and
rejects the NaN-for-NaN pairs the old code legitimately produced, so it is wrong
in both directions here.

Two behaviours are load-bearing and easy to break silently:

* a missing ``(content, family)`` cell stays **NaN**, never ``0.0``;
* a **length-1** row **broadcasts** across all ``T`` timesteps.

The second is the trap. Ragged widths raise -- so a rewrite that gets them wrong
is reported loudly -- but a length-1 row is the one width that does *not* raise,
and the obvious pad-to-max build turns ``[9.0, 9.0, 9.0]`` into
``[9.0, nan, nan]`` with no exception at all. Both wrong implementations are
therefore committed here as planted violations and asserted to be *caught*
(non-negotiable 15): a harness nobody has watched fail is not a harness.
"""

from __future__ import annotations

import random
import struct

import pytest
import torch

from spectramr.core.metrics.meta_evaluation.rankers.sim2rank import (
    _content_family_tensor,
)

NAN = float("nan")


# --------------------------------------------------------------------------- #
# Oracle: the implementation as it stood before #1532, copied verbatim.
# --------------------------------------------------------------------------- #
def _content_family_tensor_ref(traj, families, contents):
    if not families or not contents:
        return torch.zeros((0, 0, 0))
    T = len(next(iter(traj.values()))) if traj else 0
    out = torch.full((len(contents), len(families), T), float("nan"), dtype=torch.float64)
    for i, c in enumerate(contents):
        for j, fam in enumerate(families):
            arr = traj.get((c, fam))
            if arr is None:
                continue
            out[i, j] = torch.tensor(arr, dtype=torch.float64)
    return out


# --------------------------------------------------------------------------- #
# Planted violations: the two rewrites that look right and are not.
# --------------------------------------------------------------------------- #
def _zeros_pad_variant(traj, families, contents):
    """Pads absent cells with 0.0 instead of NaN."""
    if not families or not contents:
        return torch.zeros((0, 0, 0))
    width = len(next(iter(traj.values()))) if traj else 0
    grid = [[list(traj.get((c, fam)) or [0.0] * width) for fam in families] for c in contents]
    return torch.tensor(grid, dtype=torch.float64)


def _pad_to_max_variant(traj, families, contents):
    """Pads every short row -- including length-1 -- rather than broadcasting."""
    if not families or not contents:
        return torch.zeros((0, 0, 0))
    width = len(next(iter(traj.values()))) if traj else 0
    grid = []
    for c in contents:
        row = []
        for fam in families:
            arr = list(traj.get((c, fam)) or [])
            row.append(arr + [NAN] * (width - len(arr)))
        grid.append(row)
    return torch.tensor(grid, dtype=torch.float64)


def _bits(t: torch.Tensor) -> bytes:
    """Byte image of the tensor -- distinguishes -0.0 from 0.0, NaN from NaN."""
    return b"".join(struct.pack("<d", float(x)) for x in t.flatten().tolist())


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return tuple(a.shape) == tuple(b.shape) and _bits(a) == _bits(b)


# --------------------------------------------------------------------------- #
# Targeted cases
# --------------------------------------------------------------------------- #
OK_CASES: dict[str, tuple] = {
    "empty families": ({("a", "f"): [1.0]}, [], ["a"]),
    "empty contents": ({("a", "f"): [1.0]}, ["f"], []),
    "empty traj (T=0)": ({}, ["f"], ["a"]),
    "full grid": ({("a", "f"): [1.0, 2.0, 3.0], ("a", "g"): [4.0, 5.0, 6.0]}, ["f", "g"], ["a"]),
    "grid with a hole": ({("a", "f"): [1.0, 2.0, 3.0]}, ["f", "g"], ["a"]),
    "length-1 row": ({("a", "f"): [1.0, 2.0, 3.0], ("a", "g"): [9.0]}, ["f", "g"], ["a"]),
    "all rows length 1 (T=1)": ({("a", "f"): [9.0], ("a", "g"): [7.0]}, ["f", "g"], ["a"]),
    "T=0 with a length-1 row": ({("z", "z"): [], ("a", "f"): [9.0]}, ["f"], ["a"]),
    "T=0 with a length-0 row": ({("z", "z"): [], ("a", "f"): []}, ["f"], ["a"]),
    "negative zero": ({("a", "f"): [-0.0, 0.0]}, ["f"], ["a"]),
    "NaN inside a present row": ({("a", "f"): [1.0, NAN]}, ["f"], ["a"]),
    "wholly absent grid": ({("z", "z"): [1.0, 2.0]}, ["f", "g"], ["a", "b"]),
}

RAGGED_CASES: dict[str, tuple] = {
    "first-long": (
        {("a", "f"): [1.0, 2.0, 3.0, 4.0, 5.0], ("a", "g"): [1.0, 2.0, 3.0]},
        ["f", "g"],
        ["a"],
    ),
    "first-short": (
        {("a", "f"): [1.0, 2.0, 3.0], ("a", "g"): [1.0, 2.0, 3.0, 4.0, 5.0]},
        ["f", "g"],
        ["a"],
    ),
    "empty row against T=3": ({("a", "f"): [1.0, 2.0, 3.0], ("a", "g"): []}, ["f", "g"], ["a"]),
}


@pytest.mark.parametrize("label", list(OK_CASES))
def test_matches_pre_1532_oracle(label: str) -> None:
    traj, families, contents = OK_CASES[label]
    assert _same(
        _content_family_tensor(traj, families, contents),
        _content_family_tensor_ref(traj, families, contents),
    ), f"{label}: differs from the pre-#1532 implementation"


def test_matches_oracle_over_randomized_shapes() -> None:
    rng = random.Random(20260828)
    for _ in range(60):
        n_c, n_f, width = rng.randint(1, 4), rng.randint(1, 4), rng.randint(0, 6)
        contents = [f"c{i}" for i in range(n_c)]
        families = [f"f{j}" for j in range(n_f)]
        traj: dict[tuple[str, str], list[float]] = {}
        for c in contents:
            for fam in families:
                if rng.random() < 0.25:
                    continue  # a hole
                traj[(c, fam)] = [
                    NAN if rng.random() < 0.15 else rng.uniform(-5, 5) for _ in range(width)
                ]
        if not traj:
            continue
        assert _same(
            _content_family_tensor(traj, families, contents),
            _content_family_tensor_ref(traj, families, contents),
        )


# --------------------------------------------------------------------------- #
# The two must-survive behaviours, pinned directly
# --------------------------------------------------------------------------- #
def test_absent_cell_is_nan_not_zero() -> None:
    out = _content_family_tensor({("a", "f"): [1.0, 2.0, 3.0]}, ["f", "g"], ["a"])
    assert torch.isnan(out[0, 1]).all()


def test_length_one_row_broadcasts_across_all_timesteps() -> None:
    """The exception to the raise -- and the one a pad-to-max build breaks."""
    out = _content_family_tensor(
        {("a", "f"): [1.0, 2.0, 3.0], ("a", "g"): [9.0]}, ["f", "g"], ["a"]
    )
    assert out[0, 1].tolist() == [9.0, 9.0, 9.0]


@pytest.mark.parametrize("label", list(RAGGED_CASES))
def test_ragged_widths_still_raise(label: str) -> None:
    traj, families, contents = RAGGED_CASES[label]
    with pytest.raises(RuntimeError):
        _content_family_tensor(traj, families, contents)
    with pytest.raises(RuntimeError):
        _content_family_tensor_ref(traj, families, contents)


# --------------------------------------------------------------------------- #
# Planted violations -- proof the harness above discriminates
# --------------------------------------------------------------------------- #
def test_harness_catches_a_zeros_pad_rewrite() -> None:
    traj, families, contents = OK_CASES["grid with a hole"]
    assert not _same(
        _zeros_pad_variant(traj, families, contents),
        _content_family_tensor_ref(traj, families, contents),
    ), "a 0.0 pad went undetected -- the harness cannot see the NaN contract"


def test_harness_catches_a_pad_to_max_rewrite() -> None:
    traj, families, contents = OK_CASES["length-1 row"]
    assert not _same(
        _pad_to_max_variant(traj, families, contents),
        _content_family_tensor_ref(traj, families, contents),
    ), "pad-to-max went undetected -- the harness cannot see the length-1 broadcast"


def test_planted_variants_are_otherwise_plausible() -> None:
    """Both plants agree with the oracle on a full grid.

    Without this, the two tests above could pass against a variant that is
    obviously broken everywhere, which would say nothing about whether the
    harness can find a *subtle* regression.
    """
    traj, families, contents = OK_CASES["full grid"]
    ref = _content_family_tensor_ref(traj, families, contents)
    assert _same(_zeros_pad_variant(traj, families, contents), ref)
    assert _same(_pad_to_max_variant(traj, families, contents), ref)
