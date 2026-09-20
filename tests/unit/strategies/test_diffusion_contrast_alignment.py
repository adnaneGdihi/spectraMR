"""One contrast id per record, one row per slice — and the map between them.

A record carries ONE contrast id. A 2D network is fed slices: validation
flattens ``(B, C, H, W, D) -> (B*D, C, H, W)`` in ``pipelines/train.py`` and the
training prep does the same at its ``[5D->4D RESHAPE]``. So the generator sees
``B*D`` rows against ``B`` ids, and something has to expand them.

Three call sites needed that expansion and only two had it. The third,
``_build_generator_kwargs``, is what the t=0 pre-DC probe (#1682) calls, and on a
2-volume validation batch of 18-slice M4Raw volumes it reached ``complex_unet``
with 36 rows of ``t_emb`` against 2 of ``contrast_emb``::

    RuntimeError: The size of tensor a (36) must match the size of tensor b (2)
    at non-singleton dimension 0

The cascade rungs above it in the same ``validation_step`` were unaffected
because that site expanded inline — which is precisely why the defect survived:
the failure needed the one path that skipped it.

Alignment now belongs to ``_contrast_idx_from_batch``, the declared owner of the
extraction (non-negotiable 17), and the callers pass their row count.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
)

#: Bound as a plain function: the alignment is a staticmethod, and the tests
#: exercise it directly rather than standing up a whole strategy, whose
#: construction needs a config, a model and a device.
_resolve = DiffusionTrainingStrategy._contrast_idx_from_batch


def _batch(ids: list[int]) -> dict[str, torch.Tensor]:
    return {"contrast_idx": torch.tensor(ids, dtype=torch.long)}


def test_ids_expand_to_one_per_flattened_row() -> None:
    """The reported failure: 2 declared ids, 36 rows after an 18-slice flatten."""
    out = _resolve(_batch([0, 2]), torch.device("cpu"), flat_batch=36)
    assert out is not None
    assert out.shape[0] == 36


def test_the_expansion_is_volume_major_not_interleaved() -> None:
    """`repeat_interleave`, never `repeat` — and length alone cannot tell them apart.

    The flatten is `permute(0, 4, 1, 2, 3).reshape(b * d, ...)`, so slice `i` of
    volume `v` lands at `v * d + i`. `repeat` produces the right SHAPE with the
    wrong assignment, conditioning most slices on another volume's contrast,
    and nothing downstream would raise. This asserts the exact tensor.
    """
    out = _resolve(_batch([0, 2]), torch.device("cpu"), flat_batch=6)
    assert out.tolist() == [0, 0, 0, 2, 2, 2]
    assert out.tolist() != [0, 2, 0, 2, 0, 2], "this is what `repeat` would give"


def test_a_row_count_that_is_not_a_multiple_raises() -> None:
    """Refused, not guessed.

    Truncating (what the generators' own guards do) or leaning on broadcast both
    produce a batch that trains while conditioned on the wrong contrast — the
    silent-wrong-answer outcome non-negotiable 3 exists to prevent.
    """
    with pytest.raises(ValueError, match="not a whole multiple"):
        _resolve(_batch([0, 1]), torch.device("cpu"), flat_batch=35)


def test_no_flat_batch_leaves_the_ids_untouched() -> None:
    """The default, so adding the parameter cannot regress an existing caller."""
    out = _resolve(_batch([0, 1, 2]), torch.device("cpu"))
    assert out.tolist() == [0, 1, 2]


def test_an_unflattened_batch_is_a_no_op() -> None:
    """The training case at `batch_size: N` with no depth to fold: rows == ids."""
    out = _resolve(_batch([0, 1]), torch.device("cpu"), flat_batch=2)
    assert out.tolist() == [0, 1]


def test_a_batch_without_contrast_ids_stays_none() -> None:
    """Unconditioned arms must not acquire a contrast index from the alignment."""
    assert _resolve({"input": torch.zeros(4)}, torch.device("cpu"), flat_batch=36) is None


@pytest.mark.parametrize("ids", [[0, 2], [1, 1, 0]])
def test_every_row_carries_the_id_of_the_record_it_came_from(ids: list[int]) -> None:
    """The property the two tests above check by example, stated directly."""
    d = 4
    out = _resolve(_batch(ids), torch.device("cpu"), flat_batch=len(ids) * d)
    for row, value in enumerate(out.tolist()):
        assert value == ids[row // d], f"row {row} took the id of record {row // d}"
