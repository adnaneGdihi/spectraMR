"""Tests for :class:`spectramr.data.datasets.preprocessed_dataset.PreprocessedMRIDataset`.

Regression coverage for DC-2: ``_apply_split`` used to return the *full*
sample list for any split string that was neither ``'train'`` nor ``'val'``
(the ``else`` branch). A typo such as ``split='test'`` would then silently
train on the entire dataset — a NN#3 / pitfall #9 silent fallback. The fix
raises ``ValueError`` for unknown split values.
"""

from __future__ import annotations

import pytest

from spectramr.data.datasets.preprocessed_dataset import PreprocessedMRIDataset


def _apply_split(split: str):
    """Invoke the (effectively static) ``_apply_split`` without full __init__.

    ``_apply_split`` only consumes its explicit arguments, so it is safe to
    call unbound on a bare instance created via ``__new__``.
    """
    samples = [f"s{i}" for i in range(10)]
    inst = PreprocessedMRIDataset.__new__(PreprocessedMRIDataset)
    return PreprocessedMRIDataset._apply_split(
        inst, samples, split, validation_split=0.2
    )


def test_apply_split_train_returns_train_partition() -> None:
    """Validation now comes off the END, matching every other dataset type.

    This assertion changed. It used to expect ``s2..s9`` for train and
    ``s0, s1`` for val, because ``_apply_split`` sliced validation off the
    START while ``split_utils.split_index`` — the data-layer SSOT that every
    other loader routes through — takes it off the end. The old values encoded
    that divergence: an otherwise identical arm validated on different samples
    purely because its ``dataset_type`` was ``preprocessed``.
    """
    out = _apply_split("train")
    assert out == [f"s{i}" for i in range(8)]


def test_apply_split_val_returns_val_partition() -> None:
    out = _apply_split("val")
    assert out == ["s8", "s9"]


def test_apply_split_unknown_value_raises() -> None:
    """An unknown split (e.g. 'test' or a typo) must raise, not return all."""
    with pytest.raises(ValueError, match="Unknown split value"):
        _apply_split("test")


def test_apply_split_delegates_to_the_ssot() -> None:
    """The partition must BE the SSOT's, not merely resemble it.

    Pinning the two lists above would still pass if the local slice were
    "fixed" to take from the end while keeping ``int()`` truncation or dropping
    the non-empty clamp. Comparing against ``split_index`` itself is what makes
    a re-divergence impossible.
    """
    from spectramr.data.split_utils import split_index

    samples = [f"s{i}" for i in range(10)]
    expected_train, expected_val = split_index(samples, 0.2)
    assert _apply_split("train") == expected_train
    assert _apply_split("val") == expected_val


def test_rounding_matches_the_ssot_not_truncation() -> None:
    """``int()`` vs ``round()`` HALVES the val set at fraction 0.15.

    10 samples at 0.15: truncation gives 1 validation sample, the SSOT gives 2.
    0.15 is one of the two commonest validation fractions in the corpus, so
    this was not a corner case.
    """
    from spectramr.data.split_utils import split_index

    samples = [f"s{i}" for i in range(10)]
    inst = PreprocessedMRIDataset.__new__(PreprocessedMRIDataset)
    out = PreprocessedMRIDataset._apply_split(
        inst, samples, "val", validation_split=0.15
    )
    assert len(out) == 2
    assert out == split_index(samples, 0.15)[1]


def test_a_tiny_corpus_no_longer_yields_an_empty_val_split() -> None:
    """3 samples at 0.1 truncated to ``n_val = 0`` — validation was silently
    EMPTY, and nothing said so. The SSOT clamps both splits non-empty."""
    samples = [f"s{i}" for i in range(3)]
    inst = PreprocessedMRIDataset.__new__(PreprocessedMRIDataset)
    out = PreprocessedMRIDataset._apply_split(
        inst, samples, "val", validation_split=0.1
    )
    assert out == ["s2"]


def test_single_sample_corpus_raises_instead_of_emptying_a_split() -> None:
    """One file cannot form a non-overlapping split. The old code returned an
    empty val list and trained on everything; the SSOT raises."""
    inst = PreprocessedMRIDataset.__new__(PreprocessedMRIDataset)
    with pytest.raises(ValueError, match="single file"):
        PreprocessedMRIDataset._apply_split(inst, ["s0"], "train", validation_split=0.2)


# ---------------------------------------------------------------------------
# image_to_graph — a 3-D volume must contribute EVERY slice, not the center one
# ---------------------------------------------------------------------------


def test_image_to_graph_2d_grid_node_count() -> None:
    """A 2-D (C, H, W) image maps to H*W grid nodes."""
    torch = pytest.importorskip("torch")
    from spectramr.data.datasets.preprocessed_dataset import (
        GraphRepresentation,
        image_to_graph,
    )

    g = image_to_graph(torch.zeros(1, 4, 4), graph_type=GraphRepresentation.GRID_4)
    assert g["num_nodes"] == 16
    assert g["x"].shape[0] == 16


def test_image_to_graph_volume_covers_all_slices() -> None:
    """A (C, H, W, D) volume yields D*H*W nodes — every slice, not the center."""
    torch = pytest.importorskip("torch")
    from spectramr.data.datasets.preprocessed_dataset import (
        GraphRepresentation,
        image_to_graph,
    )

    c, h, w, d = 1, 4, 4, 3
    vol = torch.arange(c * h * w * d, dtype=torch.float32).reshape(c, h, w, d)
    g = image_to_graph(vol, graph_type=GraphRepresentation.GRID_4)
    # All 3 slices become nodes, not silently collapsed to the central H*W=16.
    assert g["num_nodes"] == d * h * w
    assert g["x"].shape[0] == d * h * w


def test_image_to_graph_volume_edges_stay_within_each_slice() -> None:
    """Per-slice grids are disjoint: no edge crosses the slice boundary."""
    torch = pytest.importorskip("torch")
    from spectramr.data.datasets.preprocessed_dataset import (
        GraphRepresentation,
        image_to_graph,
    )

    h, w, d = 4, 4, 3
    vol = torch.zeros(1, h, w, d)
    g = image_to_graph(vol, graph_type=GraphRepresentation.GRID_4)
    nodes_per_slice = h * w
    src = g["edge_index"][0]
    dst = g["edge_index"][1]
    # Both endpoints of every edge fall in the same slice block.
    assert torch.equal(src // nodes_per_slice, dst // nodes_per_slice)


# ---------------------------------------------------------------------------
# __init__ must set each filter exactly once (issue #671)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attr", ["contrasts", "sessions"])
def test_init_assigns_each_filter_exactly_once(attr: str) -> None:
    """``__init__`` assigned ``contrasts``/``sessions`` twice, 45 lines apart.

    Both copies were byte-identical, so nothing misbehaved and no behavioural
    test could have caught it. The hazard is the next edit: normalising the
    first copy (a locale-aware ``upper()``, a dedupe, a validation raise) would
    be silently undone by the second, and the symptom would surface far from
    the change. Asserting the *count* is the only thing that pins it, so this
    reads the source rather than the object.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(PreprocessedMRIDataset.__init__))
    )
    writes = [
        node
        for node in ast.walk(tree)
        for target in getattr(node, "targets", [])
        if isinstance(node, ast.Assign)
        and isinstance(target, ast.Attribute)
        and target.attr == attr
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ]
    assert len(writes) == 1, (
        f"self.{attr} is assigned {len(writes)}x in __init__; a second write "
        "silently discards whatever the first one computed"
    )


def test_apply_split_keeps_each_subject_on_one_side() -> None:
    """Two volumes per subject: a by-sample split put subject 2 on both sides."""
    from pathlib import Path

    from spectramr.data.datasets.preprocessed_dataset import PreprocessedSample

    samples = [
        PreprocessedSample(f"subj{i // 2}", Path(f"in{i}.npy"), Path(f"tg{i}.npy"))
        for i in range(10)
    ]
    inst = PreprocessedMRIDataset.__new__(PreprocessedMRIDataset)
    train = PreprocessedMRIDataset._apply_split(inst, samples, "train", validation_split=0.3)
    val = PreprocessedMRIDataset._apply_split(inst, samples, "val", validation_split=0.3)
    assert {s.subject_id for s in train}.isdisjoint({s.subject_id for s in val})
    assert len(train) + len(val) == 10
