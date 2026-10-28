"""``validation_rows``: one row per validation batch, identifiers attached.

The rows exist for one reason, pairing the same slice across arms, so the tests
pin the identity contract (a batch's ids reach its row, a batch without them
writes empty cells rather than another batch's), the one-sync-per-pass contract
(tensor metrics never cross to the host in ``add``), and the file contract (a
later pass with a new metric widens the header instead of dropping the column).
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from spectramr.data.batch_types import TrainingBatch
from spectramr.infrastructure.training.validation_rows import (
    IDENTIFIER_KEYS,
    ValidationBatchRows,
    batch_identifiers,
    depth_slices,
    expand_depth,
    identifier_strings,
    validation_batches_csv_for,
    validation_csv_for,
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_identifiers_are_read_through_the_batch_mapping_protocol() -> None:
    batch = TrainingBatch(
        input=torch.zeros(2, 1, 4, 4),
        target=torch.zeros(2, 1, 4, 4),
        metadata={
            "subject_id": ["2022060101", "2022060102"],
            "contrast": ["T1", "T1"],
            "slice_index": torch.tensor([7, 8]),
            "repetition_group": ["2022060101_T1", "2022060102_T1"],
        },
    )
    cells = batch_identifiers(batch, 2)
    assert cells["subject_id"] == ["2022060101", "2022060102"]
    assert torch.equal(cells["slice_index"], torch.tensor([7, 8]))
    assert cells["repetition_group"] == ["2022060101_T1", "2022060102_T1"]
    assert cells["file_id"] == ""
    rows = ValidationBatchRows(Path("unused.csv"), iteration=1, epoch=1)
    rows.add(0, 2, cells, {})
    assert rows._rows[0][2]["slice_index"].device == batch.metadata["slice_index"].device


def test_a_dict_batch_and_a_scalar_field_are_read_too() -> None:
    cells = batch_identifiers({"subject_id": "s1", "slice_index": torch.tensor(3)}, 1)
    assert cells["subject_id"] == "s1" and int(cells["slice_index"]) == 3


def test_a_field_whose_count_disagrees_with_the_batch_is_written_empty() -> None:
    """A row must never carry another sample's identity."""
    cells = batch_identifiers({"subject_id": ["a", "b", "c"]}, 2)
    assert cells["subject_id"] == ""
    assert set(cells) == set(IDENTIFIER_KEYS)


def test_add_keeps_tensors_on_the_device_and_flush_syncs_once(tmp_path: Path, monkeypatch) -> None:
    rows = ValidationBatchRows(
        tmp_path / "logs" / "validation_batches.csv", iteration=2000, epoch=3
    )
    scores = [torch.tensor(1.5), torch.tensor(2.5)]
    calls: list[str] = []
    original = torch.Tensor.tolist

    def counting_tolist(self):
        calls.append("tolist")
        return original(self)

    for i, score in enumerate(scores):
        rows.add(
            i, 1, {"subject_id": f"s{i}", "slice_index": str(i)}, {"val_psnr": score, "note": "x"}
        )
    assert len(rows) == 2 and not calls
    monkeypatch.setattr(torch.Tensor, "tolist", counting_tolist)
    assert rows.flush() == 2
    assert calls == ["tolist"], "one host transfer per tensor column per pass"
    on_disk = _rows(rows.path)
    assert [r["iteration"] for r in on_disk] == ["2000", "2000"]
    assert [r["subject_id"] for r in on_disk] == ["s0", "s1"]
    assert [float(r["val_psnr"]) for r in on_disk] == [1.5, 2.5]
    assert on_disk[0]["note"] == "x" and on_disk[0]["contrast"] == ""
    assert len(rows) == 0 and rows.flush() == 0


def test_a_later_pass_appends_and_a_new_metric_widens_the_header(tmp_path: Path) -> None:
    path = tmp_path / "validation_batches.csv"
    first = ValidationBatchRows(path, iteration=1000, epoch=1)
    first.add(0, 1, {"subject_id": "a"}, {"val_psnr": 20.0})
    first.flush()
    second = ValidationBatchRows(path, iteration=2000, epoch=2)
    second.add(
        0, 1, {"subject_id": "a"}, {"val_psnr": 21.0, "val_nex_equivalent": torch.tensor(8.5)}
    )
    second.flush()
    on_disk = _rows(path)
    assert [r["iteration"] for r in on_disk] == ["1000", "2000"]
    assert on_disk[0]["val_nex_equivalent"] == "" and float(on_disk[1]["val_nex_equivalent"]) == 8.5
    third = ValidationBatchRows(path, iteration=3000, epoch=3)
    third.add(0, 1, {"subject_id": "a"}, {"val_psnr": 22.0})
    third.flush()
    assert len(_rows(path)) == 3 and _rows(path)[2]["val_nex_equivalent"] == ""


def test_identifier_strings_render_one_batch_as_the_writer_would() -> None:
    one = identifier_strings({"subject_id": ["synthetic_0"], "slice_index": torch.tensor([0])}, 1)
    assert one["subject_id"] == "synthetic_0" and one["slice_index"] == "0"
    assert one["contrast"] == ""
    two = identifier_strings({"subject_id": ["a", "b"], "slice_index": torch.tensor([7, 8])}, 2)
    assert two["subject_id"] == "a|b" and two["slice_index"] == "7|8"


def test_a_volume_batch_is_scored_one_depth_slice_at_a_time() -> None:
    """The m4raw validation loader serves ``(B, C, H, W, S)``; each slice arrives in
    the depth-1 collate's layout with its depth index as ``slice_index``."""
    batch = {
        "input": torch.arange(2 * 2 * 4 * 4 * 3, dtype=torch.float32).reshape(2, 2, 4, 4, 3),
        "target": torch.zeros(2, 2, 4, 4, 3),
        "input_alt": torch.ones(2, 2, 4, 4, 3),
        "sensitivity_complex": torch.zeros(2, 1, 4, 4, 3, dtype=torch.complex64),
        "noise_covariance": torch.eye(1).expand(2, 1, 1),
        "kspace_scale": torch.tensor(3.0),
        "subject_id": ["a", "b"],
    }
    cuts = list(depth_slices(batch))
    assert len(cuts) == 3
    for index, cut in enumerate(cuts):
        assert cut["input"].shape == (2, 2, 4, 4)
        assert torch.equal(cut["input"], batch["input"][..., index])
        assert cut["input_alt"].shape == (2, 2, 4, 4)
        assert cut["sensitivity_complex"].shape == (2, 1, 4, 4)
        assert cut["noise_covariance"].shape == (2, 1, 1)
        assert cut["kspace_scale"].ndim == 0
        assert cut["subject_id"] == ["a", "b"]
        assert torch.equal(cut["slice_index"], torch.tensor([index, index]))
    assert batch_identifiers(cuts[2], 2)["slice_index"].tolist() == [2, 2]


def test_a_slice_batch_and_a_training_batch_pass_through_depth_slices() -> None:
    flat = {"input": torch.zeros(1, 2, 4, 4), "target": torch.zeros(1, 2, 4, 4), "slice_index": torch.tensor([7])}
    assert list(depth_slices(flat)) == [flat]
    volume = TrainingBatch(
        input=torch.zeros(1, 2, 4, 4, 2),
        target=torch.zeros(1, 2, 4, 4, 2),
        metadata={"slice_index": torch.tensor([9]), "sensitivity_complex": torch.zeros(1, 1, 4, 4, 2)},
    )
    cuts = list(expand_depth([volume, flat]))
    assert len(cuts) == 3
    assert isinstance(cuts[0], TrainingBatch) and cuts[0].input.shape == (1, 2, 4, 4)
    assert cuts[0].metadata["sensitivity_complex"].shape == (1, 1, 4, 4)
    # A per-sample slice_index the loader put on the batch is kept, not overwritten.
    assert torch.equal(cuts[1].metadata["slice_index"], torch.tensor([9]))
    assert cuts[2] is flat


def test_the_path_rules_live_beside_the_writer() -> None:
    """Readers outside ``pipelines/`` resolve the rows file through this module."""
    suffixed = Path("/runs/a/logs/training_metrics_gan.csv")
    assert validation_csv_for(suffixed) == Path("/runs/a/logs/validation_metrics_gan.csv")
    assert validation_csv_for("/runs/a/logs/curve.csv") == Path(
        "/runs/a/logs/validation_metrics.csv"
    )
    assert validation_batches_csv_for(suffixed) == Path("/runs/a/logs/validation_batches.csv")


def test_a_volume_sized_metric_tensor_is_not_retained_per_batch(tmp_path: Path) -> None:
    """A map-valued metric is kept as its mean, so a pass holds scalars, not volumes."""
    rows = ValidationBatchRows(tmp_path / "rows.csv", iteration=1, epoch=0)
    volume = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
    rows.add(0, 1, {}, {"val_map": volume, "val_psnr": torch.tensor(30.0)})
    stored = rows._rows[0][3]
    assert stored["val_map"].numel() == 1
    assert torch.isclose(stored["val_map"], volume.mean())
    rows.flush()
    written = float(_rows(tmp_path / "rows.csv")[0]["val_map"])
    assert written == __import__("pytest").approx(float(volume.mean()), rel=1e-6)
