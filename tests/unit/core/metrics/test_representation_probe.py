"""The frozen-encoder linear probe reads contrast from separable embeddings and not from noise."""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.representation_probe import (
    PROBE_METRIC_KEYS,
    RepresentationProbeAccumulator,
)


def _rows(n_subjects: int, seed: int, separable: bool):
    gen = torch.Generator().manual_seed(seed)
    acc = RepresentationProbeAccumulator()
    for subject in range(n_subjects):
        for contrast in range(3):
            feats = torch.randn(6, 8, generator=gen)
            if separable:
                feats[:, contrast] += 10.0
            acc.add(feats, torch.full((6,), contrast), groups=[f"s{subject}"] * 6)
    return acc


def test_separable_embeddings_probe_above_chance() -> None:
    summary = _rows(4, 0, separable=True).summary()
    assert set(summary) == PROBE_METRIC_KEYS
    assert summary["val_probe_contrast_acc"] == pytest.approx(1.0)
    assert summary["val_probe_contrast_above_chance"] > 0.5
    assert summary["val_probe_split_by_subject"] == 1.0


def test_noise_embeddings_probe_near_chance() -> None:
    summary = _rows(4, 1, separable=False).summary()
    assert summary["val_probe_contrast_above_chance"] < 0.4


def test_a_pass_mixing_subject_ids_and_none_raises() -> None:
    acc = RepresentationProbeAccumulator()
    acc.add(torch.randn(2, 3), torch.tensor([0, 1]), groups=["a", "a"])
    with pytest.raises(ValueError, match="every batch"):
        acc.add(torch.randn(2, 3), torch.tensor([0, 1]))


def test_row_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="probe rows"):
        RepresentationProbeAccumulator().add(torch.randn(3, 4), torch.tensor([0, 1]))


def test_a_single_contrast_pass_cannot_be_probed() -> None:
    acc = RepresentationProbeAccumulator()
    acc.add(torch.randn(8, 4), torch.zeros(8), groups=["a"] * 4 + ["b"] * 4)
    with pytest.raises(ValueError, match="2 contrast labels"):
        acc.summary()
