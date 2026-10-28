"""Tests for the tissue-dependent contrast factor (VFU-05)."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.multi_acquisition import (
    TISSUE_ORDER,
    MultiAcquisitionSimulator,
)
from spectramr.infrastructure.physics.tissue_factor import (
    N_TISSUES,
    class_agreement,
    gain_vector,
    tissue_kappa_map,
)

_GAINS = {"white_matter": 0.49, "gray_matter": 0.45, "csf": 0.87}


def test_gain_vector_follows_the_simulator_class_order() -> None:
    g = gain_vector(_GAINS)
    assert g.tolist() == pytest.approx([_GAINS[t] for t in TISSUE_ORDER])


def test_gain_vector_refuses_a_missing_class() -> None:
    with pytest.raises(ValueError, match="csf"):
        gain_vector({"white_matter": 0.5, "gray_matter": 0.5})


def test_one_hot_logits_select_the_declared_gain_exactly() -> None:
    g = gain_vector(_GAINS)
    classes = torch.tensor([[0, 1], [2, 1]]).view(1, 1, 2, 2)
    logits = torch.nn.functional.one_hot(classes[:, 0], N_TISSUES).permute(0, 3, 1, 2) * 50.0
    kappa = tissue_kappa_map(logits.float(), g)
    assert torch.allclose(kappa, g[classes].float(), atol=1e-4)
    assert float(class_agreement(logits.float(), classes)) == 1.0


def test_kappa_map_stays_inside_the_declared_range() -> None:
    g = gain_vector(_GAINS)
    kappa = tissue_kappa_map(torch.randn(2, N_TISSUES, 8, 8) * 5, g)
    assert float(kappa.min()) >= float(g.min()) - 1e-6
    assert float(kappa.max()) <= float(g.max()) + 1e-6


def test_kappa_map_refuses_the_wrong_logit_count() -> None:
    with pytest.raises(ValueError, match="logits"):
        tissue_kappa_map(torch.zeros(1, 2, 4, 4), gain_vector(_GAINS))


def test_simulator_gain_field_is_the_class_table_lookup() -> None:
    sim = MultiAcquisitionSimulator("subvoxel_sr")
    sim.set_field_transfer(_GAINS, 1.0)
    pd = torch.rand(2, 1, 16, 16)
    classes = sim.tissue_classes(pd)
    expected = gain_vector(_GAINS)[classes]
    assert torch.allclose(sim._tissue_gain_field(pd), expected)
    assert set(classes.unique().tolist()) == {0, 1, 2}
