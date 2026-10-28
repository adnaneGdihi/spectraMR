"""Tests for the Pattern-C acquisition FiLM block (LM-20)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.acquisition_film import AcquisitionTableFiLM, acquisition_table

LEIDEN_MAP = {"T1w": 0, "T2w": 1, "FLAIR": 2, "ADC": 3}
ROWS = [
    {"name": "FLAIR", "TE": 120.0, "TR": 8000.0, "TI": 2200.0, "FA": 90.0},
    {"name": "T1w", "TE": 12.0, "TR": 600.0, "TI": -1.0, "FA": 70.0},
    {"name": "ADC", "TE": 90.0, "TR": 6000.0, "TI": -1.0, "FA": 90.0},
    {"name": "T2w", "TE": 80.0, "TR": 4000.0, "TI": -1.0, "FA": 90.0},
]


class TestTable:
    def test_rows_are_ordered_by_contrast_id_not_by_listing(self):
        table = acquisition_table(LEIDEN_MAP, ROWS, 4)
        assert table[0].tolist() == [12.0, 600.0, -1.0, 70.0]  # T1w
        assert table[2].tolist() == [120.0, 8000.0, 2200.0, 90.0]  # FLAIR

    def test_a_contrast_without_a_row_raises(self):
        with pytest.raises(ValueError, match=r"missing \['ADC'\]"):
            acquisition_table(LEIDEN_MAP, ROWS[:2] + ROWS[3:], 4)

    def test_a_row_not_in_the_map_raises(self):
        """The old pattern-C map had PDw, which the Leiden data does not have."""
        rows = [*ROWS, {"name": "PDw", "TE": 15.0, "TR": 2000.0, "TI": -1.0, "FA": 90.0}]
        with pytest.raises(ValueError, match=r"not in contrast_map \['PDw'\]"):
            acquisition_table(LEIDEN_MAP, rows, 4)

    def test_a_duplicate_row_raises(self):
        with pytest.raises(ValueError, match="twice"):
            acquisition_table(LEIDEN_MAP, [*ROWS, ROWS[0]], 4)

    def test_non_contiguous_ids_raise(self):
        with pytest.raises(ValueError, match=r"exactly 0\.\.3"):
            acquisition_table({"T1w": 0, "T2w": 1, "FLAIR": 2, "ADC": 5}, ROWS, 4)


def _bound(embed_dim: int = 8) -> AcquisitionTableFiLM:
    film = AcquisitionTableFiLM(4, embed_dim)
    film.bind(LEIDEN_MAP, ROWS)
    return film


class TestFiLM:
    def test_it_starts_as_the_identity(self):
        film = _bound()
        emb = torch.randn(4, 8)
        assert torch.equal(film(emb, torch.arange(4)), emb)

    def test_gradient_reaches_the_film_head(self):
        """Identity at init must still learn: the zero head gets a gradient."""
        film = _bound()
        film(torch.randn(4, 8), torch.arange(4)).pow(2).sum().backward()
        assert film.to_gamma_beta.weight.grad.abs().sum() > 0

    def test_the_output_depends_on_the_table(self):
        torch.manual_seed(0)
        film = _bound()
        torch.nn.init.normal_(film.to_gamma_beta.weight, std=0.5)
        emb = torch.randn(1, 8).expand(4, 8)
        out = film(emb, torch.arange(4))
        assert not torch.allclose(out[0], out[2]), "T1w and FLAIR rows modulate differently"
        film.table[2] = film.table[0]
        assert torch.allclose(film(emb, torch.arange(4))[2], out[0])

    def test_unbound_raises(self):
        with pytest.raises(RuntimeError, match="no acquisition table was bound"):
            AcquisitionTableFiLM(4, 8)(torch.randn(2, 8), torch.tensor([0, 1]))

    def test_a_checkpoint_carries_the_bound_table(self):
        restored = AcquisitionTableFiLM(4, 8)
        restored.load_state_dict(_bound().state_dict())
        assert torch.equal(restored.table, _bound().table)
        restored(torch.randn(2, 8), torch.tensor([0, 3]))  # does not raise
