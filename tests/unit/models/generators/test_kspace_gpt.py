"""kspace_gpt's causal line path: row j of the output reads input rows 0..j only."""

from __future__ import annotations

import warnings

import pytest
import torch

from spectramr.models.generators.kspace_gpt import KSpaceGPT

_TINY = {
    "in_channels": 2,
    "out_channels": 2,
    "d_model": 16,
    "nhead": 2,
    "num_layers": 2,
    "dim_feedforward": 32,
    "img_size": 8,
    "patch_size": 4,
}


def _factory_build(**kwargs):
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    with warnings.catch_warnings():  # ModelFactory is the canonical (deprecated) build path
        warnings.simplefilter("ignore", DeprecationWarning)
        from spectramr.models.factories.model_factory import ModelFactory

        factory = ModelFactory()
    return factory.create_generator(model_type="kspace_gpt", **_TINY, **kwargs).eval()


def _rows_changed(model: torch.nn.Module, row: int) -> torch.Tensor:
    torch.manual_seed(0)
    x = torch.randn(1, 2, 8, 8)
    perturbed = x.clone()
    perturbed[:, :, row] += 1.0
    with torch.no_grad():
        delta = (model(perturbed) - model(x)).abs().amax(dim=(0, 1, 3))
    return delta > 0


def test_factory_builds_the_causal_path_and_no_row_sees_a_later_one() -> None:
    model = _factory_build(causal_mask=True, max_seq_length=8)
    assert isinstance(model, KSpaceGPT) and model.causal_mask
    changed = _rows_changed(model, row=3)
    assert not changed[:3].any()
    assert changed[3:].all()


def test_planted_the_bidirectional_twin_leaks_later_rows_backwards() -> None:
    # Row 7 sits in the second patch row; rows 0-3 (the first) change only through attention.
    model = _factory_build(causal_mask=False)
    assert _rows_changed(model, row=7)[:4].any()


def test_too_many_lines_raise() -> None:
    model = KSpaceGPT(**_TINY, causal_mask=True, max_seq_length=4).eval()
    with pytest.raises(ValueError, match="exceed max_seq_length"):
        model(torch.zeros(1, 2, 8, 8))


def test_a_line_length_other_than_img_size_raises() -> None:
    model = KSpaceGPT(**_TINY, causal_mask=True).eval()
    with pytest.raises(ValueError, match="img_size"):
        model(torch.zeros(1, 2, 8, 6))


def test_max_seq_length_without_the_causal_path_raises() -> None:
    with pytest.raises(ValueError, match="unread with causal_mask=False"):
        KSpaceGPT(**_TINY, max_seq_length=8)
