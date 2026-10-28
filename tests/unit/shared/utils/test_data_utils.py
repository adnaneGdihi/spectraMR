"""Tests for ``_extract_input_tensor`` (the dispatch helper).

Targets ``spectramr.shared.utils.data_utils``. The ``get_sample_batch``
function depends on a config object and a director — exercising it
requires heavy fixtures. Instead, we focus on ``_extract_input_tensor``,
the pure-function dispatcher that handles the four documented batch
formats: dict, tuple/list, TorchIO ScalarImage, bare tensor.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.shared.utils.data_utils import _extract_input_tensor


# ---------------------------------------------------------------------------
# Dict-style batches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["source", "image", "kspace", "data"])
def test_extract_from_dict_with_documented_keys(key: str) -> None:
    """Each documented key returns the associated tensor."""
    tensor = torch.zeros(2, 1, 8, 8)
    batch = {key: tensor}
    out = _extract_input_tensor(batch)
    assert torch.equal(out, tensor)


def test_extract_from_dict_torchio_scalar_image() -> None:
    """TorchIO-style ``{'image': {'data': tensor, ...}}`` is unwrapped."""
    tensor = torch.zeros(2, 1, 8, 8)
    batch = {"image": {"data": tensor, "affine": torch.eye(4)}}
    out = _extract_input_tensor(batch)
    assert torch.equal(out, tensor)


def test_extract_from_dict_fallback_first_tensor() -> None:
    """When no documented key matches, falls back to first tensor with ndim≥3."""
    tensor = torch.zeros(1, 1, 8, 8)
    batch = {"my_field": tensor, "metadata": {"foo": "bar"}}
    out = _extract_input_tensor(batch)
    assert torch.equal(out, tensor)


def test_extract_from_dict_no_tensor_raises() -> None:
    """Empty/non-tensor dict raises with actionable message (CLAUDE.md #9)."""
    with pytest.raises(ValueError, match="No suitable input tensor"):
        _extract_input_tensor({"foo": "bar", "baz": 123})


# ---------------------------------------------------------------------------
# Tuple / list batches
# ---------------------------------------------------------------------------


def test_extract_from_tuple_returns_first_element() -> None:
    """``(x, y)`` style — returns ``x``."""
    x = torch.zeros(2, 1, 8, 8)
    y = torch.ones(2, 1, 8, 8)
    out = _extract_input_tensor((x, y))
    assert torch.equal(out, x)


def test_extract_from_list_returns_first_element() -> None:
    """List-style batch is also supported."""
    x = torch.zeros(2, 1, 4, 4)
    y = torch.ones(2, 1, 4, 4)
    out = _extract_input_tensor([x, y])
    assert torch.equal(out, x)


def test_extract_from_empty_tuple_raises() -> None:
    """Empty tuple has nothing to return — fail loud."""
    with pytest.raises(ValueError, match="Empty batch"):
        _extract_input_tensor(())


def test_extract_from_tuple_first_non_tensor_raises() -> None:
    """First element non-tensor → fail loud, not silent fallback."""
    with pytest.raises(ValueError, match="expected Tensor"):
        _extract_input_tensor(("not a tensor", torch.zeros(2)))


# ---------------------------------------------------------------------------
# Bare tensor
# ---------------------------------------------------------------------------


def test_extract_from_bare_tensor() -> None:
    """A raw tensor input is returned as-is."""
    t = torch.zeros(2, 1, 4, 4)
    out = _extract_input_tensor(t)
    assert torch.equal(out, t)


# ---------------------------------------------------------------------------
# Unsupported types
# ---------------------------------------------------------------------------


def test_extract_from_unsupported_type_raises() -> None:
    """Strings, ints, None all fail with actionable error."""
    with pytest.raises(ValueError, match="Unsupported batch type"):
        _extract_input_tensor("not a batch")
    with pytest.raises(ValueError, match="Unsupported batch type"):
        _extract_input_tensor(42)


def test_dynamic_sample_batch_builds_a_non_persistent_loader(monkeypatch) -> None:
    """The helper reads one batch and exits; a persistent pool would be leaked."""
    import spectramr.infrastructure.builders.directors.data_pipeline_director as dpd
    from spectramr.shared.utils import data_utils

    seen = {}

    class _Director:
        def __init__(self, cfg):
            pass

        def build_dataloaders(self, **kwargs):
            seen.update(kwargs)
            return [{"input": torch.zeros(2, 1, 4, 4)}], None

    monkeypatch.setattr(dpd, "DataPipelineDirector", _Director)
    out = data_utils.get_sample_batch(object(), device=torch.device("cpu"))
    assert out.shape[0] == 1
    assert seen["persistent_workers"] is False
