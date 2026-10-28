"""Streaming k-space assembly from per-state renders."""

from __future__ import annotations

import weakref

import pytest
import torch

from spectramr.infrastructure.physics.degradations.blocks.time_states import assemble_from_states
from spectramr.infrastructure.physics.fft_ops import fftnc

DIMS = {"slice": (-2, -1), "volume": (-3, -2, -1)}
SHAPE = (2, 2, 3, 4, 5)


def _img(seed):
    return torch.randn(*SHAPE, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("encoding", ["slice", "volume"])
def test_single_image_for_all_states_equals_fftnc(encoding):
    img = _img(0)
    idx = torch.arange(12).view(3, 4) % 3
    k = assemble_from_states(3, idx, lambda p: img.clone(), encoding=encoding)
    assert torch.equal(k, fftnc(img, dims=DIMS[encoding]))


@pytest.mark.parametrize("encoding", ["slice", "volume"])
def test_each_line_comes_from_its_state(encoding):
    imgs = [_img(1), _img(2)]
    idx = torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 0]])
    k = assemble_from_states(2, idx, lambda p: imgs[p], encoding=encoding)
    ks = [fftnc(i, dims=DIMS[encoding]) for i in imgs]
    for p in range(2):
        sel = idx == p
        assert torch.equal(k[:, :, sel, :], ks[p][:, :, sel, :])


@pytest.mark.parametrize("bad", [-1, 2])
def test_state_index_out_of_range_raises(bad):
    idx = torch.zeros(3, 4, dtype=torch.long)
    idx[0, 0] = bad
    with pytest.raises(ValueError):
        assemble_from_states(2, idx, lambda p: _img(p), encoding="slice")


@pytest.mark.parametrize(
    "idx",
    [torch.zeros(4, 3, dtype=torch.long), torch.zeros(3, 4), torch.zeros(12, dtype=torch.long)],
)
def test_bad_state_index_shape_or_dtype_raises(idx):
    with pytest.raises(ValueError):
        assemble_from_states(1, idx, lambda p: _img(p), encoding="slice")


def test_unknown_encoding_raises():
    with pytest.raises(ValueError):
        assemble_from_states(
            1, torch.zeros(3, 4, dtype=torch.long), lambda p: _img(p), encoding="x"
        )  # type: ignore[arg-type]


def test_streaming_one_render_per_state_and_no_retained_outputs():
    refs: list[weakref.ref] = []
    alive_at_call: list[int] = []

    def render(p):
        alive_at_call.append(sum(r() is not None for r in refs))
        out = _img(p)
        refs.append(weakref.ref(out))
        return out

    idx = torch.arange(12).view(3, 4) % 4
    assemble_from_states(4, idx, render, encoding="volume")
    assert len(refs) == 4
    assert alive_at_call == [0, 0, 0, 0]
    assert all(r() is None for r in refs)


def test_non_cpu_state_index_raises():
    idx = torch.zeros(3, 4, dtype=torch.long, device="meta")
    with pytest.raises(ValueError, match="meta"):
        assemble_from_states(1, idx, lambda p: _img(p), encoding="slice")


def test_render_shape_mismatch_between_states_raises():
    idx = torch.arange(12).view(3, 4) % 2
    shapes = [SHAPE, (2, 2, 3, 4, 6)]
    with pytest.raises(ValueError, match="differs"):
        assemble_from_states(2, idx, lambda p: torch.randn(*shapes[p]), encoding="slice")
