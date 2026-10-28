"""Tests for the diffusion strategy's chunked validation forward (#1914)."""

from __future__ import annotations

import torch

from spectramr.infrastructure.training.strategies.validation_chunking import (
    forward_in_chunks,
    reconstruction_output,
    slice_batch_aligned,
)


def test_batch_aligned_tensors_are_sliced_and_shared_ones_are_not():
    batch = 5
    kwargs = {
        "smaps": torch.arange(batch),
        "mask": torch.ones(1, 1, 4, 4),
        "scalar": torch.tensor(3.0),
        "flag": True,
    }
    out = slice_batch_aligned(kwargs, 2, 4, batch)
    assert torch.equal(out["smaps"], torch.tensor([2, 3]))
    assert out["mask"] is kwargs["mask"]
    assert out["scalar"] is kwargs["scalar"]
    assert out["flag"] is True


def test_every_chunk_sees_the_rows_of_its_own_samples():
    batch, chunk = 5, 2
    x = torch.arange(batch, dtype=torch.float32).view(batch, 1)
    calls = []

    def forward(xc, ts, kw):
        calls.append((xc.clone(), ts, kw["per_sample"].clone()))
        assert torch.equal(xc, kw["per_sample"]), "a chunk met another chunk's rows"
        return xc * 10

    out = forward_in_chunks(forward, x, torch.arange(batch), {"per_sample": x.clone()}, chunk)
    assert [c[0].shape[0] for c in calls] == [2, 2, 1]
    assert torch.equal(calls[2][1], torch.tensor([4]))
    assert torch.equal(out, x * 10)


def test_a_batch_within_one_chunk_is_forwarded_whole_and_returned_as_is():
    marker = ("pred", "aux")
    out = forward_in_chunks(lambda x, t, kw: marker, torch.zeros(2, 1), None, {}, 4)
    assert out is marker


def test_tuple_and_dict_outputs_contribute_their_primary_tensor():
    x = torch.zeros(4, 1)
    tup = forward_in_chunks(lambda xc, t, kw: (xc + 1, "aux"), x, None, {}, 2)
    dic = forward_in_chunks(lambda xc, t, kw: {"image": xc + 2}, x, None, {}, 2)
    assert torch.equal(tup, x + 1)
    assert torch.equal(dic, x + 2)


def test_reconstruction_output_prefers_the_reconstruction_key():
    x = torch.ones(2, 1)
    assert reconstruction_output(x) is x
    assert reconstruction_output((x, "aux")) is x
    assert reconstruction_output({"image": x * 0, "reconstruction": x}) is x
    assert reconstruction_output({"output": x}) is x
    assert reconstruction_output({"other": x}) is x
