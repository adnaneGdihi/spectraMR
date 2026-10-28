"""Regression test for the 2026-06 audit fix to the SCAS scout pipeline.

``ScoutAcquisitionTransform`` overwrote the ``input`` image in place and emitted
no ``scout`` key, so the SCAS hypernet's scout-conditioned density penalty was
dead every batch. The transform now emits a SEPARATE ``scout`` image (preserving
the full-resolution input), and the reconstruction mixin propagates the ``scout``
key into ``batch_context``.
"""
from __future__ import annotations


import torch


class _Img:
    """Minimal torchio-Image stand-in (.data / .set_data) for apply_transform."""

    def __init__(self, data: torch.Tensor) -> None:
        self._d = data

    @property
    def data(self) -> torch.Tensor:
        return self._d

    def set_data(self, d: torch.Tensor) -> None:
        self._d = d


class _Subj(dict):
    def add_image(self, image: object, name: str) -> None:
        self[name] = image


def test_scout_transform_emits_separate_scout_preserving_input() -> None:
    from spectramr.data.transforms.scout_acquisition import ScoutAcquisitionTransform

    inp = torch.randn(1, 64, 64)
    subj = _Subj({"input": _Img(inp.clone())})
    t = ScoutAcquisitionTransform(
        scout_resolution=(16, 16), domain="kspace", keys=("input",)
    )
    out = t.apply_transform(subj)

    # Both keys present: input preserved + scout added (not an in-place overwrite).
    assert "input" in out and "scout" in out
    assert torch.equal(out["input"].data, inp)

    scout = out["scout"].data
    assert scout.shape == inp.shape
    # k-space crop: samples outside the central 16x16 window are zeroed.
    assert torch.all(scout[..., 0, 0] == 0)
    # ...and the centre retains signal.
    assert torch.any(scout[..., 32, 32] != 0)


def test_reconstruction_mixin_propagates_scout_key() -> None:
    """A batch carrying the transform's ``scout`` view reaches the batch context
    the SCAS hypernet reads (#2439: asserted on behaviour, not on source text)."""
    from types import SimpleNamespace

    from spectramr.infrastructure.training.strategies.mixins.reconstruction import (
        ReconstructionMixin,
    )

    class _Probe(ReconstructionMixin):
        env = None
        state = SimpleNamespace(config=SimpleNamespace(model=SimpleNamespace(input_type="image")))

    scout = torch.randn(1, 1, 8, 8)
    context = _Probe()._prepare_batch_context_reconstruction(
        torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8), batch={"scout": scout}
    )
    assert torch.equal(context["scout"], scout)
