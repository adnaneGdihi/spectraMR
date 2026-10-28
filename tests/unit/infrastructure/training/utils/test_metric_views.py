"""``MetricViews`` carries the complex pair to the computer only when it exists (#867)."""

from __future__ import annotations

import torch

from spectramr.infrastructure.training.utils.metric_views import (
    MetricViews,
    real_stacked_to_complex,
)


class _SpyComputer:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def compute(self, pred, target, **kwargs):
        self.kwargs = kwargs
        return {"ok": 1.0}


def test_complex_pair_requires_both_halves() -> None:
    z = torch.complex(torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
    m = z.abs()
    assert MetricViews(m, m).complex_pair is None
    assert MetricViews(m, m, z, None).complex_pair is None
    assert MetricViews(m, m, z, z).complex_pair == (z, z)


def test_compute_passes_the_pair_only_when_present() -> None:
    z = torch.complex(torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
    spy = _SpyComputer()
    MetricViews(z.abs(), z.abs()).compute(spy, only=["mae"])
    assert spy.kwargs == {"only": ["mae"]}
    MetricViews(z.abs(), z.abs(), z, z).compute(spy)
    assert spy.kwargs["complex_pair"][0] is z


def test_real_stacked_to_complex_pairs_interleaved_channels() -> None:
    t = torch.randn(2, 4, 3, 3)
    out = real_stacked_to_complex(t)
    assert out.shape == (2, 2, 3, 3)
    assert torch.equal(out.real, t[:, 0::2]) and torch.equal(out.imag, t[:, 1::2])


def test_real_stacked_to_complex_leaves_other_shapes_alone() -> None:
    odd = torch.randn(2, 3, 3, 3)
    assert real_stacked_to_complex(odd) is odd
    z = torch.complex(torch.randn(1, 1, 2, 2), torch.randn(1, 1, 2, 2))
    assert real_stacked_to_complex(z) is z
