"""MONAI surface-distance calls stay free of MONAI's internal always_return_as_numpy warning."""

from __future__ import annotations

import warnings

import pytest
import torch

pytest.importorskip("monai")

from spectramr.core.metrics.monai_compat import call_surface_metric
from spectramr.core.metrics.registry import MetricsRegistry


def _no_deprecation(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
    assert not [w for w in caught if "always_return_as_numpy" in str(w.message)]
    return out


def _blob(shift: int = 0) -> torch.Tensor:
    x = torch.zeros(1, 1, 32, 32)
    x[..., 8 + shift : 20 + shift, 8:20] = 1.0
    return x


def test_call_surface_metric_is_quiet_and_returns_the_metric_value():
    from monai.metrics import HausdorffDistanceMetric

    metric = HausdorffDistanceMetric(include_background=True, percentile=95)
    d = _no_deprecation(lambda: call_surface_metric(metric, y_pred=_blob(2), y=_blob()))
    assert float(d.mean()) == pytest.approx(2.0)


@pytest.mark.parametrize("key", ["hd95", "tissue_hd95"])
def test_registered_hausdorff_metrics_do_not_warn(key):
    if not MetricsRegistry.is_registered(key):
        pytest.skip(f"{key} not registered")
    metric = MetricsRegistry.get(key)
    img = torch.rand(1, 1, 32, 32) * 0.2 + _blob() * 0.8
    _no_deprecation(lambda: metric(img + 0.05 * torch.rand_like(img), img))
