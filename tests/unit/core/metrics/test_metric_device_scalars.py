"""HFEN, RobustMRI_PSNR and PhaseMSE return device scalars without a host sync (#2163).

The metrics computer defers every real 0-d tensor to one fused host transfer
(``fuse_to_host``). A metric that calls ``.item()``, branches in Python on a
device scalar, or boolean-indexes by a device mask forces its own sync first,
which makes that fusion a facade. The ``_legacy_*`` functions below are frozen
copies of the pre-#2163 bodies: they are the numerical oracle for the rewrite
and the planted violations that prove the sync guard turns red.
"""

from __future__ import annotations

import contextlib

import pytest
import torch
import torch.nn.functional as F

from spectramr.core.metrics.complex_input import PhaseBlindInputError
from spectramr.core.metrics.evaluation_metrics import PhaseMSE, RobustMRI_PSNR
from spectramr.core.metrics.hfen import HFENMetric
from spectramr.infrastructure.physics.fft_ops import fft2c


class HostSyncError(AssertionError):
    """Raised by the guard when a metric reaches for a host value."""


def _legacy_hfen(metric: HFENMetric, pred, target):
    pred = pred.abs() if torch.is_complex(pred) else pred
    target = target.abs() if torch.is_complex(target) else target
    numerator = (metric._apply_log(pred) - metric._apply_log(target)).abs().sum()
    return (numerator / (metric._apply_log(target).abs().sum() + 1e-8)).item()


def _magnitude(x):
    if torch.is_complex(x):
        return x.abs()
    if x.dim() == 4 and x.shape[1] == 2:
        return torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    return x.abs()


def _legacy_robust_psnr(pred, target, thresh=1e-5, percentile=0.999):
    pred_mag, target_mag = _magnitude(pred), _magnitude(target)
    if percentile < 1.0:
        robust_max = torch.quantile(target_mag.flatten(), percentile)
    else:
        robust_max = target_mag.max()
    tissue_mask = target_mag > (robust_max * 0.05 if robust_max > thresh else thresh)
    if tissue_mask.sum() == 0:
        mse = F.mse_loss(pred_mag, target_mag, reduction="mean")
        data_range = robust_max.item() if robust_max > 0 else 1.0
        return 20 * torch.log10(
            torch.tensor(data_range, device=pred.device) / (torch.sqrt(mse) + 1e-10)
        )
    mse = torch.sum((pred_mag[tissue_mask] - target_mag[tissue_mask]) ** 2) / tissue_mask.sum()
    data_range = robust_max.item()
    if data_range < 1e-10:
        data_range = 1.0
    return 20 * torch.log10(
        torch.tensor(data_range, device=pred.device, dtype=mse.dtype) / (torch.sqrt(mse) + 1e-10)
    )


def _legacy_phase_mse(pred, target):
    if pred.dim() == 4 and pred.size(1) == 2:
        pred_c = torch.complex(pred[:, 0], pred[:, 1])
        target_c = torch.complex(target[:, 0], target[:, 1])
    elif torch.is_complex(pred):
        pred_c, target_c = pred, target
    else:
        pred_c, target_c = fft2c(pred), fft2c(target)
    magnitude_mask = torch.abs(target_c) > 0.05
    diff = torch.angle(torch.exp(1j * (torch.angle(pred_c) - torch.angle(target_c))))
    if magnitude_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return (diff[magnitude_mask] ** 2).mean()


@contextlib.contextmanager
def no_host_sync():
    """Make every Python-visible device-to-host read raise.

    Covers ``.item()``/``.tolist()``, the implicit ``bool``/``float``/``int`` a
    Python ``if`` or cast takes, and boolean-mask indexing (a ``nonzero`` sync).
    """
    original_getitem = torch.Tensor.__getitem__

    def _raise(*_a, **_k):
        raise HostSyncError("metric forced a device-to-host read")

    def _getitem(self, index):
        parts = index if isinstance(index, tuple) else (index,)
        if any(isinstance(p, torch.Tensor) and p.dtype == torch.bool for p in parts):
            raise HostSyncError("boolean-mask indexing forces a nonzero() sync")
        return original_getitem(self, index)

    with pytest.MonkeyPatch.context() as mp:
        for slot in ("item", "tolist", "__bool__", "__float__", "__int__"):
            mp.setattr(torch.Tensor, slot, _raise)
        mp.setattr(torch.Tensor, "__getitem__", _getitem)
        yield


def _new(name):
    return {"hfen": HFENMetric(), "robust": RobustMRI_PSNR(), "phase": PhaseMSE()}[name]


def _legacy(name):
    hfen = HFENMetric()
    return {
        "hfen": lambda p, t: _legacy_hfen(hfen, p, t),
        "robust": _legacy_robust_psnr,
        "phase": _legacy_phase_mse,
    }[name]


def _inputs(kind, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "complex":
        return tuple(torch.randn(2, 1, 24, 24, dtype=torch.complex64, generator=g) for _ in "pt")
    if kind == "two_channel":
        return tuple(torch.randn(2, 2, 24, 24, generator=g) for _ in "pt")
    if kind == "float64":
        return tuple(torch.rand(2, 1, 24, 24, dtype=torch.float64, generator=g) for _ in "pt")
    return tuple(torch.rand(2, 1, 24, 24, generator=g) for _ in "pt")


def _assert_same(new, old):
    assert isinstance(new, torch.Tensor) and new.dim() == 0
    torch.testing.assert_close(float(new), float(old), rtol=1e-5, atol=1e-6, equal_nan=True)


# --- guard self-tests: the slot overrides actually intercept ---------------------


@pytest.mark.parametrize(
    "violation",
    [
        lambda: torch.tensor(1.0).item(),
        lambda: bool(torch.tensor(1.0) > 0),
        lambda: float(torch.tensor(1.0)),
        lambda: torch.arange(4)[torch.arange(4) > 1],
    ],
    ids=["item", "if_tensor", "float_cast", "bool_index"],
)
def test_guard_catches_each_sync_shape(violation):
    with no_host_sync(), pytest.raises(HostSyncError):
        violation()


@pytest.mark.parametrize("name", ["hfen", "robust", "phase"])
def test_guard_is_red_on_legacy_implementation(name):
    pred, target = _inputs("real", 0)
    with no_host_sync(), pytest.raises(HostSyncError):
        _legacy(name)(pred, target)


# phase_mse refuses real single-channel input (#867), so it is exercised only on
# the phase-bearing layouts; its refusal on real input is pinned separately.
_PHASE_BEARING = ("complex", "two_channel")


def _cases(kinds):
    return [
        (name, kind)
        for name in ("hfen", "robust", "phase")
        for kind in kinds
        if name != "phase" or kind in _PHASE_BEARING
    ]


@pytest.mark.parametrize(("name", "kind"), _cases(["real", "complex", "two_channel"]))
def test_metric_returns_device_scalar_without_sync(name, kind):
    pred, target = _inputs(kind, 1)
    metric = _new(name)
    with no_host_sync():
        value = metric(pred, target)
    assert isinstance(value, torch.Tensor) and value.dim() == 0


# --- numerical equivalence with the pre-#2163 bodies -----------------------------


@pytest.mark.parametrize(("name", "kind"), _cases(["real", "complex", "two_channel", "float64"]))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_legacy_on_random_inputs(name, kind, seed):
    pred, target = _inputs(kind, seed)
    _assert_same(_new(name)(pred, target), _legacy(name)(pred, target))


@pytest.mark.parametrize("kind", ["real", "float64"])
def test_phase_mse_refuses_real_input_without_sync(kind):
    pred, target = _inputs(kind, 0)
    with no_host_sync(), pytest.raises(PhaseBlindInputError):
        PhaseMSE()(pred, target)


def _robust_edge_cases():
    z = torch.zeros(1, 1, 16, 16)
    tiny = torch.full((1, 1, 16, 16), 5e-6)
    outlier = torch.zeros(1, 1, 100, 100)
    outlier[0, 0, 0, 0] = 1.0  # robust_max 0, one tissue voxel -> range 1.0
    nan_bg_target = torch.rand(1, 1, 16, 16) + 0.5
    nan_bg_target[..., :4] = 0.0
    nan_bg_pred = nan_bg_target.clone()
    nan_bg_pred[..., :4] = float("nan")
    rnd = torch.rand(1, 1, 16, 16)
    return {
        "zero_target": (torch.rand(1, 1, 16, 16), z, 0.999),
        "tiny_target_fallback": (torch.rand(1, 1, 16, 16) * 1e-5, tiny, 0.999),
        "outlier_only_tissue": (torch.rand(1, 1, 100, 100), outlier, 0.999),
        "percentile_one_max_path": (torch.rand(1, 1, 16, 16), rnd, 1.0),
        "exact_match": (rnd, rnd.clone(), 0.999),
        "nan_in_background_pred": (nan_bg_pred, nan_bg_target, 0.999),
    }


@pytest.mark.parametrize("case", list(_robust_edge_cases()))
def test_robust_psnr_matches_legacy_on_edge_cases(case):
    pred, target, pct = _robust_edge_cases()[case]
    new = RobustMRI_PSNR(data_range_percentile=pct)(pred, target)
    old = _legacy_robust_psnr(pred, target, percentile=pct)
    _assert_same(new, old)
    if case == "nan_in_background_pred":
        assert torch.isfinite(new)


@pytest.mark.parametrize(
    "case",
    ["empty_mask_two_channel", "empty_mask_complex", "partial_mask", "nan_masked_out"],
)
def test_phase_mse_matches_legacy_on_edge_cases(case):
    t = torch.randn(1, 2, 16, 16)
    p = torch.randn(1, 2, 16, 16)
    if case == "empty_mask_two_channel":
        t = t * 1e-3
    elif case == "empty_mask_complex":
        t = torch.complex(t[:, :1], t[:, 1:]) * 1e-3
        p = torch.complex(p[:, :1], p[:, 1:])
    elif case == "partial_mask":
        t[..., :8] *= 1e-3
    else:
        t[..., :8] = 0.0
        p[..., :8] = float("nan")
    new = PhaseMSE()(p, t)
    _assert_same(new, _legacy_phase_mse(p, t))
    if case.startswith("empty_mask"):
        assert float(new) == 0.0


def test_hfen_matches_legacy_on_zero_target():
    pred, target = torch.rand(1, 1, 24, 24), torch.zeros(1, 1, 24, 24)
    metric = HFENMetric()
    _assert_same(metric(pred, target), _legacy_hfen(metric, pred, target))


# --- the real CUDA sync check, where an accelerator exists -----------------------


@contextlib.contextmanager
def _cuda_sync_errors():
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sync debug mode needs CUDA")
@pytest.mark.parametrize("name", ["hfen", "robust", "phase"])
def test_cuda_sync_debug_mode(name):
    kind = "two_channel" if name == "phase" else "real"
    pred, target = (t.cuda() for t in _inputs(kind, 3))
    metric = {
        "hfen": HFENMetric(),
        "robust": RobustMRI_PSNR(device="cuda"),
        "phase": PhaseMSE(device="cuda"),
    }[name]
    metric(pred, target)  # warm-up: one-time kernel/constant uploads are not per-step syncs
    with _cuda_sync_errors():
        value = metric(pred, target)
    assert value.is_cuda and value.dim() == 0
    legacy = _legacy(name)
    legacy(pred, target)  # same warm-up, so the red half is the metric's own sync
    with _cuda_sync_errors(), pytest.raises(RuntimeError, match="synchronizing"):
        legacy(pred, target)
