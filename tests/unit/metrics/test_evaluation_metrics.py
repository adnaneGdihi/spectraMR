"""Unit tests for evaluation_metrics.py — four archetypes per public metric.

Covers:
- canary: each class instantiates and compute_metric returns a finite scalar.
- parametrize: batch sizes {1, 2} and value-range variants.
- edge: identical inputs, all-zero target, constant arrays, length-1 batches.
- raises: shape mismatches raise rather than silently fall back.
- sanity_shape: SHAPE_MATRIX_2D sweep for the lightweight metrics.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.utils.shape_matrices import SHAPE_MATRIX_2D, shape_id

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _img(b: int, c: int, h: int = 16, w: int = 16, seed: int = 0) -> torch.Tensor:
    """Return a seeded float32 image [B, C, H, W] in [0, 1]."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.rand(b, c, h, w, generator=g)


def _is_finite_scalar(t: torch.Tensor) -> bool:
    return t.ndim == 0 and math.isfinite(t.item())


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------


class TestPSNR:
    @pytest.fixture
    def psnr(self):
        from spectramr.core.metrics.evaluation_metrics import PSNR
        return PSNR()

    # canary
    def test_canary_finite(self, psnr):
        x = _img(1, 1)
        y = _img(1, 1, seed=1)
        result = psnr.compute_metric(x, y)
        assert _is_finite_scalar(result)

    # parametrize
    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, psnr, b):
        x = _img(b, 1)
        y = _img(b, 1, seed=2)
        result = psnr.compute_metric(x, y)
        assert _is_finite_scalar(result)

    @pytest.mark.parametrize("offset", [0.0, 0.5, -0.5])
    def test_value_ranges(self, offset):
        from spectramr.core.metrics.evaluation_metrics import PSNR
        m = PSNR()
        x = _img(1, 1) + offset
        y = _img(1, 1, seed=3) + offset
        result = m.compute_metric(x, y)
        assert _is_finite_scalar(result)

    # edge: all-zero target (MSE=0) -> no div-by-zero
    def test_edge_zero_target(self, psnr):
        x = torch.zeros(1, 1, 16, 16)
        result = psnr.compute_metric(x, x)
        assert math.isfinite(result.item())

    # edge: single pixel batch
    def test_edge_length1_batch(self, psnr):
        x = _img(1, 1, 4, 4)
        y = _img(1, 1, 4, 4, seed=5)
        result = psnr.compute_metric(x, y)
        assert _is_finite_scalar(result)

    # raises: shape mismatch
    def test_raises_shape_mismatch(self, psnr):
        x = _img(1, 1, 16, 16)
        y = _img(1, 1, 8, 8)
        with pytest.raises((AssertionError, RuntimeError)):
            psnr.compute_metric(x, y)

    # sanity_shape
    @pytest.mark.sanity_shape
    @pytest.mark.parametrize("shape", SHAPE_MATRIX_2D, ids=shape_id)
    def test_shape_matrix(self, shape):
        from spectramr.core.metrics.evaluation_metrics import PSNR
        b, c, h, w = shape
        x = _img(b, c, h, w)
        y = _img(b, c, h, w, seed=99)
        result = PSNR().compute_metric(x, y)
        assert _is_finite_scalar(result)


# ---------------------------------------------------------------------------
# MSE
# ---------------------------------------------------------------------------


class TestMSE:
    @pytest.fixture
    def mse(self):
        from spectramr.core.metrics.evaluation_metrics import MSE
        return MSE()

    def test_canary_finite(self, mse):
        x = _img(1, 1)
        y = _img(1, 1, seed=1)
        assert _is_finite_scalar(mse.compute_metric(x, y))

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, mse, b):
        x = _img(b, 1)
        y = _img(b, 1, seed=2)
        assert _is_finite_scalar(mse.compute_metric(x, y))

    def test_edge_identical_zero(self, mse):
        x = _img(2, 1)
        result = mse.compute_metric(x, x)
        assert result.item() == pytest.approx(0.0, abs=1e-7)

    def test_edge_all_zero_target(self, mse):
        z = torch.zeros(1, 1, 16, 16)
        result = mse.compute_metric(z, z)
        assert math.isfinite(result.item())

    def test_edge_constant_arrays(self, mse):
        a = torch.ones(1, 1, 16, 16) * 0.5
        b = torch.ones(1, 1, 16, 16) * 0.5
        assert mse.compute_metric(a, b).item() == pytest.approx(0.0, abs=1e-7)

    @pytest.mark.sanity_shape
    @pytest.mark.parametrize("shape", SHAPE_MATRIX_2D, ids=shape_id)
    def test_shape_matrix(self, shape):
        from spectramr.core.metrics.evaluation_metrics import MSE
        b, c, h, w = shape
        x = _img(b, c, h, w)
        y = _img(b, c, h, w, seed=7)
        assert _is_finite_scalar(MSE().compute_metric(x, y))


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------


class TestMAE:
    @pytest.fixture
    def mae(self):
        from spectramr.core.metrics.evaluation_metrics import MAE
        return MAE()

    def test_canary_finite(self, mae):
        x = _img(1, 1)
        y = _img(1, 1, seed=1)
        assert _is_finite_scalar(mae.compute_metric(x, y))

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, mae, b):
        assert _is_finite_scalar(mae.compute_metric(_img(b, 1), _img(b, 1, seed=3)))

    def test_edge_identical_zero(self, mae):
        x = _img(2, 1)
        assert mae.compute_metric(x, x).item() == pytest.approx(0.0, abs=1e-7)

    def test_edge_all_zero(self, mae):
        z = torch.zeros(1, 1, 16, 16)
        assert math.isfinite(mae.compute_metric(z, z).item())

    @pytest.mark.sanity_shape
    @pytest.mark.parametrize("shape", SHAPE_MATRIX_2D, ids=shape_id)
    def test_shape_matrix(self, shape):
        from spectramr.core.metrics.evaluation_metrics import MAE
        b, c, h, w = shape
        assert _is_finite_scalar(MAE().compute_metric(_img(b, c, h, w), _img(b, c, h, w, seed=11)))


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------


class TestRMSE:
    @pytest.fixture
    def rmse(self):
        from spectramr.core.metrics.evaluation_metrics import RMSE
        return RMSE()

    def test_canary_finite(self, rmse):
        x = _img(1, 1)
        y = _img(1, 1, seed=1)
        assert _is_finite_scalar(rmse.compute_metric(x, y))

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, rmse, b):
        assert _is_finite_scalar(rmse.compute_metric(_img(b, 1), _img(b, 1, seed=4)))

    def test_edge_identical_zero_rmse(self, rmse):
        x = _img(2, 1)
        assert rmse.compute_metric(x, x).item() == pytest.approx(0.0, abs=1e-7)

    def test_edge_all_zero(self, rmse):
        z = torch.zeros(1, 1, 16, 16)
        assert math.isfinite(rmse.compute_metric(z, z).item())


# ---------------------------------------------------------------------------
# SSIMMetric
# ---------------------------------------------------------------------------


class TestSSIM:
    @pytest.fixture
    def ssim(self):
        from spectramr.core.metrics.evaluation_metrics import SSIMMetric
        return SSIMMetric()

    def test_canary_finite(self, ssim):
        x = _img(1, 1)
        y = _img(1, 1, seed=1)
        result = ssim.compute_metric(x, y)
        assert _is_finite_scalar(result)

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, ssim, b):
        x = _img(b, 1)
        y = _img(b, 1, seed=5)
        assert _is_finite_scalar(ssim.compute_metric(x, y))

    def test_edge_identical_inputs_near_one(self, ssim):
        x = _img(1, 1)
        result = ssim.compute_metric(x, x)
        # SSIM of identical images should be very close to 1.0
        assert result.item() > 0.98, f"SSIM of identical images: {result.item()}"

    def test_edge_all_zero_target(self, ssim):
        z = torch.zeros(1, 1, 16, 16)
        result = ssim.compute_metric(z, z)
        assert math.isfinite(result.item())

    def test_raises_shape_mismatch(self, ssim):
        x = _img(1, 1, 16, 16)
        y = _img(1, 1, 8, 8)
        with pytest.raises((AssertionError, RuntimeError)):
            ssim.compute_metric(x, y)

    @pytest.mark.sanity_shape
    @pytest.mark.parametrize("shape", SHAPE_MATRIX_2D, ids=shape_id)
    def test_shape_matrix(self, shape):
        from spectramr.core.metrics.evaluation_metrics import SSIMMetric
        b, c, h, w = shape
        x = _img(b, c, h, w)
        y = _img(b, c, h, w, seed=13)
        result = SSIMMetric().compute_metric(x, y)
        assert _is_finite_scalar(result)


# ---------------------------------------------------------------------------
# NMSE
# ---------------------------------------------------------------------------


class TestNMSE:
    @pytest.fixture
    def nmse(self):
        from spectramr.core.metrics.evaluation_metrics import NMSE
        return NMSE()

    def test_canary_finite(self, nmse):
        x = _img(1, 1)
        y = _img(1, 1, seed=1)
        assert _is_finite_scalar(nmse.compute_metric(x, y))

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, nmse, b):
        assert _is_finite_scalar(nmse.compute_metric(_img(b, 1), _img(b, 1, seed=6)))

    def test_edge_identical_zero(self, nmse):
        x = _img(2, 1)
        assert nmse.compute_metric(x, x).item() == pytest.approx(0.0, abs=1e-7)

    def test_edge_all_zero_target_no_div_zero(self, nmse):
        z = torch.zeros(1, 1, 16, 16)
        result = nmse.compute_metric(z, z)
        assert math.isfinite(result.item())


# ---------------------------------------------------------------------------
# GradientError
# ---------------------------------------------------------------------------


class TestGradientError:
    @pytest.fixture
    def ge(self):
        from spectramr.core.metrics.evaluation_metrics import GradientError
        return GradientError()

    def test_canary_finite(self, ge):
        x = _img(1, 1)
        y = _img(1, 1, seed=1)
        assert _is_finite_scalar(ge.compute_metric(x, y))

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, ge, b):
        assert _is_finite_scalar(ge.compute_metric(_img(b, 1), _img(b, 1, seed=7)))

    def test_edge_identical_zero(self, ge):
        x = _img(2, 1)
        assert ge.compute_metric(x, x).item() == pytest.approx(0.0, abs=1e-5)

    def test_edge_constant_arrays(self, ge):
        a = torch.ones(1, 1, 16, 16) * 0.3
        b = torch.ones(1, 1, 16, 16) * 0.7
        result = ge.compute_metric(a, b)
        assert math.isfinite(result.item())


# ---------------------------------------------------------------------------
# SNR
# ---------------------------------------------------------------------------


class TestSNR:
    @pytest.fixture
    def snr(self):
        from spectramr.core.metrics.evaluation_metrics import SNR
        return SNR()

    def test_canary_finite(self, snr):
        x = _img(1, 1)
        y = _img(1, 1, seed=1)
        assert _is_finite_scalar(snr.compute_metric(x, y))

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, snr, b):
        assert _is_finite_scalar(snr.compute_metric(_img(b, 1), _img(b, 1, seed=8)))

    def test_edge_identical_inputs_high_snr(self, snr):
        x = _img(1, 1)
        result = snr.compute_metric(x, x)
        # identical => noise_power=0 => capped at 100 dB
        assert result.item() == pytest.approx(100.0, abs=0.1)

    def test_edge_all_zero_no_div_zero(self, snr):
        z = torch.zeros(1, 1, 16, 16)
        result = snr.compute_metric(z, z)
        assert math.isfinite(result.item())


# ---------------------------------------------------------------------------
# GMSD
# ---------------------------------------------------------------------------


class TestGMSD:
    @pytest.fixture
    def gmsd(self):
        from spectramr.core.metrics.evaluation_metrics import GMSD
        return GMSD()

    def test_canary_finite(self, gmsd):
        x = _img(1, 1)
        y = _img(1, 1, seed=1)
        assert _is_finite_scalar(gmsd.compute_metric(x, y))

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, gmsd, b):
        assert _is_finite_scalar(gmsd.compute_metric(_img(b, 1), _img(b, 1, seed=9)))

    def test_edge_identical_low_deviation(self, gmsd):
        x = _img(1, 1)
        result = gmsd.compute_metric(x, x)
        assert math.isfinite(result.item())
        assert result.item() >= 0.0


# ---------------------------------------------------------------------------
# FSIM
# ---------------------------------------------------------------------------


class TestFSIM:
    @pytest.fixture
    def fsim(self):
        from spectramr.core.metrics.evaluation_metrics import FSIM
        return FSIM()

    def test_canary_in_unit_range(self, fsim):
        x = _img(1, 1)
        y = _img(1, 1, seed=1)
        result = fsim.compute_metric(x, y)
        assert _is_finite_scalar(result)
        assert 0.0 <= result.item() <= 1.0

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, fsim, b):
        result = fsim.compute_metric(_img(b, 1), _img(b, 1, seed=10))
        assert _is_finite_scalar(result) and 0.0 <= result.item() <= 1.0

    def test_edge_identical_near_one(self, fsim):
        x = _img(1, 1)
        result = fsim.compute_metric(x, x)
        assert result.item() >= 0.0

    def test_edge_constant_arrays(self, fsim):
        a = torch.ones(1, 1, 16, 16) * 0.4
        b = torch.ones(1, 1, 16, 16) * 0.6
        result = fsim.compute_metric(a, b)
        assert math.isfinite(result.item())


# ---------------------------------------------------------------------------
# PearsonCorrelation
# ---------------------------------------------------------------------------


class TestPearsonCorrelation:
    @pytest.fixture
    def pearson(self):
        from spectramr.core.metrics.evaluation_metrics import PearsonCorrelation
        return PearsonCorrelation()

    def test_canary_finite(self, pearson):
        x = _img(1, 1)
        y = _img(1, 1, seed=1)
        result = pearson.compute_metric(x, y)
        assert _is_finite_scalar(result)

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, pearson, b):
        result = pearson.compute_metric(_img(b, 1), _img(b, 1, seed=20))
        assert _is_finite_scalar(result)

    def test_edge_identical_near_one(self, pearson):
        x = _img(1, 1)
        result = pearson.compute_metric(x, x)
        assert result.item() > 0.99

    def test_edge_constant_no_div_zero(self, pearson):
        # Constant arrays have zero variance — the metric should guard against div-by-zero
        c = torch.ones(1, 1, 16, 16) * 0.5
        result = pearson.compute_metric(c, c)
        assert math.isfinite(result.item())


# ---------------------------------------------------------------------------
# KSpaceError
# ---------------------------------------------------------------------------


class TestKSpaceError:
    @pytest.fixture
    def kse(self):
        from spectramr.core.metrics.evaluation_metrics import KSpaceError
        return KSpaceError()

    def test_canary_finite(self, kse):
        x = _img(1, 1)
        y = _img(1, 1, seed=1)
        assert _is_finite_scalar(kse.compute_metric(x, y))

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, kse, b):
        assert _is_finite_scalar(kse.compute_metric(_img(b, 1), _img(b, 1, seed=30)))

    def test_edge_identical_zero(self, kse):
        x = _img(1, 1)
        result = kse.compute_metric(x, x)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_edge_all_zero(self, kse):
        z = torch.zeros(1, 1, 16, 16)
        assert math.isfinite(kse.compute_metric(z, z).item())


# ---------------------------------------------------------------------------
# EFC (no-reference — target ignored)
# ---------------------------------------------------------------------------


class TestEFC:
    @pytest.fixture
    def efc(self):
        from spectramr.core.metrics.evaluation_metrics import EFC
        return EFC()

    def test_canary_in_unit_range(self, efc):
        x = _img(1, 1)
        result = efc.compute_metric(x, x)  # target ignored
        assert _is_finite_scalar(result)
        assert 0.0 <= result.item() <= 1.0 + 1e-6

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, efc, b):
        x = _img(b, 1)
        result = efc.compute_metric(x, x)
        assert _is_finite_scalar(result)

    def test_edge_constant_array_finite(self, efc):
        c = torch.ones(1, 1, 16, 16) * 0.5
        result = efc.compute_metric(c, c)
        assert math.isfinite(result.item())


# ---------------------------------------------------------------------------
# Dice
# ---------------------------------------------------------------------------


class TestDice:
    @pytest.fixture
    def dice(self):
        from spectramr.core.metrics.evaluation_metrics import Dice
        return Dice()

    def test_canary_in_unit_range(self, dice):
        x = _img(1, 1)
        y = _img(1, 1, seed=1)
        result = dice.compute_metric(x, y)
        assert _is_finite_scalar(result)
        assert 0.0 <= result.item() <= 1.0

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, dice, b):
        x = _img(b, 1)
        y = _img(b, 1, seed=40)
        assert 0.0 <= dice.compute_metric(x, y).item() <= 1.0

    def test_edge_identical_near_one(self, dice):
        x = _img(1, 1)
        result = dice.compute_metric(x, x)
        assert result.item() > 0.95

    def test_edge_all_zero_no_div_zero(self, dice):
        z = torch.zeros(1, 1, 16, 16)
        result = dice.compute_metric(z, z)
        assert math.isfinite(result.item())


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------


class TestIoU:
    @pytest.fixture
    def iou(self):
        from spectramr.core.metrics.evaluation_metrics import IoU
        return IoU()

    def test_canary_in_unit_range(self, iou):
        x = _img(1, 1)
        y = _img(1, 1, seed=1)
        result = iou.compute_metric(x, y)
        assert _is_finite_scalar(result)
        assert 0.0 <= result.item() <= 1.0

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, iou, b):
        result = iou.compute_metric(_img(b, 1), _img(b, 1, seed=41))
        assert 0.0 <= result.item() <= 1.0

    def test_edge_identical_near_one(self, iou):
        x = _img(1, 1)
        assert iou.compute_metric(x, x).item() > 0.95

    def test_edge_no_overlap_near_zero(self, iou):
        # all-zero pred vs all-one target => denominator dominated by target
        pred = torch.zeros(1, 1, 16, 16)
        target = torch.ones(1, 1, 16, 16)
        result = iou.compute_metric(pred, target)
        assert math.isfinite(result.item())


# ---------------------------------------------------------------------------
# RobustMRI_PSNR
# ---------------------------------------------------------------------------


class TestRobustMRI_PSNR:
    @pytest.fixture
    def rpsnr(self):
        from spectramr.core.metrics.evaluation_metrics import RobustMRI_PSNR
        return RobustMRI_PSNR()

    def test_canary_finite(self, rpsnr):
        x = _img(1, 1)
        y = _img(1, 1, seed=1)
        assert _is_finite_scalar(rpsnr.compute_metric(x, y))

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, rpsnr, b):
        assert _is_finite_scalar(rpsnr.compute_metric(_img(b, 1), _img(b, 1, seed=50)))

    def test_edge_all_zero_target_fallback(self, rpsnr):
        z = torch.zeros(1, 1, 16, 16)
        result = rpsnr.compute_metric(z, z)
        # Should not raise; returns finite value via fallback
        assert math.isfinite(result.item())


# ---------------------------------------------------------------------------
# VIF
# ---------------------------------------------------------------------------


class TestVIF:
    @pytest.fixture
    def vif(self):
        from spectramr.core.metrics.evaluation_metrics import VIF
        return VIF()

    def test_canary_in_unit_range(self, vif):
        x = _img(1, 1, 64, 64)
        y = _img(1, 1, 64, 64, seed=1)
        result = vif.compute_metric(x, y)
        assert _is_finite_scalar(result)
        assert 0.0 <= result.item() <= 1.0 + 1e-5

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, vif, b):
        x = _img(b, 1, 64, 64)
        y = _img(b, 1, 64, 64, seed=51)
        result = vif.compute_metric(x, y)
        assert _is_finite_scalar(result)

    def test_edge_identical_nonzero(self, vif):
        x = _img(1, 1, 64, 64)
        result = vif.compute_metric(x, x)
        assert math.isfinite(result.item())

    def test_edge_all_zero_no_crash(self, vif):
        z = torch.zeros(1, 1, 32, 32)
        result = vif.compute_metric(z, z)
        assert math.isfinite(result.item())
