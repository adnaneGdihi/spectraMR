"""Tests for the data-calibrated LDM latent scale (``latent_scaling_factor: auto``)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.latent_scale_calibration import (
    AUTO_LATENT_SCALE,
    global_latent_std,
    parse_latent_scaling_factor,
)


class TestParse:
    def test_absent_is_none(self):
        assert parse_latent_scaling_factor(None) is None

    @pytest.mark.parametrize("raw", ["auto", "AUTO", " auto "])
    def test_auto(self, raw):
        assert parse_latent_scaling_factor(raw) == AUTO_LATENT_SCALE

    def test_a_number_passes_through(self):
        assert parse_latent_scaling_factor(2.45) == pytest.approx(2.45)

    @pytest.mark.parametrize("raw", [0.0, -1.0])
    def test_non_positive_raises(self, raw):
        with pytest.raises(ValueError, match="> 0"):
            parse_latent_scaling_factor(raw)

    def test_an_unknown_string_raises(self):
        with pytest.raises(ValueError, match="'auto'"):
            parse_latent_scaling_factor("calibrate")


class TestGlobalLatentStd:
    def test_matches_torch_std_on_one_rank(self):
        mu = torch.randn(4, 4, 8, 8) * 0.37 + 0.2
        expected = mu.std(unbiased=False)
        assert float(global_latent_std(mu)) == pytest.approx(float(expected), rel=1e-5)

    def test_it_is_the_spread_of_mu_not_a_posterior_sigma(self):
        """The defect it replaces: stage 1 logged exp(0.5*logvar).mean(). A
        latent whose mu spreads by 0.5 has std 0.5 whatever its logvar."""
        mu = torch.randn(8, 4, 16, 16) * 0.5
        assert float(global_latent_std(mu)) == pytest.approx(0.5, rel=0.05)

    def test_returns_a_0d_tensor_on_the_input_device(self):
        out = global_latent_std(torch.randn(2, 4, 4, 4))
        assert out.dim() == 0
        assert out.device == torch.device("cpu")
