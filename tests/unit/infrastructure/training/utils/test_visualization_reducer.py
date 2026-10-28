"""Tests for the single raw-output → displayable reduction (#709, #390, #682).

Five render sites each grew their own copy of this chain and disagreed in ways
invisible from one figure. These pin the parts that were wrong, and — as
important — the parts that must NOT change, because the reducer now sits in
front of every preview in the tree.
"""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.infrastructure.training.utils.visualization_reducer import (
    VisualizationReducer,
    is_distribution_head,
    to_magnitude,
)


def _cfg(*, model_type="unet", log_scaling=False):
    from spectramr.config.schemas.data import DataProcessingConfigSchema

    processing = DataProcessingConfigSchema(
        enable_kspace_normalization=log_scaling, enable_log_scaling=log_scaling
    )
    return types.SimpleNamespace(
        model=types.SimpleNamespace(model_type=model_type),
        data=types.SimpleNamespace(processing=processing),
    )


class TestDistributionHeadDetection:
    """One declaration, two consumers — the width guard's and this one's.

    They asked the question separately before, and got different answers: the
    guard knew `evidential_unet` emits parameters, the visualization side did
    not, so its four channels rendered as `sqrt(Σ params²)` (#390).
    """

    def test_evidential_unet_is_one(self):
        assert is_distribution_head(_cfg(model_type="evidential_unet")) is True

    def test_a_strategy_flag_is_the_other_route(self):
        strategy = types.SimpleNamespace(predicts_distribution_params=True)
        assert is_distribution_head(_cfg(), strategy) is True

    def test_an_ordinary_model_is_not(self):
        assert is_distribution_head(_cfg(), types.SimpleNamespace()) is False

    def test_it_matches_the_width_guard_predicate(self):
        """Pins the shared declaration. If the guard's test changes and this one
        does not, the two owners are back."""
        import inspect

        from spectramr.infrastructure.training.strategies.base import (
            BaseTrainingStrategy,
        )

        src = inspect.getsource(BaseTrainingStrategy.train_step)
        assert 'model_type == "evidential_unet"' in src
        assert "predicts_distribution_params" in src


class TestPointEstimate:
    def test_a_distribution_head_reduces_to_channel_zero(self):
        """`evidential_unet` is [mean, var, alpha, beta]; the mean is the image."""
        r = VisualizationReducer(distribution_head=True)
        t = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1)

        out = r.point_estimate(t)

        assert out.shape == (1, 1, 1, 1)
        assert out[0, 0, 0, 0] == 0.0

    def test_a_complex_pair_is_left_alone(self):
        """The load-bearing non-change.

        A 2-channel real-stacked complex tensor is NOT a distribution head.
        Taking channel 0 would show the real part and call it the image; it has
        to reach `to_magnitude` intact so RSS can produce the modulus.
        """
        r = VisualizationReducer(distribution_head=False)
        t = torch.tensor([[[[3.0]], [[4.0]]]])

        assert r.point_estimate(t).shape == t.shape
        assert float(to_magnitude(r.point_estimate(t))) == pytest.approx(5.0)

    def test_a_short_tensor_is_returned_unchanged(self):
        """Never index past the end to satisfy a policy."""
        r = VisualizationReducer(distribution_head=True)
        t = torch.rand(3)
        assert r.point_estimate(t) is t


class TestUncertainty:
    def test_channel_one_is_the_variance(self):
        r = VisualizationReducer(distribution_head=True)
        t = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1)

        var = r.uncertainty(t)

        assert var is not None and var[0, 0, 0, 0] == 1.0

    def test_it_is_returned_separately_not_blended(self):
        """Variance is a different quantity in different units. Rendering it AS
        the image is #390; averaging it into the image is worse."""
        r = VisualizationReducer(distribution_head=True)
        t = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1)

        assert not torch.equal(r.point_estimate(t), r.uncertainty(t))

    def test_an_ordinary_head_has_none(self):
        assert (
            VisualizationReducer(distribution_head=False).uncertainty(
                torch.rand(1, 2, 4, 4)
            )
            is None
        )

    def test_a_one_channel_distribution_head_has_none(self):
        assert (
            VisualizationReducer(distribution_head=True).uncertainty(
                torch.rand(1, 1, 4, 4)
            )
            is None
        )


class TestToMagnitude:
    def test_complex_becomes_modulus(self):
        t = torch.complex(torch.tensor([[[[3.0]]]]), torch.tensor([[[[4.0]]]]))
        assert float(to_magnitude(t)) == pytest.approx(5.0)

    def test_multichannel_real_becomes_rss(self):
        t = torch.tensor([[[[3.0]], [[4.0]]]])
        assert float(to_magnitude(t)) == pytest.approx(5.0, abs=1e-3)

    def test_single_channel_passes_through(self):
        t = torch.rand(1, 1, 4, 4)
        assert torch.equal(to_magnitude(t), t)

    def test_output_is_one_channel(self):
        assert to_magnitude(torch.rand(2, 6, 8, 8)).shape[1] == 1


class TestToDisplayOrdering:
    """The order is the whole point; each step is wrong before the one above it."""

    def test_decompression_happens_before_the_transform(self, monkeypatch):
        """`log1p` is a per-bin nonlinearity, so the IFFT of a compressed
        spectrum is not a scaled image (#682)."""
        calls = []
        import spectramr.infrastructure.training.utils.visualization_reducer as vr

        monkeypatch.setattr(
            vr, "decompress_for_view", lambda t, **kw: (calls.append("expm1"), t)[1]
        )
        monkeypatch.setattr(
            "spectramr.infrastructure.physics.fft_ops.ifft2c",
            lambda t: (calls.append("ifft"), t)[1],
        )

        VisualizationReducer(log_scaled=True).to_display(
            torch.rand(1, 2, 8, 8), in_kspace=True
        )

        assert calls == ["expm1", "ifft"]

    def test_image_domain_skips_both(self, monkeypatch):
        """A guessed domain produces a plausible-but-wrong picture, so the
        caller declares it and an image-domain tensor is never transformed."""
        import spectramr.infrastructure.training.utils.visualization_reducer as vr

        monkeypatch.setattr(
            vr,
            "decompress_for_view",
            lambda t, **kw: pytest.fail("decompressed an image-domain tensor"),
        )

        out = VisualizationReducer(log_scaled=True).to_display(
            torch.rand(1, 2, 8, 8), in_kspace=False
        )

        assert out.shape[1] == 1

    def test_the_result_is_not_windowed(self):
        """Windowing belongs to the writer. Two owners is what produced the
        double-windowing, where one tensor came out at two contrasts."""
        t = torch.rand(1, 1, 8, 8) * 100.0
        out = VisualizationReducer().to_display(t, in_kspace=False)
        assert float(out.max()) > 1.0, "reducer must not normalise to [0, 1]"


class TestTheMixinDefaultUsesIt:
    """Anti-facade: a reducer nothing calls is exactly pitfall #16."""

    def test_the_hook_delegates(self):
        import inspect

        from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
            MetricsMixin,
        )

        src = inspect.getsource(MetricsMixin._prediction_for_visualization)
        assert "VisualizationReducer" in src
        assert "point_estimate" in src

    def test_an_evidential_config_now_reduces(self):
        """The #390 case, end to end through the public hook."""
        from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
            MetricsMixin,
        )

        obj = MetricsMixin.__new__(MetricsMixin)
        obj.config = _cfg(model_type="evidential_unet")
        t = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1)

        assert MetricsMixin._prediction_for_visualization(obj, t).shape == (1, 1, 1, 1)

    def test_an_ordinary_config_is_still_identity(self):
        """Regression guard for every non-distribution arm in the tree."""
        from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
            MetricsMixin,
        )

        obj = MetricsMixin.__new__(MetricsMixin)
        obj.config = _cfg()
        t = torch.rand(1, 2, 4, 4)

        assert torch.equal(MetricsMixin._prediction_for_visualization(obj, t), t)
