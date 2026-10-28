"""LPIPS must reject non-finite inputs at the metric boundary.

Regression for the 2026-06-24 ``exp_hm_06`` / ``exp_hm_10`` crash: a diverged
model emitted NaN predictions, the LPIPS normaliser's ``torch.clamp`` passed
them through, and the failure surfaced inside torchmetrics as a misleading
*"values in range [nan, nan] … expected [-1, 1]"* error. LPIPS must now raise a
precise, finite-guard error (naming the model as the cause) **before** reaching
the VGG backbone — so the test does not need torchmetrics installed.
"""

import pytest
import torch

from spectramr.core.metrics.evaluation_metrics import LPIPS


class _ExplodingBackbone(torch.nn.Module):
    """Stand-in for the LPIPS backbone — fails loudly if ever called.

    An ``nn.Module`` so it can be assigned to the metric's registered ``lpips``
    submodule (the real torchmetrics VGG / lpips-package AlexNet backends are both
    ``nn.Module``s, so ``metric.lpips`` is a child module that rejects non-Modules).
    """

    def forward(self, *_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("LPIPS backbone must not be reached on NaN input")

    @property
    def device(self):
        return torch.device("cpu")


class TestLpipsFiniteGuard:
    @pytest.mark.unit
    def test_lpips_raises_on_nan_pred_before_backbone(self):
        metric = LPIPS(device="cpu")
        # Force the non-torchmetrics CI path to still exercise compute_metric:
        # a truthy backbone that explodes if the guard fails to short-circuit.
        metric.lpips = _ExplodingBackbone()

        preds = torch.randn(1, 1, 16, 16)
        preds[0, 0, 0, 0] = float("nan")
        target = torch.randn(1, 1, 16, 16)

        with pytest.raises(ValueError, match="non-finite"):
            metric.compute_metric(preds, target)

    @pytest.mark.unit
    def test_lpips_error_attributes_failure_to_the_model(self):
        metric = LPIPS(device="cpu")
        metric.lpips = _ExplodingBackbone()
        preds = torch.full((1, 1, 16, 16), float("inf"))
        target = torch.randn(1, 1, 16, 16)
        with pytest.raises(ValueError) as exc:
            metric.compute_metric(preds, target)
        assert "lpips" in str(exc.value)
        assert "model" in str(exc.value).lower()


class _MeanBackbone(torch.nn.Module):
    """Per-image score = that image's mean; records the batch of every call."""

    def __init__(self):
        super().__init__()
        self.batches: list[int] = []

    def forward(self, preds, target):
        self.batches.append(preds.shape[0])
        return (preds - target).abs().mean(dim=(1, 2, 3))


class TestLpipsChunking:
    @pytest.mark.unit
    def test_chunked_mean_equals_whole_batch_and_bounds_each_call(self, monkeypatch):
        metric = LPIPS(device="cpu")
        metric.lpips = _MeanBackbone()
        metric._backend = None
        preds = torch.rand(10, 1, 16, 16)
        target = torch.rand(10, 1, 16, 16)

        monkeypatch.setattr("spectramr.core.metrics.evaluation_metrics.METRIC_CHUNK_IMAGES", 4)
        got = metric.compute_metric(preds, target)

        assert max(metric.lpips.batches) == 4 and sum(metric.lpips.batches) == 10
        from spectramr.core.metrics.channel_adapter import adapt_to_rgb

        p = adapt_to_rgb(preds, mode=metric.channel_mode).mul(2).sub(1).clamp(-1, 1)
        t = adapt_to_rgb(target, mode=metric.channel_mode).mul(2).sub(1).clamp(-1, 1)
        torch.testing.assert_close(got, (p - t).abs().mean(dim=(1, 2, 3)).mean())


class TestImageChunks:
    @pytest.mark.unit
    def test_chunks_are_aligned_bounded_and_cover_every_image(self, monkeypatch):
        from spectramr.core.metrics import evaluation_metrics as em

        monkeypatch.setattr(em, "METRIC_CHUNK_IMAGES", 3)
        a, b = torch.arange(7.0), torch.arange(7.0) * 2
        pieces = list(em._image_chunks(a, b))
        assert [len(x) for x, _ in pieces] == [3, 3, 1]
        torch.testing.assert_close(torch.cat([y for _, y in pieces]), b)

    @pytest.mark.unit
    def test_medfid_features_equal_unchunked_and_keep_row_order(self, monkeypatch):
        from spectramr.core.metrics import evaluation_metrics as em

        monkeypatch.setattr(em, "METRIC_CHUNK_IMAGES", 4)
        metric = em.MedFID.__new__(em.MedFID)
        torch.nn.Module.__init__(metric)
        metric._model_available = True
        metric.model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        x = torch.rand(10, 1, 20, 20)
        got = metric.extract_features(x)
        resized = torch.nn.functional.interpolate(x, size=(299, 299), mode="bilinear")
        torch.testing.assert_close(got, metric.model(resized.repeat(1, 3, 1, 1)))


class TestFidCountsEachBatchOnce:
    @pytest.mark.unit
    def test_compute_metric_feeds_each_image_to_the_state_once(self, monkeypatch):
        from spectramr.core.metrics import evaluation_metrics as em

        class _Recorder:
            device = torch.device("cpu")

            def __init__(self):
                self.real, self.fake = 0, 0

            def update(self, x, real):
                if real:
                    self.real += x.shape[0]
                else:
                    self.fake += x.shape[0]

        monkeypatch.setattr(em, "METRIC_CHUNK_IMAGES", 2)
        metric = em.FID.__new__(em.FID)
        torch.nn.Module.__init__(metric)
        metric.device = torch.device("cpu")
        metric.use_torchmetrics = True
        metric.channel_mode = "auto"
        metric.fid = _Recorder()
        metric.compute_metric(torch.rand(5, 1, 8, 8), torch.rand(5, 1, 8, 8))
        assert (metric.fid.real, metric.fid.fake) == (5, 5)
