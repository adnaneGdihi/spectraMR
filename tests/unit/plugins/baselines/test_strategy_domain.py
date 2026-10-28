"""The adapters receive image-domain tensors whatever domain the loader serves.

All three upstreams train and sample on a complex image, while the M4Raw arms declare
``data.domain.output: kspace``. The strategy owns that conversion through
``_ensure_image_domain_target``; these tests hand it k-space and check what reaches the
adapter, and the planted violation checks an image-domain loader is left untouched.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from spectramr_baselines.strategy import UpstreamProcessStrategy

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.training.utils.metric_views import MetricViews
from tests.utils.data_config_stub import DataConfigStub

_SIZE = 32


class _RecordingAdapter(torch.nn.Module):
    """Keeps the tensors the strategy hands over."""

    UPSTREAM_LOSS_FAMILY = "l1"

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.seen: dict[str, torch.Tensor] = {}

    def training_loss(self, x_0, batch=None):
        self.seen["train_target"] = x_0
        return (self.weight * x_0).abs().mean()

    def validation_sample(self, input_batch, target_batch, batch=None):
        self.seen["val_input"] = input_batch
        self.seen["val_target"] = target_batch
        return target_batch


def _strategy(adapter: _RecordingAdapter, output_domain: str) -> UpstreamProcessStrategy:
    strategy = UpstreamProcessStrategy.__new__(UpstreamProcessStrategy)
    strategy.config = SimpleNamespace(
        data=DataConfigStub(
            dataset_type="m4raw", output_domain=output_domain, normalize_kspace=False
        ),
        model=None,
        physics=None,
        losses=None,
        training=SimpleNamespace(output_dir=None),
        validation=SimpleNamespace(),
    )
    strategy.env = SimpleNamespace(generator=adapter)
    strategy._metric_transform_views = lambda pred, target, cfg: MetricViews(pred, target)
    return strategy


def _image_and_kspace() -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.complex(torch.randn(2, 1, _SIZE, _SIZE), torch.randn(2, 1, _SIZE, _SIZE))
    kspace = fft2c(image)
    as_stack = lambda z: torch.cat([z.real, z.imag], dim=1)  # noqa: E731
    return as_stack(image), as_stack(kspace)


def test_a_kspace_training_target_reaches_the_adapter_as_an_image() -> None:
    adapter = _RecordingAdapter()
    image, kspace = _image_and_kspace()

    _strategy(adapter, "kspace")._compute_losses_impl(torch.zeros_like(kspace), kspace, epoch=0)

    torch.testing.assert_close(adapter.seen["train_target"], image, atol=1e-5, rtol=1e-5)


def test_validation_converts_both_input_and_target(monkeypatch) -> None:
    adapter = _RecordingAdapter()
    image, kspace = _image_and_kspace()
    strategy = _strategy(adapter, "kspace")
    monkeypatch.setattr(
        type(strategy),
        "validation_metrics_computer",
        property(lambda self: SimpleNamespace(compute=lambda pred, target: {"val_psnr": 0.0})),
    )

    strategy.validation_step(kspace.clone(), kspace)

    torch.testing.assert_close(adapter.seen["val_input"], image, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(adapter.seen["val_target"], image, atol=1e-5, rtol=1e-5)


def test_validation_caches_the_scored_pair_for_the_image_logger(monkeypatch) -> None:
    """The logger would otherwise render one denoiser pass on k-space, not the sample."""
    adapter = _RecordingAdapter()
    image, kspace = _image_and_kspace()
    strategy = _strategy(adapter, "kspace")
    monkeypatch.setattr(
        type(strategy),
        "validation_metrics_computer",
        property(lambda self: SimpleNamespace(compute=lambda pred, target: {"val_psnr": 0.0})),
    )

    strategy.validation_step(kspace.clone(), kspace)

    torch.testing.assert_close(strategy._last_visual_pred, image, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(strategy._last_visual_target, image, atol=1e-5, rtol=1e-5)


def test_an_image_domain_loader_is_passed_through_unchanged() -> None:
    """PLANTED VIOLATION: converting unconditionally would IFFT an image a second time."""
    adapter = _RecordingAdapter()
    image, _ = _image_and_kspace()

    _strategy(adapter, "image")._compute_losses_impl(torch.zeros_like(image), image, epoch=0)

    assert torch.equal(adapter.seen["train_target"], image)
