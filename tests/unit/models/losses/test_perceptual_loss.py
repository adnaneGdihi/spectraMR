"""``PerceptualLoss`` (``models/losses/perceptual_loss.py``): non-finite inputs are refused."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.fixture
def perceptual(monkeypatch):
    """A PerceptualLoss over a stub VGG (37 identity layers, no download)."""
    import torchvision

    stub = SimpleNamespace(features=nn.Sequential(*[nn.Identity() for _ in range(37)]))
    monkeypatch.setattr(torchvision.models, "vgg19", lambda **_: stub)
    from spectramr.models.losses.perceptual_loss import PerceptualLoss

    return PerceptualLoss()


def test_a_nan_prediction_raises_instead_of_zeroing_the_loss(perceptual) -> None:
    """Planted violation: until 2026-09-03 this returned a zero loss and the run went on."""
    x = torch.rand(1, 3, 8, 8)
    x[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite values in the prediction"):
        perceptual(x, torch.rand(1, 3, 8, 8))


def test_an_inf_target_raises(perceptual) -> None:
    y = torch.rand(1, 3, 8, 8)
    y[0, 1, 2, 3] = float("inf")
    with pytest.raises(ValueError, match="non-finite values in the target"):
        perceptual(torch.rand(1, 3, 8, 8), y)


def test_finite_inputs_give_a_finite_loss(perceptual) -> None:
    loss = perceptual(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8))
    assert torch.isfinite(loss).all()


# ---------------------------------------------------------------------------
# A stub VGG with the real ReLU positions, so layer indices mean what they mean
# in VGG19 and every layer changes its input.
# ---------------------------------------------------------------------------


def _stub_vgg_features(seed: int = 0) -> nn.Sequential:
    from spectramr.models.losses.perceptual_loss import VGG19_RELU_INDICES

    torch.manual_seed(seed)
    relus = set(VGG19_RELU_INDICES.values())
    return nn.Sequential(
        *[nn.ReLU() if i in relus else nn.Conv2d(3, 3, 3, padding=1) for i in range(37)]
    )


@pytest.fixture
def stub_vgg(monkeypatch) -> nn.Sequential:
    import torchvision

    features = _stub_vgg_features()
    monkeypatch.setattr(torchvision.models, "vgg19", lambda **_: SimpleNamespace(features=features))
    return features


def _reference_features(features: nn.Sequential, x: torch.Tensor) -> list[torch.Tensor]:
    """Every layer's output, from one full pass."""
    outs = []
    for layer in features:
        x = layer(x)
        outs.append(x)
    return outs


def test_rss_magnitude_matches_the_interleaved_complex_modulus() -> None:
    from spectramr.models.losses.perceptual_loss import rss_magnitude

    x = torch.randn(2, 4, 5, 6)
    z = torch.complex(x[:, 0::2], x[:, 1::2])
    expected = z.abs().pow(2).sum(dim=1, keepdim=True).sqrt()
    torch.testing.assert_close(rss_magnitude(x), expected)
    torch.testing.assert_close(rss_magnitude(z), expected)


def test_a_zero_complex_pixel_has_a_finite_gradient(stub_vgg) -> None:
    """``sqrt(sum |z|^2)`` gave NaN at an all-zero (Re, Im) pixel of a 2-channel input."""
    from spectramr.models.losses.perceptual_loss import PerceptualLoss

    x = torch.rand(1, 2, 8, 8)
    x[:, :, :2, :2] = 0.0
    x.requires_grad_(True)
    PerceptualLoss()(x, torch.rand(1, 2, 8, 8)).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_an_unknown_layer_name_raises(stub_vgg) -> None:
    """A misspelt layer used to contribute nothing, so the term trained at zero."""
    from spectramr.models.losses.perceptual_loss import PerceptualLoss, VGGFeatureExtractor

    with pytest.raises(ValueError, match="relu5_5"):
        PerceptualLoss(layer_weights={"relu5_5": 1.0})
    with pytest.raises(ValueError, match="relu9_9"):
        VGGFeatureExtractor(layers=["relu9_9"])


def test_extraction_stops_at_the_deepest_requested_layer(stub_vgg) -> None:
    """Same value as a full pass, with VGG run only as far as ``relu2_2`` (index 8)."""
    from spectramr.models.losses.perceptual_loss import PerceptualLoss

    loss_fn = PerceptualLoss(layer_weights={"relu1_2": 0.5, "relu2_2": 1.0}, use_input_norm=False)
    calls: list[int] = []
    for layer in stub_vgg:
        layer.register_forward_hook(lambda *_: calls.append(1))
    x, y = torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8)
    with torch.no_grad():
        got = loss_fn(x, y)
        calls_per_branch = len(calls) // 2
        fx = _reference_features(stub_vgg, x.repeat(1, 3, 1, 1))
        fy = _reference_features(stub_vgg, y.repeat(1, 3, 1, 1))
    want = 0.5 * (fx[3] - fy[3]).abs().mean() + (fx[8] - fy[8]).abs().mean()
    torch.testing.assert_close(got, want)
    assert calls_per_branch == 9


def test_forward_to_layers_serves_several_layers_in_one_pass(stub_vgg) -> None:
    """The old loop restarted at layer 0 for every name, feeding ``relu1_1``'s
    output back through ``conv1_1`` before reading ``relu2_1``."""
    from spectramr.models.losses.perceptual_loss import PerceptualLoss

    loss_fn = PerceptualLoss(layer_weights={"relu1_1": 1.0, "relu2_1": 1.0}, use_input_norm=False)
    x = torch.rand(1, 3, 8, 8)
    got = loss_fn.forward_to_layers(x)
    with torch.no_grad():
        ref = _reference_features(stub_vgg, x)
    torch.testing.assert_close(got["relu1_1"], ref[1])
    torch.testing.assert_close(got["relu2_1"], ref[6])


def test_feature_extractor_stops_early_and_matches_a_full_pass(stub_vgg) -> None:
    from spectramr.models.losses.perceptual_loss import VGGFeatureExtractor

    extractor = VGGFeatureExtractor(layers=["relu1_1", "relu3_1"], use_input_norm=False)
    calls: list[int] = []
    for layer in stub_vgg:
        layer.register_forward_hook(lambda *_: calls.append(1))
    x = torch.rand(1, 3, 8, 8)
    with torch.no_grad():
        got = extractor(x)
        executed = len(calls)
        ref = _reference_features(stub_vgg, x)
    assert executed == 12
    assert len(got) == 2
    torch.testing.assert_close(got[0], ref[1])
    torch.testing.assert_close(got[1], ref[11])
