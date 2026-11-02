"""FNOGenerator output head: ``output_form`` complex (default) vs magnitude."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.fno_generator import FNOGenerator


def _tiny(**kw) -> FNOGenerator:
    base = {"width": 8, "modes1": 4, "modes2": 4, "depth": 1, "output_domain": "image"}
    return FNOGenerator(**{**base, **kw})


def test_magnitude_head_emits_one_nonnegative_channel():
    model = _tiny(out_channels=1, output_form="magnitude").eval()
    out = model(torch.randn(1, 2, 16, 16))
    assert out.shape == (1, 1, 16, 16)
    assert bool((out >= 0).all())


def test_complex_head_is_default_and_emits_two_channels():
    model = _tiny().eval()
    assert model.output_form == "complex"
    assert model(torch.randn(1, 2, 16, 16)).shape == (1, 2, 16, 16)


def test_unknown_output_form_raises():
    with pytest.raises(ValueError, match="output_form"):
        _tiny(out_channels=2, output_form="phase")


def test_unknown_output_domain_raises():
    with pytest.raises(ValueError, match="output_domain"):
        _tiny(output_domain="kspaec")


def test_magnitude_head_requires_out_channels_one():
    with pytest.raises(ValueError, match="out_channels"):
        _tiny(out_channels=2, output_form="magnitude")


def test_spectral_truncation_keeps_smooth_image_and_drops_high_frequency():
    from spectramr.models.blocks.fno_block import SpectralConv2d

    n = 64
    layer = SpectralConv2d(1, 1, modes1=8, modes2=8).eval()
    with torch.no_grad():
        layer.weights1.fill_(1.0)
        layer.weights2.fill_(1.0)
    cols = torch.arange(n, dtype=torch.float32) / n
    low = torch.cos(2 * torch.pi * 2 * cols).expand(n, n)[None, None]
    high = torch.cos(2 * torch.pi * 28 * cols).expand(n, n)[None, None]
    with torch.no_grad():
        assert (layer(low) - low).abs().max() < 1e-4
        assert layer(high).abs().max() < 1e-4


def test_spectral_blocks_see_image_domain_features():
    model = _tiny().eval()
    captured = []
    model.fc0.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach()))
    k = torch.randn(1, 2, 16, 16)
    model(k)
    feat = captured[0][..., :2].permute(0, 3, 1, 2)
    from spectramr.infrastructure.physics.fft_ops import fft2c

    recovered = fft2c(torch.complex(feat[:, 0], feat[:, 1]))
    assert torch.allclose(recovered.real, k[:, 0], atol=1e-4)
    assert torch.allclose(recovered.imag, k[:, 1], atol=1e-4)


def test_kspace_output_domain_is_fft_of_image_output():
    from spectramr.infrastructure.physics.fft_ops import fft2c

    k = torch.randn(1, 2, 16, 16)
    img = _tiny(output_domain="image").eval()
    ksp = _tiny(output_domain="kspace").eval()
    ksp.load_state_dict(img.state_dict())
    out_img = img(k)
    out_k = ksp(k)
    expected = fft2c(torch.complex(out_img[:, 0], out_img[:, 1]))
    assert torch.allclose(out_k[:, 0], expected.real, atol=1e-4)
    assert torch.allclose(out_k[:, 1], expected.imag, atol=1e-4)
