"""Regression tests for ``spectramr.models.vae.wavelet_kan_vae``.

The registered ``vae_wavelet_kan_vae`` headline mechanism is the wavelet
KAN convolution. The module used to guard its import with ``try/except
ImportError`` that defined a MOCK plain-``nn.Conv2d`` stand-in (even
swallowing ``wavelet_type``) — and because the guarded import targeted a
name (``WavKANConv2D``) that ``wav_kan.py`` never exported, the mock path
ALWAYS fired: the registered model was a plain conv VAE from day one
(pitfall #9/#16). These tests are the mechanism-fires probe: the real
``WavKANConv2DLayer`` must be importable and present in the module tree.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.layers.kan.kan_convs.wav_kan import (  # noqa: E402
    WavKANConv2DLayer,
)
from spectramr.models.vae.wavelet_kan_vae import WaveletKANVAE  # noqa: E402


class TestWaveletKANVAEMechanismFires:
    def test_no_mock_conv_module_level_fallback(self) -> None:
        """The module must not define its own WavKANConv2D mock class."""
        import spectramr.models.vae.wavelet_kan_vae as mod

        assert not hasattr(mod, "WavKANConv2D"), (
            "mock WavKANConv2D fallback reintroduced — the real layer is "
            "WavKANConv2DLayer from kan_convs.wav_kan"
        )

    def test_encoder_and_decoder_use_real_wavelet_kan(self) -> None:
        """All four conv stages are real WavKANConv2DLayer instances."""
        vae = WaveletKANVAE(in_channels=1, latent_dim=16)
        wav_layers = [
            m for m in vae.modules() if isinstance(m, WavKANConv2DLayer)
        ]
        assert len(wav_layers) == 4
        # wavelet_type must be threaded through, not swallowed
        for layer in wav_layers:
            assert layer.wavelet_type == "daubechies"
        # And no bare nn.Conv2d imposters at the top of encoder/decoder
        assert isinstance(vae.encoder[0], WavKANConv2DLayer)
        assert isinstance(vae.decoder_layers[0], WavKANConv2DLayer)

    def test_forward_finite_and_shapes(self) -> None:
        torch.manual_seed(0)
        vae = WaveletKANVAE(in_channels=1, latent_dim=16)
        x = torch.randn(2, 1, 64, 64)
        recon, mu, logvar = vae(x)
        assert recon.shape == (2, 1, 64, 64)
        assert mu.shape == (2, 16)
        assert logvar.shape == (2, 16)
        assert torch.isfinite(recon).all()
        assert torch.isfinite(mu).all()
        assert torch.isfinite(logvar).all()

    def test_unsupported_wavelet_type_raises(self) -> None:
        """The real layer rejects unknown wavelet types (no silent default)."""
        with pytest.raises((AssertionError, ValueError)):
            WavKANConv2DLayer(1, 8, kernel_size=3, padding=1, wavelet_type="db2")


class TestExperiment24RunsTheWaveletModel:
    """``wavelet_kan_vae`` is a registry alias of the plain KANVAEGenerator, which has no
    wavelet layer; experiment_24 once named it and trained a plain KAN VAE."""

    _YAML = "experiments/inprogress/vae_latent/experiment_24_wavelet_fourier_kan_vae.yaml"

    def _model_block(self) -> dict:
        from pathlib import Path

        import yaml

        root = Path(__file__).resolve().parents[4]
        return yaml.safe_load((root / self._YAML).read_text())["model"]

    def test_the_arm_resolves_to_the_wavelet_class(self) -> None:
        from spectramr.models.init_registry import populate_model_registry
        from spectramr.models.registry import get_model_class

        populate_model_registry()
        assert get_model_class(self._model_block()["model_type"]) is WaveletKANVAE

    def test_every_declared_model_kwarg_is_a_parameter_the_constructor_reads(self) -> None:
        """The constructor swallows ``**kwargs``, so an unread key would be discarded silently."""
        import inspect

        declared = set(self._model_block().get("model_kwargs") or {})
        named = set(inspect.signature(WaveletKANVAE.__init__).parameters) - {"self", "kwargs"}
        assert declared <= named, f"unread model_kwargs: {sorted(declared - named)}"
