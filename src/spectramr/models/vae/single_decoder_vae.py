r"""Single-decoder VAE (``single_decoder_vae``): the width-matched control for ``moe_vae``.

It keeps ``moe_vae``'s encoder trunk, latent heads and decoder shape, and replaces the
top-1 router and its ``K`` expert decoders with one decoder of width ``decoder_hidden``.
Choosing ``decoder_hidden`` so that this decoder holds as many parameters as the ``K``
experts together isolates routing from capacity: the two models differ only in whether
the decoder parameters are split across experts and selected per sample.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spectramr.models.registry import register_model
from spectramr.models.vae.moe_vae import ExpertDecoder, MoEVAE, trunk_encoder

# GroupNorm(8, width // 2) in the trunk and the decoder needs width // 2 divisible by 8.
_WIDTH_MULTIPLE = 16


def _check_width(name: str, width: int) -> None:
    if width <= 0 or width % _WIDTH_MULTIPLE:
        raise ValueError(
            f"single_decoder_vae: {name}={width} must be a positive multiple of "
            f"{_WIDTH_MULTIPLE} (GroupNorm(8, {name} // 2))"
        )


@register_model(
    name="single_decoder_vae",
    training_mode="vae",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=False,
)
class SingleDecoderVAE(nn.Module):
    r"""Gaussian VAE with ``moe_vae``'s trunk and one dense decoder.

    Args:
        in_channels: Input image channels.
        out_channels: Reconstruction channels.
        latent_dim: Gaussian latent width.
        hidden: Encoder trunk width (the same knob as ``moe_vae.hidden``).
        decoder_hidden: Decoder width; ``None`` uses ``hidden``.
        pooled_size: Spatial size pooled to before the latent heads.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_dim: int = 64,
        hidden: int = 128,
        decoder_hidden: int | None = None,
        pooled_size: int = 4,
    ) -> None:
        super().__init__()
        decoder_hidden = hidden if decoder_hidden is None else int(decoder_hidden)
        _check_width("hidden", hidden)
        _check_width("decoder_hidden", decoder_hidden)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.decoder_hidden = decoder_hidden
        self.pooled_size = pooled_size
        feature_dim = hidden * pooled_size * pooled_size

        self.encoder = trunk_encoder(in_channels, hidden)
        self.pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))
        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)
        self.decoder = ExpertDecoder(latent_dim, decoder_hidden, pooled_size, out_channels)

    @property
    def name(self) -> str:
        return "single_decoder_vae"

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = torch.flatten(self.pool(self.encoder(x)), start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Return ``reconstruction`` and ``kl_override`` as ``moe_vae`` does, without ``aux_loss``.

        ``mu``/``logvar`` are left out so ``VAETrainingStrategy`` counts the KL once, through
        ``kl_override``, exactly as it does for ``moe_vae``.
        """
        mu, logvar = self.encode(x)
        z = MoEVAE.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        if x_hat.shape[-2:] != x.shape[-2:]:
            x_hat = F.interpolate(x_hat, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"reconstruction": x_hat, "kl_override": MoEVAE.kl_standard_normal(mu, logvar)}


__all__ = ["SingleDecoderVAE"]
