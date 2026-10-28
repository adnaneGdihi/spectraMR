"""CT-Mamba: a Mamba encoder and decoder whose step sizes differ by a fixed per-channel scale.

The ULF sequence is encoded with step scale ``delta_scale_encoder`` and decoded with
``delta_scale_decoder``. With ``delta_scale_decoder < delta_scale_encoder`` the decoder
steps finer, which is the continuous-time multi-rate claim of hm_01. Both stacks are
``DeltaMambaStack``s, so the scale multiplies Δ after softplus and cannot be
reabsorbed into the frozen Δ bias.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

from spectramr.models.blocks.delta_mamba import DeltaMambaStack
from spectramr.models.blocks.grad_checkpointing import NestedGradCheckpointingMixin
from spectramr.models.generators.hilbert_mamba_generators import (
    _ConvHead,
    _ConvStem,
    _HilbertLinearize,
    _squeeze_volume_to_2d,
    _unsqueeze_to_volume,
)
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)


def _validate_delta_scales(encoder: float, decoder: float) -> None:
    for name, value in (("delta_scale_encoder", encoder), ("delta_scale_decoder", decoder)):
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"CTMamba {name} must be a finite number > 0, got {value!r}.")
    if decoder > encoder:
        raise ValueError(
            f"CTMamba needs delta_scale_decoder <= delta_scale_encoder (the decoder steps "
            f"finer), got decoder={decoder} > encoder={encoder}."
        )


@register_model(
    name="ct_mamba",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
    strict_kwargs=True,
)
class CTMamba(NestedGradCheckpointingMixin, nn.Module):
    """Continuous-time Mamba with a coarse encoder step and a finer decoder step.

    Args:
        in_channels: Input channels (ULF).
        out_channels: Output channels (HF).
        d_model: Mamba model dimension.
        num_layers: Mamba blocks per stack.
        d_state: SSM state dimension.
        linearization_mode: Hilbert-family linearization of the image.
        delta_scale_encoder: Multiplies Δ in the encoder. Must be > 0.
        delta_scale_decoder: Multiplies Δ in the decoder. Must be > 0 and <= the encoder value.
        learnable_delta_scale: Train the per-channel scales (stored as log-scales, so they stay > 0).
        freeze_dt_bias: Freeze the Δ bias in both stacks.
        scan_backend: ``cuda`` (mamba_ssm kernel) or ``reference`` (pure-torch scan, CPU tests only).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 64,
        num_layers: int = 4,
        d_state: int = 16,
        linearization_mode: str = "hilbert_2d",
        delta_scale_encoder: float = 1.0,
        delta_scale_decoder: float = 0.5,
        learnable_delta_scale: bool = False,
        freeze_dt_bias: bool = True,
        scan_backend: str = "cuda",
    ) -> None:
        super().__init__()
        _validate_delta_scales(delta_scale_encoder, delta_scale_decoder)
        self.stem = _ConvStem(in_channels, d_model)
        self.linearize = _HilbertLinearize(mode=linearization_mode)
        self.encoder = DeltaMambaStack(d_model, num_layers, d_state, backend=scan_backend)
        self.decoder = DeltaMambaStack(d_model, num_layers, d_state, backend=scan_backend)
        d_inner = self.encoder.layers[0].d_inner
        self._register_delta_scale("encoder", delta_scale_encoder, d_inner, learnable_delta_scale)
        self._register_delta_scale("decoder", delta_scale_decoder, d_inner, learnable_delta_scale)
        if freeze_dt_bias:
            self.encoder.freeze_dt_bias()
            self.decoder.freeze_dt_bias()
        self.head = _ConvHead(d_model, out_channels)

        logger.info(
            "[CTMamba] d_model=%d, layers=%d, mode=%s, delta_scale enc=%g dec=%g learnable=%s",
            d_model,
            num_layers,
            linearization_mode,
            delta_scale_encoder,
            delta_scale_decoder,
            learnable_delta_scale,
        )

    def _register_delta_scale(self, site: str, value: float, d_inner: int, learnable: bool) -> None:
        log_scale = torch.full((d_inner,), math.log(value))
        if learnable:
            self.register_parameter(f"log_delta_scale_{site}", nn.Parameter(log_scale))
        else:
            self.register_buffer(f"log_delta_scale_{site}", log_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: ULF ``[B, C_in, H, W]`` to HF ``[B, C_out, H, W]``."""
        x, vol_shape = _squeeze_volume_to_2d(x)
        feat = self.stem(x)
        seq, shape = self.linearize(feat)  # [B, D, L]
        seq = seq.permute(0, 2, 1)  # [B, L, D]

        latent = self.encoder(seq, delta_scale=self.log_delta_scale_encoder.exp())
        decoded = self.decoder(latent, delta_scale=self.log_delta_scale_decoder.exp())

        decoded = decoded.permute(0, 2, 1)  # [B, D, L]
        out = self.linearize.reverse(decoded, shape)
        out = self.head(out) + x[:, : self.head.proj[-1].out_channels]
        return _unsqueeze_to_volume(out, vol_shape)


__all__ = ["CTMamba"]
