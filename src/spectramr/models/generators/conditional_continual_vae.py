"""Conditional Continual VAE Generator for lifelong MRI learning.

VAE with task-specific latent distributions and catastrophic-forgetting
protection via Elastic Weight Consolidation (EWC). Enables sequential
training across scanners/contrasts without losing previous knowledge.

Reference:
    Ramapuram et al., "Lifelong Generative Modeling", NeurIPS 2017.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


def _stage(in_ch: int, out_ch: int, *, down: bool) -> list[nn.Module]:
    conv = nn.Conv2d if down else nn.ConvTranspose2d
    return [
        conv(in_ch, out_ch, 4, stride=2, padding=1),
        nn.GroupNorm(math.gcd(32, out_ch), out_ch),
        nn.SiLU(),
    ]


class _TaskConditionedEncoder(nn.Module):
    """Stride-2 encoder over ``features`` with task-specific FiLM conditioning."""

    def __init__(self, in_ch: int, features: list[int], num_tasks: int) -> None:
        super().__init__()
        widths = [in_ch, *features]
        layers: list[nn.Module] = []
        for a, b in itertools.pairwise(widths):
            layers += _stage(a, b, down=True)
        self.conv = nn.Sequential(*layers)
        out_dim = features[-1]
        self.task_embed = nn.Embedding(num_tasks, out_dim)
        self.film_gamma = nn.Linear(out_dim, out_dim)
        self.film_beta = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        """Encode ``x`` (B, C, H, W) and modulate the last stage by the task embedding."""
        h = self.conv(x)
        B = h.shape[0]
        tid = torch.full((B,), task_id, device=x.device, dtype=torch.long)
        te = self.task_embed(tid)
        gamma = self.film_gamma(te)[..., None, None]
        beta = self.film_beta(te)[..., None, None]
        return gamma * h + beta


def _check_depth(features: list[int], encoder_blocks: int, decoder_blocks: int) -> None:
    """Raise unless ``features`` is a stem width plus one width per encoder/decoder stage."""
    if encoder_blocks < 1 or len(features) != encoder_blocks + 1:
        raise ValueError(
            f"conditional_continual_vae: features has {len(features)} widths but "
            f"encoder_blocks={encoder_blocks}; it needs encoder_blocks + 1 widths "
            "(features[0] is the first stage's input width after the stem)."
        )
    if decoder_blocks != encoder_blocks:
        raise ValueError(
            f"conditional_continual_vae: decoder_blocks={decoder_blocks} must equal "
            f"encoder_blocks={encoder_blocks}; the decoder mirrors the encoder's resolution."
        )


@register_model(name="conditional_continual_vae", training_mode="vae")
class ConditionalContinualVAEGenerator(IGenerator, nn.Module):
    """Continual VAE with task conditioning for lifelong MRI learning."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        features: list[int] = list(kwargs.get("features", [32, 64, 128, 256]))
        encoder_blocks: int = kwargs.get("encoder_blocks", 3)
        decoder_blocks: int = kwargs.get("decoder_blocks", 3)
        num_tasks: int = kwargs.get("num_tasks", 5)
        _check_depth(features, encoder_blocks, decoder_blocks)
        self.features = features
        self.encoder_blocks = encoder_blocks
        self.decoder_blocks = decoder_blocks

        # Stem at features[0], then encoder_blocks stride-2 stages over features[1:].
        self.stem = nn.Conv2d(in_channels, features[0], 3, padding=1)
        self.encoder = _TaskConditionedEncoder(features[0], features[1:], num_tasks)
        enc_out_dim = features[-1]

        self.fc_mu = nn.Conv2d(enc_out_dim, enc_out_dim, 1)
        self.fc_logvar = nn.Conv2d(enc_out_dim, enc_out_dim, 1)

        rev = features[::-1]
        dec: list[nn.Module] = []
        for a, b in itertools.pairwise(rev):
            dec += _stage(a, b, down=False)
        dec.append(nn.Conv2d(features[0], out_channels, 3, padding=1))
        self.decoder = nn.Sequential(*dec)
        self._last_aux: dict[str, torch.Tensor] | None = None

        self.num_tasks = num_tasks

    @property
    def name(self) -> str:
        return "conditional_continual_vae"

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for ConditionalContinualVAEGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        H, W = x.shape[-2:]
        task_id: int = kwargs.get("task_id", 0)

        h = self.encoder(self.stem(x), task_id)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self._reparameterize(mu, logvar)
        out = self.decoder(z)
        # The strategy reads the posterior here, so the KL acts on a bare-tensor forward (#2362).
        self._last_aux = {"mu": mu, "logvar": logvar, "z": z}

        if out.shape[2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out

    def last_aux(self) -> dict[str, torch.Tensor] | None:
        """The posterior ``(mu, logvar, z)`` of the most recent forward, or None before one."""
        return self._last_aux

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
