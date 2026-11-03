"""Affine-coupling normalizing flow over a vector, or over a frozen AutoencoderKL latent.

With ``paired_vae: true`` the flow models the flattened latent of a frozen stage-1
AutoencoderKL (``autoencoder_kl_pretrain``), loaded by the training builder from
``vae_checkpoint_path``. ``decode_from_latent`` maps a flow sample back to image
space, so validation scores images rather than latent vectors.

With ``conditioning: input_latent`` the coupling networks read an embedding of the
input image's latent, giving a conditional density p(z_target | z_input) trained by
exact NLL (SRFlow-style; Lugmayr et al., ECCV 2020). ``forward(input)`` returns the
decoded push-forward of the base mode (temperature 0) as the reconstruction.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any, ClassVar

import torch
from torch import nn

from spectramr.models.generators.latent_diffusion_generator import (
    AutoencoderKL,
    load_autoencoder_state_into,
)
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)

CONDITIONING_MODES = ("none", "input_latent")


class FlowBlock(nn.Module):
    """RealNVP affine coupling: ``x2 -> x2 * exp(s(x1, c)) + t(x1, c)``, ``x1`` unchanged."""

    def __init__(self, dim: int, hidden_dim: int = 256, cond_dim: int = 0):
        super().__init__()
        if dim < 2 or dim % 2:
            raise ValueError(f"FlowBlock needs an even latent_dim >= 2; got {dim}.")
        self.split_dim = dim // 2
        self.net = nn.Sequential(
            nn.Linear(self.split_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def _scale_shift(
        self, x1: torch.Tensor, cond: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = x1 if cond is None else torch.cat([x1, cond], dim=1)
        scale, shift = self.net(h).chunk(2, dim=1)
        return torch.tanh(scale), shift

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(y, log|det dy/dx|)``."""
        x1, x2 = x[:, : self.split_dim], x[:, self.split_dim :]
        scale, shift = self._scale_shift(x1, cond)
        return torch.cat([x1, x2 * torch.exp(scale) + shift], dim=1), scale.sum(dim=1)

    def inverse(
        self, y: torch.Tensor, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x, log|det dx/dy|)``."""
        y1, y2 = y[:, : self.split_dim], y[:, self.split_dim :]
        scale, shift = self._scale_shift(y1, cond)
        return torch.cat([y1, (y2 - shift) * torch.exp(-scale)], dim=1), -scale.sum(dim=1)


@register_model(name="latent_flow", training_mode="generative")
class LatentFlowGenerator(nn.Module, IGenerator):
    """Coupling flow; optionally over a frozen paired VAE latent and conditioned on the input.

    A dimension reversal follows every coupling block (log-det 0), so each half of the
    latent is transformed by alternate blocks.
    """

    #: Read by the audit's advertised-options check, so an illegal value fails at load.
    OPTION_SCHEMA: ClassVar[dict[str, list[Any]]] = {
        "conditioning": list(CONDITIONING_MODES),
        "paired_vae": [True, False],
        "freeze_vae": [True],
    }

    @property
    def name(self) -> str:
        return "latent_flow"

    def __init__(
        self,
        latent_dim: int = 256,
        in_channels: int = 1,
        out_channels: int = 1,
        flow_layers: int = 8,
        hidden_dim: int = 512,
        conditioning: str = "none",
        paired_vae: bool = False,
        latent_channels: int = 4,
        base_channels: int = 64,
        vae_checkpoint_path: str | None = None,
        freeze_vae: bool = True,
    ):
        super().__init__()
        if conditioning not in CONDITIONING_MODES:
            raise ValueError(
                f"latent_flow conditioning={conditioning!r}; expected one of {CONDITIONING_MODES}."
            )
        if not isinstance(paired_vae, bool):
            raise ValueError(f"latent_flow paired_vae must be true/false; got {paired_vae!r}.")
        if conditioning == "input_latent" and not paired_vae:
            raise ValueError("latent_flow conditioning='input_latent' needs paired_vae: true.")
        # The builder reads both keys and loads the VAE (raising on a missing file).
        if paired_vae and not vae_checkpoint_path:
            raise ValueError(
                "latent_flow paired_vae: true needs vae_checkpoint_path (stage-1 VAE)."
            )
        if freeze_vae is not True:
            raise ValueError(
                "latent_flow trains an NLL on the VAE latent; an unfrozen encoder can shrink "
                "the latent to lower that NLL. Set freeze_vae: true."
            )
        self.latent_dim = int(latent_dim)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conditioning = conditioning
        self.paired_vae = paired_vae
        self.latent_channels = int(latent_channels)
        #: Read by GenerativeTrainingStrategy: log_prob takes the input as its condition.
        self.conditions_on_input = conditioning == "input_latent"

        self.autoencoder: AutoencoderKL | None = None
        if paired_vae:
            side = math.isqrt(self.latent_dim // max(self.latent_channels, 1))
            if self.latent_channels * side * side != self.latent_dim:
                raise ValueError(
                    f"latent_flow latent_dim={self.latent_dim} is not latent_channels "
                    f"({self.latent_channels}) x a square latent grid."
                )
            self._latent_side = side
            self.autoencoder = AutoencoderKL(
                in_channels=in_channels,
                out_channels=out_channels,
                latent_channels=self.latent_channels,
                base_channels=base_channels,
            )

        cond_dim = hidden_dim if self.conditions_on_input else 0
        self.cond_net = (
            nn.Sequential(nn.Linear(self.latent_dim, hidden_dim), nn.ReLU())
            if self.conditions_on_input
            else None
        )
        self.flow_blocks = nn.ModuleList(
            [FlowBlock(self.latent_dim, hidden_dim, cond_dim) for _ in range(flow_layers)]
        )
        logger.info(
            "[latent_flow] latent_dim=%d flow_layers=%d paired_vae=%s conditioning=%s",
            self.latent_dim,
            flow_layers,
            paired_vae,
            conditioning,
        )

    # ── paired VAE seam ──────────────────────────────────────────────────
    def load_autoencoder_state(
        self, state: Mapping[str, torch.Tensor], *, freeze: bool, source: str
    ) -> None:
        """Load the stage-1 AutoencoderKL, frozen (``freeze_vae`` is refused at construction)."""
        del freeze
        if self.autoencoder is None:
            raise ValueError("latent_flow has no paired VAE (paired_vae: false) to load into.")
        load_autoencoder_state_into(self.autoencoder, state, freeze=True, source=source)

    def _require_vae(self) -> AutoencoderKL:
        if self.autoencoder is None:
            raise ValueError("latent_flow paired_vae is false; there is no VAE to encode/decode.")
        return self.autoencoder

    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Image ``[B, C, H, W]`` -> flattened posterior-mean latent ``[B, latent_dim]``."""
        vae = self._require_vae()
        with torch.no_grad():
            mean, _ = vae.encode(x)
        z = mean.flatten(1)
        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent_flow: the VAE latent of a {tuple(x.shape[-2:])} image has "
                f"{z.shape[1]} values; latent_dim is {self.latent_dim}."
            )
        return z

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Flattened latent ``[B, latent_dim]`` -> image through the frozen decoder."""
        vae = self._require_vae()
        side = self._latent_side
        return vae.decode(z.reshape(z.shape[0], self.latent_channels, side, side))

    def train(self, mode: bool = True) -> LatentFlowGenerator:
        """Keep the frozen VAE in eval mode while the flow trains."""
        super().train(mode)
        if self.autoencoder is not None:
            self.autoencoder.eval()
        return self

    # ── flow ─────────────────────────────────────────────────────────────
    def _as_vector(self, x: torch.Tensor) -> torch.Tensor:
        if self.paired_vae:
            return self.encode_to_latent(x)
        z = x.flatten(1)
        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent_flow (paired_vae: false) got {z.shape[1]} values per sample; "
                f"latent_dim is {self.latent_dim}. Set paired_vae: true to model images."
            )
        return z

    def _context(self, condition: torch.Tensor | None) -> torch.Tensor | None:
        if self.cond_net is None:
            return None
        if condition is None:
            raise ValueError("latent_flow conditioning='input_latent' needs the input image.")
        return self.cond_net(self.encode_to_latent(condition))

    def flow_forward(
        self, z: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Base -> data: ``(x, log|det dx/dz|)``."""
        log_det = z.new_zeros(z.shape[0])
        for block in self.flow_blocks:
            z, ld = block(z, context)
            z = z.flip(1)
            log_det = log_det + ld
        return z, log_det

    def flow_inverse(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Data -> base: ``(z, log|det dz/dx|)``, accumulated in one inverse pass."""
        log_det = x.new_zeros(x.shape[0])
        for block in reversed(self.flow_blocks):
            x, ld = block.inverse(x.flip(1), context)
            log_det = log_det + ld
        return x, log_det

    def log_prob(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Exact ``log p(x [| condition])`` per sample, in nats of the flow's vector space."""
        z_base, log_det = self.flow_inverse(self._as_vector(x), self._context(condition))
        log_pz = -0.5 * (z_base**2).sum(dim=1) - 0.5 * self.latent_dim * math.log(2 * math.pi)
        return log_pz + log_det

    def sample(
        self,
        batch_size: int = 1,
        condition: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Draw ``z ~ N(0, temperature^2 I)``, push it through the flow, decode when paired."""
        device = next(self.parameters()).device
        context = self._context(condition)
        if context is not None:
            batch_size = context.shape[0]
        z = torch.randn(batch_size, self.latent_dim, device=device) * temperature
        x, _ = self.flow_forward(z, context)
        return self.decode_from_latent(x) if self.paired_vae else x

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Conditional: decoded base-mode reconstruction of ``x``. Paired: a decoded sample
        per row. Unpaired: the push-forward (base -> data) of the vector ``x``."""
        if self.conditions_on_input:
            return self.sample(condition=x, temperature=0.0)
        if self.paired_vae:
            return self.sample(batch_size=x.shape[0])
        out, _ = self.flow_forward(self._as_vector(x), self._context(condition))
        return out.view(x.shape)

    def generate(self, noise: torch.Tensor | None = None, **kwargs: object) -> torch.Tensor:
        """IGenerator alias of :meth:`sample`; one sample per row of ``noise``."""
        return self.sample(1 if noise is None else noise.shape[0])

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
