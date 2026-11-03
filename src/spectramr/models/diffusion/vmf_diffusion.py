r"""Hyperspherical (von Mises-Fisher) diffusion denoiser (``vmf_diffusion``).

Diffuses a latent whose channel vector at every position lies on
:math:`S^{C-1}` (the hyperspherical latent of Davidson *et al.* [1]). The
forward process is vMF noising with a decreasing concentration schedule
(:mod:`~spectramr.models.diffusion.vmf_schedule`); the network predicts the
clean direction :math:`\hat z_0` and is trained with the cosine loss
:math:`1 - \langle \hat z_0, z_0 \rangle`. Generation runs the registered
reverse sampler named by ``inference_sampler`` (default ``vmf_reverse``).

The model operates on latents only. The image <-> latent encoder/decoder and
the training strategy that drives this model on MRI data are separate
components.

Reference:
    [1] T. R. Davidson *et al.*, "Hyperspherical variational
        auto-encoders," UAI 2018.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spectramr.models.diffusion import vmf_sampler  # noqa: F401  (registers vmf_reverse)
from spectramr.models.diffusion.samplers import SamplerRegistry, get_sampler
from spectramr.models.diffusion.vmf_schedule import VMFNoiseSchedule
from spectramr.models.reconstruction.unet import SinusoidalPositionEmbeddings
from spectramr.models.registry import register_model


class _TimeResBlock(nn.Module):
    """Pre-norm residual conv block with an additive time embedding."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, hidden)
        self.conv1 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.time = nn.Linear(hidden, hidden)
        self.norm2 = nn.GroupNorm(1, hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


@register_model(
    name="vmf_diffusion",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain="latent",
    output_domain="latent",
    accepts_complex=False,
    strict_kwargs=True,
)
class VMFDiffusion(nn.Module):
    """vMF diffusion denoiser on a ``(B, C, H, W)`` hyperspherical latent.

    Args:
        in_channels: Latent channels ``C``; the sphere is :math:`S^{C-1}`, so ``C >= 2``.
        out_channels: Must equal ``in_channels`` (the prediction is a direction).
        cond_channels: Channels of an optional conditioning map concatenated to
            the input (e.g. the latent of the low-resolution image); 0 = unconditional.
        hidden: Trunk width (even, >= 4).
        num_blocks: Residual blocks in the trunk.
        timesteps: Noise levels ``T``.
        kappa_max: Concentration at ``t = 0``.
        kappa_min: Concentration at ``t = T - 1``.
        kappa_schedule: ``log_linear`` or ``linear``.
        inference_sampler: Registered sampler used by :meth:`sample`.
        sampling_steps: Model evaluations per :meth:`sample` call.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        cond_channels: int = 0,
        hidden: int = 64,
        num_blocks: int = 2,
        timesteps: int = 1000,
        kappa_max: float = 1.0e4,
        kappa_min: float = 1.0e-2,
        kappa_schedule: str = "log_linear",
        inference_sampler: str = "vmf_reverse",
        sampling_steps: int = 50,
    ) -> None:
        super().__init__()
        if int(in_channels) < 2:
            raise ValueError(
                f"vmf_diffusion needs in_channels >= 2 (latent on S^(C-1)), got {in_channels}."
            )
        if int(out_channels) != int(in_channels):
            raise ValueError(
                f"vmf_diffusion predicts a direction: out_channels ({out_channels}) must "
                f"equal in_channels ({in_channels})."
            )
        if int(cond_channels) < 0 or int(hidden) < 4 or int(hidden) % 2 or int(num_blocks) < 0:
            raise ValueError(
                "vmf_diffusion needs cond_channels >= 0, an even hidden >= 4 (sinusoidal "
                f"time embedding) and num_blocks >= 0; got "
                f"{cond_channels}, {hidden}, {num_blocks}."
            )
        if not SamplerRegistry.is_registered(inference_sampler):
            raise ValueError(
                f"vmf_diffusion inference_sampler {inference_sampler!r} is not registered; "
                f"available: {SamplerRegistry.list_available()}."
            )
        if not 1 <= int(sampling_steps) <= int(timesteps):
            raise ValueError(
                f"vmf_diffusion needs 1 <= sampling_steps <= timesteps, got "
                f"sampling_steps={sampling_steps}, timesteps={timesteps}."
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.cond_channels = int(cond_channels)
        self.hidden = int(hidden)
        self.num_blocks = int(num_blocks)
        self.inference_sampler = str(inference_sampler)
        self.sampling_steps = int(sampling_steps)
        self.schedule = VMFNoiseSchedule(timesteps, kappa_max, kappa_min, kappa_schedule)

        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.inp = nn.Conv2d(self.in_channels + self.cond_channels, self.hidden, 3, padding=1)
        self.blocks = nn.ModuleList(_TimeResBlock(self.hidden) for _ in range(self.num_blocks))
        self.out = nn.Conv2d(self.hidden, self.out_channels, 3, padding=1)

    def forward(self, z_t: Tensor, t: Tensor, cond: Tensor | None = None) -> Tensor:
        """Predict the clean unit direction from ``z_t`` at levels ``t``."""
        if (cond is None) != (self.cond_channels == 0):
            raise ValueError(
                f"vmf_diffusion built with cond_channels={self.cond_channels} but "
                f"cond is {'missing' if cond is None else 'given'}."
            )
        x = z_t if cond is None else torch.cat([z_t, cond], dim=1)
        temb = self.time_embed(t.to(z_t.dtype))
        h = self.inp(x)
        for block in self.blocks:
            h = block(h, temb)
        return F.normalize(self.out(F.silu(h)), dim=1, eps=1e-6)

    def training_loss(self, z0: Tensor, cond: Tensor | None = None) -> Tensor:
        """Cosine x0-prediction loss at uniformly drawn levels.

        ``z0`` is projected onto the channel sphere here: the model owns the manifold.
        """
        z0 = F.normalize(z0, dim=1, eps=1e-6)
        t = torch.randint(0, self.schedule.timesteps, (z0.shape[0],), device=z0.device)
        z_t = self.schedule.q_sample(z0, t)
        pred = self(z_t, t, cond)
        return (1.0 - (pred * z0).sum(dim=1)).mean()

    def sample(
        self,
        shape: tuple[int, ...],
        cond: Tensor | None = None,
        steps: int | None = None,
    ) -> Tensor:
        """Generate ``shape`` = ``(B, C, H, W)`` latents through the registered sampler."""
        sampler = get_sampler(
            self.inference_sampler, steps=self.sampling_steps if steps is None else steps
        )
        return sampler.sample(self, shape, cond)

    def provenance(self) -> dict[str, Any]:
        """Resolved knobs for the run's provenance record."""
        return {
            "model_type": "vmf_diffusion",
            "in_channels": self.in_channels,
            "cond_channels": self.cond_channels,
            "hidden": self.hidden,
            "num_blocks": self.num_blocks,
            "inference_sampler": self.inference_sampler,
            "sampling_steps": self.sampling_steps,
            **self.schedule.describe(),
        }


__all__ = ["VMFDiffusion"]
