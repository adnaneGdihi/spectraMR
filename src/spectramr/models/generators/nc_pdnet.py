"""NC-PDNet -- density-compensated unrolled reconstruction for off-grid acquisitions.

The reference baseline for non-Cartesian reconstruction: Ramzi et al., *NC-PDNet:
A Density-Compensated Unrolled Network for 2D and 3D Non-Cartesian MRI
Reconstruction*, IEEE TMI 2022. It reports up to +1.2 dB PSNR over U-Net and Deep
Image Prior, and at least +1 dB when generalising across anatomy (trained on
knee, validated on brain).

It is here as the **control the graph arms have to beat**. NC-PDNet keeps the
NUFFT and compensates sampling density rather than avoiding gridding, so if a
graph over genuinely off-grid samples does not beat it, the off-grid claim is not
earning anything and the cohort should know that before it writes a paper.

Structure, per cascade: a CNN proximal step on the image, then a sample-domain
data-consistency step through :class:`NonCartesianDataConsistency` in
``gradient`` mode -- the same layer every other arm in this cohort uses, so the
comparison isolates the regulariser rather than accidentally comparing two
fidelity implementations.

Simplifications stated rather than hidden: the primal-dual buffers of the paper
are collapsed to a single primal iterate with a learned step, and the
regulariser is a small residual CNN rather than the paper's larger one. Both are
capacity choices; the density compensation and the unrolled fidelity -- the two
things the paper's result is attributed to -- are intact.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.non_cartesian_dc import NonCartesianDataConsistency
from spectramr.models.blocks.attention_domains import (
    complex_to_interleaved,
    interleaved_to_complex,
)
from spectramr.models.generators.grad_checkpointing import GradCheckpointingMixin
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class ResidualRegulariser(nn.Module):
    """Small residual CNN proximal step on an interleaved complex image."""

    def __init__(self, channels: int, width: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(channels, width, 3, padding=1), nn.ReLU()]
        for _ in range(max(0, depth - 2)):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU()]
        layers.append(nn.Conv2d(width, channels, 3, padding=1))
        self.body = nn.Sequential(*layers)
        # Zero-init the exit so the cascade starts as pure data consistency and
        # the regulariser has to earn its contribution. Unlike the transformer
        # heads this does NOT make the output input-independent: the DC step
        # still carries the measurement through.
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


@register_model(
    name="nc_pdnet",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=True,
    expects_real_imag_interleaved=True,
)
class NCPDNet(GradCheckpointingMixin, nn.Module, IGenerator):
    """Density-compensated unrolled network over a non-Cartesian trajectory.

    Args:
        in_channels: Interleaved real/imag width, ``2 * coils``.
        out_channels: Must equal ``in_channels`` -- an unrolled net iterates on
            its own estimate, so the widths cannot differ.
        image_size: Matrix the NUFFT reconstructs onto.
        num_cascades: Unrolled iterations.
        width: Regulariser width.
        depth: Convolutions per regulariser.
        step_size: Initial DC step, in Landweber units.
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        image_size: int = 256,
        num_cascades: int = 8,
        width: int = 64,
        depth: int = 4,
        step_size: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError(
                f"NC-PDNet iterates on an interleaved real/imag image, so "
                f"in_channels must be even; got {in_channels}."
            )
        if out_channels != in_channels:
            raise ValueError(
                f"An unrolled cascade feeds its own output back in, so "
                f"out_channels ({out_channels}) must equal in_channels "
                f"({in_channels})."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        im_size = (int(image_size), int(image_size))
        self.regularisers = nn.ModuleList(
            ResidualRegulariser(in_channels, width, depth) for _ in range(num_cascades)
        )
        # One DC module per cascade so each learns its own fidelity weight, which
        # is what lets an unrolled net anneal from data-driven to prior-driven.
        self.dc_steps = nn.ModuleList(
            NonCartesianDataConsistency(
                im_size=im_size, mode="gradient", step_size=step_size, learn_step=True
            )
            for _ in range(num_cascades)
        )

    def forward(
        self,
        x: torch.Tensor,
        measured_kspace: torch.Tensor | None = None,
        trajectory: torch.Tensor | None = None,
        sample_mask: torch.Tensor | None = None,
        dcf: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Unroll regulariser/DC pairs over an interleaved image.

        ``measured_kspace``, ``trajectory`` and ``dcf`` come from the batch that
        ``NonCartesianSimulation`` stores. They are required: an unrolled network
        run without its measurement is a plain CNN wearing the name of one, and
        the DC layer raises rather than letting that happen quietly.
        """
        if measured_kspace is None:
            measured_kspace = kwargs.get("measured_samples")
        current = x
        ckpt = self._checkpointing_active()
        for regulariser, dc in zip(self.regularisers, self.dc_steps, strict=True):
            current = (
                torch.utils.checkpoint.checkpoint(regulariser, current, use_reentrant=False)
                if ckpt
                else regulariser(current)
            )
            complex_view = dc(
                interleaved_to_complex(current),
                measured_kspace,
                trajectory,
                sample_mask=sample_mask,
                dcf=dcf,
            )
            current = complex_to_interleaved(complex_view)
        return current

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    @property
    def name(self) -> str:
        return "NC-PDNet"

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generation wrapper to satisfy IGenerator."""
        return self.forward(z, **kwargs)


__all__ = ["NCPDNet", "ResidualRegulariser"]
