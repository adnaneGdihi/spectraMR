"""A 1-D convolutional denoiser whose receptive field is bounded by construction.

DTN2S (dual-traversal Noise2Self) is J-invariant only if the output at sequence position ``i``
reads no input position further than ``receptive_window`` tokens from ``i``: the mask blinds
exactly the voxels that fall inside that window. A state-space or attention backbone reads the whole
sequence, so no mask of that width can certify it. This backbone stacks plain ``Conv1d`` layers whose
half-widths add up to ``receptive_half_width``, and exposes that number so the strategy can check it
against the window instead of trusting the pairing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.registry import register_model

__all__ = ["WindowedSequenceDenoiser"]


@register_model(
    name="windowed_sequence_denoiser",
    training_mode="reconstruction",
    # Rank 1 is what the module consumes; rank 2 is how a DTN2S arm reaches it, through the
    # strategy's space-filling-curve linearisation. A direct 2-D forward raises.
    spatial_dims=(1, 2),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=False,
    strict_kwargs=True,
)
class WindowedSequenceDenoiser(nn.Module):
    """Maps a sequence ``(B, C_in, N)`` to ``(B, C_out, N)`` reading at most ``receptive_half_width`` tokens either side.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        receptive_half_width: Tokens read on each side of a position. Must be a multiple of
            ``(kernel_size - 1) // 2`` so the layers add up to it exactly.
        kernel_size: Odd convolution width of every layer.
        hidden_channels: Channel width of the hidden layers.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        receptive_half_width: int = 8,
        kernel_size: int = 3,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd and >= 3, got {kernel_size}")
        step = (kernel_size - 1) // 2
        if receptive_half_width < step or receptive_half_width % step:
            raise ValueError(
                f"receptive_half_width={receptive_half_width} must be a positive multiple of "
                f"(kernel_size - 1) // 2 = {step}: the layers' half-widths add up to it exactly."
            )
        if hidden_channels < 1:
            raise ValueError(f"hidden_channels must be >= 1, got {hidden_channels}")
        n_layers = receptive_half_width // step
        layers: list[nn.Module] = []
        width = in_channels
        for _ in range(n_layers):
            layers += [nn.Conv1d(width, hidden_channels, kernel_size, padding=step), nn.GELU()]
            width = hidden_channels
        self.body = nn.Sequential(*layers)
        self.project = nn.Conv1d(width, out_channels, kernel_size=1)
        #: Read by ``DTN2SStrategy``: the largest distance an output position reads along the sequence.
        self.receptive_half_width = int(receptive_half_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C_in, N)`` to ``(B, C_out, N)``; any other rank raises."""
        if x.ndim != 3:
            raise ValueError(
                f"WindowedSequenceDenoiser expects a (B, C, N) sequence, got {tuple(x.shape)}"
            )
        return self.project(self.body(x))
