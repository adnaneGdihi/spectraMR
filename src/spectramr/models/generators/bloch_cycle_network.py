"""Bloch Cycle Network
===================

A physics-informed network that incorporates Bloch equation constraints
or cycle consistency for MRI reconstruction.
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from spectramr.models.generators.bloch_cycle_maps import (
    BlochSequence,
    build_refinement_modules,
    maps_from_logits,
    resolve_fixed_relaxation,
)
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.reconstruction.unet import (
    AttentionType,
    BlockType,
    NormalizationType,
    UNetConfig,
)
from spectramr.models.reconstruction.unet import UNet as ConfigurableUNetGenerator
from spectramr.models.registry import register_model


@register_model(
    name="bloch_cycle",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="image",
    accepts_complex=False,
    expects_real_imag_interleaved=True,
    requires_paired_data=False,
    strict_kwargs=True,
)
@register_model(
    name="bloch_cycle_network",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="image",
    accepts_complex=False,
    expects_real_imag_interleaved=True,
    requires_paired_data=False,
    strict_kwargs=True,
)
class BlochCycleNetwork(nn.Module, IGenerator):
    """Bloch Cycle Network for MRI Reconstruction.

    A UNet predicts bounded quantitative maps (PD, T1, T2 in ms) and the analytical
    Bloch equation renders the signal at the declared sequence. With
    ``num_bloch_modules > 1`` each further module refines the map logits from the
    input and the previous render, and renders again.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, ...] = (64, 128, 256, 512, 1024),
        depth: int = 4,
        bilinear: bool = True,
        dropout: float = 0.0,
        sequence_type: str = "SE",
        bloch_config: Mapping[str, Any] | None = None,
        t1_learnable: bool = True,
        t2_learnable: bool = True,
        t1_fixed_ms: float | None = None,
        t2_fixed_ms: float | None = None,
        num_bloch_modules: int = 1,
        feature_dim: int | None = None,
        num_residual_blocks: int | None = None,
    ):
        """Initialize Bloch Cycle Network.

        Args:
            in_channels: Number of input channels (e.g. k-space or image).
            out_channels: Declared output width; the rendered signal has one channel.
            features: UNet widths, one per level (``depth + 1`` entries).
            depth: UNet depth.
            bilinear: Upsampling mode.
            dropout: Dropout rate.
            sequence_type: ``SE`` or ``GRE``, the signal equation the simulator applies.
            bloch_config: ``{TR, TE[, flip_angle, use_physics_backward]}`` in ms / deg;
                ``None`` keeps spin echo at TR 3000 / TE 100 ms.
            t1_learnable / t2_learnable: predict the map (true) or hold it at
                ``t1_fixed_ms`` / ``t2_fixed_ms`` (false).
            num_bloch_modules: Bloch render stages; stages after the first refine the logits.
            feature_dim / num_residual_blocks: width and depth of each refinement stage.
        """
        super().__init__()
        features = tuple(int(f) for f in features)
        if len(features) != depth + 1:
            raise ValueError(
                f"features={features} must give depth + 1 = {depth + 1} widths, one per level."
            )
        self.sequence = BlochSequence.from_config(bloch_config, sequence_type)
        self._t1_fixed = resolve_fixed_relaxation("t1", bool(t1_learnable), t1_fixed_ms)
        self._t2_fixed = resolve_fixed_relaxation("t2", bool(t2_learnable), t2_fixed_ms)

        # Encoder: image -> logits of the quantitative maps (PD, T1, T2).
        self.map_generator = ConfigurableUNetGenerator(
            config=UNetConfig(
                in_channels=in_channels,
                out_channels=3,
                features=features,
                depth=depth,
                bilinear=bilinear,
                dropout=float(dropout),
                attention_type=AttentionType.CHANNEL,
                norm_type=NormalizationType.BATCH,
                block_type=BlockType.STANDARD,
                use_attention=True,
                use_residual=True,
                deep_supervision=False,
            )
        )
        self.refiners = build_refinement_modules(
            in_channels, int(num_bloch_modules), feature_dim, num_residual_blocks
        )

        # Decoder: the analytical, differentiable Bloch simulator.
        from spectramr.infrastructure.physics.bloch_simulation import (
            DifferentiableBlochSimulator,
        )

        self.bloch_simulator = DifferentiableBlochSimulator(sequence_type=sequence_type)
        self.sequence_provenance: dict[str, Any] = {
            "sequence_type": self.sequence.sequence_type,
            "tr_ms": self.sequence.tr_ms,
            "te_ms": self.sequence.te_ms,
            "flip_deg": self.sequence.flip_deg,
            "num_bloch_modules": int(num_bloch_modules),
            "t1_fixed_ms": self._t1_fixed,
            "t2_fixed_ms": self._t2_fixed,
        }

    def _render(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        maps = maps_from_logits(logits, self._t1_fixed, self._t2_fixed)
        alpha = self.sequence.flip_deg if self.sequence.sequence_type == "GRE" else None
        signal = self.bloch_simulator(
            maps, TR=self.sequence.tr_ms, TE=self.sequence.te_ms, alpha=alpha
        )
        return maps, signal

    def _run(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """(signal, final maps, whether the input was NHWC) through every Bloch module."""
        needs_permute = x.ndim == 4 and x.shape[1] > 16 and x.shape[3] <= 16
        if needs_permute:
            x = x.permute(0, 3, 1, 2)
        if torch.is_complex(x):
            x = x.abs()
        logits = self.map_generator(x)
        maps, signal = self._render(logits)
        for refine in self.refiners:
            logits = refine(x, signal, logits)
            maps, signal = self._render(logits)
        return signal, maps, needs_permute

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """[B, C, H, W] (or NHWC, or complex) -> rendered signal [B, 1, H, W]."""
        signal, _, needs_permute = self._run(x)
        return signal.permute(0, 2, 3, 1) if needs_permute else signal

    def forward_with_intermediates(self, x: torch.Tensor, **kwargs):
        """The signal, plus ``[maps, signal]`` of the final Bloch module."""
        signal, maps, needs_permute = self._run(x)
        if needs_permute:
            signal = signal.permute(0, 2, 3, 1)
        return signal, [maps, signal]

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """generate.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.forward(x)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        # Output shape matches input spatial dims, 1 channel
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        B, C, H, W = input_shape
        return (B, 1, H, W)

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "bloch_cycle_network"
