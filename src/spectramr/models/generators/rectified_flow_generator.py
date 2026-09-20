"""Rectified Flow Generator for MRI Reconstruction.

Implements measurement-conditional flow matching for MRI inverse problems.
The vector field v_θ(z_t, t, c) is conditioned on the measurement embedding c.

References:
- Lipman et al., "Flow Matching for Generative Modeling," ICLR 2023
- Chung et al., "Diffusion Posterior Sampling for General Noisy Inverse Problems," ICLR 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.physics.data_consistency import NoiseSimulatingDataConsistency
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class MeasurementEncoder(nn.Module):
    """Encodes k-space measurements into conditioning vector."""

    def __init__(
        self,
        in_channels: int = 2,
        context_dim: int = 512,
        features: list[int] = [64, 128, 256],
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            context_dim (int): Description.
            features (list[int]): Description.
        """
        super().__init__()
        self.context_dim = context_dim

        layers = []
        ch = in_channels
        for feat in features:
            layers.extend(
                [
                    nn.Conv2d(ch, feat, 3, stride=2, padding=1),
                    nn.GroupNorm(8, feat),
                    nn.SiLU(),
                ]
            )
            ch = feat

        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(features[-1], context_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode measurement to context vector.

        Args:
            x: Measurement tensor [B, C, H, W]

        Returns:
            Context vector [B, context_dim]

        forward method for MeasurementEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.encoder(x)
        h = self.pool(h).flatten(1)
        return self.proj(h)


class VelocityNetwork(nn.Module):
    """Velocity field prediction network v_θ(z_t, t, c)."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        features: list[int] = [64, 128, 256, 512],
        context_dim: int = 512,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            features (list[int]): Description.
            context_dim (int): Description.
        """
        super().__init__()
        self.context_dim = context_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )

        # Context projection
        self.context_proj = nn.Linear(context_dim, 256)

        # Combined conditioning
        self.cond_proj = nn.Linear(512, features[0])

        # Encoder
        self.encoder = nn.ModuleList()
        ch = in_channels
        for feat in features:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(ch, feat, 3, padding=1),
                    nn.GroupNorm(8, feat),
                    nn.SiLU(),
                    nn.Conv2d(feat, feat, 3, padding=1),
                    nn.GroupNorm(8, feat),
                    nn.SiLU(),
                )
            )
            ch = feat

        # Decoder with skip connections
        self.decoder = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(features[i] + features[i - 1], features[i - 1], 3, padding=1),
                    nn.GroupNorm(8, features[i - 1]),
                    nn.SiLU(),
                    nn.Conv2d(features[i - 1], features[i - 1], 3, padding=1),
                    nn.GroupNorm(8, features[i - 1]),
                    nn.SiLU(),
                )
            )

        self.out = nn.Conv2d(features[0], out_channels, 3, padding=1)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field.

        Args:
            z_t: Noisy latent [B, C, H, W]
            t: Timestep [B] or [B, 1]
            context: Conditioning vector [B, context_dim]

        Returns:
            Velocity field [B, C, H, W]

        forward method for VelocityNetwork.

        Executes PyTorch tensor operations.

        Args:
            z_t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            context (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B = z_t.shape[0]

        # Time embedding
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_embed(t.float())

        # Context embedding
        c_emb = self.context_proj(context)

        # Combined conditioning
        cond = torch.cat([t_emb, c_emb], dim=-1)
        cond = self.cond_proj(cond)  # [B, features[0]]

        # Encoder path
        skips = []
        h = z_t
        for i, block in enumerate(self.encoder):
            h = block(h)
            if i == 0:
                # Add conditioning via channel-wise scaling
                h = h + cond.view(B, -1, 1, 1)
            skips.append(h)
            if i < len(self.encoder) - 1:
                h = F.avg_pool2d(h, 2)

        # Decoder path
        for i, block in enumerate(self.decoder):
            h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
            skip = skips[-(i + 2)]
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        return self.out(h)


@register_model(
    name="rectified_flow",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class RectifiedFlowGenerator(nn.Module, IGenerator):
    """Rectified Flow for MRI Reconstruction.

    Implements measurement-conditional flow matching where the vector field
    is conditioned on the k-space measurement embedding.

    The flow equation: dz/dt = v_θ(z_t, t, c)
    where c = Encoder(y) is the measurement embedding.

    Training objective (Conditional Flow Matching):
        L = E_{x_0, x_1, t} [ || v_θ(z_t, t, c) - (x_1 - x_0) ||^2 ]
    where z_t = (1-t)*x_0 + t*x_1 (linear interpolation)
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        features: list[int] = [64, 128, 256, 512],
        context_dim: int = 512,
        flow_steps: int = 10,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            features (list[int]): Description.
            context_dim (int): Description.
            flow_steps (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flow_steps = flow_steps

        # Measurement encoder
        self.measurement_encoder = MeasurementEncoder(
            in_channels=in_channels,
            context_dim=context_dim,
            features=features[:3] if len(features) >= 3 else features,
        )

        # Velocity network
        self.velocity_net = VelocityNetwork(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            context_dim=context_dim,
        )

        # Physics: Data Consistency
        self.dc_layer = NoiseSimulatingDataConsistency()

    def forward(
        self,
        x: torch.Tensor,
        measurement: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass - either training or inference.

        For training: x is the target, measurement is the conditioning
        For inference: x is the initial noise, measurement is the conditioning

        forward method for RectifiedFlowGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            measurement (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # [FIX] Alias for strategy compatibility
        if measurement is None:
            measurement = kwargs.get("kspace_measured") or kwargs.get("measured_kspace")

        if measurement is None:
            # If no measurement provided, assume x is target/noise and we can't do DC
            measurement_for_cond = x
        else:
            measurement_for_cond = measurement

        # Encode measurement
        context = self.measurement_encoder(measurement_for_cond)

        # ODE solve (Euler method)
        # If training (x is target), we usually just return velocity for loss.
        # But this forward implementation seems to do generation?
        # The training logic calls compute_loss directly.
        # So forward is likely used for inference/generation.

        z = torch.randn_like(x)  # Start from noise
        dt = 1.0 / self.flow_steps

        for step in range(self.flow_steps):
            t = torch.full((x.shape[0],), step * dt, device=x.device)
            v = self.velocity_net(z, t, context)
            z = z + v * dt

            # Physics Correction (Manifold Constraint)
            # "Project" back to the feasible set defined by k-space measurements
            if mask is not None and measurement is not None:
                z = self.dc_layer(z, measurement, mask)

        # Apply final data consistency
        if mask is not None and measurement is not None:
            z = self.dc_layer(z, measurement, mask)

        return z

    def compute_loss(
        self,
        x_0: torch.Tensor,  # Noise sample
        x_1: torch.Tensor,  # Target (ground truth)
        measurement: torch.Tensor,  # K-space measurement for conditioning
    ) -> torch.Tensor:
        """Compute Conditional Flow Matching loss.

        Args:
            x_0: Starting point (noise) [B, C, H, W]
            x_1: Target image [B, C, H, W]
            measurement: K-space measurement [B, C, H, W]

        Returns:
            Flow matching loss
        """
        B = x_1.shape[0]

        # Sample random timestep
        t = torch.rand(B, device=x_1.device)

        # Linear interpolation
        t_view = t.view(B, 1, 1, 1)
        z_t = (1 - t_view) * x_0 + t_view * x_1

        # Target velocity (straight line)
        target_velocity = x_1 - x_0

        # Encode measurement
        context = self.measurement_encoder(measurement)

        # Predict velocity
        pred_velocity = self.velocity_net(z_t, t, context)

        # MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity)

        return loss

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate output from latent."""
        return self.forward(z, **kwargs)

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "rectified_flow_generator"
