from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.generators.protocols import IGeneratorProtocol
from spectramr.models.registry import register_model


@register_model(name="hybrid_qmap_generator", training_mode="reconstruction")
class HybridQMapGenerator(nn.Module, IGeneratorProtocol):
    """
    Hybrid Quantitative Parameter Map Generator.

    Combines a CNN Grid Encoder with a coordinate-based MLP Decoder (Implicit Neural Representation).
    This architecture enables continuous-resolution querying and super-resolution quantitative mapping.

    Architecture:
    1. Grid Encoder (CNN): Extracts a spatial feature volume from weighted MRI inputs.
    2. Coordinate Sampler: Interpolates features at arbitrary (x, y) continuous coordinates.
    3. Decoder MLP: Maps sampled features + coordinates to physical parameters (T1, T2, PD, B0, B1).
    """

    #: No test-time optimisation runs here (#2307): the audit rejects ``enable_tto: true`` at
    #: load; the build-time raise below also catches ``tto_*`` keys without ``enable_tto``.
    OPTION_SCHEMA: ClassVar[dict[str, list[Any]]] = {"enable_tto": [False]}

    def __init__(self, input_dim: int = 1, feature_dim: int = 64, **kwargs: Any):
        """
        Args:
            input_dim: Number of input channels (e.g., 2 for complex or 1 for magnitude).
            feature_dim: Dimension of the latent feature grid.
            **kwargs: Absorbed for factory compatibility. Recognised aliases:
                ``in_channels`` (canonical v6.0 name) overrides ``input_dim``.
        """
        tto = sorted(k for k in kwargs if k == "enable_tto" or k.startswith("tto_"))
        if tto:
            # Nothing in this model or its strategy runs test-time optimisation (#2307).
            raise ValueError(
                f"hybrid_qmap_generator does not run test-time optimisation, so {tto} "
                "would be dropped; remove them or use a model that wires TTO."
            )
        super().__init__()
        # Accept the canonical v6.0 `in_channels` kwarg as alias for input_dim
        # so model_factory wiring honours the YAML's declared channel count
        # instead of silently dropping it into **kwargs.
        if "in_channels" in kwargs:
            input_dim = int(kwargs.pop("in_channels"))
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.use_coordinate_sampling = True

        # 1. GRID ENCODER (CNN)
        # We use a compact encoder to produce the feature grid.
        # This can be swapped with a more powerful backbone if needed.
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, feature_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feature_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 2. COORDINATE DECODER (MLP)
        # Input to MLP: features (feature_dim) + coordinates (2)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(feature_dim + 2, 128),
            nn.LayerNorm(128),
            nn.SiLU(),  # SiLU is generally preferred for INR stability
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 5),  # Outputs: T1, T2, PD, B0, B1
        )

    def forward(
        self, images: torch.Tensor, query_coords: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for hybrid physics prediction.

        Args:
            images: (B, C, H, W) Input weighted MRI images.
            query_coords: (B, N, 2) Continuous coordinates in range [-1, 1].
                         If None, defaults to the full pixel grid.

        Returns:
            Tuple[t1, t2, pd, b0, b1]: Maps at queried coordinates.
        """
        B, C, H, W = images.shape

        # A. Encode to Feature Grid
        feature_grid = self.encoder(images)  # (B, feature_dim, H, W)

        # B. Handle Coordinate Sampling
        if query_coords is None:
            # Generate standard grid if none provided: (B, H*W, 2)
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, H, device=images.device),
                torch.linspace(-1, 1, W, device=images.device),
                indexing="ij",
            )
            query_coords = torch.stack([x, y], dim=-1).view(1, H * W, 2).expand(B, -1, -1)

        # C. Sample Features at Query Coordinates
        # F.grid_sample expects coords in [B, H_out, W_out, 2]
        # We treat query_coords as (B, 1, N, 2)
        N_pts = query_coords.shape[1]
        grid_coords = query_coords.unsqueeze(1)  # (B, 1, N, 2)

        sampled_features = F.grid_sample(
            feature_grid,
            grid_coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (B, feature_dim, 1, N)

        # Reshape for MLP concatenation: (B, N, feature_dim)
        sampled_features = sampled_features.squeeze(2).permute(0, 2, 1)

        # D. Decode to Physics Parameters
        # mlp_input: (B, N, feature_dim + 2)
        mlp_input = torch.cat([sampled_features, query_coords], dim=-1)
        raw_out = self.decoder_mlp(mlp_input)  # (B, N, 5)

        # E. Map to Physical Ranges
        # T1: Range [10, 3010] ms
        t1 = (torch.sigmoid(raw_out[..., 0]) * 3000.0 + 10.0).unsqueeze(1)
        # T2: Range [1, 501] ms
        t2 = (torch.sigmoid(raw_out[..., 1]) * 500.0 + 1.0).unsqueeze(1)
        # PD: Strict positivity using softplus
        pd = (F.softplus(raw_out[..., 2]) * 10.0).unsqueeze(1)
        # B0: Range [-200, 200] Hz
        b0 = (torch.tanh(raw_out[..., 3]) * 200.0).unsqueeze(1)
        # B1: Range [0.5, 1.5]
        b1 = (torch.tanh(raw_out[..., 4]) * 0.5 + 1.0).unsqueeze(1)

        # F. Reshape back to grid if needed (default case)
        if N_pts == H * W:
            t1 = t1.view(B, 1, H, W)
            t2 = t2.view(B, 1, H, W)
            pd = pd.view(B, 1, H, W)
            b0 = b0.view(B, 1, H, W)
            b1 = b1.view(B, 1, H, W)

        return t1, t2, pd, b0, b1

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "hybrid_qmap_generator"

    def generate(self, z: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        """generate.

        Args:
            z (torch.Tensor): Description.
        Returns:
            tuple[torch.Tensor, ...]: Description.
        """
        query_coords = kwargs.get("query_coords")
        return self.forward(z, query_coords=query_coords)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return (input_shape[0], 1, input_shape[2], input_shape[3])
