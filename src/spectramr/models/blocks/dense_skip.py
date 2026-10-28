"""Time-conditioned dense skip block used by the configurable UNet."""

import torch
from torch import nn

from spectramr.models.blocks.normalization import get_adaptive_groups


class DenseSkipConnection(nn.Module):
    """Dense layers over concatenated features, a 1x1 transition and a time-conditioned residual."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        num_layers: int,
        time_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        current = in_channels
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.GroupNorm(get_adaptive_groups(current), current),
                    nn.SiLU(),
                    nn.Conv2d(current, growth_rate, 3, padding=1),
                    nn.Dropout(dropout),
                )
            )
            current += growth_rate
        self.transition = nn.Sequential(
            nn.GroupNorm(get_adaptive_groups(current), current),
            nn.SiLU(),
            nn.Conv2d(current, in_channels, 1),
            nn.Dropout(dropout),
        )
        self.time_mlp = nn.Linear(time_dim, in_channels)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None) -> torch.Tensor:
        if time_emb is None:
            raise ValueError("DenseSkipConnection needs a time embedding; pass `time`/`timesteps`.")
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, 1)))
        out = self.transition(torch.cat(features, 1))
        return out + self.time_mlp(time_emb)[:, :, None, None] + x
