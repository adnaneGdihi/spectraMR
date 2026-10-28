"""B0-conditioned hypernetwork generator for field-specific output weights.

A hypernetwork ``H_phi(B0)`` emits the weights of the last ``n_generated_tensors``
parameter tensors of a template ResNet generator (its output conv by default).
Every other parameter is an ordinary, field-independent parameter. The generated weights
are applied with ``torch.func.functional_call`` under ``vmap``, so gradients
reach ``H_phi`` through the loss and no batch loop runs in Python.

Only the 64 mT and 3 T domains exist in the ULF data, so the hypernetwork is a
two-point conditioning, not a continuous field interpolator.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.func import functional_call, vmap

from spectramr.models.generators.cycle_gan import ResNetGenerator
from spectramr.models.registry import register_model


class B0HyperNetwork(nn.Module):
    """Map a scalar B0 (Tesla) to a flat vector of generated weights.

    Args:
        b0_embed_dim: Dimension of the sinusoidal B0 embedding.
        hidden_dim: Width of the MLP backbone.
        num_layers: Number of linear layers in the backbone plus one.
        target_param_count: Length of the flat weight vector.
        chunk_size: Output heads are split into chunks of this many weights.
        b0_range: (min, max) B0 range used to normalise the input.
    """

    def __init__(
        self,
        b0_embed_dim: int = 64,
        hidden_dim: int = 512,
        num_layers: int = 3,
        target_param_count: int = 100000,
        chunk_size: int = 50000,
        b0_range: tuple[float, float] = (0.01, 7.0),
    ) -> None:
        super().__init__()
        self.target_param_count = target_param_count
        self.chunk_size = chunk_size
        self.b0_min, self.b0_max = b0_range
        self.b0_embed_dim = b0_embed_dim

        self.register_buffer(
            "freq_bands",
            torch.exp(torch.linspace(0, math.log(1000), b0_embed_dim // 2)),
        )

        layers: list[nn.Module] = []
        in_dim = b0_embed_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        n_chunks = max(1, (target_param_count + chunk_size - 1) // chunk_size)
        self.heads = nn.ModuleList()
        remaining = target_param_count
        for _ in range(n_chunks):
            out_size = min(chunk_size, remaining)
            self.heads.append(nn.Linear(hidden_dim, out_size))
            remaining -= out_size

    def init_from_template(self, flat_template: torch.Tensor) -> None:
        """Set the head biases to the template weights, so step 0 reproduces the template."""
        if flat_template.numel() != self.target_param_count:
            raise ValueError(
                f"template has {flat_template.numel()} values, hypernetwork emits "
                f"{self.target_param_count}."
            )
        offset = 0
        with torch.no_grad():
            for head in self.heads:
                n = head.out_features
                head.bias.copy_(flat_template[offset : offset + n])
                offset += n

    def _embed_b0(self, b0: torch.Tensor) -> torch.Tensor:
        """Sinusoidal embedding of B0, shape [B, b0_embed_dim]."""
        b0 = b0.reshape(-1)
        b0_norm = ((b0 - self.b0_min) / (self.b0_max - self.b0_min + 1e-8)).unsqueeze(-1)
        phases = b0_norm * self.freq_bands.unsqueeze(0)
        return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)

    def forward(self, b0: torch.Tensor) -> torch.Tensor:
        """Return generated weights, shape [B, target_param_count]."""
        h = self.backbone(self._embed_b0(b0))
        return torch.cat([head(h) for head in self.heads], dim=-1)


@register_model(name="b0_hypernetwork_generator", training_mode="gan")
class B0ConditionedGenerator(nn.Module):
    """ResNet generator whose declared output weights come from a B0 hypernetwork.

    Args:
        in_channels / out_channels: Image channels of the template generator.
        ngf / n_blocks: Template ResNet generator size.
        n_generated_tensors: The last N template parameter tensors (the output conv
            weight and bias with the default 2) are generated; the rest stay ordinary.
        hidden_dim / b0_embed_dim / num_layers: Hypernetwork size.
        b0_range: (min, max) field strength used to normalise the B0 input.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        ngf: int = 64,
        n_blocks: int = 6,
        *,
        n_generated_tensors: int = 2,
        hidden_dim: int = 512,
        b0_embed_dim: int = 64,
        num_layers: int = 3,
        b0_range: tuple[float, float] = (0.01, 7.0),
    ) -> None:
        super().__init__()
        self._cfg = {
            "ngf": ngf,
            "n_blocks": n_blocks,
            "n_generated_tensors": int(n_generated_tensors),
            "hidden_dim": hidden_dim,
            "b0_embed_dim": b0_embed_dim,
            "num_layers": num_layers,
            "b0_range": tuple(b0_range),
        }
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base = ResNetGenerator(in_channels, out_channels, ngf=ngf, n_blocks=n_blocks)

        names = [n for n, _ in self.base.named_parameters()]
        k = self._cfg["n_generated_tensors"]
        if not 1 <= k <= len(names):
            raise ValueError(
                f"n_generated_tensors={k} must be in [1, {len(names)}] "
                "(the template's parameter tensor count)."
            )
        self._gen_names = names[-k:]
        shapes = dict(self.base.named_parameters())
        self._gen_shapes = [tuple(shapes[n].shape) for n in self._gen_names]
        self._gen_sizes = [int(torch.Size(s).numel()) for s in self._gen_shapes]
        n_gen = sum(self._gen_sizes)

        self.hyper_network = B0HyperNetwork(
            b0_embed_dim=b0_embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            target_param_count=n_gen,
            chunk_size=max(1, n_gen),
            b0_range=b0_range,
        )
        template = torch.cat([shapes[n].detach().reshape(-1) for n in self._gen_names])
        self.hyper_network.init_from_template(template)

    def reverse(self, in_channels: int, out_channels: int) -> B0ConditionedGenerator:
        """Build the same architecture with channels swapped (the CycleGAN B->A side)."""
        return B0ConditionedGenerator(in_channels, out_channels, **self._cfg)

    def _unflatten(self, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        parts = torch.split(flat, self._gen_sizes)
        return {
            n: p.reshape(s)
            for n, p, s in zip(self._gen_names, parts, self._gen_shapes, strict=True)
        }

    def forward(
        self, x: torch.Tensor, field_strength: torch.Tensor | float | None = None
    ) -> torch.Tensor:
        """Translate ``x`` with the weights generated for its output field.

        Args:
            x: Input [B, C, H, W].
            field_strength: Output-domain B0 in Tesla, scalar or [B]. Required.
        """
        if field_strength is None:
            raise ValueError(
                "B0ConditionedGenerator requires field_strength (the output B0). "
                "Set data.expose.field_strength_pair so the batch carries it."
            )
        b0 = torch.as_tensor(field_strength, device=x.device, dtype=x.dtype).reshape(-1)
        if b0.numel() == 1:
            b0 = b0.expand(x.shape[0])
        if b0.numel() != x.shape[0]:
            raise ValueError(
                f"field_strength has {b0.numel()} entries for a batch of {x.shape[0]}."
            )
        flat = self.hyper_network(b0)  # [B, P]
        shared = {n: p for n, p in self.base.named_parameters() if n not in self._gen_names}

        def apply_one(w: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
            params = {**shared, **self._unflatten(w)}
            return functional_call(self.base, params, (xi.unsqueeze(0),)).squeeze(0)

        return vmap(apply_one)(flat, x)


__all__ = ["B0ConditionedGenerator", "B0HyperNetwork"]
