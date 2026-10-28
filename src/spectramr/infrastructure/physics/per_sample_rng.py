"""Random draws laid out one sample at a time, so a sample's draw does not depend on the batch.

``torch.randn(B, n)`` is not a batch of independent draws in the sense a degradation needs:
on CPU the Gaussian fill is vectorised in chunks, so element ``i`` depends on the total size,
and on CUDA the thread layout of every distribution kernel depends on it too. The same
subject then gets a different realisation in a batch of 2 than alone. Drawing each sample
in its own call, in order, makes sample ``b`` depend only on the generator state and the
samples before it, on every device. With ``batch = 1`` the stream is the one a single
``torch.rand(shape)`` / ``normal_`` call would have produced.
"""

from __future__ import annotations

import torch

__all__ = ["per_sample_normal_pair", "per_sample_rand", "per_sample_randn"]


def per_sample_rand(
    batch: int,
    *shape: int,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Uniform ``[0, 1)`` draws of ``shape`` for each of ``batch`` samples: ``[batch, *shape]``."""
    # A loop over the batch of RNG calls only; the cost is negligible next to the FFTs.
    return torch.stack(
        [torch.rand(shape, generator=generator, device=device, dtype=dtype) for _ in range(batch)]
    )


def per_sample_randn(
    batch: int,
    *shape: int,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Standard-normal draws of ``shape`` for each of ``batch`` samples: ``[batch, *shape]``."""
    return torch.stack(
        [
            torch.empty(shape, device=device, dtype=dtype).normal_(0.0, 1.0, generator=generator)
            for _ in range(batch)
        ]
    )


def per_sample_normal_pair(
    batch: int,
    *shape: int,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two standard-normal fields per sample, drawn first-then-second within each sample.

    The real and imaginary parts of complex noise (or the two Rician components).
    Returns two ``[batch, *shape]`` tensors.
    """

    def draw() -> torch.Tensor:
        return torch.empty(shape, device=device, dtype=dtype).normal_(0.0, 1.0, generator=generator)

    first, second = [], []
    for _ in range(batch):
        first.append(draw())
        second.append(draw())
    return torch.stack(first), torch.stack(second)
