"""Per-sample virtual-fiducial realisations with disjoint seed families (VFU-02).

With one cached marker, the super-Nyquist probe target is the same image on
every step, so a network can memorise it and the "certificate" measures recall,
not recovery. Here each sample gets its own jittered lattice, keyed by a seed
that is a pure function of ``(base seed, family, epoch, ordinal)``:

* the TRAIN family draws even seeds, the VALIDATION family odd ones, so no
  validation marker can ever have been seen in training;
* validation ignores the epoch, so every pass over one checkpoint scores the
  same realisations;
* :func:`wrong_seed` flips a bit that preserves the family, giving a plausible
  but WRONG realisation for the mismatched-target control.

:func:`marker_batch` reproduces ``VirtualFiducial._build_gaussian_grid`` exactly
for a given seed (the jitter is drawn from the same CPU generator in the same
order), then builds all profiles in one batched ``einsum`` on the device.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

FAMILIES = {"train": 0, "val": 1}
_MASK64 = (1 << 64) - 1
_SEED_SPACE = 1 << 30


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def realisation_seed(base_seed: int, family: str, epoch: int, ordinal: int) -> int:
    """Seed of one sample's marker; even for ``train``, odd for ``val``."""
    if family not in FAMILIES:
        raise ValueError(f"unknown marker family {family!r}; expected one of {sorted(FAMILIES)}")
    if family == "val":
        epoch = 0  # one checkpoint must score the same realisations every pass
    h = int(base_seed) & _MASK64
    for part in (FAMILIES[family], int(epoch), int(ordinal)):
        h = _splitmix64(h ^ (part & _MASK64))
    return 2 * (h % _SEED_SPACE) + FAMILIES[family]


def realisation_seeds(
    base_seed: int, family: str, epoch: int, ordinals: Sequence[int]
) -> tuple[int, ...]:
    return tuple(realisation_seed(base_seed, family, epoch, o) for o in ordinals)


def wrong_seed(seed: int) -> int:
    """A different realisation from the same family (bit 1 flipped, parity kept)."""
    return int(seed) ^ 2


def _jittered_centres(
    im_size: tuple[int, ...], spacing_px: tuple[float, ...], jitter: float, seed: int
) -> list[Tensor]:
    """Peak centres per axis, drawn exactly as ``VirtualFiducial`` draws them."""
    axes = [
        torch.arange(sp / 2.0, float(n), sp, dtype=torch.float32)
        for n, sp in zip(im_size, spacing_px, strict=True)
    ]
    centres = [c.reshape(-1) for c in torch.meshgrid(*axes, indexing="ij")]
    if jitter > 0.0:
        g = torch.Generator().manual_seed(int(seed))
        centres = [
            (c + (torch.rand(c.shape, generator=g) - 0.5) * (2.0 * jitter * sp)).clamp(
                0.0, float(n) - 1.0
            )
            for c, sp, n in zip(centres, spacing_px, im_size, strict=True)
        ]
    return centres


def marker_batch(
    seeds: Sequence[int],
    im_size: tuple[int, int],
    spacing_px: tuple[float, float],
    sigma_px: tuple[float, float],
    jitter: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """One real fiducial per seed, ``[B, 1, H, W]``, each normalised to max 1."""
    if len(im_size) != 2:
        raise ValueError(f"marker_batch is 2-D, got im_size={im_size}")
    if not seeds:
        raise ValueError("marker_batch needs at least one seed")
    per = [_jittered_centres(im_size, spacing_px, jitter, s) for s in seeds]
    # One pinned, async H2D copy per call: a pageable .to("cuda") blocks the host every step.
    centres = torch.stack([torch.stack(c) for c in per], dim=1)  # [2, B, P]
    if device is not None and torch.device(device).type == "cuda":
        centres = centres.pin_memory().to(device, non_blocking=True)
    else:
        centres = centres.to(device)
    cy, cx = centres[0], centres[1]
    (h, w), (sy, sx) = im_size, sigma_px
    ry = torch.arange(h, device=device, dtype=torch.float32)
    rx = torch.arange(w, device=device, dtype=torch.float32)
    py = torch.exp(-((ry.view(1, 1, h) - cy.unsqueeze(-1)) ** 2) / (2 * sy**2))  # [B, P, H]
    px = torch.exp(-((rx.view(1, 1, w) - cx.unsqueeze(-1)) ** 2) / (2 * sx**2))  # [B, P, W]
    grid = torch.einsum("bph,bpw->bhw", py, px)
    grid = grid / grid.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    return grid.unsqueeze(1).to(dtype)


__all__ = [
    "FAMILIES",
    "marker_batch",
    "realisation_seed",
    "realisation_seeds",
    "wrong_seed",
]
