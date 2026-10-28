"""FDB's test-time acquisition mask, and the reverse schedule that mask has to survive.

The paper undersamples "based on a normal density across the two-dimensional transverse
plane" at R = 4-8 (arXiv:2308.01096) and ships its masks inside the test ``.mat`` files
(``us_masks``, read by ``read_data.get_us_singlecoil``), not as code or as files in the
repository. :func:`normal_density_mask` regenerates a pattern from that description; its
width is this module's choice, and the adapter records it.

The pattern also has to satisfy upstream's ``DiffusionBridge.create_mask``
(``utils/fdb.py``), which removes the unacquired points in ``T`` rounds by rejection
sampling outside a radius shrinking from ``max(H, W) / 2`` towards ``r_end``, and loops
forever once a round has fewer eligible points than it wants. :func:`check_bridge_feasible`
replays that schedule and raises instead.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import numpy as np
import torch

#: ``create_mask``'s inner radius per ``data_type`` (``utils/fdb.py``).
BRIDGE_R_END = {"singlecoil": 3.0, "multicoil": 4.0}


def bridge_test_steps(train_steps: int, train_rate: int, test_rate: int) -> int:
    """Reverse steps for a bridge trained at ``train_rate`` and tested at ``test_rate``.

    The paper's ``T_r = floor(T_f (R - 1) R' / ((R' - 1) R))``, which gives the README's
    ``--diffusion_steps 1500`` at R = 4 and 1750 at R = 8 for ``T_f = 1000``, R' = 2.
    """
    if train_rate <= 1 or test_rate <= 1:
        raise ValueError(
            f"FDB undersampling rates must exceed 1 (train {train_rate}, test {test_rate}): "
            "the step formula divides by R' - 1."
        )
    return (train_steps * (test_rate - 1) * train_rate) // ((train_rate - 1) * test_rate)


def bridge_radii(height: int, width: int, steps: int, data_type: str) -> np.ndarray:
    """``create_mask``'s exclusion radius at each of its ``steps`` rounds, float64."""
    if data_type not in BRIDGE_R_END:
        raise ValueError(
            f"Unknown FDB data_type {data_type!r}; expected one of {sorted(BRIDGE_R_END)}."
        )
    r_start = max(height, width) / 2
    r_end = BRIDGE_R_END[data_type]
    t = np.arange(steps, dtype=np.float64)
    return r_start - t * (r_start - r_end) / steps


def _centre_distance_sq(height: int, width: int) -> np.ndarray:
    """``(x - rows/2)**2 + (y - cols/2)**2``, the centre ``create_mask`` measures from."""
    x = np.arange(height, dtype=np.float64)[:, None] - height / 2
    y = np.arange(width, dtype=np.float64)[None, :] - width / 2
    return x**2 + y**2


def normal_density_mask(
    height: int,
    width: int,
    acceleration: int,
    *,
    sigma: float,
    centre_radius: float,
    seed: int,
) -> torch.Tensor:
    """A 2-D point mask at exactly ``round(H * W / R)`` samples, float 0/1 ``[H, W]``.

    Points closer to the centre than ``centre_radius`` are always acquired; the rest are
    drawn without replacement with weight ``exp(-d^2 / (2 (sigma * max(H, W) / 2)^2))``.

    Args:
        acceleration: The test acceleration R.
        sigma: Gaussian width as a fraction of the half field of view.
        centre_radius: Radius of the fully acquired disk, in k-space samples.
        seed: Seed of the draw; the same arguments give the same mask.
    """
    if acceleration <= 1 or sigma <= 0:
        raise ValueError(f"Need acceleration > 1 and sigma > 0, got {acceleration}, {sigma}.")
    d2 = torch.from_numpy(_centre_distance_sq(height, width)).flatten()
    budget = round(height * width / acceleration)
    centre = d2 < centre_radius**2
    remaining = budget - int(centre.sum())
    if remaining < 0:
        raise ValueError(
            f"The acquired disk of radius {centre_radius:.2f} holds {int(centre.sum())} "
            f"points, more than R={acceleration} allows at {height}x{width} ({budget})."
        )
    weight = torch.exp(-d2 / (2 * (sigma * max(height, width) / 2) ** 2))
    weight[centre] = 0
    generator = torch.Generator().manual_seed(seed)
    picked = torch.multinomial(weight, remaining, replacement=False, generator=generator)
    mask = centre.clone()
    mask[picked] = True
    return mask.reshape(height, width).to(torch.float32)


def check_bridge_feasible(mask: torch.Tensor, steps: int, data_type: str) -> None:
    """Raise if ``create_mask`` would spin forever on ``mask`` over ``steps`` rounds.

    Round ``t`` removes ``p = int(remaining / rounds_left)`` points, each an unacquired
    point outside radius ``r_t``. The eligible set only grows as ``r_t`` shrinks, so the
    schedule completes iff the removals through every round fit inside that round's
    eligible set.
    """
    m = mask.detach().cpu().numpy() != 0
    height, width = m.shape
    d2 = np.sort(_centre_distance_sq(height, width)[~m])
    radii = bridge_radii(height, width, steps, data_type)
    eligible = d2.size - np.searchsorted(d2, radii**2, side="left")
    remaining, left, removed = d2.size, steps, 0
    for t in range(steps):
        p = int(remaining / left)
        removed += p
        if removed > eligible[t]:
            raise ValueError(
                f"FDB's create_mask cannot finish on this mask: round {t} of {steps} needs "
                f"{removed} unacquired points outside radius {radii[t]:.2f}, and the mask "
                f"has {int(eligible[t])}. Upstream would loop forever here; acquire more "
                "of the centre or widen the density."
            )
        remaining -= p
        left -= 1


def last_round_radius(height: int, width: int, steps: int, data_type: str) -> float:
    """The smallest radius ``create_mask`` excludes; an acquired disk this wide leaves no hole
    the final round cannot reach."""
    return float(bridge_radii(height, width, steps, data_type)[-1])


@contextlib.contextmanager
def seeded_numpy(seed: int) -> Iterator[None]:
    """Seed the global ``np.random`` that ``create_mask`` draws from, then restore it."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


__all__ = [
    "BRIDGE_R_END",
    "bridge_radii",
    "bridge_test_steps",
    "check_bridge_feasible",
    "last_round_radius",
    "normal_density_mask",
    "seeded_numpy",
]
