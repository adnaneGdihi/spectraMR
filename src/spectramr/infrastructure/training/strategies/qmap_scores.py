"""Tissue reference masks and across-subject spread for the gain-calibrated q-map arms.

``reference_masks`` is the one rule for CSF and white matter in noise-unit triplets; the
series-gain gate (``scripts/analysis/m4raw_series_gain_gate.py``) and the q-map
strategy's validation both use it, so the gate and the arm measure the same tissue.
"""

from __future__ import annotations

import torch


def reference_masks(
    z_t1: torch.Tensor, z_t2: torch.Tensor, z_flair: torch.Tensor, brain: torch.Tensor
) -> dict[str, torch.Tensor]:
    """CSF: bright T2 and dark FLAIR. WM: bright T1 and dark T2 (terciles within brain)."""

    def q(x: torch.Tensor, p: float) -> torch.Tensor:
        return torch.quantile(x[brain].float(), p)

    csf = brain & (z_t2 >= q(z_t2, 0.9)) & (z_flair <= q(z_flair, 0.33))
    wm = brain & (z_t1 >= q(z_t1, 0.67)) & (z_t2 <= q(z_t2, 0.33)) & ~csf
    return {"csf": csf, "wm": wm}


def white_matter_medians(maps: dict[str, torch.Tensor], z: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-sample median T1 and T2 inside each sample's white-matter mask.

    ``z`` is the (B, 3, H, W) noise-unit triplet. The brain is T1 strictly above its
    median, the series-gain gate's rule: it relies on air filling most of the slice.
    """
    t1s, t2s = [], []
    for b in range(z.shape[0]):
        z_t1, z_t2, z_fl = z[b, 0], z[b, 1], z[b, 2]
        brain = z_t1 > torch.quantile(z_t1.flatten().float(), 0.5)
        if not brain.any():  # a constant slice has no voxel above its own median
            t1s.append(maps["t1"].new_tensor(float("nan")))
            t2s.append(maps["t2"].new_tensor(float("nan")))
            continue
        wm =reference_masks(z_t1, z_t2, z_fl, brain)["wm"]
        t1s.append(maps["t1"][b, 0][wm].median())
        t2s.append(maps["t2"][b, 0][wm].median())
    return {"t1": torch.stack(t1s), "t2": torch.stack(t2s)}


def coefficient_of_variation(x: torch.Tensor) -> torch.Tensor:
    """Population std over mean: the across-subject spread of one tissue value."""
    x = x.flatten().float()
    return x.std(unbiased=False) / x.mean().abs().clamp_min(1e-12)
