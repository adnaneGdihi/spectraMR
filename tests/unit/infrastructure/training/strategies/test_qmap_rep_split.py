"""Repetition-split risk: a label-free error estimate, unbiased in the power domain."""

from __future__ import annotations

import torch

from spectramr.infrastructure.training.strategies.qmap_rep_split import repetition_risk


def _rss(
    amplitude: torch.Tensor, coils: int, gen: torch.Generator, fraction: float = 1.0
) -> torch.Tensor:
    """RSS magnitude in noise units; ``fraction`` of k-space acquired scales the noise power."""
    s = amplitude / coils**0.5
    std = (fraction / 2) ** 0.5
    re = s + torch.randn(coils, *amplitude.shape, generator=gen) * std
    im = torch.randn(coils, *amplitude.shape, generator=gen) * std
    return (re**2 + im**2).sum(0).sqrt()


def _setup(coils: int = 4, n: int = 200_000):
    gen = torch.Generator().manual_seed(0)
    amplitude = torch.full((1, 1, n, 1), 6.0)  # low SNR, where the magnitude bias matters
    x = torch.cat([_rss(amplitude, coils, gen) for _ in range(3)], dim=1)
    y = torch.cat([_rss(amplitude, coils, gen) for _ in range(3)], dim=1)
    noise_power = torch.tensor([[float(coils)] * 3])
    return amplitude, x, y, noise_power


def test_the_power_target_is_unbiased_for_the_true_render() -> None:
    amplitude, x, y, noise_power = _setup()
    renders = [amplitude.clone() for _ in range(3)]
    out = repetition_risk(renders, x, y, noise_power, signal_floor=0.0)
    # Both the gain (fitted on the input) and the target are bias-corrected powers, so the
    # mean residual sits near 0. A magnitude-domain gain left it about 3.5 (near C = 4) high.
    assert abs(out["val_rep_split_bias"]) < 0.15 * 4


def test_a_render_with_a_shape_error_scores_a_higher_risk() -> None:
    amplitude, x, y, noise_power = _setup(n=20_000)
    pattern = torch.ones_like(amplitude)
    pattern[..., ::2, :] = 1.3  # wrong contrast between voxels, not a global scale
    good = repetition_risk([amplitude] * 3, x, y, noise_power, signal_floor=0.0)
    bad = repetition_risk([amplitude * pattern] * 3, x, y, noise_power, signal_floor=0.0)
    assert bad["val_rep_split_risk"] > good["val_rep_split_risk"]


def test_the_gain_never_reads_the_target_repetition() -> None:
    amplitude, x, y, noise_power = _setup(n=1000)
    a = repetition_risk([amplitude] * 3, x, y, noise_power, signal_floor=0.0)
    b = repetition_risk([amplitude] * 3, x, 2.0 * y, noise_power, signal_floor=0.0)
    # a target-fitted gain would track the 2x and leave the bias unchanged
    assert b["val_rep_split_bias"] != a["val_rep_split_bias"]


def test_each_contrast_is_reported_and_averaged() -> None:
    amplitude, x, y, noise_power = _setup(n=1000)
    out = repetition_risk([amplitude] * 3, x, y, noise_power, signal_floor=0.0)
    for name in ("t1", "t2", "flair"):
        assert f"val_rep_split_risk_{name}" in out
    per = [out[f"val_rep_split_risk_{c}"] for c in ("t1", "t2", "flair")]
    assert abs(out["val_rep_split_risk"] - sum(per) / 3) < 1e-6 * max(per)


def test_a_zero_filled_scan_is_unbiased_with_its_acquired_noise_power() -> None:
    # The reviewer's case: a spatially varying render on a scan with 76 % of k-space
    # acquired. Subtracting C instead of f C left a render-dependent bias of about 0.3.
    gen = torch.Generator().manual_seed(1)
    coils, f = 4, 0.76
    amplitude = torch.linspace(2.0, 12.0, 100_000).view(1, 1, -1, 1)
    x = torch.cat([_rss(amplitude, coils, gen, f) for _ in range(3)], dim=1)
    y = torch.cat([_rss(amplitude, coils, gen, f) for _ in range(3)], dim=1)
    out = repetition_risk(
        [amplitude] * 3, x, y, torch.tensor([[coils * f] * 3]), signal_floor=0.0
    )
    assert abs(out["val_rep_split_bias"]) < 0.05
