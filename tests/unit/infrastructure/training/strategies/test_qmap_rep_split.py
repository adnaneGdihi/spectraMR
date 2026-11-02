"""Repetition-split risk: a label-free error estimate, unbiased in the power domain."""

from __future__ import annotations

import torch

from spectramr.infrastructure.training.strategies.qmap_rep_split import (
    power_projected_gains,
    power_residual,
    repetition_loss,
    repetition_risk,
)


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


def test_the_training_loss_is_minimised_at_the_noise_free_amplitude() -> None:
    """#2976: a magnitude loss against M_b is minimised near E[M_b] > A at low SNR.

    The power loss with a known gain must prefer the true amplitude's gain (1) over the
    gain that matches the noisy magnitude's mean.
    """
    amplitude, x, y, noise_power = _setup()
    render = amplitude.clone()
    biased = float((y[:, :1].mean() / amplitude.mean()) ** 2)
    assert biased > 1.05  # the Rician floor this test is about (about 1.10 at A = 6, C = 4)
    at_truth = repetition_loss(render, x[:, :1], y[:, :1], noise_power[:, 0], gain2=torch.ones(()))
    at_bias = repetition_loss(
        render, x[:, :1], y[:, :1], noise_power[:, 0], gain2=torch.tensor(biased)
    )
    assert at_truth < at_bias


def test_the_training_mask_is_built_from_the_input_repetition() -> None:
    """A target voxel outside the input's signal mask never reaches the loss."""
    amplitude, x, y, noise_power = _setup(n=1_000)
    x, y = x[:, :1].clone(), y[:, :1].clone()
    x[..., :100, :] = 0.0  # below the floor in the INPUT
    base = repetition_loss(amplitude, x, y, noise_power[:, 0])
    y[..., :100, :] = 1e3  # bright in the target only
    assert torch.equal(base, repetition_loss(amplitude, x, y, noise_power[:, 0]))


def test_the_training_loss_is_in_magnitude_units() -> None:
    """At A = 30 noise units the loss matches the magnitude MSE, not the 4 A^2 larger power MSE.

    A gauge prior of weight 1 is calibrated against the magnitude loss, so the data term
    must keep that scale.
    """
    gen = torch.Generator().manual_seed(0)
    amplitude = torch.full((1, 1, 50_000, 1), 30.0)
    x, y = _rss(amplitude, 4, gen), _rss(amplitude, 4, gen)
    power = torch.tensor([4.0])
    render = amplitude * 1.02  # a small shape error, true gain
    loss = repetition_loss(render, x, y, power, gain2=torch.ones(()))
    magnitude = ((render - y) ** 2).mean()
    assert 0.5 < float(loss / magnitude) < 2.0
    resid, w = power_residual(render, x, y, power, gain2=torch.ones(()))
    unweighted = resid.pow(2).sum() / w.sum()
    assert float(unweighted / loss) > 1000.0  # about 4 A^2 = 3600


def test_the_prior_gains_are_fitted_on_the_input_only() -> None:
    amplitude, x, _, noise_power = _setup(n=1_000)
    renders = [amplitude] * 3
    g = power_projected_gains(renders, x, noise_power)
    assert g.shape == (1, 3)
    assert torch.allclose(g, torch.ones_like(g), atol=0.05)
