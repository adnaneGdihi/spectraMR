"""Tests for the SSDU reference-free reconstruction strategy."""

import torch

from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)
from spectramr.infrastructure.training.strategies.ssdu_strategy import (
    SSDUReconstructionStrategy,
    inject_noisier_kspace,
    split_acquired_mask,
)
from spectramr.models.losses.ssdu_loss import SSDULoss


def test_split_is_disjoint_and_covers_acquired():
    # acquired mask: 1 where k-space was sampled
    acq = torch.zeros(1, 1, 8, 8)
    acq[..., ::2] = 1.0  # half the columns acquired
    lam, theta = split_acquired_mask(
        acq, theta_fraction=0.4, generator=torch.Generator().manual_seed(0)
    )
    # Lambda and Theta are disjoint
    assert torch.all((lam * theta) == 0)
    # Lambda union Theta == acquired (no leakage outside acquired support)
    assert torch.allclose(lam + theta, acq)
    # Theta holds ~40% of acquired points
    n_acq = int(acq.sum().item())
    assert abs(int(theta.sum().item()) - round(0.4 * n_acq)) <= 1


def test_split_rejects_bad_fraction():
    acq = torch.ones(1, 1, 4, 4)
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            split_acquired_mask(acq, theta_fraction=bad)
        except ValueError:
            continue
        raise AssertionError(f"theta_fraction={bad} should have raised")


def test_split_handles_empty_mask():
    acq = torch.zeros(1, 1, 4, 4)
    lam, theta = split_acquired_mask(acq, theta_fraction=0.4)
    assert int(theta.sum().item()) == 0
    assert torch.allclose(lam, acq)


def test_ssdu_loss_consumes_strategy_context():
    """The strategy's context dict must satisfy SSDULoss's contract."""
    b, c, h, w = 1, 1, 8, 8
    pred_image = torch.randn(b, c, h, w, requires_grad=True)
    target_kspace = torch.randn(b, c, h, w, dtype=torch.complex64)
    theta = torch.zeros(b, 1, h, w)
    theta[..., ::3] = 1.0
    loss = SSDULoss()(
        prediction=pred_image,
        target=torch.zeros_like(pred_image),
        context={"target_kspace": target_kspace, "theta_mask": theta},
    )
    assert loss.requires_grad and loss.ndim == 0


def test_strategy_class_extends_reconstruction():
    assert issubclass(SSDUReconstructionStrategy, ReconstructionTrainingStrategy)


# --- Robust SSDU (Noisier2Noise) -------------------------------------------
def test_noisier_injection_zero_std_is_identity():
    """Corner case: σ=0 → no extra noise → vanilla SSDU."""
    y = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    acq = torch.zeros(1, 1, 8, 8)
    acq[..., ::2] = 1.0
    z = inject_noisier_kspace(y, acq, noise_std=0.0)
    assert torch.equal(z, y)


def test_noisier_injection_adds_complex_noise_on_acquired_only():
    y = torch.zeros(1, 1, 16, 16, dtype=torch.complex64)
    acq = torch.zeros(1, 1, 16, 16)
    acq[..., ::2] = 1.0
    z = inject_noisier_kspace(y, acq, noise_std=0.5, generator=torch.Generator().manual_seed(0))
    diff = z - y
    assert z.is_complex()
    # noise only on acquired support; unacquired entries stay exactly zero
    assert torch.all(diff[acq == 0] == 0)
    assert diff.abs().sum() > 0
    # injected magnitude std is in the right ballpark of noise_std
    acq_diff = diff[acq > 0]
    assert 0.2 < acq_diff.abs().std().item() < 0.9


def test_robust_ssdu_strategy_wires_noisier2noise():
    """The strategy must consume the noisier2noise knob + the robust_ssdu mode."""
    import inspect

    src = inspect.getsource(SSDUReconstructionStrategy)
    assert "inject_noisier_kspace" in src
    assert "noisier2noise" in src.lower()
    assert "robust_ssdu" in src


def test_it_declares_its_own_loss_ownership() -> None:
    """Issue #1918: it iterates env.losses but its loop body skips every name without 'ssdu'.

    Read off ``__dict__``, never the inherited value: this class sits under
    ``ReconstructionTrainingStrategy``, whose ``folds_image_losses = True`` is truthful for ITSELF and
    becomes a lie the moment a subclass replaces ``_compute_losses_impl``. An
    inherited True makes the audit's ``image_losses_reach_the_objective`` witness
    PASS every declared ``losses.image_losses`` entry on this strategy's arms
    while the training step discards them.
    """
    assert SSDUReconstructionStrategy.__dict__["folds_image_losses"] is False
    assert SSDUReconstructionStrategy.__dict__["inline_losses"] == frozenset()
