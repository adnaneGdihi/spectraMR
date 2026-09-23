"""Tests for complex-valued losses.

Targets ``spectramr.models.losses.complex_losses``:

- ``ComplexL1Loss`` — magnitude of complex residual
- ``WeightedKSpaceL1Loss`` — frequency-weighted L1 (high-freq emphasis)
- ``LogSpectralPhaseLoss`` — log-magnitude + magnitude-weighted phase
- ``ComplexMSELoss`` — Re² + Im² of residual
- ``FrequencyDomainLoss`` — FFT-based image-vs-image loss
- ``SENSEAdjointL1Loss`` — SENSE-projected complex L1 (skipped here;
  needs k-space + smaps fixtures, covered in integration tier)

Common contracts: zero-residual ⇒ near-zero loss, gradient flow,
reduction modes, registry presence.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.complex_losses import (
    ComplexL1Loss,
    ComplexMSELoss,
    FrequencyDomainLoss,
    LogSpectralPhaseLoss,
    SENSEAdjointL1Loss,
    WeightedKSpaceL1Loss,
)


def _complex_tensor(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    """Helper: random complex tensor."""
    torch.manual_seed(seed)
    return torch.complex(torch.randn(*shape), torch.randn(*shape))


# ---------------------------------------------------------------------------
# ComplexL1Loss
# ---------------------------------------------------------------------------


def test_complex_l1_zero_residual_near_zero() -> None:
    """``loss(x, x)`` is bounded by the eps-stabilisation term."""
    loss_fn = ComplexL1Loss()
    x = _complex_tensor((1, 2, 8, 8))
    out = loss_fn(x, x)
    # eps = 1e-8 → loss ≈ eps per voxel
    assert out.item() < 1e-6


def test_complex_l1_gradient_flows_to_input() -> None:
    """Backward through the loss reaches the input tensor."""
    loss_fn = ComplexL1Loss()
    real = torch.randn(1, 2, 8, 8, requires_grad=True)
    target = torch.randn(1, 2, 8, 8)
    out = loss_fn(real, target)
    out.backward()
    assert real.grad is not None
    assert torch.isfinite(real.grad).all()


def test_complex_l1_reduction_modes() -> None:
    """``mean``, ``sum``, default-not-mean reductions all return finite values."""
    loss_mean = ComplexL1Loss(reduction="mean")
    loss_sum = ComplexL1Loss(reduction="sum")
    x = _complex_tensor((1, 2, 4, 4))
    y = _complex_tensor((1, 2, 4, 4), seed=1)
    m = loss_mean(x, y)
    s = loss_sum(x, y)
    assert m.dim() == 0
    assert s.dim() == 0
    # sum >= mean (since loss is non-negative and per-voxel)
    assert s.item() >= m.item()


def test_complex_l1_real_2channel_input() -> None:
    """``[B, 2, H, W]`` real-stacked input is auto-converted to complex."""
    loss_fn = ComplexL1Loss()
    pred = torch.randn(1, 2, 8, 8)
    target = torch.zeros(1, 2, 8, 8)
    out = loss_fn(pred, target)
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# WeightedKSpaceL1Loss — frequency emphasis
# ---------------------------------------------------------------------------


def test_weighted_kspace_uses_radial_weight() -> None:
    """Higher exponent → higher loss for high-frequency-only differences."""
    loss_low = WeightedKSpaceL1Loss(exponent=0.0)  # uniform weight
    loss_high = WeightedKSpaceL1Loss(exponent=2.0)  # strong high-freq emphasis

    # Plant a residual only at the corners (high frequency).
    pred = torch.zeros(1, 1, 16, 16, dtype=torch.complex64)
    target = torch.zeros_like(pred)
    target[..., 0, 0] = 1.0 + 0.0j  # corner = high frequency

    out_low = loss_low(pred, target)
    out_high = loss_high(pred, target)
    assert out_high.item() > out_low.item()


def test_weighted_kspace_zero_residual_is_zero() -> None:
    """``loss(x, x) == 0``."""
    loss_fn = WeightedKSpaceL1Loss(exponent=1.0)
    x = _complex_tensor((1, 1, 8, 8))
    out = loss_fn(x, x)
    assert out.item() == 0.0


def test_weighted_kspace_caches_weight_mask() -> None:
    """Calling twice with same shape reuses cached mask."""
    loss_fn = WeightedKSpaceL1Loss(exponent=1.0)
    x = _complex_tensor((1, 1, 8, 8))
    y = _complex_tensor((1, 1, 8, 8), seed=1)
    _ = loss_fn(x, y)
    _ = loss_fn(x, y)
    cache_keys = list(loss_fn._weight_cache.keys())
    assert len(cache_keys) == 1


# ---------------------------------------------------------------------------
# LogSpectralPhaseLoss
# ---------------------------------------------------------------------------


def test_log_spectral_phase_zero_residual_near_zero() -> None:
    """``loss(x, x)`` is bounded by eps-stabilisation."""
    loss_fn = LogSpectralPhaseLoss(phase_weight=0.1)
    x = _complex_tensor((1, 1, 8, 8))
    out = loss_fn(x, x)
    assert out.item() < 1e-3


def test_log_spectral_phase_phase_weight_scales_loss() -> None:
    """Larger ``phase_weight`` → larger loss when only phase differs."""
    pred_mag = torch.ones(1, 1, 8, 8, dtype=torch.complex64)
    target_mag = pred_mag.clone()
    # Same magnitude, different phase
    target_mag = target_mag * torch.exp(1j * torch.full((1, 1, 8, 8), 0.5))

    loss_low = LogSpectralPhaseLoss(phase_weight=0.0)
    loss_high = LogSpectralPhaseLoss(phase_weight=1.0)

    out_low = loss_low(pred_mag, target_mag)
    out_high = loss_high(pred_mag, target_mag)
    assert out_high.item() > out_low.item()


# ---------------------------------------------------------------------------
# ComplexMSELoss
# ---------------------------------------------------------------------------


def test_complex_mse_zero_residual_zero() -> None:
    """MSE on identity input is exactly 0."""
    loss_fn = ComplexMSELoss()
    x = _complex_tensor((1, 1, 8, 8))
    assert loss_fn(x, x).item() == 0.0


def test_complex_mse_quadratic_in_residual() -> None:
    """Doubling residual → 4× loss (quadratic)."""
    loss_fn = ComplexMSELoss(reduction="mean")
    target = torch.zeros(1, 1, 8, 8, dtype=torch.complex64)
    pred1 = torch.full((1, 1, 8, 8), 1.0 + 0.0j)
    pred2 = torch.full((1, 1, 8, 8), 2.0 + 0.0j)
    l1 = loss_fn(pred1, target).item()
    l2 = loss_fn(pred2, target).item()
    assert abs(l2 / l1 - 4.0) < 1e-5


# ---------------------------------------------------------------------------
# FrequencyDomainLoss
# ---------------------------------------------------------------------------


def test_frequency_domain_zero_residual_near_zero() -> None:
    """``loss(x, x)`` is small (< 1e-3) due to log-eps stabilisation."""
    loss_fn = FrequencyDomainLoss(use_log=True)
    x = torch.randn(1, 1, 8, 8)
    out = loss_fn(x, x)
    assert out.item() < 1e-3


def test_frequency_domain_complex_input_collapses_to_magnitude() -> None:
    """Complex input is reduced to magnitude before FFT."""
    loss_fn = FrequencyDomainLoss(use_log=False)
    cplx = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    out = loss_fn(cplx, cplx)
    # Same input → near-zero loss.
    assert out.item() < 1e-3


def test_frequency_domain_reduction_modes() -> None:
    """All reduction modes return finite values."""
    loss_fn_none = FrequencyDomainLoss(reduction="none")
    pred = torch.randn(1, 1, 8, 8)
    target = torch.randn(1, 1, 8, 8)
    out = loss_fn_none(pred, target)
    assert out.shape == pred.shape


# ---------------------------------------------------------------------------
# Sanity-shape matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 8, 8), (2, 2, 16, 16), (1, 4, 16, 32)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_complex_l1_shape_matrix(shape: tuple[int, ...]) -> None:
    """ComplexL1Loss handles any ``[B, C, H, W]`` shape (C even)."""
    loss_fn = ComplexL1Loss()
    pred = torch.randn(*shape)
    target = torch.randn(*shape)
    out = loss_fn(pred, target)
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_complex_losses_registered() -> None:
    """All five complex losses are registered under their canonical names."""
    from spectramr.models.losses.registry import list_available

    available = set(list_available())
    expected = {
        "complex_l1",
        "complex_mse",
        "weighted_kspace_l1",
        "log_spectral_phase",
        "frequency_domain_consistency",
    }
    assert expected <= available, f"missing: {expected - available}"


# ---------------------------------------------------------------------------
# SENSEAdjointL1Loss — missing-smaps guard (DC-blob root-cause pattern)
# ---------------------------------------------------------------------------


def test_sense_adjoint_l1_raises_on_missing_smaps_during_training() -> None:
    """A 0.3-weighted SENSE fidelity term must NOT silently zero in training.

    Returning ``pred.sum() * 0.0`` when smaps is absent is the exact DC-blob
    root-cause pattern (a data-fidelity term multiplied by zero), so during
    training (``self.training`` True) it must raise instead.
    """
    loss = SENSEAdjointL1Loss()
    loss.train()
    pred = torch.randn(1, 2, 8, 8)
    target = torch.randn(1, 2, 8, 8)
    with pytest.raises(ValueError, match="smaps=None during training"):
        loss(pred, target, smaps=None)


def test_sense_adjoint_l1_zero_in_eval_mode() -> None:
    """In eval mode (startup / no-coil audit) missing smaps -> legitimate 0."""
    loss = SENSEAdjointL1Loss()
    loss.eval()
    pred = torch.randn(1, 2, 8, 8)
    target = torch.randn(1, 2, 8, 8)
    assert loss(pred, target, smaps=None).item() == 0.0


def test_sense_adjoint_l1_zero_with_explicit_optout() -> None:
    """An explicit ``allow_missing_smaps`` opt-out preserves the zero path."""
    loss = SENSEAdjointL1Loss()
    loss.train()
    loss.allow_missing_smaps = True
    pred = torch.randn(1, 2, 8, 8)
    target = torch.randn(1, 2, 8, 8)
    assert loss(pred, target, smaps=None).item() == 0.0


def test_sense_adjoint_l1_raises_on_non_finite_smaps_during_training() -> None:
    """Non-finite coil maps (rank-deficient ESPIRiT) must raise in training,
    not poison the 0.3-weighted fidelity gradient (the exp_11 divergence)."""
    loss = SENSEAdjointL1Loss()
    loss.train()
    pred = torch.randn(1, 2, 8, 8)
    target = torch.randn(1, 2, 8, 8)
    smaps = torch.randn(1, 2, 8, 8)
    smaps[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite smaps"):
        loss(pred, target, smaps=smaps)


def test_sense_adjoint_l1_tolerates_non_finite_smaps_in_eval() -> None:
    """The finite-guard is train-only: eval must not raise (validation runs on
    possibly-degraded maps without crashing)."""
    loss = SENSEAdjointL1Loss()
    loss.eval()
    pred = torch.randn(1, 2, 8, 8)
    target = torch.randn(1, 2, 8, 8)
    smaps = torch.randn(1, 2, 8, 8)
    smaps[0, 0, 0, 0] = float("inf")
    loss(pred, target, smaps=smaps)  # no exception


# ─────────────────────────────────────────────────────────────────────────────
# Fourier-bridge contract (issue #467)
#
# SENSEAdjointL1Loss bridges from k-space itself: it iFFTs the prediction before
# the coil-adjoint combination. The whole kspace_filling cohort declared it under
# ``losses.image_losses`` with ``losses.output_domain: kspace``, so LossBuilder
# wrapped it in a SECOND iFFT->magnitude bridge. The loss then re-paired the
# 4 real magnitude channels into 2 "complex" coils and silently sliced its 4-coil
# sensitivity maps down to 2 to match. Finite value, no warning, wrong objective.
# ─────────────────────────────────────────────────────────────────────────────

_COILS = 4


def _interleaved_kspace(img: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] complex image -> [B, 2C, H, W] real-interleaved k-space."""
    from spectramr.infrastructure.physics.fft_ops import fft2c

    k = fft2c(img)
    return torch.stack([k.real, k.imag], dim=2).flatten(1, 2)


def _outer_bridge(kspace: torch.Tensor) -> torch.Tensor:
    """What LossBuilder's ``image_losses`` wrapper hands the loss."""
    from spectramr.models.losses.physics_losses import DifferentiableFourierBridge

    return DifferentiableFourierBridge(spatial_loss_fn=None, return_complex=False)._kspace_to_image(
        kspace
    )


def _phase_only_pair() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """k_pred, k_target with identical per-coil magnitudes but different phase."""
    torch.manual_seed(0)
    shape = (1, _COILS, 16, 16)
    mag = torch.rand(shape) + 0.5
    img_t = mag * torch.exp(1j * torch.rand(shape) * 2 * torch.pi)
    img_p = mag * torch.exp(1j * torch.rand(shape) * 2 * torch.pi)
    smaps = torch.complex(torch.rand(shape) + 0.5, torch.rand(shape) * 0.1)
    return _interleaved_kspace(img_p), _interleaved_kspace(img_t), smaps


def test_sense_adjoint_l1_exposes_the_fourier_bridge_flag() -> None:
    """The attribute LossBuilder's double-bridge guard inspects."""
    assert SENSEAdjointL1Loss().use_fourier_bridge is True
    assert SENSEAdjointL1Loss(use_fourier_bridge=False).use_fourier_bridge is False


def test_sense_adjoint_l1_declares_the_kspace_domain() -> None:
    from spectramr.models.losses.registry import LossRegistry

    caps = LossRegistry.get_loss_capabilities("sense_adjoint_l1")
    assert caps is not None, "registration must carry domain metadata"
    assert caps.domain == "kspace"


def test_sense_adjoint_l1_single_bridge_sees_the_phase_error() -> None:
    """Correct placement (kspace_losses): the SENSE-adjoint image is complex, so a
    pure phase error registers."""
    k_pred, k_target, smaps = _phase_only_pair()
    loss = SENSEAdjointL1Loss()
    loss.train()

    value = loss(k_pred, k_target, smaps=smaps)
    assert torch.isfinite(value)
    assert value > 1e-3, f"phase error must register; got {value.item():.3e}"


def test_sense_adjoint_l1_double_bridge_is_phase_blind() -> None:
    """#467: the pre-fix path scored ~0 on a pure phase error.

    Both sides collapse to the same magnitude image before the loss ever runs, so
    a 0.3-weighted data-fidelity term contributed nothing on phase.
    """
    k_pred, k_target, smaps = _phase_only_pair()
    loss = SENSEAdjointL1Loss()
    loss.train()

    doubled = loss(_outer_bridge(k_pred), _outer_bridge(k_target), smaps=smaps)
    single = loss(k_pred, k_target, smaps=smaps)

    assert doubled < 1e-6, f"expected a collapsed value; got {doubled.item():.3e}"
    assert single > 1e3 * doubled


def test_sense_adjoint_l1_bridge_flag_off_skips_the_internal_ifft() -> None:
    """With an outer bridge in play the transform must be applied exactly once."""
    k_pred, k_target, smaps = _phase_only_pair()
    once = SENSEAdjointL1Loss(use_fourier_bridge=False)
    once.train()
    twice = SENSEAdjointL1Loss(use_fourier_bridge=True)
    twice.train()

    img_pred, img_target = _outer_bridge(k_pred), _outer_bridge(k_target)
    # Same inputs, and the only difference is the skipped iFFT.
    assert not torch.isclose(
        once(img_pred, img_target, smaps=smaps),
        twice(img_pred, img_target, smaps=smaps),
    )


# ---------------------------------------------------------------------------
# SENSEAdjointL1Loss(normalize=...) — the shading is a spatial weight on a
# fidelity term, and nobody chose it.
#
# A^H returns (sum_c |S_c|^2) * m, so the SAME error costs less where the array
# is less sensitive. That is a per-subject, per-slice reweighting of a
# data-fidelity term. `normalize` divides it out through `coil_combine_sense`,
# the owner of that divide and of the support floor it needs.
# ---------------------------------------------------------------------------


def _sense_phantom(seed: int = 0):
    """``(k_target, smaps, m)`` with deliberately SHADED maps (not unit RSS)."""
    import torch

    from spectramr.infrastructure.physics.fft_ops import fft2c

    torch.manual_seed(seed)
    b, c, s = 2, 4, 16
    m = torch.randn(b, 1, s, s, dtype=torch.complex64)
    maps = torch.randn(b, c, s, s, dtype=torch.complex64)
    maps = maps / maps.abs().pow(2).sum(1, keepdim=True).sqrt()
    maps = maps * (0.3 + 0.7 * torch.rand(b, 1, s, s))
    return fft2c(maps * m), maps, m


def test_normalize_defaults_off_so_declaring_arms_do_not_move():
    """68 cohort arms plus 6 elsewhere declare this term; the default is the
    pre-change matched filter, bit-for-bit."""
    import torch

    from spectramr.infrastructure.physics.fft_ops import ifft2c

    k, maps, _ = _sense_phantom()
    assert SENSEAdjointL1Loss().normalize is False
    got = SENSEAdjointL1Loss().eval()(k, k, smaps=maps)
    expected = torch.nn.functional.l1_loss(
        torch.view_as_real((ifft2c(k) * maps.conj()).sum(1, keepdim=True)),
        torch.view_as_real((ifft2c(k) * maps.conj()).sum(1, keepdim=True)),
    )
    assert torch.equal(got, expected)


def test_the_adjoint_weights_the_same_error_by_where_it_lands():
    """The planted defect, stated as a measurement: one identical perturbation,
    two locations, and the loss differs by the coil intensity ratio."""
    import torch

    from spectramr.infrastructure.physics.fft_ops import fft2c

    k, maps, m = _sense_phantom()
    support = (maps.abs() ** 2).sum(1, keepdim=True)
    weak = support.flatten(2).argmin(-1)
    strong = support.flatten(2).argmax(-1)

    def perturbed(idx):
        d = torch.zeros_like(m)
        for b in range(m.shape[0]):
            d[b, 0].view(-1)[idx[b, 0]] = 0.5
        return fft2c(maps * (m + d))

    adjoint = SENSEAdjointL1Loss().eval()
    lo = float(adjoint(perturbed(weak), k, smaps=maps))
    hi = float(adjoint(perturbed(strong), k, smaps=maps))
    assert hi > 3 * lo, f"expected a strong shading, got {hi / lo:.2f}x"


def test_normalize_removes_that_weighting():
    """The fix, stated as the same measurement: the ratio goes to one."""
    import torch

    from spectramr.infrastructure.physics.fft_ops import fft2c

    k, maps, m = _sense_phantom()
    support = (maps.abs() ** 2).sum(1, keepdim=True)
    weak = support.flatten(2).argmin(-1)
    strong = support.flatten(2).argmax(-1)

    def perturbed(idx):
        d = torch.zeros_like(m)
        for b in range(m.shape[0]):
            d[b, 0].view(-1)[idx[b, 0]] = 0.5
        return fft2c(maps * (m + d))

    roemer = SENSEAdjointL1Loss(normalize=True).eval()
    lo = float(roemer(perturbed(weak), k, smaps=maps))
    hi = float(roemer(perturbed(strong), k, smaps=maps))
    assert abs(hi / lo - 1.0) < 0.05, f"expected parity, got {hi / lo:.3f}x"


def test_normalize_is_zero_on_an_exact_match_and_differentiable():
    import torch

    k, maps, _ = _sense_phantom()
    pred = k.clone().requires_grad_(True)
    loss = SENSEAdjointL1Loss(normalize=True).eval()
    assert float(loss(k, k, smaps=maps)) < 1e-6
    loss(pred + 0.1, k, smaps=maps).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_the_support_floor_is_reachable_from_the_yaml_kwargs():
    """``min_support_frac`` is the knob that stops out-of-support air noise from
    becoming the dominant term; an unread knob here would be pitfall #15."""
    import torch

    from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

    k, maps, _m = _sense_phantom()
    dead = maps.clone()
    dead[..., :4, :] *= 1e-4
    noisy = fft2c(ifft2c(k) + 0.01 * torch.randn_like(k))

    floored = float(SENSEAdjointL1Loss(normalize=True).eval()(noisy, k, smaps=dead))
    unprotected = float(
        SENSEAdjointL1Loss(normalize=True, min_support_frac=0.0).eval()(noisy, k, smaps=dead)
    )
    assert unprotected > 5 * floored, (unprotected, floored)
