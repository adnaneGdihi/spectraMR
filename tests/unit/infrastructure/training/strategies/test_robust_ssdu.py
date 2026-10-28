r"""The Noisier2Noise correction makes Robust SSDU unbiased where SSDU is not.

Per k-space location on the lines the network input carries: ``y = k0 + n`` with
``n ~ CN(0, s^2)``, ``y~ = y + n~`` with ``n~ ~ CN(0, a^2 s^2)``. Robust SSDU scores
``f(y~)`` against ``y`` there, so ``f`` converges to ``E[y | y~]``, which still
contains ``E[n | y~]``; the correction ``((1+a^2) f - y~) / a^2`` removes it.

"Unbiased" is tested as conditional unbiasedness, ``E[k_hat - k0 | y~] = 0``: for a
linear estimate of a Gaussian that is orthogonality of the error to ``y~``, which a
regression coefficient measures. An unconditional mean would not separate the two
estimators -- both are zero-mean here.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.training.strategies.robust_ssdu import (
    noisier2noise_correct,
    robust_loss_weights,
)
from spectramr.infrastructure.training.strategies.ssdu_strategy import inject_noisier_kspace

ALPHA = 0.75
SIGMA = 0.5
TAU = 1.0
N = 200_000


def _draw(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(k0, y, y~)`` for N independent locations, all on the acquired support."""
    g = torch.Generator().manual_seed(seed)
    k0 = (TAU / math.sqrt(2.0)) * torch.complex(
        torch.randn(1, 1, 1, N, generator=g), torch.randn(1, 1, 1, N, generator=g)
    )
    acquired = torch.ones(1, 1, 1, N)
    y = inject_noisier_kspace(k0, acquired, SIGMA, generator=g)
    y_tilde = inject_noisier_kspace(y, acquired, ALPHA * SIGMA, generator=g)
    return k0, y, y_tilde


def _slope(error: torch.Tensor, regressor: torch.Tensor) -> float:
    """Least-squares coefficient of ``error`` on ``regressor``: 0 iff orthogonal."""
    return float((error * regressor.conj()).sum().real / regressor.abs().pow(2).sum())


def _fit_robust_loss(y: torch.Tensor, y_tilde: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """The linear ``f(y~) = w y~`` that minimises Robust SSDU's weighted loss."""
    weights = robust_loss_weights(lam, torch.ones_like(lam), ALPHA).pow(2)
    w = (weights * y * y_tilde.conj()).sum() / (weights * y_tilde.abs().pow(2)).sum()
    return w * y_tilde


def test_the_oracle_conditional_mean_is_corrected_exactly() -> None:
    """With k0 known, E[y | y~] = k0 + (y~ - k0)/(1+a^2), and the correction is k0."""
    k0, _, y_tilde = _draw()
    f_star = k0 + (y_tilde - k0) / (1.0 + ALPHA**2)
    lam = torch.ones(1, 1, 1, N)
    corrected = noisier2noise_correct(f_star, y_tilde, lam, ALPHA)
    assert torch.allclose(corrected, k0, atol=1e-4)
    assert not torch.allclose(f_star, k0, atol=1e-2), "the uncorrected estimate is noisy"


def test_the_corrected_estimate_is_unbiased_on_the_input_lines_and_the_raw_one_is_not() -> None:
    """The planted Gaussian case the correction exists for.

    Trained through the weighted loss, ``f`` recovers E[y | y~] with coefficient
    (t^2+s^2)/(t^2+(1+a^2)s^2); corrected, it is E[k0 | y~] = t^2/(t^2+(1+a^2)s^2).
    """
    k0, y, y_tilde = _draw(1)
    lam = torch.ones(1, 1, 1, N)
    raw = _fit_robust_loss(y, y_tilde, lam)
    corrected = noisier2noise_correct(raw, y_tilde, lam, ALPHA)

    raw_bias = _slope(raw - k0, y_tilde)
    corrected_bias = _slope(corrected - k0, y_tilde)
    expected_raw = SIGMA**2 / (TAU**2 + (1 + ALPHA**2) * SIGMA**2)
    assert abs(corrected_bias) < 0.01, corrected_bias
    assert raw_bias == pytest.approx(expected_raw, abs=0.01)


def test_a_misspecified_alpha_leaves_a_bias() -> None:
    """Red if the factor or the alpha it is given is wrong: the correction is not generic."""
    k0, y, y_tilde = _draw(2)
    lam = torch.ones(1, 1, 1, N)
    raw = _fit_robust_loss(y, y_tilde, lam)
    wrong = noisier2noise_correct(raw, y_tilde, lam, 2.0 * ALPHA)
    assert abs(_slope(wrong - k0, y_tilde)) > 0.05


def test_the_correction_leaves_lines_the_input_did_not_carry() -> None:
    predicted = torch.randn(1, 1, 4, 4, dtype=torch.complex64)
    network_input = torch.randn(1, 1, 4, 4, dtype=torch.complex64)
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., ::2, :] = 1.0
    out = noisier2noise_correct(predicted, network_input, mask, ALPHA)
    assert torch.equal(out[..., 1::2, :], predicted[..., 1::2, :])
    expected = ((1 + ALPHA**2) * predicted - network_input) / ALPHA**2
    assert torch.allclose(out[..., ::2, :], expected[..., ::2, :])


def test_the_loss_weights_follow_eq_35() -> None:
    acquired = torch.tensor([[[[1.0, 1.0, 1.0, 0.0]]]])
    lam = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
    weights = robust_loss_weights(lam, acquired, ALPHA)
    on_lambda = (1 + ALPHA**2) / ALPHA**2
    assert torch.allclose(weights, torch.tensor([[[[on_lambda, 1.0, on_lambda, 0.0]]]]))


@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_a_non_positive_alpha_raises(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha > 0"):
        robust_loss_weights(torch.ones(1), torch.ones(1), alpha)
    with pytest.raises(ValueError, match="alpha > 0"):
        noisier2noise_correct(torch.ones(1), torch.ones(1), torch.ones(1), alpha)


def test_the_committed_coil_shape_colours_the_injection() -> None:
    """Eq. 29 needs the injected noise distributed as alpha times the measurement noise;
    M4Raw's coil noise is correlated, so a white draw left the correction approximate."""
    from spectramr.infrastructure.training.strategies.ssdu_strategy import _committed_coil_factor

    factor = _committed_coil_factor()
    shape = factor @ factor.conj().T
    assert float(shape.diagonal().real.sum()) == pytest.approx(4.0, rel=1e-5)
    y = torch.zeros(4000, 4, 4, 4, dtype=torch.complex64)
    acquired = torch.ones(1, 1, 4, 4)
    gen = torch.Generator().manual_seed(0)
    coloured = inject_noisier_kspace(y, acquired, 0.1, generator=gen, coil_factor=factor)
    draws = coloured.permute(1, 0, 2, 3).reshape(4, -1)
    empirical = draws @ draws.conj().T / draws.shape[1]
    assert float((empirical - 0.01 * shape).abs().max()) < 0.02 * 0.01 * 4
    # Planted: the white draw has no coil correlation.
    white = inject_noisier_kspace(y, acquired, 0.1, generator=torch.Generator().manual_seed(0))
    wd = white.permute(1, 0, 2, 3).reshape(4, -1)
    white_cov = wd @ wd.conj().T / wd.shape[1]
    off = ~torch.eye(4, dtype=torch.bool)
    assert float(shape[off].abs().max()) > 5 * float(white_cov[off].abs().max() / 0.01)


def test_a_mismatched_coil_factor_raises() -> None:
    y = torch.zeros(1, 2, 4, 4, dtype=torch.complex64)
    with pytest.raises(ValueError, match="does not match the 2 coils"):
        inject_noisier_kspace(
            y, torch.ones(1, 1, 4, 4), 0.1, coil_factor=torch.eye(4, dtype=torch.complex64)
        )


def test_the_module_does_not_claim_the_p_weight_is_one() -> None:
    """PAPER: Omega is drawn per slice, so P is about 13.7 off the ACS lines and Claim 3 does not hold."""
    from spectramr.infrastructure.training.strategies import robust_ssdu

    assert "is 1 here" not in (robust_ssdu.__doc__ or "")
    assert "Claim 3" in (robust_ssdu.__doc__ or "")
