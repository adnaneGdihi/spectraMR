"""Tests for the frame_coherence loss (A-6.5 SCO-Frame, the one-knob penalty)."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.models.generators.tight_frame_learner import TightFrameLearner
from spectramr.models.losses.frame_coherence_loss import (
    FrameCoherenceLoss,
    gram_offdiagonal_penalty,
    smooth_max_coherence_penalty,
)
from spectramr.models.losses.registry import create_loss, list_available


def test_reads_frame_module() -> None:
    m = TightFrameLearner(num_atoms=8, kernel_size=3)
    w = m.atoms()
    assert float(FrameCoherenceLoss(penalty="frobenius")(frame=m).detach()) == pytest.approx(
        float(gram_offdiagonal_penalty(w).detach()), rel=1e-6
    )
    assert float(FrameCoherenceLoss(beta=50.0)(frame=m).detach()) == pytest.approx(
        float(smooth_max_coherence_penalty(w, 50.0).detach()), rel=1e-6
    )


def test_reads_atoms_tensor() -> None:
    atoms = torch.randn(6, 9)
    assert float(FrameCoherenceLoss(penalty="frobenius")(atoms=atoms).detach()) == pytest.approx(
        float(gram_offdiagonal_penalty(atoms).detach()), rel=1e-6
    )


def test_default_penalty_is_smooth_max() -> None:
    loss = FrameCoherenceLoss()
    assert loss.penalty == "smooth_max" and loss.beta == 1000.0


def test_orthonormal_atoms_zero() -> None:
    q = torch.linalg.qr(torch.randn(9, 4))[0].t()  # [4, 9] orthonormal rows
    assert float(FrameCoherenceLoss(penalty="frobenius")(atoms=q).detach()) == pytest.approx(
        0.0, abs=1e-5
    )
    # All six G_ij^2 are 0, so L_beta = log(6)/beta exactly (the smooth-max offset).
    loss = FrameCoherenceLoss(beta=10.0)
    assert float(loss(atoms=q).detach()) == pytest.approx(loss.penalty_offset(4), abs=1e-5)


def test_smooth_max_brackets_max_squared_coherence() -> None:
    torch.manual_seed(0)
    atoms = torch.randn(10, 6)
    wn = atoms / atoms.norm(dim=1, keepdim=True)
    g2 = (wn @ wn.t() - torch.eye(10)) ** 2
    beta = 200.0
    val = float(smooth_max_coherence_penalty(atoms, beta))
    assert float(g2.max()) - 1e-6 <= val <= float(g2.max()) + math.log(45) / beta + 1e-6


def _lines_frame(angles_deg: list[float]) -> torch.Tensor:
    a = torch.tensor(angles_deg, dtype=torch.float64) * math.pi / 180.0
    return torch.stack([a.cos(), a.sin()], dim=1)


def test_frobenius_ties_where_smooth_max_ranks_the_less_coherent_frame() -> None:
    # Two unit-norm tight frames of 4 atoms in R^2: both sit at the Frobenius floor
    # m^2/n - m = 4, but duplicated atoms have mu = 1 and the 45-degree packing mu = 0.707.
    dup = _lines_frame([0.0, 0.0, 90.0, 90.0])
    spread = _lines_frame([0.0, 45.0, 90.0, 135.0])
    assert float(gram_offdiagonal_penalty(dup)) == pytest.approx(4.0, abs=1e-9)
    assert float(gram_offdiagonal_penalty(spread)) == pytest.approx(4.0, abs=1e-9)
    assert (
        float(smooth_max_coherence_penalty(spread, 50.0))
        < float(smooth_max_coherence_penalty(dup, 50.0)) - 0.4
    )


def _descend(penalty: FrameCoherenceLoss, steps: int = 300) -> float:
    # Start next to the duplicated tight frame (Frobenius floor, mu ~ 1) and descend.
    torch.manual_seed(0)
    w = _lines_frame([0.0, 0.0, 90.0, 90.0]) + 0.01 * torch.randn(4, 2, dtype=torch.float64)
    w = w.requires_grad_(True)
    opt = torch.optim.SGD([w], lr=0.01)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        penalty(atoms=w).backward()
        opt.step()
    with torch.no_grad():
        wn = w / w.norm(dim=1, keepdim=True)
        return float((wn @ wn.t() - torch.eye(4, dtype=w.dtype)).abs().max())


def test_minimisers_differ_smooth_max_reaches_lower_mutual_coherence() -> None:
    mu_frob = _descend(FrameCoherenceLoss(penalty="frobenius"))
    mu_smooth = _descend(FrameCoherenceLoss(penalty="smooth_max", beta=50.0))
    assert mu_frob > 0.95  # already at the Frobenius floor: no pull towards lower mu
    assert mu_smooth < 0.75  # near the optimal packing cos(45 deg) = 0.707


@pytest.mark.parametrize("kwargs", [{"penalty": "max"}, {"beta": 0.0}, {"beta": float("inf")}])
def test_invalid_knobs_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        FrameCoherenceLoss(**kwargs)


def test_smooth_max_needs_two_atoms() -> None:
    with pytest.raises(ValueError, match=">= 2 atoms"):
        smooth_max_coherence_penalty(torch.randn(1, 4))


def test_differentiable_in_atoms() -> None:
    atoms = torch.randn(6, 9, requires_grad=True)
    pen = FrameCoherenceLoss()(atoms=atoms)
    pen.backward()
    assert atoms.grad is not None and torch.isfinite(atoms.grad).all()


def test_raises_without_frame_or_atoms() -> None:
    with pytest.raises(ValueError, match="PARAMETER penalty"):
        FrameCoherenceLoss()(torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8))


def test_gram_penalty_accepts_conv_weight_shape() -> None:
    # Contract: axis 0 = atoms; trailing axes flatten. A raw conv weight [m,C,k,k]
    # is accepted and equals the penalty on its [m, C*k*k] flatten.
    w = torch.randn(5, 1, 3, 3)
    assert float(gram_offdiagonal_penalty(w).detach()) == pytest.approx(
        float(gram_offdiagonal_penalty(w.reshape(5, -1)).detach()), rel=1e-6
    )


def test_gram_penalty_raises_on_rank_below_2() -> None:
    # A rank-<2 tensor has no atom/feature split -> raise, not silently reshape (#9).
    with pytest.raises(ValueError, match=">=2 dims"):
        gram_offdiagonal_penalty(torch.randn(9))


def test_accepts_positional_frame() -> None:
    m = TightFrameLearner(num_atoms=8, kernel_size=3)
    assert float(FrameCoherenceLoss()(m).detach()) == pytest.approx(
        float(smooth_max_coherence_penalty(m.atoms()).detach()), rel=1e-6
    )


def test_registered_and_constructible() -> None:
    assert "frame_coherence" in list_available()
    assert isinstance(create_loss("mutual_coherence_penalty"), FrameCoherenceLoss)  # via alias
    built = create_loss("frame_coherence", penalty="frobenius", beta=7.0)
    assert built.penalty == "frobenius" and built.beta == 7.0
