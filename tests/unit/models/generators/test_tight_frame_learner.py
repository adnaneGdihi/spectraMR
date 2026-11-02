"""Tests for TightFrameLearner — learnable sparsifying frame (A-6.5 SCO-Frame)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.tight_frame_learner import TightFrameLearner
from spectramr.models.losses.frame_coherence_loss import gram_offdiagonal_penalty


def test_forward_preserves_shape() -> None:
    m = TightFrameLearner(in_channels=1, out_channels=1, num_atoms=16, kernel_size=5)
    x = torch.rand(2, 1, 24, 24)
    assert m(x).shape == x.shape


def test_analyze_synthesize_are_adjoints() -> None:
    # <D x, c> == <x, D^T c> for random x, c (conv_transpose2d is the adjoint of conv2d).
    torch.manual_seed(0)
    m = TightFrameLearner(num_atoms=8, kernel_size=3)
    x = torch.randn(1, 1, 16, 16)
    c = torch.randn(1, 8, 16, 16)
    lhs = float((m.analyze(x) * c).sum().detach())
    rhs = float((x * m.synthesize(c)).sum().detach())
    assert lhs == pytest.approx(rhs, rel=1e-4)


def test_coherence_penalty_zero_for_orthonormal_atoms() -> None:
    # Set the atoms to mutually orthonormal rows -> Gram = I -> off-diagonal 0.
    m = TightFrameLearner(in_channels=1, num_atoms=4, kernel_size=3)  # feat = 9 >= 4
    feat = m.atoms().shape[1]
    q = torch.linalg.qr(torch.randn(feat, 4))[0].t()  # [4, feat] orthonormal rows
    with torch.no_grad():
        m.analysis.weight.copy_(q.reshape(4, 1, 3, 3))
    assert float(gram_offdiagonal_penalty(m.atoms().detach())) == pytest.approx(0.0, abs=1e-5)
    assert m.mutual_coherence() == pytest.approx(0.0, abs=1e-5)


def test_coherence_penalty_positive_for_identical_atoms() -> None:
    # All atoms equal -> Gram all-ones -> maximal coherence.
    m = TightFrameLearner(num_atoms=4, kernel_size=3)
    with torch.no_grad():
        m.analysis.weight.copy_(torch.ones_like(m.analysis.weight))
    assert float(gram_offdiagonal_penalty(m.atoms().detach())) > 1.0
    assert m.mutual_coherence() == pytest.approx(1.0, abs=1e-4)


def test_mutual_coherence_in_unit_range() -> None:
    torch.manual_seed(0)
    m = TightFrameLearner(num_atoms=12, kernel_size=3)
    mu = m.mutual_coherence()
    assert 0.0 <= mu <= 1.0 + 1e-6


def test_mutual_coherence_single_atom_is_zero() -> None:
    # Edge case (A-6.5 review): one atom has no off-diagonal pairs -> coherence 0,
    # not the -1.0 the old "subtract 2*I then max" masking produced.
    m = TightFrameLearner(num_atoms=1, kernel_size=3)
    assert m.mutual_coherence() == 0.0


def test_parseval_penalty_nonnegative_and_finite() -> None:
    m = TightFrameLearner(num_atoms=16, kernel_size=3)
    p = m.parseval_penalty()
    assert torch.isfinite(p) and float(p) >= 0.0


def test_parseval_penalty_zero_is_a_conv_parseval_frame() -> None:
    # #2368: W^T W = I/n makes D^T D = I; the old (m/n) target made D^T D = m I.
    torch.manual_seed(0)
    m_atoms, k = 64, 7
    n = k * k
    model = TightFrameLearner(num_atoms=m_atoms, kernel_size=k, sparsity_threshold=0.05).double()
    q, _ = torch.linalg.qr(torch.randn(m_atoms, n, dtype=torch.float64))
    x = torch.randn(1, 1, 48, 48, dtype=torch.float64)
    with torch.no_grad():
        model.analysis.weight.copy_((q / n**0.5).reshape(m_atoms, 1, k, k))
        dtdx = model.synthesize(model.analyze(x))
    assert float(model.parseval_penalty()) < 1e-20
    assert torch.allclose(dtdx[..., 8:-8, 8:-8], x[..., 8:-8, 8:-8], atol=1e-10)
    with torch.no_grad():
        model.analysis.weight.copy_(((m_atoms / n) ** 0.5 * q).reshape(m_atoms, 1, k, k))
    assert float(model.parseval_penalty()) > 1.0


# --------------------------------------------------------------------------- #
# THE LOAD-BEARING ORACLE: the coherence penalty actually lowers the learned
# mutual coherence vs an un-penalised control AT MATCHED RECONSTRUCTION.
# --------------------------------------------------------------------------- #


def _train_frame(use_coherence: bool, steps: int = 150) -> tuple[float, float]:
    """Train a frame to denoise a fixed batch; return (final recon MSE, mutual coherence)."""
    torch.manual_seed(0)
    m = TightFrameLearner(num_atoms=24, kernel_size=5, sparsity_threshold=0.02)
    # A fixed low-rank-ish "clean" batch + noise -> a real reconstruction task.
    torch.manual_seed(1)
    clean = torch.rand(4, 1, 20, 20)
    noisy = clean + 0.1 * torch.randn_like(clean)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    mse = torch.nn.functional.mse_loss
    last_recon = 0.0
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        recon = mse(m(noisy), clean)
        loss = recon + 0.01 * m.parseval_penalty()
        if use_coherence:
            loss = loss + 0.05 * gram_offdiagonal_penalty(m.atoms())
        loss.backward()
        opt.step()
        last_recon = float(recon.detach())
    return last_recon, m.mutual_coherence()


def test_coherence_penalty_lowers_learned_coherence_at_matched_reconstruction() -> None:
    recon_off, mu_off = _train_frame(use_coherence=False)
    recon_on, mu_on = _train_frame(use_coherence=True)
    # The coherence-penalised frame is measurably LESS coherent...
    assert mu_on < mu_off
    # ...while reconstruction stays comparable (matched task, not wrecked by the penalty).
    assert recon_on < recon_off * 3.0


def test_in_out_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="in==out"):
        TightFrameLearner(in_channels=1, out_channels=2)


def test_even_kernel_raises() -> None:
    with pytest.raises(ValueError, match="odd"):
        TightFrameLearner(kernel_size=4)


def test_registered_as_model() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert "tight_frame_learner" in MODEL_REGISTRY
