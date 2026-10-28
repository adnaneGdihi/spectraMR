"""Unit tests for BeltramiDiagnosticLoss.

Pin the fundamental quasiconformal-geometry properties:

- Identity displacement ⇒ |mu| = 0 (no distortion).
- Pure shear ⇒ |mu| > 0 (smooth distortion).
- Loss is autograd-traceable end-to-end.
- Output shape is correct for both 2D (B,2,H,W) and 3D (B,2,D,H,W).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.losses.beltrami_diagnostic_loss import (  # noqa: E402
    BeltramiDiagnosticLoss,
)


def test_identity_field_has_zero_beltrami() -> None:
    """f(z) = z everywhere ⇒ ∂f/∂z̄ = 0 ⇒ μ ≡ 0."""
    H = W = 16
    # Identity displacement: u(x,y) = x, v(x,y) = y. The constant
    # field (u=0, v=0) is also conformal (just translation), so we
    # use that as the simpler analytic case.
    field = torch.zeros(1, 2, H, W)
    loss = BeltramiDiagnosticLoss(reduction="mean")(field)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_shear_field_has_nonzero_beltrami() -> None:
    """A pure shear deformation has positive |μ|."""
    H = W = 16
    # Displacement u = a*y, v = eps*x: the map z + d has |mu| = |a + eps| / |2 + i(eps - a)|.
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    a = 0.5
    eps = 0.1
    u = a * yy
    v = eps * xx
    field = torch.stack([u, v], dim=0).unsqueeze(0)  # (1, 2, H, W)
    loss = BeltramiDiagnosticLoss(reduction="mean")(field)
    assert loss.item() > 0.0


def test_loss_is_autograd_traceable() -> None:
    """Backprop through Beltrami must produce non-trivial gradients."""
    field = torch.randn(1, 2, 16, 16, requires_grad=True)
    loss = BeltramiDiagnosticLoss(reduction="mean")(field)
    loss.backward()
    assert field.grad is not None
    assert torch.any(field.grad != 0)


def test_3d_input_per_slice_average() -> None:
    """Per-slice 2D Beltrami over a 3D volume must produce a scalar."""
    field = torch.randn(2, 2, 4, 16, 16)  # (B, 2, D, H, W)
    loss = BeltramiDiagnosticLoss(reduction="mean")(field)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


def test_pred_target_fallback_signature() -> None:
    """forward(pred, target) treats residual as a (u, v) field."""
    pred = torch.randn(1, 2, 16, 16)
    target = torch.randn(1, 2, 16, 16)
    loss = BeltramiDiagnosticLoss(reduction="mean")(pred, target)
    assert torch.isfinite(loss)


def test_pred_target_fallback_rejects_one_channel_intensity_residual() -> None:
    """A 1-channel intensity residual is not a displacement and must raise."""
    pred = torch.randn(1, 1, 16, 16)
    target = torch.randn(1, 1, 16, 16)
    with pytest.raises(ValueError, match="displacement"):
        BeltramiDiagnosticLoss(reduction="mean")(pred, target)


def test_3d_zero_displacement_has_zero_beltrami() -> None:
    field = torch.zeros(1, 3, 4, 8, 8)
    assert BeltramiDiagnosticLoss(reduction="infty")(torch.zeros(0), displacement=field).item() == (
        pytest.approx(0.0, abs=1e-6)
    )


def test_3d_planted_inplane_shear_has_positive_beltrami() -> None:
    """u = a * y on every slice, with a z component present, gives |mu| > 0."""
    H = W = 8
    y = torch.arange(H, dtype=torch.float32).view(1, 1, 1, H, 1).expand(1, 1, 4, H, W)
    field = torch.zeros(1, 3, 4, H, W)
    field[:, 0:1] = 0.2 * y
    field[:, 2] = 0.1
    value = BeltramiDiagnosticLoss(reduction="infty")(torch.zeros(0), displacement=field)
    assert value.item() > 0.0


def test_matches_the_epi_beltrami_for_a_pure_pe_shift() -> None:
    """Characterisation against infrastructure's EPI Beltrami, without importing it here."""
    from spectramr.infrastructure.physics.epi_forward import beltrami_from_b0_and_gnl

    H = W = 12
    a = 0.1
    y = torch.arange(H, dtype=torch.float64).view(1, 1, H, 1).expand(1, 1, H, W)
    # The EPI model shifts by n_pe * t_esp * df; with t_esp = 1 and n_pe = H, df = a*y/H
    # gives a displacement of a*y along PE (y).
    reference = beltrami_from_b0_and_gnl(
        a * y / H, torch.zeros(1, 2, H, W, dtype=torch.float64), t_esp=1.0, phase_encode_axis=-2
    )
    field = torch.zeros(1, 2, H, W, dtype=torch.float64)
    field[:, 1:2] = a * y
    ours = BeltramiDiagnosticLoss(eps=0.0)._beltrami_2d(field)
    torch.testing.assert_close(ours[:, 1:-1, 1:-1], reference.abs()[:, 1:-1, 1:-1])


def test_penalty_reduction_zero_for_quasiconformal() -> None:
    """`relu(|mu| - 1)^2` is exactly 0 for fields with |mu| <= 1."""
    # Zero field has mu = 0, so penalty must be 0.
    field = torch.zeros(1, 2, 16, 16)
    loss = BeltramiDiagnosticLoss(reduction="penalty")(field)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def _grid(h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.meshgrid(
        torch.arange(h, dtype=torch.float64),
        torch.arange(w, dtype=torch.float64),
        indexing="ij",
    )


def test_small_shear_displacement_gives_small_mu() -> None:
    """d = eps * conj(z) makes z + d a near-identity map with |mu| = eps.

    Reading the displacement as the map itself (the old formula) put 2*eps over a
    vanishing denominator and reported |mu| = 20 for eps = 0.01.
    """
    yy, xx = _grid(16, 16)
    eps = 0.01
    field = torch.stack([eps * xx, -eps * yy]).unsqueeze(0)
    mu = BeltramiDiagnosticLoss(reduction="infty")(field, displacement=field)
    assert mu.item() == pytest.approx(eps, rel=1e-3)
    assert BeltramiDiagnosticLoss(reduction="penalty")(field, displacement=field).item() == 0.0


def test_matches_the_physics_beltrami_on_the_interior() -> None:
    from spectramr.infrastructure.physics.beltrami_faithfulness import (
        beltrami_from_displacement,
    )

    torch.manual_seed(0)
    field = 0.3 * torch.randn(2, 2, 12, 12, dtype=torch.float64)
    ours = BeltramiDiagnosticLoss(reduction="mean", eps=0.0)._beltrami_2d(field)
    reference = beltrami_from_displacement(field).abs()
    # The two differ only in how the border derivative is padded.
    torch.testing.assert_close(ours[:, 1:-1, 1:-1], reference[:, 1:-1, 1:-1])


@pytest.mark.parametrize(
    "field",
    [torch.zeros(1, 2, 8, 8), torch.full((1, 2, 8, 8), 0.3)],
    ids=["identity", "translation"],
)
def test_conformal_field_has_a_finite_gradient(field: torch.Tensor) -> None:
    """mu = 0 exactly here, and sqrt'(0) turned the zero upstream gradient into NaN."""
    field = field.clone().requires_grad_(True)
    BeltramiDiagnosticLoss(reduction="mean")(field, displacement=field).backward()
    assert field.grad is not None
    assert torch.isfinite(field.grad).all()


def test_equal_prediction_and_target_have_a_finite_gradient() -> None:
    pred = torch.rand(1, 2, 8, 8, requires_grad=True)
    BeltramiDiagnosticLoss(reduction="mean")(pred, pred.detach().clone()).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
