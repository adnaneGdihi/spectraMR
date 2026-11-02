"""Tests for the susceptibility-aware EPI forward operator.

The displacement law is ``Δpix = ΔB0[Hz] · t_esp · N_pe`` (Jezzard & Balaban
1995). A prior bug multiplied the (already-Hz) field by the gyromagnetic ratio
``γ ≈ 4.26e7 Hz/T``, overshooting the true shift by ~10^6 so the downstream
``grid_sample`` clamp saturated every voxel to the FOV edge — a degenerate,
measurement-independent warp. These tests pin the physical magnitude.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.epi_forward import (
    apply_epi_distortion,
    b0_to_displacement,
    beltrami_from_b0,
)


@pytest.mark.physics
class TestB0ToDisplacement:
    def test_magnitude_is_physical_not_gamma_scaled(self) -> None:
        # 50 Hz over t_esp=0.5 ms with N_pe=128 -> 50*0.5e-3*128 = 3.2 px.
        n_pe = 128
        b0 = torch.full((1, 1, n_pe, 64), 50.0)
        disp = b0_to_displacement(b0, t_esp=0.5e-3)
        assert torch.allclose(disp, torch.full_like(disp, 3.2), atol=1e-4)
        # The old gamma-scaled bug would have produced ~1e6 pixels.
        assert disp.abs().max().item() < 10.0

    def test_scales_with_phase_encode_line_count(self) -> None:
        b0 = torch.full((1, 1, 256, 64), 50.0)
        disp = b0_to_displacement(b0, t_esp=0.5e-3)  # N_pe=256 -> 6.4 px
        assert torch.allclose(disp, torch.full_like(disp, 6.4), atol=1e-4)

    def test_transverse_axis_uses_width(self) -> None:
        b0 = torch.full((1, 1, 128, 64), 50.0)
        disp = b0_to_displacement(b0, t_esp=0.5e-3, phase_encode_axis=-1)
        # N_pe = W = 64 -> 50*0.5e-3*64 = 1.6 px
        assert torch.allclose(disp, torch.full_like(disp, 1.6), atol=1e-4)


@pytest.mark.physics
class TestApplyEpiDistortion:
    def test_zero_field_is_identity(self) -> None:
        torch.manual_seed(0)
        img = torch.randn(2, 1, 32, 32)
        b0 = torch.zeros(2, 1, 32, 32)
        out = apply_epi_distortion(img, b0, t_esp=0.5e-3)
        assert torch.allclose(out, img, atol=1e-5)

    def test_modest_field_does_not_saturate(self) -> None:
        # A modest field should warp the image, not collapse it to the border.
        torch.manual_seed(1)
        img = torch.randn(1, 1, 64, 64)
        b0 = torch.full((1, 1, 64, 64), 40.0)  # ~1.3 px shift
        out = apply_epi_distortion(img, b0, t_esp=0.5e-3)
        assert not torch.allclose(out, img, atol=1e-3)  # it did move
        # Saturation would make large contiguous border regions identical;
        # assert the output still has comparable dynamic range to the input.
        assert out.std().item() > 0.5 * img.std().item()

    @pytest.mark.parametrize("slope", [0.1, 0.3, 0.5, -0.3])
    def test_constant_gradient_field_conserves_column_mass(self, slope: float) -> None:
        # #2364: the EPI signal equation conserves signal along each phase-encode
        # line. Pure resampling gave a column-mass ratio of 1/(1+d') instead of 1.
        h = w = 128
        t_esp = 0.5e-3
        yy = torch.arange(h, dtype=torch.float64).view(1, 1, h, 1).expand(1, 1, h, w)
        img = torch.exp(-0.5 * ((yy - 64) / 10) ** 2)
        db0 = slope / (t_esp * h) * (yy - 64)  # d' = slope exactly
        out = apply_epi_distortion(img, db0, t_esp=t_esp)
        ratio = out.sum(-2) / img.sum(-2)
        assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-2)

    def test_transverse_axis_conserves_row_mass(self) -> None:
        h = w = 96
        t_esp = 0.5e-3
        xx = torch.arange(w, dtype=torch.float64).view(1, 1, 1, w).expand(1, 1, h, w)
        img = torch.exp(-0.5 * ((xx - 48) / 8) ** 2)
        db0 = 0.3 / (t_esp * w) * (xx - 48)
        out = apply_epi_distortion(img, db0, t_esp=t_esp, phase_encode_axis=-1)
        ratio = out.sum(-1) / img.sum(-1)
        assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-2)


@pytest.mark.physics
class TestBeltramiFromB0:
    def test_zero_field_zero_coefficient(self) -> None:
        b0 = torch.zeros(1, 1, 32, 32)
        mu = beltrami_from_b0(b0, t_esp=0.5e-3)
        assert torch.allclose(mu, torch.zeros_like(mu), atol=1e-6)

    def test_bounded_below_one(self) -> None:
        torch.manual_seed(2)
        b0 = torch.randn(1, 1, 48, 48) * 30.0
        mu = beltrami_from_b0(b0, t_esp=0.5e-3)
        assert mu.abs().max().item() < 1.0

    def test_matches_closed_form_mu_equals_minus_s_over_2_plus_s(self) -> None:
        # Linear ramp along PE: d/dpix(ΔB0) is constant, so s = N_pe·t_esp·slope is
        # constant and mu must equal -s/(2+s) exactly (no sqrt: the old form gave
        # (1-sqrt(1+s))/(1+sqrt(1+s)) ≈ -s/4, half the correct value).
        n_pe = 64
        t_esp = 0.5e-3
        slope = 2.0  # Hz per pixel
        y = torch.arange(n_pe, dtype=torch.float32).view(1, 1, n_pe, 1)
        b0 = (slope * y).expand(1, 1, n_pe, 32).contiguous()
        mu = beltrami_from_b0(b0, t_esp=t_esp)
        s = n_pe * t_esp * slope
        expected = -s / (2.0 + s)
        interior = mu[:, :, 2:-2, :]  # avoid replicate-pad boundary
        assert torch.allclose(interior, torch.full_like(interior, expected), atol=1e-5)
        # Guard against a regression to the sqrt form (which would be ~half).
        sqrt_form = (1.0 - (1.0 + s) ** 0.5) / (1.0 + (1.0 + s) ** 0.5)
        assert abs(expected - sqrt_form) > 0.4 * abs(expected)

    def test_agrees_with_wirtinger_form_on_pure_b0(self) -> None:
        # The specialised one-axial closed form must agree with the general
        # Wirtinger construction on a pure-B0 field varying only along PE.
        from spectramr.infrastructure.physics.epi_forward import beltrami_from_b0_and_gnl

        h = w = 48
        yy, _ = torch.meshgrid(
            torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
        )
        b0 = (25.0 * yy)[None, None]
        mu_closed = beltrami_from_b0(b0, t_esp=0.5e-3)[:, 0]  # [B,H,W] real
        mu_wirt = beltrami_from_b0_and_gnl(b0, torch.zeros(1, 2, h, w), t_esp=0.5e-3)
        # Compare interior real parts (Wirtinger returns complex; imag ~ 0 here).
        ci, wi = mu_closed[:, 2:-2, 2:-2], mu_wirt[:, 2:-2, 2:-2]
        assert torch.allclose(ci, wi.real, atol=5e-4)
        assert wi.imag.abs().max().item() < 5e-4


class TestUnifiedBeltramiB0GNL:
    """A-3.1: unified Beltrami coefficient of the combined B0 (EPI) + gradient-nonlinearity warp."""

    @staticmethod
    def _scene():
        import torch

        h = w = 32
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        b0 = (30.0 * yy)[None, None]
        gnl = torch.stack([0.5 * xx, 0.5 * (xx**2 + yy**2)], 0)[None]
        return b0, gnl

    def test_identity_when_no_distortion(self) -> None:
        import torch

        from spectramr.infrastructure.physics.epi_forward import beltrami_from_b0_and_gnl

        b0, gnl = self._scene()
        mu = beltrami_from_b0_and_gnl(torch.zeros_like(b0), torch.zeros_like(gnl))
        assert float(mu.abs().max()) < 1e-6  # no distortion -> identity map -> mu=0

    def test_unifies_both_sources(self) -> None:
        import torch

        from spectramr.infrastructure.physics.epi_forward import beltrami_from_b0_and_gnl

        b0, gnl = self._scene()
        mu_b0 = beltrami_from_b0_and_gnl(b0, torch.zeros_like(gnl))
        mu_gnl = beltrami_from_b0_and_gnl(torch.zeros_like(b0), gnl)
        mu_both = beltrami_from_b0_and_gnl(b0, gnl)
        assert float(mu_b0.abs().max()) > 1e-3  # captures B0
        assert float(mu_gnl.abs().max()) > 1e-3  # captures GNL
        assert float((mu_both - mu_b0).abs().max()) > 1e-3  # GNL genuinely unifies in

    def test_quasiconformal_and_solver_compatible(self) -> None:
        from spectramr.infrastructure.physics.conformal_geometry import LinearBeltramiSolver
        from spectramr.infrastructure.physics.epi_forward import beltrami_from_b0_and_gnl

        b0, gnl = self._scene()
        mu = beltrami_from_b0_and_gnl(b0, gnl)
        assert mu.dim() == 3 and mu.is_complex()  # [B,H,W] complex
        assert float(mu.abs().max()) < 0.9 + 1e-6  # |mu| < k_max (quasiconformal)
        res = LinearBeltramiSolver(max_iter=10).forward(mu)  # the existing solver accepts it
        assert res.phi.shape == mu.shape

    def test_matches_analytic_affine_warp(self) -> None:
        # Analytic ground truth: displacement d = c*conj(z) = (c*x, -c*y) gives phi(z)=z+c*conj(z),
        # whose Beltrami coefficient is the constant mu = c. This pins the Wirtinger math against a
        # KNOWN value (not just non-zero), so a sign/convention error in d_z / d_zbar would fail.
        import torch

        from spectramr.infrastructure.physics.epi_forward import beltrami_from_b0_and_gnl

        c = 0.2
        h = w = 48
        gx, gy = torch.meshgrid(
            torch.arange(w, dtype=torch.float32), torch.arange(h, dtype=torch.float32), indexing="xy"
        )
        gnl = torch.stack([c * gx, -c * gy], 0)[None]  # d = (c*x, -c*y) = c*conj(z)
        mu = beltrami_from_b0_and_gnl(torch.zeros(1, 1, h, w), gnl, k_max=0.99)
        mu_in = mu[0, 2:-2, 2:-2]  # interior (avoid replicate-pad boundary)
        assert abs(float(mu_in.real.mean()) - c) < 1e-3
        assert float(mu_in.imag.abs().mean()) < 1e-3

    def test_rejects_invalid_shapes(self) -> None:
        import pytest
        import torch

        from spectramr.infrastructure.physics.epi_forward import beltrami_from_b0_and_gnl

        with pytest.raises(ValueError):
            beltrami_from_b0_and_gnl(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))  # gnl needs 2 ch
        with pytest.raises(ValueError):
            beltrami_from_b0_and_gnl(torch.zeros(1, 2, 8, 8), torch.zeros(1, 2, 8, 8))  # b0 needs 1 ch
