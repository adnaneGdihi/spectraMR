"""End-to-end registration + smoke tests for the fMRI + MRF 2026 plan
(``TODO/audit_plan_novel_fmri.md``).

Verifies (i) every advertised name is in its registry, (ii) each
component performs one forward pass, (iii) every strategy's
``_compute_losses_impl`` runs to a finite total loss.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

# Force-populate registries.
import spectramr.models.generators  # noqa: F401
import spectramr.models.diffusion  # noqa: F401
import spectramr.models.losses  # noqa: F401
import spectramr.core.metrics  # noqa: F401


# --- Shared physics smoke ---------------------------------------------------


def test_beltrami_4d_solver_runs() -> None:
    from spectramr.infrastructure.physics.manifolds.beltrami_4d import (
        SeparableBeltrami4DSolver,
    )
    s = SeparableBeltrami4DSolver(planar_max_iter=8, tol=1e-3)
    mu = 0.3 * torch.exp(1j * torch.zeros(1, 4, 8, 8))
    nu = 0.2 * torch.tanh(torch.randn(1, 4))
    res = s(mu, nu)
    assert res.phi_spatial.shape == (1, 4, 8, 8)
    assert res.phi_temporal.shape == (1, 4)
    assert res.K > 1.0


def test_spd_manifold_exp_log_roundtrip() -> None:
    from spectramr.infrastructure.physics.manifolds.spd_manifold import SPDManifold
    M = SPDManifold(n=4)
    C0 = torch.eye(4).expand(2, 4, 4).contiguous()
    V = 0.1 * torch.randn(2, 4, 4)
    V = 0.5 * (V + V.transpose(-1, -2))
    C1 = M.exp_map(C0, V)
    back = M.log_map(C0, C1)
    assert torch.allclose(back, V, atol=1e-4)


def test_srv_transform_roundtrip() -> None:
    from spectramr.infrastructure.physics.manifolds.time_reparameterisation import (
        inverse_srv_transform,
        srv_transform,
    )
    tau = torch.linspace(0, 31, 32).unsqueeze(0).expand(2, 32)
    q = srv_transform(tau)
    back = inverse_srv_transform(q, T=32)
    assert torch.allclose(back, tau, atol=1e-4)


def test_cayley_hilbert_chart_roundtrip() -> None:
    from spectramr.infrastructure.physics.manifolds.bloch_mrf_manifold import (
        cayley_hilbert_chart,
        inverse_cayley_hilbert_chart,
    )
    theta = torch.tensor([[1.0, 0.1, 1.0, 0.0, 1.0]])
    z = cayley_hilbert_chart(theta)
    back = inverse_cayley_hilbert_chart(z)
    assert torch.allclose(back, theta, atol=1e-5)


def test_epi_forward_warp_preserves_shape() -> None:
    from spectramr.infrastructure.physics.epi_forward import (
        apply_epi_distortion,
        beltrami_from_b0,
    )
    img = torch.randn(2, 1, 16, 16)
    b0 = 0.001 * torch.randn(2, 1, 16, 16)  # small, smooth
    warped = apply_epi_distortion(img, b0, t_esp=0.5e-3)
    assert warped.shape == img.shape
    mu = beltrami_from_b0(b0)
    assert mu.abs().max().item() < 1.0


# --- Registration checks ----------------------------------------------------


def test_6_fmri_mrf_models_registered() -> None:
    from spectramr.models.registry import MODEL_REGISTRY
    for n in (
        "b0_beltrami_estimator", "hrf_diffusion_prior", "tangent_spd_diffusion",
        "mrf_tangent_score", "conformal_fp_embedding", "scanner_time_reparam",
    ):
        assert n in MODEL_REGISTRY, f"model {n!r} missing"


def test_3_fmri_mrf_losses_registered() -> None:
    from spectramr.models.losses.registry import LossRegistry
    names = set(LossRegistry.list_available()) | set(LossRegistry._custom_losses)
    for n in ("beltrami_epi_residual", "hrf_likelihood", "conformality_jacobian"):
        assert n in names


def test_4_fmri_mrf_metrics_registered() -> None:
    from spectramr.core.metrics.registry import MetricsRegistry
    names = set(MetricsRegistry.list_available())
    for n in (
        "geodesic_fc_error", "geodesic_mrf_parameter_error",
        "cosine_preservation_score", "cross_scanner_t1t2_concordance",
    ):
        assert n in names


def test_10_fmri_mrf_strategies_registered() -> None:
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
    sm = TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    for n in (
        "spatiotemporal_adaptive_sfc_recon", "beltrami_epi_distortion",
        "cortical_conformal_fmri_recon", "riemannian_dfc_diffusion",
        "hrf_manifold_diffusion",
        "spatiotemporal_mrf_recon", "riemannian_mrf_diffusion",
        "conformal_mrf_dictless_recon", "crlb_mrf_pulse_design",
        "cross_scanner_mrf_harmonisation",
    ):
        assert n in sm


# --- Strategy smoke (bypass BaseTrainingStrategy.__init__) -----------------


def _alloc(strategy_cls, generator: nn.Module):
    inst = strategy_cls.__new__(strategy_cls)
    inst.config = SimpleNamespace(training=SimpleNamespace(seed=0))
    inst.device = torch.device("cpu")
    inst.env = SimpleNamespace(
        models={"generator": generator},
        device=torch.device("cpu"),
        config=inst.config,
    )
    type(inst).generator_model = property(  # type: ignore[assignment]
        lambda s: s.env.models["generator"]
    )
    return inst


class _Identity(nn.Module):
    def forward(self, x, *a, **kw):
        return x


class _ScoreStub(nn.Module):
    def forward(self, x, t=None, *a, **kw):
        return torch.zeros_like(x)


def test_spatiotemporal_adaptive_sfc_recon_smoke() -> None:
    from spectramr.infrastructure.training.strategies.fmri_kspace_strategies import (
        SpatiotemporalAdaptiveSFCReconStrategy,
    )
    s = _alloc(SpatiotemporalAdaptiveSFCReconStrategy, _Identity())
    s.lambda_mu = 0.01
    s.lambda_T = 0.01
    s.lambda_l1 = 10.0
    x = torch.randn(1, 1, 4, 8, 8)
    out = s._compute_losses_impl(x, x, epoch=0)
    assert torch.isfinite(out["g_total_loss"])


def test_beltrami_epi_distortion_smoke() -> None:
    from spectramr.infrastructure.training.strategies.fmri_kspace_strategies import (
        BeltramiEPIDistortionStrategy,
    )

    class _B0Head(nn.Module):
        def forward(self, x, *a, **kw):
            return 0.001 * torch.randn(x.shape[0], 1, x.shape[-2], x.shape[-1])

    s = _alloc(BeltramiEPIDistortionStrategy, _B0Head())
    s.t_esp = 0.5e-3
    s.lambda_mu = 0.01
    x = torch.randn(2, 1, 16, 16)
    out = s._compute_losses_impl(x, x, epoch=0)
    assert torch.isfinite(out["g_total_loss"])


def test_cortical_conformal_fmri_recon_smoke() -> None:
    from spectramr.infrastructure.training.strategies.fmri_surface_strategies import (
        CorticalConformalFMRIReconStrategy,
    )
    s = _alloc(CorticalConformalFMRIReconStrategy, _Identity())
    s.lambda_curv = 0.1
    x = torch.randn(1, 1, 16, 16)
    out = s._compute_losses_impl(x, x, epoch=0)
    assert torch.isfinite(out["g_total_loss"])


def test_riemannian_dfc_diffusion_smoke() -> None:
    from spectramr.infrastructure.training.strategies.fmri_surface_strategies import (
        RiemannianDFCDiffusionStrategy,
    )

    class _SPDScore(nn.Module):
        def forward(self, C, sigma=None, *a, **kw):
            return torch.zeros_like(C)

    s = _alloc(RiemannianDFCDiffusionStrategy, _SPDScore())
    s.sigma_min, s.sigma_max = 0.01, 0.5
    s.use_cayley = False
    n = 4
    C = torch.eye(n).expand(2, n, n).contiguous()
    out = s._compute_losses_impl(C, C, epoch=0)
    assert torch.isfinite(out["g_total_loss"])


def test_hrf_manifold_diffusion_smoke() -> None:
    from spectramr.infrastructure.training.strategies.fmri_surface_strategies import (
        HRFManifoldDiffusionStrategy,
    )
    s = _alloc(HRFManifoldDiffusionStrategy, _ScoreStub())
    s.sigma_min, s.sigma_max = 0.01, 0.5
    s.tau_eps = 0.3
    h = torch.randn(2, 32)
    out = s._compute_losses_impl(h, h, epoch=0)
    assert torch.isfinite(out["g_total_loss"])


def test_spatiotemporal_mrf_recon_smoke() -> None:
    from spectramr.infrastructure.physics.manifolds.beltrami_4d import (
        SeparableBeltrami4DSolver,
    )
    from spectramr.infrastructure.training.strategies.mrf_kspace_strategies import (
        SpatiotemporalMRFReconStrategy,
    )
    s = _alloc(SpatiotemporalMRFReconStrategy, _Identity())
    s.lambda_mu, s.lambda_nu = 0.01, 0.01
    s.solver = SeparableBeltrami4DSolver(planar_max_iter=4, tol=1e-2)
    x = torch.randn(1, 1, 4, 8, 8)
    out = s._compute_losses_impl(x, x, epoch=0)
    assert torch.isfinite(out["g_total_loss"])


def test_riemannian_mrf_diffusion_smoke() -> None:
    from spectramr.infrastructure.training.strategies.mrf_kspace_strategies import (
        RiemannianMRFDiffusionStrategy,
    )
    from spectramr.infrastructure.physics.manifolds.bloch_mrf_manifold import (
        BlochMRFManifold,
    )

    class _ScoreT(nn.Module):
        def forward(self, theta, t=None, fp=None, *a, **kw):
            return torch.zeros_like(theta)

    s = _alloc(RiemannianMRFDiffusionStrategy, _ScoreT())
    s.sigma_min, s.sigma_max = 0.01, 0.5
    s.manifold = BlochMRFManifold()
    theta = torch.tensor([[1.0, 0.1, 1.0, 0.0, 1.0]])
    # The fingerprint is now required (#347). This call used to omit it, so the
    # smoke test certified the UNCONDITIONAL mode — a finite loss with the
    # conditioning silently dropped — as the strategy working.
    out = s._compute_losses_impl(
        theta, theta, epoch=0, fingerprint=torch.randn(1, 8)
    )
    assert torch.isfinite(out["g_total_loss"])


def test_riemannian_mrf_diffusion_without_a_fingerprint_raises() -> None:
    """#347: dropping the conditioning is a different model, not a fallback."""
    import pytest

    from spectramr.infrastructure.physics.manifolds.bloch_mrf_manifold import (
        BlochMRFManifold,
    )
    from spectramr.infrastructure.training.strategies.mrf_kspace_strategies import (
        RiemannianMRFDiffusionStrategy,
    )

    class _ScoreT(nn.Module):
        def forward(self, theta, t=None, fp=None, *a, **kw):
            return torch.zeros_like(theta)

    s = _alloc(RiemannianMRFDiffusionStrategy, _ScoreT())
    s.sigma_min, s.sigma_max = 0.01, 0.5
    s.manifold = BlochMRFManifold()
    theta = torch.tensor([[1.0, 0.1, 1.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="'fingerprint' key on the batch"):
        s._compute_losses_impl(theta, theta, epoch=0)


def test_bloch_mrf_manifold_fisher_metric_is_well_formed() -> None:
    """metric_tensor must be a real Fisher [..., 5, 5] (JᵀJ), symmetric & PSD.

    Regression: it previously reused the same summed gradient 5× and returned a
    meaningless [..., 5, 5, 5] tensor.
    """
    from spectramr.infrastructure.physics.manifolds.bloch_mrf_manifold import (
        BlochMRFManifold,
    )

    m = BlochMRFManifold()
    theta = torch.rand(3, 5) + 0.5  # (T1, T2, M0, B0, B1)
    g = m.metric_tensor(theta)
    assert g.shape == (3, 5, 5)
    assert torch.allclose(g, g.transpose(-1, -2), atol=1e-5)  # symmetric
    evals = torch.linalg.eigvalsh(g)
    assert (evals > 0).all()  # PSD (EPS ridge keeps it strictly positive)


def test_conformal_mrf_dictless_recon_smoke() -> None:
    from spectramr.infrastructure.training.strategies.mrf_acquisition_strategies import (
        ConformalMRFDictlessReconStrategy,
    )
    from spectramr.models.generators.mrf_models import ConformalFPEmbedding
    model = ConformalFPEmbedding(fingerprint_length=32, embedding_dim=5, hidden=16, n_blocks=2)
    s = _alloc(ConformalMRFDictlessReconStrategy, model)
    s.lambda_c = 0.0  # disable jacobian computation to keep test fast
    f = torch.randn(4, 32)
    out = s._compute_losses_impl(f, f, epoch=0)
    assert torch.isfinite(out["g_total_loss"])


def test_crlb_mrf_pulse_design_smoke() -> None:
    from spectramr.infrastructure.physics.acquisition.mrf_acquisition_manifold import (
        AcquisitionBound,
    )
    from spectramr.infrastructure.training.strategies.mrf_acquisition_strategies import (
        CRLBMRFPulseDesignStrategy,
    )
    from spectramr.infrastructure.training.strategies.rng_streams import (
        rank_seeded_generator,
    )
    from spectramr.models.generators.mrf_pulse_design import MRFPulseDesign

    s = _alloc(
        CRLBMRFPulseDesignStrategy,
        MRFPulseDesign(n_tr=8, alpha_init_rad=1.0, tr_init_ms=10.0),
    )
    s.lambda_beltrami = 1e-3
    s.t1_range_ms, s.t2_range_ms, s.prior_samples, s.te_ms = (100.0, 3000.0), (20.0, 500.0), 8, 2.0
    s.bound = AcquisitionBound(4.0, 200.0)
    s._prior_rng = rank_seeded_generator(0)
    out = s._compute_losses_impl(torch.randn(2, 8), torch.randn(2, 8), epoch=0)
    assert torch.isfinite(out["g_total_loss"])


def test_cross_scanner_mrf_harmonisation_smoke() -> None:
    from spectramr.infrastructure.training.strategies.mrf_acquisition_strategies import (
        CrossScannerMRFHarmonisationStrategy,
    )
    from spectramr.models.generators.mrf_models import ScannerTimeReparam
    model = ScannerTimeReparam(fingerprint_length=32, eps=0.1)
    s = _alloc(CrossScannerMRFHarmonisationStrategy, model)
    s.tau_eps = 0.1
    s.lambda_anchor = 1e-3
    s.requires_scanner_id = False
    observed = torch.randn(2, 32)
    canonical = torch.randn(2, 32)
    out = s._compute_losses_impl(observed, canonical, epoch=0)
    assert torch.isfinite(out["g_total_loss"])
