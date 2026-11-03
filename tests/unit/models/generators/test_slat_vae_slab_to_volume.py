"""Tests for the TRELLIS-style SLAT slab→HF volume generator.

Pins the contract of the structured-latent VAE that succeeds the dense
``sparse_vae_slab_to_volume`` baseline:

1. Accepts both 4D (single-slice) and 5D (slab) input.
2. Output spatial depth equals ``d_out`` regardless of input depth.
3. ``encode_structured`` returns the (S_coarse, mu, logvar) triple
   with mu/logvar gated by S_coarse so KL is computed on active
   voxels only.
4. ``occupancy_loss`` is finite and accepts an optional ground-truth
   mask of mismatched spatial shape (it interpolates).
5. Registry entry advertises ``spatial_dims=(2, 3)`` and
   ``training_mode=vae`` — the May 2026 stage-1 LDM blocker.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.slat_vae_slab_to_volume import (
    SLATVAESlabToVolumeGenerator,
)


def _small(d_in: int = 1, d_out: int = 8):
    return SLATVAESlabToVolumeGenerator(
        in_channels=1,
        out_channels=1,
        d_in=d_in,
        d_out=d_out,
        base_channels=16,
        latent_channels=4,
        coarse_stride=4,
        num_blocks=2,
    )


class TestSLATVAESlabToVolume:
    def test_4d_slice_to_volume(self) -> None:
        m = _small(d_in=1, d_out=8)
        x = torch.randn(2, 1, 32, 32)
        y = m(x)
        assert y.shape == (2, 1, 32, 32, 8), (
            f"expected (2,1,32,32,8); got {tuple(y.shape)}"
        )
        assert torch.isfinite(y).all()

    def test_5d_slab_to_volume(self) -> None:
        m = _small(d_in=3, d_out=12)
        x = torch.randn(2, 1, 32, 32, 3)
        y = m(x)
        assert y.shape == (2, 1, 32, 32, 12)
        assert torch.isfinite(y).all()

    def test_output_depth_independent_of_input_depth(self) -> None:
        m = _small(d_in=1, d_out=16)
        y1 = m(torch.randn(1, 1, 32, 32, 1))
        y_extra = m(torch.randn(1, 1, 32, 32))  # 4D promoted
        assert y1.shape[-1] == 16
        assert y_extra.shape[-1] == 16

    def test_encode_returns_finite_mu_logvar(self) -> None:
        m = _small()
        mu, logvar = m.encode(torch.randn(1, 1, 32, 32))
        assert torch.isfinite(mu).all()
        assert torch.isfinite(logvar).all()
        assert mu.shape == logvar.shape

    def test_encode_structured_returns_active_set(self) -> None:
        m = _small()
        s, mu, logvar = m.encode_structured(torch.randn(1, 1, 32, 32))
        assert s.shape[1] == 1, "S_coarse should have a single mask channel"
        assert s.dtype == mu.dtype
        # Some voxels must be active for a non-zero foreground input.
        assert s.sum() > 0
        # mu/logvar are gated by S_coarse → strictly zero where S=0.
        zero_mask = (s == 0).expand_as(mu)
        assert torch.allclose(mu[zero_mask], torch.zeros_like(mu[zero_mask]))

    def test_kl_loss_scalar_and_finite(self) -> None:
        m = _small()
        x = torch.randn(1, 1, 32, 32)
        _ = m(x)  # populate cache
        aux = m.last_aux()
        kl = m.kl_loss(aux["mu"], aux["logvar"])
        assert kl.dim() == 0
        assert torch.isfinite(kl)

    def test_sparsity_loss_is_differentiable(self) -> None:
        m = _small()
        x = torch.randn(1, 1, 32, 32, requires_grad=False)
        _ = m(x)
        sp = m.sparsity_loss()
        assert sp.requires_grad
        sp.backward()
        # Some occupancy_head parameter should have received a gradient.
        any_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in m.decoder.occupancy_head.parameters()
        )
        assert any_grad

    def test_occupancy_loss_with_mismatched_mask(self) -> None:
        m = _small(d_in=1, d_out=8)
        _ = m(torch.randn(1, 1, 32, 32))
        # Provide a coarser mask — occupancy_loss should interpolate.
        gt = torch.ones(1, 1, 8, 8, 2)
        loss = m.occupancy_loss(gt)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_registry_metadata(self) -> None:
        from spectramr.models.registry import MODEL_REGISTRY

        entry = MODEL_REGISTRY.get("slat_vae_slab_to_volume")
        assert entry is not None
        caps = entry["capabilities"]
        assert 2 in caps.spatial_dims and 3 in caps.spatial_dims
        assert entry["mode"] == "vae"

    def test_decode_without_prior_forward_raises(self) -> None:
        m = _small()
        with pytest.raises(RuntimeError, match="active set"):
            _ = m.decode(torch.randn(1, 4, 8, 8, 1))

    def test_structural_prior_is_the_declared_weighted_sum(self) -> None:
        """#642: occupancy_lambda / sparsity_lambda were stored and never reached a loss."""
        m = _small(d_in=3, d_out=3)
        x = torch.rand(1, 1, 32, 32, 3)
        _ = m(x)
        mask = m.encoder._foreground_mask(x, m.occupancy_threshold)
        expected = m.occupancy_lambda * m.occupancy_loss(mask) + m.sparsity_lambda * m.sparsity_loss()
        torch.testing.assert_close(m.structural_prior_loss(x), expected)
        m.occupancy_lambda, m.sparsity_lambda = 0.0, 0.0
        assert m.structural_prior_loss(x) == 0

    def test_beta_kl_is_refused_not_swallowed(self) -> None:
        with pytest.raises(ValueError, match=r"training\.latent\.kl_beta_end"):
            SLATVAESlabToVolumeGenerator(beta_kl=1.0e-4)
