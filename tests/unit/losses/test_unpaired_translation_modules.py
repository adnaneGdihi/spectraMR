"""Unit tests for unpaired translation modules: GRL, SpectralBandSplit, B0HyperNetwork."""


import torch

# ──────────────────────────────────────────────────────────────────
# GRL + Domain Adversarial
# ──────────────────────────────────────────────────────────────────


class TestGradientReversalLayer:
    def test_forward_identity(self):
        from spectramr.models.blocks.domain_adaptation import GradientReversalLayer
        grl = GradientReversalLayer(alpha=1.0)
        x = torch.randn(2, 8)
        y = grl(x)
        assert torch.allclose(y, x), "Forward should be identity"

    def test_backward_negates(self):
        from spectramr.models.blocks.domain_adaptation import GradientReversalLayer
        grl = GradientReversalLayer(alpha=1.0)
        x = torch.randn(2, 4, requires_grad=True)
        y = grl(x)
        loss = y.sum()
        loss.backward()
        # Gradient should be -1 (negated from +1)
        assert torch.allclose(x.grad, -torch.ones_like(x)), (
            f"Gradient should be -1, got {x.grad}"
        )

    def test_alpha_scaling(self):
        from spectramr.models.blocks.domain_adaptation import GradientReversalLayer
        grl = GradientReversalLayer(alpha=0.5)
        x = torch.randn(2, 4, requires_grad=True)
        y = grl(x)
        y.sum().backward()
        assert torch.allclose(x.grad, -0.5 * torch.ones_like(x))


class TestDomainAdversarialLoss:
    def test_output_keys(self):
        from spectramr.models.losses.domain_adversarial_grl import DomainAdversarialLoss
        dal = DomainAdversarialLoss(latent_dim=16)
        z = torch.randn(4, 16)
        labels = torch.tensor([[0], [0], [1], [1]]).float()
        out = dal(z, labels)
        assert "domain_loss" in out
        assert "domain_acc" in out

    def test_loss_is_scalar(self):
        from spectramr.models.losses.domain_adversarial_grl import DomainAdversarialLoss
        dal = DomainAdversarialLoss(latent_dim=16)
        z = torch.randn(4, 16)
        labels = torch.tensor([[0], [0], [1], [1]]).float()
        out = dal(z, labels)
        assert out["domain_loss"].ndim == 0

    def test_4d_input(self):
        from spectramr.models.losses.domain_adversarial_grl import DomainAdversarialLoss
        dal = DomainAdversarialLoss(latent_dim=16)
        z = torch.randn(2, 16, 8, 8)  # Feature map
        labels = torch.tensor([[0], [1]]).float()
        out = dal(z, labels)
        assert out["domain_loss"].ndim == 0

    def test_anneal_alpha(self):
        from spectramr.models.losses.domain_adversarial_grl import DomainAdversarialLoss
        dal = DomainAdversarialLoss(latent_dim=16)
        a0 = dal.anneal_alpha(0.0)
        a1 = dal.anneal_alpha(1.0)
        assert a0 < a1, "Alpha should increase with training progress"
        assert a1 > 0.9, f"Alpha at progress=1.0 should be ~1.0, got {a1}"


# ──────────────────────────────────────────────────────────────────
# Spectral Band-Split
# ──────────────────────────────────────────────────────────────────


class TestSpectralBandSplitLoss:
    def test_output_keys(self):
        from spectramr.models.losses.spectral_band_split_loss import SpectralBandSplitLoss
        sbs = SpectralBandSplitLoss()
        pred = torch.randn(1, 1, 16, 16)
        src = torch.randn(1, 1, 16, 16)
        tgt = torch.randn(1, 1, 16, 16)
        out = sbs(pred, src, tgt)
        assert "loss_low" in out
        assert "loss_high" in out
        assert "spectral_total" in out

    def test_perfect_match_zero_loss(self):
        from spectramr.models.losses.spectral_band_split_loss import SpectralBandSplitLoss
        sbs = SpectralBandSplitLoss()
        img = torch.randn(1, 1, 16, 16)
        out = sbs(img, img, img)
        assert out["loss_low"].item() < 1e-5
        assert out["loss_high"].item() < 1e-5

    def test_unpaired_mode(self):
        from spectramr.models.losses.spectral_band_split_loss import SpectralBandSplitLoss
        sbs = SpectralBandSplitLoss()
        pred = torch.randn(1, 1, 16, 16)
        src = torch.randn(1, 1, 16, 16)
        out = sbs(pred, src, target=None)
        assert out["loss_high"].item() == 0.0

    def test_masks_complement(self):
        from spectramr.models.losses.spectral_band_split_loss import _build_frequency_mask
        m_low = _build_frequency_mask((16, 16), 0.3, torch.device("cpu"), "low")
        m_high = _build_frequency_mask((16, 16), 0.3, torch.device("cpu"), "high")
        # Low + high should cover everything
        assert torch.allclose(m_low + m_high, torch.ones_like(m_low))


# ──────────────────────────────────────────────────────────────────
# B₀ HyperNetwork
# ──────────────────────────────────────────────────────────────────


class TestB0HyperNetwork:
    def test_output_shape(self):
        from spectramr.models.generators.b0_hypernetwork import B0HyperNetwork
        hyper = B0HyperNetwork(target_param_count=1000, hidden_dim=64, num_layers=2)
        b0 = torch.tensor([0.064])
        weights = hyper(b0)
        assert weights.shape == (1, 1000)

    def test_different_b0_different_weights(self):
        from spectramr.models.generators.b0_hypernetwork import B0HyperNetwork
        hyper = B0HyperNetwork(target_param_count=500, hidden_dim=64, num_layers=2)
        w_ulf = hyper(torch.tensor([0.064]))
        w_3t = hyper(torch.tensor([3.0]))
        diff = (w_ulf - w_3t).abs().sum()
        assert diff > 0.1, "Different B₀ should produce different weights"

    def test_continuous_interpolation(self):
        from spectramr.models.generators.b0_hypernetwork import B0HyperNetwork
        hyper = B0HyperNetwork(target_param_count=500, hidden_dim=64, num_layers=2)
        b0_values = torch.linspace(0.064, 3.0, 10)
        weights = []
        for b0 in b0_values:
            w = hyper(b0.unsqueeze(0))
            weights.append(w)
        # Weights should change smoothly (not jumps)
        diffs = []
        for i in range(len(weights) - 1):
            diffs.append((weights[i + 1] - weights[i]).abs().mean().item())
        # All steps should be similar magnitude (smooth)
        assert max(diffs) < 10 * min(diffs) + 1e-6, (
            "Weight changes should be smooth across B₀"
        )

    def test_differentiable(self):
        from spectramr.models.generators.b0_hypernetwork import B0HyperNetwork
        hyper = B0HyperNetwork(target_param_count=500, hidden_dim=64, num_layers=2)
        b0 = torch.tensor([1.5], requires_grad=True)
        w = hyper(b0)
        w.sum().backward()
        assert b0.grad is not None, "Gradients should flow through B₀"

    def test_batch_b0(self):
        from spectramr.models.generators.b0_hypernetwork import B0HyperNetwork
        hyper = B0HyperNetwork(target_param_count=500, hidden_dim=64, num_layers=2)
        b0 = torch.tensor([0.064, 1.5, 3.0])
        w = hyper(b0)
        assert w.shape == (3, 500)
