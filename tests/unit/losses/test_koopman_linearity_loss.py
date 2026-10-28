"""Unit tests for koopman_linearity_loss.py.

The Koopman-linearity loss fits a best-fit linear operator K to the
prediction's own temporal snapshots in a learned observable space Phi and
penalizes the linear-evolution residual ||Phi(x_{t+1}) - K Phi(x_t)||.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.koopman_linearity_loss import KoopmanLinearityLoss
from spectramr.models.losses.registry import LossRegistry, create_loss
from spectramr.models.temporal.koopman_operator import KoopmanMotionFilter

B, T, H, W = 2, 6, 8, 8
#: Small enough that T - 1 > LATENT, so the operator fit does not interpolate.
LATENT = 3


def _seq(b=B, t=T, h=H, w=W, seed=0) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.randn(b, t, h, w, generator=gen)


class TestKoopmanLinearityLoss:

    # -- Canary ---------------------------------------------------------------

    def test_canary_builds_and_forward_scalar(self):
        loss_fn = KoopmanLinearityLoss(latent_dim=LATENT)
        x = _seq()
        out = loss_fn(x, x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == ()
        assert torch.isfinite(out)

    # -- Registry -------------------------------------------------------------

    def test_registry_name_aliases_domain(self):
        loss_fn = create_loss("koopman_linearity")
        assert isinstance(loss_fn, KoopmanLinearityLoss)
        # domain metadata recorded as "image"
        assert LossRegistry._loss_domains["koopman_linearity"]["domain"] == "image"

    def test_registry_alias_resolves(self):
        loss_fn = create_loss("koopman_dmd_residual")
        assert isinstance(loss_fn, KoopmanLinearityLoss)

    # -- Primitive exercised --------------------------------------------------

    def test_primitive_is_used(self):
        loss_fn = KoopmanLinearityLoss(latent_dim=LATENT)
        assert isinstance(loss_fn.koopman, KoopmanMotionFilter)
        # The observable space is the primitive's (frozen) encoder.
        x = _seq()
        flat = x.reshape(B, T, H * W)
        z = loss_fn.koopman.encode(flat)
        assert z.shape == (B, T, loss_fn.koopman.latent_dim)

    # -- Property: linear-in-observable-space sequence -> ~0 residual ----------

    def test_property_constant_sequence_zero_residual(self):
        """A constant sequence is trivially linear (identity dynamics):
        Phi(x_{t+1}) == Phi(x_t), so the best-fit K is identity and the
        residual must be ~0."""
        loss_fn = KoopmanLinearityLoss(latent_dim=LATENT)
        x = torch.ones(B, T, H, W)
        out = loss_fn(x, x)
        assert out.item() == pytest.approx(0.0, abs=1e-6)

    def test_property_linear_observable_sequence_small_residual(self):
        """Build a sequence that evolves *linearly in the observable space*
        by propagating an initial embedding through the primitive's own K and
        decoding. The fitted Koopman residual on the re-encoded snapshots is
        far smaller than for a random sequence."""
        loss_fn = KoopmanLinearityLoss(latent_dim=LATENT)
        koop = loss_fn.koopman
        with torch.no_grad():
            z0 = koop.encode(torch.randn(B, 1, H * W))  # [B, 1, latent]
            zs = [z0[:, 0]]
            for _ in range(T - 1):
                zs.append(zs[-1] @ koop.K.t())
            z_seq = torch.stack(zs, dim=1)  # [B, T, latent]
            x_lin = koop.decode(z_seq).reshape(B, T, H, W)
        lin_residual = loss_fn(x_lin, x_lin)

        rand = _seq(seed=7)
        rand_residual = loss_fn(rand, rand)
        assert lin_residual.item() < rand_residual.item()

    def test_property_random_sequence_positive_residual(self):
        loss_fn = KoopmanLinearityLoss(latent_dim=LATENT)
        x = _seq(seed=3)
        out = loss_fn(x, x)
        assert out.item() > 0.0

    # -- Reduction ------------------------------------------------------------

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_parametric_reduction(self, reduction: str):
        loss_fn = KoopmanLinearityLoss(latent_dim=LATENT, reduction=reduction)
        x = _seq()
        out = loss_fn(x, x)
        assert torch.isfinite(out)
        assert out.item() >= 0.0

    def test_property_sum_ge_mean(self):
        x = _seq(seed=5)
        s = KoopmanLinearityLoss(latent_dim=LATENT, reduction="sum")(x, x)
        m = KoopmanLinearityLoss(latent_dim=LATENT, reduction="mean")(x, x)
        assert s.item() >= m.item()

    # -- Gradient -------------------------------------------------------------

    def test_property_gradient_reaches_prediction(self):
        loss_fn = KoopmanLinearityLoss(latent_dim=LATENT)
        x = _seq().requires_grad_(True)
        loss_fn(x, x).backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    # -- Channel-as-time axis -------------------------------------------------

    def test_channel_as_time_axis(self):
        """[B, C, H, W] with C interpreted as the time axis is accepted."""
        loss_fn = KoopmanLinearityLoss(latent_dim=LATENT)
        x = _seq()  # T==C here
        out = loss_fn(x, x)
        assert torch.isfinite(out)

    # -- Raises ---------------------------------------------------------------

    def test_raises_invalid_reduction(self):
        with pytest.raises(ValueError, match="reduction"):
            KoopmanLinearityLoss(reduction="median")

    def test_raises_wrong_ndim(self):
        loss_fn = KoopmanLinearityLoss(latent_dim=LATENT)
        bad = torch.randn(B, H, W)  # 3-D, no time axis
        with pytest.raises(ValueError):
            loss_fn(bad, bad)

    def test_raises_too_short_time_axis(self):
        loss_fn = KoopmanLinearityLoss(latent_dim=LATENT)
        short = torch.randn(B, 1, H, W)  # only one snapshot -> no transition
        with pytest.raises(ValueError):
            loss_fn(short, short)

    def test_raises_shape_mismatch_pred_target(self):
        loss_fn = KoopmanLinearityLoss(latent_dim=LATENT)
        x = _seq()
        y = torch.randn(B, T, H, W + 1)
        with pytest.raises(ValueError):
            loss_fn(x, y)

    # -- Interpolation regime and frozen observable (#2279) -------------------

    @pytest.mark.parametrize("t", [2, 6, 33])
    def test_raises_when_the_fit_interpolates(self, t: int):
        """With T - 1 <= latent_dim the fitted K reproduces any series exactly.

        The only arm feeds T = 2 with latent_dim = 32, where the residual of a
        random series was 1e-9.
        """
        loss_fn = KoopmanLinearityLoss()
        x = _seq(t=t)
        with pytest.raises(ValueError, match="latent_dim \\+ 2"):
            loss_fn(x, x)

    def test_accepts_the_first_non_interpolating_length(self):
        loss_fn = KoopmanLinearityLoss(latent_dim=LATENT)
        x = _seq(t=LATENT + 2, seed=4)
        assert loss_fn(x, x).item() > 1e-4

    def test_encoder_is_frozen(self):
        loss_fn = KoopmanLinearityLoss(latent_dim=LATENT)
        x = _seq().requires_grad_(True)
        loss_fn(x, x).backward()
        assert loss_fn.koopman is not None
        assert all(not p.requires_grad for p in loss_fn.koopman.parameters())
        assert all(p.grad is None for p in loss_fn.koopman.parameters())

    def test_rebuild_is_deterministic_and_leaves_global_rng_alone(self):
        """A rebuild for a new H*W draws the same weights and no global randomness."""
        a = KoopmanLinearityLoss(latent_dim=LATENT)
        b = KoopmanLinearityLoss(latent_dim=LATENT)
        torch.manual_seed(123)
        before = torch.random.get_rng_state()
        a(_seq(), _seq())
        b(_seq(h=H, w=W), _seq(h=H, w=W))
        assert torch.equal(torch.random.get_rng_state(), before)
        assert a.koopman is not None and b.koopman is not None
        for pa, pb in zip(a.koopman.parameters(), b.koopman.parameters(), strict=True):
            assert torch.equal(pa, pb)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA generator")
def test_building_the_encoder_leaves_the_cuda_stream_alone():
    """``torch.manual_seed`` inside ``fork_rng(devices=[])`` reseeded CUDA to 0."""
    torch.manual_seed(123)
    expected = torch.randn(4, device="cuda")
    torch.manual_seed(123)
    KoopmanLinearityLoss()
    assert torch.equal(torch.randn(4, device="cuda"), expected)


def test_building_the_encoder_leaves_the_cpu_stream_alone():
    torch.manual_seed(123)
    expected = torch.randn(4)
    torch.manual_seed(123)
    KoopmanLinearityLoss()
    assert torch.equal(torch.randn(4), expected)
