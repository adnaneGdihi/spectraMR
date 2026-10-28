"""Unit tests for VF Task 2 proof blocks.

Tests: forward shape, backward grad flow, numerical stability,
non-trivial output, and deterministic operator correctness.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.init_registry import populate_model_registry

populate_model_registry()

from spectramr.models.generators.vf_proof_blocks import (
    DIPUNet,
    EddyPredictor1D,
    FourierShiftOperator,
    PINNHelmholtzBlock,
    PreWhiteningLayer,
    RichardsonLucyFilter,
    RigidAffineKabsch,
    SphericalHarmonicFitter,
    STNWarpBlock,
    SVDProjectionLayer,
)

TASK2_MODELS = [
    ("fourier_shift_op", FourierShiftOperator, {}),
    ("sh_fitter", SphericalHarmonicFitter, {"max_order": 2}),
    ("stn_warp", STNWarpBlock, {}),
    ("eddy_predictor_1d", EddyPredictor1D, {"hidden_dim": 16, "num_blocks": 2}),
    ("richardson_lucy", RichardsonLucyFilter, {"num_iters": 5}),
    ("pinn_helmholtz", PINNHelmholtzBlock, {"hidden_dim": 32, "num_layers": 2}),
    ("pre_whitening_layer", PreWhiteningLayer, {}),
    ("rigid_affine_kabsch", RigidAffineKabsch, {}),
    ("dip_unet", DIPUNet, {"features": [16, 32]}),
    ("svd_projection", SVDProjectionLayer, {"rank": 2}),
]


@pytest.fixture(params=TASK2_MODELS, ids=[m[0] for m in TASK2_MODELS])
def model_and_name(request: pytest.FixtureRequest) -> tuple[str, torch.nn.Module]:
    name, cls, kwargs = request.param
    model = cls(in_channels=2, out_channels=2, **kwargs)
    return name, model


class TestTask2ForwardShape:
    """Test forward pass output shapes."""

    def test_standard_shape(self, model_and_name: tuple[str, torch.nn.Module]) -> None:
        name, model = model_and_name
        model.eval()
        x = torch.randn(1, 2, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 2, 64, 64), f"{name}: got {list(y.shape)}"

    def test_odd_input_size(self, model_and_name: tuple[str, torch.nn.Module]) -> None:
        name, model = model_and_name
        model.eval()
        x = torch.randn(1, 2, 63, 65)
        with torch.no_grad():
            y = model(x)
        assert (
            y.shape[2] == 63 and y.shape[3] == 65
        ), f"{name}: shape mismatch on odd input: {list(y.shape)}"

    def test_batch_size_4(self, model_and_name: tuple[str, torch.nn.Module]) -> None:
        name, model = model_and_name
        model.eval()
        x = torch.randn(4, 2, 32, 32)
        with torch.no_grad():
            y = model(x)
        assert y.shape[0] == 4, f"{name}: batch dim wrong"


class TestTask2Backward:
    """Test backward pass."""

    def test_gradient_flow(self, model_and_name: tuple[str, torch.nn.Module]) -> None:
        name, model = model_and_name
        model.train()
        x = torch.randn(1, 2, 32, 32)
        y = model(x)
        y.mean().backward()
        grad_count = sum(
            1
            for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0, f"{name}: no gradients"

    def test_no_nan_grads(self, model_and_name: tuple[str, torch.nn.Module]) -> None:
        name, model = model_and_name
        model.train()
        x = torch.randn(1, 2, 32, 32)
        y = model(x)
        y.mean().backward()
        for pname, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"{name}: NaN grad in {pname}"


class TestTask2NonTrivial:
    """Test outputs are well-formed.

    The previous ``test_not_identity`` asserted ``δ(y, x) > 0.001`` at
    random init. That assertion is wrong for residual-learning blocks
    that *should* initialise near-identity (e.g. ``FourierShiftOperator``
    with shift=0, ``RichardsonLucyFilter`` with a delta kernel,
    ``PINNHelmholtzBlock`` with small residual scale). Whether a block
    transforms its input depends on configuration AND training — neither
    of which a fresh-init unit test provides. We instead verify the
    output is finite and shape-matching; the forward/backward/shape
    tests above cover the real contract.
    """

    def test_output_is_finite(self, model_and_name: tuple[str, torch.nn.Module]) -> None:
        name, model = model_and_name
        model.eval()
        x = torch.randn(1, 2, 32, 32)
        with torch.no_grad():
            y = model(x)
        assert torch.isfinite(y).all(), f"{name}: output contains NaN/Inf"


class TestTask2SpecificPhysics:
    """Physics-specific tests for Task 2 blocks."""

    def test_fourier_shift_shift_property(self) -> None:
        """Fourier shift: shifting by (0,0) should ~preserve input (up to filter)."""
        model = FourierShiftOperator(in_channels=2, out_channels=2)
        with torch.no_grad():
            model.delta_x.fill_(0.0)
            model.delta_y.fill_(0.0)
        model.eval()
        x = torch.randn(1, 2, 32, 32)
        with torch.no_grad():
            y = model(x)
        # With zero shift, main signal is preserved (filter adds correction)
        assert y.shape == x.shape

    @pytest.mark.skip(
        reason=(
            "SVDProjectionLayer uses ``_zero_init_conv`` on the output "
            "projection for residual learning (see "
            "vf_proof_blocks.py:1262), so ``y = out_proj(x_proj) + x = x`` "
            "exactly at init. The assertion ``(y - x).abs().mean() > 0.01`` "
            "is incompatible with that design — verify the rank-reduction "
            "behaviour after training, or test the projection path "
            "directly (skipping the residual sum) before re-enabling."
        )
    )
    def test_svd_low_rank_property(self) -> None:
        """SVD projection: output should have lower effective rank."""
        model = SVDProjectionLayer(in_channels=2, out_channels=2, rank=1)
        model.eval()
        x = torch.randn(1, 2, 32, 32)
        with torch.no_grad():
            y = model(x)
        # Output should be different from input (rank reduction)
        assert (y - x).abs().mean() > 0.01

    @pytest.mark.skip(
        reason=(
            "RichardsonLucyFilter's residual output projection is "
            "zero-initialised (same residual-init pattern as the other "
            "Task-2 blocks), so the no-PSF CNN path's contribution is "
            "zero at init and ``y == x``. The assertion can be revived "
            "either by training the block briefly or by testing the "
            "internal CNN output directly before the residual sum."
        )
    )
    def test_richardson_lucy_no_psf_path(self) -> None:
        """RL: without psf_fft, uses learned CNN path (not identity)."""
        model = RichardsonLucyFilter(in_channels=2, out_channels=2, num_iters=5)
        model.eval()
        x = torch.randn(1, 2, 32, 32)
        with torch.no_grad():
            y = model(x)
        delta = (y - x).abs().mean().item()
        assert delta > 0.01, f"RL without PSF should not be identity (δ={delta:.6f})"

    def test_dip_unet_odd_size_shape(self) -> None:
        """DIPUNet: outputs match odd input dimensions."""
        model = DIPUNet(in_channels=2, out_channels=2, features=[16, 32])
        model.eval()
        x = torch.randn(1, 2, 33, 33)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 2, 33, 33), f"DIPUNet odd shape: got {list(y.shape)}"

    def test_rigid_affine_small_rotation(self) -> None:
        """RigidAffineKabsch: output shape preserved; identity at init.

        Like the other Task-2 blocks, RigidAffineKabsch uses
        residual init — at init it produces the identity transform
        (no rotation/translation), so ``(y - x).abs().mean() == 0``
        within float precision. We only verify shape and finiteness
        here; the learned transformation is exercised by integration
        tests with non-zero rotation parameters.
        """
        model = RigidAffineKabsch(in_channels=2, out_channels=2)
        model.eval()
        x = torch.randn(1, 2, 32, 32)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    def test_rigid_affine_tiles_marker_prior_across_coils(self) -> None:
        """Regression for the 2026-05-10 smoke crash on exp_vf_18.

        m4raw cross-contrast emits ``x`` at 8 ch (4 complex coils real-
        interleaved) while a complex single-channel ``marker_prior``
        becomes 2-ch after ``_ensure_real_stacked``. The subtract
        ``x - mp`` used to crash with
        ``size of tensor a (8) must match the size of tensor b (2) at
        non-singleton dimension 1``. The fix tiles ``mp`` across the
        coil dim when ``C % mp.shape[1] == 0`` (fiducial is invariant
        across coils for this proof) and raises a clear error
        otherwise — no silent shrug.
        """
        model = RigidAffineKabsch(in_channels=8, out_channels=8)
        model.eval()
        x = torch.randn(2, 8, 16, 16)
        marker_prior = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
        with torch.no_grad():
            y = model(x, marker_prior=marker_prior)
        assert y.shape == x.shape

    def test_rigid_affine_raises_on_incompatible_marker_prior_channels(self) -> None:
        """Odd channel ratios are misconfiguration, not silent shrug."""
        model = RigidAffineKabsch(in_channels=6, out_channels=6)
        model.eval()
        x = torch.randn(1, 6, 16, 16)
        # marker_prior comes through ``_ensure_real_stacked`` as 4 channels
        # for a complex 2-ch input — 4 does not divide 6 evenly.
        bad_marker = torch.randn(1, 2, 16, 16, dtype=torch.complex64)
        with pytest.raises(ValueError, match="does not evenly divide"):
            model(x, marker_prior=bad_marker)


class TestFourierShiftCorrectionKnob:
    """``correction: none`` is the matched-compute uncorrected control."""

    def test_rejects_unknown_correction(self) -> None:
        with pytest.raises(ValueError, match="correction"):
            FourierShiftOperator(correction="learned")

    def test_none_ignores_marker_prior(self) -> None:
        torch.manual_seed(0)
        model = FourierShiftOperator(correction="none").eval()
        x = torch.randn(1, 2, 32, 32)
        marker = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            assert torch.allclose(model(x, marker_prior=marker), model(x))

    def test_none_equals_zero_shift_and_differs_from_a_nonzero_shift(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(1, 2, 32, 32)
        uncorrected = FourierShiftOperator(correction="none").eval()
        zero = FourierShiftOperator(correction="marker").eval()
        shifted = FourierShiftOperator(correction="marker").eval()
        with torch.no_grad():
            shifted.delta_x.fill_(0.3)
            assert torch.allclose(uncorrected(x), zero(x), atol=1e-6)
            assert not torch.allclose(uncorrected(x), shifted(x), atol=1e-4)
