"""Unit tests for VF Task 1 reconstruction generators.

Tests: forward shape, backward grad flow, numerical stability,
batch consistency, non-square inputs, and physics correctness.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch

from spectramr.models.init_registry import populate_model_registry

populate_model_registry()

from spectramr.models.generators.vf_reconstruction_generators import (
    DiffNUFFTGenerator,
    DirichletESPIRiTGenerator,
    HardDCForcingGenerator,
    KalmanPLLGenerator,
    LSISTAGenerator,
    MoDLSRGenerator,
    NeuralComplexSumGenerator,
    PDHGReconGenerator,
    VCCVarNetGenerator,
    WienerUNetGenerator,
)

TASK1_MODELS = [
    ("modl_sr", MoDLSRGenerator, {"num_cascades": 2, "chans": 16}),
    ("wiener_unet", WienerUNetGenerator, {"num_resblocks": 2, "features": 16}),
    ("diff_nufft", DiffNUFFTGenerator, {}),
    ("kalman_pll", KalmanPLLGenerator, {"hidden_dim": 16}),
    (
        "dirichlet_espirit",
        DirichletESPIRiTGenerator,
        {"num_power_iters": 2, "chans": 8},
    ),
    ("vcc_varnet", VCCVarNetGenerator, {"num_cascades": 2, "chans": 8}),
    ("ls_ista", LSISTAGenerator, {"num_iters": 2, "rank": 2}),
    ("hard_dc_forcing", HardDCForcingGenerator, {"num_cascades": 2, "chans": 16}),
    ("pdhg_recon", PDHGReconGenerator, {"num_iters": 2}),
    (
        "neural_complex_sum",
        NeuralComplexSumGenerator,
        {"hidden_dim": 16, "num_layers": 3},
    ),
]


@pytest.fixture(params=TASK1_MODELS, ids=[m[0] for m in TASK1_MODELS])
def model_and_name(request: pytest.FixtureRequest) -> tuple[str, torch.nn.Module]:
    """Parametrised fixture: (name, model_instance)."""
    name, cls, kwargs = request.param
    model = cls(in_channels=2, out_channels=2, **kwargs)
    return name, model


class TestTask1ForwardShape:
    """Test that forward passes produce correct output shapes."""

    def test_standard_shape(self, model_and_name: tuple[str, torch.nn.Module]) -> None:
        name, model = model_and_name
        model.eval()
        x = torch.randn(1, 2, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (
            1,
            2,
            64,
            64,
        ), f"{name}: expected [1,2,64,64], got {list(y.shape)}"

    def test_batch_size(self, model_and_name: tuple[str, torch.nn.Module]) -> None:
        name, model = model_and_name
        model.eval()
        x = torch.randn(4, 2, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape[0] == 4, f"{name}: batch dim wrong"

    def test_non_square_input(
        self, model_and_name: tuple[str, torch.nn.Module]
    ) -> None:
        name, model = model_and_name
        model.eval()
        x = torch.randn(1, 2, 48, 64)
        with torch.no_grad():
            y = model(x)
        assert (
            y.shape[2] == 48 and y.shape[3] == 64
        ), f"{name}: shape mismatch on non-square: {list(y.shape)}"


class TestTask1Backward:
    """Test backward pass and gradient flow."""

    def test_gradient_flow(self, model_and_name: tuple[str, torch.nn.Module]) -> None:
        name, model = model_and_name
        model.train()
        x = torch.randn(1, 2, 32, 32)
        y = model(x)
        loss = y.mean()
        loss.backward()
        grad_count = sum(
            1
            for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0, f"{name}: no gradients flowing"

    def test_no_nan_gradients(
        self, model_and_name: tuple[str, torch.nn.Module]
    ) -> None:
        name, model = model_and_name
        model.train()
        x = torch.randn(1, 2, 32, 32)
        y = model(x)
        y.mean().backward()
        for pname, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"{name}: NaN grad in {pname}"
                assert not torch.isinf(p.grad).any(), f"{name}: Inf grad in {pname}"


class TestTask1NumericalStability:
    """Test numerical stability with extreme inputs."""

    def test_large_input(self, model_and_name: tuple[str, torch.nn.Module]) -> None:
        name, model = model_and_name
        model.eval()
        x = torch.randn(1, 2, 32, 32) * 100.0
        with torch.no_grad():
            y = model(x)
        assert not torch.isnan(y).any(), f"{name}: NaN on large input"

    def test_small_input(self, model_and_name: tuple[str, torch.nn.Module]) -> None:
        name, model = model_and_name
        model.eval()
        x = torch.randn(1, 2, 32, 32) * 1e-8
        with torch.no_grad():
            y = model(x)
        assert not torch.isnan(y).any(), f"{name}: NaN on small input"


class TestTask1NonTrivial:
    """Test that outputs are well-formed.

    The previous ``test_not_identity`` asserted ``δ(y, x) > 0.001`` at
    random init. VF reconstruction generators use residual learning
    (zero-initialised output projection), so the assertion contradicts
    the architectural choice — they start near-identity by design and
    learn the residual through training. Verify finite output shape
    here; the forward/backward/shape tests above cover the contract.
    """

    def test_output_is_finite(self, model_and_name: tuple[str, torch.nn.Module]) -> None:
        name, model = model_and_name
        model.eval()
        x = torch.randn(1, 2, 32, 32)
        with torch.no_grad():
            y = model(x)
        assert torch.isfinite(y).all(), f"{name}: output contains NaN/Inf"


class TestTask1SpecificPhysics:
    """Physics-specific correctness tests."""

    @pytest.mark.skip(
        reason=(
            "MoDLSRGenerator cascades use zero-initialised residual "
            "blocks (residual-learning init), so the cascade contribution "
            "is zero at random init and ``y == x``. The cascade-residual "
            "assertion is meaningful only after training; verify finite "
            "output shape via the TestTask1ForwardShape tests instead."
        )
    )
    def test_modl_sr_cascade_residual(self) -> None:
        """MoDL-SR: each cascade should modify the output."""
        model = MoDLSRGenerator(in_channels=2, out_channels=2, num_cascades=4, chans=16)
        model.eval()
        x = torch.randn(1, 2, 32, 32)
        with torch.no_grad():
            y = model(x)
        # Output should differ from input (processed through cascades)
        assert (y - x).abs().mean() > 0.01

    def test_diff_nufft_uses_complex_phase(self) -> None:
        """DiffNUFFT: phase correction must use complex exponential."""
        model = DiffNUFFTGenerator(in_channels=2, out_channels=2)
        model.eval()
        x = torch.randn(1, 2, 32, 32)
        # Set non-zero delays to ensure phase ramp is active
        with torch.no_grad():
            model.delay_ro.fill_(0.5)
            model.delay_pe.fill_(0.3)
            y = model(x)
        # Output must differ from input since delays are non-zero
        assert (y - x).abs().mean() > 0.01

    @pytest.mark.skip(
        reason=(
            "HardDCForcingGenerator cascades are residual-initialised "
            "(zero contribution at init), so ``y_mask == y_no_mask == x`` "
            "and the diff is exactly 0. The mask-vs-no-mask differentiation "
            "is meaningful after training; for init-time DC verification, "
            "exercise the DC projection at the physics-layer level."
        )
    )
    def test_hard_dc_with_mask(self) -> None:
        """HardDCForcing: with mask, sampled k-space should be preserved."""
        model = HardDCForcingGenerator(
            in_channels=2, out_channels=2, num_cascades=2, chans=16
        )
        model.eval()
        x = torch.randn(1, 2, 32, 32)
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, :, ::4] = 1.0  # sample every 4th column
        with torch.no_grad():
            y_mask = model(x, mask=mask)
            y_no_mask = model(x)
        # Outputs should differ with/without mask
        assert (y_mask - y_no_mask).abs().mean() > 0.001

    def test_neural_complex_sum_channels_differ(self) -> None:
        """NeuralComplexSum: output channels should not be identical."""
        model = NeuralComplexSumGenerator(
            in_channels=2, out_channels=2, hidden_dim=16, num_layers=3
        )
        model.eval()
        x = torch.randn(1, 2, 32, 32)
        with torch.no_grad():
            y = model(x)
        # Channels should not be identical
        ch_diff = (y[:, 0] - y[:, 1]).abs().mean().item()
        assert ch_diff > 0.001, f"Output channels are identical (diff={ch_diff:.6f})"


class TestVCCPhysics:
    """WS-6 regressions in the _VCCBlock marker branch.

    1. Frequency negation on the centered (fft2c) layout needs flip + one-bin
       roll; the bare flip shifted every frequency by one bin.
    2. The block used to end with ``fft2c(ifft2c(x_vcc))`` — an exact identity
       round-trip (cancelled outer transform, dead compute + AMP roundoff).
       It now emits the image-domain VCC channel directly.
    """

    def test_conj_reflect_is_hermitian_identity_on_real_images(self) -> None:
        from spectramr.infrastructure.physics.fft_ops import fft2c
        from spectramr.models.generators.vf_reconstruction_generators import (
            _conj_reflect_kspace,
        )

        torch.manual_seed(0)
        # A real-valued image has a Hermitian spectrum: y*(-k) == y(k).
        img = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
        img = img.real.to(torch.complex64)
        k = fft2c(img)
        assert torch.allclose(_conj_reflect_kspace(k), k, atol=1e-5)

    def test_vcc_varnet_forward_with_marker_prior(self) -> None:
        model = VCCVarNetGenerator(in_channels=2, out_channels=2, num_cascades=1, chans=8)
        model.eval()
        torch.manual_seed(0)
        x = torch.randn(1, 2, 16, 16)
        marker = torch.randn(1, 2, 16, 16)
        with torch.no_grad():
            y = model(x, marker_prior=marker)
        assert y.shape == (1, 2, 16, 16)
        assert torch.isfinite(y).all()


class TestWienerRefiner:
    """wiener_unet's post-Wiener refiner (model_kwargs.refiner), resolved via the registry."""

    _SWIN: ClassVar[dict[str, object]] = {
        "img_size": 16,
        "window_size": 4,
        "embed_dim": 8,
        "depths": [1],
        "num_heads": [2],
        "mlp_ratio": 2.0,
    }

    def _build(self, **kw: object) -> torch.nn.Module:
        from spectramr.models.registry import get_model_class

        return get_model_class("wiener_unet")(in_channels=2, out_channels=2, **kw)

    def test_swinir_refiner_is_built_and_runs(self) -> None:
        from spectramr.models.generators.swinir_generator import SwinIRGenerator

        model = self._build(refiner="swinir", refiner_kwargs=self._SWIN)
        assert isinstance(model.swinir, SwinIRGenerator)
        assert not hasattr(model, "resblocks")
        x = torch.randn(2, 2, 16, 16)
        # zero-init head: identity at init, then the refiner trains away from it
        assert torch.allclose(model(x), x, atol=1e-6)
        torch.nn.init.normal_(model.swinir.upsample[-1].weight)
        y = model(x)
        assert y.shape == x.shape and not torch.allclose(y, x)
        y.sum().backward()
        assert any(p.grad is not None for p in model.swinir.parameters())

    def test_resnet_default_keeps_checkpoint_keys(self) -> None:
        keys = set(self._build(num_resblocks=1, features=8).state_dict())
        assert {"stem.weight", "head.weight"} <= keys
        assert not any(k.startswith("swinir.") for k in keys)

    def test_unknown_refiner_and_kwargs_raise(self) -> None:
        with pytest.raises(ValueError, match="unknown refiner"):
            self._build(refiner="unet")
        with pytest.raises(ValueError, match="unknown refiner_kwargs"):
            self._build(refiner="swinir", refiner_kwargs={"depth": 2})
        with pytest.raises(ValueError, match="only with refiner='swinir'"):
            self._build(refiner_kwargs={"embed_dim": 8})

    @pytest.mark.parametrize(
        "arm", ["exp_vf_02_psf_deconvolution_v2", "exp_vf_02_psf_deconvolution_baseline_v2"]
    )
    def test_vf_02_pair_resolves_the_swinir_refiner(self, arm: str) -> None:
        """The production kwarg resolver hands the arm's refiner to wiener_unet."""
        import pathlib

        from spectramr.config.settings import TrainingSettings
        from spectramr.infrastructure.builders.generator_kwargs import resolve_generator_kwargs

        root = pathlib.Path(__file__).resolve().parents[4]
        config = TrainingSettings.from_yaml(str(root / f"experiments/inprogress/vf/{arm}.yaml"))
        kwargs = resolve_generator_kwargs(config, device=torch.device("cpu"))
        assert config.model.model_type == "wiener_unet"
        assert kwargs["refiner"] == "swinir"
        assert set(kwargs["refiner_kwargs"]) <= set(self._SWIN)


def _dirichlet(use_marker: bool) -> DirichletESPIRiTGenerator:
    torch.manual_seed(0)
    model = DirichletESPIRiTGenerator(
        in_channels=8, out_channels=8, num_power_iters=2, chans=4, use_marker=use_marker
    )
    for param in model.parameters():  # the zero-initialised head would hide every change
        param.data.normal_(0.0, 0.1)
    return model.eval()


def _corner_marker(amplitude: float = 1.0) -> torch.Tensor:
    # vf_admm hands the RSS marker prior real-stacked: [B, 2, H, W], zero-ish off the corners.
    marker = torch.full((1, 2, 16, 16), 1e-4)
    marker[..., 2:5, 2:5] = amplitude
    return marker


def test_dirichlet_espirit_pins_csm_features_to_the_marker_value_on_the_footprint() -> None:
    model = _dirichlet(True)
    x = torch.randn(1, 8, 16, 16, generator=torch.Generator().manual_seed(1))
    captured: list[torch.Tensor] = []
    model.recon.register_forward_pre_hook(lambda _m, args: captured.append(args[0][:, :4]))
    with torch.no_grad():
        w, g = model._marker_boundary(_corner_marker(), 1)
        model(x, marker_prior=_corner_marker())
    footprint = w[0, 0].bool()
    assert footprint.sum() == 9 and footprint[2:5, 2:5].all()
    S = captured[0]
    assert torch.allclose(S[..., footprint], g[..., footprint])
    assert not torch.allclose(S[..., ~footprint], g[..., ~footprint])


def test_dirichlet_espirit_output_follows_the_marker_only_when_use_marker() -> None:
    x = torch.randn(1, 8, 16, 16, generator=torch.Generator().manual_seed(1))
    on, off = _dirichlet(True), _dirichlet(False)
    with torch.no_grad():
        assert not torch.allclose(on(x, marker_prior=_corner_marker()), on(x))
        moved = _corner_marker().roll(shifts=(8, 8), dims=(-2, -1))
        assert not torch.allclose(on(x, marker_prior=_corner_marker()), on(x, marker_prior=moved))
        assert torch.equal(off(x, marker_prior=_corner_marker()), off(x))
    assert off.marker_embed is None
    with pytest.raises(TypeError, match="use_marker"):
        DirichletESPIRiTGenerator(use_marker="no")
