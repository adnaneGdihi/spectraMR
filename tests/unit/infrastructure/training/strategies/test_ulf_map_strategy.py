"""Tests for UlfMapStrategy: Hoult-weighted physics-noise MAP (B-2.1)."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.training.strategies.ulf_map_strategy import (
    _fft2c_dc_index,
    _lowpass_otf,
    resolve_ulf_map_config,
    ulf_data_weight_normalised,
    ulf_degrade,
    ulf_map_solve,
    ulf_sigma,
)
from spectramr.infrastructure.training.strategies.ulf_lowpass_operator import UlfLowpassOperator


@pytest.mark.parametrize("n", [8, 16, 32, 256])
def test_lowpass_otf_peaks_on_fft2c_dc_and_is_real_for_even(n: int) -> None:
    # Regression for the off-center OTF (#10): the Gaussian peak MUST sit on the
    # fft2c DC bin (was index h//2-1), and for even sizes filtering a real image
    # must stay real (the .real cast was silently discarding up to ~47% imag energy).
    import torch as _t

    from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

    dc = _fft2c_dc_index(n, n, _t.device("cpu"))
    otf = _lowpass_otf(n, n, 0.3, _t.device("cpu"), _t.float32)[0, 0]
    assert divmod(int(otf.argmax()), n) == dc  # peak on the true DC
    x = _t.zeros(1, 1, n, n)
    x[0, 0, n // 2, n // 2] = 1.0
    filt = ifft2c(otf.view(1, 1, n, n) * fft2c(x))
    assert float(filt.imag.norm() / filt.real.norm().clamp_min(1e-12)) < 1e-5


@pytest.mark.parametrize("n", [7, 9])
def test_lowpass_otf_centered_on_odd_sizes(n: int) -> None:
    # Odd-axis fft2c uses a roll that h//2 misses; the empirical DC location must
    # still place the peak on DC (residual imag is tiny vs the ~0.9 pre-fix bug).
    import torch as _t

    from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

    dc = _fft2c_dc_index(n, n, _t.device("cpu"))
    otf = _lowpass_otf(n, n, 0.3, _t.device("cpu"), _t.float32)[0, 0]
    assert divmod(int(otf.argmax()), n) == dc
    x = _t.zeros(1, 1, n, n)
    x[0, 0, n // 2, n // 2] = 1.0
    filt = ifft2c(otf.view(1, 1, n, n) * fft2c(x))
    assert float(filt.imag.norm() / filt.real.norm().clamp_min(1e-12)) < 0.05


def test_ulf_sigma_increases_as_field_drops() -> None:
    # Hoult: lower field -> noisier -> larger sigma.
    s_ulf = ulf_sigma(0.1, sigma_floor=0.05)
    s_mid = ulf_sigma(1.5, sigma_floor=0.05)
    s_hf = ulf_sigma(3.0, sigma_floor=0.05)
    assert s_ulf > s_mid > s_hf
    # At the 3T reference the Hoult scale is 1.0 -> sigma == floor.
    assert abs(s_hf - 0.05) < 1e-9


def test_forward_operator_is_a_nontrivial_lowpass() -> None:
    # A is a genuine field-dependent low-pass (NOT identity): it blurs and reduces
    # high-frequency energy. (Fixes the review's A=identity facade finding.)
    from spectramr.infrastructure.physics.fft_ops import fft2c

    x = torch.rand(1, 1, 16, 16)
    ax = ulf_degrade(x, blur_cutoff=0.3)
    assert not torch.allclose(ax, x, atol=1e-3)  # not identity
    # high-frequency (off-centre) spectral energy is attenuated by A.
    hf = lambda im: fft2c(im).abs()[..., :4, :4].sum()  # noqa: E731 - corner band
    assert float(hf(ax)) < float(hf(x))


def test_high_field_data_consistency() -> None:
    # Tiny sigma (high field) -> huge data weight -> the solution is data-consistent
    # under the forward operator: A(x_hat) ~= y (the correct MAP property; the output
    # itself is A^{-1}(y), the deconvolved estimate, not y).
    y = torch.rand(1, 1, 8, 8)
    out = ulf_map_solve(
        y, lambda x: x, sigma=1e-3, rho=1.0, iterations=60,
        blur_cutoff=2.0, data_weight_max=1e6,
    )
    assert torch.allclose(ulf_degrade(out, 2.0), y, atol=2e-2)


def test_data_term_constrains_passband() -> None:
    # With a real low-pass A + high data weight + identity prior, the passband (DC)
    # of the output tracks the measurement (H=1 at DC -> exact data consistency).
    from spectramr.infrastructure.physics.fft_ops import fft2c

    y = torch.rand(1, 1, 16, 16)
    out = ulf_map_solve(
        y, lambda x: x, sigma=1e-2, rho=1.0, iterations=60,
        blur_cutoff=0.5, data_weight_max=1e6,
    )
    dc_y = fft2c(y).abs()[..., 8, 8]
    dc_o = fft2c(out).abs()[..., 8, 8]
    assert (dc_o - dc_y).abs().item() < 0.05 * float(dc_y.abs().item() + 1e-6)


def test_ulf_prior_dominates_pulls_toward_denoiser() -> None:
    # Large sigma (ULF) -> tiny data weight -> output is pulled toward the denoiser
    # fixed point (a constant blob here), not the measurement.
    y = torch.rand(1, 1, 8, 8)
    blob = torch.full((1, 1, 8, 8), 0.5)
    out = ulf_map_solve(
        y, lambda x: blob, sigma=50.0, rho=1.0, iterations=50, data_weight_max=1e3
    )
    assert (out - blob).abs().mean() < (y - blob).abs().mean()


def test_output_depends_on_measurement() -> None:
    # Anti-degeneracy (pitfall #20): the output varies with the measurement, even
    # at the ULF operating point (small data weight) — the blur A still couples y.
    den = lambda x: 0.9 * x  # noqa: E731 - simple contractive denoiser
    y1 = torch.zeros(1, 1, 8, 8)
    y2 = torch.ones(1, 1, 8, 8)
    o1 = ulf_map_solve(y1, den, sigma=ulf_sigma(0.1, 0.05), rho=1.0, iterations=10)
    o2 = ulf_map_solve(y2, den, sigma=ulf_sigma(0.1, 0.05), rho=1.0, iterations=10)
    assert not torch.allclose(o1, o2)


def test_field_knob_changes_solution() -> None:
    # The physics knob (source_field via sigma) is live: different fields give
    # different reconstructions for the same measurement + denoiser.
    den = lambda x: 0.5 * x  # noqa: E731
    y = torch.rand(1, 1, 8, 8)
    o_ulf = ulf_map_solve(y, den, sigma=ulf_sigma(0.1, 0.05), rho=1.0, iterations=15)
    o_hf = ulf_map_solve(y, den, sigma=ulf_sigma(3.0, 0.05), rho=1.0, iterations=15)
    assert not torch.allclose(o_ulf, o_hf)


def test_strategy_registered_in_factory() -> None:
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "ulf_map" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS


def test_validation_forward_runs_map() -> None:
    # _validation_forward must run the MAP solve (not a single forward). Build a
    # bare instance with a stub denoiser and the ulf_map config knobs.
    import types

    from spectramr.infrastructure.training.strategies.ulf_map_strategy import UlfMapStrategy

    strat = object.__new__(UlfMapStrategy)
    strat._denoiser = lambda x: x  # identity prox
    strat._pnp_cfg = types.SimpleNamespace(enabled=True, rho=1.0, iterations=10)
    strat._ulf_cfg = types.SimpleNamespace(
        source_field=3.0, sigma_floor=0.05, data_weight_max=1e6,
        solve_mode="map", data_weight_mode="hoult",
        relative_data_weight=None, calibration_field=None,
    )
    strat._ulf_lowpass = UlfLowpassOperator.constant_cutoff(10.0)
    y = torch.rand(1, 1, 8, 8)
    out = strat._validation_forward(y, {"use_dc": False})
    # high field + identity denoiser + huge weight -> data-consistent: A(out) ~= y
    assert torch.allclose(ulf_degrade(out, 10.0), y, atol=2e-2)


def test_config_mounted_and_frozen() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
    from spectramr.config.schemas.training.ulf_map import UlfMapConfig

    assert "ulf_map" in TrainingStrategyConfigSchema.model_fields
    c = UlfMapConfig()
    assert c.source_field > 0.0 and c.sigma_floor > 0.0
    with pytest.raises(Exception):
        c.source_field = 1.0


def test_normalised_weight_binds_at_declared_field() -> None:
    # At the declared 64 mT field the normalised weight is ratio * rho, so the data
    # term keeps a non-negligible passband share (w*H^2 / (w*H^2 + rho) at H=1).
    w = ulf_data_weight_normalised(0.064, 0.064, ratio=1.0, rho=1.0)
    assert abs(w - 1.0) < 1e-9
    assert w / (w + 1.0) > 0.4
    # The Hoult-mode weight at the same field is ~6e-4 of rho (the inert case).
    assert 1.0 / ulf_sigma(0.064, 0.05) ** 2 / 1.0 < 1e-3


def test_mis_set_source_field_changes_weight_by_field_law_only() -> None:
    # Calibration ablation: same ratio, rho and calibration field; only the believed
    # source field changes, so w scales by (B_src/B_cal)^{7/2} (a 2x field = 11.3x).
    w_true = ulf_data_weight_normalised(0.064, 0.064, ratio=1.0, rho=1.0)
    w_mis = ulf_data_weight_normalised(0.128, 0.064, ratio=1.0, rho=1.0)
    assert abs(w_mis / w_true - 2.0**3.5) < 1e-6


def test_normalised_weight_solve_is_data_sensitive() -> None:
    # The normalised weight changes the solve at the declared field (the data term
    # is live, not the 12-denoiser-application limit of the Hoult weight at 64 mT).
    den = lambda x: 0.5 * x  # noqa: E731
    y = torch.rand(1, 1, 8, 8)
    w = ulf_data_weight_normalised(0.064, 0.064, ratio=1.0, rho=1.0)
    o_bind = ulf_map_solve(y, den, sigma=ulf_sigma(0.064, 0.05), rho=1.0, iterations=12,
                           data_weight=w)
    o_inert = ulf_map_solve(y, den, sigma=ulf_sigma(0.064, 0.05), rho=1.0, iterations=12)
    assert not torch.allclose(o_bind, o_inert)


def test_resolve_ulf_map_config_raises_when_block_absent() -> None:
    import types

    with pytest.raises(ValueError, match=r"training\.ulf_map"):
        resolve_ulf_map_config(types.SimpleNamespace(ulf_map=None))


def test_ulf_map_config_weight_mode_validation() -> None:
    from pydantic import ValidationError

    from spectramr.config.schemas.training.ulf_map import UlfMapConfig

    with pytest.raises(ValidationError):
        UlfMapConfig(data_weight_mode="normalised")  # missing ratio / calibration_field
    with pytest.raises(ValidationError):
        UlfMapConfig(relative_data_weight=1.0)  # ignored under hoult -> rejected
    with pytest.raises(ValidationError):
        UlfMapConfig(solve_mode="single_pass")  # unknown value raises
    ok = UlfMapConfig(
        data_weight_mode="normalised", relative_data_weight=1.0, calibration_field=0.064
    )
    assert ok.solve_mode == "map"


def test_prior_only_validation_is_single_denoiser_pass() -> None:
    # The prior-only control returns D(y) exactly: one application, no data term.
    import types

    from spectramr.infrastructure.training.strategies.ulf_map_strategy import UlfMapStrategy

    calls: list[int] = []

    def den(x: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return 0.5 * x

    strat = object.__new__(UlfMapStrategy)
    strat._denoiser = den
    strat._pnp_cfg = types.SimpleNamespace(enabled=True, rho=1.0, iterations=12)
    strat._ulf_cfg = types.SimpleNamespace(
        source_field=0.064, sigma_floor=0.05, data_weight_max=1e3,
        solve_mode="prior_only", data_weight_mode="hoult",
        relative_data_weight=None, calibration_field=None,
    )
    y = torch.rand(1, 1, 8, 8)
    out = strat._validation_forward(y, {"use_dc": False})
    assert torch.allclose(out, 0.5 * y)
    assert len(calls) == 1


def test_pnp_disabled_raises_instead_of_passing_input_through() -> None:
    import types

    from spectramr.infrastructure.training.strategies.ulf_map_strategy import UlfMapStrategy

    strat = object.__new__(UlfMapStrategy)
    strat._denoiser = lambda x: x
    strat._pnp_cfg = types.SimpleNamespace(enabled=False, rho=1.0, iterations=2)
    strat._ulf_cfg = types.SimpleNamespace(solve_mode="map")
    with pytest.raises(ValueError, match="pnp"):
        strat.run_ulf_map(torch.rand(1, 1, 8, 8))


def test_the_solve_matches_the_inline_prox_it_replaced() -> None:
    """Characterisation: the x-update moved to ``lowpass_dc_prox`` without changing a value."""
    from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

    torch.manual_seed(0)
    y = torch.rand(1, 1, 16, 16)
    w, rho, cutoff = 4.0, 0.5, 0.3

    def den(x: torch.Tensor) -> torch.Tensor:
        return 0.9 * x

    h = _lowpass_otf(16, 16, cutoff, torch.device("cpu"), torch.float32)
    y_k, x, u = fft2c(y), y.clone(), torch.zeros_like(y)
    z = x.clone()
    for _ in range(3):
        x = ifft2c((w * h * y_k + rho * fft2c(z - u)) / (w * h * h + rho)).real
        z = den(x + u)
        u = u + (x - z)
    got = ulf_map_solve(y, den, sigma=1.0, rho=rho, iterations=3, blur_cutoff=cutoff, data_weight=w)
    assert torch.equal(got, z)


def test_validation_forward_uses_the_per_contrast_otf() -> None:
    # voxel_ratio: the contrast_id folded by validation_step selects each sample's OTF,
    # so the same y solved under two contrast ids gives two different MAP estimates.
    import types

    from spectramr.infrastructure.training.strategies.ulf_map_strategy import UlfMapStrategy

    strat = object.__new__(UlfMapStrategy)
    strat._denoiser = lambda x: x
    strat._pnp_cfg = types.SimpleNamespace(enabled=True, rho=1.0, iterations=5)
    strat._ulf_cfg = types.SimpleNamespace(
        source_field=3.0, sigma_floor=0.05, data_weight_max=1e6,
        solve_mode="map", data_weight_mode="hoult",
        relative_data_weight=None, calibration_field=None,
    )
    strat._ulf_lowpass = UlfLowpassOperator(axis_cutoffs={0: (0.2, 0.2), 1: (2.0, 2.0)})
    y = torch.rand(1, 1, 16, 16).expand(2, 1, 16, 16).contiguous()
    out = strat._validation_forward(y, {"contrast_id": torch.tensor([0, 1])})
    assert not torch.allclose(out[0], out[1])
    with pytest.raises(ValueError, match="contrast_id"):
        strat._validation_forward(y, {})


def test_validation_step_end_to_end_without_contrast_id() -> None:
    # The b21 arms validate through the new validation_step fold with a constant
    # operator and no contrast_id; the whole chain must still run.
    import types

    from spectramr.infrastructure.training.strategies.ulf_map_strategy import UlfMapStrategy
    from tests.utils.config_block_stub import block_stub

    strat = object.__new__(UlfMapStrategy)
    strat.env = types.SimpleNamespace(generator=torch.nn.Identity())
    strat._denoiser = lambda x: x
    strat._pnp_cfg = types.SimpleNamespace(enabled=True, rho=1.0, iterations=3)
    strat._ulf_cfg = types.SimpleNamespace(
        source_field=0.064, sigma_floor=0.05, data_weight_max=1e3,
        solve_mode="map", data_weight_mode="hoult",
        relative_data_weight=None, calibration_field=None,
    )
    strat._ulf_lowpass = UlfLowpassOperator.constant_cutoff(0.5)
    cfg = types.SimpleNamespace(
        training=None, validation=block_stub("validation", compute_image_metrics=False)
    )
    strat.state = types.SimpleNamespace(config=cfg)
    strat.config = cfg
    out = strat.validation_step(torch.rand(2, 1, 16, 16), torch.rand(2, 1, 16, 16))
    assert isinstance(out, dict)


def test_ulf_map_declares_its_own_parallel_compat() -> None:
    """A parent's permission is not inherited, so without its own declaration the audit
    flags every ulf_map arm as unaudited (#2629). It adds no parameters to PnP's."""
    from spectramr.infrastructure.distributed.strategy_compat import evaluate_parallel_compat
    from spectramr.infrastructure.training.strategies.ulf_map_strategy import UlfMapStrategy

    verdict = evaluate_parallel_compat(UlfMapStrategy, "ddp")
    assert not verdict.refused and not verdict.undeclared
    assert evaluate_parallel_compat(UlfMapStrategy, "fsdp").refused
