"""Regression tests for the PMPS corruption operators (#2395)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from spectramr.models.generators.pmps_corruption_operators import (  # noqa: E402
    HFCorruptionOperator,
    ULFCorruptionOperator,
    _random_fourier_shift,
)


def _run(op: torch.nn.Module, x: torch.Tensor, seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    with torch.no_grad():
        return op(x)


def test_ulf_noise_floor_is_comparable_to_signal() -> None:
    ramp = torch.sin(torch.linspace(0, torch.pi, 64))
    x = (ramp.view(64, 1) * ramp.view(1, 64)).expand(4, 1, 64, 64)  # smooth phantom in [0, 1]
    y = _run(ULFCorruptionOperator(), x)
    assert y.mean().item() < 2.0
    corr = torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1]
    assert corr.item() > 0.5


def test_every_ulf_parameter_receives_a_gradient() -> None:
    torch.manual_seed(0)
    op = ULFCorruptionOperator()
    x = torch.rand(4, 1, 32, 32)
    # A non-symmetric statistic, so a zero-mean shift still moves it.
    (op(x) * torch.linspace(0, 1, 32).view(1, 1, 1, 32)).pow(2).sum().backward()
    for name, p in op.named_parameters():
        assert p.grad is not None, name
        assert p.grad.abs().sum().item() > 0.0, name


def test_hf_motion_severity_receives_a_gradient() -> None:
    torch.manual_seed(0)
    op = HFCorruptionOperator(initial_motion_severity=0.5)
    x = torch.rand(2, 1, 32, 32)
    (op(x) * torch.linspace(0, 1, 32).view(1, 1, 1, 32)).pow(2).sum().backward()
    assert op.motion_severity.grad is not None
    assert op.motion_severity.grad.abs().item() > 0.0


def test_motion_severity_is_not_piecewise_constant() -> None:
    x = torch.rand(2, 1, 32, 32)
    a, b = ULFCorruptionOperator(), ULFCorruptionOperator()
    with torch.no_grad():
        b.motion_severity.fill_(0.101)
    assert (_run(a, x) - _run(b, x)).abs().max().item() > 0.0


def test_artifact_severity_and_coil_bias_are_separately_identified() -> None:
    x = torch.rand(2, 1, 32, 32)
    a, b = ULFCorruptionOperator(), ULFCorruptionOperator()
    with torch.no_grad():
        a.recon_artifact_severity.fill_(0.3)
        a.coil_bias.fill_(0.0)
        b.recon_artifact_severity.fill_(-0.1)
        b.coil_bias.fill_(0.2)
    assert (_run(a, x) - _run(b, x)).abs().max().item() > 1e-3


def test_hoult_law_scales_the_reference_sigma_to_the_operating_field() -> None:
    hoult = ULFCorruptionOperator(b0_t=0.064, reference_noise_sigma=0.01, reference_b0_t=0.128)
    learned = ULFCorruptionOperator(
        b0_t=0.064, reference_noise_sigma=0.01, reference_b0_t=0.128, snr_scaling_law="learned"
    )
    assert hoult.noise_sigma().item() == pytest.approx(0.01 * 2.0**1.75, rel=1e-5)
    assert learned.noise_sigma().item() == pytest.approx(0.01, rel=1e-5)


@pytest.mark.parametrize("reference_b0_t", [None, 0.064])
def test_hoult_law_without_a_distinct_reference_field_raises(reference_b0_t: float | None) -> None:
    with pytest.raises(ValueError, match="reference_b0_t"):
        ULFCorruptionOperator(b0_t=0.064, reference_b0_t=reference_b0_t)


def test_stage_b_arm_kwargs_make_the_hoult_law_change_sigma() -> None:
    arm = (
        Path(__file__).parents[4]
        / "experiments/inprogress/pmps/stage_b_corruption_calibration.yaml"
    )
    kwargs = yaml.safe_load(arm.read_text())["model"]["model_kwargs"]
    assert kwargs["snr_scaling_law"] == "hoult"
    hoult = ULFCorruptionOperator(**kwargs).noise_sigma().item()
    learned = ULFCorruptionOperator(**{**kwargs, "snr_scaling_law": "learned"}).noise_sigma().item()
    assert hoult == pytest.approx(0.03, rel=1e-3)
    assert learned == pytest.approx(kwargs["reference_noise_sigma"])
    assert hoult > 100 * learned


def test_unknown_scaling_law_raises() -> None:
    with pytest.raises(ValueError, match="snr_scaling_law"):
        ULFCorruptionOperator(snr_scaling_law="bogus")


def test_channel_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="expects"):
        ULFCorruptionOperator(in_channels=2)(torch.rand(1, 1, 16, 16))


def test_fourier_shift_matches_roll_on_an_integer_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    x = torch.rand(1, 1, 16, 16)
    monkeypatch.setattr(
        torch, "randn", lambda *a, **k: torch.tensor([2.0, -3.0]).view(2, 1, 1, 1, 1)
    )
    y = _random_fourier_shift(x, torch.tensor(1.0))
    assert torch.allclose(y, torch.roll(x, shifts=(2, -3), dims=(-2, -1)), atol=1e-5)
