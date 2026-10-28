"""Edge-case tests for ``acquisition_flow_generator`` (background voxels, unused parameters)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.acq_flow_lie_loss import acquisition_flow_generator

_ACQ = {"TR": 2000.0, "TE": 20.0, "TI": 500.0}


def _params_with_background() -> torch.Tensor:
    rho = torch.tensor([[[[1.0, 0.0]]]])
    t1 = torch.tensor([[[[800.0, 0.0]]]])
    t2 = torch.tensor([[[[80.0, 0.0]]]])
    return torch.cat([rho, t1, t2], dim=1)


@pytest.mark.parametrize("contrast_type", ["spin_echo", "inversion_recovery"])
@pytest.mark.parametrize("wrt", ["TR", "TE"])
def test_generator_is_finite_and_zero_on_background_voxels(wrt: str, contrast_type: str) -> None:
    v = acquisition_flow_generator(
        _params_with_background(), _ACQ, wrt=wrt, contrast_type=contrast_type
    )
    assert torch.isfinite(v).all()
    assert float(v[..., 1]) == 0.0
    assert float(v[..., 0].abs()) > 0.0


def test_generator_matches_the_closed_form_te_derivative() -> None:
    params = _params_with_background()[..., :1].double()
    v = acquisition_flow_generator(params, _ACQ, wrt="TE", contrast_type="spin_echo")
    rho, t1, t2 = 1.0, 800.0, 80.0
    tr, te = _ACQ["TR"], _ACQ["TE"]
    expected = (
        -rho * (1 - torch.exp(torch.tensor(-tr / t1))) * torch.exp(torch.tensor(-te / t2)) / t2
    )
    torch.testing.assert_close(v.flatten()[0], expected.double())


def test_generator_is_zero_for_a_parameter_the_contrast_ignores() -> None:
    v = acquisition_flow_generator(
        _params_with_background(), _ACQ, wrt="TI", contrast_type="spin_echo"
    )
    assert torch.equal(v, torch.zeros_like(v))


def test_generator_runs_under_no_grad() -> None:
    with torch.no_grad():
        v = acquisition_flow_generator(
            _params_with_background(), _ACQ, wrt="TE", contrast_type="spin_echo"
        )
    assert torch.isfinite(v).all()
