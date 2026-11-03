"""The symbolic term library: validation raises, dimensional typing, and the batched head."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.symbolic_term_library import (
    PhysicsSymbolicHead,
    build_symbolic_spec,
)

UNITS = {"TR": "time", "TE": "time", "T1": "time", "T2": "time", "M0": "magnetization"}
OPS = ["exp", "add", "mul", "div", "neg"]


def _spec(**over):
    kw = {
        "head_type": "PhysicsPySR",
        "constraints": {"dimensional_consistency": True, "units": dict(UNITS)},
        "allowed_operators": list(OPS),
        "num_terms": 5,
    }
    kw.update(over)
    return build_symbolic_spec(**kw)


def test_arm_library_is_dimensionally_typed():
    spec = _spec()
    exprs = {spec.expression(i) for i in range(len(spec.terms))}
    # 12 same-unit time ratios; M0 prefactor with 0, 1 or 2 decay factors: 1 + 12 + 66.
    assert len(spec.terms) == 79
    assert "M0*exp(-TE/T2)" in exprs
    assert "M0*exp(-TR/T1)*exp(-TE/T2)" in exprs
    assert not any("/M0" in e or "M0/" in e for e in exprs)
    assert all(e.startswith("M0") for e in exprs)


def test_dimensional_consistency_off_admits_cross_unit_terms():
    spec = _spec(constraints={"dimensional_consistency": False, "units": dict(UNITS)})
    exprs = {spec.expression(i) for i in range(len(spec.terms))}
    assert "TR*exp(-M0/T1)" in exprs


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"head_type": "GeneticPySR"}, "model_kwargs.type"),
        ({"allowed_operators": ["exp", "sin"]}, "not implemented"),
        ({"allowed_operators": []}, "allowed_operators"),
        ({"allowed_operators": ["exp", "mul", "div"]}, "needs 'add'"),
        ({"constraints": None}, "constraints"),
        ({"constraints": {"dimensional_consistency": True, "units": {"T1": "tesla"}}}, "unknown"),
        ({"constraints": {"dimensional_consistency": "yes", "units": UNITS}}, "true or false"),
        ({"constraints": {"dimensional_consistency": True, "units": UNITS, "x": 1}}, "not read"),
        ({"constraints": {"dimensional_consistency": True, "units": {"T1": "time"}}}, "magnet"),
        ({"num_terms": 0}, "positive integer"),
        ({"num_terms": 80}, "exceeds"),
    ],
)
def test_invalid_declarations_raise(over, match):
    with pytest.raises(ValueError, match=match):
        _spec(**over)


def test_head_uses_only_selected_terms_and_matches_closed_form():
    spec = _spec(num_terms=1)
    head = PhysicsSymbolicHead(4, spec)
    target = [spec.expression(i) for i in range(len(spec.terms))].index("M0*exp(-TE/T2)")
    with torch.no_grad():
        head.coefficients.zero_()
        head.coefficients[target] = 2.0
    h = torch.randn(1, 4, 3, 3)
    out = head(h)
    v = torch.nn.functional.softplus(head.to_variables(h)) + 1e-6
    te, t2, m0 = v[:, 1], v[:, 3], v[:, 4]
    torch.testing.assert_close(out[:, 0], 2.0 * m0 * torch.exp(-te / t2), rtol=1e-4, atol=1e-6)
    assert head.selected_expressions() == ["M0*exp(-TE/T2)"]


def test_unselected_coefficients_still_receive_gradient():
    head = PhysicsSymbolicHead(4, _spec(num_terms=2))
    head(torch.randn(2, 4, 3, 3)).pow(2).mean().backward()
    assert int((head.coefficients.grad != 0).sum()) > 2
