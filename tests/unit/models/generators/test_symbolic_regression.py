"""``symbolic_regression`` through the production factory: knobs read, terms in the objective."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.factories.model_factory import ModelCreationError, ModelFactory

ARM_KWARGS = {
    "num_symbolic_terms": 5,
    "type": "PhysicsPySR",
    "constraints": {
        "dimensional_consistency": True,
        "units": {"TR": "time", "TE": "time", "T1": "time", "T2": "time", "M0": "magnetization"},
    },
    "allowed_operators": ["exp", "add", "mul", "div", "neg"],
}


def _build(**over):
    kw = {**ARM_KWARGS, "feature_dim": 4, "num_layers": 1, **over}
    # The model builder passes the arm's model_kwargs names as ``_declared_keys``.
    return ModelFactory().create_generator(
        "symbolic_regression",
        in_channels=1,
        out_channels=1,
        base_channels=64,
        _declared_keys=frozenset(kw),
        **kw,
    )


def test_symbolic_terms_enter_the_objective():
    model = _build()
    x = torch.randn(1, 1, 8, 8)
    y = model(x)
    assert y.shape == (1, 1, 8, 8)
    torch.nn.functional.mse_loss(y, torch.zeros_like(y)).backward()
    head = model.symbolic_head
    selected = head.selection_mask().bool()
    assert int(selected.sum()) == 5
    assert torch.all(head.coefficients.grad[selected] != 0)
    assert head.to_variables.weight.grad.abs().sum() > 0


def test_solver_record_stamps_the_declaration():
    from spectramr.infrastructure.logging.provenance import count_parameters

    record = count_parameters(_build())["solver"]
    assert record["type"] == "PhysicsPySR"
    assert record["allowed_operators"] == sorted(ARM_KWARGS["allowed_operators"])
    assert record["dimensional_consistency"] is True
    assert record["units"]["M0"] == "magnetization"
    assert record["library_size"] == 79


def test_unread_model_kwarg_raises():
    with pytest.raises(ModelCreationError, match="features"):
        _build(features=[32, 64])


def test_unknown_operator_raises_through_factory():
    with pytest.raises(Exception, match="not implemented"):
        _build(allowed_operators=["exp", "add", "mul", "div", "log"])
