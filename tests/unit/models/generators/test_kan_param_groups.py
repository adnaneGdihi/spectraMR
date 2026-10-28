"""The KAN parameter-group ratios are validated once, for the build and the audit."""

from __future__ import annotations

import math

import pytest

from spectramr.models.generators.kan_param_groups import (
    KAN_RATIO_DEFAULTS,
    resolve_kan_param_group_ratios,
)

pytestmark = pytest.mark.unit


def test_undeclared_ratios_default_to_one() -> None:
    assert resolve_kan_param_group_ratios({}) == (1.0, 1.0)
    assert set(KAN_RATIO_DEFAULTS) == {"kan_lr_ratio", "kan_weight_decay_ratio"}


def test_declared_ratios_are_returned() -> None:
    assert resolve_kan_param_group_ratios({"kan_lr_ratio": 0.1, "kan_weight_decay_ratio": 1}) == (
        0.1,
        1.0,
    )


@pytest.mark.parametrize("key", sorted(KAN_RATIO_DEFAULTS))
@pytest.mark.parametrize("bad", ["fast", "0.1", None, True, 0, -0.1, math.inf, math.nan])
def test_a_malformed_ratio_is_refused_and_named(key: str, bad: object) -> None:
    """PLANTED: every shape a YAML typo can take, string, null, bool, zero, negative, non-finite."""
    with pytest.raises(ValueError, match=key):
        resolve_kan_param_group_ratios({key: bad})


def test_the_audit_refuses_what_the_build_refuses() -> None:
    """Wiring: the strict audit fails a KAN ablation with a malformed ratio, at load."""
    from pathlib import Path

    import yaml

    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker

    arm = (
        Path(__file__).resolve().parents[4]
        / "experiments/inprogress/kspace_filling/ablations_kan_dual_domain/experiment_11_kan_ablation_no_csg.yaml"
    )
    doc = yaml.safe_load(arm.read_text())
    good = ConfigHealthChecker().check_reverse_sampling_declaration(
        TrainingSettings.settings_from_dict(doc)
    )
    assert good.passed, good.message
    doc["model"]["model_kwargs"]["kan_lr_ratio"] = "fast"
    bad = ConfigHealthChecker().check_reverse_sampling_declaration(
        TrainingSettings.settings_from_dict(doc)
    )
    assert not bad.passed and "kan_lr_ratio" in bad.message


# --- #2105: the group holds the B-spline coefficients and nothing else ---------------------


def _toy_kan_block():
    """A KAN-tagged block in miniature: one KANLayer beside a plain Linear with a bias."""
    from torch import nn

    from spectramr.models.blocks.kan_layer import KANLayer

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = KANLayer(3, 2, grid_size=5, spline_order=3)
            self.proj = nn.Linear(4, 4)

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = _Block()
            self.head = nn.Linear(4, 1)

    return _Model(), _Block


def _assert_spline_only(selected: list, model) -> None:
    """The invariant: the selected set is exactly the ``spline_weight`` of every KANLayer."""
    from spectramr.models.blocks.kan_layer import KANLayer

    expected = {id(m.spline_weight) for m in model.modules() if isinstance(m, KANLayer)}
    got = {id(p) for p in selected}
    extra = [n for n, p in model.named_parameters() if id(p) in got - expected]
    assert got == expected, f"non-spline parameters on the KAN LR: {extra}"


def test_spline_coefficients_are_the_whole_group() -> None:
    from spectramr.models.generators.kan_param_groups import kan_spline_coefficients

    model, _ = _toy_kan_block()
    _assert_spline_only(kan_spline_coefficients(model), model)


def test_planted_whole_layer_selection_is_refused() -> None:
    """PLANTED: the pre-#2105 predicate collected every parameter of a KANLayer (base_weight too)."""
    from spectramr.models.blocks.kan_layer import KANLayer

    model, _ = _toy_kan_block()
    old = [p for m in model.modules() if isinstance(m, KANLayer) for p in m.parameters()]
    with pytest.raises(AssertionError, match="base_weight"):
        _assert_spline_only(old, model)


def test_planted_whole_block_selection_is_refused() -> None:
    """PLANTED: the pre-#2105 predicate collected a KAN-tagged block's sibling layers too."""
    model, block_cls = _toy_kan_block()
    old = [p for m in model.modules() if isinstance(m, block_cls) for p in m.parameters()]
    with pytest.raises(AssertionError, match="proj"):
        _assert_spline_only(old, model)


def test_groups_split_lr_and_record_the_fraction() -> None:
    from spectramr.models.generators.kan_param_groups import build_kan_param_groups

    model, _ = _toy_kan_block()
    groups, record = build_kan_param_groups(
        model, base_lr=1e-3, weight_decay=0.01, lr_ratio=0.1, weight_decay_ratio=0.5
    )
    main, kan = groups
    assert kan["name"] == "kan" and kan["lr"] == pytest.approx(1e-4)
    assert kan["weight_decay"] == pytest.approx(0.005)
    assert main["lr"] == 1e-3 and main["weight_decay"] == 0.01
    _assert_spline_only(kan["params"], model)
    n_spline = 2 * 3 * (5 + 3)
    n_total = sum(p.numel() for p in model.parameters())
    assert len(main["params"]) + len(kan["params"]) == len(list(model.parameters()))
    assert record["n_tensors"] == 1 and record["n_params"] == n_spline
    assert record["n_trainable_params"] == n_total
    assert record["fraction_of_trainable"] == pytest.approx(n_spline / n_total)
    assert record["kan_lr_ratio"] == 0.1 and record["kan_weight_decay_ratio"] == 0.5


def test_a_model_without_kan_layers_still_records_an_empty_group() -> None:
    """A declared ratio that reaches no parameter must read ``n_params: 0``, not vanish."""
    from torch import nn

    from spectramr.models.generators.kan_param_groups import build_kan_param_groups

    groups, record = build_kan_param_groups(
        nn.Linear(2, 2), base_lr=1e-3, weight_decay=0.0, lr_ratio=0.1, weight_decay_ratio=0.1
    )
    assert [g.get("name") for g in groups] == [None]
    assert record["n_params"] == 0 and record["fraction_of_trainable"] == 0.0
