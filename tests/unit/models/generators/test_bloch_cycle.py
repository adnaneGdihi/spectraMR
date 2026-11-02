import pytest
import torch

from spectramr.models.factories.model_factory import ModelCreationError, ModelFactory
from spectramr.models.generators.bloch_cycle_network import BlochCycleNetwork
from spectramr.models.registry import is_strict_kwargs

TINY = {"in_channels": 1, "features": (4, 8, 16), "depth": 2}
SEQ_2000 = {"TR": 2000.0, "TE": 100.0, "flip_angle": 90, "use_physics_backward": True}


def _out(x, **kw):
    torch.manual_seed(0)
    model = BlochCycleNetwork(**{**TINY, **kw}).eval()
    with torch.no_grad():
        return model(x)


def test_bloch_cycle_forward():
    x = torch.randn(2, 1, 16, 16)
    model = BlochCycleNetwork(**TINY)
    assert model(x).shape == (2, 1, 16, 16)
    out, (maps, signal) = model.forward_with_intermediates(x)
    assert maps.shape == (2, 3, 16, 16)
    assert (maps >= 0).all()
    torch.testing.assert_close(out, signal)


def test_bloch_cycle_gradients():
    x = torch.randn(1, 1, 16, 16, requires_grad=True)
    model = BlochCycleNetwork(**TINY)
    model(x).mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_the_render_does_not_underflow_at_init():
    """Bounded ms maps: an unbounded O(1) head gave exp(-TE/T2) == 0 everywhere."""
    out = _out(torch.randn(2, 1, 16, 16))
    assert (out > 0).float().mean() > 0.99


def test_bloch_config_sets_the_rendered_sequence():
    x = torch.randn(2, 1, 16, 16)
    assert (_out(x) - _out(x, bloch_config=SEQ_2000)).abs().max() > 1e-4
    model = BlochCycleNetwork(**TINY, bloch_config=SEQ_2000)
    assert model.sequence_provenance["tr_ms"] == 2000.0


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        ({**SEQ_2000, "field_strength": 3.0}, "no reader"),
        ({**SEQ_2000, "use_physics_backward": False}, "learned decoder"),
        ({**SEQ_2000, "flip_angle": 30}, "90 degree"),
        ({"TR": 2000.0}, "omits"),
    ],
)
def test_bloch_config_raises_on_what_it_cannot_honour(cfg, match):
    with pytest.raises(ValueError, match=match):
        BlochCycleNetwork(**TINY, bloch_config=cfg)


def test_gre_reads_the_flip_angle():
    x = torch.randn(2, 1, 16, 16)
    a = _out(x, sequence_type="GRE", bloch_config={**SEQ_2000, "flip_angle": 30})
    b = _out(x, sequence_type="GRE", bloch_config={**SEQ_2000, "flip_angle": 60})
    assert (a - b).abs().max() > 1e-4


def test_num_bloch_modules_adds_refinement_stages():
    x = torch.randn(2, 1, 16, 16)
    deep = {"num_bloch_modules": 3, "feature_dim": 8, "num_residual_blocks": 2}
    assert len(BlochCycleNetwork(**TINY, **deep).refiners) == 2
    assert (_out(x) - _out(x, **deep)).abs().max() > 1e-4
    with pytest.raises(ValueError, match="num_bloch_modules > 1"):
        BlochCycleNetwork(**TINY, feature_dim=8)


def test_a_non_learnable_relaxation_is_held_at_its_fixed_value():
    model = BlochCycleNetwork(**TINY, t2_learnable=False, t2_fixed_ms=80.0)
    _, (maps, _) = model.forward_with_intermediates(torch.randn(1, 1, 16, 16))
    assert torch.all(maps[:, 2] == 80.0)
    with pytest.raises(ValueError, match="t1_fixed_ms"):
        BlochCycleNetwork(**TINY, t1_learnable=False)


def test_features_must_match_depth():
    with pytest.raises(ValueError, match="depth \\+ 1"):
        BlochCycleNetwork(in_channels=1, features=(4, 8), depth=2)


def test_both_names_are_strict_and_a_planted_kwarg_raises_through_the_factory():
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    assert is_strict_kwargs("bloch_cycle_network") and is_strict_kwargs("bloch_cycle")
    with pytest.raises(ModelCreationError, match="num_layers"):
        ModelFactory().create_generator(
            "bloch_cycle_network",
            **TINY,
            out_channels=1,
            _declared_keys=frozenset({"num_layers"}),
            num_layers=3,
        )


def test_the_factory_threads_the_declared_knobs():
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    declared = {
        "bloch_config": SEQ_2000,
        "num_bloch_modules": 2,
        "feature_dim": 8,
        "num_residual_blocks": 1,
    }
    model = ModelFactory().create_generator(
        "bloch_cycle_network",
        **TINY,
        out_channels=1,
        _declared_keys=frozenset(declared),
        **declared,
    )
    assert model.sequence_provenance["tr_ms"] == 2000.0
    assert len(model.refiners) == 1


@pytest.mark.parametrize(
    "arm",
    [
        "bloch_cycle/experiment_42_bloch_cycles.yaml",
        "bloch_cycle/experiment_42_bloch_cycles_no_cycle.yaml",
        "bloch_cycle/experiment_52_bloch_cycle_network.yaml",
        "bloch_cycle/experiment_56_bloch_cycle_translation.yaml",
    ],
)
def test_inprogress_arm_model_kwargs_are_all_consumed(arm):
    """Strict kwargs: every declared key reaches the constructor (no model is built)."""
    import logging
    from pathlib import Path

    import yaml

    from spectramr.models.factories.model_factory import _filter_kwargs_to_signature

    model = yaml.safe_load(Path("experiments/inprogress", arm).read_text())["model"]
    declared = dict(model.get("model_kwargs") or {})
    kept = _filter_kwargs_to_signature(
        declared,
        BlochCycleNetwork,
        logger_=logging.getLogger(__name__),
        site="test_inprogress_arm_model_kwargs_are_all_consumed",
        declared=frozenset(declared),
        strict=True,
    )
    assert kept == declared
