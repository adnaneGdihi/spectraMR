"""``neural_cache``: the cross-slice cache fires in ``sequential`` mode and only there."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from spectramr.models.generators.neural_cache import NeuralCacheGenerator

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[4]
_ARMS = _REPO / "experiments/inprogress/novel_architectures"
_TINY = {"feature_dim": 8, "num_layers": 2, "cache_size": 4, "key_value_dim": 8}


def _model(mode: str, **extra) -> NeuralCacheGenerator:
    torch.manual_seed(0)
    kwargs = {**_TINY, "cache_mode": mode, "cache_pool_size": 2, "spatial_dims": 3, **extra}
    return NeuralCacheGenerator(in_channels=1, out_channels=1, **kwargs).eval()


def _slice1_moves_with_slice0(model: NeuralCacheGenerator) -> bool:
    x = torch.randn(1, 1, 8, 8, 2)
    y = x.clone()
    y[..., 0] += 1.0
    with torch.no_grad():
        a, b = model(x), model(y)
    assert a.shape == x.shape
    return not torch.allclose(a[..., 1], b[..., 1])


def test_sequential_reads_earlier_slices_and_off_does_not() -> None:
    assert _slice1_moves_with_slice0(_model("sequential"))
    assert not _slice1_moves_with_slice0(_model("off"))


def test_cache_is_causal() -> None:
    """Slice 0 has no predecessors, so a later slice cannot change it."""
    model = _model("sequential")
    x = torch.randn(1, 1, 8, 8, 3)
    y = x.clone()
    y[..., 2] += 1.0
    with torch.no_grad():
        assert torch.allclose(model(x)[..., 0], model(y)[..., 0])


def test_cache_window_bounds_the_read() -> None:
    """With ``cache_window: 1`` slice 2 reads slice 1 only, so slice 0 cannot reach it."""
    x = torch.randn(1, 1, 8, 8, 3)
    y = x.clone()
    y[..., 0] += 1.0
    with torch.no_grad():
        windowed = _model("sequential", cache_window=1)
        assert torch.allclose(windowed(x)[..., 2], windowed(y)[..., 2])
        unbounded = _model("sequential")
        assert not torch.allclose(unbounded(x)[..., 2], unbounded(y)[..., 2])


def test_modes_share_parameters() -> None:
    on, off = _model("sequential"), _model("off")
    assert on.get_parameter_count() == off.get_parameter_count()


def test_off_mode_matches_per_slice_2d_forward() -> None:
    model = _model("off")
    x = torch.randn(2, 1, 8, 8, 3)
    with torch.no_grad():
        seq = model(x)
        per_slice = torch.stack([model(x[..., s]) for s in range(3)], dim=-1)
    assert torch.allclose(seq, per_slice, atol=1e-5)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"cache_mode": "bank"}, "cache_mode"),
        ({"cache_mode": False}, "cache_mode"),
        ({"cache_mode": "sequential", "spatial_dims": 2}, "spatial_dims"),
        ({"cache_pool_size": 0}, "cache_pool_size"),
        ({"cache_window": 0}, "cache_window"),
    ],
)
def test_invalid_knobs_raise(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        NeuralCacheGenerator(in_channels=1, out_channels=1, **{**_TINY, **kwargs})


def test_sequential_refuses_a_4d_input() -> None:
    with pytest.raises(ValueError, match="slice"):
        _model("sequential")(torch.randn(1, 1, 8, 8))


def test_mechanism_record() -> None:
    record = _model("sequential").mechanism_record
    assert record["cache_mode"] == "sequential"
    assert record["cache_tokens_per_slice"] == 4


@pytest.mark.parametrize(
    ("arm", "mode"),
    [
        ("experiment_106_neural_cache.yaml", "sequential"),
        ("experiment_106b_neural_cache_no_cache.yaml", "off"),
    ],
)
def test_arm_builds_through_the_production_path(arm: str, mode: str) -> None:
    """YAML -> TrainingSettings -> ModelBuilder (registry) -> the declared cache mode fires."""
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.training.builders.model_builder import ModelBuilder

    config = TrainingSettings.from_yaml(str(_ARMS / arm))
    assert config.data.sampling.patch_size[2] > 1, "the arm must feed slice sequences"
    model = ModelBuilder(config, torch.device("cpu")).build_generator().build()["generator"]
    assert isinstance(model, NeuralCacheGenerator)
    assert model.cache_mode.value == mode
    assert model.cache_window == config.data.sampling.patch_size[2] - 1
    assert _slice1_moves_with_slice0(model.eval()) is (mode == "sequential")
