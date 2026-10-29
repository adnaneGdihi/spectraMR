"""The depth-split wrapper's single-process contracts: refusals and scoping."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

pytestmark = [pytest.mark.unit]


@pytest.fixture
def two_ranks(monkeypatch):
    """A SpatialModel can be built without a process group once the size is known."""
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)


def test_a_2d_model_is_refused_before_any_collective() -> None:
    from spectramr.infrastructure.distributed.spatial import wrap_spatial

    with pytest.raises(ValueError, match="needs a 3D model"):
        wrap_spatial(nn.Conv2d(1, 1, 3), None, (1, 1, 8, 8))


def test_a_converted_layer_refuses_a_call_outside_the_scope() -> None:
    from spectramr.infrastructure.distributed.spatial.conv import SpatialConv3d

    conv = SpatialConv3d.adopt(nn.Conv3d(1, 1, 3, padding=1), group=None)
    with pytest.raises(RuntimeError, match="outside a spatial scope"):
        conv(torch.zeros(1, 1, 4, 4, 4))


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(1, 1, 1)
        self.scalars = {"kl": 0.5}
        self.cached = None

    def forward(self, x):
        return x

    def encode(self, x):
        return x


def _wrapped(inner: nn.Module):
    from spectramr.infrastructure.distributed.spatial import SpatialModel, SpatialReport

    return SpatialModel(inner, None, SpatialReport(0.0, (True,)), (1, 1, 4, 4, 4))


def test_the_wrapper_exposes_submodules_and_plain_attributes(two_ranks) -> None:
    wrapper = _wrapped(_Inner())
    assert isinstance(wrapper.conv, nn.Conv3d)
    assert wrapper.scalars == {"kl": 0.5}
    with pytest.raises(AttributeError):
        _ = wrapper.not_a_thing
    assert getattr(wrapper, "not_a_thing", None) is None


def test_a_method_is_a_loud_error_not_a_none(two_ranks) -> None:
    wrapper = _wrapped(_Inner())
    with pytest.raises(RuntimeError, match="runs only through forward"):
        _ = wrapper.encode
    with pytest.raises(RuntimeError, match="runs only through forward"):
        getattr(wrapper, "encode", None)


def test_a_cached_slab_tensor_is_refused(two_ranks) -> None:
    inner = _Inner()
    inner.cached = torch.zeros(2)
    wrapper = _wrapped(inner)
    with pytest.raises(RuntimeError, match="holds only this rank's slab"):
        _ = wrapper.cached
    inner.cached = {"mu": torch.zeros(2)}
    with pytest.raises(RuntimeError, match="holds only this rank's slab"):
        _ = wrapper.cached
