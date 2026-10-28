"""Depth-split layers against the unsplit ones, on 2 and 3 real ranks (gloo, CPU).

Every claim the package makes is checked by the equivalence gate itself: output slab,
input gradient and summed parameter gradients equal the unsplit model's. The planted
models are the functional ops a module list cannot see; each must turn the gate red.
"""

from __future__ import annotations

import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional

pytestmark = [pytest.mark.unit]

SHAPE = (2, 4, 12, 6, 6)  # depth 12: 6 planes per rank at 2 ranks, 4 at 3


def _layers():
    return {
        "conv k3 p1": nn.Conv3d(4, 4, 3, padding=1),
        "conv k1": nn.Conv3d(4, 4, 1),
        "conv k5 p2": nn.Conv3d(4, 4, 5, padding=2),
        "conv dilated": nn.Conv3d(4, 4, 3, padding=2, dilation=2),
        "conv k3 s2": nn.Conv3d(4, 4, 3, stride=2, padding=1),
        "conv k4 s2": nn.Conv3d(4, 4, 4, stride=2, padding=1),
        "conv k2 s2": nn.Conv3d(4, 4, 2, stride=2),
        "convT k2 s2": nn.ConvTranspose3d(4, 4, 2, stride=2),
        "convT k4 s2 p1": nn.ConvTranspose3d(4, 4, 4, stride=2, padding=1),
        "upsample nearest": nn.Upsample(scale_factor=2, mode="nearest"),
        "upsample trilinear": nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        "groupnorm": nn.GroupNorm(2, 4),
        "instancenorm": nn.InstanceNorm3d(4, affine=True),
        "batchnorm": nn.BatchNorm3d(4),
        "global pool": nn.AdaptiveAvgPool3d(1),
        "composite": nn.Sequential(
            nn.Conv3d(4, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.PReLU(),
            nn.Conv3d(8, 8, 3, stride=2, padding=1),
            nn.GroupNorm(4, 8),
            nn.ConvTranspose3d(8, 4, 2, stride=2),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(4, 4, 3, stride=2, padding=1),
        ),
    }


class _Fn(nn.Module):
    """A conv followed by a functional op that mixes depth."""

    def __init__(self, fn) -> None:
        super().__init__()
        self.conv, self.fn = nn.Conv3d(4, 4, 3, padding=1), fn

    def forward(self, x):
        return self.fn(self.conv(x))


def _reflect_neighbours(y):
    """Read each plane's neighbours through a reflect pad: wrong at every internal face."""
    z = F.pad(y, (0, 0, 0, 0, 1, 1), mode="reflect")
    return z[:, :, 2:] + z[:, :, :-2]


def _planted():
    return {
        "fft along depth": _Fn(lambda y: torch.fft.fft(y, dim=2).real),
        "mean over depth": _Fn(lambda y: y - y.mean(dim=2, keepdim=True)),
        "functional trilinear": _Fn(lambda y: F.interpolate(y, scale_factor=2, mode="trilinear")),
        "reflect pad": _Fn(_reflect_neighbours),
        "flatten": nn.Sequential(nn.Conv3d(4, 1, 1), nn.Flatten(), nn.Linear(12 * 36, 3)),
    }


def _scenario_layers() -> None:
    from spectramr.infrastructure.distributed.spatial import make_spatial

    group = torch.distributed.group.WORLD
    torch.manual_seed(0)  # identical weights on every rank, as training guarantees
    for name, layer in _layers().items():
        try:
            make_spatial(layer, group, SHAPE)
        except Exception as exc:
            raise AssertionError(f"{name}: {exc}") from exc


def _scenario_planted() -> None:
    from spectramr.infrastructure.distributed.spatial import SpatialEquivalenceError, make_spatial

    group = torch.distributed.group.WORLD
    torch.manual_seed(0)
    for name, model in _planted().items():
        with pytest.raises(SpatialEquivalenceError):
            make_spatial(model, group, SHAPE)
            raise AssertionError(f"planted {name!r} passed the gate")


def _scenario_rank_dependent_weights() -> None:
    """Planted: a rank-specific init must fail before any output is compared."""
    from spectramr.infrastructure.distributed.spatial import SpatialEquivalenceError, make_spatial

    torch.manual_seed(torch.distributed.get_rank())
    with pytest.raises(SpatialEquivalenceError, match="weights differ across ranks"):
        make_spatial(nn.Conv3d(4, 4, 3, padding=1), torch.distributed.group.WORLD, SHAPE)


def _scenario_refused() -> None:
    from spectramr.infrastructure.distributed.spatial import (
        SpatialEquivalenceError,
        convert_to_spatial,
    )

    model = nn.Sequential(nn.Conv3d(4, 4, 1), nn.LayerNorm(6))
    with pytest.raises(SpatialEquivalenceError, match="LayerNorm"):
        convert_to_spatial(model, torch.distributed.group.WORLD)


def _scenario_keys_and_running_stats() -> None:
    import copy

    from spectramr.infrastructure.distributed.spatial import convert_to_spatial, split_depth

    group = torch.distributed.group.WORLD
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv3d(4, 4, 3, padding=1), nn.BatchNorm3d(4))
    reference = copy.deepcopy(model)
    keys = list(model.state_dict())
    convert_to_spatial(model, group)
    assert list(model.state_dict()) == keys, "conversion renamed state-dict keys"
    x = torch.randn(SHAPE)
    reference(x)
    model(split_depth(x, group))
    for key in ("1.running_mean", "1.running_var"):
        torch.testing.assert_close(model.state_dict()[key], reference.state_dict()[key])


def _scenario_models() -> None:
    from spectramr.infrastructure.distributed.spatial import SpatialEquivalenceError, make_spatial
    from spectramr.models.generators.dcsrn_generator import DCSRNGenerator
    from spectramr.models.generators.vnet_generator import VNetGenerator

    group = torch.distributed.group.WORLD
    # VNetDecoderBlock builds an untrained random Conv3d inside forward, so even the
    # unsplit model disagrees with itself call to call; the gate must refuse it.
    torch.manual_seed(0)
    with pytest.raises(SpatialEquivalenceError, match="does not reproduce"):
        make_spatial(VNetGenerator(1, 1, base_channels=4, num_levels=2), group, (1, 1, 16, 8, 8))
    # Depth must divide by ranks x total downsampling: 16 = 2 ranks x 2 x 2 x 2.
    torch.manual_seed(0)
    make_spatial(
        DCSRNGenerator(1, 1, num_blocks=1, growth_rate=4, num_dense_layers=2),
        group,
        (1, 1, 16, 8, 8),
    )
    # Planted: 12 planes leave 3 per rank after one stride-2 level, which cannot halve again.
    torch.manual_seed(0)
    with pytest.raises(SpatialEquivalenceError, match="not splittable"):
        make_spatial(VNetGenerator(1, 1, base_channels=4, num_levels=2), group, (1, 1, 12, 8, 8))


def _worker(rank: int, world: int, store: str, scenario: str) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    try:
        torch.set_num_threads(1)
        globals()[f"_scenario_{scenario}"]()
    finally:
        dist.destroy_process_group()


def _spawn(world: int, scenario: str) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            _worker, args=(world, os.path.join(tmp, "store"), scenario), nprocs=world, join=True
        )


@pytest.mark.parametrize("world", [2, 3])
@pytest.mark.parametrize(
    "scenario",
    ["layers", "planted", "rank_dependent_weights", "refused", "keys_and_running_stats"],
)
def test_depth_split(world: int, scenario: str) -> None:
    _spawn(world, scenario)


def test_vnet_and_dcsrn_convert() -> None:
    _spawn(2, "models")
