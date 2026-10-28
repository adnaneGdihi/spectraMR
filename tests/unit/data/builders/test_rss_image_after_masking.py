"""``rss_image`` with loader-side Cartesian masking: the mask comes first, then the coil combine."""

from __future__ import annotations

import pytest
import torch
import torchio as tio

from spectramr.data.builders.torchio_transform_builder import (
    TorchIOTransformBuilder,
    TorchIOTransformConfig,
)
from spectramr.data.transforms.kspace_coil_transforms import CoilCombineTransform
from spectramr.data.transforms.tio_physics import PhysicsInformedMasking
from spectramr.infrastructure.physics.fft_ops import ifft2c

COILS, H, W = 4, 32, 48


def _config(**kw) -> TorchIOTransformConfig:
    base = {
        "patch_size": (H, W, 1),
        "coil_processing_mode": "rss_image",
        "trajectory_type": "cartesian",
        "acceleration": 4,
        "center_fraction": 0.08,
    }
    base.update(kw)
    return TorchIOTransformConfig(**base)


def _index(chain: tio.Compose, cls: type) -> int | None:
    names = [type(t) for t in chain.transforms]
    return names.index(cls) if cls in names else None


@pytest.mark.parametrize("build", ["build_train_transforms", "build_val_transforms"])
def test_the_mask_runs_before_the_coil_combine(build) -> None:
    chain = getattr(TorchIOTransformBuilder, build)(_config())
    mask, combine = _index(chain, PhysicsInformedMasking), _index(chain, CoilCombineTransform)
    assert mask is not None and combine is not None
    assert mask < combine


@pytest.mark.parametrize("build", ["build_train_transforms", "build_val_transforms"])
def test_an_unmasked_rss_image_arm_keeps_the_combine_first(build) -> None:
    chain = getattr(TorchIOTransformBuilder, build)(_config(acceleration=1))
    assert _index(chain, PhysicsInformedMasking) is None
    assert _index(chain, CoilCombineTransform) is not None


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"acceleration": 1},
        {"trajectory_type": None},
        {"trajectory_type": "radial"},
        {"coil_processing_mode": "rss"},
    ],
)
def test_the_predicate_matches_the_physics_transform_it_mirrors(kw) -> None:
    config = _config(**kw)
    builds_the_mask = isinstance(
        TorchIOTransformBuilder._build_physics_transform(config), PhysicsInformedMasking
    )
    deferred = TorchIOTransformBuilder._rss_image_follows_masking(config)
    assert deferred == (builds_the_mask and config.coil_processing_mode == "rss_image")


def test_the_masked_arm_yields_a_zero_filled_image_input_and_a_full_image_target() -> None:
    torch.manual_seed(0)
    kspace = torch.randn(COILS, H, W, 1, dtype=torch.complex64)
    stacked = torch.stack([kspace.real, kspace.imag], dim=1).flatten(0, 1)  # [Re0, Im0, Re1, ...]
    subject = tio.Subject(kspace=tio.ScalarImage(tensor=stacked))
    out = TorchIOTransformBuilder.build_val_transforms(_config())(subject)

    assert out["input"].shape[0] == out["target"].shape[0] == 1
    assert out["measured_kspace"].shape[0] == 2 * COILS  # the measurement stays multi-coil
    assert not torch.allclose(out["input"].data, out["target"].data)  # the mask removed content

    full = torch.sqrt((ifft2c(kspace.permute(0, 3, 1, 2)).abs() ** 2).sum(0, keepdim=True))
    torch.testing.assert_close(
        out["target"].data[..., 0],
        full.permute(0, 2, 3, 1)[..., 0].squeeze(-1),
        atol=1e-4,
        rtol=1e-4,
    )
