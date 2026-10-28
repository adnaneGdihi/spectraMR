"""Inference must normalize k-space exactly the way training did (issue #572).

A model trained on k-space that was percentile-divided and ``log1p``-compressed
sees a *different distribution* at inference if the inference path divides by a
different percentile, measures the scale in a different domain, or skips the
log compression entirely. The prediction is then decoded with the wrong inverse.

The training-side normalizer is ``KSpaceNormalizationTransform``, driven by
``data.kspace_percentile`` / ``data.log_scaling`` / ``data.kspace_scale_domain``.
The inference strategies used to read an unrelated block
(``data.normalization_kwargs``, a k-space magnitude percentile defaulting to
0.99) and never applied ``log_scaling`` at all.

See ``docs/kspace_normalization_ssot.rst``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectramr.data.transforms.normalization import (
    denormalize_kspace_robust,
    kspace_image_domain_scale,
    normalize_kspace_robust,
)
from spectramr.infrastructure.physics.fft_ops import fft2c

# The exp_11 kspace_filling attention-shootout settings.
ARM_DATA_CFG = {
    "normalize_kspace": True,
    "kspace_percentile": 0.95,
    "log_scaling": True,
    "kspace_scale_domain": "image",
    # Legacy IMAGE-normalization block. Inert for training; the inference
    # strategies used to key off it. Left set to a DIFFERENT percentile on
    # purpose so a regression to the old resolver is visible.
    "normalization_kwargs": {"percentile": 0.99, "clamp": False},
}


def _multicoil_kspace(batch: int = 1, coils: int = 4, hw: int = 32) -> torch.Tensor:
    """(B, 2C, H, W) real-interleaved k-space with a realistic DC spike."""
    torch.manual_seed(0)
    img = torch.zeros(batch, coils, hw, hw, dtype=torch.complex64)
    img[:, :, hw // 4 : 3 * hw // 4, hw // 4 : 3 * hw // 4] = 1.0
    img = img + 0.02 * torch.randn(batch, coils, hw, hw, dtype=torch.complex64)
    k = fft2c(img.reshape(-1, hw, hw)).reshape(batch, coils, hw, hw)
    out = torch.empty(batch, coils * 2, hw, hw)
    out[:, 0::2] = k.real
    out[:, 1::2] = k.imag
    return out


def _training_normalization(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exactly what KSpaceNormalizationTransform applies for ARM_DATA_CFG."""
    scale = kspace_image_domain_scale(x[0], percentile=0.95, channel_dim=0)
    normed, _ = normalize_kspace_robust(x, scale=scale, log_scaling=True, channel_dim=1)
    return normed, scale


class _IdentityModel(nn.Module):
    def forward(self, x, *a, **k):  # pragma: no cover - never invoked here
        return x


def _cold_strategy(cfg: dict):
    from spectramr.infrastructure.inference.cold_diffusion_inference_strategy import (
        ColdDiffusionInferenceStrategy,
    )

    return ColdDiffusionInferenceStrategy(
        model=_IdentityModel(), device=torch.device("cpu"), config={"data": cfg}
    )


def test_cold_diffusion_preprocess_matches_training_normalization():
    """The tensor handed to the network must be the tensor training produced.

    Fails on the old resolver three ways at once: percentile 0.99 instead of
    0.95, the scale measured on |k| instead of the image domain, and no log1p.
    """
    x = _multicoil_kspace()
    expected, _ = _training_normalization(x)

    got = _cold_strategy(ARM_DATA_CFG).preprocess_input(x.clone())

    assert torch.allclose(
        got, expected, rtol=1e-4, atol=1e-6
    ), "inference normalization diverges from training normalization"


def test_cold_diffusion_round_trip_recovers_physical_kspace():
    """Denormalizing the network's output must undo BOTH the log and the scale."""
    strategy = _cold_strategy(ARM_DATA_CFG)
    x = _multicoil_kspace()

    normed = strategy.preprocess_input(x.clone())
    restored = strategy.denormalize_kspace(normed)

    assert torch.allclose(
        restored, x, rtol=1e-3, atol=1e-4 * x.abs().max()
    ), "output inversion does not return to the physical k-space scale"


def test_scale_uses_all_coils_not_just_the_first():
    """The magnitude must come from the SSOT, not ``x[:, 0]**2 + x[:, 1]**2``.

    The old cold-diffusion path read channels 0 and 1 only — the FIRST coil's
    real/imag — so an 8-channel 4-coil arm derived its entire scale from one
    coil. Scaling the other coils must therefore move the scale.
    """
    strategy = _cold_strategy(ARM_DATA_CFG)
    x = _multicoil_kspace(coils=4)

    boosted = x.clone()
    boosted[:, 2:] *= 8.0  # leave coil 0 (channels 0,1) untouched

    s_ref = strategy.compute_kspace_scale(x)
    s_boost = strategy.compute_kspace_scale(boosted)

    assert not torch.allclose(
        s_ref, s_boost, rtol=1e-3
    ), "scale ignored every coil but the first"


def test_no_normalization_when_flag_is_off():
    """``normalize_kspace: false`` must leave the tensor untouched."""
    cfg = dict(ARM_DATA_CFG, normalize_kspace=False)
    x = _multicoil_kspace()
    assert torch.allclose(_cold_strategy(cfg).preprocess_input(x.clone()), x)


@pytest.mark.parametrize("domain", ["kspace", "image"])
def test_diffusion_strategy_round_trip(domain):
    """The same parity contract holds for DiffusionInferenceStrategy."""
    from spectramr.infrastructure.inference.diffusion_inference_strategy import (
        DiffusionInferenceStrategy,
    )

    cfg = dict(ARM_DATA_CFG, kspace_scale_domain=domain)
    strategy = DiffusionInferenceStrategy(
        model=_IdentityModel(), device=torch.device("cpu"), config={"data": cfg}
    )
    x = _multicoil_kspace()

    normed, scale = strategy.kspace_norm.normalize(x, channel_dim=1)
    restored = strategy.kspace_norm.denormalize(normed, scale, channel_dim=1)

    assert torch.allclose(restored, x, rtol=1e-3, atol=1e-4 * x.abs().max())


def test_spec_reads_training_knobs_not_the_legacy_block():
    """The resolver must key off kspace_percentile, never normalization_kwargs."""
    from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

    spec = KSpaceNormalizationSpec.from_data_config(ARM_DATA_CFG)
    assert spec.enabled is True
    assert spec.percentile == 0.95  # NOT normalization_kwargs' 0.99
    assert spec.log_scaling is True
    assert spec.scale_domain == "image"


def test_spec_rejects_unknown_scale_domain():
    """Closed set — a typo must raise, not silently pick a domain."""
    from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

    with pytest.raises(ValueError, match="scale_domain"):
        KSpaceNormalizationSpec.from_data_config(
            dict(ARM_DATA_CFG, kspace_scale_domain="parseval")
        )


def test_the_volume_scale_does_not_depend_on_the_batch_size():
    """Training measures one scale per volume; predict measured one per batch.

    Normalising a 4-slice volume in batches of 1 and of 2 with the volume's scale gives
    the slices the full-volume normalisation. Planted: the per-batch scale does not.
    """
    strategy = _cold_strategy(ARM_DATA_CFG)
    volume = _multicoil_kspace(batch=4)
    volume[2:] *= 5.0  # slices of one volume at different intensities
    scale = strategy.volume_kspace_scale(volume, per_slice=False)
    assert scale.shape == (4,) and torch.all(scale == scale[0])

    full = strategy.preprocess_input(volume.clone(), kspace_scale=scale)
    for size in (1, 2):
        parts = [
            strategy.preprocess_input(volume[i : i + size].clone(), kspace_scale=scale[i : i + size])
            for i in range(0, 4, size)
        ]
        torch.testing.assert_close(torch.cat(parts), full)
    per_batch = torch.cat([strategy.preprocess_input(volume[i : i + 2].clone()) for i in (0, 2)])
    assert not torch.allclose(per_batch, full)


def test_a_slice_level_arm_gets_one_scale_per_slice():
    strategy = _cold_strategy(ARM_DATA_CFG)
    volume = _multicoil_kspace(batch=3)
    volume[1] *= 4.0
    scale = strategy.volume_kspace_scale(volume, per_slice=True)
    assert scale[1] == pytest.approx(4.0 * scale[0], rel=1e-2)


def test_the_volume_scale_round_trips_through_denormalize():
    strategy = _cold_strategy(ARM_DATA_CFG)
    volume = _multicoil_kspace(batch=2)
    scale = strategy.volume_kspace_scale(volume, per_slice=False)
    normed = strategy.preprocess_input(volume.clone(), kspace_scale=scale)
    restored = strategy.denormalize_kspace(normed)
    assert torch.allclose(restored, volume, rtol=1e-3, atol=1e-4 * volume.abs().max())
