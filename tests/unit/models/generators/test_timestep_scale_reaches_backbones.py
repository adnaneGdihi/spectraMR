"""``model_kwargs.timestep_embedding_scale`` reaches every cohort backbone (#2315).

Driven through ``KSpaceColdDiffusionGenerator.forward``, the production path,
with a spy on the value the backbone's sinusoid receives. A backbone that drops
the ``timestep_scale`` kwarg keeps dividing by the horizon, and its spy then
records 3/29 instead of 3.
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch

import spectramr.models.blocks.timestep_embedding as te
from spectramr.models.generators.complex_unet import ComplexUNet
from spectramr.models.generators.kspace_cold_diffusion_generator import (
    KSpaceColdDiffusionGenerator,
)

pytestmark = pytest.mark.unit

T = 29
STEP = 3


def _generator(backbone: str, scale: str | None, **extra):
    kwargs = dict(
        in_channels=2,
        out_channels=2,
        base_channels=24,
        num_layers=2,
        attention_type="none",
        backbone_type=backbone,
        condition_with_smaps=False,
        timesteps=T,
        dc_method="none",
        kspace_log_scaled=False,
        **({"force_pure_kspace": True, "image_size": (32, 32)}
           if backbone in {"kspace_ar", "kspace_masked"} else {}),
        **extra,
    )
    if scale is not None:
        kwargs["timestep_embedding_scale"] = scale
    return KSpaceColdDiffusionGenerator(**kwargs).eval()


def _seen_timesteps(model) -> list[tuple[float, str, float | None]]:
    """``(value, scale, horizon)`` for every sinusoid call during one forward at t=STEP.

    ``value`` is what the legacy ladder turns into phases: ``t / T`` under the
    dividing form, the raw step otherwise. complex_unet's private sinusoid, which
    owns its ``unit_interval`` and ``integer`` codes, is labelled ``"private"``.
    """
    seen: list[tuple[float, str, float | None]] = []
    shared = te.sinusoidal_timestep_embedding
    private = ComplexUNet._get_sinusoidal_embedding

    def spy_shared(t, dim, *, max_timesteps=None, scale="unit_interval"):
        value = float(t.float().flatten()[0])
        if scale == "unit_interval" and max_timesteps is not None:
            value /= max_timesteps
        seen.append((value, scale, max_timesteps))
        return shared(t, dim, max_timesteps=max_timesteps, scale=scale)

    def spy_private(self, t, dim):
        seen.append((float(t.flatten()[0]), "private", None))
        return private(self, t, dim)

    targets = [
        "spectramr.models.generators.complex_unet.sinusoidal_timestep_embedding",
        "spectramr.models.generators.diffusion_transformer_stem.sinusoidal_timestep_embedding",
        "spectramr.models.generators.swin_diff_rec.sinusoidal_timestep_embedding",
        "spectramr.models.generators.diff_varnet.sinusoidal_timestep_embedding",
    ]
    patches = [mock.patch(p, spy_shared) for p in targets]
    patches.append(mock.patch.object(ComplexUNet, "_get_sinusoidal_embedding", spy_private))
    for p in patches:
        p.start()
    try:
        with torch.no_grad():
            # The v2 generator builds its geometry planes from the rung mask.
            model(
                torch.randn(1, 2, 32, 32),
                timesteps=torch.tensor([STEP]),
                mask=torch.ones(1, 1, 32, 32),
            )
    finally:
        for p in patches:
            p.stop()
    return seen


BACKBONES = ["complex_unet", "dit", "hat", "uvit", "diffit", "swin_diff_rec", "diff_varnet",
             "kspace_ar", "kspace_masked"]


@pytest.mark.parametrize("backbone", BACKBONES)
def test_integer_scale_reaches_the_backbone_sinusoid(backbone: str) -> None:
    seen = _seen_timesteps(_generator(backbone, "integer"))
    assert seen, f"{backbone}: no sinusoid was called -- the spy targets are stale"
    assert all(v == pytest.approx(STEP) and s in {"integer", "private"} for v, s, _ in seen), seen


@pytest.mark.parametrize("backbone", BACKBONES)
def test_horizon_band_scale_reaches_the_backbone_sinusoid(backbone: str) -> None:
    """The band is derived from the horizon, so the sinusoid must receive it."""
    seen = _seen_timesteps(_generator(backbone, "horizon_band"))
    assert seen and all(
        v == pytest.approx(STEP) and s == "horizon_band" and m == pytest.approx(T)
        for v, s, m in seen
    ), seen


@pytest.mark.parametrize("backbone", BACKBONES)
def test_default_keeps_the_dividing_form(backbone: str) -> None:
    """PLANTED: the legacy codes, which is what a dropped kwarg also produces."""
    seen = _seen_timesteps(_generator(backbone, None))
    assert seen and all(v == pytest.approx(STEP / T) for v, _, _ in seen), seen


def test_unknown_scale_raises_at_build() -> None:
    with pytest.raises(ValueError, match="timestep embedding scale"):
        _generator("complex_unet", "log")
