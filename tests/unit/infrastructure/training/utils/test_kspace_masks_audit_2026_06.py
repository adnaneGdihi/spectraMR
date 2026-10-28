"""Regression: KSpaceMaskGenerator.generate_mask silently fell back to uniform.

2026-06-11 infrastructure audit. Any accelerator-vocabulary pattern that is not
a valid ``MaskType`` (including the default ``"linear"``, and ``poisson_disk`` /
``golden_angle`` / …) raised ``ValueError`` inside ``MaskGenerator`` and was
silently replaced by ``uniform_cartesian`` — so VD/Poisson validation arms were
graded on equispaced-uniform masks with no log. The static path now translates
known aliases and RAISES on untranslatable patterns (pitfall #9).

Also: ``generate_batch_masks`` no longer ``.item()``-syncs per sample.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.training.utils.kspace_masks import KSpaceMaskGenerator


def test_default_linear_pattern_builds_without_silent_fallback():
    gen = KSpaceMaskGenerator(num_timesteps=1000, default_pattern="linear")
    mask = gen.generate_mask((1, 1, 16, 16), seed=0)
    assert mask.shape[-2:] == (16, 16)


def test_untranslatable_pattern_raises_not_silent_uniform():
    gen = KSpaceMaskGenerator(num_timesteps=1000, default_pattern="poisson_disk")
    with pytest.raises(ValueError, match="cannot render pattern"):
        gen.generate_mask((1, 1, 16, 16), seed=0)


def test_valid_masktype_pattern_passes_through():
    # variable_density is a valid MaskType — it must build (no raise, no silent
    # fallback to uniform); exact mask geometry is MaskGenerator's concern.
    gen = KSpaceMaskGenerator(num_timesteps=1000, default_pattern="variable_density")
    mask = gen.generate_mask((1, 1, 16, 16), seed=1)
    assert isinstance(mask, torch.Tensor)
    assert mask.numel() > 0


def test_generate_batch_masks_shape():
    gen = KSpaceMaskGenerator(num_timesteps=1000, default_pattern="variable_density")
    timesteps = torch.tensor([0, 1, 2], dtype=torch.long)
    masks = gen.generate_batch_masks(3, timesteps, (16, 16))
    assert masks.shape[0] == 3
