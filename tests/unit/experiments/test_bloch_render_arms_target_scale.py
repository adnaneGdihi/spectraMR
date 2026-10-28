"""The M4Raw arms that grade a Bloch render declare a target in the render's range.

`DisentangledTrainingStrategy._synthesize_via_bloch_fp32` divides the signal by its per-sample
maximum and returns ``signal * 2 - 1``, so the render lives in [-1, 1] with the brightest
pixel at +1. The first sabine run of experiment 50 declared no ``normalization_type`` (its
percentile kwargs were dead and the target was the raw RSS magnitude: train MSE ~340, train
PSNR -19 dB); experiment 51 scaled an unclamped p95 into (0, 1); experiment 32a declared
``rescale_range: [-1, 1]``, which the image normaliser never reads, so it resolved to (0, 1).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from spectramr.config.settings import TrainingSettings
from spectramr.data.transforms.normalization import normalize_tensor, resolve_image_normalization

ROOT = Path(__file__).resolve().parents[3] / "experiments/inprogress"
ARMS = [
    "vae_latent/experiment_32a_vae_hf_multicontrast.yaml",
    "physics_driven/experiment_50_strict_physical_disentangled.yaml",
    "physics_driven/experiment_51_fno_epg_multicontrast.yaml",
]


def _normaliser(arm: str):
    settings = TrainingSettings.from_yaml(str(ROOT / arm))
    processing = settings.data.processing
    spec = resolve_image_normalization(
        normalization_type=processing.normalization_type,
        dataset_type=settings.data.dataset_type,
        normalization_kwargs=processing.normalization_kwargs,
        kspace_normalization_enabled=processing.enable_kspace_normalization,
    )
    return settings, spec


@pytest.mark.parametrize("arm", ARMS)
def test_the_resolved_target_spans_the_renders_range(arm: str) -> None:
    settings, spec = _normaliser(arm)
    assert spec is not None and spec.enabled, "a raw-magnitude target is not in the render's range"

    image = torch.rand(1, 1, 8, 8) * 40.0 + 3.0  # raw RSS-magnitude scale, not [0, 1]
    out = normalize_tensor(image, spec.config)

    assert float(out.min()) == pytest.approx(-1.0, abs=1e-4)
    assert float(out.max()) == pytest.approx(1.0, abs=1e-4)
    assert settings.data.processing.data_range == 2.0  # PSNR/SSIM read this as the span
