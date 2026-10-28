"""`vae_latent/experiment_50` is the image-decoder control for the Bloch-render arms.

It was marked ``needs_implementation``: single-contrast data with no label, and no metric that
could read the content/style factorisation. It now takes the multi-contrast M4Raw route that
publishes ``contrast_idx`` (which the validation probes and the swap test read), emits an image
directly, and shares its target scaling and objective with the Bloch-render arms so the two
differ only in the decoder.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from spectramr.config.schemas.base import LAUNCH_REFUSED_STATUSES
from spectramr.config.settings import TrainingSettings
from spectramr.data.transforms.normalization import normalize_tensor, resolve_image_normalization

ARM = (
    Path(__file__).resolve().parents[3]
    / "experiments/inprogress/vae_latent/experiment_50_disentangled_anatomy_contrast.yaml"
)


@pytest.fixture(scope="module")
def settings() -> TrainingSettings:
    return TrainingSettings.from_yaml(str(ARM))


def test_the_launcher_no_longer_refuses_the_arm(settings: TrainingSettings) -> None:
    assert settings.metadata.status not in LAUNCH_REFUSED_STATUSES


def test_the_data_route_publishes_a_contrast_label_per_sample(settings: TrainingSettings) -> None:
    assert settings.data.dataset_type == "m4raw"
    assert settings.data.pairing.single_contrast is True  # per-sample contrast, one contrast each


def test_the_decoder_emits_an_image_so_no_sequence_map_is_declared(
    settings: TrainingSettings,
) -> None:
    kwargs = settings.model.model_kwargs or {}
    assert kwargs.get("output_mode") != "tissue_params"
    assert "bloch_sequence_by_contrast" not in kwargs
    assert kwargs["num_classes"] >= 4  # contrast_idx runs 0..3 (T1, T2, FLAIR, PD)


def test_the_target_spans_the_tanh_decoders_range(settings: TrainingSettings) -> None:
    import torch

    processing = settings.data.processing
    spec = resolve_image_normalization(
        normalization_type=processing.normalization_type,
        dataset_type=settings.data.dataset_type,
        normalization_kwargs=processing.normalization_kwargs,
        kspace_normalization_enabled=processing.enable_kspace_normalization,
    )
    assert spec is not None and spec.enabled
    out = normalize_tensor(torch.rand(1, 1, 8, 8) * 40.0 + 3.0, spec.config)
    assert float(out.min()) == pytest.approx(-1.0, abs=1e-4)
    assert float(out.max()) == pytest.approx(1.0, abs=1e-4)
    assert processing.data_range == 2.0


def test_the_objective_matches_the_bloch_render_arms(settings: TrainingSettings) -> None:
    assert settings.losses.gan.enable_adversarial
    assert settings.losses.latent.lambda_kl > 0  # use_vae is on
    assert "lambda_patch_nce" in ARM.read_text()


def test_the_arm_reports_its_disentanglement_readings() -> None:
    """tab_run_summary lists every recorded validation key, which a report would otherwise omit."""
    cfg = yaml.safe_load(ARM.read_text())
    assert "tab_run_summary" in cfg["reporting"]["tables"]


def test_the_arm_writes_only_under_its_own_name() -> None:
    cfg = yaml.safe_load(ARM.read_text())
    paths = [
        cfg["training"]["output_dir"],
        cfg["metrics"]["output_dir"],
        cfg["logging"]["sinks"]["dir"],
    ]
    assert all("experiment_50_disentangled_anatomy_contrast" in p for p in paths), paths
    assert not [p for p in paths if "strict_physical" in p]
