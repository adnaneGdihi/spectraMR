"""Schema tests for ``training.disentangled_vae``."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training.disentangled_vae import DisentangledVAEConfig

_ARMS = Path(__file__).resolve().parents[5] / "experiments" / "inprogress" / "vae_latent"


def test_default_keeps_the_fixed_pair():
    assert DisentangledVAEConfig().physics_source == "fixed_t1w_64mt_3t"


@pytest.mark.parametrize("source", ["paired_field_stamps", "single_field"])
def test_advertised_sources_load(source):
    assert DisentangledVAEConfig(physics_source=source).physics_source == source


def test_unknown_source_raises():
    with pytest.raises(ValidationError, match="physics_source"):
        DisentangledVAEConfig(physics_source="batch")


def test_unknown_key_raises():
    with pytest.raises(ValidationError, match="contrast"):
        DisentangledVAEConfig(contrast="T1w")


@pytest.mark.parametrize(
    ("arm", "source"),
    [
        ("experiment_52_disentangled_vae.yaml", "paired_field_stamps"),
        ("experiment_55_disentangled_representation.yaml", "single_field"),
    ],
)
def test_arms_declare_their_source(arm, source):
    from spectramr.config.settings import TrainingSettings

    settings = TrainingSettings.from_yaml(_ARMS / arm)
    assert settings.training.disentangled_vae.physics_source == source
