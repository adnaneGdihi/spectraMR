"""The key tables are class-level constants and must not be editable through a caller."""

from __future__ import annotations

import pytest

from spectramr.models.losses.loss_key_registry import LossKeyRegistry


@pytest.mark.parametrize(
    "table",
    [
        "GAN_COMPONENTS",
        "DIFFUSION_COMPONENTS",
        "VAE_COMPONENTS",
        "VQVAE_COMPONENTS",
        "RECONSTRUCTION_COMPONENTS",
        "METRIC_KEYS",
    ],
)
def test_the_class_tables_are_read_only(table: str) -> None:
    with pytest.raises(TypeError):
        getattr(LossKeyRegistry, table)["injected"] = "x"  # type: ignore[index]


def test_a_caller_editing_the_mode_keys_cannot_rewrite_the_registry() -> None:
    keys = LossKeyRegistry.get_training_mode_keys("gan")
    assert isinstance(keys, dict)
    keys["injected"] = "x"
    assert "injected" not in LossKeyRegistry.get_training_mode_keys("gan")
    assert "injected" not in LossKeyRegistry.get_all_loss_keys()


def test_the_mode_keys_are_the_class_table() -> None:
    assert LossKeyRegistry.get_training_mode_keys("vqvae") == dict(LossKeyRegistry.VQVAE_COMPONENTS)
