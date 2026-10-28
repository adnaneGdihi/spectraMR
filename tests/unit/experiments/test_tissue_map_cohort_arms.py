"""The tissue-map cohort's arms resolve to the model and objective they claim.

A generated B2 arm once carried two top-level ``model:`` blocks; YAML kept the last, so it
resolved to the control model while naming B2 (#2963). The raw text is read here because the
parsed settings cannot show a duplicate.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from spectramr.config.settings import TrainingSettings

ROOT = Path(__file__).resolve().parents[3] / "experiments/inprogress/physics_driven"
ARMS = {
    "qmap_m4raw_unrolled_b2.yaml": "unrolled_tissue_net",
    "qmap_m4raw_tissue_unet_c1.yaml": "bloch_parameter_bottleneck",
    "qmap_m4raw_unrolled_b2_repsplit.yaml": "unrolled_tissue_net",
    "qmap_m4raw_masked_unet_c2.yaml": "masked_contrast_unet",
    "qmap_m4raw_unrolled_b2_b1init.yaml": "unrolled_tissue_net",
}


def test_only_the_repetition_arm_reads_independent_repetitions() -> None:
    roots = {
        arm: TrainingSettings.from_yaml(str(ROOT / arm)).data.m4raw_triplet.raw_repetition_root
        for arm in ARMS
    }
    assert roots == {
        "qmap_m4raw_unrolled_b2.yaml": None,
        "qmap_m4raw_tissue_unet_c1.yaml": None,
        "qmap_m4raw_unrolled_b2_repsplit.yaml": "databases/m4raw/data_slicechunked",
        "qmap_m4raw_masked_unet_c2.yaml": None,
        "qmap_m4raw_unrolled_b2_b1init.yaml": None,
    }


@pytest.mark.parametrize("arm", ARMS)
def test_every_top_level_key_appears_once(arm: str) -> None:
    keys = re.findall(r"^([a-z_]+):", (ROOT / arm).read_text(), flags=re.M)
    assert len(keys) == len(set(keys)), sorted(k for k in set(keys) if keys.count(k) > 1)


@pytest.mark.parametrize(("arm", "model_type"), ARMS.items())
def test_the_arm_builds_the_model_it_names(arm: str, model_type: str) -> None:
    assert TrainingSettings.from_yaml(str(ROOT / arm)).model.model_type == model_type


@pytest.mark.parametrize("arm", ARMS)
def test_b2_and_its_control_train_and_select_the_same_way(arm: str) -> None:
    settings = TrainingSettings.from_yaml(str(ROOT / arm))
    mk = settings.model.model_kwargs
    assert mk["gain_mode"] == "projected" and mk["contrast_dropout"] == 0.3
    assert set(mk["gain_ratio_prior"]) == {"weight", "T2", "FLAIR"}
    assert settings.data.dataset_type == "m4raw_triplet"
    assert settings.metrics.best_metric_name == "val_psnr_loo"


def test_only_the_b1_arm_initialises_b2_with_the_contrast_transformer() -> None:
    inits = {
        arm: TrainingSettings.from_yaml(str(ROOT / arm)).model.model_kwargs.get("initializer")
        for arm in ARMS
    }
    assert inits.pop("qmap_m4raw_unrolled_b2_b1init.yaml") == "contrast_transformer"
    assert set(inits.values()) == {None}
