"""The synthetic arms declare their transforms by registered name.

``data.custom_transforms`` and ``data.augmentation.custom_transforms`` were
never read, so the entries were discarded and the arms trained without the
simulator. Both spellings now refuse to load.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")

from spectramr.config.settings import TrainingSettings
from spectramr.data.transforms.registry import build_transform

REPO = Path(__file__).resolve().parents[4]
SYNTHETIC_ARMS = [
    "experiments/inprogress/geomamba_ulf/geomamba_ulf_v0_synthetic.yaml",
    "experiments/inprogress/geomamba_ulf/baselines/baseline_lf_synth_sr.yaml",
]


@pytest.mark.parametrize("rel", SYNTHETIC_ARMS)
def test_synthetic_arm_declares_and_builds_both_transforms(rel: str) -> None:
    cfg = TrainingSettings.from_yaml(str(REPO / rel))
    specs = cfg.data.processing.transforms
    assert [spec.name for spec in specs] == ["simulate_ulf_from_hf", "foreground_mask"]
    for spec in specs:
        assert build_transform(spec.name, **spec.resolved_kwargs()) is not None


def test_slice_profile_arm_declares_its_transform() -> None:
    cfg = TrainingSettings.from_yaml(
        str(REPO / "experiments/inprogress/multi_contrast/exp_slice_profile_sr.yaml")
    )
    assert [spec.name for spec in cfg.data.processing.transforms] == ["slice_profile"]


@pytest.mark.parametrize(
    ("block_path", "planted_key"),
    [
        (("data",), "custom_transforms"),
        (("data", "augmentation"), "custom_transforms"),
    ],
)
def test_retired_custom_transforms_key_raises(tmp_path: Path, block_path, planted_key) -> None:
    doc = yaml.safe_load((REPO / SYNTHETIC_ARMS[0]).read_text())
    node = doc
    for part in block_path:
        node = node.setdefault(part, {})
    node[planted_key] = [{"name": "simulate_ulf_from_hf"}]
    planted = tmp_path / "planted.yaml"
    planted.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match=planted_key):
        TrainingSettings.from_yaml(str(planted))
