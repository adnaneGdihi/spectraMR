"""The validation-metric migration swaps psnr/ssim for their tissue-support forms, nothing else."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]

ARM = """\
model:
  model_type: {model_type}
validation:
  scoring:
    measurement_maps: metrics
    compute:
    - psnr
    - robust_mri_psnr  # tissue only
    - hfen
other:
  keep: 1
"""


def _load():
    path = REPO / "scripts/migrations/robust_validation_metrics.py"
    spec = importlib.util.spec_from_file_location("robust_validation_metrics", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["psnr", "robust_mri_psnr", "hfen"], ["robust_mri_psnr", "clinical_ssim", "hfen"]),
        (["psnr", "ssim", "hfen"], ["robust_mri_psnr", "clinical_ssim", "hfen"]),
        (["robust_mri_psnr", "clinical_ssim"], ["robust_mri_psnr", "clinical_ssim"]),
        (["hfen"], ["hfen"]),
    ],
)
def test_migrate_names(names: list[str], expected: list[str]) -> None:
    assert _load().migrate_names(names) == expected


def test_process_rewrites_the_list_keeps_comments_and_is_idempotent(tmp_path: Path) -> None:
    mod = _load()
    arm = tmp_path / "arm.yaml"
    arm.write_text(ARM.format(model_type="kspace_cold_diffusion"))
    assert mod.process(arm, apply=True).startswith("WROTE")
    text = arm.read_text()
    assert "# tissue only" in text
    cfg = yaml.safe_load(text)
    assert cfg["validation"]["scoring"]["compute"] == ["robust_mri_psnr", "clinical_ssim", "hfen"]
    assert cfg["other"] == {"keep": 1}
    assert mod.process(arm, apply=True).startswith("OK")


def test_baselines_keep_their_published_metrics(tmp_path: Path) -> None:
    mod = _load()
    arm = tmp_path / "baseline.yaml"
    arm.write_text(ARM.format(model_type="fdb_baseline"))
    assert mod.process(arm, apply=True).startswith("SKIP")
    assert yaml.safe_load(arm.read_text())["validation"]["scoring"]["compute"][0] == "psnr"
