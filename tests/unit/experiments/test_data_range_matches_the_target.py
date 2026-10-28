"""An arm's declared ``data_range`` is the span its image normaliser actually produces.

PSNR and SSIM take ``data.processing.data_range`` as the span of the signal. 30 arms in
``ulf_paired_restoration`` and ``ldm_two_stage_ulf_to_hf`` declared ``rescale_range: [-1, 1]``
(read by nothing) and ``data_range: 2.0`` while ``minmax`` resolved to (0, 1), so their PSNR
read about 6 dB high (#2739). The check applies the arm's resolved normaliser to a raw
magnitude and compares the span it produces with the declaration.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from spectramr.data.transforms.normalization import ImageNormalizationSpec, normalize_tensor

ROOT = Path(__file__).resolve().parents[3] / "experiments/inprogress"


def _bounded_arms() -> list[tuple[str, float, ImageNormalizationSpec]]:
    """Arms whose normaliser has a fixed output span: minmax, or a clamped percentile."""
    out = []
    for path in sorted(ROOT.rglob("*.yaml")):
        try:
            cfg = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        data = cfg.get("data") if isinstance(cfg, dict) else None
        proc = data.get("processing") if isinstance(data, dict) else None
        if not isinstance(proc, dict) or proc.get("data_range") is None:
            continue
        if proc.get("enable_kspace_normalization"):
            continue
        ntype = proc.get("normalization_type", "none")
        if ntype in (None, "none"):
            continue
        spec = ImageNormalizationSpec.from_declared(
            ntype, data.get("dataset_type"), proc.get("normalization_kwargs")
        )
        clamped = spec.config.strategy.value == "minmax" or spec.config.clamp
        if spec.enabled and clamped:
            out.append((str(path.relative_to(ROOT)), float(proc["data_range"]), spec))
    return out


ARMS = _bounded_arms()


def test_the_corpus_has_bounded_arms_to_check() -> None:
    assert len(ARMS) >= 30


@pytest.mark.parametrize(("arm", "declared", "spec"), ARMS, ids=[a[0] for a in ARMS])
def test_the_declared_data_range_is_the_produced_span(
    arm: str, declared: float, spec: ImageNormalizationSpec
) -> None:
    raw = torch.rand(1, 1, 16, 16, generator=torch.Generator().manual_seed(0)) * 400.0 + 7.0
    out = normalize_tensor(raw, spec.config)
    span = float(out.max() - out.min())
    assert span == pytest.approx(declared, abs=0.05), (
        f"{arm}: data_range {declared} but the normaliser produces a span of {span:.2f} "
        f"({spec.config.strategy.value}, out_range {spec.config.out_range})"
    )


def test_the_check_catches_a_planted_mismatch() -> None:
    """Planted violation: the #2739 shape (minmax with no out_range, data_range 2.0)."""
    spec = ImageNormalizationSpec.from_declared("minmax", "nifti_paired", {})
    out = normalize_tensor(torch.rand(1, 1, 8, 8) * 50 + 3, spec.config)
    assert float(out.max() - out.min()) != pytest.approx(2.0, abs=0.05)
