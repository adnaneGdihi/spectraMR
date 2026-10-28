"""Every k-space cold-diffusion arm validates through the reverse path predict uses (#2067).

Predict reconstructs a ``kspace_cold_diffusion`` model through
``generator.sample()``. Validation does the same only when
``validation.sampling.enable_multistep_cold`` is true; otherwise it scores the
single-step forward, so its metrics describe a reconstruction predict never
returns. The schema default is ``false``, so an arm that omits the key is in
violation too.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

INPROGRESS = Path(__file__).resolve().parents[3] / "experiments" / "inprogress"


def validates_single_step(doc: dict) -> bool:
    """True for a k-space cold-diffusion arm whose validation skips ``sample()``."""
    model_type = str((doc.get("model") or {}).get("model_type") or "").lower()
    if "kspace_cold_diffusion" not in model_type:
        return False
    validation = doc.get("validation") or {}
    sampling = validation.get("sampling") or {}
    flag = sampling.get("enable_multistep_cold")
    if flag is None:
        flag = validation.get("multistep_cold_sampling")  # retired flat spelling, folded at load
    return flag is not True


def _corpus() -> list[tuple[str, dict]]:
    docs = []
    for path in sorted(INPROGRESS.rglob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict):
            docs.append((str(path.relative_to(INPROGRESS)), doc))
    return docs


def test_no_inprogress_kspace_cold_arm_validates_single_step():
    offenders = [name for name, doc in _corpus() if validates_single_step(doc)]
    assert offenders == [], (
        "these kspace_cold_diffusion arms validate with the single-step forward while "
        f"predict runs generator.sample(); set validation.sampling.enable_multistep_cold: true: "
        f"{offenders}"
    )


_COLD = {"model": {"model_type": "kspace_cold_diffusion"}}


@pytest.mark.parametrize(
    "validation",
    [
        {},
        {"sampling": {"steps": 50}},
        {"sampling": {"enable_multistep_cold": False}},
        {"multistep_cold_sampling": False},
    ],
    ids=["no_validation_keys", "sampling_without_flag", "explicit_false", "retired_flat_false"],
)
def test_the_detector_is_red_on_each_single_step_shape(validation):
    assert validates_single_step({**_COLD, "validation": validation})


@pytest.mark.parametrize(
    "doc",
    [
        {**_COLD, "validation": {"sampling": {"enable_multistep_cold": True}}},
        {**_COLD, "validation": {"multistep_cold_sampling": True}},
        {"model": {"model_type": "image_cold_diffusion"}, "validation": {}},
    ],
    ids=["canonical_true", "retired_flat_true", "not_kspace_cold"],
)
def test_the_detector_passes_what_it_should(doc):
    assert not validates_single_step(doc)
