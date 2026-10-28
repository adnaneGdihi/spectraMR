"""The Bloch-render arms report their disentanglement readings.

The four `val_*` readings of the disentangled strategy reach the validation CSV, but no
requested table or figure lists them: a report built on a copy of a sabine run with the columns
added contained only `tab_2_1_main_results`. `tab_run_summary` lists every recorded metric per
split, so the arms request it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from spectramr.infrastructure.reporting.catalog.registry import TABLE_EXAMPLES

ROOT = Path(__file__).resolve().parents[3] / "experiments/inprogress"
ARMS = [
    "vae_latent/experiment_32a_vae_hf_multicontrast.yaml",
    "physics_driven/experiment_50_strict_physical_disentangled.yaml",
    "physics_driven/experiment_51_fno_epg_multicontrast.yaml",
]


def test_the_summary_table_is_a_registered_report_artifact() -> None:
    assert "tab_run_summary" in TABLE_EXAMPLES


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_requests_the_summary_table(arm: str) -> None:
    tables = yaml.safe_load((ROOT / arm).read_text())["reporting"]["tables"]
    assert "tab_run_summary" in tables
