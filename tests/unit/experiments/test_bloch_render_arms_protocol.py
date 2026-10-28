"""The Bloch-render arms declare the acquisition each M4Raw contrast was scanned with (#2618).

The values are the carya header census recorded in the arms' own comments: T1w spin echo
(TR 500, TE 18.4), T2w fast spin echo (TR 5500, TE 128), FLAIR (TR 7500, TE 98, TI 1655), ms.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from spectramr.infrastructure.training.strategies.disentangled_bloch_inputs import (
    check_flip_angle_is_read,
    check_protocol_matches_families,
    parse_protocol_map,
    parse_sequence_map,
)

ROOT = Path(__file__).resolve().parents[3] / "experiments/inprogress"
ARMS = [
    "vae_latent/experiment_32a_vae_hf_multicontrast.yaml",
    "physics_driven/experiment_50_strict_physical_disentangled.yaml",
    "physics_driven/experiment_51_fno_epg_multicontrast.yaml",
]
CENSUS = {
    "T1": {"TR": 500.0, "TE": 18.4, "TI": 0.0},
    "T2": {"TR": 5500.0, "TE": 128.0, "TI": 0.0},
    "FLAIR": {"TR": 7500.0, "TE": 98.0, "TI": 1655.0},
}


def test_only_the_fno_arm_declares_a_surrogate_and_it_names_its_weights() -> None:
    """use_fno_epg without fno_epg_weights raises at start; weights without it are unread."""
    declared = {}
    for arm in ARMS:
        kwargs = yaml.safe_load((ROOT / arm).read_text())["model"]["model_kwargs"]
        declared[arm.split("/")[-1]] = (
            bool(kwargs.get("use_fno_epg")),
            kwargs.get("fno_epg_weights"),
        )
    assert declared == {
        "experiment_32a_vae_hf_multicontrast.yaml": (False, None),
        "experiment_50_strict_physical_disentangled.yaml": (False, None),
        "experiment_51_fno_epg_multicontrast.yaml": (
            True,
            "checkpoints/fno_epg/fno_epg_surrogate.pt",
        ),
    }


@pytest.mark.parametrize("arm", ARMS)
def test_no_arm_declares_a_flip_angle_that_nothing_reads(arm: str) -> None:
    kwargs = yaml.safe_load((ROOT / arm).read_text())["model"]["model_kwargs"]
    assert not kwargs.get("learnable_flip_angle", False)
    check_flip_angle_is_read(
        bool(kwargs.get("learnable_flip_angle", False)),
        parse_sequence_map(kwargs["bloch_sequence_by_contrast"]),
    )


@pytest.mark.parametrize("arm", ARMS)
def test_the_declared_protocol_is_the_census_and_agrees_with_the_families(arm: str) -> None:
    kwargs = yaml.safe_load((ROOT / arm).read_text())["model"]["model_kwargs"]
    protocol = parse_protocol_map(kwargs["bloch_protocol_by_contrast"])
    assert protocol == CENSUS
    check_protocol_matches_families(
        protocol, parse_sequence_map(kwargs["bloch_sequence_by_contrast"])
    )
