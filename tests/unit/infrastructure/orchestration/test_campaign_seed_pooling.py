"""Seed-replicate pooling: arms that differ only by ``tags.seed`` become one arm."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spectramr.config.schemas.campaign import CampaignConfigSchema, CampaignExperimentSchema
from spectramr.infrastructure.orchestration.campaign_seed_pooling import (
    pool_replicates,
    seed_members,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def _exp(name: str, role: str = "variant", **tags: str) -> CampaignExperimentSchema:
    return CampaignExperimentSchema(name=name, config=f"{name}.yaml", role=role, tags=tags)


def test_group_defaults_to_the_name_without_its_seed_suffix():
    members = seed_members([_exp("arm_seed42", seed="42"), _exp("arm_seed43", seed="43")])
    assert {m.group for m in members.values()} == {"arm"}


def test_tags_arm_overrides_the_name_stem():
    members = seed_members([_exp("x", seed="1", arm="grp"), _exp("y", seed="2", arm="grp")])
    assert {m.group for m in members.values()} == {"grp"}


def test_arms_without_a_seed_tag_are_not_members():
    assert seed_members([_exp("plain")]) == {}


def test_mixed_roles_in_a_group_raise():
    with pytest.raises(ValueError, match="mixes roles"):
        seed_members(
            [
                _exp("a_seed1", role="baseline", seed="1"),
                _exp("a_seed2", role="variant", seed="2"),
            ]
        )


def test_repeated_seed_in_a_group_raises():
    with pytest.raises(ValueError, match="repeats a seed"):
        seed_members([_exp("p", seed="1", arm="a"), _exp("q", seed="1", arm="a")])


def test_non_integer_seed_is_rejected_by_the_schema():
    with pytest.raises(ValueError, match="integer"):
        _exp("a", seed="abc")


def test_pooling_averages_per_sample_and_keeps_other_arms():
    members = seed_members([_exp("b_seed1", seed="1"), _exp("b_seed2", seed="2")])
    metrics = {
        "b_seed1": {"psnr": np.array([1.0, 2.0])},
        "b_seed2": {"psnr": np.array([3.0, 4.0])},
        "x": {"psnr": np.array([9.0, 9.0])},
    }
    pooled = pool_replicates(metrics, members)
    assert set(pooled) == {"b", "x"}
    np.testing.assert_allclose(pooled["b"]["psnr"], [2.0, 3.0])


def test_unequal_replicate_lengths_raise():
    members = seed_members([_exp("b_seed1", seed="1"), _exp("b_seed2", seed="2")])
    metrics = {
        "b_seed1": {"psnr": np.array([1.0, 2.0])},
        "b_seed2": {"psnr": np.array([3.0])},
    }
    with pytest.raises(ValueError, match="unequal lengths"):
        pool_replicates(metrics, members)


def test_group_name_colliding_with_an_arm_raises():
    members = seed_members([_exp("b_seed1", seed="1"), _exp("b_seed2", seed="2")])
    metrics = {
        "b_seed1": {"psnr": np.array([1.0])},
        "b_seed2": {"psnr": np.array([1.0])},
        "b": {"psnr": np.array([1.0])},
    }
    with pytest.raises(ValueError, match="collides"):
        pool_replicates(metrics, members)


def test_declared_replicate_absent_from_metrics_raises():
    members = seed_members([_exp("b_seed1", seed="1"), _exp("b_seed2", seed="2")])
    metrics = {"b_seed1": {"psnr": np.array([1.0, 2.0])}, "x": {"psnr": np.array([9.0])}}
    with pytest.raises(ValueError, match="missing declared replicates \\['b_seed2'\\]"):
        pool_replicates(metrics, members)


def test_whole_replicate_group_absent_from_metrics_raises():
    members = seed_members([_exp("b_seed1", seed="1"), _exp("b_seed2", seed="2")])
    with pytest.raises(ValueError, match="missing declared replicates"):
        pool_replicates({"x": {"psnr": np.array([9.0])}}, members)


def test_cs_mno_campaign_pools_one_group_per_arm():
    path = REPO_ROOT / "experiments" / "campaigns" / "cs_mno_2d_warmup.yaml"
    cfg = CampaignConfigSchema.from_yaml(str(path))
    members = seed_members(cfg.enabled_experiments)
    groups: dict[str, int] = {}
    for member in members.values():
        groups[member.group] = groups.get(member.group, 0) + 1
    assert groups == {
        "cs_mno_v0": 3,
        "cs_mno_ablation_raster_scan": 3,
        "cs_mno_ablation_no_physical_arc": 3,
    }


def test_unpaired_ulf_campaign_pools_seven_groups_of_three():
    path = REPO_ROOT / "experiments" / "campaigns" / "unpaired_ulf_transport.yaml"
    cfg = CampaignConfigSchema.from_yaml(str(path))
    groups: dict[str, int] = {}
    for member in seed_members(cfg.enabled_experiments).values():
        groups[member.group] = groups.get(member.group, 0) + 1
    assert groups == {
        "cyclegan_control": 3,
        "paired_kspace_cold_diffusion": 3,
        "mind_topo_lock": 3,
        "grl_disentangle": 3,
        "spectral_band_split": 3,
        "b0_hypernetwork": 3,
        "paired_unet_reference": 3,
    }
