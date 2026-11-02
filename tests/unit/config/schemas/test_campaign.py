"""Campaign schema: a seeded arm must have a replicate partner at load."""

from __future__ import annotations

import pytest

from spectramr.config.schemas.campaign import CampaignConfigSchema, CampaignExperimentSchema


def _exp(name: str, enabled: bool = True, **tags: str) -> CampaignExperimentSchema:
    return CampaignExperimentSchema(name=name, config=f"{name}.yaml", tags=tags, enabled=enabled)


def test_lone_seeded_arm_is_rejected_at_load():
    with pytest.raises(ValueError, match="alone in their replicate group"):
        CampaignConfigSchema(name="c", experiments=[_exp("cyclegan_s42", seed="42", arm="cg")])


def test_lone_seed_arm_named_by_its_suffix_is_rejected():
    with pytest.raises(ValueError, match="\\['cg'\\]"):
        CampaignConfigSchema(name="c", experiments=[_exp("cg_seed42", seed="42")])


def test_a_disabled_partner_does_not_make_a_pair():
    with pytest.raises(ValueError, match="alone in their replicate group"):
        CampaignConfigSchema(
            name="c",
            experiments=[
                _exp("cg_seed42", seed="42"),
                _exp("cg_seed43", enabled=False, seed="43"),
            ],
        )


def test_a_replicate_pair_loads():
    campaign = CampaignConfigSchema(
        name="c",
        experiments=[_exp("cg_seed42", seed="42"), _exp("cg_seed43", seed="43")],
    )
    assert {e.replicate_group for e in campaign.experiments} == {"cg"}


def test_unseeded_arms_are_not_replicates():
    campaign = CampaignConfigSchema(name="c", experiments=[_exp("plain"), _exp("other")])
    assert len(campaign.experiments) == 2
