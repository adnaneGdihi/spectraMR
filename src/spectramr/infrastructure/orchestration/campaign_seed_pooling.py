"""Seed replicates of one arm pooled into a single campaign arm.

An arm declares ``tags.seed`` (an integer). Arms that share a replicate group
(``tags.arm``, else the name with its ``_seedN`` suffix stripped) are averaged
per sample into one arm, so a second seed is not a second baseline. The average
is over the shared test set, so per-sample pairing still holds; replicates with
unequal lengths or a missing metric raise.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from spectramr.config.schemas.campaign import CampaignExperimentSchema

__all__ = ["SeedMember", "pool_replicates", "seed_members"]


@dataclass(frozen=True)
class SeedMember:
    """One declared arm that carries ``tags.seed``."""

    name: str
    group: str
    role: str
    seed: int


def seed_members(declared: Iterable[CampaignExperimentSchema]) -> dict[str, SeedMember]:
    """Map each arm that declares ``tags.seed`` to its replicate group.

    Raises:
        ValueError: a group mixes roles or repeats a seed.
    """
    members: dict[str, SeedMember] = {}
    for exp in declared:
        if "seed" not in exp.tags:
            continue
        group = exp.replicate_group
        members[exp.name] = SeedMember(exp.name, group, exp.role, int(exp.tags["seed"]))
    by_group: dict[str, list[SeedMember]] = {}
    for member in members.values():
        by_group.setdefault(member.group, []).append(member)
    for group, group_members in by_group.items():
        roles = {m.role for m in group_members}
        if len(roles) > 1:
            raise ValueError(f"Replicate group '{group}' mixes roles {sorted(roles)}")
        seeds = [m.seed for m in group_members]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Replicate group '{group}' repeats a seed: {sorted(seeds)}")
    return members


def pool_replicates(
    all_metrics: dict[str, dict[str, np.ndarray]],
    members: dict[str, SeedMember],
) -> dict[str, dict[str, np.ndarray]]:
    """Collapse each replicate group into one arm named by its group.

    Raises:
        ValueError: a declared replicate is absent from ``all_metrics``, a group
            name collides with another arm, or replicates disagree on a metric's
            length or availability.
    """
    pooled: dict[str, dict[str, np.ndarray]] = {}
    groups: dict[str, list[str]] = {}
    declared: dict[str, set[str]] = {}
    for name, values in all_metrics.items():
        member = members.get(name)
        if member is None:
            pooled[name] = values
        else:
            groups.setdefault(member.group, []).append(name)
    for name, member in members.items():
        declared.setdefault(member.group, set()).add(name)
    for group, expected in declared.items():
        missing = sorted(expected - set(groups.get(group, [])))
        if missing:
            raise ValueError(
                f"Replicate group '{group}' is missing declared replicates {missing} "
                "(failed or not evaluable); a pooled arm must not shrink silently"
            )
    for group, names in groups.items():
        if group in pooled:
            raise ValueError(f"Replicate group '{group}' collides with an arm of that name")
        metric_names = set().union(*(all_metrics[n].keys() for n in names))
        out: dict[str, np.ndarray] = {}
        for metric in sorted(metric_names):
            present = [n for n in names if metric in all_metrics[n]]
            if len(present) != len(names):
                raise ValueError(f"Replicates of '{group}' disagree on metric '{metric}'")
            lengths = {len(all_metrics[n][metric]) for n in names}
            if len(lengths) != 1:
                raise ValueError(f"Replicates of '{group}' have unequal lengths for '{metric}'")
            out[metric] = np.mean(np.stack([all_metrics[n][metric] for n in names]), axis=0)
        pooled[group] = out
    return pooled
