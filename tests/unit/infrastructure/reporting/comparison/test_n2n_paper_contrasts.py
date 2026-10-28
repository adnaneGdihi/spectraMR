"""The n2n contrast file and the paper script's CONTRASTS name the same comparisons."""

import sys
from pathlib import Path

from spectramr.infrastructure.reporting.comparison.contrasts import load_contrast_set

REPO = Path(__file__).resolve().parents[5]


def test_the_contrast_file_matches_the_paper_scripts_contrasts():
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        from n2n_paper_common import CONTRASTS
    finally:
        sys.path.remove(str(REPO / "scripts"))
    declared = load_contrast_set(REPO / "experiments/contrasts/n2n_paper.yaml")
    in_file = [
        (next(a for a in c.labels if a != c.baseline_label), c.baseline_label, c.family)
        for c in declared.contrasts
        if len(c.arms) == 2  # the group contrasts (the ladder) are not in CONTRASTS
    ]
    assert in_file == list(CONTRASTS)


def test_the_ladder_group_reads_the_nex_equivalent_against_one_two_and_three_averages():
    declared = load_contrast_set(REPO / "experiments/contrasts/n2n_paper.yaml")
    group = next(c for c in declared.contrasts if c.name == "nex_ladder")
    assert group.ladder.metric == "nex_equivalent"
    assert list(group.ladder.references.values()) == [1, 2, 3]
    assert sorted(group.labels) == ["A", "B", "C", "G"] and group.baseline_label == "B"
    assert group.group_map == {
        "B": "acquired averages",
        "C": "acquired averages",
        "A": "single excitation",
        "G": "single excitation",
    }


def test_the_r2r_family_groups_the_constraint_arms_and_the_noise_model_arms():
    declared = load_contrast_set(REPO / "experiments/contrasts/n2n_paper.yaml")
    family = next(c for c in declared.contrasts if c.name == "r2r_family")
    assert family.baseline_label == "A"
    assert family.group_map == {
        "A": "baseline",
        "D": "coil constraint",
        "F": "coil constraint",
        "E": "noise model",
        "G": "noise model",
    }
