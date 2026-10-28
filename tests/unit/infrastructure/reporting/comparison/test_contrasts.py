"""``comparison/contrasts.py``: the contrast file is strict, and a name cannot leave the folder."""

import pytest
import yaml

from spectramr.infrastructure.reporting.comparison.contrasts import load_contrast_set

BASE = {
    "results_root": "runs",
    "metrics": ["val_psnr"],
    "contrasts": [{"name": "a_vs_b", "arms": ["arm_a", "arm_b"], "baseline": "arm_b"}],
}


def _load(tmp_path, **over):
    path = tmp_path / "set.yaml"
    path.write_text(yaml.safe_dump({**BASE, **over}))
    return load_contrast_set(path)


def test_a_bare_run_name_is_an_arm_labelled_by_its_directory(tmp_path):
    contrast = _load(tmp_path).contrasts[0]
    assert contrast.labels == ["arm_a", "arm_b"]
    assert contrast.baseline_label == "arm_b"


def test_the_baseline_defaults_to_the_first_arm(tmp_path):
    only = {"name": "x", "arms": [{"run": "r1", "label": "one"}, "r2"]}
    contrast = _load(tmp_path, contrasts=[only]).contrasts[0]
    assert contrast.labels == ["one", "r2"]
    assert contrast.baseline_label == "one"


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"name": "../escape", "arms": ["a", "b"]}, "safe folder name"),
        ({"name": "/abs", "arms": ["a", "b"]}, "safe folder name"),
        ({"name": "x", "arms": ["only_one"]}, "at least two arms"),
        ({"name": "x", "arms": ["a", "a"]}, "duplicate arm labels"),
        ({"name": "x", "arms": ["a", "b"], "baseline": "c"}, "not an arm"),
        ({"name": "x", "arms": ["a", "b"], "famly": "typo"}, "Extra inputs"),
    ],
)
def test_an_inconsistent_contrast_raises(tmp_path, bad, match):
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, contrasts=[bad])


def test_metrics_are_optional_but_not_empty(tmp_path):
    path = tmp_path / "set.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in BASE.items() if k != "metrics"}))
    assert load_contrast_set(path).metrics is None
    with pytest.raises(ValueError, match="omit it to test every metric"):
        _load(tmp_path, metrics=[])


def test_duplicate_contrast_names_raise(tmp_path):
    twice = [{"name": "x", "arms": ["a", "b"]}, {"name": "x", "arms": ["c", "d"]}]
    with pytest.raises(ValueError, match="duplicate contrast names"):
        _load(tmp_path, contrasts=twice)


def test_an_unknown_view_or_top_level_key_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown checkpoint view"):
        _load(tmp_path, view="best")
    with pytest.raises(ValueError, match="Extra inputs"):
        _load(tmp_path, metric=["typo"])


def test_relative_runs_resolve_under_the_results_root(tmp_path):
    cs = _load(tmp_path)
    assert cs.run_dir(cs.contrasts[0].arms[0]).as_posix() == "runs/arm_a"
    assert (
        cs.run_dir(cs.contrasts[0].arms[0], tmp_path).as_posix() == f"{tmp_path.as_posix()}/arm_a"
    )


def test_phases_default_to_all_three_and_reject_unknown_ones(tmp_path):
    assert _load(tmp_path).phases == ("val", "train", "test")
    assert _load(tmp_path, phases=["val", "test"]).phases == ("val", "test")
    with pytest.raises(ValueError, match="unknown phase"):
        _load(tmp_path, phases=["inference"])
    with pytest.raises(ValueError, match="at least one"):
        _load(tmp_path, phases=[])


def test_a_ladder_is_optional_and_strict(tmp_path):
    arms = ["a", "b"]
    plain = _load(tmp_path, contrasts=[{"name": "x", "arms": arms}]).contrasts[0]
    assert plain.ladder is None
    ladder = {"metric": "nex_equivalent", "references": {"identity": 1, "two averages": 2}}
    got = _load(tmp_path, contrasts=[{"name": "x", "arms": arms, "ladder": ladder}]).contrasts[0]
    assert got.ladder.metric == "nex_equivalent" and got.ladder.references["two averages"] == 2
    with pytest.raises(ValueError, match="Extra inputs"):
        _load(
            tmp_path, contrasts=[{"name": "x", "arms": arms, "ladder": {"metric": "m", "refs": {}}}]
        )


def test_an_arm_may_declare_a_group_and_the_key_is_strict(tmp_path):
    arms = [{"run": "r1", "label": "one", "group": "single"}, {"run": "r2", "label": "two"}]
    contrast = _load(tmp_path, contrasts=[{"name": "x", "arms": arms}]).contrasts[0]
    assert contrast.group_map == {"one": "single"}
    bad = [{"run": "r1", "grp": "single"}, "r2"]
    with pytest.raises(ValueError, match="Extra inputs"):
        _load(tmp_path, contrasts=[{"name": "x", "arms": bad}])
