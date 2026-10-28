"""Tests for ``spectramr.config.io`` — raw YAML I/O for build-time tools.

This helper is the SSOT-compliant escape valve: legitimate build-time
code paths (ablation generation, HPO winner-config materialisation)
that need a mutable raw dict route through here instead of calling
``yaml.safe_load`` directly. The pre-commit guard
``tests/unit/test_no_yaml_reparse.py`` still passes because
``src/config/`` is the allowed subtree.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from spectramr.config.io import DuplicateKeyError, read_yaml_dict, write_yaml_dict


def test_read_yaml_dict_returns_dict(tmp_path: Path) -> None:
    """A simple YAML file round-trips through ``read_yaml_dict`` as a dict."""
    p = tmp_path / "config.yaml"
    p.write_text("a: 1\nb:\n  c: 2\n")
    out = read_yaml_dict(p)
    assert out == {"a": 1, "b": {"c": 2}}


def test_read_yaml_dict_empty_file_returns_empty_dict(tmp_path: Path) -> None:
    """An empty YAML file yields an empty dict, never ``None``."""
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert read_yaml_dict(p) == {}


def test_read_yaml_dict_raises_on_missing_file(tmp_path: Path) -> None:
    """Missing file raises ``FileNotFoundError`` (not a silent empty result)."""
    with pytest.raises(FileNotFoundError, match="YAML file not found"):
        read_yaml_dict(tmp_path / "does_not_exist.yaml")


def test_write_yaml_dict_creates_parent_dirs(tmp_path: Path) -> None:
    """Nested output directories are created on demand."""
    target = tmp_path / "deep" / "nested" / "config.yaml"
    write_yaml_dict({"x": 1}, target)
    assert target.exists()


def test_round_trip_preserves_structure(tmp_path: Path) -> None:
    """``read_yaml_dict ∘ write_yaml_dict`` preserves dict structure."""
    data = {"model": {"type": "unet", "depth": 4}, "lr": 1e-4, "seeds": [0, 1, 42]}
    target = tmp_path / "config.yaml"
    write_yaml_dict(data, target)
    assert read_yaml_dict(target) == data


# PyYAML keeps the last of two equal keys without a word; a generated arm once
# carried two ``model:`` blocks and trained the control model under the B2 name (#2963).


@pytest.mark.parametrize(
    ("text", "key", "line"),
    [
        ("model:\n  a: 1\ndata:\n  b: 2\nmodel:\n  a: 3\n", "model", 5),
        ("model:\n  depth: 4\n  width: 8\n  depth: 6\n", "depth", 4),
        ("losses:\n  - name: l1\n    weight: 1\n    name: l2\n", "name", 4),
        ("model: {depth: 4, depth: 6}\n", "depth", 1),
    ],
    ids=["top_level", "nested", "inside_list_item", "flow_style_same_line"],
)
def test_read_yaml_dict_rejects_a_repeated_key(
    tmp_path: Path, text: str, key: str, line: int
) -> None:
    p = tmp_path / "dup.yaml"
    p.write_text(text)
    with pytest.raises(DuplicateKeyError, match=rf"'{key}'.*line {line}"):
        read_yaml_dict(p)


def test_a_key_overriding_a_merged_anchor_is_not_a_duplicate(tmp_path: Path) -> None:
    p = tmp_path / "merge.yaml"
    p.write_text("base: &b\n  lr: 1\n  wd: 0\narm:\n  <<: *b\n  lr: 2\n")
    assert read_yaml_dict(p)["arm"] == {"lr": 2, "wd": 0}


def test_training_settings_rejects_a_repeated_key(tmp_path: Path) -> None:
    from spectramr.config.settings import TrainingSettings

    p = tmp_path / "arm.yaml"
    p.write_text("config_version: '1.0'\nmodel:\n  model_type: a\nmodel:\n  model_type: b\n")
    with pytest.raises(DuplicateKeyError, match="'model'"):
        TrainingSettings.from_yaml(p)


def test_an_anchor_that_itself_merges_and_overrides_is_not_a_duplicate(tmp_path: Path) -> None:
    # Merging rewrites an aliased node in place, so a check run after it saw the anchor's
    # own override next to the key it had merged in.
    p = tmp_path / "nested_merge.yaml"
    p.write_text(
        "a:\n  base: &base\n    lr: 1\n  opt: &opt\n    <<: *base\n    lr: 2\nb:\n  <<: *opt\n"
    )
    assert read_yaml_dict(p)["b"] == {"lr": 2}


def test_an_unhashable_key_stays_a_yaml_error(tmp_path: Path) -> None:
    p = tmp_path / "unhashable.yaml"
    p.write_text("? [a, b]\n: 1\n")
    with pytest.raises(yaml.YAMLError, match="unhashable"):
        read_yaml_dict(p)
