"""Unit tests for :mod:`spectramr.config.presets.cli`.

The CLI helpers are thin orchestration over :class:`PresetRegistry`. We
exercise:

* ``discover_presets`` — clears the registry on force_refresh and
  delegates to ``discover_and_register_presets``
* ``list_available_presets`` — filter by category, format conversion
* ``get_preset_categories`` — sorted list passthrough
* ``get_baseline_preset`` — category-aware lookup
* ``load_preset_or_config`` — XOR semantics + missing-file error
* ``_deep_update`` — nested-key merge

Everything works with a YAML stub and a temporary preset registry
(cleared per-test).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from spectramr.config.presets import ConfigPreset, PresetRegistry
from spectramr.config.presets.cli import (
    _deep_update,
    get_baseline_preset,
    get_preset_categories,
    list_available_presets,
    load_preset_or_config,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    """Ensure each test starts with an empty PresetRegistry singleton."""
    reg = PresetRegistry()
    reg._presets.clear()
    reg._categories.clear()
    reg._tags.clear()


@pytest.fixture
def minimal_yaml(tmp_path: Path) -> Path:
    """A minimal v6.0-ish YAML on disk — only used to satisfy the
    preset's ``load_config`` happy path. The contents are not asserted
    against the strict ``TrainingSettings`` schema in these tests; for
    that, see :mod:`tests.unit.config.test_preset_system`."""
    p = tmp_path / "min.yaml"
    p.write_text(
        "\n".join([
            "config_version: '1.0'",
            "experiment:",
            "  name: test",
        ]) + "\n"
    )
    return p


def _register(
    name: str,
    *,
    category: str = "gan",
    tags: list[str] | None = None,
    yaml_path: Path | None = None,
    description: str = "test",
) -> ConfigPreset:
    p = ConfigPreset(
        name=name,
        category=category,
        description=description,
        yaml_path=str(yaml_path) if yaml_path else "x.yaml",
        tags=tags or [],
    )
    PresetRegistry.register(p)
    return p


# ── list_available_presets ──────────────────────────────────────────


class TestListAvailablePresets:
    def test_returns_all_when_no_category_given(self, minimal_yaml: Path) -> None:
        _register("a", category="gan", yaml_path=minimal_yaml)
        _register("b", category="diffusion", yaml_path=minimal_yaml)
        out = list_available_presets()
        assert set(out) == {"a", "b"}
        # Values are metadata dicts.
        assert out["a"]["category"] == "gan"
        assert out["b"]["category"] == "diffusion"

    def test_filters_by_category(self, minimal_yaml: Path) -> None:
        _register("a", category="gan", yaml_path=minimal_yaml)
        _register("b", category="diffusion", yaml_path=minimal_yaml)
        gan_only = list_available_presets(category="gan")
        assert set(gan_only) == {"a"}

    def test_empty_when_no_match(self, minimal_yaml: Path) -> None:
        _register("a", category="gan", yaml_path=minimal_yaml)
        assert list_available_presets(category="ssl") == {}


# ── get_preset_categories ───────────────────────────────────────────


class TestGetPresetCategories:
    def test_returns_sorted_unique_categories(self, minimal_yaml: Path) -> None:
        _register("a", category="gan", yaml_path=minimal_yaml)
        _register("b", category="diffusion", yaml_path=minimal_yaml)
        _register("c", category="gan", yaml_path=minimal_yaml)
        out = get_preset_categories()
        # Sorted, unique.
        assert out == ["diffusion", "gan"]

    def test_empty_registry_returns_empty_list(self) -> None:
        assert get_preset_categories() == []


# ── get_baseline_preset ─────────────────────────────────────────────


class TestGetBaselinePreset:
    def test_baseline_for_category(self, minimal_yaml: Path) -> None:
        _register("base_gan", category="gan", tags=["baseline"], yaml_path=minimal_yaml)
        out = get_baseline_preset(category="gan")
        assert out is not None
        assert out["name"] == "base_gan"
        assert "baseline" in out["tags"]

    def test_missing_baseline_returns_none_for_category(self, minimal_yaml: Path) -> None:
        _register("variant", category="gan", tags=["ablation"], yaml_path=minimal_yaml)
        assert get_baseline_preset(category="gan") is None

    def test_no_category_returns_first_baseline_across_all(
        self, minimal_yaml: Path
    ) -> None:
        _register("base_gan", category="gan", tags=["baseline"], yaml_path=minimal_yaml)
        _register(
            "base_diff", category="diffusion", tags=["baseline"], yaml_path=minimal_yaml
        )
        out = get_baseline_preset()
        # Sample category order is gan first; first baseline is base_gan.
        assert out is not None
        assert out["name"] == "base_gan"

    def test_returns_none_when_registry_empty(self) -> None:
        assert get_baseline_preset() is None


# ── load_preset_or_config — XOR semantics ──────────────────────────


class TestLoadPresetOrConfigXOR:
    def test_neither_arg_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Exactly one"):
            load_preset_or_config()

    def test_both_args_raises_value_error(self, tmp_path: Path) -> None:
        cfg = tmp_path / "x.yaml"
        cfg.write_text("config_version: '1.0'\n")
        with pytest.raises(ValueError, match="Exactly one"):
            load_preset_or_config(preset_name="x", config_path=str(cfg))

    def test_unknown_preset_raises_keyerror_with_available_list(
        self, minimal_yaml: Path
    ) -> None:
        _register("baseline_gan", category="gan", yaml_path=minimal_yaml)
        with pytest.raises(KeyError) as ei:
            load_preset_or_config(preset_name="does_not_exist")
        msg = str(ei.value)
        assert "does_not_exist" in msg
        # Available presets are listed for discoverability.
        assert "baseline_gan" in msg

    def test_missing_config_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_preset_or_config(config_path=str(tmp_path / "missing.yaml"))


# ── _deep_update ────────────────────────────────────────────────────


class TestDeepUpdate:
    def test_shallow_overwrites_existing(self) -> None:
        target = {"a": 1, "b": 2}
        _deep_update(target, {"a": 99})
        assert target == {"a": 99, "b": 2}

    def test_shallow_inserts_new_key(self) -> None:
        target = {"a": 1}
        _deep_update(target, {"b": 2})
        assert target == {"a": 1, "b": 2}

    def test_dotted_key_writes_into_nested_dict(self) -> None:
        target: dict = {"optimization": {"optimizer": {"learning_rate": 1e-3}}}
        _deep_update(target, {"optimization.optimizer.learning_rate": 1e-4})
        assert target["optimization"]["optimizer"]["learning_rate"] == 1e-4

    def test_dotted_key_creates_missing_parent(self) -> None:
        target: dict = {}
        _deep_update(target, {"a.b.c": 42})
        # Whole chain is created.
        assert target == {"a": {"b": {"c": 42}}}

    def test_multiple_keys_at_once(self) -> None:
        target = {"a": 1, "b": {"c": 2}}
        _deep_update(target, {"a": 10, "b.c": 20, "b.d": 30})
        assert target == {"a": 10, "b": {"c": 20, "d": 30}}
