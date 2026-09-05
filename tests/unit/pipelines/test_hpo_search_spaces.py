"""Unit tests for HPO search-space specs and the dotted-path override util.

CPU-only — these tests stub out any Optuna trial via a small mock that
records the suggest_* calls, so we can verify the search-space spec
without actually running an HPO study.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

# The dotted-override half moved to infrastructure/hpo/dotted_override.py and
# this import was not updated with it, so the module raised ImportError at
# COLLECTION time -- which pytest answers with "Interrupted: N errors during
# collection", running zero tests in the whole session rather than failing one
# file. Import each name from the module that defines it; a re-export from
# hpo_search_spaces would make that module a second owner of names it no longer
# has (non-negotiable 17).
from spectramr.infrastructure.hpo.dotted_override import (
    _ListSelector,
    _tokenize_path,
    apply_dotted_override,
)
from spectramr.pipelines.hpo_search_spaces import (
    PRESETS,
    Distribution,
    SearchSpace,
    enumerate_schema_paths,
    list_presets,
    load_preset,
    load_presets,
    make_search_space_template,
)
from tests.utils.repo_scripts import require_repo_file

# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------


class TestDistribution:
    def test_categorical_requires_choices(self):
        with pytest.raises(ValueError, match="categorical distribution requires"):
            Distribution(kind="categorical")

    def test_continuous_requires_low_high(self):
        with pytest.raises(ValueError, match="requires both low and high"):
            Distribution(kind="loguniform", low=1.0)

    def test_low_must_be_below_high(self):
        with pytest.raises(ValueError, match="must be < high"):
            Distribution(kind="uniform", low=1.0, high=1.0)

    def test_loguniform_requires_positive_low(self):
        with pytest.raises(ValueError, match="loguniform distributions require low > 0"):
            Distribution(kind="loguniform", low=0.0, high=1.0)

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match=r"Distribution\.kind must be"):
            Distribution(kind="exponential", low=0.1, high=1.0)

    def test_sample_dispatches_to_correct_optuna_method(self):
        """Distribution.sample calls the matching trial.suggest_* method."""
        trial = MagicMock()
        Distribution(kind="uniform", low=0.0, high=1.0).sample(trial, "x")
        trial.suggest_float.assert_called_with("x", 0.0, 1.0)

        Distribution(kind="loguniform", low=1e-5, high=1e-3).sample(trial, "lr")
        trial.suggest_float.assert_called_with("lr", 1e-5, 1e-3, log=True)

        Distribution(kind="int_uniform", low=2, high=8).sample(trial, "k")
        trial.suggest_int.assert_called_with("k", 2, 8)

        Distribution(kind="categorical", choices=[4, 5, 6]).sample(trial, "g")
        trial.suggest_categorical.assert_called_with("g", [4, 5, 6])


# ---------------------------------------------------------------------------
# SearchSpace
# ---------------------------------------------------------------------------


class TestSearchSpaceConstruction:
    def test_from_dict_with_dict_specs(self):
        space = SearchSpace.from_dict(
            {
                "optimization.optimizer.learning_rate": {
                    "dist": "loguniform",
                    "low": 1e-5,
                    "high": 1e-3,
                },
                "model.model_kwargs.k": {"dist": "categorical", "choices": [4, 8, 16]},
            }
        )
        assert len(space) == 2
        assert "optimization.optimizer.learning_rate" in space
        assert space.params["optimization.optimizer.learning_rate"].kind == "loguniform"
        assert space.params["model.model_kwargs.k"].choices == [4, 8, 16]

    def test_from_dict_with_tuple_specs(self):
        """Terse tuple form: (kind, low, high) or ('categorical', *choices)."""
        space = SearchSpace.from_dict(
            {
                "lr": ("loguniform", 1e-5, 1e-3),
                "k": ("categorical", 4, 8, 16),
            }
        )
        assert space.params["lr"].kind == "loguniform"
        assert space.params["lr"].low == 1e-5
        assert space.params["k"].choices == [4, 8, 16]

    def test_from_dict_rejects_missing_dist_kind(self):
        with pytest.raises(ValueError, match="requires a 'dist' or 'kind' key"):
            SearchSpace.from_dict({"x": {"low": 0.0, "high": 1.0}})

    def test_from_yaml_roundtrip(self, tmp_path: Path):
        yaml_path = tmp_path / "space.yaml"
        with open(yaml_path, "w") as fh:
            yaml.safe_dump(
                {
                    "optimization.optimizer.learning_rate": {
                        "dist": "loguniform",
                        "low": 1e-5,
                        "high": 1e-3,
                    },
                    "model.model_kwargs.kan_grid_size": {
                        "dist": "categorical",
                        "choices": [4, 5, 6, 8],
                    },
                },
                fh,
            )
        space = SearchSpace.from_yaml(yaml_path)
        assert len(space) == 2
        assert "optimization.optimizer.learning_rate" in space


class TestSearchSpaceOverridesFn:
    """The closure returned by make_overrides_fn must sample every param."""

    def test_overrides_fn_returns_dotted_path_dict(self):
        space = SearchSpace.from_dict(
            {
                "optimization.optimizer.learning_rate": ("loguniform", 1e-5, 1e-3),
                "model.foo": ("categorical", "a", "b", "c"),
            }
        )
        # Mock the trial to return predictable values
        trial = MagicMock()
        trial.suggest_float.return_value = 5e-5
        trial.suggest_categorical.return_value = "b"

        overrides = space.make_overrides_fn()(trial)
        assert overrides == {
            "optimization.optimizer.learning_rate": 5e-5,
            "model.foo": "b",
        }
        # And the sample names match the dotted paths
        trial.suggest_float.assert_called_with(
            "optimization.optimizer.learning_rate", 1e-5, 1e-3, log=True
        )
        trial.suggest_categorical.assert_called_with("model.foo", ["a", "b", "c"])


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class TestPresets:
    def test_list_presets_returns_known_names(self):
        names = list_presets()
        assert "kan_dual_domain" in names
        assert "loss_weights_only" in names
        assert "optimizer_basic" in names
        assert "full_kan_block" in names
        assert "diffusion_curriculum" in names

    def test_load_preset_returns_search_space(self):
        space = load_preset("kan_dual_domain")
        assert isinstance(space, SearchSpace)
        # Plan §4.2 specifies 14 dimensions — the preset should match exactly.
        # If this number drifts, it means someone changed the canonical
        # search space without updating the test.
        assert len(space) >= 12, f"kan_dual_domain shrank to {len(space)} dims"

    def test_load_preset_unknown_name_raises(self):
        with pytest.raises(KeyError, match="Unknown preset"):
            load_preset("not_a_real_preset")

    def test_all_presets_construct_cleanly(self):
        """Every preset listed in PRESETS must build into a valid SearchSpace."""
        for name in PRESETS:
            space = load_preset(name)
            assert len(space) > 0, f"preset {name} is empty"


class TestSearchSpaceMerge:
    """Compositing presets: union of disjoint specs, conflict handling."""

    def test_merge_disjoint_unions_paths(self):
        a = SearchSpace.from_dict({"x": ("uniform", 0.0, 1.0)})
        b = SearchSpace.from_dict({"y": ("loguniform", 1e-5, 1e-3)})
        merged = a.merge(b)
        assert len(merged) == 2
        assert "x" in merged and "y" in merged

    def test_merge_does_not_mutate_inputs(self):
        a = SearchSpace.from_dict({"x": ("uniform", 0.0, 1.0)})
        b = SearchSpace.from_dict({"y": ("uniform", 0.0, 1.0)})
        _ = a.merge(b)
        assert "y" not in a
        assert "x" not in b

    def test_merge_conflict_raises_by_default(self):
        a = SearchSpace.from_dict({"x": ("uniform", 0.0, 1.0)})
        b = SearchSpace.from_dict({"x": ("loguniform", 1e-3, 1e-1)})
        with pytest.raises(ValueError, match="merge conflict on path 'x'"):
            a.merge(b)

    def test_merge_conflict_override_lets_other_win(self):
        a = SearchSpace.from_dict({"x": ("uniform", 0.0, 1.0)})
        b = SearchSpace.from_dict({"x": ("loguniform", 1e-3, 1e-1)})
        merged = a.merge(b, on_conflict="override")
        assert merged.params["x"].kind == "loguniform"

    def test_merge_conflict_keep_keeps_self(self):
        a = SearchSpace.from_dict({"x": ("uniform", 0.0, 1.0)})
        b = SearchSpace.from_dict({"x": ("loguniform", 1e-3, 1e-1)})
        merged = a.merge(b, on_conflict="keep")
        assert merged.params["x"].kind == "uniform"

    def test_merge_invalid_on_conflict_raises(self):
        a = SearchSpace()
        b = SearchSpace()
        with pytest.raises(ValueError, match="on_conflict must be"):
            a.merge(b, on_conflict="surprise")

    def test_load_presets_merges_in_order(self):
        """Composite of optimizer_basic + loss_weights_only is the union."""
        space = load_presets(["optimizer_basic", "loss_weights_only"])
        # optimizer_basic adds LR + WD; loss_weights_only adds 4 loss weights
        assert len(space) == len(load_preset("optimizer_basic")) + len(
            load_preset("loss_weights_only")
        )
        assert "optimization.optimizer.learning_rate" in space
        assert "optimization.optimizer.weight_decay" in space
        # at least one loss weight made it in
        assert any("losses." in p for p in space.params)

    def test_load_presets_empty_list_returns_empty_space(self):
        assert len(load_presets([])) == 0

    def test_load_presets_propagates_conflict(self):
        """If two presets share a key, load_presets raises by default."""
        # full_kan_block and kan_dual_domain both touch
        # model.model_kwargs.kan_dual_domain_kwargs.kan_grid_size
        with pytest.raises(ValueError, match="merge conflict"):
            load_presets(["kan_dual_domain", "full_kan_block"])


# ---------------------------------------------------------------------------
# Path tokenization + dotted-path override
# ---------------------------------------------------------------------------


class TestTokenizePath:
    def test_plain_dotted_path(self):
        assert _tokenize_path("a.b.c") == ["a", "b", "c"]

    def test_with_list_selector(self):
        toks = _tokenize_path("a.b[name=foo].c")
        assert toks[0] == "a"
        assert toks[1] == "b"
        assert isinstance(toks[2], _ListSelector)
        assert toks[2].key == "name"
        assert toks[2].value == "foo"
        assert toks[3] == "c"

    def test_selector_value_type_coercion(self):
        """[k=42] coerces value to int; [k=true] to bool; [k=foo] stays string."""
        toks = _tokenize_path("a[idx=42].b")
        assert toks[1].value == 42
        assert isinstance(toks[1].value, int)
        toks = _tokenize_path("a[flag=true].b")
        assert toks[1].value is True
        toks = _tokenize_path("a[name=foo].b")
        assert toks[1].value == "foo"

    def test_selector_without_equals_rejected(self):
        with pytest.raises(ValueError, match=r"\[foo\] must be of form"):
            _tokenize_path("a[foo].b")


class TestApplyDottedOverride:
    def test_plain_path(self):
        cfg: dict[str, Any] = {"a": {"b": {"c": 1}}}
        apply_dotted_override(cfg, "a.b.c", 42)
        assert cfg["a"]["b"]["c"] == 42

    def test_creates_missing_intermediate_keys(self):
        cfg: dict[str, Any] = {"a": {}}
        apply_dotted_override(cfg, "a.b.c.d", 99)
        assert cfg["a"]["b"]["c"]["d"] == 99

    def test_list_selector_targets_match(self):
        """losses.kspace_losses[name=log_spectral].weight selects by name."""
        cfg = {
            "losses": {
                "kspace_losses": [
                    {"name": "complex_l1", "weight": 1.0},
                    {"name": "log_spectral", "weight": 0.1},
                    {"name": "sobolev_kspace", "weight": 0.05},
                ],
            },
        }
        apply_dotted_override(cfg, "losses.kspace_losses[name=log_spectral].weight", 0.3)
        kspace = cfg["losses"]["kspace_losses"]
        assert kspace[0]["weight"] == 1.0  # unchanged
        assert kspace[1]["weight"] == 0.3  # the one we targeted
        assert kspace[2]["weight"] == 0.05  # unchanged

    def test_list_selector_no_match_raises(self):
        cfg = {"x": [{"name": "a"}]}
        with pytest.raises(ValueError, match="no list item with"):
            apply_dotted_override(cfg, "x[name=missing].y", 1)

    def test_list_selector_multiple_matches_raises(self):
        cfg = {"x": [{"name": "a", "v": 1}, {"name": "a", "v": 2}]}
        with pytest.raises(ValueError, match="multiple list items match"):
            apply_dotted_override(cfg, "x[name=a].v", 99)

    def test_kan_dual_domain_preset_against_real_yaml(self, tmp_path: Path):
        """End-to-end: load the kan_dual_domain preset, mock-sample one trial,
        and apply every sampled override to a real headline YAML — every
        single override must succeed without raising."""
        headline = require_repo_file(
            "experiments/inprogress/kspace_filling/experiment_11_kan_dual_domain.yaml"
        )
        with open(headline) as fh:
            cfg = yaml.safe_load(fh)

        space = load_preset("kan_dual_domain")
        # Mock trial returns the *low* end of every distribution
        trial = MagicMock()
        trial.suggest_float.side_effect = lambda name, low, high, **kw: float(low)
        trial.suggest_int.side_effect = lambda name, low, high, **kw: int(low)
        trial.suggest_categorical.side_effect = lambda name, choices: choices[0]

        overrides = space.make_overrides_fn()(trial)
        # Every dotted path in the preset must apply cleanly to the real YAML
        for path, value in overrides.items():
            apply_dotted_override(cfg, path, value)

        # Spot check: the LR override actually landed -- read back the SAME
        # dotted path that was written. It used to assert against the flat
        # ``optimization.learning_rate`` while the preset overrode
        # ``optimization.optimizer.learning_rate`` (phase 8 decomposed the block
        # and moved the preset, not this read-back), so the spot check was
        # inspecting a key the override never touched.
        lr_path = "optimization.optimizer.learning_rate"
        landed = cfg
        for part in lr_path.split("."):
            landed = landed[part]
        assert landed == overrides[lr_path]

    @pytest.mark.parametrize("preset_name", sorted(PRESETS))
    def test_every_preset_applies_cleanly_to_headline_yaml(self, preset_name: str) -> None:
        """Every built-in preset's list selectors must resolve against a real arm.

        Plain dotted keys auto-create missing dicts, so a ``model_kwargs`` path
        an arm never declares is harmless. Only ``[name=...]`` list selectors
        raise, which makes them the one part of a preset that can silently drift
        away from the config it targets -- and drift they did (#631): two loss
        paths named ``image_losses`` while both losses live in ``kspace_losses``,
        so any HPO run using ``kan_dual_domain`` or ``loss_weights_only`` died on
        trial 1 with "no list item with name=...".
        """
        headline = require_repo_file(
            "experiments/inprogress/kspace_filling/experiment_11_kan_dual_domain.yaml"
        )
        with open(headline) as fh:
            cfg = yaml.safe_load(fh)

        trial = MagicMock()
        trial.suggest_float.side_effect = lambda n, low, high, **kw: float(low)
        trial.suggest_int.side_effect = lambda n, low, high, **kw: int(low)
        trial.suggest_categorical.side_effect = lambda n, choices: choices[0]

        overrides = load_preset(preset_name).make_overrides_fn()(trial)
        selectors = 0
        for path, value in overrides.items():
            apply_dotted_override(cfg, path, value)
            if "[" in path:
                selectors += 1

        # Guard the guard: if a refactor drops the selector paths from the two
        # loss presets, this invariant would still pass while testing nothing.
        if preset_name in ("kan_dual_domain", "loss_weights_only"):
            assert selectors == 4, (
                f"{preset_name} should carry 4 list-selector loss paths; found {selectors}"
            )


# ---------------------------------------------------------------------------
# Schema enumeration
# ---------------------------------------------------------------------------


class TestEnumerateSchemaPaths:
    def test_enumerate_returns_known_paths(self):
        """The enumerator returns nested-dotted-path strings for the expected
        common fields. Slightly loose because the schema may grow."""
        paths = enumerate_schema_paths()
        assert "optimization.optimizer.learning_rate" in paths
        # At least some training and data fields are exposed
        assert any(p.startswith("training.") for p in paths)
        assert any(p.startswith("data.") for p in paths)
        # Free-form dict fields are reported as `*` leaves
        assert any(p.endswith(".*") for p in paths)

    def test_enumerate_respects_max_depth(self):
        paths_deep = enumerate_schema_paths(max_depth=6)
        paths_shallow = enumerate_schema_paths(max_depth=1)
        assert len(paths_deep) > len(paths_shallow)

    def test_template_yaml_is_parseable(self):
        text = make_search_space_template()
        parsed = yaml.safe_load(text)
        assert isinstance(parsed, dict)
        assert "optimization.optimizer.learning_rate" in parsed
        # Roundtrip: the template should be loadable as a SearchSpace
        space = SearchSpace.from_dict(parsed)
        assert "optimization.optimizer.learning_rate" in space
