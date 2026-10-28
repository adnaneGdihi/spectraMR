"""Unit tests for CollationStrategySelector auto-selection (WS4).

The E1/E2/E5 + Phase-4 dataset families (bart_kspace, bids_paired, cine,
quantitative, ...) used to be absent from ``AUTO_SELECT_MAP`` and so silently
fell through to the ``robust`` fallback (pitfall #9). They now map explicitly to
``image`` — the strategy their input/target tio.Subject contract requires —
and are registered as compatible so an explicit strategy no longer warns.
"""

from types import SimpleNamespace

import pytest

from spectramr.data.collation.strategy_selector import CollationStrategySelector

_NEW_TYPES = [
    "bart_kspace",
    "ismrmrd_kspace",
    "bids_paired",
    "png_paired",
    "field_ref",
    "oracle_bssfp",
    "mrixfields",
    "quantitative",
    "cine",
    "pde_synthetic",
]


def _auto_config():
    # strategy=None → auto-select; the rest are the image-strategy kwargs
    # select_strategy reads when populating the collation kwargs.
    return SimpleNamespace(
        strategy=None,
        log_strategy_selection=False,
        padding_mode="constant",
        padding_value=0.0,
        allow_variable_shapes=False,
        squeeze_depth_dim=None,
        validate_nans=True,
    )


class TestAutoSelectCoverage:
    def test_new_families_are_mapped_not_defaulted(self):
        for dtype in _NEW_TYPES:
            assert CollationStrategySelector.AUTO_SELECT_MAP.get(dtype) == "image", (
                f"{dtype} must map explicitly, not fall through to robust"
            )

    def test_auto_select_returns_image_for_new_types(self):
        for dtype in _NEW_TYPES:
            strategy, _ = CollationStrategySelector.select_strategy(
                _auto_config(), dataset_type=dtype
            )
            assert strategy == "image", f"{dtype} auto-selected {strategy!r}"

    def test_new_families_are_image_compatible(self):
        compat = CollationStrategySelector.COMPATIBILITY_MAP["image"]
        for dtype in _NEW_TYPES:
            assert dtype in compat

    def test_unknown_type_still_falls_back_to_robust(self):
        # Genuinely-unknown types keep the documented safety fallback.
        strategy, _ = CollationStrategySelector.select_strategy(
            _auto_config(), dataset_type="totally_unregistered_xyz"
        )
        assert strategy == "robust"


class TestTheTwoStrategyVocabulariesAgree:
    """A8. `COMPATIBILITY_MAP` and `CollateStrategyFactory.STRATEGIES` are two
    tables for one concept, and the permissive one is consulted FIRST — so a
    name present in the map and absent from the factory passes validation and
    then dies at construction. That is how `"universal"` survived: it passed
    the Literal, passed the map, then raised `Unknown data type: universal`.
    """

    def test_every_advertised_strategy_is_constructible(self) -> None:
        from spectramr.data.collation.strategies import CollateStrategyFactory
        from spectramr.data.collation.strategy_selector import (
            CollationStrategySelector,
        )

        advertised = set(CollationStrategySelector.COMPATIBILITY_MAP)
        buildable = set(CollateStrategyFactory.STRATEGIES)
        assert advertised - buildable == set(), (
            "advertised but not constructible — passes validation, fails at "
            "construction"
        )

    def test_universal_is_gone(self) -> None:
        from spectramr.data.collation.strategy_selector import (
            CollationStrategySelector,
        )

        assert "universal" not in CollationStrategySelector.COMPATIBILITY_MAP

    def test_the_guard_fires_on_a_reintroduction(self, monkeypatch) -> None:
        """A check that cannot fail is not a check."""
        from spectramr.data.collation.strategy_selector import (
            CollationStrategySelector,
        )

        monkeypatch.setitem(
            CollationStrategySelector.COMPATIBILITY_MAP, "phantom", {"*"}
        )
        with pytest.raises(ValueError, match="not constructible"):
            CollationStrategySelector._assert_vocabularies_agree()

    def test_the_guard_runs_on_the_live_path(self, monkeypatch) -> None:
        """Wired into `select_strategy`, not only asserted under test — the
        whole failure mode was a check that ran too late."""
        from spectramr.config.schemas.data import CollationConfigSchema
        from spectramr.data.collation.strategy_selector import (
            CollationStrategySelector,
        )

        monkeypatch.setitem(
            CollationStrategySelector.COMPATIBILITY_MAP, "phantom", {"*"}
        )
        with pytest.raises(ValueError, match="not constructible"):
            CollationStrategySelector.select_strategy(
                config=CollationConfigSchema(strategy="image"),
                dataset_type="nifti",
                enable_slab_mode=False,
                patch_size=(64, 64, 1),
            )


class TestIncompatiblePairsRaise:
    """A13. This is the one place the subsystem KNOWS the pair is wrong.

    It used to log "Proceeding anyway (user-specified choice)" — pitfall #10.
    The collation strategy decides how samples are stacked, and stacking them
    wrongly does not fail loudly downstream: it yields a batch of the wrong
    shape or contents, surfacing as a model error far from the cause.

    Measured before flipping: 105 arms declare `data.collation.strategy`, all
    of them `image`, and ZERO form an incompatible pair. This raises for nobody
    today.
    """

    @staticmethod
    def _select(strategy, dataset_type):
        from spectramr.config.schemas.data import CollationConfigSchema
        from spectramr.data.collation.strategy_selector import (
            CollationStrategySelector,
        )

        return CollationStrategySelector.select_strategy(
            config=CollationConfigSchema(strategy=strategy),
            dataset_type=dataset_type,
            enable_slab_mode=False,
            patch_size=(64, 64, 1),
        )

    def test_an_incompatible_pair_raises(self) -> None:
        with pytest.raises(ValueError, match="not compatible with dataset_type"):
            self._select("slab", "graph")

    def test_the_message_names_the_compatible_set(self) -> None:
        with pytest.raises(ValueError) as exc:
            self._select("physics", "nifti")
        assert "'kspace'" in str(exc.value)

    def test_it_offers_robust_as_the_deliberate_escape(self) -> None:
        """A pairing that is intentional rather than mistaken needs a way out
        that does not require lying about the dataset_type."""
        with pytest.raises(ValueError, match="robust"):
            self._select("slab", "graph")
        assert self._select("robust", "graph")[0] == "robust"

    def test_the_corpus_pairing_still_works(self) -> None:
        """All 105 arms that declare a strategy declare `image`."""
        for dataset_type in ("nifti", "kspace", "m4raw", "mrixfields"):
            assert self._select("image", dataset_type)[0] == "image"


class TestStrategyKwargsTable:
    """``_build_strategy_kwargs`` is a table of per-strategy builders, not an if-chain."""

    @staticmethod
    def _config(**over):
        base = {
            "padding_mode": "zeros",
            "padding_value": 0.0,
            "allow_variable_shapes": False,
            "squeeze_depth_dim": None,
            "validate_nans": True,
            "slab_target_mode": "middle",
        }
        return SimpleNamespace(**{**base, **over})

    def test_image_kwargs_keep_their_keys_and_order(self) -> None:
        kwargs = CollationStrategySelector._build_strategy_kwargs(
            "image", self._config(), False, (256, 256, 1)
        )
        assert list(kwargs) == [
            "padding_mode",
            "padding_value",
            "allow_variable_shapes",
            "squeeze_depth_dim",
            "validate_nans",
        ]
        assert kwargs["squeeze_depth_dim"] is True  # derived from patch_size[-1] == 1

    def test_slab_kwargs_add_target_mode_only_in_slab_mode(self) -> None:
        plain = CollationStrategySelector._build_strategy_kwargs(
            "slab", self._config(), False, None
        )
        slab = CollationStrategySelector._build_strategy_kwargs("slab", self._config(), True, None)
        assert "target_mode" not in plain and "flat_input" not in plain
        assert (slab["target_mode"], slab["flat_input"]) == ("middle", True)

    def test_slab_still_refuses_a_config_that_asks_for_the_opposite(self) -> None:
        with pytest.raises(ValueError, match="allow_variable_shapes"):
            CollationStrategySelector._build_strategy_kwargs(
                "slab", self._config(allow_variable_shapes=True), False, None
            )
        with pytest.raises(ValueError, match="squeeze_depth_dim"):
            CollationStrategySelector._build_strategy_kwargs(
                "slab", self._config(squeeze_depth_dim=True), False, None
            )

    def test_an_unlisted_strategy_takes_no_kwargs(self) -> None:
        assert CollationStrategySelector._build_strategy_kwargs(
            "anything_else", self._config(), True, None
        ) == {}
