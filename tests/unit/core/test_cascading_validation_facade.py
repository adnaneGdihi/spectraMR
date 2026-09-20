"""Pins the cascading_validation facade from the Wave 0 exit-criterion split (#1400).

The module was 464 LOC against the 300 ceiling (NN20) and held three
independent concerns: which severity levels an arm evaluates, whether the
timestep<->acceleration round trip holds, and how a result row is assembled.
The first two moved to sibling modules. 24 importers resolve them through this
path, so the facade must expose the definitions themselves, not copies.
"""

from __future__ import annotations

import pytest

from spectramr.core import cascade_levels, cascade_round_trip
from spectramr.core import cascading_validation as cv


class TestFacadeIdentity:
    @pytest.mark.parametrize(
        "name",
        [
            "CASCADE_ID_COLUMNS",
            "CASCADING_LEVELS",
            "UNRECORDED_SKIP_REASON",
            "normalize_cascade_levels",
            "reconcile_skipped_levels",
            "resolve_cascade_levels",
        ],
    )
    def test_level_symbols_are_the_definitions(self, name: str) -> None:
        assert getattr(cv, name) is getattr(cascade_levels, name)

    @pytest.mark.parametrize(
        "name",
        [
            "IDENTITY_ACCELERATION",
            "RoundTrip",
            "check_round_trip",
            "legacy_linear_timestep",
            "training_band",
        ],
    )
    def test_round_trip_symbols_are_the_definitions(self, name: str) -> None:
        assert getattr(cv, name) is getattr(cascade_round_trip, name)

    def test_row_assembly_still_lives_in_the_facade(self) -> None:
        """The third concern was not moved; it must not have become a re-export."""
        assert cv.build_cascade_row.__module__ == cv.__name__
        assert cv.aggregate_cascade_rows.__module__ == cv.__name__
        assert cv.flatten_band_records.__module__ == cv.__name__

    def test_all_covers_every_public_name_the_split_touched(self) -> None:
        assert set(cv.__all__) == set(cascade_levels.__all__) | set(
            cascade_round_trip.__all__
        ) | {"build_cascade_row", "aggregate_cascade_rows", "flatten_band_records"}


class TestSiblingsAreIndependent:
    """The three blocks referenced nothing of each other's -- that is why the
    split is a DAG with no shared helper module. Pinned by AST over the real
    import statements, not by scanning source text: both docstrings *name* the
    other module in prose, so a text match reports a dependency that is not one.
    """

    @staticmethod
    def _imported_modules(module: object) -> set[str]:
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(module))
        out: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                out.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                out.add(node.module)
        return out

    def test_levels_does_not_import_round_trip(self) -> None:
        assert not any(
            "cascade_round_trip" in m for m in self._imported_modules(cascade_levels)
        )

    def test_round_trip_does_not_import_levels(self) -> None:
        assert not any(
            "cascade_levels" in m for m in self._imported_modules(cascade_round_trip)
        )

    def test_the_import_scan_can_fire(self) -> None:
        """Anti-vacuity: the facade DOES import both, so the scan must see them."""
        found = self._imported_modules(cv)
        assert any("cascade_levels" in m for m in found)
        assert any("cascade_round_trip" in m for m in found)
