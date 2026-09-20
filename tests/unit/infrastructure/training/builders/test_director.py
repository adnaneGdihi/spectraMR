"""Unit tests for TrainingEnvironmentDirector.

Focus: the WS-D calling-convention migration. ``__init__`` was moved onto the
canonical :class:`BuilderContext` shape behind the ``accepts_builder_context``
back-compat shim, so both the legacy ``Director(config)`` call and the canonical
``Director(BuilderContext(config=config))`` call must construct an equivalent
object (same stored ``._config``).
"""

from unittest.mock import MagicMock, patch

import torch

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.builders.context import BuilderContext
from spectramr.infrastructure.training.builders.director import (
    TrainingEnvironmentDirector,
)


def _make_config() -> MagicMock:
    """Minimal stub TrainingSettings (device wiring is patched out)."""
    config = MagicMock(spec=TrainingSettings)
    config.device = "cpu"
    return config


class TestDirectorBuilderContextMigration:
    """Both construction shapes are accepted and produce equivalent state."""

    @patch.object(
        TrainingEnvironmentDirector,
        "_resolve_device",
        return_value=torch.device("cpu"),
    )
    def test_legacy_config_shape(self, _mock_resolve) -> None:
        """Legacy ``Director(config)`` still works via the compat shim."""
        config = _make_config()

        director = TrainingEnvironmentDirector(config)  # type: ignore[arg-type]

        assert director._config is config
        assert director._device == torch.device("cpu")

    @patch.object(
        TrainingEnvironmentDirector,
        "_resolve_device",
        return_value=torch.device("cpu"),
    )
    def test_builder_context_shape(self, _mock_resolve) -> None:
        """Canonical ``Director(BuilderContext(config=config))`` works."""
        config = _make_config()

        director = TrainingEnvironmentDirector(BuilderContext(config=config))

        assert director._config is config
        assert director._device == torch.device("cpu")

    @patch.object(
        TrainingEnvironmentDirector,
        "_resolve_device",
        return_value=torch.device("cpu"),
    )
    def test_both_shapes_equivalent(self, _mock_resolve) -> None:
        """Legacy and context construction store the same config + device."""
        config = _make_config()

        legacy = TrainingEnvironmentDirector(config)  # type: ignore[arg-type]
        canonical = TrainingEnvironmentDirector(BuilderContext(config=config))

        assert legacy._config is canonical._config is config
        assert legacy._device == canonical._device

    def test_init_marked_as_accepting_builder_context(self) -> None:
        """The shim records its marker so the fitness check sees the migration."""
        assert (
            getattr(
                TrainingEnvironmentDirector.__init__,
                "__accepts_builder_context__",
                False,
            )
            is True
        )


class TestCompileIsPlacedNotFixed:
    """The director must apply compilation at the stage the table names.

    Structural rather than behavioural: driving `build_environment` needs a real
    config, models, data and a device. The risk being guarded is a wiring
    regression -- an application point dropped, or the fixed step-1 position
    coming back -- and that is visible in the source.
    """

    @staticmethod
    def _director_source() -> str:
        import inspect

        from spectramr.infrastructure.training.builders import director as mod

        return inspect.getsource(mod)

    @staticmethod
    def _stages_the_director_applies() -> set[str]:
        """Stage names the director compares `placement.stage` against.

        Read off the AST rather than by substring: a substring test passes on a
        mention in a comment, and the comments here name the stages to explain
        them. Only a real `==` comparison is an application point.
        """
        import ast

        tree = ast.parse(TestCompileIsPlacedNotFixed._director_source())
        found: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare) or not isinstance(node.left, ast.Attribute):
                continue
            if node.left.attr != "stage":
                continue
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    found.add(comparator.value)
        return found

    @staticmethod
    def _stages_the_table_returns() -> set[str]:
        """Every stage any row of `_PLACEMENTS` can actually produce."""
        from spectramr.infrastructure.training.builders.compile_placement import (
            _PLACEMENTS,
            resolve_compile_placement,
        )

        stages: set[str] = set()
        for strategy in _PLACEMENTS:
            # zero_stage spans the DeepSpeed rows, which branch on it.
            for zero_stage in (None, 0, 1, 2, 3):
                placement = resolve_compile_placement(strategy, zero_stage=zero_stage)
                if placement.stage is not None:
                    stages.add(placement.stage)
        return stages

    def test_the_table_and_the_director_agree_in_both_directions(self) -> None:
        """Planted, both ways round.

        A stage the table returns and the director never applies is an arm
        declared compiled that silently runs eager. A stage the director
        applies and no row returns is dead code -- which `after_models` was:
        it survived the plan's correction to `after_stage_b` and no row could
        reach it.
        """
        table = self._stages_the_table_returns()
        director = self._stages_the_director_applies()
        assert table == director, (
            f"unapplied stages {sorted(table - director)}, "
            f"unreachable branches {sorted(director - table)}"
        )

    def test_every_declared_stage_is_reachable(self) -> None:
        """The `CompileStage` Literal is the published vocabulary; a member no
        row returns advertises a placement that does not exist."""
        import typing

        from spectramr.infrastructure.training.builders.compile_placement import CompileStage

        assert set(typing.get_args(CompileStage)) == self._stages_the_table_returns()

    def test_the_model_builder_chain_no_longer_compiles(self) -> None:
        """The fixed step-1 position is what this work removed.

        Matches the chained call specifically -- the surrounding comment names
        `.compile()` to explain its absence, and a bare substring test would
        read that as the thing it forbids.
        """
        chained = [
            line
            for line in self._director_source().splitlines()
            if line.strip() == ".compile()"
        ]
        assert chained == []

    def test_a_refusal_raises_before_anything_is_built(self) -> None:
        """A refused combination must not cost a GPU allocation to discover, so
        the check precedes model construction."""
        source = self._director_source()
        refusal = source.index("placement.refused")
        models = source.index("ModelBuilder(")
        assert refusal < models

    def test_the_applied_record_reaches_the_environment(self) -> None:
        """Otherwise the run cannot say afterwards WHERE it compiled, which is
        the distinction the whole placement change turns on.

        AST again: the previous substring check would have passed on the word
        appearing in a comment.
        """
        import ast

        tree = ast.parse(self._director_source())
        passed = {
            keyword.arg
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg is not None
        }
        assert "compile" in passed, "the CompileRecord never reaches TrainingEnvironment"

    def test_the_environment_carries_a_compile_field(self) -> None:
        """A record the dataclass cannot hold is a record that is dropped."""
        import dataclasses

        from spectramr.infrastructure.training.builders.environment import (
            TrainingEnvironment,
        )

        assert "compile" in {f.name for f in dataclasses.fields(TrainingEnvironment)}
