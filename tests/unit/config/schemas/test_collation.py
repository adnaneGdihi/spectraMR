"""Tests for ``CollationConfigSchema``.

Targets ``spectramr.config.schemas.collation``. Pydantic schema for batch
collation strategy selection. Has custom validators for:

- ``strategy`` ∈ documented set (or None)
- ``padding_mode`` ∈ {zeros, reflect, replicate}
- ``padding_value`` is finite (no NaN/Inf)

Also enforces ``frozen=True`` and ``extra='forbid'``.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.collation import CollationConfigSchema


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_construction() -> None:
    """All defaults applied; ``strategy=None`` triggers auto-selection."""
    cfg = CollationConfigSchema()
    assert cfg.strategy is None
    assert cfg.padding_mode == "zeros"
    assert cfg.padding_value == 0.0
    assert cfg.allow_variable_shapes is False
    assert cfg.squeeze_depth_dim is None
    assert cfg.validate_nans is True
    assert cfg.log_strategy_selection is True


# ---------------------------------------------------------------------------
# Strategy validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    ["image", "slab", "graph", "sequence", "mixed", "robust", "physics"],
)
def test_documented_strategies_accepted(strategy: str) -> None:
    """Each documented strategy literal is accepted."""
    cfg = CollationConfigSchema(strategy=strategy)
    assert cfg.strategy == strategy


def test_retired_universal_strategy_rejected() -> None:
    """``universal`` was removed 2026-08-09: CollateStrategyFactory has no such strategy."""
    with pytest.raises(ValidationError):
        CollationConfigSchema(strategy="universal")


def test_none_strategy_accepted() -> None:
    """``strategy=None`` (auto-selection) accepted."""
    cfg = CollationConfigSchema(strategy=None)
    assert cfg.strategy is None


def test_invalid_strategy_raises() -> None:
    """Unknown strategy → ``ValidationError``."""
    with pytest.raises(ValidationError):
        CollationConfigSchema(strategy="not_a_strategy")


# ---------------------------------------------------------------------------
# Padding mode validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate"])
def test_padding_modes_accepted(mode: str) -> None:
    """Each documented padding mode is accepted."""
    cfg = CollationConfigSchema(padding_mode=mode)
    assert cfg.padding_mode == mode


def test_invalid_padding_mode_raises() -> None:
    """Unknown padding mode → ``ValidationError``."""
    with pytest.raises(ValidationError):
        CollationConfigSchema(padding_mode="circular")


# ---------------------------------------------------------------------------
# Padding value validator
# ---------------------------------------------------------------------------


def test_padding_value_finite_accepted() -> None:
    """Finite values accepted."""
    cfg = CollationConfigSchema(padding_value=-1.5)
    assert cfg.padding_value == -1.5


def test_padding_value_nan_rejected() -> None:
    """NaN padding value raises."""
    with pytest.raises(ValidationError, match="must be finite"):
        CollationConfigSchema(padding_value=float("nan"))


def test_padding_value_inf_rejected() -> None:
    """Inf padding value raises."""
    with pytest.raises(ValidationError, match="must be finite"):
        CollationConfigSchema(padding_value=math.inf)


# ---------------------------------------------------------------------------
# Frozen + extra='forbid'
# ---------------------------------------------------------------------------


def test_schema_is_frozen() -> None:
    """Cannot mutate after construction."""
    cfg = CollationConfigSchema()
    with pytest.raises(ValidationError):
        cfg.strategy = "image"


def test_extra_fields_rejected() -> None:
    """Unknown fields → ``ValidationError`` (``extra='forbid'``)."""
    with pytest.raises(ValidationError):
        CollationConfigSchema(unknown_field=1)


def test_squeeze_depth_dim_accepts_three_states() -> None:
    """``squeeze_depth_dim`` accepts True, False, None."""
    for v in (True, False, None):
        cfg = CollationConfigSchema(squeeze_depth_dim=v)
        assert cfg.squeeze_depth_dim is v
