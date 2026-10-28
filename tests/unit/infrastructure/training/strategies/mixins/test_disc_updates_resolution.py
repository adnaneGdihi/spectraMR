"""Regression tests for the disc_updates resolution helper.

Pin the post-audit-05-F10 fix: discriminator-updates-per-G-step lives at
``config.losses.gan.disc_updates`` (v6.0 SSOT). Reading
``config.disc_updates`` is the legacy flat pattern (CLAUDE.md pitfall
#1) and silently falls back to 1 because ``extra="forbid"`` rejects the
unknown top-level field.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectramr.infrastructure.training.strategies.mixins.adversarial import (
    _resolve_disc_updates,
)


def _config_with_disc_updates(n: int, interval: int = 1) -> SimpleNamespace:
    gan = SimpleNamespace(disc_updates=n, disc_update_interval=interval)
    return SimpleNamespace(losses=SimpleNamespace(gan=gan))


def test_resolve_disc_updates_reads_from_losses_gan() -> None:
    assert _resolve_disc_updates(_config_with_disc_updates(3), iteration=0) == 3


def test_resolve_disc_updates_returns_int() -> None:
    out = _resolve_disc_updates(_config_with_disc_updates(5), iteration=0)
    assert isinstance(out, int)
    assert out == 5


def test_resolve_disc_updates_raises_when_gan_block_missing() -> None:
    cfg = SimpleNamespace(losses=SimpleNamespace(gan=None))
    with pytest.raises(ValueError, match="config.losses.gan"):
        _resolve_disc_updates(cfg, iteration=0)


def test_resolve_disc_updates_raises_when_losses_missing() -> None:
    cfg = SimpleNamespace()
    with pytest.raises(ValueError, match="config.losses.gan"):
        _resolve_disc_updates(cfg, iteration=0)


def test_resolve_disc_updates_does_not_silently_fall_back_to_1() -> None:
    """CLAUDE.md pitfall #1 + #9: no flat-config fallback, no silent default."""
    cfg = SimpleNamespace(disc_updates=7, losses=SimpleNamespace(gan=None))
    with pytest.raises(ValueError):
        _resolve_disc_updates(cfg, iteration=0)


def test_the_critic_updates_only_on_its_interval() -> None:
    # losses.gan.disc_update_interval: N critic updates every k-th iteration, none between.
    cfg = _config_with_disc_updates(2, interval=20)
    assert [_resolve_disc_updates(cfg, iteration=i) for i in (1000, 1001, 1019, 1020)] == [2, 0, 0, 2]


def test_interval_one_updates_every_iteration() -> None:
    cfg = _config_with_disc_updates(1)
    assert {_resolve_disc_updates(cfg, iteration=i) for i in range(5)} == {1}
