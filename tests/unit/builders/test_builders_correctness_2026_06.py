"""Regression: builder correctness fixes (2026-06-11 infrastructure audit).

* ``InfrastructureBuilder.build_metrics`` silently skipped an explicitly-enabled
  metric whose name wasn't registered (DEBUG log + continue) — pitfall #18 (a
  best/early-stopping metric that is never computed). Now raises.
* ``ModelBuilder`` read ``config.training.gan.discriminator`` directly, raising
  AttributeError whenever ``training.gan`` was set (the schema has no such
  field). Now uses getattr → safe no-op.
* ``LossBuilder`` built ``_fallback_registry_map`` but never consulted it, so a
  migrated config using a legacy alias (``edge``/``l2``/``ffl``/``dc``) whose
  registry twin is in the declarative lists was falsely rejected as unmigrated.
"""

from __future__ import annotations

import inspect

import pytest

from spectramr.config.settings import TrainingSettings
from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.training.builders import infrastructure_builder as ib_mod
from spectramr.infrastructure.training.builders import loss_builder as lb_mod
from spectramr.infrastructure.training.builders import model_builder as mb_mod
from tests.utils.repo_scripts import require_repo_file


def _config():
    return TrainingSettings.from_yaml(
        str(require_repo_file("experiments/inprogress/dummy/dummy_gan.yaml"))
    )


def test_enabled_but_unregistered_metric_raises(monkeypatch):
    cfg = _config()
    builder = ib_mod.InfrastructureBuilder(cfg, device="cpu")
    # Force every registry membership check to fail; compute_psnr defaults True.
    monkeypatch.setattr(
        ib_mod.MetricsRegistry, "is_registered", staticmethod(lambda name: False)
    )
    with pytest.raises(ConfigurationError, match="not a registered"):
        builder.build_metrics()


def test_model_builder_gan_discriminator_branch_uses_getattr():
    # The schema (GANSubConfigSchema) has no ``discriminator`` field, so direct
    # attribute access crashed. The fix must read it defensively.
    src = inspect.getsource(mb_mod)
    assert 'getattr(self._config.training.gan, "discriminator", None)' in src
    assert "and self._config.training.gan.discriminator\n" not in src


def test_loss_builder_consults_fallback_registry_map():
    src = inspect.getsource(lb_mod.LossBuilder)
    # The previously-dead map must now translate before the membership check.
    assert "_fallback_registry_map.get(loss_name" in src
    assert "effective in self._losses" in src
