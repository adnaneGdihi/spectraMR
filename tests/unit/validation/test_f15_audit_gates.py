"""Regression tests for F15 (smoke_audit_20260521 round 9) — Tier-1
audit gates that convert F11-class runtime crashes into audit-time
failures.

Round 6 (F11) fixed 5 individual YAMLs that crashed at runtime with:
- ``[PINN Strategy] config.losses.pinn is None``
- ``step_configs[1] (name='discriminator') has optimizer=None``
- ``Missing required config: training.diffusion.num_timesteps``

The YAMLs themselves are patched, but the underlying CLASS of bug
can recur on any new experiment. F15 adds two Tier-1 audit checks
that catch the bug at submission time (the smoke wrapper's
``audit --strict`` gate runs BEFORE training starts):

1. **PINN paradigm requires ``losses.pinn``** — added to
   ``_PARADIGM_REQUIREMENTS`` so the existing
   ``check_paradigm_required_fields`` walks it. Fires when the
   strategy_class or training_mode contains "pinn" but
   ``losses.pinn`` is None.

2. **Discriminator declared → ``losses.gan`` required** — new check
   ``check_discriminator_requires_gan_losses``. Fires when
   ``model.discriminator_component`` (v6.0) or ``model.discriminator``
   (legacy) is set but ``losses.gan.enable_adversarial`` is falsy.

3. **Diffusion paradigm requires ``training.diffusion``** — already
   in ``_PARADIGM_REQUIREMENTS`` from a prior round; pinned here for
   completeness.

These tests build minimal mock TrainingSettings objects and verify
each rule fires (or doesn't fire) appropriately.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _mock_settings(**overrides) -> SimpleNamespace:
    """Minimal TrainingSettings mock with just enough to drive the checks."""
    training = SimpleNamespace(
        strategy_class=overrides.pop("strategy_class", ""),
        training_mode=overrides.pop("training_mode", ""),
        diffusion=overrides.pop("diffusion", None),
        vae=overrides.pop("vae", None),
    )
    model = SimpleNamespace(
        discriminator_component=overrides.pop("discriminator_component", None),
        discriminator=overrides.pop("discriminator", None),
        model_kwargs=overrides.pop("model_kwargs", {}),
    )
    losses = SimpleNamespace(
        pinn=overrides.pop("losses_pinn", None),
        gan=overrides.pop("losses_gan", None),
    )
    return SimpleNamespace(
        training=training,
        model=model,
        losses=losses,
        **overrides,
    )


# ── PINN paradigm requirement ───────────────────────────────────────────────


def test_paradigm_required_fires_for_pinn_without_losses_block() -> None:
    """When the strategy looks like PINN but ``losses.pinn`` is None,
    ``check_paradigm_required_fields`` must emit a failing result."""
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    checker = ConfigHealthChecker()
    cfg = _mock_settings(
        strategy_class="spectramr.infrastructure.training.strategies.pinn_strategy.PINNStrategy",
        training_mode="pinn",
        losses_pinn=None,
    )
    results = checker.check_paradigm_required_fields(cfg)
    failing = [r for r in results if not r.passed]
    assert any("pinn" in r.message.lower() and "losses.pinn" in str(r.yaml_keys).lower()
               for r in failing), (
        "F15: PINN paradigm without losses.pinn must fire a "
        "check_paradigm_required_fields failure. "
        "See TODO/audit/smoke_audit_20260521.md §F15."
    )


def test_paradigm_required_passes_when_pinn_block_present() -> None:
    """Positive case: PINN strategy WITH losses.pinn block passes."""
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    checker = ConfigHealthChecker()
    cfg = _mock_settings(
        strategy_class="spectramr.infrastructure.training.strategies.pinn_strategy.PINNStrategy",
        training_mode="pinn",
        losses_pinn=SimpleNamespace(lambda_pde=0.1),  # any non-None
    )
    results = checker.check_paradigm_required_fields(cfg)
    # Should not have a failing PINN-related result.
    failing_pinn = [
        r for r in results
        if not r.passed and "losses.pinn" in str(r.yaml_keys).lower()
    ]
    assert not failing_pinn, (
        "When losses.pinn is set, the paradigm-required check must pass."
    )


# ── Discriminator → losses.gan requirement ──────────────────────────────────


def test_discriminator_without_gan_losses_fires() -> None:
    """When ``model.discriminator_component`` is set but
    ``losses.gan.enable_adversarial`` is missing/false, the new
    check must fail."""
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    checker = ConfigHealthChecker()
    cfg = _mock_settings(
        discriminator_component=SimpleNamespace(name="patch_gan_discriminator"),
        losses_gan=None,
    )
    result = checker.check_discriminator_requires_gan_losses(cfg)
    assert not result.passed, (
        "Discriminator declared without losses.gan must fail the audit gate. "
        "See §F15."
    )
    assert "optimizer=None" in result.message or "discriminator" in result.message.lower()


def test_discriminator_with_gan_losses_passes() -> None:
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    checker = ConfigHealthChecker()
    cfg = _mock_settings(
        discriminator_component=SimpleNamespace(name="patch_gan_discriminator"),
        losses_gan=SimpleNamespace(enable_adversarial=True),
    )
    result = checker.check_discriminator_requires_gan_losses(cfg)
    assert result.passed


def test_no_discriminator_returns_pass() -> None:
    """When no discriminator is declared, the check is a no-op pass."""
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    checker = ConfigHealthChecker()
    cfg = _mock_settings()
    result = checker.check_discriminator_requires_gan_losses(cfg)
    assert result.passed


def test_legacy_model_discriminator_layout_also_triggers() -> None:
    """The legacy ``model.discriminator`` (without _component suffix)
    must also be detected. Pinned so future schema migrations don't
    silently break the legacy path."""
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    checker = ConfigHealthChecker()
    cfg = _mock_settings(
        discriminator=SimpleNamespace(some_field=True),  # legacy block
        losses_gan=None,
    )
    result = checker.check_discriminator_requires_gan_losses(cfg)
    assert not result.passed, (
        "Legacy model.discriminator block must also trigger the audit gate."
    )


def test_adversarial_loss_without_a_discriminator_fires() -> None:
    """The converse: ``enable_adversarial`` with nothing to build a critic from.

    Under ``training_mode: gan`` the optimizer builder raises at build time; under the
    disentangled strategy the adversarial step was silently skipped (sabine job 17885803).
    """
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    for mode in ("gan", ""):
        cfg = _mock_settings(
            training_mode=mode, losses_gan=SimpleNamespace(enable_adversarial=True)
        )
        result = ConfigHealthChecker().check_discriminator_requires_gan_losses(cfg)
        assert not result.passed and result.severity == "error"
        assert "model.discriminator_component" in result.message


def test_a_discriminator_under_model_kwargs_is_named_as_the_cause() -> None:
    """The shape the three disentangled arms had: the block one level too deep, never read."""
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    cfg = _mock_settings(
        losses_gan=SimpleNamespace(enable_adversarial=True),
        model_kwargs={"discriminator_component": {"name": "patchgan"}},
    )
    result = ConfigHealthChecker().check_discriminator_requires_gan_losses(cfg)
    assert not result.passed
    assert "model.model_kwargs.discriminator_component" in result.message


def test_adversarial_off_without_a_discriminator_passes() -> None:
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    cfg = _mock_settings(losses_gan=SimpleNamespace(enable_adversarial=False))
    assert ConfigHealthChecker().check_discriminator_requires_gan_losses(cfg).passed


# ── Diffusion paradigm requirement (already exists, pinned for completeness) ─


def test_diffusion_without_training_diffusion_block_fires() -> None:
    """Sanity-check the pre-existing diffusion paradigm requirement."""
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    checker = ConfigHealthChecker()
    cfg = _mock_settings(
        strategy_class="spectramr.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy",
        training_mode="diffusion",
        diffusion=None,
    )
    results = checker.check_paradigm_required_fields(cfg)
    failing = [r for r in results if not r.passed]
    assert any("training.diffusion" in str(r.yaml_keys).lower() for r in failing), (
        "Diffusion paradigm without training.diffusion must fire "
        "(pre-existing rule; pinned for regression safety)."
    )
