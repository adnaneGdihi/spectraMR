"""Planted violations for ``LossBuilder``'s ``unmigrated`` guard (non-negotiable 15).

The guard's own message says it refuses to train an arm whose declared supervision
would be silently skipped (CLAUDE.md #9/#10). It could not do that, because its
only input was ``losses.get_enabled_losses()`` — which keeps a ``lambda_<name>``
only when ``enable_<name>`` is true, and every ``enable_*`` defaults ``False``.
A weight declared *only* as a lambda therefore resolved in
``build_loss_weight_table`` and was built by nothing, with no error.

Measured 2026-09-07 on ``experiment_11_attention_none.yaml``: deleting its five
``kspace_losses`` entries while keeping the matching lambdas left the weight table
unchanged and raised nothing, while the built module count fell from 6 to 1. That
is ``MUTATION_B`` below, and it is the shape this guard now has to fail on.

The guard reads the weight table as a second pass, so the tests that matter most
here are the ones asserting it stays *quiet*: a guard that raises on a legitimate
arm is worse than one that misses, because it blocks training. The
``marker``/``gan`` cases below are regressions the 656-arm corpus caught.
"""

from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures, frozen from the producer
# ---------------------------------------------------------------------------

#: ``experiment_11_attention_none.yaml`` with the five *duplicated* domain-list
#: entries deleted and their lambdas left behind — the edit an author makes when
#: they resolve the redundancy on the wrong surface. The list-only entries
#: (``hfen``, ``null_space_content``) and ``policy`` stay, because that is what
#: the arm looks like after such an edit.
MUTATION_B = {
    "reconstruction": {
        "lambda_complex_l1": 1.0,
        "lambda_log_spectral": 0.1,
        "lambda_sobolev_kspace": 0.05,
        "lambda_sense_adjoint_l1": 0.3,
        "log_spectral_skip_fft": True,
        "lambda_pre_dc_kspace": 0.3,
        "lambda_perceptual": 0.0,
    },
    "physics": {
        "lambda_complex_spatial_gradient": 1.0,
        "lambda_bloch_residual": 0.0,
        "lambda_physics_constraint": 0.0,
        "lambda_snr_preserving": 0.0,
    },
    "image_losses": [
        {
            "name": "hfen",
            "weight": 0.3,
            "enabled": True,
            "kwargs": {"kernel_size": 15, "sigma": 1.5, "normalize": True},
        }
    ],
    "kspace_losses": [{"name": "null_space_content", "weight": 0.25, "enabled": False}],
    "complex_losses": [],
    "policy": {"output_domain": "kspace"},
}

MUTATION_B_UNBUILT = (
    "complex_l1",
    "complex_spatial_gradient",
    "log_spectral",
    "sense_adjoint_l1",
    "sobolev_kspace",
)

#: ``losses:`` of ``pillars/exp_pillar_07_vf_fourier_shift.yaml``, verbatim.
#: ``marker`` canonicalises to ``marker_corruption`` while ``STRATEGY_MANAGED_LOSSES``
#: is authored in schema spellings, so a naive weight-table pass flagged this
#: legitimate arm. Three of the 29 managed names drift this way
#: (``content``/``marker``/``patch_nce``); the guard tests the raw declared name.
PILLAR07_LOSSES = {
    "reconstruction": {"enable_marker_loss": True, "lambda_marker": 2.0},
    "image_losses": [
        {"name": "mse", "weight": 1.0, "enabled": True},
        {"name": "ssim", "weight": 0.1, "enabled": False},
    ],
    "kspace_losses": [],
    "complex_losses": [],
    "policy": {"output_domain": "image"},
}


def _settings(losses_block: dict):
    from spectramr.config.schemas.data import DataConfigSchema
    from spectramr.config.schemas.logging import LoggingConfigSchema
    from spectramr.config.schemas.loss import LossConfigSchema
    from spectramr.config.schemas.metrics import MetricsConfigSchema
    from spectramr.config.schemas.model import ModelConfigSchema
    from spectramr.config.schemas.optimization import OptimizationConfigSchema
    from spectramr.config.settings import TrainingSettings

    return TrainingSettings(
        model=ModelConfigSchema(),
        data=DataConfigSchema(),
        optimization=OptimizationConfigSchema(),
        logging=LoggingConfigSchema(),
        metrics=MetricsConfigSchema(),
        losses=LossConfigSchema(**losses_block),
    )


def _build(losses_block: dict):
    """The full builder chain, in the order the training path drives it."""
    from spectramr.infrastructure.training.builders.loss_builder import LossBuilder

    return (
        LossBuilder(_settings(losses_block), device=torch.device("cpu"))
        .build_reconstruction_losses()
        .build_adversarial_losses()
        .build_physics_losses()
        .build_regularization_losses()
        .build_diffusion_losses()
        .build_latent_losses()
        .build_ssl_losses()
        .build()
    )


# ---------------------------------------------------------------------------
# The violation the guard exists for
# ---------------------------------------------------------------------------


class TestLambdaOnlyDeclarationIsRefused:
    def test_mutation_b_raises_naming_every_unbuilt_loss(self):
        from spectramr.domain.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError) as excinfo:
            _build(MUTATION_B)
        message = str(excinfo.value)
        for name in MUTATION_B_UNBUILT:
            assert f"'{name}'" in message, f"{name} not named in:\n{message}"

    def test_the_message_reports_the_weight_that_would_have_been_lost(self):
        """A name alone does not tell the author how much supervision vanished."""
        from spectramr.domain.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError) as excinfo:
            _build(MUTATION_B)
        message = str(excinfo.value)
        for name, weight in (
            ("complex_l1", 1.0),
            ("log_spectral", 0.1),
            ("sobolev_kspace", 0.05),
            ("sense_adjoint_l1", 0.3),
            ("complex_spatial_gradient", 1.0),
        ):
            assert f"'{name}' (weight={weight})" in message, message

    def test_fix_hint_names_a_live_spelling(self):
        """``objectives.*`` is ``extra=forbid``-rejected (#547, #1850).

        A hint routing the author from one raising spelling to another is worse
        than no hint.
        """
        from spectramr.domain.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError) as excinfo:
            _build(MUTATION_B)
        message = str(excinfo.value)
        assert "losses.kspace_losses" in message
        assert "losses.image_losses" in message
        assert "objectives." not in message

    def test_a_zero_weight_lambda_is_not_reported(self):
        """``lambda_perceptual: 0.0`` declares nothing to lose."""
        from spectramr.domain.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError) as excinfo:
            _build(MUTATION_B)
        assert "'perceptual'" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# The arms the guard must not block
# ---------------------------------------------------------------------------


class TestLegitimateArmsStillBuild:
    def test_list_declared_losses_do_not_raise(self):
        """The migrated shape: lists present, no redundant lambdas."""
        _build(
            {
                "reconstruction": {
                    "log_spectral_skip_fft": True,
                    "lambda_pre_dc_kspace": 0.3,
                    "lambda_perceptual": 0.0,
                },
                "kspace_losses": [
                    {"name": "complex_l1", "weight": 1.0, "enabled": True},
                    {"name": "log_spectral", "weight": 0.1, "enabled": True},
                    {"name": "sobolev_kspace", "weight": 0.05, "enabled": True},
                    {
                        "name": "complex_spatial_gradient",
                        "weight": 1.0,
                        "enabled": True,
                    },
                    {"name": "sense_adjoint_l1", "weight": 0.3, "enabled": True},
                ],
                "policy": {"output_domain": "kspace"},
            }
        )

    def test_marker_inline_shape_does_not_raise(self):
        """``pillars/exp_pillar_07_vf_fourier_shift.yaml`` — canonicalisation drift.

        Canonicalising the skip set instead would also exempt ``perceptual``,
        which #421 establishes is a distinct buildable loss from ``content``.
        """
        _build(PILLAR07_LOSSES)

    @pytest.mark.parametrize(
        "name,block",
        [
            # Read inline by the strategy with no ``enable_*`` gate and no list
            # entry possible — declared exemptions, verified at their read sites.
            ("pre_dc_kspace", {"reconstruction": {"lambda_pre_dc_kspace": 0.3}}),
            ("cycle_adv", {"gan": {"lambda_cycle_adv": 1.0}}),
            ("cycle_bloch", {"gan": {"lambda_cycle_bloch": 1.0}}),
            # Applied inside the discriminator step, never built standalone.
            ("gradient_penalty", {"gan": {"lambda_gp": 10.0}}),
            ("r1", {"gan": {"lambda_r1": 10.0}}),
            # #1468: ``losses.adversarial`` names no field; the real home is
            # ``losses.gan.lambda_adv``.
            ("adversarial", {"gan": {"lambda_adv": 1.0}}),
        ],
    )
    def test_strategy_managed_lambda_only_losses_do_not_raise(self, name, block):
        _build(block)


class TestGuardScopeIsDeclaredNotAssumed:
    """What this guard does NOT cover, stated so it cannot be mistaken for coverage.

    The guard runs inside ``_build_list_based_losses``, which ``_build_all_dynamic``
    reaches only when ``uses_list_based_losses`` is true — and ``_build_all_dynamic``
    returns even earlier when ``get_enabled_losses()`` is empty, which it is for a
    lambda-only arm (every ``enable_*`` defaults ``False``).

    So an arm that has *not started* migrating — no domain lists at all — is not
    guarded here. That is a scope boundary, not an oversight: widening it makes
    every unmigrated arm in the corpus refuse to train, which is a corpus decision
    rather than a detector change. The duplicate-declaration shape is covered
    corpus-wide by ``check_loss_weight_declaration_ssot`` at audit time.
    """

    def test_an_arm_with_no_domain_lists_is_not_reached_by_this_guard(self):
        block = {
            "reconstruction": {
                "lambda_complex_l1": 1.0,
                "lambda_log_spectral": 0.1,
            },
        }
        settings = _settings(block)
        assert settings.losses.uses_list_based_losses is False
        assert settings.losses.get_enabled_losses() == {}
        _build(block)  # documents the boundary; must not raise

    def test_one_list_entry_is_enough_to_arm_the_guard(self):
        """The discriminator between the two cases above.

        The entry is a k-space loss on purpose. It used to be ``hfen``, which is
        registered ``domain="image"`` -- under ``kspace_losses`` with
        ``output_domain: kspace`` nothing bridges it, so it would have scored raw
        k-space as anatomy. That is a real defect the zero-bridge guard now
        rejects before this one is reached, and the fixture only ever needed
        *some* list entry to arm the guard under test.
        """
        from spectramr.domain.exceptions import ConfigurationError

        block = {
            "reconstruction": {"lambda_complex_l1": 1.0},
            "kspace_losses": [{"name": "null_space_content", "weight": 0.3, "enabled": True}],
            "policy": {"output_domain": "kspace"},
        }
        assert _settings(block).losses.uses_list_based_losses is True
        with pytest.raises(ConfigurationError, match="complex_l1"):
            _build(block)
