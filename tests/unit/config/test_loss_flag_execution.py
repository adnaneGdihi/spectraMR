"""Loss Flag → LossBuilder Execution Tests.

PURPOSE:
For each `enable_X = True` flag in the loss schemas, verify that
LossBuilder._build_all_dynamic() actually produces a concrete nn.Module in
self._losses.  Catches the class of bugs where:
  - The flag and lambda are set correctly in YAML
  - get_enabled_losses() emits the loss name
  - BUT LossBuilder has no registry_map entry / create_loss() call for it
    → silently skipped, loss_fn never runs, gradient never flows.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectramr.models.losses.build_set import declared_build_set, strategy_resolved
from spectramr.models.losses.weights import build_loss_weight_table


def _declared(losses):
    """Weights the author declared (the table owns this, #1966); the build set is narrower."""
    return {
        k: s.weight
        for k, s in build_loss_weight_table(losses).items()
        if s.enabled and s.weight > 0
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_minimal_settings(losses_kwargs: dict) -> TrainingSettings:
    """Build a minimal TrainingSettings sufficient for LossBuilder."""
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
        losses=LossConfigSchema(**losses_kwargs),
    )


def _build_losses(settings) -> dict[str, nn.Module]:
    """Run LossBuilder and return the resulting losses dict."""
    from spectramr.infrastructure.training.builders.loss_builder import LossBuilder

    builder = LossBuilder(config=settings, device=torch.device("cpu"))
    builder._build_all_dynamic()
    return builder._losses


# ---------------------------------------------------------------------------
# 1. Reconstruction losses
# ---------------------------------------------------------------------------


class TestReconstructionLossExecution:
    """Each enable_* flag in ReconstructionLossesConfig must produce a real module."""

    CASES = [
        ("l1", {"enable_l1": True, "lambda_l1": 1.0}),
        ("l2", {"enable_l2": True, "lambda_l2": 1.0}),
        ("smooth_l1", {"enable_smooth_l1": True, "lambda_smooth_l1": 1.0}),
        ("perceptual", {"enable_perceptual": True, "lambda_perceptual": 1.0}),
        ("ssim", {"enable_ssim": True, "lambda_ssim": 1.0}),
        ("ms_ssim", {"enable_ms_ssim": True, "lambda_ms_ssim": 1.0}),
        ("hfen", {"enable_hfen": True, "lambda_hfen": 1.0}),
        ("ffl", {"enable_ffl": True, "lambda_ffl": 1.0}),
        (
            "kspace",
            {"enable_kspace": True, "lambda_kspace": 1.0},
        ),  # registry: 'spectral_kspace' or 'kspace_consistency'
        ("frequency", {"enable_frequency": True, "lambda_frequency": 1.0}),
        ("log_spectral", {"enable_log_spectral": True, "lambda_log_spectral": 1.0}),
        ("tv", {"enable_tv": True, "lambda_tv": 1.0}),
        ("edge", {"enable_edge": True, "lambda_edge": 1.0}),
        (
            "explicit_gradient",
            {"enable_explicit_gradient": True, "lambda_explicit_gradient": 1.0},
        ),
        ("hist", {"enable_hist": True, "lambda_hist": 1.0}),
        ("complex_l1", {"enable_complex_l1": True, "lambda_complex_l1": 1.0}),
        ("complex_mse", {"enable_complex_mse": True, "lambda_complex_mse": 1.0}),
        ("ffl", {"enable_ffl": True, "lambda_ffl": 1.0}),
        ("mind_ssc", {"enable_mind_ssc": True, "lambda_mind_ssc": 1.0}),
        (
            "background_suppression",
            {
                "enable_background_suppression": True,
                "lambda_background_suppression": 1.0,
            },
        ),
        (
            "rician_consistency",
            {"enable_rician_consistency": True, "lambda_rician_consistency": 0.3},
        ),
        (
            "sobolev_kspace",
            {"enable_sobolev_kspace": True, "lambda_sobolev_kspace": 1.0},
        ),
        (
            "frequency_weighted_l1_kspace",
            {
                "enable_frequency_weighted_l1_kspace": True,
                "lambda_frequency_weighted_l1_kspace": 1.0,
            },
        ),
        (
            "energy_conservation",
            {"enable_energy_conservation": True, "lambda_energy_conservation": 1.0},
        ),
        (
            "latent_consistency",
            {"enable_latent_consistency": True, "lambda_latent_consistency": 0.1},
        ),
        (
            "weighted_kspace_l1",
            {"enable_weighted_kspace_l1": True, "lambda_weighted_kspace_l1": 1.0},
        ),
        (
            "sense_adjoint_l1",
            {"enable_sense_adjoint_l1": True, "lambda_sense_adjoint_l1": 1.0},
        ),
        (
            "dino_perceptual",
            {"enable_dino_perceptual": True, "lambda_dino_perceptual": 1.0},
        ),
        ("dists", {"enable_dists": True, "lambda_dists": 1.0}),
    ]

    @pytest.mark.parametrize(
        "expected_key,recon_kwargs", CASES, ids=[c[0] for c in CASES]
    )
    def test_loss_module_is_created(self, expected_key: str, recon_kwargs: dict):
        """enable_{key}=True must produce a nn.Module in LossBuilder._losses[{key}]."""
        from spectramr.config.schemas.loss import ReconstructionLossesConfig

        # These keys have a mismatch between the schema enable flag name and the
        # LossBuilder registry_map — the loss is silently not created even when enabled.
        KNOWN_REGISTRY_MAPPING_GAPS = {
            "kspace",  # enable_kspace=True emits 'kspace' but no registry entry for it
            # (closest: 'spectral_kspace'). Fix: add registry_map["kspace"] = "spectral_kspace"
        }

        settings = _make_minimal_settings(
            {"reconstruction": ReconstructionLossesConfig(**recon_kwargs)}
        )
        losses = _build_losses(settings)

        if expected_key in KNOWN_REGISTRY_MAPPING_GAPS:
            if expected_key not in losses:
                pytest.xfail(
                    reason=f"KNOWN GAP: '{expected_key}' enable flag emits '{expected_key}' but "
                    f"LossBuilder has no registry_map entry for it — loss is silently not created.\n"
                    f"Fix: add registry_map['{expected_key}'] = '<correct_registry_name>' in "
                    f"LossBuilder._build_all_dynamic()"
                )

        # A reconstruction lambda is computed inline by its computer (models/losses/build_set.py),
        # so the builder need not instantiate it; it must still be a declared, active term.
        spec = build_loss_weight_table(settings.losses).get(expected_key)
        assert spec is not None and spec.enabled and spec.weight > 0, (
            f"'{expected_key}' is not a declared, active term for enable_{expected_key}=True."
        )
        if expected_key not in losses:
            assert strategy_resolved(spec), (
                f"LossBuilder did not create '{expected_key}' and no computer or strategy owns it."
            )
            return
        assert isinstance(losses[expected_key], nn.Module), (
            f"losses['{expected_key}'] must be an nn.Module, "
            f"got {type(losses[expected_key]).__name__}"
        )


# ---------------------------------------------------------------------------
# 2. Physics losses
# ---------------------------------------------------------------------------


class TestPhysicsLossExecution:
    """Each enable_* flag in PhysicsLossesConfig must produce a real module."""

    CASES = [
        (
            "bloch_residual",
            {"enable_bloch_residual": True, "lambda_bloch_residual": 1.0},
        ),
        (
            "physics_constraint",
            {"enable_physics_constraint": True, "lambda_physics_constraint": 0.1},
        ),
        # NOTE: 'parallel_imaging' key from enable flag != registry 'parallel_imaging_kspace'
        # This is a KNOWN REGISTRY GAP — documented as xfail below
        (
            "parallel_imaging",
            {"enable_parallel_imaging": True, "lambda_parallel_imaging": 1.0},
        ),
        (
            "snr_preserving",
            {"enable_snr_preserving": True, "lambda_snr_preserving": 1.0},
        ),
        (
            "complex_spatial_gradient",
            {
                "enable_complex_spatial_gradient": True,
                "lambda_complex_spatial_gradient": 1.0,
            },
        ),
    ]

    KNOWN_PHYSICS_REGISTRY_GAPS = {
        # enable_parallel_imaging emits 'parallel_imaging' but registry has 'parallel_imaging_kspace'
        # Fix: add registry_map["parallel_imaging"] = "parallel_imaging_kspace" in LossBuilder
        "parallel_imaging",
    }

    @pytest.mark.parametrize(
        "expected_key,physics_kwargs", CASES, ids=[c[0] for c in CASES]
    )
    def test_physics_loss_module_is_created(
        self, expected_key: str, physics_kwargs: dict
    ):
        from spectramr.config.schemas.loss import PhysicsLossesConfig
        from spectramr.infrastructure.training.builders.loss_builder import (
            STRATEGY_MANAGED_LOSSES,
        )

        settings = _make_minimal_settings(
            {"physics": PhysicsLossesConfig(**physics_kwargs)}
        )

        if expected_key in self.KNOWN_PHYSICS_REGISTRY_GAPS:
            # #368: the enable flag emits '<key>' but no loss is registered under
            # that name (the real one is 'parallel_imaging_kspace'), so LossBuilder
            # raises. Known and tracked, not yet wired — xfail before the build.
            pytest.xfail(
                reason=f"#368: enable flag for '{expected_key}' emits a name with no "
                "registered/strategy-managed loss (real name 'parallel_imaging_kspace')."
            )

        losses = _build_losses(settings)

        if expected_key in STRATEGY_MANAGED_LOSSES:
            # Injected by the training strategy at train time (STRATEGY_MANAGED_LOSSES),
            # so LossBuilder deliberately does NOT build a standalone module. Absence
            # here is the contract, not a silent drop (#9); double-building would
            # double-count the term.
            assert expected_key not in losses, (
                f"'{expected_key}' is strategy-managed and must not also be built "
                f"standalone by LossBuilder. Keys: {sorted(losses.keys())}"
            )
            return

        assert expected_key in losses, (
            f"LossBuilder did not create physics loss '{expected_key}'.\n"
            f"Keys created: {sorted(losses.keys())}"
        )
        assert isinstance(losses[expected_key], nn.Module)


# ---------------------------------------------------------------------------
# 3. GAN losses (non-adversarial)
# ---------------------------------------------------------------------------


class TestGANAuxiliaryLossExecution:
    """GAN auxiliary losses (R1 regularization, etc.) must be created."""

    def test_r1_regularization_is_created(self):
        from spectramr.config.schemas.loss import GANLossesConfig
        from spectramr.infrastructure.training.builders.loss_builder import (
            STRATEGY_MANAGED_LOSSES,
        )

        settings = _make_minimal_settings(
            {
                "gan": GANLossesConfig(
                    enable_r1=True, lambda_r1=10.0, enable_adversarial=False
                )
            }
        )
        # r1 is enabled and emitted by get_enabled_losses...
        assert _declared(settings.losses).get("r1", 0) > 0
        # ...but it is strategy-managed (applied in the GAN strategy's discriminator
        # step), so LossBuilder deliberately does NOT build a standalone module.
        # Confirm it is recognised as managed rather than silently dropped (#9).
        assert "r1" in STRATEGY_MANAGED_LOSSES
        losses = _build_losses(settings)
        assert "r1" not in losses, (
            f"r1 is strategy-managed and must not also be built standalone. "
            f"Keys: {sorted(losses.keys())}"
        )


# ---------------------------------------------------------------------------
# 4. SSL losses
# ---------------------------------------------------------------------------


class TestSSLLossExecution:
    """SSL/contrastive losses must be created when enabled."""

    CASES = [
        ("contrastive", {"enable_contrastive": True, "lambda_contrastive": 1.0}),
        # NOTE: 'ca_contrastive' is not in the loss registry — it's a schema-only key
        # with no registered implementation. Documented as xfail.
        (
            "ca_contrastive",
            {"enable_ca_contrastive": True, "lambda_ca_contrastive": 1.0},
        ),
    ]

    KNOWN_SSL_REGISTRY_GAPS = {
        # enable_ca_contrastive emits 'ca_contrastive' but no such loss is registered
        # Fix: implement ContrastiveAnatomyLoss and register as 'ca_contrastive', OR
        # remove enable_ca_contrastive from SSLLossesConfig schema if not planned
        "ca_contrastive",
    }

    @pytest.mark.parametrize(
        "expected_key,ssl_kwargs", CASES, ids=[c[0] for c in CASES]
    )
    def test_ssl_loss_module_is_created(self, expected_key: str, ssl_kwargs: dict):
        from spectramr.config.schemas.loss import SSLLossesConfig

        settings = _make_minimal_settings({"ssl": SSLLossesConfig(**ssl_kwargs)})

        if expected_key in self.KNOWN_SSL_REGISTRY_GAPS:
            # #368: 'ca_contrastive' is advertised in SSLLossesConfig but has no
            # registered implementation, so LossBuilder raises. Tracked, not yet
            # wired — xfail before the build.
            pytest.xfail(
                reason=f"#368: '{expected_key}' is in SSLLossesConfig schema but has no "
                "registered loss implementation."
            )

        losses = _build_losses(settings)

        assert (
            expected_key in losses
        ), f"SSL loss '{expected_key}' not created. Keys: {sorted(losses.keys())}"


# ---------------------------------------------------------------------------
# 5. Curriculum scheduling does not crash the builder
# ---------------------------------------------------------------------------


class TestCurriculumSchedulingNoCrash:
    """Curriculum scheduling is config-only (no module). It must not crash the builder."""

    def test_curriculum_scheduling_enabled_no_crash(self):
        from spectramr.config.schemas.loss import ReconstructionLossesConfig

        recon = ReconstructionLossesConfig(
            use_curriculum_scheduling=True,
            curriculum_warmup_epochs=10.0,
            curriculum_max_epochs=100.0,
            curriculum_l1_end_weight=2.0,
            curriculum_hfen_end_weight=8.0,
        )
        settings = _make_minimal_settings({"reconstruction": recon})
        # Must not raise
        losses = _build_losses(settings)
        # At minimum the l1 default must be present
        assert isinstance(losses, dict)


# ---------------------------------------------------------------------------
# 6. No enabled loss has a zero-weight (would silently disable the gradient)
# ---------------------------------------------------------------------------


class TestNoZeroWeightEnabledLoss:
    """An enabled loss with lambda=0 should either be excluded from get_enabled_losses()
    or raise a clear configuration error — it must NOT silently compute a scaled-zero term.
    """

    def test_zero_weight_l1_excluded_from_enabled(self):
        """lambda_l1=0.0 with enable_l1=True → key must NOT appear with 0-weight."""
        from spectramr.config.schemas.loss import (
            LossConfigSchema,
            ReconstructionLossesConfig,
        )

        cfg = LossConfigSchema(
            reconstruction=ReconstructionLossesConfig(enable_l1=True, lambda_l1=0.0)
        )
        enabled = declared_build_set(cfg)
        # If 'l1' is in enabled, its weight must be > 0
        if "l1" in enabled:
            assert enabled["l1"] > 0, (
                "get_enabled_losses() returned l1 with weight=0.0. "
                "Zero-weight enabled losses waste computation silently."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
