import pytest
from pydantic import ValidationError

from spectramr.config.schemas.loss import (
    DiffusionLossesConfig,
    GANLossesConfig,
    LatentLossesConfig,
    LossConfigSchema,
    MetricsConfig,
    ReconstructionLossesConfig,
)
from spectramr.models.losses.weights import build_loss_weight_table


def _declared(losses):
    """Weights the author declared (the table owns this, #1966); the build set is narrower."""
    return {
        k: s.weight
        for k, s in build_loss_weight_table(losses).items()
        if s.enabled and s.weight > 0
    }



class TestLossConfigSchema:
    def test_defaults(self):
        schema = LossConfigSchema()
        # enable_* flags are opt-in (default False) — YAML must turn the
        # ones the experiment wants on. Lambdas keep documented baselines.
        assert schema.reconstruction.enable_l1 is False
        assert schema.reconstruction.enable_perceptual is False
        # GAN losses are None by default in LossConfigSchema
        assert schema.gan is None
        # Diffusion losses are None by default in LossConfigSchema
        assert schema.diffusion is None

    def test_custom_values(self):
        schema = LossConfigSchema(
            reconstruction=ReconstructionLossesConfig(enable_l1=True, enable_l2=True),
            gan=GANLossesConfig(enable_gradient_penalty=False),
            policy={"exclude_defaults": ["mse"]},
        )
        assert schema.reconstruction.enable_l1 is True
        assert schema.reconstruction.enable_l2 is True
        assert schema.gan.enable_gradient_penalty is False
        # `disable_default_losses` is a fold-posture legacy spelling: the
        # validator moves it to `policy.exclude_defaults` at parse time, so the
        # attribute of that name no longer exists. Asserting the legacy name is
        # what made this stale -- read where the value actually lands.
        assert schema.policy.exclude_defaults == ["mse"]

    def test_the_three_inert_global_knobs_are_gone(self):
        """`normalize_losses` / `clip_loss_value` / `loss_scaling` were deleted
        2026-08-03 (issue #676): zero readers, zero corpus declarations.

        This replaces a `gt=0` bounds test on `loss_scaling`, which asserted
        that a knob nothing reads rejects one of the values it would have
        ignored. `loss_scaling`'s apparent readers were
        `MixedPrecisionConfig.loss_scaling` -- a different class, and the
        leaf-name blind spot of issue #674.

        Note the block is ``extra="ignore"``, so re-adding one of these names to
        a YAML is silently dropped rather than rejected. That is visible via the
        Tier-1 ``declared_keys_are_not_discarded`` check, not here.
        """
        for field in ("normalize_losses", "clip_loss_value", "loss_scaling"):
            assert field not in LossConfigSchema.model_fields, (
                f"{field} is back on LossConfigSchema; it had no reader when it "
                "was removed, so re-adding it needs one (pitfall #15)"
            )


class TestReconstructionLossesConfig:
    def test_defaults(self):
        schema = ReconstructionLossesConfig()
        # enable_* flags are opt-in (see TestLossConfigSchema.test_defaults).
        assert schema.enable_l1 is False
        assert schema.enable_perceptual is False

    def test_enable_recon_surfaces_recon_loss(self):
        """Regression WS1-core-02: ``lambda_recon`` pairs with ``enable_recon``,
        not ``enable_l1``. get_enabled_losses() previously checked enable_l1 for
        the reconstruction-paradigm 'recon' loss, so ``enable_recon=True`` +
        ``lambda_recon=1.0`` silently produced nothing (disentangled-VAE arms).
        """
        cfg = LossConfigSchema(
            reconstruction=ReconstructionLossesConfig(
                enable_recon=True, lambda_recon=1.0
            )
        )
        enabled = _declared(cfg)
        assert enabled.get("recon") == 1.0

    def test_enable_l1_independent_of_recon(self):
        """enable_l1 has its own lambda_l1 pair and must not be conflated."""
        cfg = LossConfigSchema(
            reconstruction=ReconstructionLossesConfig(enable_l1=True, lambda_l1=2.0)
        )
        enabled = _declared(cfg)
        assert enabled.get("l1") == 2.0
        assert "recon" not in enabled

    def test_reconstruction_managed_losses_reports_enabled_names(self):
        """``reconstruction_managed_losses`` names the losses the reconstruction
        computer resolves from ``reconstruction.lambda_*`` — so LossBuilder's
        unmigrated-key guard can skip them when they are deliberately kept out
        of the declarative image/kspace/complex lists (direct_ulf_to_hf_sr).
        """
        cfg = LossConfigSchema(
            reconstruction=ReconstructionLossesConfig(
                enable_l1=True,
                lambda_l1=1.0,
                enable_ssim=True,
                lambda_ssim=0.5,
                enable_l2=False,  # disabled → not managed
                lambda_l2=3.0,
            )
        )
        managed = cfg.reconstruction_managed_losses()
        assert "l1" in managed
        assert "ssim" in managed
        assert "l2" not in managed  # enable flag off

    def test_reconstruction_managed_losses_excludes_zero_weight(self):
        """An enabled term with lambda==0 is inert; it must not be reported as
        managed (else a genuinely-dropped loss would be silently excused)."""
        cfg = LossConfigSchema(
            reconstruction=ReconstructionLossesConfig(enable_l1=True, lambda_l1=0.0)
        )
        assert "l1" not in cfg.reconstruction_managed_losses()


class TestGANLossesConfig:
    def test_defaults(self):
        schema = GANLossesConfig()
        # GAN flags follow the same opt-in convention.
        assert schema.enable_adversarial is False
        assert schema.enable_gradient_penalty is False


class TestDiffusionLossesConfig:
    def test_defaults(self):
        schema = DiffusionLossesConfig()
        # Diffusion losses only have enable_* flags
        # Lambda weights are in objectives.diffusion
        assert schema.enable_diffusion is False


class TestLatentLossesConfig:
    def test_defaults(self):
        schema = LatentLossesConfig()
        # Latent losses only have enable_* flags
        # Lambda weights are in objectives.latent
        assert schema.enable_reconstruction is False
        assert schema.enable_kl is False


class TestPreDcKspaceWeight:
    """The opt-in pre-DC fidelity weight (Experiment-11 DC-blob L1+)."""

    def test_default_is_zero_noop(self):
        """Defaults to 0.0 so existing configs are unaffected (no extra term)."""
        assert ReconstructionLossesConfig().lambda_pre_dc_kspace == 0.0

    def test_accepts_positive_weight(self):
        assert (
            ReconstructionLossesConfig(lambda_pre_dc_kspace=0.5).lambda_pre_dc_kspace
            == 0.5
        )

    def test_rejects_negative_weight(self):
        """ge=0 — a negative fidelity weight is a ValidationError."""
        with pytest.raises(ValidationError):
            ReconstructionLossesConfig(lambda_pre_dc_kspace=-0.1)


class TestMetricsConfig:
    def test_defaults(self):
        schema = MetricsConfig()
        assert schema.compute_psnr is True
        assert schema.compute_ssim is True
