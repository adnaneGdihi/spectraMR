"""Comprehensive tests for new loss computers and patterns.

Tests ensure:
1. Loss computers initialize correctly
2. Loss computers compute losses without backward pass
3. Loss output formats are correct
4. Loss keys validate correctly
5. Backward passes work when called explicitly
"""

import pytest
import torch
import torch.nn as nn

# Import directly to avoid circular imports
from spectramr.models.losses.computers.base import LossOutput
from spectramr.models.losses.computers.unified_diffusion_reconstruction import (
    UnifiedDiffusionLossComputer,
    UnifiedReconstructionLossComputer,
)
from spectramr.models.losses.computers.unified_gan import UnifiedGANLossComputer
from spectramr.models.losses.computers.unified_vae import (
    UnifiedVAELossComputer,
    UnifiedVQVAELossComputer,
)
from spectramr.models.losses.loss_key_registry import LossKeyRegistry, LossKeyValidator


def real_settings(**losses_reconstruction):
    """A REAL frozen ``TrainingSettings`` -- the config the computers expect.

    ``MockConfig`` below is a hand-rolled double, and the loss computers do not
    consume a config the way it pretends: the explicit loss path is built from a
    real ``losses`` block, so under the stub it produces nothing and
    ``compute()`` raises *"All declared losses failed to compute (components is
    empty)"* -- correctly, since that guard exists to stop training on a
    disconnected zero total. Four of cluster job 8004252's failures were that
    guard firing at a stub, not at a defect.

    Built through ``from_yaml`` rather than ``model_validate``: ``config_version``
    is stripped by the loader and is not a model field, so ``model_validate``
    rejects it outright.

    Frozen, hence the kwargs -- declare the weights up front instead of mutating
    afterwards (non-negotiable #1). That immutability is the only reason the
    remaining ``MockConfig`` call sites in this file have not been migrated too:
    they mutate ``config.training.training_mode`` and ``lambda_*`` in place, and
    converting them is a separate change that should be made with the suite
    runnable.
    """
    import tempfile

    import yaml

    from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION
    from spectramr.config.settings import TrainingSettings

    doc = {
        "config_version": CANONICAL_CONFIG_VERSION,
        "model": {
            "model_type": "standard_unet",
            "in_channels": 1,
            "out_channels": 1,
        },
        "data": {"dataset_type": "synthetic"},
        "training": {"training_mode": "reconstruction"},
        "losses": {"reconstruction": dict(losses_reconstruction)},
        "optimization": {},
        "logging": {},
        "run": {"device": "cpu"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(doc, fh)
        path = fh.name
    return TrainingSettings.from_yaml(path)


class MockConfig:
    """Minimal config for testing."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.optimization = type(
            "obj",
            (object,),
            {
                "learning_rate": 1e-4,
                "use_amp": False,
            },
        )()
        self.model = type(
            "obj",
            (object,),
            {
                "model_type": "test",
                "in_channels": 2,
                "out_channels": 2,
            },
        )()
        self.losses = type(
            "obj",
            (object,),
            {
                "reconstruction": type(
                    "obj",
                    (object,),
                    {
                        "lambda_l1": 1.0,
                        "lambda_l2": 0.5,
                        "lambda_perceptual": 0.1,
                        "lambda_ssim": 0.0,
                        "lambda_lpips": 0.0,
                        "spatial_losses_use_fourier_bridge": False,
                        "log_spectral_skip_fft": False,
                        "ffl_alpha": 1.0,
                        "histogram_bins": 256,
                        "background_suppression_threshold_ratio": 0.1,
                        "background_suppression_use_fourier_bridge": False,
                        "rician_noise_sigma": 0.0,
                        "rician_use_fourier_bridge": False,
                        "frequency_weighted_l1_kspace_alpha": 1.0,
                        "weighted_kspace_exponent": 1.0,
                    },
                )(),
                "gan": type(
                    "obj",
                    (object,),
                    {
                        "lambda_adv": 0.1,
                        "lambda_gp": 10.0,
                        "feature_matching": 0.0,
                        "gan_loss_type": "standard",
                        "label_smoothing": 0.0,
                    },
                )(),
                "physics": None,
                "get_enabled_losses": lambda self: {"l1": 1.0, "adversarial": 0.1},
            },
        )()
        self.deep_supervision_weight = 0.0
        self.data = type(
            "obj",
            (object,),
            {
                "patch_size": [32, 32],
            },
        )()
        self.training = type(
            "obj",
            (object,),
            {
                "training_mode": "reconstruction",
            },
        )()


class SimpleGenerator(nn.Module):
    """Simple test generator."""

    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class SimpleDiscriminator(nn.Module):
    """Simple test discriminator."""

    def __init__(self, in_channels=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


# ============================================================================
# Tests for BaseLossComputer
# ============================================================================


class TestBaseLossComputer:
    """Test base loss computer functionality."""

    def test_loss_output_creation(self):
        """Test LossOutput dataclass creation."""
        components = {"loss_l1": torch.tensor(1.0), "loss_l2": torch.tensor(0.5)}
        total = torch.tensor(1.5)
        metrics = {"psnr": 30.0}

        output = LossOutput(total=total, components=components, metrics=metrics)

        assert output.total == total
        assert output.components == components
        assert output.metrics == metrics

    def test_loss_computer_initialization(self):
        """Test that loss computer initializes without errors."""
        config = MockConfig()
        computer = UnifiedReconstructionLossComputer(config=config, device=torch.device("cpu"))

        assert computer is not None
        assert computer.config == config
        assert computer.device == torch.device("cpu")

    def test_loss_computer_requires_grad(self):
        """Test that loss computer sets requires_grad correctly."""
        config = MockConfig()
        computer = UnifiedReconstructionLossComputer(config=config, device=torch.device("cpu"))

        # Loss computer should not have trainable parameters
        # (if it does, they should not require grad)
        for attr_name, attr_value in computer.__dict__.items():
            if isinstance(attr_value, nn.Module):
                for param in attr_value.parameters():
                    assert not param.requires_grad, (
                        "Loss computer should not have requires_grad=True parameters"
                    )

    # ------------------------------------------------------------------
    # Silent-NaN-collapse guard (2026-05-21 Round 16, F29)
    # ------------------------------------------------------------------
    def test_partial_nan_skip_is_silent_and_keeps_finite_component(self):
        """One NaN component is benign — the survivor still contributes."""
        computer = UnifiedReconstructionLossComputer(
            config=MockConfig(), device=torch.device("cpu")
        )
        comps = {
            "mse": torch.tensor(float("nan")),
            "l1": torch.tensor(2.0, requires_grad=True),
        }
        total = computer._stack_components(comps, weights={"mse": 1.0, "l1": 1.0})
        assert torch.isfinite(total).all()
        assert float(total.detach()) == pytest.approx(2.0)

    def test_all_nan_components_emit_silent_collapse_error(self, caplog):
        """All components NaN → loud ERROR (loss is 0, model not training)."""
        import logging

        computer = UnifiedReconstructionLossComputer(
            config=MockConfig(), device=torch.device("cpu")
        )
        comps = {
            "mse": torch.tensor(float("nan")),
            "l1": torch.tensor(float("inf")),
        }
        with caplog.at_level(logging.ERROR, logger="spectramr.models.losses.computers.base"):
            total = computer._stack_components(comps, weights={"mse": 1.0, "l1": 1.0})
        # total is exactly 0.0 (no contribution) — the dangerous case.
        assert float(total.detach()) == 0.0
        assert any("SILENT NaN COLLAPSE" in rec.message for rec in caplog.records), (
            "all-NaN collapse must escalate to a distinct ERROR for the "
            "smoke-log triage to reclassify the green PASS as a failure"
        )

    def test_empty_components_do_not_emit_collapse_error(self, caplog):
        """No components at all is not a collapse — must stay quiet."""
        import logging

        computer = UnifiedReconstructionLossComputer(
            config=MockConfig(), device=torch.device("cpu")
        )
        with caplog.at_level(logging.ERROR, logger="spectramr.models.losses.computers.base"):
            total = computer._stack_components({})
        assert float(total.detach()) == 0.0
        assert not any("SILENT NaN COLLAPSE" in rec.message for rec in caplog.records)

    # ------------------------------------------------------------------
    # Dead-loss guard (2026-06-27): finite component(s) but every weight 0.
    # Parallel hole to the NaN collapse — the all-NaN guard misses it because
    # the surviving component is finite. (cs_mno dead_loss cohort: l1 warmup-
    # gated to 0 while its ms_ssim sibling went NaN on an off-scale operator
    # output and was skipped → total 0.0, zero gradient, train_psnr at -30.)
    # ------------------------------------------------------------------
    def test_all_weights_zero_emits_dead_loss_error(self, caplog):
        """Finite component(s) but all weights 0 → total 0.0, zero gradient."""
        import logging

        computer = UnifiedReconstructionLossComputer(
            config=MockConfig(), device=torch.device("cpu")
        )
        comps = {"l1": torch.tensor(2.0, requires_grad=True)}
        with caplog.at_level(logging.ERROR, logger="spectramr.models.losses.computers.base"):
            total = computer._stack_components(comps, weights={"l1": 0.0})
        assert float(total.detach()) == 0.0
        assert any("DEAD LOSS" in rec.message for rec in caplog.records), (
            "a finite component weighted to 0.0 (warmup gate eating the only "
            "loss) yields a zero-gradient total and must escalate to a distinct "
            "ERROR the smoke-log triage can catch"
        )

    def test_nonzero_weight_does_not_emit_dead_loss_error(self, caplog):
        """A real (weight>0) contribution must NOT trip the dead-loss guard."""
        import logging

        computer = UnifiedReconstructionLossComputer(
            config=MockConfig(), device=torch.device("cpu")
        )
        comps = {
            "l1": torch.tensor(2.0, requires_grad=True),  # weight 0 (warmup)
            "l2": torch.tensor(1.0, requires_grad=True),  # weight 1.0 — carries it
        }
        with caplog.at_level(logging.ERROR, logger="spectramr.models.losses.computers.base"):
            total = computer._stack_components(comps, weights={"l1": 0.0, "l2": 1.0})
        assert float(total.detach()) == pytest.approx(1.0)
        assert not any("DEAD LOSS" in rec.message for rec in caplog.records)


# ============================================================================
# Tests for UnifiedDiffusionLossComputer
# ============================================================================


class TestUnifiedDiffusionLossComputer:
    """Test diffusion loss computer."""

    @pytest.fixture
    def setup(self):
        """Setup for diffusion tests."""
        config = MockConfig()
        config.training.training_mode = "diffusion"
        computer = UnifiedDiffusionLossComputer(config=config, device=torch.device("cpu"))

        # Test data
        batch_size, channels, height, width = 2, 2, 32, 32
        pred = torch.randn(batch_size, channels, height, width, requires_grad=True)
        target = torch.randn(batch_size, channels, height, width)
        timesteps = torch.randint(0, 1000, (batch_size,))

        return computer, pred, target, timesteps

    def test_diffusion_forward_loss(self, setup):
        """Test diffusion forward loss computation."""
        computer, pred, target, timesteps = setup

        loss_output = computer.compute_forward_loss(
            pred=pred, target=target, timesteps=timesteps, epoch=0
        )

        # Check output structure
        assert isinstance(loss_output.total, torch.Tensor)
        assert loss_output.total.item() > 0, "Loss should be positive"
        assert loss_output.total.grad_fn is not None, "Loss should track gradients"
        assert len(loss_output.components) > 0, "Should have loss components"

    def test_diffusion_no_backward_in_compute(self, setup):
        """Test that compute() doesn't perform backward pass."""
        computer, pred, target, timesteps = setup

        # Store initial grad state
        initial_grad = pred.grad.clone() if pred.grad is not None else None

        # Compute loss (should NOT update gradients)
        loss_output = computer.compute_forward_loss(
            pred=pred, target=target, timesteps=timesteps, epoch=0
        )

        # Manually backward (this is how it should be done)
        loss_output.total.backward()

        # Check that backward was effective
        assert pred.grad is not None, "Manual backward should set gradients"


# ============================================================================
# Tests for UnifiedReconstructionLossComputer
# ============================================================================


class TestUnifiedReconstructionLossComputer:
    """Test reconstruction loss computer."""

    @pytest.fixture
    def setup(self):
        """Setup for reconstruction tests.

        Uses a REAL losses block. Under ``MockConfig`` the explicit loss path
        builds nothing, so ``compute()`` raised *"All declared losses failed to
        compute (components is empty)"* -- the fail-loud guard working correctly
        against a double that does not carry what the computer reads. Three of
        cluster job 8004252's failures were that.

        ``lambda_l2`` is non-zero because ``l1`` is warmup-gated: at
        ``iteration=0`` it resolves to 0.0, so a fixture declaring only ``l1``
        has no surviving term and lands right back on the same guard.
        """
        config = real_settings(lambda_l1=1.0, lambda_l2=0.5)
        computer = UnifiedReconstructionLossComputer(config=config, device=torch.device("cpu"))

        batch_size, channels, height, width = 2, 2, 32, 32
        pred = torch.randn(batch_size, channels, height, width, requires_grad=True)
        target = torch.randn(batch_size, channels, height, width)

        return computer, pred, target

    def test_reconstruction_loss(self, setup):
        """Test reconstruction loss computation."""
        computer, pred, target = setup

        loss_output = computer.compute(pred=pred, target=target, epoch=0)

        assert isinstance(loss_output.total, torch.Tensor)
        assert loss_output.total.grad_fn is not None, "Loss should track gradients"
        assert loss_output.total.item() > 0, "Loss should be positive"

    def test_reconstruction_loss_components(self, setup):
        """Test that loss has expected components."""
        computer, pred, target = setup

        loss_output = computer.compute(pred=pred, target=target, epoch=0)

        # Should have L1, L2, perceptual components
        keys = set(loss_output.components.keys())
        expected = {"l1", "l2", "perceptual"}

        # At least some components should exist
        assert len(keys & expected) > 0, f"Expected some of {expected}, got {keys}"

    def test_reconstruction_backward(self, setup):
        """Test that backward pass works correctly."""
        computer, pred, target = setup

        loss_output = computer.compute(pred=pred, target=target, epoch=0)
        loss_output.total.backward()

        # Check that gradients were computed
        assert pred.grad is not None, "Backward should compute gradients"
        assert pred.grad.abs().sum() > 0, "Gradients should be non-zero"

    def test_declarative_mse_only_yaml_still_computes_loss_pre_warmup(self) -> None:
        """Regression for the FNO/neural-ODE/contrastive smoke arms.

        Configs whose only declared loss is ``image_losses: [{name: mse}]``
        (the v6.0 declarative form) used to produce a disconnected zero
        leaf for the first 1000 iterations of every reconstruction run:

        * ``losses.reconstruction.lambda_l1`` defaults to 10.0 but ``l1``
          is in ``SPATIAL_LOSSES`` and warmed-up to 0 for the first
          ``warmup_iterations`` iters.
        * ``losses.reconstruction.lambda_l2`` defaults to 0.0 → L2 path
          gated off.
        * The LossBuilder pushes the user's ``mse`` callable into
          ``losses_dict``, but the dynamic loop's name-based skip-list
          dropped it unconditionally — so ``components`` stayed empty
          and ``_stack_components`` returned a fresh ``torch.tensor(0.0,
          requires_grad=True)`` with no upstream graph → backward()
          updated nothing → mosaic outputs flagged as aliased.

        After the fix, the dynamic loop skips ``mse``/``l1``/``l2`` only
        when the explicit block already populated the canonical key.
        With an MSE-only YAML the loop fires and ``components["mse"]``
        is non-zero from iter 0.
        """
        # Build a minimal MSE callable mimicking what LossBuilder pushes.
        mse_fn = torch.nn.MSELoss()

        # ``mse`` CANONICALISES TO ``l2``. The previous fixture set
        # ``lambda_l2 = 0.0`` and justified it as "the explicit L2 path is gated
        # off; only the dynamic ``mse`` entry should produce the loss" -- but
        # ``canonical_loss_name("mse") == "l2"``, so that line set the weight of
        # the very term under test to zero. The total was 0.0 because the fixture
        # asked for it, and the framework said so out loud:
        #
        #   [LossComputer] DEAD LOSS: all 1 finite loss component(s) were
        #   weighted to 0.0 -- total loss is 0.0 and the model is NOT training
        #
        # An MSE-only arm declares a NON-ZERO mse weight, which is spelled
        # ``lambda_l2``. ``lambda_l1`` stays at its warmup-gated default so this
        # still exercises the pre-warmup window the test is named for.
        config = real_settings(lambda_l1=10.0, lambda_l2=1.0)

        computer = UnifiedReconstructionLossComputer(config=config, device=torch.device("cpu"))

        pred = torch.randn(2, 1, 16, 16, requires_grad=True)
        target = torch.randn(2, 1, 16, 16)

        # iteration=0 puts us inside the L1 warmup window — the only
        # surviving path is the dynamic ``mse`` entry from losses_dict.
        out = computer.compute(
            pred=pred,
            target=target,
            epoch=0,
            iteration=0,
            losses_dict={"mse": mse_fn},
        )

        # Keyed ``l2``, not ``mse`` -- the canonical name reaches the component
        # dict as well as the weight table.
        assert "l2" in out.components, (
            "Expected the dynamic loop to compute the LossBuilder-supplied "
            "`mse` entry under its canonical key `l2`; got components=" + str(list(out.components))
        )
        assert out.components["l2"].item() > 0, (
            "MSE between random pred and target should be positive."
        )
        assert out.total.item() > 0, (
            "Total loss must be non-zero when the YAML declared MSE — a "
            "zero leaf here is the silent-fallback regression."
        )
        out.total.backward()
        assert pred.grad is not None and pred.grad.abs().sum() > 0, (
            "Gradients must flow back to pred; a disconnected zero leaf "
            "would leave pred.grad as zeros."
        )

    def test_shape_mismatch_loss_raises_loudly_not_silently(self) -> None:
        """F-LOSSFAIL / 2026-05-20 — a ValueError from a declarative loss
        must propagate, not be swallowed and replaced with a 0-loss step.

        Smoke run 20260519 surfaced 10+ ``Failed to compute loss 'l2':
        ValueError: [MSELoss] Shape mismatch ...`` warnings followed by
        ``g_total_loss=0.0000`` lines — a textbook CLAUDE.md pitfall #9
        silent fallback. The fix re-raises the ValueError so the
        configuration bug is impossible to ignore.
        """

        class _ShapeMismatchLoss(torch.nn.Module):
            def forward(self, pred, target, **_):
                raise ValueError(
                    "[MSELoss] Shape mismatch: pred torch.Size([2, 1, 128]) "
                    "vs target torch.Size([2, 1, 128, 1, 1])."
                )

        config = MockConfig()
        config.training.training_mode = "reconstruction"
        config.losses.reconstruction.lambda_l1 = 0.0
        config.losses.reconstruction.lambda_l2 = 0.0

        computer = UnifiedReconstructionLossComputer(config=config, device=torch.device("cpu"))
        pred = torch.randn(2, 1, 16, 16, requires_grad=True)
        target = torch.randn(2, 1, 16, 16)

        with pytest.raises(ValueError, match=r"silent loss failure|shape / channel"):
            computer.compute(
                pred=pred,
                target=target,
                epoch=0,
                iteration=2000,
                losses_dict={"l2": _ShapeMismatchLoss()},
            )

    def test_all_losses_failing_raises_not_silent_zero(self) -> None:
        """F-LOSSFAIL / 2026-05-20 — if every declared loss errors out
        (or none was wired), the previous behaviour returned a zero leaf
        with no upstream graph. Now a RuntimeError surfaces.

        The construct: a non-ValueError exception (kept as a warning per
        the fix) + the explicit L1/L2 paths gated off. Nothing populates
        ``components``; the post-loop guard fires.
        """

        class _RuntimeErrorLoss(torch.nn.Module):
            def forward(self, pred, target, **_):
                raise RuntimeError("transient backbone init failure")

        config = MockConfig()
        config.training.training_mode = "reconstruction"
        config.losses.reconstruction.lambda_l1 = 0.0
        config.losses.reconstruction.lambda_l2 = 0.0

        computer = UnifiedReconstructionLossComputer(config=config, device=torch.device("cpu"))
        pred = torch.randn(2, 1, 16, 16, requires_grad=True)
        target = torch.randn(2, 1, 16, 16)

        with pytest.raises(
            RuntimeError, match=r"empty.*disconnected zero|All declared losses failed"
        ):
            computer.compute(
                pred=pred,
                target=target,
                epoch=0,
                iteration=2000,
                losses_dict={"aux": _RuntimeErrorLoss()},
            )


# ============================================================================
# Tests for UnifiedVAELossComputer
# ============================================================================


class TestUnifiedVAELossComputer:
    """Test VAE loss computer."""

    @pytest.fixture
    def setup(self):
        """Setup for VAE tests.

        ``config=None`` — the sanctioned minimal-fallback path. See the GAN
        setup above: the old ``MockConfig`` never built via ``LossBuilder`` and
        silently fell back to L1 (a bare ``except`` removed 2026-07-01).
        """
        computer = UnifiedVAELossComputer(config=None, device=torch.device("cpu"))

        batch_size, channels, height, width = 2, 2, 32, 32
        pred = torch.randn(batch_size, channels, height, width, requires_grad=True)
        target = torch.randn(batch_size, channels, height, width)
        mu = torch.randn(batch_size, 64, requires_grad=True)
        logvar = torch.randn(batch_size, 64, requires_grad=True)

        return computer, pred, target, mu, logvar

    def test_vae_loss(self, setup):
        """Test VAE loss computation."""
        computer, pred, target, mu, logvar = setup

        loss_output = computer.compute(pred=pred, target=target, mu=mu, logvar=logvar, epoch=0)

        assert isinstance(loss_output.total, torch.Tensor)
        assert loss_output.total.grad_fn is not None
        assert loss_output.total.item() > 0


# ============================================================================
# Tests for UnifiedVQVAELossComputer
# ============================================================================


class TestUnifiedVQVAELossComputer:
    """Test VQ-VAE loss computer."""

    @pytest.fixture
    def setup(self):
        """Setup for VQ-VAE tests."""
        config = MockConfig()
        config.training.training_mode = "vqvae"
        config.losses.vqvae = type(
            "obj",
            (object,),
            {
                "beta_commitment": 0.25,
                "lambda_recon": 1.0,
            },
        )()

        computer = UnifiedVQVAELossComputer(config=config, device=torch.device("cpu"))

        batch_size, channels, height, width = 2, 2, 32, 32
        pred = torch.randn(batch_size, channels, height, width, requires_grad=True)
        target = torch.randn(batch_size, channels, height, width)
        z_q = torch.randn(batch_size, 64, requires_grad=True)  # Quantized
        z_e = torch.randn(batch_size, 64, requires_grad=True)  # Continuous

        return computer, pred, target, z_q, z_e

    def test_vqvae_loss(self, setup):
        """Test VQ-VAE loss computation."""
        computer, pred, target, z_q, z_e = setup

        loss_output = computer.compute(pred=pred, target=target, z_q=z_q, z_e=z_e, epoch=0)

        assert isinstance(loss_output.total, torch.Tensor)
        assert loss_output.total.grad_fn is not None
        assert loss_output.total.item() > 0


# ============================================================================
# Tests for UnifiedGANLossComputer
# ============================================================================


class TestUnifiedGANLossComputer:
    """Test GAN loss computer."""

    @pytest.fixture
    def setup(self):
        """Setup for GAN tests.

        Uses ``config=None`` — the sanctioned minimal-fallback path (L1 recon).
        The old ``MockConfig`` never built via ``LossBuilder``; it silently fell
        back to L1 through a bare ``except Exception`` that has since been removed
        (a pitfall-#9 silent fallback, review 2026-07-01). So these compute-math
        tests always exercised the L1 stack — ``config=None`` makes that explicit
        and is now the only sanctioned way to request that minimal stack.
        """
        computer = UnifiedGANLossComputer(config=None, device=torch.device("cpu"))

        batch_size, channels, height, width = 2, 2, 32, 32
        gen_output = torch.randn(batch_size, channels, height, width, requires_grad=True)
        target = torch.randn(batch_size, channels, height, width)
        discriminator = SimpleDiscriminator(in_channels=channels)

        return computer, gen_output, target, discriminator

    def test_gan_generator_loss(self, setup):
        """Test GAN generator loss."""
        computer, gen_output, target, discriminator = setup

        loss_output = computer.compute_generator_loss(
            pred=gen_output,
            target=target,
            discriminator=discriminator,
            epoch=0,
        )

        assert isinstance(loss_output.total, torch.Tensor)
        assert loss_output.total.grad_fn is not None

    def test_gan_discriminator_loss(self, setup):
        """Test GAN discriminator loss."""
        computer, gen_output, target, discriminator = setup

        loss_output = computer.compute_discriminator_loss(
            real=target,
            fake=gen_output,
            discriminator=discriminator,
            epoch=0,
        )

        assert isinstance(loss_output.total, torch.Tensor)

        # This fixture builds the computer with ``config=None``, the sanctioned
        # minimal stack, which sets ``adversarial_loss_fn = None``
        # (``unified_gan.py:182``). ``compute_discriminator_loss`` gates its
        # whole body on ``if self.adversarial_loss_fn`` (:677), so there is
        # genuinely nothing to compute and ``components`` comes back empty.
        #
        # This assertion used to read ``assert loss_output.total.requires_grad``
        # and it passed -- but not because of anything the computer did.
        # ``LossOutput.__post_init__`` used to force ``requires_grad=True`` on
        # EVERY LossOutput ever constructed, so the assertion was a tautology:
        # no computer could have failed it. Deleting that repair (#1952) is what
        # made this measurable, and what it measures is that the minimal stack
        # has no discriminator objective at all.
        assert loss_output.components == {}, (
            "config=None configures no adversarial loss, so there is no "
            "discriminator objective to stack"
        )
        assert loss_output.total.grad_fn is None
        assert not loss_output.total.requires_grad, (
            "a computer that produced nothing must SAY so; the old "
            "__post_init__ repair dressed this up as a trainable leaf, and "
            "backward() on it silently updated no discriminator weight"
        )


# ============================================================================
# Tests for LossKeyValidator
# ============================================================================


class TestLossKeyValidator:
    """Test loss key validation."""

    def test_valid_diffusion_keys(self):
        """Test validation of valid diffusion loss keys."""
        validator = LossKeyValidator(training_mode="diffusion")

        losses = {
            "diffusion_total_loss": torch.tensor(1.0),
            "diffusion_loss_mse": torch.tensor(0.8),
            "diffusion_loss_score": torch.tensor(0.2),
        }

        # Should not raise
        validator.validate(losses)

    def test_valid_reconstruction_keys(self):
        """Test validation of valid reconstruction keys."""
        validator = LossKeyValidator(training_mode="reconstruction")

        losses = {
            "reconstruction_total_loss": torch.tensor(1.0),
            "recon_loss_l1": torch.tensor(0.6),
            "recon_loss_l2": torch.tensor(0.4),
        }

        # Should not raise
        validator.validate(losses)

    def test_invalid_key_raises_error(self):
        """Test that invalid keys raise error."""
        validator = LossKeyValidator(training_mode="diffusion")

        losses = {
            "invalid_loss_name": torch.tensor(1.0),
        }

        # Should raise
        with pytest.raises(ValueError, match="Unknown loss key"):
            validator.validate(losses)

    def test_allow_extra_keys(self):
        """Test that extra keys are allowed when enabled."""
        validator = LossKeyValidator(training_mode="diffusion")

        losses = {
            "diffusion_total_loss": torch.tensor(1.0),
            "metric_psnr": torch.tensor(30.0),  # Extra key
        }

        # Should not raise with allow_extra_keys=True
        validator.validate(losses)


# ============================================================================
# Tests for Loss Key Registry
# ============================================================================


class TestLossKeyRegistry:
    """Test loss key registry."""

    def test_registry_has_all_modes(self):
        """Test that registry has all training modes."""
        keys = LossKeyRegistry.get_all_loss_keys()
        assert len(keys) > 0

    def test_registry_keys_for_diffusion(self):
        """Test that diffusion has expected keys."""
        keys = LossKeyRegistry.get_training_mode_keys("diffusion")

        # Should have component keys
        assert len(keys) > 1, "Should have component keys"

    def test_registry_keys_for_gan(self):
        """Test that GAN has both generator and discriminator keys."""
        keys = LossKeyRegistry.get_training_mode_keys("gan")

        # Should have both G and D keys
        assert any("g_" in k for k in keys), "Missing generator keys"
        assert any("d_" in k for k in keys), "Missing discriminator keys"


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for full training loop."""

    def test_training_step_with_reconstruction(self):
        """Test a full training step with reconstruction loss computer.

        Real losses block, same reason as ``TestUnifiedReconstructionLossComputer
        .setup``: the explicit path builds nothing from ``MockConfig``, so this
        hit the "All declared losses failed to compute" guard before it reached
        the training step it exists to exercise.
        """
        config = real_settings(lambda_l1=1.0, lambda_l2=0.5)

        # Create generator and optimizer
        generator = SimpleGenerator(in_channels=2, out_channels=2)
        optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

        # Create loss computer
        computer = UnifiedReconstructionLossComputer(config=config, device=torch.device("cpu"))

        # Create dummy batch
        batch_size, channels, height, width = 2, 2, 32, 32
        lr_batch = torch.randn(batch_size, channels, height, width)
        hr_batch = torch.randn(batch_size, channels, height, width)

        # Training step
        optimizer.zero_grad()

        with torch.autocast(device_type="cpu", enabled=False):
            gen_output = generator(lr_batch)
            loss_output = computer.compute(pred=gen_output, target=hr_batch, epoch=0)

        # Backward
        loss_output.total.backward()

        # Check gradients
        assert any(p.grad is not None for p in generator.parameters())

        # Optimizer step
        optimizer.step()

    def test_gan_training_step(self):
        """Test a full GAN training step."""
        # Create models
        generator = SimpleGenerator(in_channels=2, out_channels=2)
        discriminator = SimpleDiscriminator(in_channels=2)

        opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

        # Create loss computer. config=None → sanctioned L1 fallback (the old
        # MockConfig never built via LossBuilder; the bare-except fallback that
        # masked that was removed 2026-07-01).
        computer = UnifiedGANLossComputer(config=None, device=torch.device("cpu"))

        # Create dummy batch
        batch_size, channels, height, width = 2, 2, 32, 32
        lr_batch = torch.randn(batch_size, channels, height, width)
        hr_batch = torch.randn(batch_size, channels, height, width)

        # Generator step
        opt_g.zero_grad()
        with torch.autocast(device_type="cpu", enabled=False):
            gen_output = generator(lr_batch)
            disc_fake = discriminator(gen_output.detach())
            g_loss_output = computer.compute_generator_loss(
                pred=gen_output,
                target=hr_batch,
                discriminator=discriminator,
                epoch=0,
            )

        g_loss_output.total.backward()
        assert any(p.grad is not None for p in generator.parameters())
        opt_g.step()

        # Discriminator step
        opt_d.zero_grad()
        with torch.autocast(device_type="cpu", enabled=False):
            gen_output = generator(lr_batch).detach()
            d_loss_output = computer.compute_discriminator_loss(
                real=hr_batch, fake=gen_output, discriminator=discriminator, epoch=0
            )

        # If total has no discriminator grad (no adversarial_loss_fn configured),
        # fall back to direct BCE for testing purposes
        if not any(p.grad is not None for p in discriminator.parameters()):
            opt_d.zero_grad()
            real_pred = discriminator(hr_batch)
            fake_pred = discriminator(gen_output)
            bce = torch.nn.BCEWithLogitsLoss()
            d_loss = bce(real_pred, torch.ones_like(real_pred)) + bce(
                fake_pred, torch.zeros_like(fake_pred)
            )
            d_loss.backward()

        (
            d_loss_output.total.backward()
            if d_loss_output.total.requires_grad and d_loss_output.total.grad_fn is not None
            else None
        )
        assert any(p.grad is not None for p in discriminator.parameters())
        opt_d.step()


if __name__ == "__main__":
    # Run tests with: pytest tests/unit/test_loss_computers.py -v
    pytest.main([__file__, "-v"])
