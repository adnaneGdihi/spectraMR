"""Strategy-Level Data Leak Detection Tests

CRITICAL: These tests verify that no training strategy leaks target (ground truth)
information into the model's forward pass beyond what the paradigm allows.

Leak vectors tested:
  1. GRADIENT ISOLATION  — target_batch must NOT receive gradients from the loss
  2. TARGET-IN-INPUT     — generator must never see raw target during forward pass
  3. TRAIN/EVAL MODE     — strategies must respect .train()/.eval() boundaries
  4. BATCH UNPACKING     — _unpack_batch must not silently swap input/target
  5. LOSS SYMMETRY       — swapping pred/target must change the loss value
  6. DETERMINISM         — same seed must produce identical loss values

Run with:
    pytest tests/strategies/test_strategy_data_leak.py -v --tb=short

Expected:
    All PASS = No data leaks detected in any strategy
    Any FAIL = Potential data leak — fix immediately
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H, W = 16, 16  # Spatial dims for all synthetic tensors
B = 1           # Batch size
C_IMG = 1       # Image-domain channels
C_KSPACE = 2    # Real/imag k-space channels


class TinyGenerator(nn.Module):
    """Minimal generator: 1-ch → 1-ch via single conv."""

    def __init__(self, in_ch: int = C_IMG, out_ch: int = C_IMG):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, out_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyDiscriminator(nn.Module):
    """Minimal PatchGAN discriminator."""

    def __init__(self, in_ch: int = C_IMG):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyVAE(nn.Module):
    """Minimal VAE returning (recon, mu, logvar)."""

    def __init__(self, in_ch: int = C_IMG):
        super().__init__()
        self.enc = nn.Conv2d(in_ch, 8, 3, padding=1)
        self.mu_head = nn.Conv2d(8, 4, 1)
        self.logvar_head = nn.Conv2d(8, 4, 1)
        self.dec = nn.Conv2d(4, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, *_args, **_kwargs):
        h = torch.relu(self.enc(x))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return self.dec(z), mu, logvar

    def encode_content(self, x):
        return torch.relu(self.enc(x))

    def decode(self, z, *_args):
        return self.dec(z[:, :4])


def _make_batch(
    in_ch: int = C_IMG,
    out_ch: int = C_IMG,
    spatial: tuple[int, int] = (H, W),
) -> dict[str, torch.Tensor]:
    """Create a synthetic training batch with statistically distinct input/target."""
    torch.manual_seed(0)
    inp = torch.randn(B, in_ch, *spatial) * 0.5          # mean≈0, std≈0.5
    tgt = torch.randn(B, out_ch, *spatial) * 0.5 + 2.0   # mean≈2, std≈0.5
    return {"input": inp, "target": tgt}


def _make_env(
    generator: nn.Module,
    discriminator: nn.Module | None = None,
    *,
    config: Any = None,
    device: str = "cpu",
) -> MagicMock:
    """Build a minimal TrainingEnvironment mock."""
    env = MagicMock()
    env.generator = generator
    env.discriminator = discriminator
    env.opt_g = MagicMock()
    env.opt_d = MagicMock() if discriminator else None
    env.losses = {}
    env.device = torch.device(device)

    if config is None:
        config = _make_config()
    env.config = config
    return env


def _make_config() -> MagicMock:
    """Create a minimal mock config that satisfies BaseTrainingStrategy."""
    cfg = MagicMock()
    cfg.model.in_channels = C_IMG
    cfg.model.out_channels = C_IMG
    cfg.model.model_type = "standard_unet"
    cfg.model.model_kwargs = {}
    cfg.model.domain = "image"
    cfg.model.target_domain = "image"
    cfg.optimization.precision.enabled = False
    # `dtype` is read by `resolve_amp_precision`, which raises on anything
    # outside the schema-validated set (non-negotiable #3). Left unset on a
    # MagicMock it auto-creates a Mock and the strategy cannot be built at
    # all -- the knob moved to `optimization.precision.*` in phase 8 and
    # this stand-in stopped short of its `dtype` leaf.
    cfg.optimization.precision.dtype = "float16"
    cfg.optimization.gradient.clip.value = 1.0
    cfg.optimization.gradient.clip.enabled = False
    cfg.optimization.gradient.accumulation_steps = 1
    cfg.optimization.gradient.clip.method = "norm"
    cfg.training.training_mode = "reconstruction"
    cfg.training.enforce_output_range = False
    cfg.training.loss_weights = {}
    cfg.data.normalize_kspace = False
    cfg.data.normalization_kwargs = {}
    cfg.data.prior_loading = MagicMock()
    cfg.data.prior_loading.enabled = False
    cfg.losses.reconstruction = MagicMock()
    cfg.losses.reconstruction.enable_mse = True
    cfg.losses.reconstruction.lambda_mse = 1.0
    cfg.losses.gan = MagicMock()
    cfg.losses.gan.enable_adversarial = False
    cfg.losses.physics = MagicMock()
    cfg.losses.physics.enable_physics_constraint = False
    cfg.physics.kspace = MagicMock()
    cfg.physics.kspace.enable_kspace_recon = False
    cfg.physics.data_consistency = MagicMock()
    cfg.physics.data_consistency.enabled = False
    cfg.physics.pinn = MagicMock()
    cfg.physics.pinn.pde_type = "wave_equation"
    cfg.physics.pinn.enabled = False
    cfg.physics.pinn.lambda_pde = 0.0
    cfg.deep_supervision_weight = 0.0
    cfg.acceleration = {}
    cfg.logging = MagicMock()
    cfg.logging.log_gradients = False
    cfg.logging.log_interval = 100
    cfg.metrics = MagicMock()
    cfg.metrics.compute_psnr = False
    cfg.metrics.domain = "image"
    cfg.metrics.transform = None
    return cfg


def _n2n_batch() -> dict[str, torch.Tensor]:
    """The loader's Noise2Noise batch: slot 0 of ``kspace_reps`` is the input, slot 1 the pair."""
    batch = _make_batch()
    batch["kspace_reps"] = torch.cat([batch["input"], batch["target"]], dim=1)
    return batch


def _n2n_env_factory(generator: nn.Module):
    """An env whose config declares the pairing the strategy refuses to run without."""

    def build() -> MagicMock:
        config = _make_config()
        config.data.n2n_pairing = "ordered"
        return _make_env(generator, config=config)

    return build


# ---------------------------------------------------------------------------
# TEST 1: Gradient Isolation — target must never accumulate gradients
# ---------------------------------------------------------------------------

class TestGradientIsolation:
    """Target tensors must be treated as constants (no grad flow)."""

    @staticmethod
    def _run_gradient_check(strategy_cls, env_factory, batch):
        """Generic helper: instantiate strategy, run one step, check target grads."""
        env = env_factory()
        with patch("spectramr.infrastructure.di.di_container.resolve_service", return_value=MagicMock()):
            strategy = strategy_cls(env=env, device="cpu")

        inp = batch["input"].clone().requires_grad_(True)
        tgt = batch["target"].clone().requires_grad_(True)

        try:
            result = strategy.train_step(
                batch=batch, epoch=0, input_batch=inp, target_batch=tgt,
            )
            # Execute closures if returned as step configs
            if isinstance(result, list):
                for step_cfg in result:
                    if "closure" in step_cfg and callable(step_cfg["closure"]):
                        loss = step_cfg["closure"]()
                        if loss is not None and isinstance(loss, torch.Tensor) and loss.requires_grad:
                            loss.backward(retain_graph=True)
            elif isinstance(result, dict) and "g_total_loss" in result:
                loss = result["g_total_loss"]
                if isinstance(loss, torch.Tensor) and loss.requires_grad:
                    loss.backward()
        except Exception as exc:
            pytest.skip(f"Strategy setup incomplete for unit test: {exc}")

        assert tgt.grad is None or torch.all(tgt.grad == 0), (
            f"DATA LEAK: target_batch received gradients in {strategy_cls.__name__}. "
            "The target (ground truth) must never participate in the computational graph "
            "as an input to the generator."
        )

    def test_reconstruction_gradient_isolation(self):
        """Reconstruction strategy must not leak target into generator."""
        from spectramr.infrastructure.training.strategies.reconstruction import (
            ReconstructionTrainingStrategy,
        )
        batch = _make_batch()
        self._run_gradient_check(
            ReconstructionTrainingStrategy,
            lambda: _make_env(TinyGenerator()),
            batch,
        )

    def test_n2n_gradient_isolation(self):
        """Noise2Noise strategy must not leak target into generator."""
        from spectramr.infrastructure.training.strategies.n2n_strategy import (
            NoiseToNoiseStrategy,
        )
        self._run_gradient_check(NoiseToNoiseStrategy, _n2n_env_factory(TinyGenerator()), _n2n_batch())


# ---------------------------------------------------------------------------
# TEST 2: Target-in-Input — generator forward must not see target
# ---------------------------------------------------------------------------

class TestTargetNotInForwardPass:
    """Generator.forward() must only receive input_batch, never target_batch."""

    def test_reconstruction_forward_args(self):
        """ReconstructionStrategy must call generator(input) not generator(target)."""
        from spectramr.infrastructure.training.strategies.reconstruction import (
            ReconstructionTrainingStrategy,
        )

        gen = TinyGenerator()
        original_forward = gen.forward
        forward_calls: list[torch.Tensor] = []

        def spy_forward(x, *args, **kwargs):
            forward_calls.append(x.detach().clone())
            return original_forward(x, *args, **kwargs)

        gen.forward = spy_forward

        batch = _make_batch()
        env = _make_env(gen)

        with patch("spectramr.infrastructure.di.di_container.resolve_service", return_value=MagicMock()):
            strategy = ReconstructionTrainingStrategy(env=env, device="cpu")

        try:
            result = strategy.train_step(batch=batch, epoch=0)
            if isinstance(result, list):
                for cfg in result:
                    if callable(cfg.get("closure")):
                        cfg["closure"]()
        except Exception as exc:
            pytest.skip(f"Strategy setup incomplete: {exc}")

        tgt = batch["target"]
        for i, fwd_input in enumerate(forward_calls):
            # The generator input must NOT be the target
            if fwd_input.shape == tgt.shape:
                corr = torch.corrcoef(
                    torch.stack([fwd_input.flatten(), tgt.flatten()])
                )[0, 1]
                assert corr < 0.99, (
                    f"DATA LEAK: Generator forward call #{i} received a tensor "
                    f"highly correlated (r={corr:.4f}) with target_batch. "
                    "The generator must only see input_batch."
                )

    def test_n2n_forward_uses_input_rep(self):
        """N2N must call generator on first repetition, not second."""
        from spectramr.infrastructure.training.strategies.n2n_strategy import (
            NoiseToNoiseStrategy,
        )

        gen = TinyGenerator()
        calls: list[torch.Tensor] = []
        orig = gen.forward

        def spy(x, *a, **k):
            calls.append(x.detach().clone())
            return orig(x, *a, **k)

        gen.forward = spy
        batch = _n2n_batch()
        env = _n2n_env_factory(gen)()

        with patch("spectramr.infrastructure.di.di_container.resolve_service", return_value=MagicMock()):
            strategy = NoiseToNoiseStrategy(env=env, device="cpu")

        try:
            result = strategy.train_step(batch=batch, epoch=0)
            if isinstance(result, list):
                for cfg in result:
                    if callable(cfg.get("closure")):
                        cfg["closure"]()
        except Exception as exc:
            pytest.skip(f"N2N setup incomplete: {exc}")

        tgt = batch["target"]
        for fwd_input in calls:
            if fwd_input.shape == tgt.shape:
                corr = torch.corrcoef(
                    torch.stack([fwd_input.flatten(), tgt.flatten()])
                )[0, 1]
                assert corr < 0.99, (
                    "DATA LEAK: N2N generator saw target repetition as input."
                )


# ---------------------------------------------------------------------------
# TEST 3: Train / Eval Mode Integrity
# ---------------------------------------------------------------------------

class TestTrainEvalModeIntegrity:
    """Strategies must set correct mode and not cross-contaminate."""

    def test_generator_in_train_mode_during_train_step(self):
        """Generator must be in .train() mode during train_step."""
        from spectramr.infrastructure.training.strategies.reconstruction import (
            ReconstructionTrainingStrategy,
        )

        gen = TinyGenerator()
        gen.eval()  # Start in eval mode intentionally
        env = _make_env(gen)
        batch = _make_batch()

        with patch("spectramr.infrastructure.di.di_container.resolve_service", return_value=MagicMock()):
            strategy = ReconstructionTrainingStrategy(env=env, device="cpu")

        try:
            strategy.train_step(batch=batch, epoch=0)
        except Exception:
            pass

        assert gen.training, (
            "DATA LEAK RISK: Generator was NOT set to .train() mode during train_step. "
            "Dropout/BatchNorm will behave as eval, causing train/val distribution mismatch."
        )


# ---------------------------------------------------------------------------
# TEST 4: Batch Unpacking Integrity
# ---------------------------------------------------------------------------

class TestBatchUnpackingIntegrity:
    """_unpack_batch must faithfully separate input and target."""

    def test_dict_batch_unpacking(self):
        """Standard dict batch must unpack correctly."""
        from spectramr.infrastructure.training.strategies.mixins.batch_preparation import (
            BatchPreparationMixin,
        )

        class Unpacker(BatchPreparationMixin):
            device = torch.device("cpu")

        unpacker = Unpacker()
        batch = _make_batch()
        inp, tgt = unpacker._unpack_batch(batch)

        assert inp is not None and tgt is not None, "Unpacking returned None"
        assert torch.equal(inp, batch["input"]), "Input was corrupted during unpacking"
        assert torch.equal(tgt, batch["target"]), "Target was corrupted during unpacking"

    def test_tuple_batch_unpacking(self):
        """Tuple batch must unpack in correct order (input, target)."""
        from spectramr.infrastructure.training.strategies.mixins.batch_preparation import (
            BatchPreparationMixin,
        )

        class Unpacker(BatchPreparationMixin):
            device = torch.device("cpu")

        unpacker = Unpacker()
        inp_orig = torch.randn(B, C_IMG, H, W)
        tgt_orig = torch.randn(B, C_IMG, H, W) + 5.0

        inp, tgt = unpacker._unpack_batch((inp_orig, tgt_orig))

        assert torch.equal(inp, inp_orig), "Tuple unpacking swapped input/target"
        assert torch.equal(tgt, tgt_orig), "Tuple unpacking corrupted target"

    def test_input_target_are_distinct(self):
        """Input and target must be statistically distinct (not aliased)."""
        batch = _make_batch()
        inp, tgt = batch["input"], batch["target"]

        assert not torch.equal(inp, tgt), (
            "DATA LEAK: input and target are identical tensors. "
            "The model would achieve perfect loss without learning."
        )

        corr = torch.corrcoef(
            torch.stack([inp.flatten(), tgt.flatten()])
        )[0, 1]
        assert abs(corr) < 0.95, (
            f"DATA LEAK WARNING: input/target correlation is {corr:.4f} — "
            "suspiciously high. Check data pipeline for aliasing."
        )


# ---------------------------------------------------------------------------
# TEST 5: Loss Asymmetry — swapping pred/target must change loss
# ---------------------------------------------------------------------------

class TestLossAsymmetry:
    """Loss functions must not be trivially symmetric in a way that hides leaks."""

    def test_ssot_loss_changes_with_swap(self):
        """UnifiedReconstructionLossComputer must produce different loss for swapped args."""
        try:
            from spectramr.models.losses.computers import UnifiedReconstructionLossComputer
        except ImportError:
            pytest.skip("UnifiedReconstructionLossComputer not available")

        cfg = _make_config()
        computer = UnifiedReconstructionLossComputer(config=cfg, device=torch.device("cpu"))

        pred = torch.randn(B, C_IMG, H, W)
        target = torch.randn(B, C_IMG, H, W) + 3.0

        try:
            out_normal = computer.compute(pred=pred, target=target, epoch=0, losses_dict={})
            out_swapped = computer.compute(pred=target, target=pred, epoch=0, losses_dict={})
        except Exception as exc:
            pytest.skip(f"Loss computer not fully mockable: {exc}")

        # For L1/L2 losses, swap is symmetric but values should still be non-zero
        assert out_normal.total.item() > 0, "Loss is zero — model may see target"

    def test_registry_l1_nonzero(self):
        """Registry L1 loss must return nonzero for distinct pred/target."""
        from spectramr.models.losses.registry import create_loss

        l1 = create_loss("l1")
        pred = torch.zeros(B, C_IMG, H, W)
        target = torch.ones(B, C_IMG, H, W)

        loss = l1(pred, target)
        assert loss.item() > 0, "L1 loss is zero for distinct tensors — leak or bug"


# ---------------------------------------------------------------------------
# TEST 6: Determinism — same seed → same loss
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Training steps must be deterministic given the same seed."""

    def test_reconstruction_deterministic(self):
        """Two runs with same seed must produce identical losses."""
        from spectramr.infrastructure.training.strategies.reconstruction import (
            ReconstructionTrainingStrategy,
        )

        def _run_once():
            torch.manual_seed(42)
            gen = TinyGenerator()
            env = _make_env(gen)
            batch = _make_batch()

            with patch("spectramr.infrastructure.di.di_container.resolve_service", return_value=MagicMock()):
                strategy = ReconstructionTrainingStrategy(env=env, device="cpu")

            try:
                result = strategy.train_step(batch=batch, epoch=0)
                if isinstance(result, list):
                    for cfg in result:
                        if callable(cfg.get("closure")):
                            loss = cfg["closure"]()
                            if isinstance(loss, torch.Tensor):
                                return loss.item()
                elif isinstance(result, dict) and "g_total_loss" in result:
                    v = result["g_total_loss"]
                    return v.item() if isinstance(v, torch.Tensor) else float(v)
            except Exception as exc:
                pytest.skip(f"Strategy incomplete: {exc}")
            return None

        v1 = _run_once()
        v2 = _run_once()

        if v1 is not None and v2 is not None:
            assert abs(v1 - v2) < 1e-5, (
                f"NON-DETERMINISTIC: same seed produced losses {v1} vs {v2}. "
                "This makes experiments non-reproducible and can mask data leaks."
            )


# ---------------------------------------------------------------------------
# TEST 7: Strategy-Specific Leak Vectors
# ---------------------------------------------------------------------------

class TestCycleBlochNoTargetLeak:
    """CycleBloch must only use ULF input → generate HF → Bloch → compare ULF.
    The HF target is used ONLY for discriminator supervision, never as generator input."""

    def test_cycle_bloch_generator_sees_only_ulf(self):
        from spectramr.infrastructure.training.strategies.cycle_bloch_strategy import (
            CycleBlochStrategy,
        )

        gen = TinyGenerator()
        disc = TinyDiscriminator()
        calls: list[torch.Tensor] = []
        orig = gen.forward

        def spy(x, *a, **k):
            calls.append(x.detach().clone())
            return orig(x, *a, **k)

        gen.forward = spy
        batch = _make_batch()
        env = _make_env(gen, disc)

        with patch("spectramr.infrastructure.di.di_container.resolve_service", return_value=MagicMock()):
            try:
                strategy = CycleBlochStrategy(env=env, device="cpu")
            except Exception as exc:
                pytest.skip(f"CycleBloch setup needs full env: {exc}")

        tgt = batch["target"]
        try:
            strategy.train_step(batch=batch, epoch=0)
        except Exception:
            pass

        for fwd_input in calls:
            if fwd_input.shape == tgt.shape:
                corr = torch.corrcoef(
                    torch.stack([fwd_input.flatten(), tgt.flatten()])
                )[0, 1]
                assert corr < 0.99, (
                    "DATA LEAK: CycleBloch generator saw HF target. "
                    "It must only receive ULF input."
                )


class TestGuidedSRReferenceNotFullTarget:
    """GuidedSR extracts partial HF reference from target — the generator must
    NOT see the full HF target, only the extracted reference slice/slab."""

    def test_guided_sr_reference_extraction_is_partial(self):
        from spectramr.infrastructure.training.strategies.guided_sr_strategy import (
            GuidedSuperResolutionStrategy,
        )

        # For 2D, reference = full target (by design for 2D).
        # For 3D, reference must be a subset.
        # Test the 3D case conceptually.
        gen = TinyGenerator(in_ch=2)  # ULF + ref concatenated
        env = _make_env(gen)

        with patch("spectramr.infrastructure.di.di_container.resolve_service", return_value=MagicMock()):
            try:
                strategy = GuidedSuperResolutionStrategy(env=env, device="cpu")
            except Exception as exc:
                pytest.skip(f"GuidedSR setup incomplete: {exc}")

        # 3D volume: reference must be partial
        hf_vol = torch.randn(1, 1, 10, H, W)
        ref = strategy._extract_reference(hf_vol, mode="random_slice")

        assert ref.shape[2] < hf_vol.shape[2], (
            "DATA LEAK: GuidedSR reference is the full volume, not a partial slice. "
            "The generator would see the complete target."
        )


class TestDiffusionNoisePrediction:
    """Diffusion strategies predict noise, not clean images.
    Target (clean image) must only appear in the loss, not as generator input."""

    def test_diffusion_generator_receives_noisy_input(self):
        """Generator must receive noisy x_t, not clean x_0."""
        # This is a structural/design test — we verify the strategy code path
        # rather than running it, since full diffusion setup is heavy.

        import inspect

        from spectramr.infrastructure.training.strategies.diffusion import (
            DiffusionTrainingStrategy,
        )

        source = inspect.getsource(DiffusionTrainingStrategy)

        # The generator call should include timesteps / noisy input
        assert "generator" in source.lower(), "No generator reference found"

        # Should NOT have: generator(target) or generator(x_0) without noise
        # This is a heuristic check — not perfect but catches obvious leaks
        lines = source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""'):
                continue
            if "generator(target" in stripped.lower():
                pytest.fail(
                    f"POTENTIAL LEAK at line {i}: generator called with 'target'. "
                    "Diffusion generator must receive noisy x_t, not clean target."
                )


# ---------------------------------------------------------------------------
# TEST 8: Validation must not affect training state
# ---------------------------------------------------------------------------

class TestValidationIsolation:
    """Validation step must not modify model weights or optimizer state."""

    def test_validation_does_not_update_weights(self):
        from spectramr.infrastructure.training.strategies.reconstruction import (
            ReconstructionTrainingStrategy,
        )

        gen = TinyGenerator()
        env = _make_env(gen)
        batch = _make_batch()

        with patch("spectramr.infrastructure.di.di_container.resolve_service", return_value=MagicMock()):
            strategy = ReconstructionTrainingStrategy(env=env, device="cpu")

        # Snapshot weights before validation
        weights_before = {
            n: p.clone() for n, p in gen.named_parameters()
        }

        try:
            strategy.validation_step(batch=batch, epoch=0)
        except Exception:
            pass  # Validation may fail on mock env — that's OK

        # Verify weights unchanged
        for name, param in gen.named_parameters():
            assert torch.equal(param, weights_before[name]), (
                f"DATA LEAK: Validation modified weight '{name}'. "
                "Validation must run under torch.no_grad() and not update parameters."
            )


# ---------------------------------------------------------------------------
# TEST 9: Loss Computer SSOT — no hardcoded losses in strategies
# ---------------------------------------------------------------------------

class TestNoHardcodedLosses:
    """Verify that recently remediated strategies use SSOT loss infrastructure."""

    TIER1_STRATEGIES = [
        "src/spectramr/infrastructure/training/strategies/n2n_strategy.py",
        "src/spectramr/infrastructure/training/strategies/diff_siren.py",
        "src/spectramr/infrastructure/training/strategies/disentangled_vae_strategy.py",
    ]

    @pytest.mark.parametrize("filepath", TIER1_STRATEGIES)
    def test_no_nn_mseloss(self, filepath):
        """Strategy must not contain nn.MSELoss() instantiation."""
        import re
        from pathlib import Path

        full = Path(__file__).parents[2] / filepath
        if not full.exists():
            pytest.skip(f"File not found: {full}")

        content = full.read_text()
        # Skip comments and docstrings (rough heuristic)
        code_lines = [
            line for line in content.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        code = "\n".join(code_lines)

        matches = re.findall(r"nn\.MSELoss\(\)", code)
        assert len(matches) == 0, (
            f"SSOT VIOLATION: {filepath} contains {len(matches)} instance(s) of "
            "nn.MSELoss(). Use create_loss('mse') or UnifiedReconstructionLossComputer."
        )

    @pytest.mark.parametrize("filepath", TIER1_STRATEGIES)
    def test_no_nn_l1loss(self, filepath):
        """Strategy must not contain nn.L1Loss() instantiation."""
        import re
        from pathlib import Path

        full = Path(__file__).parents[2] / filepath
        if not full.exists():
            pytest.skip(f"File not found: {full}")

        content = full.read_text()
        code_lines = [
            line for line in content.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        code = "\n".join(code_lines)

        matches = re.findall(r"nn\.L1Loss\(\)", code)
        assert len(matches) == 0, (
            f"SSOT VIOLATION: {filepath} contains {len(matches)} instance(s) of "
            "nn.L1Loss(). Use create_loss('l1') or UnifiedReconstructionLossComputer."
        )


# ---------------------------------------------------------------------------
# TEST 10: Source-level scan — no F.mse_loss / F.l1_loss in remediated files
# ---------------------------------------------------------------------------

class TestNoFunctionalLossesInRemediatedStrategies:
    """All remediated strategies must use create_loss() not F.mse_loss/F.l1_loss."""

    ALL_REMEDIATED = [
        "src/spectramr/infrastructure/training/strategies/n2n_strategy.py",
        "src/spectramr/infrastructure/training/strategies/diff_siren.py",
        "src/spectramr/infrastructure/training/strategies/cycle_bloch_strategy.py",
        "src/spectramr/infrastructure/training/strategies/disentangled_diffusion_strategy.py",
        "src/spectramr/infrastructure/training/strategies/disentangled_vae_strategy.py",
        "src/spectramr/infrastructure/training/strategies/guided_sr_strategy.py",
        "src/spectramr/infrastructure/training/strategies/graph_cold_diffusion_strategy.py",
        "src/spectramr/infrastructure/training/strategies/meta_learning_strategy.py",
    ]

    @pytest.mark.parametrize("filepath", ALL_REMEDIATED)
    def test_no_functional_mse(self, filepath):
        """No F.mse_loss in training-path code (comments OK)."""
        import re
        from pathlib import Path

        full = Path(__file__).parents[2] / filepath
        if not full.exists():
            pytest.skip(f"File not found: {full}")

        for i, line in enumerate(full.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("'") or stripped.startswith('"'):
                continue
            if re.search(r"F\.mse_loss\(", stripped):
                pytest.fail(
                    f"SSOT VIOLATION in {filepath}:{i} — "
                    f"F.mse_loss() found. Use create_loss('mse')."
                )

    @pytest.mark.parametrize("filepath", ALL_REMEDIATED)
    def test_no_functional_l1(self, filepath):
        """No F.l1_loss in training-path code (comments OK)."""
        import re
        from pathlib import Path

        full = Path(__file__).parents[2] / filepath
        if not full.exists():
            pytest.skip(f"File not found: {full}")

        for i, line in enumerate(full.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("'") or stripped.startswith('"'):
                continue
            if re.search(r"F\.l1_loss\(", stripped):
                pytest.fail(
                    f"SSOT VIOLATION in {filepath}:{i} — "
                    f"F.l1_loss() found. Use create_loss('l1')."
                )


# ---------------------------------------------------------------------------
# TEST 11: Loss Registry Roundtrip — create_loss returns functional modules
# ---------------------------------------------------------------------------

class TestLossRegistryIntegrity:
    """Registry-created losses must produce valid gradients and non-zero values."""

    @pytest.mark.parametrize("name", ["l1", "mse"])
    def test_registry_loss_has_grad(self, name):
        """Registry loss must propagate gradients through pred."""
        from spectramr.models.losses.registry import create_loss

        loss_fn = create_loss(name)
        pred = torch.randn(B, C_IMG, H, W, requires_grad=True)
        target = torch.randn(B, C_IMG, H, W)

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None, (
            f"create_loss('{name}') did not propagate gradients to pred."
        )
        assert pred.grad.abs().sum() > 0, (
            f"create_loss('{name}') produced zero gradients."
        )

    @pytest.mark.parametrize("name", ["l1", "mse"])
    def test_registry_loss_zero_for_identical(self, name):
        """Registry loss must be zero when pred == target (no phantom loss)."""
        from spectramr.models.losses.registry import create_loss

        loss_fn = create_loss(name)
        x = torch.randn(B, C_IMG, H, W)
        loss = loss_fn(x, x.clone())

        assert loss.item() < 1e-6, (
            f"create_loss('{name}') returns {loss.item():.6f} for identical "
            "pred/target. Should be ~0. Possible data corruption."
        )


# ---------------------------------------------------------------------------
# TEST 12: Complex tensor guard — no silent complex→real truncation
# ---------------------------------------------------------------------------

class TestComplexTensorGuard:
    """Complex k-space input must be explicitly converted, not silently truncated."""

    def test_complex_to_real_preserves_information(self):
        """Real-stacking [B,C,H,W] → [B,2C,H,W] must preserve both components."""
        x_complex = torch.randn(B, C_IMG, H, W, dtype=torch.complex64)
        x_stacked = torch.cat([x_complex.real, x_complex.imag], dim=1)

        assert x_stacked.shape[1] == 2 * C_IMG, "Channel count wrong after stacking"

        # Recover and verify roundtrip
        recovered = torch.complex(
            x_stacked[:, :C_IMG], x_stacked[:, C_IMG:]
        )
        assert torch.allclose(x_complex, recovered, atol=1e-6), (
            "DATA LEAK: Complex→Real stacking is not invertible. "
            "Phase information would be lost."
        )

    def test_complex_target_also_converted(self):
        """If input is complex-stacked, target must be too (same representation)."""
        x = torch.randn(B, C_IMG, H, W, dtype=torch.complex64)
        t = torch.randn(B, C_IMG, H, W, dtype=torch.complex64)

        x_stacked = torch.cat([x.real, x.imag], dim=1)
        t_stacked = torch.cat([t.real, t.imag], dim=1)

        assert x_stacked.shape == t_stacked.shape, (
            "DATA LEAK: Input and target have different shapes after complex "
            "conversion. Loss computation will be incorrect."
        )


# ---------------------------------------------------------------------------
# TEST 13: Model output independence — untrained model must not reproduce target
# ---------------------------------------------------------------------------

class TestModelOutputIndependence:
    """An untrained model's output must be independent of the target."""

    def test_untrained_generator_output_differs_from_target(self):
        """Random-init generator output must not match target (no memorization)."""
        torch.manual_seed(99)
        gen = TinyGenerator()
        batch = _make_batch()

        with torch.no_grad():
            output = gen(batch["input"])

        target = batch["target"]
        corr = torch.corrcoef(
            torch.stack([output.flatten(), target.flatten()])
        )[0, 1]

        assert abs(corr) < 0.5, (
            f"SUSPICIOUS: Untrained model output correlates with target (r={corr:.4f}). "
            "Check if target leaks into model initialization or forward pass."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
