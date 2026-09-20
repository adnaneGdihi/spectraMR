from itertools import pairwise
from typing import ClassVar

"""Tests for KSpaceUndersamplingProcess.

Validates physics-informed cold diffusion degradation operator.
"""

import pytest
import torch

from spectramr.models.diffusion.kspace_process import (
    KSpaceUndersamplingProcess,
    PhysicsInformedColdDiffusion,
)


class TestKSpaceUndersamplingProcess:
    """Unit tests for KSpaceUndersamplingProcess."""

    @pytest.fixture
    def process(self) -> KSpaceUndersamplingProcess:
        return KSpaceUndersamplingProcess(
            num_timesteps=100,
            max_acceleration=8.0,
            center_fraction=0.08,
            mask_type="variable_density",
            seed=42,
        )

    def test_init(self, process: KSpaceUndersamplingProcess) -> None:
        """Test process initialization."""
        assert process.num_timesteps == 100
        assert process.max_accel == 8.0
        assert process.center_fraction == 0.08

    def test_q_sample_shape(self, process: KSpaceUndersamplingProcess) -> None:
        """Test q_sample returns correct shapes."""
        x_start = torch.randn(2, 2, 64, 64)  # [B, C, H, W]
        t = torch.tensor([50, 75])

        x_t, masks = process.q_sample(x_start, t)

        assert x_t.shape == x_start.shape
        assert masks.shape == (2, 1, 64, 64)

    def test_q_sample_full_sampling_at_t0(
        self, process: KSpaceUndersamplingProcess
    ) -> None:
        """At t=0, should have full sampling (R=1)."""
        x_start = torch.randn(1, 2, 64, 64)
        t = torch.tensor([0])

        x_t, masks = process.q_sample(x_start, t)

        # At t=0, mask should be all 1s (full sampling)
        assert masks.sum() == masks.numel(), "t=0 should have full sampling"
        assert torch.allclose(x_t, x_start), "t=0 output should equal input"

    def test_q_sample_max_undersampling_at_tmax(
        self, process: KSpaceUndersamplingProcess
    ) -> None:
        """At t=T-1, should have maximum undersampling."""
        x_start = torch.randn(1, 2, 64, 64)
        t = torch.tensor([99])  # T-1 for 100 timesteps

        x_t, masks = process.q_sample(x_start, t)

        # At t=T-1, mask should be sparse (R=max_accel)
        expected_fraction = 1.0 / process.max_accel
        actual_fraction = masks.sum() / masks.numel()

        # Allow some tolerance due to center fraction
        assert (
            actual_fraction < 0.3
        ), f"t=T-1 should have sparse sampling, got {actual_fraction}"

    def test_q_sample_monotonic_degradation(
        self, process: KSpaceUndersamplingProcess
    ) -> None:
        """Higher t should have sparser masks (more degradation)."""
        x_start = torch.randn(1, 2, 64, 64)

        fractions = []
        for t_val in [0, 25, 50, 75, 99]:
            t = torch.tensor([t_val])
            _, masks = process.q_sample(x_start, t)
            fractions.append((masks.sum() / masks.numel()).item())

        # Fractions should decrease with t
        for i in range(len(fractions) - 1):
            assert (
                fractions[i] >= fractions[i + 1]
            ), f"Sampling fraction should decrease: {fractions}"

    def test_center_always_sampled(self, process: KSpaceUndersamplingProcess) -> None:
        """Center frequencies (ACS) should always be sampled."""
        x_start = torch.randn(1, 2, 64, 64)
        t = torch.tensor([99])  # Maximum undersampling

        _, masks = process.q_sample(x_start, t)

        # Check center region
        # masks is shape [B, 1, H, W]
        # VariableDensityKSpaceAccelerator samples a 2D center patch in 2D sampling mode
        H, W = 64, 64
        num_rows = H
        num_cols = W

        # Calculate exactly the way `_apply_center_patch` calculates the center bounds
        center_h_size = max(1, int(num_rows * process.center_fraction))
        center_w_size = max(1, int(num_cols * process.center_fraction))
        start_h = (num_rows - center_h_size) // 2
        start_w = (num_cols - center_w_size) // 2
        end_h = start_h + center_h_size
        end_w = start_w + center_w_size

        # Extract the exact center square
        center_region = masks[0, 0, start_h:end_h, start_w:end_w]
        assert (
            center_region.float().min() == 1.0
        ), f"Center frequencies must always be sampled, got min {center_region.float().min()}"

    def test_acceleration_schedule(self, process: KSpaceUndersamplingProcess) -> None:
        """Test acceleration schedule is correct."""
        schedule = process.get_acceleration_schedule()

        assert len(schedule) == process.num_timesteps
        assert schedule[0] == 1.0, "R(0) should be 1"
        assert abs(schedule[-1] - process.max_accel) < 0.01, "R(T) should be max_accel"

    def test_mask_determinism_with_seed(self) -> None:
        """Test that same seed produces same masks."""
        process1 = KSpaceUndersamplingProcess(
            num_timesteps=100, max_acceleration=4.0, seed=42
        )
        process2 = KSpaceUndersamplingProcess(
            num_timesteps=100, max_acceleration=4.0, seed=42
        )

        x = torch.randn(1, 2, 32, 32)
        t = torch.tensor([50])

        _, mask1 = process1.q_sample(x, t)
        _, mask2 = process2.q_sample(x, t)

        assert torch.equal(mask1, mask2), "Same seed should produce same masks"

    def test_base_acceleration_stored(self) -> None:
        """Test that base_acceleration is stored when provided."""
        process = KSpaceUndersamplingProcess(
            num_timesteps=100,
            max_acceleration=8.0,
            base_acceleration=2.0,
            seed=42,
        )
        assert process.base_acceleration == 2.0

    def test_base_acceleration_default(self) -> None:
        """Test that base_acceleration defaults to 1.0."""
        process = KSpaceUndersamplingProcess(
            num_timesteps=100,
            max_acceleration=8.0,
            seed=42,
        )
        assert process.base_acceleration == 1.0

    def test_base_acceleration_causes_undersampling_at_t0(self) -> None:
        """Regression: base_acceleration>1 should degrade even at t=0.

        This test guards against the identity collapse bug where
        base_acceleration was not forwarded, defaulting to 1.0 and
        producing full sampling at low timesteps.
        """
        process = KSpaceUndersamplingProcess(
            num_timesteps=100,
            max_acceleration=8.0,
            base_acceleration=2.0,
            center_fraction=0.08,
            mask_type="uniform_cartesian",
            seed=42,
        )

        x_start = torch.randn(1, 2, 64, 64)
        t = torch.tensor([0])  # t=0 with base_acceleration=2.0

        x_t, masks = process.q_sample(x_start, t)

        # With base_acceleration=2.0, even t=0 should have ~50% sampling
        actual_fraction = (masks.sum() / masks.numel()).item()
        assert actual_fraction < 0.75, (
            f"base_acceleration=2.0 at t=0 should produce ~50% sampling, "
            f"got {actual_fraction * 100:.1f}%"
        )


class TestPriorChannelRange:
    """Pin the cross-contrast prior behaviour: q_sample must keep the
    declared channel range fully sampled while still degrading the rest.

    Motivated by ``experiment_cross_contrast_kspace_diffusion.yaml`` where
    the M4Raw loader concatenates ``[T1 || T2/FLAIR]`` along the coil axis
    and the metadata claims "T1 fully sampled prior". Without this
    parameter the broadcast mask multiplied every channel uniformly and
    T1 was as undersampled as T2.
    """

    def test_invalid_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            KSpaceUndersamplingProcess(num_timesteps=100, prior_channel_range=(5, 5))
        with pytest.raises(ValueError):
            KSpaceUndersamplingProcess(num_timesteps=100, prior_channel_range=(-1, 4))

    def test_out_of_bounds_raises_in_q_sample(self) -> None:
        proc = KSpaceUndersamplingProcess(
            num_timesteps=100,
            prior_channel_range=(0, 10),
        )
        with pytest.raises(ValueError):
            proc.q_sample(torch.randn(1, 4, 16, 16), torch.tensor([50]))

    def test_prior_channels_unchanged_at_max_t(self) -> None:
        """Even at t=T-1 (max undersampling) the prior channels must
        equal the input exactly — only the non-prior channels are
        degraded."""
        proc = KSpaceUndersamplingProcess(
            num_timesteps=100,
            max_acceleration=8.0,
            center_fraction=0.08,
            mask_type="variable_density",
            seed=42,
            # 8-channel tensor: first 4 = T1 (prior), last 4 = T2 (target).
            prior_channel_range=(0, 4),
        )
        x_start = torch.randn(2, 8, 32, 32)
        t = torch.tensor([99, 99])

        x_t, mask = proc.q_sample(x_start, t)

        # Prior channels: bit-exact preservation.
        assert torch.equal(x_t[:, :4], x_start[:, :4])
        # Non-prior channels: should be at least partially zeroed by the
        # mask (max acceleration, so most rows are dropped).
        non_prior_changed = not torch.allclose(x_t[:, 4:], x_start[:, 4:])
        assert (
            non_prior_changed
        ), "Non-prior channels must be degraded at max undersampling"
        # Mask shape unchanged.
        assert mask.shape == (2, 1, 32, 32)

    def test_no_prior_range_is_legacy_behaviour(self) -> None:
        """Without ``prior_channel_range`` every channel is degraded
        uniformly — i.e. behaviour is identical to before the patch."""
        proc_legacy = KSpaceUndersamplingProcess(
            num_timesteps=100,
            max_acceleration=8.0,
            center_fraction=0.08,
            mask_type="variable_density",
            seed=42,
        )
        x_start = torch.randn(1, 4, 32, 32)
        t = torch.tensor([99])
        x_t, _ = proc_legacy.q_sample(x_start, t)
        # With acceleration > 1 the broadcast mask zeros most rows on
        # every channel — pin that none of the channels is bit-exact to
        # the input (any of the 4 may differ).
        any_degraded = not torch.equal(x_t, x_start)
        assert any_degraded, "Legacy q_sample must still degrade x_start"


class TestPhysicsInformedColdDiffusion:
    """Unit tests for PhysicsInformedColdDiffusion."""

    @pytest.fixture
    def mock_model(self):
        """Simple identity model for testing."""

        class MockModel(torch.nn.Module):
            def forward(self, x, t):
                return x  # Identity

        return MockModel()

    @pytest.fixture
    def diffusion(self, mock_model) -> PhysicsInformedColdDiffusion:
        return PhysicsInformedColdDiffusion(
            model=mock_model,
            num_timesteps=10,
            max_acceleration=4.0,
            center_fraction=0.08,
            dc_method="hard",
            kspace_log_scaled=False,
        )

    def test_p_sample_hard_dc(self, diffusion: PhysicsInformedColdDiffusion) -> None:
        """Test hard data consistency is enforced."""
        B, C, H, W = 2, 2, 32, 32
        measurement = torch.randn(B, C, H, W)
        mask = torch.zeros(B, 1, H, W)
        mask[:, :, :, 10:20] = 1.0  # Some sampled columns

        x_t = torch.randn(B, C, H, W)
        # Use t=0 so we get final output without q_sample degradation
        t = torch.tensor([0, 0])

        x_out = diffusion.p_sample(x_t, t, measurement, mask)

        # Where mask=1, output should equal measurement (hard DC at t=0)
        mask_expanded = mask.expand_as(x_out)
        assert torch.allclose(
            x_out * mask_expanded, measurement * mask_expanded
        ), "Hard DC should preserve measured values"

    def test_sample_loop(self, diffusion: PhysicsInformedColdDiffusion) -> None:
        """Test full sampling loop completes."""
        measurement = torch.randn(1, 2, 32, 32)
        mask = torch.ones(1, 1, 32, 32)
        mask[:, :, :, ::4] = 0  # 4x undersampling

        x_out = diffusion.sample(measurement, mask)

        assert x_out.shape == measurement.shape
        # With hard DC and identity model, output should equal measurement
        # (since model returns input and DC replaces with measurement)

    def test_unknown_dc_method_raises_at_construction(self, mock_model) -> None:
        """An unimplemented dc_method must fail at BUILD, not mid-validation.

        Validated against the physics SSOT ``VALID_DC_METHODS`` in ``__init__``
        so a bad value fails loud at load (pitfall #9/#15), never silently mid
        reverse-diffusion.
        """
        with pytest.raises(ValueError, match="dc_method"):
            PhysicsInformedColdDiffusion(
                model=mock_model,
                num_timesteps=10,
                max_acceleration=4.0,
                center_fraction=0.08,
                dc_method="bogus",
                kspace_log_scaled=False,
            )

    def test_p_sample_adaptive_delegates_to_model_dc_layer(self, mock_model) -> None:
        """dc_method='adaptive' must delegate to the model's learned dc_layer
        (the same operator it trained with), not raise.

        Regression for the kspace_filling kan_ablation_cnn_adc /
        legacy_dual_domain validation crash: the generator built
        ``AdaptiveDataConsistency`` but the sampler only knew hard/soft and
        raised ``Unknown dc_method 'adaptive'`` on the first validation sample.
        """
        from spectramr.infrastructure.physics.data_consistency import (
            AdaptiveDataConsistency,
        )

        mock_model.dc_layer = AdaptiveDataConsistency()
        diff = PhysicsInformedColdDiffusion(
            model=mock_model,
            num_timesteps=10,
            max_acceleration=4.0,
            center_fraction=0.08,
            dc_method="adaptive",
            kspace_log_scaled=False,
        )
        B, C, H, W = 2, 2, 32, 32
        x_t = torch.randn(B, C, H, W)
        t = torch.tensor([0, 0])
        measurement = torch.randn(B, C, H, W)
        mask = torch.zeros(B, 1, H, W)
        mask[:, :, :, 10:20] = 1.0

        x_out = diff.p_sample(x_t, t, measurement, mask)  # must not raise
        assert x_out.shape == x_t.shape
        assert torch.isfinite(x_out).all()

    def test_p_sample_learned_dc_method_without_dc_layer_raises(
        self, mock_model
    ) -> None:
        """A learned dc_method with no ``model.dc_layer`` to delegate to must
        fail loudly (pitfall #9), never silently skip DC."""
        diff = PhysicsInformedColdDiffusion(
            model=mock_model,  # bare identity model, no dc_layer attribute
            num_timesteps=10,
            max_acceleration=4.0,
            center_fraction=0.08,
            dc_method="adaptive",
            kspace_log_scaled=False,
        )
        x_t = torch.randn(2, 2, 32, 32)
        t = torch.tensor([0, 0])
        measurement = torch.randn(2, 2, 32, 32)
        mask = torch.zeros(2, 1, 32, 32)
        mask[:, :, :, 10:20] = 1.0
        with pytest.raises(ValueError, match="dc_layer"):
            diff.p_sample(x_t, t, measurement, mask)


class TestSamplerScheduleReuse:
    """Regression: the reverse-process degradation MUST use the same schedule
    the model was trained on.

    ``PhysicsInformedColdDiffusion`` previously built a *fresh*
    ``KSpaceUndersamplingProcess`` from the handful of scalars passed to it,
    falling back to library defaults for everything else: ``schedule_type
    ="linear"``, ``base_acceleration=1.0``, ``mask_type="variable_density"``.
    But the generator trains with a YAML-configured process (e.g. the
    experiment_11 step schedule, ``base_acceleration=2.0``, equispaced masks,
    7-bucket ``acceleration_range``). So the iterative sampler restored the
    model from masks it never saw at each timestep — a silent train/sample
    degradation-manifold desync that made multi-step ("true") cold diffusion
    operate on the wrong manifold (experiment_11 fix, 2026-06-09).

    The generator already exposes the correctly-configured ``kspace_process``;
    the sampler must reuse it as the SSOT.
    """

    def _model_with_process(self) -> torch.nn.Module:
        class GenLike(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Distinctive NON-default schedule (mirrors the exp_11 YAML).
                self.kspace_process = KSpaceUndersamplingProcess(
                    num_timesteps=28,
                    max_acceleration=32.0,
                    base_acceleration=2.0,
                    center_fraction=0.08,
                    mask_type="uniform_cartesian",
                    schedule_type="step",
                    schedule_kwargs={
                        "acceleration_range": [2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0]
                    },
                    seed=42,
                )

            def forward(self, x, t):
                return x

        return GenLike()

    def test_reuses_model_kspace_process_as_ssot(self) -> None:
        """The sampler reuses the model's configured process, not a fresh default."""
        model = self._model_with_process()
        diffusion = PhysicsInformedColdDiffusion(
            model=model,
            num_timesteps=28,
            max_acceleration=32.0,
            center_fraction=0.08,
            kspace_log_scaled=False,
        )
        # SSOT identity: same object, so the reverse trajectory degrades with
        # exactly the schedule the model trained against.
        assert diffusion.process is model.kspace_process
        # ...and therefore carries the training schedule, NOT the linear/base-1
        # variable-density default the bug produced.
        assert diffusion.process.base_acceleration == 2.0
        assert diffusion.process.mask_type == "uniform_cartesian"

    def test_reused_process_degrades_identically_to_training(self) -> None:
        """The sampler's per-timestep mask must match the model's training mask."""
        model = self._model_with_process()
        diffusion = PhysicsInformedColdDiffusion(
            model=model,
            num_timesteps=28,
            max_acceleration=32.0,
            center_fraction=0.08,
            kspace_log_scaled=False,
        )
        x = torch.randn(1, 2, 64, 64)
        for t_val in (2, 10, 25):
            t = torch.tensor([t_val])
            _, train_mask = model.kspace_process.q_sample(x, t)
            _, sampler_mask = diffusion.process.q_sample(x, t)
            assert torch.equal(
                train_mask, sampler_mask
            ), f"sampler mask at t={t_val} must match the training mask"

    def test_falls_back_to_constructed_process_without_model_attr(self) -> None:
        """A model exposing no ``kspace_process`` keeps the legacy construct path."""

        class Bare(torch.nn.Module):
            def forward(self, x, t):
                return x

        diffusion = PhysicsInformedColdDiffusion(
            model=Bare(),
            num_timesteps=10,
            max_acceleration=4.0,
            center_fraction=0.08,
            kspace_log_scaled=False,
        )
        assert isinstance(diffusion.process, KSpaceUndersamplingProcess)
        assert diffusion.process.max_accel == 4.0


class TestDynamicMask:
    """``enable_dynamic_mask`` was a dead façade knob — declared in
    ``AccelerationConfigSchema`` and set in every kspace_filling YAML, but read
    *nowhere* in ``src/`` (pitfall #15). So the cold-diffusion model trained on a
    SINGLE undersampling pattern per acceleration level: at R=2 every sample and
    every call produced the byte-identical mask.

    Wired 2026-06-09: when ``enable_dynamic_mask`` is on AND the process is in
    training mode, ``q_sample`` draws a fresh accelerator seed per sample, so the
    model sees many distinct patterns per R. The acceleration FRACTION is
    unchanged (still ``R(t)``) — only WHICH lines are kept varies. Gated on
    ``self.training`` so validation keeps the fixed seed and stays reproducible.
    """

    def _proc(self, dynamic: bool) -> KSpaceUndersamplingProcess:
        return KSpaceUndersamplingProcess(
            num_timesteps=28,
            max_acceleration=32.0,
            base_acceleration=2.0,
            center_fraction=0.08,
            mask_type="equispaced",
            schedule_type="step",
            schedule_kwargs={
                "acceleration_range": [2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0]
            },
            seed=42,
            enable_dynamic_mask=dynamic,
        )

    @staticmethod
    def _n_distinct(masks: torch.Tensor) -> int:
        return int(torch.unique(masks.reshape(masks.shape[0], -1), dim=0).shape[0])

    def test_off_is_one_mask_per_acceleration(self) -> None:
        """Default (knob off): all samples at R=2 get the identical mask (the bug)."""
        proc = self._proc(dynamic=False).train()
        x = torch.randn(8, 4, 256, 256)
        t = torch.full((8,), 2, dtype=torch.long)  # R=2 bucket centre
        _, masks = proc.q_sample(x, t)
        assert self._n_distinct(masks) == 1

    def test_on_varies_pattern_per_sample_in_training(self) -> None:
        """Knob on + training: samples at the same R get DIFFERENT patterns."""
        proc = self._proc(dynamic=True).train()
        x = torch.randn(8, 4, 256, 256)
        t = torch.full((8,), 2, dtype=torch.long)
        _, masks = proc.q_sample(x, t)
        assert self._n_distinct(masks) > 1

    def test_on_preserves_acceleration_fraction(self) -> None:
        """Randomising the pattern must NOT change how many lines are kept."""
        proc = self._proc(dynamic=True).train()
        x = torch.randn(8, 4, 256, 256)
        t = torch.full((8,), 10, dtype=torch.long)  # R=8 bucket
        _, masks = proc.q_sample(x, t)
        counts = masks.reshape(8, -1).sum(dim=1)
        assert (
            int(counts.unique().numel()) == 1
        ), f"all samples at the same R must keep the same #lines, got {counts.tolist()}"

    def test_on_is_deterministic_in_eval(self) -> None:
        """Eval mode keeps the fixed seed → reproducible masks (validation-safe)."""
        proc = self._proc(dynamic=True).eval()
        x = torch.randn(8, 4, 256, 256)
        t = torch.full((8,), 2, dtype=torch.long)
        _, masks = proc.q_sample(x, t)
        assert self._n_distinct(masks) == 1


class TestApplyDataConsistency:
    """KSpaceUndersamplingProcess.apply_data_consistency soft/hard/raise.

    Regression for the byte-identical soft==hard branch (a silent no-op that
    ignored ``lambda_weight`` — pitfall #15) and the missing raise-on-unknown
    (pitfall #9).
    """

    @pytest.fixture
    def process(self) -> KSpaceUndersamplingProcess:
        return KSpaceUndersamplingProcess(num_timesteps=10, seed=0)

    def test_hard_replaces_measured_bins(
        self, process: KSpaceUndersamplingProcess
    ) -> None:
        pred = torch.zeros(1, 2, 8, 8)
        meas = torch.ones(1, 2, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., :4, :] = 1.0  # sample the top half
        out = process.apply_data_consistency(pred, meas, mask, method="hard")
        assert torch.allclose(out[..., :4, :], torch.ones(1, 2, 4, 8))
        assert torch.allclose(out[..., 4:, :], torch.zeros(1, 2, 4, 8))

    def test_soft_is_not_hard_and_uses_lambda(
        self, process: KSpaceUndersamplingProcess
    ) -> None:
        """Soft DC must produce the closed-form blend, NOT a hard replacement."""
        pred = torch.zeros(1, 2, 8, 8)
        meas = torch.ones(1, 2, 8, 8)
        mask = torch.ones(1, 1, 8, 8)  # fully sampled isolates the blend
        hard = process.apply_data_consistency(pred, meas, mask, method="hard")
        soft = process.apply_data_consistency(
            pred, meas, mask, method="soft", lambda_weight=0.5
        )
        # closed form at m=1: (0 + 0.5*1)/(1 + 0.5) = 1/3, NOT hard's 1.0
        assert not torch.allclose(soft, hard)
        assert torch.allclose(soft, torch.full_like(soft, 1.0 / 3.0), atol=1e-6)

    def test_soft_lambda_controls_blend(
        self, process: KSpaceUndersamplingProcess
    ) -> None:
        pred = torch.zeros(1, 2, 4, 4)
        meas = torch.ones(1, 2, 4, 4)
        mask = torch.ones(1, 1, 4, 4)
        small = process.apply_data_consistency(
            pred, meas, mask, method="soft", lambda_weight=0.1
        )
        large = process.apply_data_consistency(
            pred, meas, mask, method="soft", lambda_weight=10.0
        )
        # larger lambda -> closer to the measurement (1.0)
        assert large.mean() > small.mean()
        assert large.mean() > 0.9  # 10/11 ≈ 0.909

    def test_unknown_method_raises(self, process: KSpaceUndersamplingProcess) -> None:
        pred = torch.zeros(1, 2, 4, 4)
        meas = torch.ones(1, 2, 4, 4)
        mask = torch.ones(1, 1, 4, 4)
        with pytest.raises(ValueError, match="Unknown data-consistency method"):
            process.apply_data_consistency(pred, meas, mask, method="bogus")


class TestReverseModeReplaceFreeze:
    """The corrected ``replace_freeze`` reverse process vs legacy ``additive``.

    Root cause (2026-07-11): the ``additive`` loop ``x += x0*(M_{t-1}-M_t)`` with
    an unbounded head, fed its own growing partial sum, blows the k-space
    magnitude up at inference (never at single-step training). ``replace_freeze``
    writes each revealed coefficient once (hard-DC'd, magnitude-bounded) and
    reveals from the NEXT SCHEDULED level, which provably bounds the output.
    """

    @pytest.fixture
    def huge_model(self):
        """A head that emits large constants regardless of input.

        Stands in for the unbounded ``complex_unet`` head drifting off-range on
        out-of-distribution multi-step states.
        """

        class HugeModel(torch.nn.Module):
            def forward(self, x, t):
                return torch.full_like(x, 100.0)

        return HugeModel()

    @staticmethod
    def _make(model, reverse_mode: str, reverse_clip_ratio: float = 4.0):
        return PhysicsInformedColdDiffusion(
            model=model,
            num_timesteps=10,
            max_acceleration=4.0,
            center_fraction=0.08,
            dc_method="hard",
            reverse_mode=reverse_mode,
            reverse_clip_ratio=reverse_clip_ratio,
            kspace_log_scaled=False,
        )

    def test_default_reverse_mode_is_additive(self, huge_model) -> None:
        diff = PhysicsInformedColdDiffusion(model=huge_model, num_timesteps=10, kspace_log_scaled=False)
        assert diff.reverse_mode == "additive"

    def test_unknown_reverse_mode_raises_at_construction(self, huge_model) -> None:
        with pytest.raises(ValueError, match="reverse_mode"):
            self._make(huge_model, reverse_mode="bogus")

    def test_nonpositive_clip_ratio_raises(self, huge_model) -> None:
        with pytest.raises(ValueError, match="reverse_clip_ratio"):
            self._make(
                huge_model, reverse_mode="replace_freeze", reverse_clip_ratio=0.0
            )

    def test_replace_freeze_bounds_output_but_additive_blows_up(
        self, huge_model
    ) -> None:
        """The discriminating A/B: same unbounded head, same inputs.

        ``additive`` lets the output exceed the physical measured scale;
        ``replace_freeze`` caps every written magnitude at ``ratio·max|observed|``.
        """
        torch.manual_seed(0)
        b, c, h, w = 1, 2, 32, 32
        measurement = torch.randn(b, c, h, w)
        mask = torch.zeros(b, 1, h, w)
        mask[:, :, :, 12:20] = 1.0  # a measured band

        ratio = 4.0
        obs = (mask > 0).float()
        ceil = ratio * float((measurement.abs() * obs).amax())

        add_out = self._make(huge_model, "additive").sample(measurement, mask)
        rf_out = self._make(huge_model, "replace_freeze", ratio).sample(
            measurement, mask
        )

        # additive: unbounded head reaches ~100 >> the measured scale.
        assert float(add_out.abs().max()) > ceil
        # replace_freeze: never exceeds the fixed per-sample ceiling.
        assert float(rf_out.abs().max()) <= ceil + 1e-4

    def test_replace_freeze_preserves_observed_support(self, huge_model) -> None:
        """Observed coefficients stay pinned to the measurement (exact, idempotent)."""
        torch.manual_seed(1)
        measurement = torch.randn(1, 2, 32, 32)
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, :, 8:16] = 1.0

        out = self._make(huge_model, "replace_freeze").sample(measurement, mask)
        m = mask.expand_as(out)
        assert torch.allclose(out * m, measurement * m, atol=1e-5)

    def test_replace_freeze_full_coverage_no_holes(self) -> None:
        """Every coefficient ends committed (observed or infilled) — no zero holes.

        Guards the strided-schedule ``t-1`` skip that would otherwise leave
        interior bands unrevealed.
        """
        torch.manual_seed(2)

        class ConstModel(torch.nn.Module):
            def forward(self, x, t):
                return torch.full_like(x, 3.0)

        diff = self._make(ConstModel(), "replace_freeze")
        measurement = torch.randn(1, 2, 24, 24)
        mask = torch.zeros(1, 1, 24, 24)
        mask[:, :, :, 10:14] = 1.0

        out = diff.sample(measurement, mask)
        # Non-observed entries were all filled from the (nonzero) prediction, so
        # essentially no coefficient remains exactly zero.
        frac_zero = float((out.abs() < 1e-8).float().mean())
        assert frac_zero < 0.01, f"coverage holes: {frac_zero:.3f} of entries are zero"

    class _NestedCenterProcess(KSpaceUndersamplingProcess):
        """Deterministic nested center-block masks that CAP at 50% at t=0.

        Reproduces ``base_acceleration=2.0``: the least-degraded (t=0) mask keeps
        only the central half of k-space, and every higher-t mask is a strict
        subset (nested). Overriding ``q_sample`` sidesteps the stochastic mask
        generator so the terminal-reveal coverage bug is deterministic.
        """

        def q_sample(self, x_start, t, noise=None):  # type: ignore[override]
            t0 = int(t.reshape(-1)[0].item())
            w = x_start.shape[-1]
            frac = 0.5 * (1.0 - t0 / max(self.num_timesteps - 1, 1)) + 1.0 / w
            keep = max(1, min(w, round(frac * w)))
            lo = (w - keep) // 2
            mask = torch.zeros(
                x_start.shape[0], 1, *x_start.shape[2:], device=x_start.device
            )
            mask[..., lo : lo + keep] = 1.0
            return x_start * mask, mask

    @pytest.mark.parametrize("reverse_mode", ["replace_freeze", "replace_freeze_dc"])
    def test_terminal_reveal_fills_full_kspace_under_base_acceleration_gt1(
        self, reverse_mode: str
    ) -> None:
        """Regression (experiment_11 'DC blob', 2026-07-21): the FINAL reveal step
        must fill ALL of k-space even when the tail mask covers < full support.

        Pre-fix the terminal reveal used ``q_sample(x0, t=0)`` == the
        base-acceleration mask (support ``1/base_acceleration``). With
        ``base_acceleration=2.0`` that left ~50% of k-space permanently zero — a
        hard low-pass that capped even a perfect (oracle) denoiser at ~20 dB. The
        fix reveals every remaining line from the model's own prediction (never
        the measurement), so coverage is complete and no ground truth leaks.
        """
        torch.manual_seed(7)

        from spectramr.infrastructure.physics.data_consistency import (
            SoftDataConsistency,
        )

        class ModelBase2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.kspace_process = TestReverseModeReplaceFreeze._NestedCenterProcess(
                    num_timesteps=10,
                    max_acceleration=8.0,
                    base_acceleration=2.0,
                    center_fraction=0.08,
                )
                # ``dc_method='soft'`` (the ``replace_freeze_dc`` leg below) now
                # delegates to the model's trained dc_layer, mirroring what
                # ``KSpaceColdDiffusionGenerator`` builds for that method.
                self.dc_layer = SoftDataConsistency(lambda_init=0.5)

            def forward(self, x, t):
                return torch.full_like(x, 3.0)

        diff = PhysicsInformedColdDiffusion(
            model=ModelBase2(),
            num_timesteps=10,
            max_acceleration=8.0,
            center_fraction=0.08,
            dc_method="soft" if reverse_mode == "replace_freeze_dc" else "hard",
            dc_weight=0.5,
            reverse_mode=reverse_mode,
            reverse_clip_ratio=1.3,
            kspace_log_scaled=False,
        )
        # Anchor the bug condition: the least-degraded (t=0) mask keeps only ~half
        # of k-space, so a terminal reveal that stopped there leaves holes.
        _, base_mask = diff.process.q_sample(
            torch.randn(1, 2, 32, 32), torch.zeros(1, dtype=torch.long)
        )
        assert 0.4 < float(base_mask.float().mean()) < 0.6, "t=0 mask must cap ~50%"

        # A genuinely UNDERsampled measurement: exactly zero off the observed
        # band, so unrevealed k-space stays zero and ``frac_zero`` measures the
        # reveal's coverage (a dense measurement would mask the holes).
        measurement, mask = self._band(h=32, w=32, cols=(14, 18))
        measurement = measurement * mask
        out = diff.sample(measurement, mask)
        frac_zero = float((out.abs() < 1e-8).float().mean())
        assert frac_zero < 0.02, (
            f"{reverse_mode}: terminal reveal left {frac_zero:.3f} of k-space zero "
            "(base_acceleration=2.0 low-pass 'DC blob' not fixed)"
        )

    # ---- replace_freeze_dc: observed lines HONOR dc_method (denoise) ----------

    @staticmethod
    def _make_dc(
        model,
        reverse_mode: str,
        dc_method: str,
        dc_weight: float = 1.0,
        reverse_clip_ratio: float = 4.0,
    ):
        return PhysicsInformedColdDiffusion(
            model=model,
            num_timesteps=10,
            max_acceleration=4.0,
            center_fraction=0.08,
            dc_method=dc_method,
            dc_weight=dc_weight,
            reverse_mode=reverse_mode,
            reverse_clip_ratio=reverse_clip_ratio,
            kspace_log_scaled=False,
        )

    @staticmethod
    def _band(b=1, c=2, h=32, w=32, cols=(12, 20)):
        measurement = torch.randn(b, c, h, w)
        mask = torch.zeros(b, 1, h, w)
        mask[:, :, :, cols[0] : cols[1]] = 1.0
        return measurement, mask

    def test_replace_freeze_dc_is_a_valid_mode(self, huge_model) -> None:
        diff = self._make_dc(huge_model, "replace_freeze_dc", "hard")
        assert diff.reverse_mode == "replace_freeze_dc"

    def test_replace_freeze_dc_hard_matches_freeze(self, huge_model) -> None:
        """``dc_method='hard'`` under replace_freeze_dc pins observed == measurement."""
        torch.manual_seed(3)
        measurement, mask = self._band()
        out = self._make_dc(huge_model, "replace_freeze_dc", "hard").sample(
            measurement, mask
        )
        m = mask.expand_as(out)
        assert torch.allclose(out * m, measurement * m, atol=1e-5)

    @staticmethod
    def _const_model_with_soft_dc_layer(value: float = 3.0, lambda_init: float = 0.5):
        """A constant-output model carrying the trained ``SoftDataConsistency``
        layer the generator builds for ``dc_method in ('soft', 'noise_adjusted')``
        (``kspace_cold_diffusion_generator.py:2738-2739``). Reproduces that wiring
        so the sampler's delegation path has a real layer to reach.
        """
        from spectramr.infrastructure.physics.data_consistency import (
            SoftDataConsistency,
        )

        class ConstModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dc_layer = SoftDataConsistency(lambda_init=lambda_init)

            def forward(self, x, t):
                return torch.full_like(x, value)

        return ConstModel()

    def test_replace_freeze_dc_soft_denoises_observed(self) -> None:
        """``soft`` blends the observed lines through the trained ``dc_layer``
        (the proximal blend ``(x0 + lam*y)/(1+lam)``), not a hand-rolled convex
        blend that discards the layer's ``lambda_param``.
        """
        torch.manual_seed(4)
        model = self._const_model_with_soft_dc_layer(value=3.0, lambda_init=0.5)
        measurement, mask = self._band()
        out = self._make_dc(model, "replace_freeze_dc", "soft", 0.5).sample(
            measurement, mask
        )
        m = mask.expand_as(out)
        # Observed lines are the proximal blend (3 + 0.5*y)/1.5, NOT the raw
        # measurement y and NOT the old convex blend 1.5 + 0.5*y.
        assert not torch.allclose(out * m, measurement * m, atol=1e-3)
        old_buggy_formula = (1.5 + 0.5 * measurement) * m
        assert not torch.allclose(out * m, old_buggy_formula, atol=1e-3)
        expected_obs = ((3.0 + 0.5 * measurement) / 1.5) * m
        assert torch.allclose(out * m, expected_obs, atol=1e-4)

    def test_replace_freeze_dc_soft_reads_trained_lambda(self) -> None:
        """Moving the trained ``lambda_param`` must change the sampled output.

        Before this fix ``dc_weight`` was read as a fixed convex-blend fraction
        and ``lambda_param`` was never consulted at sampling, so a live
        (optimizer-moved) lambda had zero effect on the reconstruction.
        """
        torch.manual_seed(9)
        measurement, mask = self._band()

        low = self._const_model_with_soft_dc_layer(value=3.0, lambda_init=0.5)
        out_low = self._make_dc(low, "replace_freeze_dc", "soft", 0.5).sample(
            measurement, mask
        )

        high = self._const_model_with_soft_dc_layer(value=3.0, lambda_init=0.5)
        with torch.no_grad():
            high.dc_layer.lambda_param.fill_(7.0)
        out_high = self._make_dc(high, "replace_freeze_dc", "soft", 0.5).sample(
            measurement, mask
        )

        m = mask.expand_as(out_low)
        assert not torch.allclose(out_low * m, out_high * m, atol=1e-4)

    def test_replace_freeze_dc_noise_adjusted_matches_soft(self) -> None:
        """``noise_adjusted`` is documented as an alias onto the SAME
        ``SoftDataConsistency`` branch (``dc_settings.py``), so it must produce
        the identical output given the identical trained layer.
        """
        torch.manual_seed(10)
        measurement, mask = self._band()

        soft_model = self._const_model_with_soft_dc_layer(value=3.0, lambda_init=0.5)
        soft_out = self._make_dc(soft_model, "replace_freeze_dc", "soft", 0.5).sample(
            measurement, mask
        )

        na_model = self._const_model_with_soft_dc_layer(value=3.0, lambda_init=0.5)
        na_out = self._make_dc(
            na_model, "replace_freeze_dc", "noise_adjusted", 0.5
        ).sample(measurement, mask)

        assert torch.allclose(soft_out, na_out, atol=1e-6)

    def test_replace_freeze_dc_soft_without_dc_layer_raises(self, huge_model) -> None:
        """``soft`` now delegates like every other non-hard method, so a model
        with no trained ``dc_layer`` must fail loudly rather than silently fall
        back to a hand-rolled blend (pitfall #9)."""
        diff = self._make_dc(huge_model, "replace_freeze_dc", "soft", 0.5)
        measurement, mask = self._band()
        with pytest.raises(ValueError, match="dc_layer"):
            diff.sample(measurement, mask)

    def test_replace_freeze_dc_bounds_output_with_huge_model(self) -> None:
        """The magnitude clamp still bounds output when the observed DC is soft."""
        torch.manual_seed(5)
        measurement, mask = self._band()
        ratio = 4.0
        obs = (mask > 0).float()
        ceil = ratio * float((measurement.abs() * obs).amax())
        model = self._const_model_with_soft_dc_layer(value=100.0, lambda_init=0.5)
        out = self._make_dc(model, "replace_freeze_dc", "soft", 0.5, ratio).sample(
            measurement, mask
        )
        assert float(out.abs().max()) <= ceil + 1e-4

    def test_replace_freeze_dc_learned_without_dc_layer_raises(
        self, huge_model
    ) -> None:
        """A learned dc_method needs the model's dc_layer to delegate to."""
        diff = self._make_dc(huge_model, "replace_freeze_dc", "noise_adaptive")
        measurement, mask = self._band()
        with pytest.raises(ValueError, match="dc_layer"):
            diff.sample(measurement, mask)

    def test_replace_freeze_dc_noise_adaptive_denoises(self) -> None:
        """``noise_adaptive`` routes to the model's dc_layer and denoises observed."""
        from spectramr.infrastructure.physics.data_consistency import (
            NoiseAdaptiveDataConsistency,
        )

        torch.manual_seed(6)

        class ModelWithDC(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dc_layer = NoiseAdaptiveDataConsistency(beta=1.0)

            def forward(self, x, t):
                return torch.full_like(x, 3.0)

        measurement, mask = self._band()
        out = self._make_dc(
            ModelWithDC(), "replace_freeze_dc", "noise_adaptive"
        ).sample(measurement, mask)
        m = mask.expand_as(out)
        assert torch.isfinite(out).all()
        # Wiener trust blends observed lines away from the raw measurement.
        assert not torch.allclose(out * m, measurement * m, atol=1e-2)


class TestSoftDCReachesTheGeneratorsOwnDCLayer:
    """Non-negotiable 16: the findings-3/6 fix is real only if the PRODUCTION
    call chain reaches the generator's own trained ``dc_layer`` -- not just the
    hand-built stand-ins ``TestReverseModeReplaceFreeze`` constructs directly.

    ``KSpaceColdDiffusionGenerator.sample()`` passes ``model=self`` to the
    sampler it builds (kspace_cold_diffusion_generator.py:4161), so
    ``self.model.dc_layer`` inside ``_apply_observed_dc`` IS the generator's
    own ``SoftDataConsistency`` — verified end to end here.
    """

    @staticmethod
    def _generator(**over):
        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            KSpaceColdDiffusionGenerator,
        )

        return KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            base_channels=8,
            num_res_blocks=1,
            backbone_type="complex_unet",
            attention_type="none",
            timesteps=8,
            sampling_steps=4,
            dc_method="soft",
            dc_weight=0.5,
            reverse_sampling_mode="replace_freeze_dc",
            kspace_log_scaled=False,
            force_pure_kspace=True,
            condition_with_smaps=False,
            acceleration_type="equispaced",
            base_acceleration=1.0,
            max_acceleration=4.0,
            center_fraction=0.08,
            **over,
        )

    @staticmethod
    def _measurement():
        measurement = torch.randn(1, 2, 32, 32)
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, :, 10:20] = 1.0
        return measurement, mask

    def test_soft_dc_sample_does_not_raise(self):
        """Confirms the delegation lands on the generator's own constructed
        dc_layer (not None): a wrong wiring here raises inside sample()."""
        gen = self._generator()
        measurement, mask = self._measurement()
        out = gen.sample(measurement, mask=mask, inference_timesteps=4)
        assert torch.isfinite(out).all()

    def test_soft_dc_trained_lambda_changes_the_generators_own_sample_output(self):
        """Moving the SAME dc_layer the generator trains with must change what
        ``sample()`` returns."""
        torch.manual_seed(11)
        gen = self._generator()
        measurement, mask = self._measurement()

        out_low = gen.sample(measurement, mask=mask, inference_timesteps=4)
        with torch.no_grad():
            gen.dc_layer.lambda_param.fill_(7.0)
        out_high = gen.sample(measurement, mask=mask, inference_timesteps=4)

        assert not torch.allclose(out_low, out_high, atol=1e-4)


class TestReverseInputLabelPairing:
    """#2067 step 2 (finding 4): does an EXECUTED call's actual input support
    match the degradation its own ``t`` label declares?

    ``replace_freeze_dc`` is self-consistent by construction: the reveal at
    step ``i`` targets ``schedule[i+1]``, so step ``i``'s input -- written by
    step ``i-1``'s reveal, which targeted ``schedule[i]`` -- matches step
    ``i``'s own label. ``CURRENT_STEP_KEYED_REVERSE_MODES``
    (``replace_freeze_dc_t0``) keys the reveal off ``schedule[i]`` instead
    (kspace_process.py:2104), which lags EVERY executed call's input one
    schedule index behind its label -- worst at the terminal call, labelled
    ``0``, whose input is measurably not fully sampled.

    BLOCKED, not fixed here: the two candidate repairs both break a
    ``test_reverse_stats.py`` assertion outside this file's scope --
    (a) keying off ``schedule[i+1]`` unconditionally makes ``_t0`` byte-
    identical to the default mode, so the terminal call is skipped again
    (breaks ``test_t0_keying_calls_the_model_at_the_terminal_rung``); (b)
    keeping the default reveal and forcing the terminal call to run anyway
    adds one extra model call (breaks ``test_t0_keying_costs_no_extra_model_
    calls``) and, under ``dc_method='hard'``, that forced call is a
    structural no-op (empty reveal, ``own == obs``). Planted as a strict
    xfail so a real fix is provable and a regression that widens the lag is
    caught immediately.
    """

    @staticmethod
    def _ladder(**over):
        return KSpaceUndersamplingProcess(
            num_timesteps=29,
            max_acceleration=32.0,
            base_acceleration=1.0,
            mask_type="variable_density",
            seed=42,
            train_identity_rung=True,
            device="cpu",
            **over,
        )

    class _InputSupportRecorder(torch.nn.Module):
        """Records the OBSERVED support of ``x`` (this call's actual input),
        keyed by the ``t`` label the call was made with."""

        def __init__(self, process):
            super().__init__()
            self.kspace_process = process
            self.calls: list[tuple[int, torch.Tensor]] = []

        def forward(self, x, t):
            from spectramr.models.diffusion.kspace_process import paired_magnitude

            support = (paired_magnitude(x) > 1e-8).float()
            self.calls.append((int(t[0]), support.clone()))
            return torch.ones_like(x)

    @pytest.mark.parametrize(
        "mode",
        [
            "replace_freeze_dc",
            pytest.param(
                "replace_freeze_dc_t0",
                marks=pytest.mark.xfail(
                    strict=True,
                    reason=(
                        "finding 4 / #2067 step 2: CURRENT_STEP_KEYED_REVERSE_MODES "
                        "keys the reveal off schedule[i] instead of schedule[i+1], so "
                        "every executed call's input support lags its own label by "
                        "one schedule index (worst at the terminal t=0 call, whose "
                        "input is not fully sampled). Structural tension against "
                        "test_reverse_stats.py's pinned head-skip/equal-call-count "
                        "contract -- see the task report for the two candidate "
                        "fixes and which pinned assertion each one breaks."
                    ),
                ),
            ),
        ],
    )
    def test_executed_calls_input_support_matches_their_own_label(self, mode):
        process = self._ladder()
        model = self._InputSupportRecorder(process)
        sampler = PhysicsInformedColdDiffusion(
            model=model,
            num_timesteps=29,
            max_acceleration=32.0,
            dc_method="hard",
            reverse_mode=mode,
            sampling_steps=8,
            kspace_log_scaled=False,
        )
        x = torch.randn(1, 2, 64, 64)
        _, mask = process.q_sample(x, torch.full((1,), 6, dtype=torch.long))
        sampler.sample(x * mask, mask, start_timestep=6)

        assert model.calls, "no call was made"
        for label, support in model.calls:
            _, label_mask = process.q_sample(
                x, torch.full((1,), label, dtype=torch.long)
            )
            assert torch.equal(support, label_mask), (
                f"label={label}: input support does not equal mask(label)"
            )


class TestPriorChannelRangeReversePath:
    """Finding 7: the reverse loop derives its observed support from the
    single-channel sampling mask alone (``obs = (mask > 0).float()``,
    ``_sample_replace_freeze_dc`` / ``_sample_replace_freeze``), so it has no
    way to know ``prior_channel_range`` channels are meant to stay measured.

    Demonstrated with a measurement ``q_sample`` ITSELF produced, so this is
    not an artefact of building the measurement wrong: even a measurement
    q_sample legitimately marks fully-sampled on the prior channels gets those
    bins overwritten by the model's own prediction during ``sample()``.

    BLOCKED, not fixed here: making ``_sample_replace_freeze_dc`` /
    ``_sample_replace_freeze`` honour ``prior_channel_range`` in isolation
    would be actively harmful in production. The real validation measurement
    is built by ``diffusion.py``'s ``_apply_masking_and_verify`` (outside this
    file's scope) as a uniform ``input_batch * mask`` with no channel
    carve-out, so there the prior arrives genuinely undersampled -- treating
    it as trusted would hard-pin ``dc_method='hard'`` to the undersampled
    zeros instead of letting the model predict them, which is WORSE than
    today. Both sites must change together; tracked for a coordinated fix.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "finding 7: the reverse loop's obs support ignores "
            "prior_channel_range and overwrites a fully-sampled prior with "
            "the model's own prediction. A same-file fix is blocked: it would "
            "be harmful without a paired fix to diffusion.py's "
            "_apply_masking_and_verify, which is outside this file's scope "
            "-- see the task report."
        ),
    )
    def test_reverse_loop_overwrites_a_fully_sampled_prior(self):
        process = KSpaceUndersamplingProcess(
            num_timesteps=4,
            max_acceleration=4.0,
            center_fraction=0.125,
            prior_channel_range=(0, 2),
        )

        class PriorModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.kspace_process = process

            def forward(self, x, t):
                return torch.full_like(x, -999.0)

        x0 = torch.ones(1, 4, 16, 16)
        t = torch.full((1,), 3, dtype=torch.long)
        measurement, mask = process.q_sample(x0, t)
        assert bool((measurement[:, 0:2] == 1.0).all()), (
            "q_sample itself must mark the prior fully sampled -- what "
            "follows is about what the reverse loop does with that, not "
            "about how the measurement was built"
        )

        sampler = PhysicsInformedColdDiffusion(
            model=PriorModel(),
            num_timesteps=4,
            reverse_mode="replace_freeze_dc",
            sampling_steps=4,
            dc_method="hard",
            kspace_log_scaled=False,
        ).eval()
        out = sampler.sample(measurement=measurement, mask=mask, start_timestep=3)

        overwritten = int((out[:, 0:2] != 1.0).sum())
        assert overwritten == 0, (
            f"{overwritten} of {out[:, 0:2].numel()} fully-sampled prior "
            "coefficients were overwritten by the model's own prediction"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Acceleration-ladder realisability (issue #534)
# ---------------------------------------------------------------------------


def _exp11_process(min_center_fraction):
    """The exp_11 cohort's acceleration block, parameterised on the ACS floor."""
    from spectramr.models.diffusion.kspace_process import KSpaceUndersamplingProcess

    return KSpaceUndersamplingProcess(
        num_timesteps=28,
        max_acceleration=32.0,
        base_acceleration=2.0,
        center_fraction=0.08,
        min_center_fraction=min_center_fraction,
        mask_type="equispaced",
        seed=42,
        schedule_type="step",
        schedule_kwargs={"acceleration_range": [2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0]},
    )


class TestMinCenterFractionIsForwarded:
    """``min_center_fraction`` was stored on the process and never forwarded."""

    def test_forwarded_to_the_accelerator(self):
        process = _exp11_process(0.02)
        inner = process.mask_generator._get_accelerator("equispaced").accelerator
        assert inner.min_center_fraction == 0.02

    def test_static_acs_caps_the_ladder_at_12x(self):
        """center_fraction == min_center_fraction: the bug the cohort shipped with.

        An always-sampled 8% ACS IS the whole budget at R=12.5, so the declared
        32x rung realises ~12x and R=16/32 become the same mask.
        """
        ladder = {
            t: eff
            for t, _nom, eff, _kept in _exp11_process(0.08).describe_ladder((256, 256))
        }
        assert ladder[24] == pytest.approx(12.19, abs=0.1)
        assert ladder[20] == pytest.approx(ladder[24], abs=1e-6)

    def test_shrinking_acs_makes_every_rung_reachable(self):
        ladder = {
            t: eff
            for t, _nom, eff, _kept in _exp11_process(0.02).describe_ladder((256, 256))
        }
        assert ladder[20] == pytest.approx(16.0, rel=0.05)
        assert ladder[27] == pytest.approx(32.0, rel=0.05)


class TestDeclaredLadderDefects:
    def test_static_acs_is_reported(self):
        defects = _exp11_process(0.08).declared_ladder_defects((256, 256))
        assert defects
        assert any("R=32" in d for d in defects)
        assert any("same" in d for d in defects)

    def test_shrinking_acs_is_clean(self):
        assert _exp11_process(0.02).declared_ladder_defects((256, 256)) == []

    def test_assert_raises_only_when_defective(self):
        _exp11_process(0.02).assert_ladder_realisable((256, 256))  # no raise
        with pytest.raises(ValueError, match="not realisable"):
            _exp11_process(0.08).assert_ladder_realisable((256, 256))

    def test_continuous_schedule_is_out_of_scope(self):
        """A continuous schedule MUST have duplicate masks by pigeonhole.

        T timesteps cannot map to more than (num_bins) distinct masks, so
        duplicates there are not a defect and must not be reported.
        """
        from spectramr.models.diffusion.kspace_process import KSpaceUndersamplingProcess

        process = KSpaceUndersamplingProcess(
            num_timesteps=100,
            max_acceleration=32.0,
            base_acceleration=1.0,
            center_fraction=0.08,
            mask_type="equispaced",
            seed=42,
            schedule_type="linear",
        )
        assert process.declared_ladder_defects((256, 256)) == []

    def test_acceleration_is_measured_as_sampled_fraction(self):
        """Not as a line count: a line count reads ~R=1 on a 2-D mask."""
        ladder = _exp11_process(0.02).describe_ladder((256, 256))
        _t, _nom, eff, kept = ladder[0]
        assert kept == 256 * 256 / eff


# ---------------------------------------------------------------------------
# Reverse-loop inert steps and the timestep floor (issue #535)
# ---------------------------------------------------------------------------


class _CountingStub(torch.nn.Module):
    """Deterministic, input- and t-dependent stand-in that records its calls."""

    def __init__(self, process):
        super().__init__()
        self.kspace_process = process
        self.calls: list[int] = []

    def forward(self, x, t):
        self.calls.append(int(t[0]))
        return x * (1.0 + 0.01 * int(t[0])) + 0.001 * int(t[0])


def _sampler(min_center_fraction, reverse_mode="replace_freeze_dc"):
    from spectramr.models.diffusion.kspace_process import (
        KSpaceUndersamplingProcess,
        PhysicsInformedColdDiffusion,
    )

    process = KSpaceUndersamplingProcess(
        num_timesteps=28,
        max_acceleration=32.0,
        base_acceleration=2.0,
        center_fraction=0.08,
        min_center_fraction=min_center_fraction,
        mask_type="equispaced",
        seed=42,
        schedule_type="step",
        schedule_kwargs={"acceleration_range": [2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0]},
    )
    process.eval()
    model = _CountingStub(process)
    model.eval()
    sampler = PhysicsInformedColdDiffusion(
        model,
        num_timesteps=28,
        dc_method="hard",
        dc_weight=0.5,
        sampling_steps=20,
        reverse_mode=reverse_mode,
        reverse_clip_ratio=1.3,
        kspace_log_scaled=False,
    )
    return sampler, model, process


def _r2_measurement(process):
    torch.manual_seed(0)
    full = torch.randn(1, 8, 64, 64)
    _, mask = process.q_sample(full, torch.tensor([2]))
    mask = process.mask_generator.expand_mask_to_channels(mask, 8)
    return full * mask, mask


class TestInertStepSkipping:
    @pytest.mark.parametrize("mode", ["replace_freeze_dc", "replace_freeze"])
    def test_skipping_does_not_change_the_output(self, mode):
        """The load-bearing property: a skipped step must be a PURE no-op."""
        sampler, model, process = _sampler(0.02, mode)
        measurement, mask = _r2_measurement(process)
        with_skip = sampler.sample(measurement, mask)
        calls_with_skip = len(model.calls)

        full_sampler, full_model, _ = _sampler(0.02, mode)
        full_sampler._step_reveals_anything = lambda *a, **k: True
        without_skip = full_sampler.sample(measurement, mask)

        assert torch.equal(with_skip, without_skip)
        assert calls_with_skip < len(full_model.calls)

    def test_static_acs_made_almost_every_step_inert(self):
        """Reproduces the pre-#534 pathology: 2 of 21 steps do anything at R=2.

        The reconstruction was a single-shot fill wearing a 20-step trajectory's
        clothes.
        """
        sampler, model, process = _sampler(0.08)
        measurement, mask = _r2_measurement(process)
        sampler.sample(measurement, mask)
        assert len(model.calls) == 2
        assert sampler.last_skipped_steps == 19

    def test_shrinking_acs_makes_most_steps_meaningful(self):
        sampler, model, process = _sampler(0.02)
        measurement, mask = _r2_measurement(process)
        sampler.sample(measurement, mask)
        assert len(model.calls) > 2
        assert sampler.last_effective_steps == len(model.calls)

    def test_effective_and_skipped_sum_to_the_schedule(self):
        sampler, _model, process = _sampler(0.02)
        measurement, mask = _r2_measurement(process)
        sampler.sample(measurement, mask)
        assert sampler.last_effective_steps + sampler.last_skipped_steps == 21


class TestTimestepFloor:
    def test_floor_is_zero_when_t0_still_undersamples(self):
        process = _sampler(0.02)[2]
        # step schedule with an explicit range: R(0) = range[0] = 2.0 > 1
        assert process.min_meaningful_timestep() == 0

    def test_floor_is_one_when_t0_is_the_identity(self):
        from spectramr.models.diffusion.kspace_process import KSpaceUndersamplingProcess

        process = KSpaceUndersamplingProcess(
            num_timesteps=28,
            max_acceleration=32.0,
            base_acceleration=1.0,
            center_fraction=0.08,
            mask_type="equispaced",
            seed=42,
            schedule_type="linear",
        )
        assert process.min_meaningful_timestep() == 1

    def test_identity_rung_knob_lowers_the_floor_to_zero(self):
        """The opt-in overrides the ``R(0) == 1 -> 1`` exclusion.

        Same construction as ``test_floor_is_one_when_t0_is_the_identity``
        above, differing ONLY in the knob -- so a green result here that came
        from something other than the knob would redden that test too.
        """
        from spectramr.models.diffusion.kspace_process import KSpaceUndersamplingProcess

        process = KSpaceUndersamplingProcess(
            num_timesteps=28,
            max_acceleration=32.0,
            base_acceleration=1.0,
            center_fraction=0.08,
            mask_type="equispaced",
            seed=42,
            schedule_type="linear",
            train_identity_rung=True,
        )
        assert process.min_meaningful_timestep() == 0

    def test_identity_rung_knob_defaults_off(self):
        """Absent the knob the floor is unchanged, on both branches.

        The three pins above are the real regression guard; this states the
        default explicitly so a future default flip cannot be read as an
        unrelated failure.
        """
        from spectramr.models.diffusion.kspace_process import KSpaceUndersamplingProcess

        process = KSpaceUndersamplingProcess(
            num_timesteps=28,
            max_acceleration=32.0,
            base_acceleration=1.0,
            center_fraction=0.08,
            mask_type="equispaced",
            seed=42,
            schedule_type="linear",
        )
        assert process.train_identity_rung is False
        assert process.min_meaningful_timestep() == 1

    def test_identity_rung_is_a_noop_where_the_floor_is_already_zero(self):
        """``R(0) > 1`` already trains t=0; the knob must not change it."""
        process = _sampler(0.02)[2]
        assert process.min_meaningful_timestep() == 0
        process.train_identity_rung = True
        assert process.min_meaningful_timestep() == 0

    def test_reverse_range_guard_admits_t0_only_under_the_knob(self):
        """The floor is the SSOT for training AND the reverse range (#535).

        ``sample`` REJECTS a ``start_timestep`` below the floor rather than
        clamping it, so the guard is a direct, observable read of the resolved
        floor on the reverse path -- and it is the assertion that would go red
        if someone lowered the training floor without the reverse half moving
        with it. Asserted by running the sampler, not by re-deriving the bound.

        The evaluated-timestep list is deliberately NOT the assertion here: the
        loop skips inert steps, which on a static-ACS ladder collapses the very
        low rungs under test and would make this pass for the wrong reason.
        """
        from spectramr.models.diffusion.kspace_process import (
            KSpaceUndersamplingProcess,
            PhysicsInformedColdDiffusion,
        )

        kwargs = {
            "num_timesteps": 28,
            "max_acceleration": 32.0,
            "base_acceleration": 1.0,
            "center_fraction": 0.08,
            "min_center_fraction": 0.02,
            "mask_type": "equispaced",
            "seed": 42,
            "schedule_type": "linear",
        }

        def sample_from_t0(identity_rung: bool):
            process = KSpaceUndersamplingProcess(**kwargs, train_identity_rung=identity_rung)
            process.eval()
            model = _CountingStub(process)
            model.eval()
            sampler = PhysicsInformedColdDiffusion(
                model,
                num_timesteps=28,
                dc_method="hard",
                sampling_steps=20,
                reverse_mode="replace_freeze_dc",
                reverse_clip_ratio=1.3,
                kspace_log_scaled=False,
            )
            measurement, mask = _r2_measurement(process)
            sampler.sample(measurement, mask, start_timestep=0)

        with pytest.raises(ValueError, match="outside the meaningful reverse range"):
            sample_from_t0(False)
        sample_from_t0(True)  # must not raise

    def test_reverse_schedule_never_goes_below_the_floor(self):
        """Every timestep the loop evaluates must be one training can draw."""
        from spectramr.models.diffusion.kspace_process import (
            KSpaceUndersamplingProcess,
            PhysicsInformedColdDiffusion,
        )

        process = KSpaceUndersamplingProcess(
            num_timesteps=28,
            max_acceleration=32.0,
            base_acceleration=1.0,
            center_fraction=0.08,
            mask_type="equispaced",
            seed=42,
            schedule_type="linear",
        )
        process.eval()
        model = _CountingStub(process)
        model.eval()
        sampler = PhysicsInformedColdDiffusion(
            model,
            num_timesteps=28,
            dc_method="hard",
            sampling_steps=20,
            reverse_mode="replace_freeze_dc",
            reverse_clip_ratio=1.3,
            kspace_log_scaled=False,
        )
        measurement, mask = _r2_measurement(process)
        sampler.sample(measurement, mask)
        floor = process.min_meaningful_timestep()
        assert model.calls, "sampler made no model calls"
        assert min(model.calls) >= floor


# ---------------------------------------------------------------------------
# Band-local magnitude ceiling (issue #536)
# ---------------------------------------------------------------------------


def _decaying_kspace(h=64, w=64, channels=8):
    """Synthetic k-space with a realistic steep radial magnitude decay."""
    yy = torch.arange(h) - h // 2
    xx = torch.arange(w) - w // 2
    radius = torch.sqrt((yy[:, None] ** 2 + xx[None, :] ** 2).float())
    radius = radius / radius.max()
    torch.manual_seed(0)
    return torch.randn(1, channels, h, w) * torch.exp(-6 * radius)


def _r2_mask(h=64, w=64):
    mask = torch.zeros(1, 1, h, w)
    mask[:, :, ::2, :] = 1
    mask[:, :, h // 2 - 4 : h // 2 + 4, :] = 1  # ACS
    return mask


class TestBandLocalMagnitudeCeiling:
    def test_periphery_is_far_tighter_than_the_global_bound(self):
        """The point of the change: a global ratio bounds nothing in the periphery."""
        from spectramr.models.diffusion.kspace_process import (
            band_local_magnitude_ceiling,
            paired_magnitude,
        )

        measured = _decaying_kspace() * _r2_mask()
        # The global reference must use the SAME modulus the ceiling now uses.
        # ``measured.abs().amax()`` is the elementwise max over interleaved
        # Re/Im channels, which under-reads the true complex modulus by up to
        # sqrt(2) -- correcting exactly that is the point of issue #1281, so
        # pinning against it would re-assert the defect.
        global_ceiling = 1.3 * paired_magnitude(measured).amax()
        band = band_local_magnitude_ceiling(measured, 1.3, log_scaled=False)
        assert band[0, 0, 32, 32] == pytest.approx(float(global_ceiling), rel=1e-5)
        assert band[0, 0, 1, 1] < global_ceiling / 20

    def test_ceiling_is_monotone_non_increasing_with_radius(self):
        from spectramr.models.diffusion.kspace_process import (
            band_local_magnitude_ceiling,
        )

        band = band_local_magnitude_ceiling(_decaying_kspace() * _r2_mask(), 1.3, log_scaled=False)
        profile = [float(band[0, 0, 32, 32 + d]) for d in range(0, 32, 2)]
        assert all(b <= a + 1e-9 for a, b in pairwise(profile))

    def test_never_zero_so_prediction_is_never_forbidden(self):
        """A zero ceiling would hard low-pass the output instead of bounding it."""
        from spectramr.models.diffusion.kspace_process import (
            band_local_magnitude_ceiling,
        )

        # ACS-only measurement: every outer band is unsampled.
        measured = _decaying_kspace()
        acs = torch.zeros_like(measured)
        acs[:, :, 30:34, :] = measured[:, :, 30:34, :]
        band = band_local_magnitude_ceiling(acs, 1.3, log_scaled=False)
        assert bool((band > 0).all())

    def test_rejects_a_too_small_tensor(self):
        from spectramr.models.diffusion.kspace_process import (
            band_local_magnitude_ceiling,
        )

        with pytest.raises(ValueError, match="at least"):
            band_local_magnitude_ceiling(torch.zeros(4, 4), 1.3, log_scaled=False)

    def test_no_host_sync_per_call(self):
        """The old per-band loop's ``bool(sel.any())`` forced 16 device->host
        syncs per call on a partition that is a compile-time constant
        (non-negotiable 9). The ``scatter_reduce`` rewrite must introduce none.
        """
        from spectramr.models.diffusion.kspace_process import (
            band_local_magnitude_ceiling,
        )

        measured = _decaying_kspace() * _r2_mask()
        calls = {"bool": 0}
        orig_bool = torch.Tensor.__bool__

        def counting_bool(self):
            calls["bool"] += 1
            return orig_bool(self)

        torch.Tensor.__bool__ = counting_bool
        try:
            band_local_magnitude_ceiling(measured, 1.3, log_scaled=False)
        finally:
            torch.Tensor.__bool__ = orig_bool
        assert calls["bool"] == 0, f"{calls['bool']} host sync(s) via bool(); expected 0"

    def test_radial_band_partition_is_cached_across_calls(self):
        """The partition is a function of ``(h, w, num_bands, device)`` alone,
        so repeated calls at the SAME shape must reuse it rather than rebuild
        it. Verified through the cache's own hit/miss accounting (not
        ``len(cache)``/the number of stored keys): a cache that is written but
        never READ keeps exactly one entry while still rebuilding on every
        call, and that shape passed a previous version of this check.
        """
        from spectramr.models.diffusion.kspace_process import (
            _radial_band_partition,
            band_local_magnitude_ceiling,
        )

        _radial_band_partition.cache_clear()
        measured = _decaying_kspace() * _r2_mask()
        for _ in range(3):
            band_local_magnitude_ceiling(measured, 1.3, log_scaled=False)
        info = _radial_band_partition.cache_info()
        assert info.misses == 1, f"expected exactly one build, got {info.misses} misses"
        assert info.hits == 2, f"expected the other two calls to hit cache, got {info.hits}"


class TestClipReferenceWiring:
    def test_default_is_the_legacy_global_max(self):
        sampler, _model, _process = _sampler(0.02)
        assert sampler.clip_reference == "global_max"

    def test_unknown_reference_raises_at_build(self):
        from spectramr.models.diffusion.kspace_process import (
            KSpaceUndersamplingProcess,
            PhysicsInformedColdDiffusion,
        )

        process = KSpaceUndersamplingProcess(num_timesteps=8, mask_type="equispaced")
        with pytest.raises(ValueError, match="clip_reference"):
            PhysicsInformedColdDiffusion(
                _CountingStub(process),
                num_timesteps=8,
                dc_method="hard",
                clip_reference="nonsense",
                kspace_log_scaled=False,
            )

    def test_band_local_bounds_the_output_more_tightly(self):
        """An over-energetic model must be clamped harder under band_local."""
        from spectramr.models.diffusion.kspace_process import (
            KSpaceUndersamplingProcess,
            PhysicsInformedColdDiffusion,
        )

        class Loud(torch.nn.Module):
            def __init__(self, process):
                super().__init__()
                self.kspace_process = process

            def forward(self, x, t):
                # DC-peak-magnitude junk everywhere, the divergence signature
                return torch.full_like(x, float(x.abs().max()))

        outs = {}
        for reference in ("global_max", "band_local"):
            process = KSpaceUndersamplingProcess(
                num_timesteps=28,
                max_acceleration=32.0,
                base_acceleration=2.0,
                center_fraction=0.08,
                min_center_fraction=0.02,
                mask_type="equispaced",
                seed=42,
                schedule_type="step",
                schedule_kwargs={
                    "acceleration_range": [2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0]
                },
            )
            process.eval()
            model = Loud(process)
            model.eval()
            sampler = PhysicsInformedColdDiffusion(
                model,
                num_timesteps=28,
                dc_method="hard",
                sampling_steps=20,
                reverse_mode="replace_freeze_dc",
                reverse_clip_ratio=1.3,
                clip_reference=reference,
                kspace_log_scaled=False,
            )
            measured = _decaying_kspace() * _r2_mask()
            outs[reference] = sampler.sample(measured, _r2_mask().expand(1, 8, 64, 64))

        # Total energy written into the unobserved support must be far lower.
        unobserved = 1.0 - _r2_mask()
        energy = {
            k: float(((v * unobserved) ** 2).sum()) for k, v in outs.items()
        }
        assert energy["band_local"] < energy["global_max"] / 10, energy


# ---------------------------------------------------------------------------
# resolve_undersampling_kwargs (issue #550)
# ---------------------------------------------------------------------------


class TestResolveUndersamplingKwargs:
    """The generator and the CI ladder gate must resolve the SAME kwargs.

    They used to derive them independently, with different defaults, and the
    gate read raw YAML while the runtime read a schema that silently dropped
    keys. So the gate could certify a ladder the runtime never built: it
    reported ``defects: none`` for the whole exp_11 cohort while every arm ran
    a collapsed one.
    """

    def _accel_block(self, **overrides):
        block = {
            "acceleration_type": "equispaced",
            "schedule_type": "step",
            "base_acceleration": 2.0,
            "max_acceleration": 32.0,
            "center_fraction": 0.08,
            "min_center_fraction": 0.02,
            "acceleration_range": [2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0],
        }
        block.update(overrides)
        return block

    def test_str_enum_is_unwrapped_to_its_value(self):
        """``AccelerationSchedule`` is a ``(str, Enum)``.

        ``str(AccelerationSchedule.STEP)`` is ``'AccelerationSchedule.STEP'``,
        not ``'step'``. The gate compared with ``str(...)``, so routing it
        through the schema without unwrapping would have made every arm look
        like a non-step schedule and skip the check entirely.
        """
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema
        from spectramr.models.diffusion.kspace_process import (
            resolve_undersampling_kwargs,
        )

        cfg = AccelerationConfigSchema(**self._accel_block())
        assert resolve_undersampling_kwargs(cfg)["schedule_type"] == "step"

    def test_pydantic_object_dict_and_dump_agree(self):
        """Three call sites pass three different shapes of the same config.

        ``model_builder`` passes the live schema object, ``ModelFactory`` passes
        its ``model_dump()``, tests and scripts pass raw dicts. A pydantic model
        has no ``.get``, so the object path was an AttributeError waiting on any
        caller that skipped the factory.
        """
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema
        from spectramr.models.diffusion.kspace_process import (
            resolve_undersampling_kwargs,
        )

        cfg = AccelerationConfigSchema(**self._accel_block())
        from_object = resolve_undersampling_kwargs(cfg)
        from_dump = resolve_undersampling_kwargs(cfg.model_dump())
        from_raw = resolve_undersampling_kwargs(self._accel_block())
        assert from_object == from_dump == from_raw

    def test_none_config_falls_back_to_overrides(self):
        from spectramr.models.diffusion.kspace_process import (
            resolve_undersampling_kwargs,
        )

        resolved = resolve_undersampling_kwargs(None, {"max_acceleration": 12.0})
        assert resolved["max_acceleration"] == 12.0

    def test_acceleration_config_wins_over_overrides(self):
        from spectramr.models.diffusion.kspace_process import (
            resolve_undersampling_kwargs,
        )

        resolved = resolve_undersampling_kwargs(
            {"max_acceleration": 32.0}, {"max_acceleration": 8.0}
        )
        assert resolved["max_acceleration"] == 32.0

    def test_acceleration_range_reaches_schedule_kwargs(self):
        """Without it the step schedule spans [1, max], so t=0 is the identity."""
        from spectramr.models.diffusion.kspace_process import (
            resolve_undersampling_kwargs,
        )

        resolved = resolve_undersampling_kwargs(self._accel_block())
        assert resolved["schedule_kwargs"]["acceleration_range"] == [
            2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0
        ]

    def test_caller_schedule_kwargs_is_not_mutated(self):
        """The old inline code mutated the caller's dict in place."""
        from spectramr.models.diffusion.kspace_process import (
            resolve_undersampling_kwargs,
        )

        caller = {"density_power": 1.6}
        resolve_undersampling_kwargs(self._accel_block(), {"schedule_kwargs": caller})
        assert caller == {"density_power": 1.6}

    def test_schema_to_process_realises_every_declared_rung(self):
        """End-to-end: the path that was broken, asserted at the seam.

        The process-level tests above already passed while the cohort ran a
        collapsed ladder, because they constructed the process directly and
        never went through the schema. This one starts where the YAML does.
        """
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema
        from spectramr.models.diffusion.kspace_process import (
            KSpaceUndersamplingProcess,
            resolve_undersampling_kwargs,
        )

        cfg = AccelerationConfigSchema(**self._accel_block())
        process = KSpaceUndersamplingProcess(
            num_timesteps=28, **resolve_undersampling_kwargs(cfg)
        )
        assert process.min_center_fraction == 0.02
        assert process.declared_ladder_defects((256, 256)) == []

    def test_density_power_undeclared_keeps_the_stated_default(self):
        """0 of the cohort's arms declare this knob today; the fix must not
        silently move their runtime value off the literal they compute now."""
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema
        from spectramr.models.diffusion.kspace_process import (
            resolve_undersampling_kwargs,
        )

        cfg = AccelerationConfigSchema(**self._accel_block())
        assert "density_power" not in cfg.model_fields_set
        resolved = resolve_undersampling_kwargs(cfg)
        assert resolved["schedule_kwargs"]["density_power"] == 1.6

    def test_declared_density_power_reaches_schedule_kwargs(self):
        """The schema field (audit-advertised, schema default 2.0) was read
        nowhere before this fix: every declared value fell through to 1.6."""
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema
        from spectramr.models.diffusion.kspace_process import (
            resolve_undersampling_kwargs,
        )

        cfg = AccelerationConfigSchema(**self._accel_block(density_power=4.0))
        assert "density_power" in cfg.model_fields_set
        resolved = resolve_undersampling_kwargs(cfg)
        assert resolved["schedule_kwargs"]["density_power"] == 4.0

    def test_density_power_dump_cannot_be_told_apart_from_default(self):
        """A ``model_dump()`` carries no field-set info, so a caller on that
        path (``ModelFactory``) cannot distinguish "declared 2.0" from "schema
        default 2.0" -- falls to the stated literal rather than guessing."""
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema
        from spectramr.models.diffusion.kspace_process import (
            resolve_undersampling_kwargs,
        )

        cfg = AccelerationConfigSchema(**self._accel_block(density_power=4.0))
        resolved = resolve_undersampling_kwargs(cfg.model_dump())
        assert resolved["schedule_kwargs"]["density_power"] == 1.6

    def test_density_power_declared_twice_with_different_values_raises(self):
        """The schema field and the live undocumented ``schedule_kwargs``
        spelling (``model.model_kwargs.schedule_kwargs.density_power``) are two
        owners of the same fact; a disagreement must raise, not pick one
        silently (NN17)."""
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema
        from spectramr.models.diffusion.kspace_process import (
            resolve_undersampling_kwargs,
        )

        cfg = AccelerationConfigSchema(**self._accel_block(density_power=4.0))
        with pytest.raises(ValueError, match="density_power"):
            resolve_undersampling_kwargs(
                cfg, {"schedule_kwargs": {"density_power": 6.0}}
            )

    def test_density_power_declared_twice_with_same_value_resolves(self):
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema
        from spectramr.models.diffusion.kspace_process import (
            resolve_undersampling_kwargs,
        )

        cfg = AccelerationConfigSchema(**self._accel_block(density_power=4.0))
        resolved = resolve_undersampling_kwargs(
            cfg, {"schedule_kwargs": {"density_power": 4.0}}
        )
        assert resolved["schedule_kwargs"]["density_power"] == 4.0

    def test_dropping_the_key_collapses_the_ladder(self):
        """Pin the cost, so a future ``extra="ignore"`` drop fails loudly here."""
        from spectramr.models.diffusion.kspace_process import (
            KSpaceUndersamplingProcess,
            resolve_undersampling_kwargs,
        )

        block = self._accel_block()
        del block["min_center_fraction"]
        process = KSpaceUndersamplingProcess(
            num_timesteps=28, **resolve_undersampling_kwargs(block)
        )
        defects = process.declared_ladder_defects((256, 256))
        assert any("R=32" in d for d in defects)
        assert any("same" in d for d in defects)


class TestDynamicMaskBypassesNestingEnforcement:
    """``enable_dynamic_mask`` + ``enforce_nested`` was a 30x training slowdown.

    ``_generate_batch_masks_dynamic`` mutates the inner accelerator's seed per
    sample, and ``ColdDiffusionAccelerator`` caches its first-drop map on
    ``(shape, device, seed)``. So every sample missed the cache, rebuilt the whole
    cascade at ``num_timesteps`` mask evaluations, and left a permanent entry
    behind. Measured at 256x256 batch 2: 11 ms/step -> 334 ms/step, with the cache
    growing one entry per sample for the life of the run.

    ``get_acceleration_mask`` already documented that enforcement is meant to
    apply to the fixed-seed cascade only; nothing implemented it.
    """

    @staticmethod
    def _process(enforce: bool):
        from spectramr.models.diffusion.kspace_process import KSpaceUndersamplingProcess

        return KSpaceUndersamplingProcess(
            num_timesteps=28,
            max_acceleration=32.0,
            base_acceleration=2.0,
            center_fraction=0.08,
            min_center_fraction=0.02,
            mask_type="random",
            seed=42,
            schedule_type="step",
            schedule_kwargs={"acceleration_range": [2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0]},
            mask_direction="phase",
            enable_dynamic_mask=True,
            enforce_nested=enforce,
        )

    def test_dynamic_path_leaves_no_cache_entries(self):
        """The leak is the observable signature of the cliff: one cache entry per
        SAMPLE, never evicted. Asserting on entries rather than wall-clock keeps
        this deterministic on a shared runner."""
        import torch

        p = self._process(enforce=True)
        for _ in range(4):
            p._generate_batch_masks_dynamic(
                2, torch.randint(0, 28, (2,)), (64, 64), torch.device("cpu")
            )
        acc = p.mask_generator._get_accelerator("random")
        assert len(acc._nested_cache) == 0, (
            f"{len(acc._nested_cache)} first-drop maps cached from the dynamic "
            "path; enforcement must be bypassed there"
        )

    def test_enforce_flag_is_restored(self):
        """The bypass must be invisible outside the loop — otherwise validation
        silently loses the nesting guarantee it depends on."""
        import torch

        p = self._process(enforce=True)
        acc = p.mask_generator._get_accelerator("random")
        p._generate_batch_masks_dynamic(
            2, torch.randint(0, 28, (2,)), (64, 64), torch.device("cpu")
        )
        assert acc.enforce_nested is True

    def test_fixed_seed_cascade_still_nests(self):
        """The whole point of the flag. Bypassing it for dynamic masks must not
        weaken the path the reverse trajectory and validation actually walk."""
        import torch

        p = self._process(enforce=True)
        acc = p.mask_generator._get_accelerator("random")
        p._generate_batch_masks_dynamic(
            2, torch.randint(0, 28, (2,)), (64, 64), torch.device("cpu")
        )
        masks = [
            acc.get_acceleration_mask((1, 64, 64), t)[0].bool() for t in range(28)
        ]
        violations = sum(1 for t in range(27) if (masks[t + 1] & ~masks[t]).any())
        assert violations == 0, f"{violations}/27 nesting violations after the bypass"


# ---------------------------------------------------------------------------
# Schedule certification audits (papers' C1 leak-freeness, inert levels, C4)
# ---------------------------------------------------------------------------


def _exp11_cascade(*, enforce_nested):
    """The exp_11 arm parameterised on nesting enforcement (equispaced/step)."""
    from spectramr.models.diffusion.kspace_process import KSpaceUndersamplingProcess

    return KSpaceUndersamplingProcess(
        num_timesteps=28,
        max_acceleration=32.0,
        base_acceleration=2.0,
        center_fraction=0.08,
        min_center_fraction=0.02,
        mask_type="equispaced",
        seed=42,
        schedule_type="step",
        schedule_kwargs={"acceleration_range": [2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0]},
        enforce_nested=enforce_nested,
    )


def _planted(masks):
    """A process whose audits walk a hand-built cascade instead of the generator's."""
    process = _exp11_cascade(enforce_nested=False)
    # ``**_`` so the stub keeps accepting keyword extensions to the real signature
    # (``raw=`` arrived with the enforced-vs-family leak distinction); a planted
    # cascade is enforcement-free by definition, so the flag has nothing to change.
    process._cascade_masks = lambda image_shape, **_: iter(enumerate(masks))
    return process


def _mask4(*removed):
    """4x4 all-kept mask with the given (row, col) bins removed."""
    m = torch.ones(4, 4, dtype=torch.bool)
    for r, c in removed:
        m[r, c] = False
    return m


class TestNestingLeakReport:
    A, B = (0, 0), (1, 1)

    def test_enforced_cascade_is_leak_free(self):
        assert _exp11_cascade(enforce_nested=True).nesting_leak_report((64, 64)) == []

    def test_unenforced_step_schedule_leaks(self):
        """The defect the audit exists for: equispaced stride sets do not nest."""
        report = _exp11_cascade(enforce_nested=False).nesting_leak_report((64, 64))
        assert report, "equispaced/step without enforcement is known to leak"
        for entry in report:
            assert 0 < entry["leak_fraction"] <= 1
            # Adjacency violations are a subset of the running-union count.
            assert entry["consecutive_bins"] <= entry["reintroduced_bins"]

    def test_planted_reintroduction_fields_are_exact(self):
        """Bin A removed at level 0, back at level 2, still held at level 3.

        Level 3 keeps the same mask as level 2, so A is no adjacency violation
        there -- only the running union still knows it was ever removed. A
        consecutive-pair check would call level 3 clean.
        """
        masks = [
            _mask4(self.A),
            _mask4(self.A, self.B),
            _mask4(self.B),
            _mask4(self.B),
        ]
        assert _planted(masks).nesting_leak_report((4, 4)) == [
            {"t": 2, "reintroduced_bins": 1, "consecutive_bins": 1,
             "leak_fraction": 1 / 15},
            {"t": 3, "reintroduced_bins": 1, "consecutive_bins": 0,
             "leak_fraction": 1 / 15},
        ]

    def test_nested_planted_cascade_is_clean(self):
        masks = [_mask4(), _mask4(self.A), _mask4(self.A, self.B)]
        assert _planted(masks).nesting_leak_report((4, 4)) == []


class TestInertStepReport:
    A = (0, 0)

    def test_planted_inert_levels_are_indexed(self):
        masks = [_mask4(), _mask4(), _mask4(self.A), _mask4(self.A)]
        assert _planted(masks).inert_step_report((4, 4)) == [1, 3]

    def test_strictly_shrinking_cascade_has_none(self):
        masks = [_mask4(), _mask4(self.A), _mask4(self.A, (1, 1))]
        assert _planted(masks).inert_step_report((4, 4)) == []

    def test_step_schedule_pigeonhole_is_visible(self):
        """28 levels over 7 rungs leave exactly 28 - 7 levels with nothing to do.

        This is the forward-side twin of issue #535: the nested cascade is fine
        (leak-free), but the timestep axis is degenerate inside each rung.
        """
        inert = _exp11_cascade(enforce_nested=True).inert_step_report((64, 64))
        assert len(inert) == 28 - 7


class TestRemovedLineEnergyStats:
    A, B, C = (0, 0), (1, 1), (2, 2)

    def _flat_kspace(self):
        return torch.ones(1, 1, 4, 4, dtype=torch.complex64)

    def test_flat_spectrum_fractions_count_bins(self):
        """Uniform energy: per-level fraction is bins_removed / total bins."""
        masks = [_mask4(), _mask4(self.A, self.B), _mask4(self.A, self.B, self.C)]
        stats = _planted(masks).removed_line_energy_stats(
            (4, 4), self._flat_kspace(), domain="kspace"
        )
        assert [(s["t"], s["bins_removed"]) for s in stats] == [(1, 2), (2, 1)]
        assert stats[0]["energy_fraction"] == pytest.approx(2 / 16)
        assert stats[1]["energy_fraction"] == pytest.approx(1 / 16)
        assert stats[0]["share"] == pytest.approx(2 / 3)
        assert sum(s["share"] for s in stats) == pytest.approx(1.0)

    def test_nested_fractions_sum_to_total_removed_energy(self):
        """delta_t bookkeeping: on a nested cascade nothing is double-counted."""
        masks = [_mask4(), _mask4(self.A, self.B), _mask4(self.A, self.B, self.C)]
        stats = _planted(masks).removed_line_energy_stats(
            (4, 4), self._flat_kspace(), domain="kspace"
        )
        kept_first, kept_last = int(masks[0].sum()), int(masks[-1].sum())
        assert sum(s["energy_fraction"] for s in stats) == pytest.approx(
            (kept_first - kept_last) / 16
        )

    def test_image_domain_routes_through_centred_fft(self):
        """A constant image is a DC spike at (H//2, W//2) under fft2c.

        Removing the centre bin must therefore carry ~all the energy, and
        removing a corner bin ~none -- this pins the shifted convention; a raw
        (unshifted) FFT would score the two levels the other way around.
        """
        dc, corner = (2, 2), (0, 0)
        masks = [_mask4(), _mask4(dc), _mask4(dc, corner)]
        stats = _planted(masks).removed_line_energy_stats(
            (4, 4), torch.ones(1, 1, 4, 4), domain="image"
        )
        assert stats[0]["energy_fraction"] == pytest.approx(1.0, abs=1e-6)
        assert stats[1]["energy_fraction"] == pytest.approx(0.0, abs=1e-6)

    def test_unknown_domain_raises(self):
        with pytest.raises(ValueError, match="domain"):
            _planted([_mask4()]).removed_line_energy_stats(
                (4, 4), self._flat_kspace(), domain="fourier"
            )

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            _exp11_cascade(enforce_nested=True).removed_line_energy_stats(
                (64, 64), torch.ones(1, 1, 32, 32)
            )

    def test_real_cascade_bookkeeping_is_consistent(self):
        process = _exp11_cascade(enforce_nested=True)
        stats = process.removed_line_energy_stats(
            (64, 64), torch.rand(2, 3, 64, 64)
        )
        masks = [kept for _t, kept in process._cascade_masks((64, 64))]
        removed_bins = int(masks[0].sum()) - int(masks[-1].sum())
        assert sum(s["bins_removed"] for s in stats) == removed_bins
        assert sum(s["share"] for s in stats) == pytest.approx(1.0)


class TestScheduleCertificationReport:
    A, B, C = (0, 0), (1, 1), (2, 2)

    def test_bundles_verdicts_without_batch(self):
        process = _exp11_cascade(enforce_nested=True)
        report = process.schedule_certification_report((64, 64))
        assert report["leak_free"] is True
        assert report["leaks"] == []
        assert report["inert_steps"] == process.inert_step_report((64, 64))
        assert report["allocation"] is None

    def test_allocation_verdict_tracks_the_threshold(self):
        masks = [_mask4(), _mask4(self.A, self.B), _mask4(self.A, self.B, self.C)]
        batch = torch.ones(1, 1, 4, 4, dtype=torch.complex64)
        report = _planted(masks).schedule_certification_report(
            (4, 4), batch, max_level_share=0.5, domain="kspace"
        )
        alloc = report["allocation"]
        assert alloc["argmax_level"] == 1
        assert alloc["max_level_share"] == pytest.approx(2 / 3)
        assert alloc["ok"] is False
        relaxed = _planted(masks).schedule_certification_report(
            (4, 4), batch, max_level_share=0.7, domain="kspace"
        )
        assert relaxed["allocation"]["ok"] is True


class TestDeclaredLadderDefectStringsAreStable:
    """The CI ratchet baselines 62 arms against these EXACT strings.

    ``scripts/ci/check_acceleration_ladder_realisable.py`` compares
    ``declared_ladder_defects`` output byte-for-byte against its stored
    baseline, so any rewording silently un-baselines every known-defective
    arm. The schedule-certification audits above were added as NEW methods for
    exactly this reason; this test makes a drive-by "improvement" of the old
    strings fail loudly instead.
    """

    def test_exp11_static_acs_defect_strings_are_byte_identical(self):
        assert _exp11_process(0.08).declared_ladder_defects((256, 256)) == [
            "declared rung R=32 realises R=12.19 at t=24 (off by 62% > 25%)",
            "declared rungs R=[16.0, 32.0] all realise the same 5376-bin mask "
            "(5376 of 65536 bins)",
        ]


class TestSamplerDeterminismValidation:
    """C6 knobs fail loud at BUILD (the dc_method/reverse_mode pattern):
    a bad sampler_sigma or selection_rule must never surface mid-sampling."""

    @staticmethod
    def _build(**kwargs):
        class _Identity(torch.nn.Module):
            def forward(self, x, t):
                return x

        return PhysicsInformedColdDiffusion(
            model=_Identity(),
            num_timesteps=10,
            max_acceleration=4.0,
            center_fraction=0.08,
            **kwargs,
            kspace_log_scaled=False,
        )

    def test_defaults_are_deterministic(self):
        diffusion = self._build()
        assert diffusion.sampler_sigma == 0.0
        assert diffusion.sampler_seed is None
        assert diffusion.selection_rule == "fixed"

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sampler_sigma"):
            self._build(sampler_sigma=-0.1)

    def test_non_finite_sigma_raises(self):
        with pytest.raises(ValueError, match="sampler_sigma"):
            self._build(sampler_sigma=float("nan"))
        with pytest.raises(ValueError, match="sampler_sigma"):
            self._build(sampler_sigma=float("inf"))

    def test_unknown_selection_rule_raises(self):
        with pytest.raises(ValueError, match="selection_rule"):
            self._build(selection_rule="greedy")

    def test_valid_knobs_are_stored(self):
        diffusion = self._build(sampler_sigma=0.1, sampler_seed=7)
        assert diffusion.sampler_sigma == 0.1
        assert diffusion.sampler_seed == 7


class TestAcceleratorKwargsTranslation:
    """The config → ``create_kspace_accelerator`` mapping (issues #1059, #534).

    Three call sites build a ``KSpaceMaskGenerator`` from an
    ``AccelerationConfigSchema``. Two of them used to re-derive the kwargs by
    filtering ``model_dump()``, which forwards every schema field — defaults
    included — and never translates ``mask_seed`` to the accelerator's ``seed``.
    Once ``_reject_unknown_accelerator_kwargs`` landed that raised a TypeError
    the first time validation asked for a mask; before it existed the same
    kwargs were discarded silently, leaving ``seed=None`` and a cascade that is
    no longer nested. These tests pin the shared allowlist that replaced them.
    """

    ARM: ClassVar[dict[str, object]] = {
        "acceleration_type": "density_nested",
        "base_acceleration": 2.0,
        "max_acceleration": 32.0,
        "center_fraction": 0.08,
        "min_center_fraction": 0.02,
        "acceleration_range": [2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0],
        "mask_direction": "phase",
        "schedule_type": "step",
        "mask_seed": 42,
        "enforce_nested": True,
        "enable_dynamic_mask": True,
    }

    def test_config_kwargs_construct_an_accelerator(self) -> None:
        """The kwargs must survive the gate that rejects unread names."""
        from spectramr.infrastructure.physics.sampling import create_kspace_accelerator
        from spectramr.models.diffusion.kspace_process import (
            accelerator_kwargs_from_config,
        )

        pattern, kwargs = accelerator_kwargs_from_config(self.ARM)
        assert pattern == "density_nested"
        accelerator = create_kspace_accelerator(
            acceleration_type=pattern, num_timesteps=28, **kwargs
        )
        assert accelerator is not None

    def test_mask_seed_is_translated_to_seed(self) -> None:
        """YAML says ``mask_seed``; the accelerator only knows ``seed``.

        Dropping it is not cosmetic: ``seed=None`` sends mask generation to the
        global RNG, so each call draws a fresh permutation instead of truncating
        one fixed ranking, and the cascade stops being nested (issue #1059).
        """
        from spectramr.models.diffusion.kspace_process import (
            accelerator_kwargs_from_config,
        )

        _, kwargs = accelerator_kwargs_from_config(self.ARM)
        assert kwargs["seed"] == 42
        assert "mask_seed" not in kwargs

    def test_schema_defaults_do_not_ride_along(self) -> None:
        """A dump-and-filter forwards every field; an allowlist forwards ten."""
        from spectramr.models.diffusion.kspace_process import (
            accelerator_kwargs_from_config,
        )

        _, kwargs = accelerator_kwargs_from_config(self.ARM)
        # Names the accelerator does not read, present in every ``model_dump()``.
        for junk in (
            "mixed_precision",
            "use_compile",
            "use_distributed",
            "gradient_accumulation_steps",
            "ground_truth_folder",
            "sampling_pattern",
            "schedule_steps",
            "enable_dynamic_mask",
            "acceleration_type",
        ):
            assert junk not in kwargs, f"{junk} would reach the accelerator"

    def test_schedule_type_becomes_acceleration_schedule(self) -> None:
        from spectramr.models.diffusion.kspace_process import (
            accelerator_kwargs_from_config,
        )

        _, kwargs = accelerator_kwargs_from_config(self.ARM)
        assert kwargs["acceleration_schedule"] == "step"
        assert "schedule_type" not in kwargs

    def test_schedule_kwargs_are_flattened(self) -> None:
        """``density_power`` is a top-level accelerator kwarg, not a nested dict."""
        from spectramr.models.diffusion.kspace_process import (
            accelerator_kwargs_from_config,
        )

        _, kwargs = accelerator_kwargs_from_config(self.ARM)
        assert kwargs["density_power"] == 1.6
        assert kwargs["acceleration_range"] == [2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0]
        assert "schedule_kwargs" not in kwargs

    def test_absent_mask_direction_is_omitted_not_none(self) -> None:
        """``line_axis=None`` is rejected downstream; absent means "keep default"."""
        from spectramr.models.diffusion.kspace_process import (
            accelerator_kwargs_from_config,
        )

        arm = {k: v for k, v in self.ARM.items() if k != "mask_direction"}
        _, kwargs = accelerator_kwargs_from_config(arm)
        assert "mask_direction" not in kwargs

    def test_min_center_fraction_defaults_to_center_fraction(self) -> None:
        """Unforwarded, it collapsed every rung above R≈12 onto one mask (#534)."""
        from spectramr.models.diffusion.kspace_process import build_accelerator_kwargs

        kwargs = build_accelerator_kwargs(
            max_acceleration=32.0,
            base_acceleration=2.0,
            center_fraction=0.08,
            min_center_fraction=None,
        )
        assert kwargs["min_center_fraction"] == 0.08

    def test_process_and_strategy_paths_agree(self) -> None:
        """The regression that stops the two paths drifting apart again.

        ``KSpaceUndersamplingProcess`` (the generator side) and
        ``accelerator_kwargs_from_config`` (the strategy/builder side) must build
        the same accelerator from the same YAML block — otherwise training and
        validation undersample differently and the metric is meaningless.
        """
        from spectramr.models.diffusion.kspace_process import (
            accelerator_kwargs_from_config,
            resolve_undersampling_kwargs,
        )

        _, strategy_kwargs = accelerator_kwargs_from_config(self.ARM)
        process = KSpaceUndersamplingProcess(
            num_timesteps=28, **resolve_undersampling_kwargs(self.ARM)
        )
        assert process.mask_generator._accelerator_kwargs == strategy_kwargs


class TestRungsRealisedAtNoTimestepAreReported:
    """Issue #1171: ``per_rung`` is built from timesteps, so a skipped rung vanishes.

    ``declared_ladder_defects`` collects one representative timestep per declared
    rung and then checks realised-vs-nominal and duplicate-mask collisions over
    that mapping. A rung that no timestep reaches never enters the mapping at all,
    so both checks skipped it and the gate reported "defects: none" for a ladder
    that silently drops rungs.

    The step index is ``min(int(t/(T-1) * K), K - 1)`` and takes at most ``T``
    distinct values, so ``K > T`` guarantees unreachable entries. This is the
    config-time counterpart of the raise in
    ``KSpaceAccelerator.timestep_for_acceleration``.
    """

    def _process(self, ladder, timesteps):
        from spectramr.models.diffusion.kspace_process import KSpaceUndersamplingProcess

        return KSpaceUndersamplingProcess(
            num_timesteps=timesteps,
            max_acceleration=float(max(ladder)),
            base_acceleration=float(min(ladder)),
            center_fraction=0.08,
            min_center_fraction=0.02,
            mask_type="equispaced",
            seed=42,
            schedule_type="step",
            schedule_kwargs={"acceleration_range": [float(r) for r in ladder]},
        )

    def test_more_rungs_than_timesteps_is_reported(self):
        """8 rungs over 4 timesteps: half of them are unreachable."""
        process = self._process([2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 32.0], 4)

        realised = {nom for _t, nom, _eff, _kept in process.describe_ladder((256, 256))}
        skipped = [r for r in (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 32.0) if r not in realised]
        assert skipped, "premise: 8 rungs over 4 timesteps must skip some"

        defects = process.declared_ladder_defects((256, 256))
        assert any("realised at NO timestep" in d for d in defects), (
            f"a ladder skipping {skipped} must be reported, got {defects}"
        )

    def test_the_message_names_the_counts_that_have_to_change(self):
        """A defect a reader cannot act on is barely better than silence."""
        process = self._process([2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 32.0], 4)
        message = " | ".join(process.declared_ladder_defects((256, 256)))
        assert "8 entries" in message
        assert "4 timesteps" in message

    def test_one_rung_per_timestep_is_clean(self):
        """The shape #1155 moved the kspace_filling cohort to; must stay silent."""
        ladder = [2.0, 4.0, 8.0, 16.0]
        assert self._process(ladder, 4).declared_ladder_defects((256, 256)) == []

    def test_fewer_rungs_than_timesteps_is_clean(self):
        """7 rungs over 28 timesteps — the shape all 7 mismatched corpus arms have.

        Measured 2026-08-17: every arm under ``experiments/inprogress/`` whose
        ``len(acceleration_range) != timesteps`` has FEWER rungs than timesteps, so
        this new defect class must not move the ratchet baseline for any of them.
        """
        ladder = [2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0]
        assert self._process(ladder, 28).declared_ladder_defects((256, 256)) == []

    def test_assert_ladder_realisable_also_raises_on_a_skipped_rung(self):
        """The hard assertion must cover the new defect class, not just #534's."""
        process = self._process([2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 32.0], 4)
        with pytest.raises(ValueError, match="not realisable"):
            process.assert_ladder_realisable((256, 256))


class TestReverseTrajectoryStartTimestep:
    """The reverse schedule starts where the measurement actually is (#535/#1388).

    ``sample()`` used to build every trajectory from ``num_timesteps - 1``, which is
    only the correct head when the input sits at MAX acceleration. Cascading
    validation evaluates several rungs against one sampler, so every rung below the
    top replayed the fully-degraded schedule. Those extra steps cannot write
    anything -- with a nested mask family the ``mask_next`` of any step above the
    measurement's own timestep is already inside the observed support, so the reveal
    is empty -- but each one still pays a ``q_sample`` plus a device sync in the
    inert-step guard.

    The suite pins both halves: the schedule SHAPE (via ``dc_method="soft"``, which
    disables the inert-step skip so every scheduled step calls the model and the
    call trace IS the schedule) and the RESULT (via ``dc_method="hard"``, where the
    surviving trajectory must be bit-identical to the old one).
    """

    TIMESTEPS = 8

    @pytest.fixture
    def nested_process(self) -> KSpaceUndersamplingProcess:
        """A certified-nested cascade -- the precondition the equivalence needs."""
        process = KSpaceUndersamplingProcess(
            num_timesteps=self.TIMESTEPS,
            max_acceleration=8.0,
            base_acceleration=1.0,
            center_fraction=0.1,
            mask_type="density_nested",
            enforce_nested=True,
            nested_tolerance=1.0,
            seed=42,
            schedule_type="linear",
        )
        # Guard the premise rather than assume it: without nesting the steps above
        # the head DO reveal coefficients and the equivalence below is false.
        assert process.nesting_leak_report((32, 32)) == []
        return process

    class _SpyModel(torch.nn.Module):
        """Records the timestep of every model call; output depends on ``t``.

        The ``t``-dependence matters: it makes the bit-identity assertion sensitive
        to ANY divergence in the trajectory, not just to a different step count.

        Carries a ``dc_layer`` because the ``dc_method="soft"`` cases below use
        'soft' only as a means to disable the hard-DC inert-step skip (so the
        call trace IS the schedule); soft now delegates to the trained layer
        like every other non-hard method.
        """

        def __init__(self, process: KSpaceUndersamplingProcess) -> None:
            super().__init__()
            from spectramr.infrastructure.physics.data_consistency import (
                SoftDataConsistency,
            )

            self.kspace_process = process
            self.dc_layer = SoftDataConsistency(lambda_init=1.0)
            self.seen: list[int] = []

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            self.seen.append(int(t[0]))
            return x * 0.5 + float(t[0]) * 0.01

    def _sampler(
        self, process: KSpaceUndersamplingProcess, dc_method: str
    ) -> tuple["TestReverseTrajectoryStartTimestep._SpyModel", PhysicsInformedColdDiffusion]:
        model = self._SpyModel(process)
        sampler = PhysicsInformedColdDiffusion(
            model=model,
            num_timesteps=self.TIMESTEPS,
            sampling_steps=self.TIMESTEPS,
            dc_method=dc_method,
            reverse_mode="replace_freeze_dc",
            kspace_log_scaled=False,
        )
        return model, sampler

    @staticmethod
    def _measurement(
        process: KSpaceUndersamplingProcess, t_used: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """k-space degraded at ``t_used`` -- what a cascade rung actually hands in."""
        torch.manual_seed(0)
        clean = torch.randn(1, 2, 32, 32)
        _, mask = process.q_sample(clean, torch.tensor([t_used]))
        return clean * mask, mask

    def test_none_preserves_the_legacy_fully_degraded_head(self, nested_process):
        """Default is unchanged: head at ``num_timesteps - 1``, floor at the minimum."""
        model, sampler = self._sampler(nested_process, "soft")
        measurement, mask = self._measurement(nested_process, 3)

        sampler.sample(measurement, mask)

        floor = nested_process.min_meaningful_timestep()
        assert model.seen == list(range(self.TIMESTEPS - 1, floor - 1, -1))

    def test_start_timestep_becomes_the_trajectory_head(self, nested_process):
        """The schedule starts at the rung's own timestep and still lands on the floor."""
        model, sampler = self._sampler(nested_process, "soft")
        measurement, mask = self._measurement(nested_process, 3)

        sampler.sample(measurement, mask, start_timestep=3)

        assert model.seen == [3, 2, 1]

    def test_steps_above_the_head_were_provably_inert(self, nested_process):
        """Old behaviour SCHEDULED 7 steps but only 3 could write: the wasted 4."""
        model, sampler = self._sampler(nested_process, "hard")
        measurement, mask = self._measurement(nested_process, 3)

        sampler.sample(measurement, mask)

        # The inert-step guard already skipped everything above t=3 -- at a cost.
        assert model.seen == [3, 2, 1]

    def test_result_is_bit_identical_to_the_legacy_trajectory(self, nested_process):
        """The decisive property: this removes only steps that never wrote anything.

        Under ``dc_method="hard"`` on a nested cascade the old schedule's surviving
        steps ARE the new schedule, so the reconstruction must match exactly -- not
        approximately. A tolerance here would hide precisely the regression the test
        exists to catch.
        """
        measurement, mask = self._measurement(nested_process, 3)

        _, legacy = self._sampler(nested_process, "hard")
        _, scoped = self._sampler(nested_process, "hard")

        assert torch.equal(
            legacy.sample(measurement, mask),
            scoped.sample(measurement, mask, start_timestep=3),
        )

    def test_head_at_the_horizon_reproduces_the_default(self, nested_process):
        """``start_timestep=num_timesteps-1`` is the boundary, and it is legal."""
        model, sampler = self._sampler(nested_process, "soft")
        measurement, mask = self._measurement(nested_process, self.TIMESTEPS - 1)

        sampler.sample(measurement, mask, start_timestep=self.TIMESTEPS - 1)

        floor = nested_process.min_meaningful_timestep()
        assert model.seen == list(range(self.TIMESTEPS - 1, floor - 1, -1))

    def test_head_above_the_horizon_raises(self, nested_process):
        """No silent clamp to the horizon (non-negotiable 3)."""
        _, sampler = self._sampler(nested_process, "soft")
        measurement, mask = self._measurement(nested_process, 3)

        with pytest.raises(ValueError, match="outside the meaningful reverse range"):
            sampler.sample(measurement, mask, start_timestep=self.TIMESTEPS)

    def test_head_below_the_floor_raises(self, nested_process):
        """The inverted-range trap: ``linspace`` ASCENDS below the floor.

        ``torch.linspace(0, 1, n)`` counts UP, and the ``sorted(..., reverse=True)``
        normalisation would then hand the loop a schedule whose head is ABOVE the
        requested one -- silently restoring the trajectory the caller asked to skip.
        Rejected rather than clamped for exactly that reason.
        """
        assert nested_process.min_meaningful_timestep() == 1
        _, sampler = self._sampler(nested_process, "soft")
        measurement, mask = self._measurement(nested_process, 3)

        with pytest.raises(ValueError, match="outside the meaningful reverse range"):
            sampler.sample(measurement, mask, start_timestep=0)

    def test_negative_head_raises(self, nested_process):
        """Guards the lower bound independently of where the floor happens to sit."""
        _, sampler = self._sampler(nested_process, "soft")
        measurement, mask = self._measurement(nested_process, 3)

        with pytest.raises(ValueError, match="outside the meaningful reverse range"):
            sampler.sample(measurement, mask, start_timestep=-1)


class TestMaskDeviceInheritance:
    """The process inherits its mask device from the run configuration (#1508).

    ``KSpaceMaskGenerator`` serves masks from a device-resident ``[T, 1, H, W]``
    table only when BOTH its own device and the incoming ``timesteps`` are
    non-CPU. Built device-less, this process pinned the generator to CPU, so the
    table was never reached and every ``q_sample`` paid the
    ``timesteps.to("cpu").tolist()`` host sync the table exists to remove --
    while the strategy-side generator, constructed with ``device=self.device``,
    did take the fast path. The device is threaded in from the configuration
    rather than sniffed off a tensor, which non-negotiable 9b forbids.
    """

    @staticmethod
    def _process(**kwargs) -> KSpaceUndersamplingProcess:
        return KSpaceUndersamplingProcess(
            num_timesteps=8,
            max_acceleration=8.0,
            center_fraction=0.08,
            mask_type="variable_density",
            seed=42,
            **kwargs,
        )

    def test_default_is_cpu_unchanged(self) -> None:
        """``None`` keeps the historical behaviour; it never resolves a device.

        Resolving here would raise on every legitimately CPU-only construction
        (the witness schedule certification, unit tests, CI wiring), because the
        accelerated-run contract makes resolution fail without an accelerator.
        """
        process = self._process()
        assert process.mask_device is None
        assert process.mask_generator.device.type == "cpu"

    def test_declared_device_reaches_the_mask_generator(self) -> None:
        """The wire itself, checkable without an accelerator present.

        ``torch.device("cuda")`` is constructible on a CPU-only host, and the
        generator only stores it -- so dropping ``device=`` from the
        ``create_kspace_mask_generator`` call turns this red everywhere, not just
        on a GPU runner.
        """
        process = self._process(device="cuda")
        assert process.mask_device == torch.device("cuda")
        assert process.mask_generator.device.type == "cuda"

    def test_accepts_a_torch_device_object(self) -> None:
        process = self._process(device=torch.device("cpu"))
        assert process.mask_device == torch.device("cpu")
        assert process.mask_generator.device.type == "cpu"

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cuda_q_sample_populates_the_mask_table(self) -> None:
        """The observation from #1508, inverted: the table is now reached.

        Before the fix this asserted ``len(...) == 0`` after a CUDA ``q_sample``
        -- the generator was CPU-pinned, so the fast-path gate never opened.
        """
        process = self._process(device="cuda").eval()
        cache = process.mask_generator._mask_tables
        assert len(cache) == 0

        x_start = torch.randn(2, 2, 64, 64, device="cuda")
        t = torch.full((2,), 5, device="cuda", dtype=torch.long)
        with torch.no_grad():
            _, masks = process.q_sample(x_start, t)

        assert masks.device.type == "cuda"
        assert len(cache) == 1

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_device_less_process_still_takes_the_slow_path(self) -> None:
        """The defect shape itself, pinned: no declared device, no table.

        Keeps the two halves of the gate distinguishable -- a future change that
        made the generator sniff the tensor device would turn this red, which is
        the behaviour non-negotiable 9b rules out.
        """
        process = self._process().eval()
        x_start = torch.randn(2, 2, 64, 64, device="cuda")
        t = torch.full((2,), 5, device="cuda", dtype=torch.long)
        with torch.no_grad():
            process.q_sample(x_start, t)
        assert len(process.mask_generator._mask_tables) == 0


# ---------------------------------------------------------------------------
# train_identity_rung threading (issue #535)
# ---------------------------------------------------------------------------


class TestIdentityRungResolution:
    """The knob must survive the config -> process translation, both ways.

    ``resolve_undersampling_kwargs`` is the SSOT the generator, the CI ladder
    gate and the witness checks all call, so a knob that stops here is declared
    and dead -- the non-negotiable 8 shape. These pin the translation itself
    rather than the floor, which its own tests cover.
    """

    def test_resolver_forwards_the_declared_knob(self) -> None:
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema
        from spectramr.models.diffusion.kspace_process import resolve_undersampling_kwargs

        cfg = AccelerationConfigSchema(base_acceleration=1.0, train_identity_rung=True)
        assert resolve_undersampling_kwargs(cfg)["train_identity_rung"] is True

    def test_resolver_defaults_the_knob_off(self) -> None:
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema
        from spectramr.models.diffusion.kspace_process import resolve_undersampling_kwargs

        cfg = AccelerationConfigSchema(base_acceleration=1.0)
        assert resolve_undersampling_kwargs(cfg)["train_identity_rung"] is False

    def test_resolved_kwargs_construct_a_process_with_the_lowered_floor(self) -> None:
        """The end-to-end translation, not the dict key in isolation.

        ``**resolve_undersampling_kwargs(...)`` is exactly how the generator
        builds the process, so this is the path a run takes.
        """
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema
        from spectramr.models.diffusion.kspace_process import (
            KSpaceUndersamplingProcess,
            resolve_undersampling_kwargs,
        )

        base = {
            "base_acceleration": 1.0,
            "max_acceleration": 32.0,
            "center_fraction": 0.08,
            "acceleration_type": "equispaced",
            "schedule_type": "linear",
        }
        off = KSpaceUndersamplingProcess(
            num_timesteps=28,
            **resolve_undersampling_kwargs(AccelerationConfigSchema(**base)),
        )
        on = KSpaceUndersamplingProcess(
            num_timesteps=28,
            **resolve_undersampling_kwargs(
                AccelerationConfigSchema(**base, train_identity_rung=True)
            ),
        )
        assert off.min_meaningful_timestep() == 1
        assert on.min_meaningful_timestep() == 0

    def test_accelerator_kwargs_do_not_carry_the_knob(self) -> None:
        """It is a property of the diffusion process, not of a mask pattern.

        Forwarding it would reach an accelerator that has no notion of a
        timestep floor; the forwarded set is enumerated explicitly, so this
        pins that enumeration rather than an absence of a typo.
        """
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema
        from spectramr.models.diffusion.kspace_process import accelerator_kwargs_from_config

        _pattern, kwargs = accelerator_kwargs_from_config(
            AccelerationConfigSchema(base_acceleration=1.0, train_identity_rung=True)
        )
        assert "train_identity_rung" not in kwargs

    def test_a_declared_false_beats_an_override_saying_true(self) -> None:
        """The falsy-``or`` reversion, pinned.

        Every numeric key beside this one chains with ``or`` for backwards
        compatibility, and ``False or True`` is ``True`` -- so "tidying" this
        read into the surrounding idiom would silently switch the rung ON for
        an arm that turned it off, with no other test in this file going red.
        The source comment at the read claims this property; nothing executed
        it until here.

        The second half is what makes the first half discriminating: a resolver
        that ignored ``overrides`` altogether would satisfy "declared False
        wins" for the wrong reason.
        """
        from spectramr.models.diffusion.kspace_process import resolve_undersampling_kwargs

        declared_off = resolve_undersampling_kwargs(
            {"train_identity_rung": False}, {"train_identity_rung": True}
        )
        assert declared_off["train_identity_rung"] is False

        # Absent on the arm -> the override is the live source of the value.
        from_override = resolve_undersampling_kwargs({}, {"train_identity_rung": True})
        assert from_override["train_identity_rung"] is True


def _stochastic_sampler(sigma: float, seed: int | None = 7):
    """``_sampler`` with the C6 knobs on, so two calls can be compared stream for stream."""
    from spectramr.models.diffusion.kspace_process import (
        KSpaceUndersamplingProcess,
        PhysicsInformedColdDiffusion,
    )

    process = KSpaceUndersamplingProcess(
        num_timesteps=28,
        max_acceleration=32.0,
        base_acceleration=2.0,
        center_fraction=0.08,
        min_center_fraction=0.02,
        mask_type="equispaced",
        seed=42,
        schedule_type="step",
        schedule_kwargs={"acceleration_range": [2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0]},
    )
    process.eval()
    model = _CountingStub(process)
    model.eval()
    sampler = PhysicsInformedColdDiffusion(
        model,
        num_timesteps=28,
        dc_method="hard",
        dc_weight=0.5,
        sampling_steps=20,
        reverse_mode="replace_freeze",
        reverse_clip_ratio=1.3,
        sampler_sigma=sigma,
        sampler_seed=seed,
        kspace_log_scaled=False,
    )
    return sampler, process


class TestEnsembleSeedOffset:
    """``sample(seed_offset=i)`` -- one noise stream per ensemble member (2026-09).

    The C6 contract reseeds at every ``sample()`` entry, so N calls at the same
    seed were N identical reconstructions; a validation ensemble needs member
    ``i`` to be a deterministic function of (measurement, mask, seed, i).
    """

    def test_offset_zero_is_the_legacy_stream(self) -> None:
        sampler, process = _stochastic_sampler(0.1)
        measurement, mask = _r2_measurement(process)
        legacy = sampler.sample(measurement, mask)
        member_zero = sampler.sample(measurement, mask, seed_offset=0)
        assert torch.equal(legacy, member_zero)

    def test_distinct_offsets_draw_distinct_reproducible_noise(self) -> None:
        sampler, process = _stochastic_sampler(0.1)
        measurement, mask = _r2_measurement(process)
        member_zero = sampler.sample(measurement, mask, seed_offset=0)
        member_one = sampler.sample(measurement, mask, seed_offset=1)
        assert not torch.equal(member_zero, member_one)
        assert torch.equal(member_one, sampler.sample(measurement, mask, seed_offset=1))

    def test_the_offset_has_nothing_to_shift_at_sigma_zero(self) -> None:
        """The deterministic path stays bit-for-bit what it was."""
        sampler, process = _stochastic_sampler(0.0)
        measurement, mask = _r2_measurement(process)
        assert torch.equal(
            sampler.sample(measurement, mask, seed_offset=0),
            sampler.sample(measurement, mask, seed_offset=3),
        )

    def test_the_stream_is_seed_plus_offset(self) -> None:
        diffusion = TestSamplerDeterminismValidation._build(sampler_sigma=0.1, sampler_seed=7)
        diffusion._reseed_sampler_generator(2)
        drawn = torch.randn(4, generator=diffusion._active_sampler_generator())
        expected = torch.randn(4, generator=torch.Generator().manual_seed(9))
        assert torch.equal(drawn, expected)

    @pytest.mark.parametrize("bad", [-1, True, 1.5])
    def test_an_illegal_offset_is_refused(self, bad) -> None:
        diffusion = TestSamplerDeterminismValidation._build(sampler_sigma=0.1, sampler_seed=7)
        with pytest.raises(ValueError, match="seed_offset"):
            diffusion._reseed_sampler_generator(bad)


# ---------------------------------------------------------------------------
# `mask_at` must be able to hit the memoised mask table.
#
# The fast path in `generate_batch_masks` is gated on BOTH the timestep tensor
# and the generator being off-CPU -- "a host-side table would need the index
# moved back, reintroducing this very sync". A bare `torch.tensor([t])` is
# always CPU, so every call missed the table and rebuilt the mask. The
# attribution path calls this once per scheduled step, per rung, per batch, and
# the cache's own note records a Scalene profile charging 24.24% of a run to
# the host copy it exists to avoid.
#
# Asserted as AGREEMENT with the generator's device rather than as "is cuda",
# so the test says the same thing on a CPU box and on an accelerator.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mask_at_builds_its_timestep_on_the_generators_device():
    from spectramr.models.diffusion.kspace_process import KSpaceUndersamplingProcess

    process = KSpaceUndersamplingProcess(
        num_timesteps=8,
        base_acceleration=1.0,
        max_acceleration=4.0,
        center_fraction=0.08,
        mask_type="equispaced",
        train_identity_rung=True,
    )
    seen: dict[str, torch.Tensor] = {}
    real = process.mask_generator.generate_batch_masks

    def spy(*args, **kwargs):
        seen["timesteps"] = kwargs["timesteps"]
        return real(*args, **kwargs)

    process.mask_generator.generate_batch_masks = spy
    process.mask_at(3, (16, 16))

    assert seen["timesteps"].device == process.mask_generator.device, (
        "the timestep index is on a different device from the mask generator, "
        "so the memoised table is skipped and the mask is rebuilt every call"
    )


@pytest.mark.unit
def test_mask_at_names_the_device_explicitly_not_by_accident():
    """The agreement test above is VACUOUS on a CPU box.

    `torch.tensor([t])` and a default-constructed generator are both CPU, so the
    devices match whether or not `mask_at` asks for one -- and this box cannot
    run CUDA kernels (sm_110 against a cu126 build), so the accelerator case is
    unreachable here. Assert the request itself, which is what carries to a GPU
    node (non-negotiable 15: a gate that cannot fail is not a gate).
    """
    import ast
    import inspect
    import textwrap

    from spectramr.models.diffusion.kspace_process import KSpaceUndersamplingProcess

    tree = ast.parse(textwrap.dedent(inspect.getsource(KSpaceUndersamplingProcess.mask_at)))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and "generate_batch_masks" in ast.unparse(n.func)
    ]
    assert len(calls) == 1, "expected exactly one mask build in mask_at"

    timesteps = next(k.value for k in calls[0].keywords if k.arg == "timesteps")
    rendered = ast.unparse(timesteps)
    assert "device=" in rendered, (
        f"mask_at builds its timestep index as `{rendered}` with no device, so it "
        "lands on CPU and misses the memoised mask table on every call"
    )
