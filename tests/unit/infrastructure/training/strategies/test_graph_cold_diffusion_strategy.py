"""Degradation-pattern resolution for :class:`GraphColdDiffusionStrategy` (#1092).

The strategy used to resolve its k-space mask through code that could not run:
``_setup_accelerator`` imported ``ACCELERATOR_REGISTRY`` (the name is
``_ACCELERATOR_REGISTRY``), so the import raised ``ImportError`` on every run, was
caught, and logged at ``debug``. ``self.accelerator`` was therefore always ``None``, the
NUFFT branch it gated was unreachable, and every run fell through to a mask **hardcoded**
to ``random_cartesian`` — ignoring ``physics.compressed_sensing.sampling_pattern``
entirely, on eleven arms including three literature baselines.

These tests pin the three properties that replacement has to hold:

1. the eight existing arms keep training on exactly the pattern they trained on before
   (a wiring fix must not silently change anyone's science);
2. an unresolvable pattern RAISES instead of falling back (CLAUDE.md #9); and
3. the non-Cartesian patterns stay refused for as long as their schedule is broken —
   which is a property of the mask stack, so it is asserted against the mask stack
   rather than trusted to a comment.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.training.strategies.graph_cold_diffusion_strategy import (
    GraphColdDiffusionStrategy as S,
)


def _resolve(pattern: str) -> str:
    """Call the resolver with a minimal stand-in for ``self``.

    Deliberately not a real strategy instance: constructing one needs a
    ``TrainingEnvironment`` (model, optimizers, dataloaders), none of which the resolver
    touches. Binding the plain function to a namespace keeps the test on the logic under
    test instead of on a fixture that could pass for the wrong reason.
    """
    fake_self = SimpleNamespace(
        # the class attributes the resolver reads off `self`
        _PATTERN_ALIASES=S._PATTERN_ALIASES,
        _NON_CARTESIAN_PATTERNS=S._NON_CARTESIAN_PATTERNS,
        config=SimpleNamespace(
            physics=SimpleNamespace(
                compressed_sensing=SimpleNamespace(sampling_pattern=pattern)
            )
        )
    )
    return S._resolve_degradation_pattern(fake_self)


class TestPatternResolution:
    def test_schema_default_cartesian_resolves_and_does_not_raise(self) -> None:
        """``cartesian`` is the schema default and what all eight arms declare — but it
        is NOT a key in the mask registry. It must be mapped, not rejected."""
        assert _resolve("cartesian") == "random_cartesian"

    def test_the_alias_preserves_what_the_hardcoded_fallback_used(self) -> None:
        """The behaviour-preservation contract. The old code hardcoded
        ``pattern="random_cartesian"``; the alias must land on the same string, or this
        'wiring fix' silently changes what eight arms train on."""
        assert S._PATTERN_ALIASES["cartesian"] == "random_cartesian"

    def test_a_registered_cartesian_pattern_passes_through_unchanged(self) -> None:
        assert _resolve("uniform_cartesian") == "uniform_cartesian"

    def test_an_unknown_pattern_raises_and_names_the_vocabulary(self) -> None:
        """`sampling_pattern` is typed `str`, not a Literal, so a typo cannot be caught
        at schema-validation time. It has to be caught here or not at all."""
        with pytest.raises(ValueError, match="not a registered k-space pattern"):
            _resolve("cartesain")  # transposed letters

    def test_the_unknown_pattern_error_lists_what_is_available(self) -> None:
        with pytest.raises(ValueError) as exc:
            _resolve("definitely_not_a_pattern")
        assert "random_cartesian" in str(exc.value)

    @pytest.mark.parametrize("pattern", ["radial", "spiral", "golden_angle"])
    def test_non_cartesian_patterns_are_refused_with_the_reason(self, pattern) -> None:
        """Refused rather than silently accepted: the mask stack DOES produce these, so
        without an explicit guard they would look like they work."""
        with pytest.raises(ValueError, match="non-Cartesian family"):
            _resolve(pattern)


class TestTheScheduleThatJustifiesTheRefusal:
    """The refusal above is only correct while the schedule is actually broken.

    Asserted against the live mask generator rather than taken on faith, so that if
    someone fixes the non-Cartesian schedule this test fails and points at the guard
    that should then be removed — instead of the guard quietly outliving its reason.
    """

    @staticmethod
    def _fractions(pattern: str, timesteps=(0, 100, 200, 400, 999)):
        from spectramr.infrastructure.training.utils.kspace_masks import KSpaceMaskGenerator

        g = KSpaceMaskGenerator(num_timesteps=1000)
        return [
            g.generate_acceleration_mask(
                timestep=t, image_shape=(64, 64), pattern=pattern
            )
            .float()
            .mean()
            .item()
            for t in timesteps
        ]

    @pytest.mark.parametrize("pattern", ["random_cartesian", "uniform_cartesian"])
    def test_cartesian_patterns_start_fully_sampled(self, pattern) -> None:
        """t=0 is the CLEAN end of a cold-diffusion schedule."""
        assert self._fractions(pattern)[0] == pytest.approx(1.0)

    @pytest.mark.parametrize("pattern", ["random_cartesian", "uniform_cartesian"])
    def test_cartesian_degradation_never_decreases_with_t(self, pattern) -> None:
        f = self._fractions(pattern)
        assert all(f[i] >= f[i + 1] - 1e-6 for i in range(len(f) - 1)), f

    @pytest.mark.parametrize("pattern", ["radial", "spiral", "golden_angle"])
    def test_non_cartesian_rungs_are_unrealisable_on_this_schedule(self, pattern) -> None:
        """The reason for the guard since #612: the families nest and start at the
        identity, but this schedule's first rungs sit between R=1 and R=1.33, which no
        rasterised trajectory covers, so the mask stack refuses them at build. If the
        schedule moves its first rung above 1.33 this fails and points at the guard."""
        with pytest.raises(ValueError, match="covers at most"):
            self._fractions(pattern)


class TestTheUndersamplingBlockReachesTheAccelerator:
    """#2060. The pattern was wired by #1092; everything else still was not.

    ``__init__`` built the generator with only a timestep count and a device, so
    ``accelerator_kwargs`` defaulted to ``{}`` and the accelerator used its own
    defaults for the whole ladder. The three arms on this strategy are the
    literature baselines for the experiment_11 shootout, and all three were
    degraded on a sweep to R=64 from the global RNG while declaring R=8 or R=32
    with ``mask_seed: 42``.
    """

    @staticmethod
    def _generator(undersampling, *, timesteps: int = 1000, phase_encode_axis: int = -1):
        """Bind the builder to a stand-in, as the resolver tests above do.

        A real instance needs a ``TrainingEnvironment``; the builder reads three
        config paths and a device, so a namespace keeps the test on the wiring.
        """
        fake_self = SimpleNamespace(
            config=SimpleNamespace(
                undersampling=undersampling,
                data=SimpleNamespace(
                    acquisition=SimpleNamespace(phase_encode_axis=phase_encode_axis)
                ),
                training=SimpleNamespace(diffusion=SimpleNamespace(timesteps=timesteps)),
            ),
            device=torch.device("cpu"),
        )
        return S._build_mask_generator(fake_self)

    @staticmethod
    def _cdiffmr_block():
        """``baseline_cdiffmr.yaml``'s block, including the ACS floor #2060 added."""
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema

        return AccelerationConfigSchema(
            acceleration_type="equispaced",
            base_acceleration=2.0,
            max_acceleration=32.0,
            center_fraction=0.04,
            min_center_fraction=0.02,
            acceleration_range=[32.0],
            mask_direction="phase",
            schedule_type="power_law",
            schedule_steps=1000,
            enable_dynamic_mask=True,
            mask_seed=42,
        )

    def test_the_declared_ladder_is_the_one_built(self):
        """Unwired this reaches R=64 at the top of a declared 32x sweep."""
        accelerator = self._generator(self._cdiffmr_block())._get_accelerator("random_cartesian")
        assert float(accelerator.get_acceleration_factor(999)) == pytest.approx(32.0)
        assert float(accelerator.get_acceleration_factor(0)) == pytest.approx(2.0)

    def test_the_declared_seed_reaches_the_accelerator(self):
        """``seed=None`` is the global RNG, so the cascade stops being nested.

        Cold diffusion's forward process assumes ``M_{t+1}`` is a subset of
        ``M_t``; without a fixed seed each call draws a fresh permutation
        instead of truncating one ranking (#1059).
        """
        accelerator = self._generator(self._cdiffmr_block())._get_accelerator("random_cartesian")
        assert accelerator.accelerator.seed == 42

    def test_an_unreachable_acs_floor_is_refused_rather_than_approximated(self):
        """A declared 4% ACS floor cannot fit the 3.1% budget at R=32, so the sweep is
        unrealisable. Failing at build beats reporting R=32 while realising ~25.
        Only a declared floor is refused (#2516); an undeclared one is forwarded as None.
        """
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema

        block = AccelerationConfigSchema(
            acceleration_type="equispaced",
            base_acceleration=2.0,
            max_acceleration=32.0,
            center_fraction=0.04,
            min_center_fraction=0.04,
            mask_seed=42,
        )
        with pytest.raises(ValueError, match="sampling budget"):
            self._generator(block)._get_accelerator("random_cartesian")

    def test_an_absent_block_leaves_the_accelerator_on_its_own_defaults(self):
        """No declaration must not synthesise one (non-negotiable 3)."""
        assert self._generator(None)._accelerator_kwargs == {}


class TestTheNameDoesNotPromiseAGraph:
    """The class is named for a method it does not implement.

    The warning block has said "CARTESIAN-only, despite its name" since #1092, but the
    feature list 100 lines below it went on asserting "Graph architecture handles
    arbitrary k-space trajectories" through two fix passes. These pin the claims to the
    code so the contradiction cannot silently return (#2082).
    """

    @staticmethod
    def _docstring() -> str:
        from spectramr.infrastructure.training.strategies.graph_cold_diffusion_strategy import (
            GraphColdDiffusionStrategy,
        )

        return GraphColdDiffusionStrategy.__doc__ or ""

    def test_no_graph_capability_is_claimed_outside_the_aspirational_section(self) -> None:
        doc = self._docstring()
        head, marker, tail = doc.partition("ASPIRATIONAL, NOT IMPLEMENTED")
        assert marker, "the aspirational marker is what separates spec from description"
        # Everything before the marker describes CURRENT behaviour, and the feature list
        # after it must not re-assert the capability the marker just disclaimed.
        live = head + tail.split("## Training Process", 1)[-1]
        for claim in ("Graph architecture handles", "GNN forward passes", "GNN: Aggregates"):
            assert claim not in live, f"docstring re-asserts an unimplemented capability: {claim!r}"

    def test_the_cartesian_only_warning_is_still_present(self) -> None:
        assert "CARTESIAN-only, despite its name" in self._docstring()

    def test_no_dead_nufft_projection_helper(self) -> None:
        from spectramr.infrastructure.training.strategies.graph_cold_diffusion_strategy import (
            GraphColdDiffusionStrategy,
        )

        # `self.nufft_op` is never assigned, so a caller would AttributeError into a bare
        # `except Exception` that returned the IMAGE where k-space was expected.
        assert not hasattr(GraphColdDiffusionStrategy, "_project_to_kspace")

    def test_documented_attributes_are_ones_the_strategy_actually_sets(self) -> None:
        doc = self._docstring()
        attrs = doc.split("Attributes:", 1)[1].split("References:", 1)[0]
        for gone in ("k_space_mask_gen", "trajectory:"):
            assert gone not in attrs, f"Attributes names something the instance lacks: {gone!r}"
        assert "mask_generator" in attrs


class TestOneDegradationLevelPerBatch:
    """#1760: the loss drew a timestep per sample, degraded the whole batch with
    ``t[0]``'s mask and ``t[0].item()`` (a host sync), then labelled each sample
    with its own ``t``. B-1 of every B samples were told a noise level they were
    not degraded at."""

    @staticmethod
    def _drive(pred_fn):
        import types

        from torch import nn

        from spectramr.infrastructure.training.strategies.graph_cold_diffusion_strategy import (
            GraphColdDiffusionStrategy,
        )
        from spectramr.models.losses.computers.base import LossOutput

        record: dict = {}

        class _Gen(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = nn.Parameter(torch.ones(()))

            def forward(self, x, timesteps=None):
                record["timesteps"] = timesteps
                return pred_fn(x) * self.w

        def _degrade(x_0, t, *, t_val, **_):
            record["mask_level"] = t_val
            record["t"] = t
            return x_0, None, x_0, torch.ones_like(x_0)

        st = object.__new__(GraphColdDiffusionStrategy)
        st.timesteps = 1000
        st.env = types.SimpleNamespace(generator=_Gen(), losses={})
        st._apply_degradation = _degrade
        st.loss_computer = types.SimpleNamespace(
            compute=lambda pred, target, **_: LossOutput(
                total=(pred - target).abs().mean(), components={}, metrics={}
            )
        )
        x = torch.rand(6, 1, 8, 8)
        out = st._compute_losses_impl(x, x.clone(), epoch=0)
        return out, record

    def test_every_sample_is_labelled_with_the_level_it_was_degraded_at(self) -> None:
        _, record = self._drive(lambda x: x)
        ts = record["timesteps"]
        assert ts is not None and ts.shape == (6,)
        assert torch.all(ts == record["mask_level"])
        assert isinstance(record["mask_level"], int)

    def test_a_non_finite_prediction_reaches_the_executor_instead_of_a_zero_leaf(self) -> None:
        out, _ = self._drive(lambda x: x * float("nan"))
        loss = out["g_total_loss"]
        assert not torch.isfinite(loss)
        assert loss.grad_fn is not None
        assert "nan_skip" not in out


def test_the_snapshot_names_the_pattern_every_mask_call_passes() -> None:
    """The generator's ``linear`` default is never used here; reporting it
    would name a mask this strategy does not build."""
    st = object.__new__(S)
    st.degradation_pattern = "random_cartesian"
    assert st._snapshot_mask_pattern() == "random_cartesian"


class _GeometryGen(torch.nn.Module):
    """Stand-in for a geometry-plane generator: raises without ``mask=``, records it."""

    kspace_geometry_channels = 6

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(()))
        self.masks: list[torch.Tensor] = []

    def forward(self, x, timesteps=None, mask=None):
        if mask is None:
            raise ValueError("forward needs mask=")
        self.masks.append(mask)
        return x * self.w


class TestTheSupportReachesAGeometryModel:
    """``cold_diffusion`` + ``kspace_cold_diffusion`` routes here, and that generator
    raises without the support its input carries."""

    @staticmethod
    def _strategy(gen):
        import types

        from spectramr.models.losses.computers.base import LossOutput

        st = object.__new__(S)
        st.timesteps = 4
        st.env = types.SimpleNamespace(generator=gen, losses={})
        st.device = torch.device("cpu")
        st.validation_timestep = 0
        st.loss_computer = types.SimpleNamespace(
            compute=lambda pred, target, **_: LossOutput(
                total=(pred - target).abs().mean(), components={}, metrics={}
            )
        )
        return st

    def test_training_passes_the_rung_mask_the_input_was_sampled_on(self) -> None:
        gen = _GeometryGen()
        st = self._strategy(gen)
        rung = torch.zeros(1, 1, 8, 8)
        rung[..., ::2] = 1.0

        def _degrade(x_0, t, *, t_val, **_):
            return x_0 * rung, None, x_0, rung

        st._apply_degradation = _degrade
        x = torch.rand(3, 1, 8, 8)
        st._compute_losses_impl(x, x.clone(), epoch=0)
        assert gen.masks and gen.masks[0] is rung

    def test_the_degradation_returns_the_mask_it_applied(self) -> None:
        from spectramr.infrastructure.training.utils.kspace_masks import KSpaceMaskGenerator
        from spectramr.infrastructure.training.utils.transform_ops import FFTTransformer

        st = object.__new__(S)
        st.timesteps = 10
        st.fft_transformer = FFTTransformer()
        st.mask_generator = KSpaceMaskGenerator(num_timesteps=10)
        st.degradation_pattern = "random_cartesian"
        st.degradation = "kspace_mask"
        st.b0_simulator = None
        x = torch.rand(2, 2, 16, 16)
        x_t, _traj, _target, mask = st._apply_degradation(
            x, torch.full((2,), 5), t_val=5
        )
        assert mask.dtype == torch.float32 and mask.shape[-2:] == (16, 16)
        torch.testing.assert_close(x_t, st.fft_transformer.fft2c(x) * mask)

    def test_validation_passes_the_measurement_support(self) -> None:
        import types

        from spectramr.models.blocks.kspace_geometry_encoding import measurement_support

        gen = _GeometryGen()
        st = self._strategy(gen)
        st.config = types.SimpleNamespace(
            validation=types.SimpleNamespace(
                scoring=types.SimpleNamespace(enable_image_metrics=False)
            )
        )
        st._log_validation_images_to_tensorboard = lambda **_: None
        y = torch.zeros(2, 2, 8, 8)
        y[..., 1::2] = 1.0
        st.validation_step(y, y.clone())
        torch.testing.assert_close(gen.masks[0], measurement_support(y))

    def test_a_model_without_geometry_planes_gets_no_mask(self) -> None:
        import types

        class _Plain(torch.nn.Module):
            def forward(self, x, timesteps=None, **kwargs):
                return x

        st = object.__new__(S)
        st.env = types.SimpleNamespace(generator=_Plain())
        kwargs = st._build_forward_kwargs(
            torch.zeros(1, 2, 4, 4), torch.zeros(1), None, mask=torch.ones(1, 1, 4, 4)
        )
        assert "mask" not in kwargs


class TestRicianForwardProcess:
    """``training.diffusion.degradation: rician`` builds x_t from noise, not a mask."""

    @staticmethod
    def _strategy(degradation: str, *, timesteps: int = 10, sigma_max: float = 0.5):
        from spectramr.infrastructure.training.utils.kspace_masks import KSpaceMaskGenerator
        from spectramr.infrastructure.training.utils.transform_ops import FFTTransformer

        st = object.__new__(S)
        st.timesteps = timesteps
        st.fft_transformer = FFTTransformer()
        st.mask_generator = KSpaceMaskGenerator(num_timesteps=timesteps)
        st.degradation_pattern = "random_cartesian"
        st.degradation = degradation
        st.rician_sigma_max = sigma_max
        st.b0_simulator = None
        return st

    @staticmethod
    def _image() -> torch.Tensor:
        torch.manual_seed(0)
        return torch.rand(2, 1, 16, 16) + 0.5

    def test_the_clean_end_is_the_clean_kspace(self) -> None:
        st, x = self._strategy("rician"), self._image()
        x_t, _, _, mask = st._apply_degradation(x, torch.zeros(2), t_val=0)
        torch.testing.assert_close(x_t, st.fft_transformer.fft2c(x))
        assert torch.all(mask == 1)

    def test_noise_grows_along_the_schedule_and_keeps_full_support(self) -> None:
        st, x = self._strategy("rician"), self._image()
        errs = []
        for t in (3, 9):
            x_t, _, _, mask = st._apply_degradation(x, torch.full((2,), t), t_val=t)
            assert torch.all(mask == 1) and mask.dtype == torch.float32
            image = st.fft_transformer.ifft2c(x_t)
            assert image.imag.abs().max() < 1e-4  # the magnitude is real
            errs.append((image.real - x).abs().mean())
        assert errs[0] > 0 and errs[1] > errs[0]

    def test_the_two_processes_feed_the_model_different_inputs(self) -> None:
        """The knob changes the training input at the same level, hence the loss."""
        x = self._image()
        rician, _, _, _ = self._strategy("rician")._apply_degradation(x, torch.full((2,), 6), t_val=6)
        masked, _, _, _ = self._strategy("kspace_mask")._apply_degradation(
            x, torch.full((2,), 6), t_val=6
        )
        assert not torch.allclose(rician, masked)

    def test_a_real_multichannel_image_is_refused(self) -> None:
        st = self._strategy("rician")
        with pytest.raises(ValueError, match="cannot pair"):
            st._apply_degradation(torch.rand(1, 2, 8, 8), torch.full((1,), 4), t_val=4)

    def test_setup_refuses_an_unknown_degradation(self) -> None:
        st = object.__new__(S)
        st.config = SimpleNamespace(
            training=SimpleNamespace(
                diffusion=SimpleNamespace(
                    timesteps=10, degradation="blur", rician_sigma_max=None, validation_timestep=None
                )
            ),
            physics=SimpleNamespace(compressed_sensing=SimpleNamespace(sampling_pattern="cartesian")),
        )
        with pytest.raises(ValueError, match="not a forward process"):
            st._setup_cold_diffusion_components()

    def test_validation_tells_the_model_the_declared_level(self) -> None:
        import types

        seen: list[torch.Tensor] = []

        class _Gen(torch.nn.Module):
            def forward(self, x, timesteps=None):
                seen.append(timesteps)
                return x

        st = object.__new__(S)
        st.env = types.SimpleNamespace(generator=_Gen())
        st.device = torch.device("cpu")
        st.validation_timestep = 7
        st.config = types.SimpleNamespace(
            validation=types.SimpleNamespace(scoring=types.SimpleNamespace(enable_image_metrics=False))
        )
        st._log_validation_images_to_tensorboard = lambda **_: None
        st.validation_step(torch.rand(2, 2, 8, 8), torch.rand(2, 2, 8, 8))
        assert torch.all(seen[0] == 7)
