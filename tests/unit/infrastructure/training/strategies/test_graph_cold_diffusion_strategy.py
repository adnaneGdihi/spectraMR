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
    def test_non_cartesian_has_no_clean_end(self, pattern) -> None:
        """The reason for the guard: at t=0 these are already heavily undersampled, so
        the model never sees x_0."""
        assert self._fractions(pattern)[0] < 0.5

    @pytest.mark.parametrize("pattern", ["radial", "spiral", "golden_angle"])
    def test_non_cartesian_schedule_saturates(self, pattern) -> None:
        """The second reason: the schedule goes FLAT, so the timestep conditioning
        carries almost no information over most of its range.

        Measured from t=400 rather than t=200: radial and spiral are already flat at
        200, but golden_angle still moves (0.086 -> 0.057) between 200 and 400. Pinning
        the looser, true bound rather than the tidier, false one — the claim being
        tested is 'saturates', not 'saturates at exactly 200'."""
        f = self._fractions(pattern, timesteps=(400, 600, 999))
        assert max(f) - min(f) < 0.01, f


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
    def _generator(undersampling, *, timesteps: int = 1000):
        """Bind the builder to a stand-in, as the resolver tests above do.

        A real instance needs a ``TrainingEnvironment``; the builder reads two
        config paths and a device, so a namespace keeps the test on the wiring.
        """
        fake_self = SimpleNamespace(
            config=SimpleNamespace(
                undersampling=undersampling,
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
        """What the arm declared before #2060 added the floor.

        A 4% ACS cannot fit the 3.1% budget at R=32, so the declared sweep is
        unrealisable. Failing at build beats reporting R=32 while realising ~25.
        """
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema

        block = AccelerationConfigSchema(
            acceleration_type="equispaced",
            base_acceleration=2.0,
            max_acceleration=32.0,
            center_fraction=0.04,
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
