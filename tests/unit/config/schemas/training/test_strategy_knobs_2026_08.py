"""The 2026-08 knob-wiring batch: paradigm blocks that arrived as raw dicts.

``TrainingStrategyConfigSchema`` is ``extra="allow"``, so a paradigm block the
schema never declares is stored as the plain ``dict`` it arrived as. Every
strategy reads its knobs with ``getattr(cfg, "knob", default)`` — and ``getattr``
on a dict returns the **default**. The YAML validates, the key reaches
provenance, and the declared value is discarded.

The tests below are ordered by what they prove:

1. the blocks coerce at all (the mechanism);
2. every default equals the strategy's own fallback, so an arm that does NOT
   declare the block is bit-for-bit unaffected (the no-regression control);
3. a typo raises instead of sitting unread (what ``extra="forbid"`` buys);
4. **the ablation ablates** — the assertion that actually mattered. Two real
   corpus arms whose declared ``lambda_t`` differs used to resolve to the same
   value, so any measured delta between them was noise.

(4) is the red-then-green case: before the blocks were declared it failed,
because both arms resolved ``lambda_t == 0.01``.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
from spectramr.config.schemas.training.strategy_knobs_2026_08 import (
    AdaptiveSFCHSSCTrainingConfigSchema,
    BeltramiEPIDistortionTrainingConfigSchema,
    BlochEquivariantTranslationTrainingConfigSchema,
    ConformalDiffusionReconTrainingConfigSchema,
    ConformalMRFDictlessReconTrainingConfigSchema,
    CRLBMRFPulseDesignTrainingConfigSchema,
    CrossScannerMRFHarmonisationTrainingConfigSchema,
    DTN2STrainingConfigSchema,
    IBActiveAcquisitionTrainingConfigSchema,
    PrivilegedLearningTrainingConfigSchema,
    RiemannianBlochDiffusionTrainingConfigSchema,
    SpatiotemporalAdaptiveSFCReconTrainingConfigSchema,
)

#: block name -> its schema. The four this batch declares.
BLOCKS = {
    "spatiotemporal_adaptive_sfc_recon": SpatiotemporalAdaptiveSFCReconTrainingConfigSchema,
    "beltrami_epi_distortion": BeltramiEPIDistortionTrainingConfigSchema,
    "adaptive_sfc_hssc": AdaptiveSFCHSSCTrainingConfigSchema,
    "conformal_diffusion_recon": ConformalDiffusionReconTrainingConfigSchema,
    "conformal_mrf_dictless_recon": ConformalMRFDictlessReconTrainingConfigSchema,
    "crlb_mrf_pulse_design": CRLBMRFPulseDesignTrainingConfigSchema,
    "cross_scanner_mrf_harmonisation": CrossScannerMRFHarmonisationTrainingConfigSchema,
    "bloch_equivariant_translation": BlochEquivariantTranslationTrainingConfigSchema,
    "ib_active_acquisition": IBActiveAcquisitionTrainingConfigSchema,
    "riemannian_bloch_diffusion": RiemannianBlochDiffusionTrainingConfigSchema,
    "privileged": PrivilegedLearningTrainingConfigSchema,
    "dtn2s": DTN2STrainingConfigSchema,
}

#: (block, knob, the literal default the STRATEGY falls back to).
#: Read off the strategy source, not the schema — that is the direction that
#: catches a drifted default. `fmri_kspace_strategies.py:55-57,133-135`;
#: `sfc_conformal_kspace_strategies.py:59-63,140-143`.
STRATEGY_FALLBACKS = [
    ("spatiotemporal_adaptive_sfc_recon", "lambda_mu", 0.01),
    ("spatiotemporal_adaptive_sfc_recon", "lambda_t", 0.01),
    ("beltrami_epi_distortion", "t_esp", 0.5e-3),
    ("beltrami_epi_distortion", "lambda_mu", 0.01),
    ("adaptive_sfc_hssc", "grid_size", 32),
    ("adaptive_sfc_hssc", "in_channels", 1),
    ("adaptive_sfc_hssc", "lambda_mu", 0.01),
    ("adaptive_sfc_hssc", "lambda_traj", 0.01),
    ("conformal_diffusion_recon", "dc_alpha", 1.0),
    ("conformal_diffusion_recon", "sigma_min", 1e-2),
    ("conformal_diffusion_recon", "sigma_max", 1.0),
    ("conformal_mrf_dictless_recon", "lambda_conformality", 1e-2),
    ("crlb_mrf_pulse_design", "lambda_beltrami", 1e-3),
    ("cross_scanner_mrf_harmonisation", "lambda_anchor", 1e-3),
    ("bloch_equivariant_translation", "t1_scale_ulf", 0.30),
    ("bloch_equivariant_translation", "tr_ulf", 8.0),
    ("bloch_equivariant_translation", "flip_angle_deg", 15.0),
    ("bloch_equivariant_translation", "lambda_bloch_eq", 1.0),
    ("bloch_equivariant_translation", "unpaired_patch", 32),
    ("ib_active_acquisition", "beta", 1.0),
    ("riemannian_bloch_diffusion", "t_min", 1e-3),
    ("riemannian_bloch_diffusion", "metric_cache_resolution", 0),
    ("riemannian_bloch_diffusion", "metric_refresh_interval", 50),
    ("privileged", "alpha", 1.0),
    ("privileged", "beta", 0.5),
    ("privileged", "delta", 0.1),
    ("privileged", "curriculum_tau0", 5.0),
    ("privileged", "min_keep_fraction", 0.0),
    ("privileged", "initial_keep_fraction", 1.0),
    ("privileged", "disc_hidden_dim", 128),
    ("dtn2s", "receptive_window", 8),
]


class TestTheBlocksAreActuallyMounted:
    @pytest.mark.parametrize("block", sorted(BLOCKS))
    def test_block_coerces_to_its_schema(self, block: str) -> None:
        """A raw dict here is the whole defect — `getattr` would return defaults."""
        arrived = getattr(TrainingStrategyConfigSchema(**{block: {}}), block)
        assert not isinstance(arrived, dict), (
            f"training.{block} arrives as a raw dict, so every "
            f"getattr(cfg, knob, default) in its strategy returns the DEFAULT "
            f"and the arm's declared value is discarded."
        )
        assert isinstance(arrived, BLOCKS[block])

    @pytest.mark.parametrize("block", sorted(BLOCKS))
    def test_a_declared_value_survives(self, block: str) -> None:
        knob, value = next(
            (k, v)
            for b, k, d in STRATEGY_FALLBACKS
            if b == block
            for v in [
                d + 1 if isinstance(d, int) and not isinstance(d, bool) else d * 2
            ]
        )
        arrived = getattr(TrainingStrategyConfigSchema(**{block: {knob: value}}), block)
        assert getattr(arrived, knob) == value


class TestDefaultsMatchTheStrategyFallbacks:
    """Declaring the block must not change an arm that does not declare it.

    Every default is the value the strategy already falls back to, so this batch
    is a no-op for the arms it does not touch. A drifted default here would
    silently change those arms — the mirror of the bug being fixed.
    """

    @pytest.mark.parametrize("block,knob,fallback", STRATEGY_FALLBACKS)
    def test_schema_default_equals_strategy_fallback(
        self, block: str, knob: str, fallback: object
    ) -> None:
        assert BLOCKS[block]().__getattribute__(knob) == fallback, (
            f"{block}.{knob} defaults to "
            f"{BLOCKS[block]().__getattribute__(knob)!r} but the strategy falls "
            f"back to {fallback!r}. An arm that omits the block would change "
            f"behaviour, which this batch must not do."
        )


class TestTyposRaiseInsteadOfSittingUnread:
    """`extra="forbid"` is what converts the old silent discard into an error."""

    @pytest.mark.parametrize("block", sorted(BLOCKS))
    def test_unknown_knob_raises(self, block: str) -> None:
        with pytest.raises(ValidationError):
            BLOCKS[block](an_invented_knob_no_schema_declares=1)

    def test_out_of_range_raises(self) -> None:
        """A bound that never ran is a spec with zero percent execution."""
        with pytest.raises(ValidationError):
            BeltramiEPIDistortionTrainingConfigSchema(t_esp=0.0)  # gt=0
        with pytest.raises(ValidationError):
            AdaptiveSFCHSSCTrainingConfigSchema(grid_size=0)  # ge=1


class TestTheAblationAblates:
    """The assertion this batch exists for, on the real corpus arms.

    `idea_1_spatial_only_sfc` declares `lambda_t: 0.0` and its baseline declares
    `0.01`. Before the block was declared BOTH resolved to 0.01, so the two arms
    were runtime-identical and any measured delta between them was noise. The
    ablation's own YAML comment reasons explicitly about de-confounding.
    """

    ABLATION = "experiments/inprogress/fmri_2026/ablations/idea_1_spatial_only_sfc.yaml"
    BASELINE = (
        "experiments/inprogress/fmri_2026/idea_1_spatiotemporal_adaptive_sfc.yaml"
    )
    ECHO_SPACING = (
        "experiments/inprogress/fmri_2026/ablations/idea_2_long_echo_spacing.yaml"
    )

    @staticmethod
    def _load(path: str):
        from pathlib import Path

        from spectramr.config.settings import TrainingSettings

        p = Path(path)
        if not p.exists():
            pytest.skip(f"{path} not present (curated branch?)")
        return TrainingSettings.from_yaml(str(p))

    def test_the_two_arms_resolve_different_lambda_t(self) -> None:
        ablation = self._load(self.ABLATION).training.spatiotemporal_adaptive_sfc_recon
        baseline = self._load(self.BASELINE).training.spatiotemporal_adaptive_sfc_recon
        assert ablation is not None and baseline is not None
        assert ablation.lambda_t != baseline.lambda_t, (
            "the ablation and its own baseline resolve the SAME lambda_t, so the "
            "arm does not ablate anything and no comparison between them means "
            "anything. This is the defect the 2026-08 batch fixes."
        )
        assert ablation.lambda_t == 0.0 and baseline.lambda_t == 0.01

    def test_the_shared_knob_is_still_shared(self) -> None:
        """Control: the pair must differ in EXACTLY one knob, or the ablation is
        confounded for a different reason (pitfall #17)."""
        ablation = self._load(self.ABLATION).training.spatiotemporal_adaptive_sfc_recon
        baseline = self._load(self.BASELINE).training.spatiotemporal_adaptive_sfc_recon
        assert ablation.lambda_mu == baseline.lambda_mu

    def test_echo_spacing_resolves_to_what_the_arm_declares(self) -> None:
        """`t_esp` IS that arm's stated axis; it ran at the 5e-4 default."""
        cfg = self._load(self.ECHO_SPACING).training.beltrami_epi_distortion
        assert cfg is not None
        assert cfg.t_esp == 1.0e-3, (
            f"resolved t_esp={cfg.t_esp}, but the arm declares 1.0e-3. At the "
            "0.5e-3 default this 'long echo spacing' arm has the SHORT spacing."
        )


class TestMRFArmsResolveWhatTheyDeclare:
    """The mrf cohort. Only one of its three arms actually changes.

    `idea_4_strict_sar_bound` and `idea_5_tight_reparameterisation` happen to
    declare exactly the default, so the fix is a runtime no-op for them — it
    restores the MEANING of the declaration, not its value. That distinction is
    worth pinning: a test that only checked "it loads" would not tell them apart.
    """

    @staticmethod
    def _block(path: str, block: str):
        from pathlib import Path

        from spectramr.config.settings import TrainingSettings

        p = Path(path)
        if not p.exists():
            pytest.skip(f"{path} not present (curated branch?)")
        return getattr(TrainingSettings.from_yaml(str(p)).training, block)

    def test_conformality_weight_is_the_declared_one(self) -> None:
        """The one arm that changes: declared 0.05, ran at the 0.01 default."""
        cfg = self._block(
            "experiments/inprogress/mrf_2026/ablations/"
            "idea_3_overcompressed_embedding.yaml",
            "conformal_mrf_dictless_recon",
        )
        assert cfg is not None and cfg.lambda_conformality == 0.05, (
            "the arm declares lambda_conformality: 0.05; at the 0.01 default its "
            "conformality term is 5x under-weighted"
        )

    @pytest.mark.parametrize(
        "path,block,knob,value",
        [
            (
                "experiments/inprogress/mrf_2026/ablations/idea_4_strict_sar_bound.yaml",
                "crlb_mrf_pulse_design",
                "lambda_beltrami",
                0.001,
            ),
            (
                "experiments/inprogress/mrf_2026/ablations/"
                "idea_5_tight_reparameterisation.yaml",
                "cross_scanner_mrf_harmonisation",
                "lambda_anchor",
                0.001,
            ),
        ],
    )
    def test_declared_equals_default_arms_are_unchanged(
        self, path: str, block: str, knob: str, value: float
    ) -> None:
        cfg = self._block(path, block)
        assert cfg is not None and getattr(cfg, knob) == value


class TestPhysicsEqSSLIsDeliberatelyNotMounted:
    """`physics_eq_ssl` solved this the OTHER way and must stay a dict.

    `physics_equivariant_ssl_strategy.py:138-160` asserts the block IS a dict,
    reads it with `pe.get(...)`, and does its own typo rejection. Mounting a
    typed block would make that `isinstance(pe, dict)` guard fail and raise
    `TypeError` on the one arm that uses it — so this batch, which is
    schema-only by design, must leave it alone.
    """

    def test_it_still_arrives_as_a_dict(self) -> None:
        from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema

        arrived = TrainingStrategyConfigSchema(physics_eq_ssl={"latent_dim": 8})
        assert isinstance(arrived.physics_eq_ssl, dict), (
            "physics_eq_ssl now coerces to a schema, which BREAKS "
            "PhysicsEquivariantSSLStrategy: its isinstance(pe, dict) guard "
            "raises TypeError. Mounting it requires editing the strategy to "
            "read the typed block."
        )

    def test_the_arm_that_uses_it_still_loads(self) -> None:
        from pathlib import Path

        from spectramr.config.settings import TrainingSettings

        p = Path("experiments/inprogress/novel_2026/idea_6_phys_eq_ssl_brain.yaml")
        if not p.exists():
            pytest.skip("arm not present (curated branch?)")
        cfg = TrainingSettings.from_yaml(str(p))
        assert isinstance(cfg.training.physics_eq_ssl, dict)


class TestRiemannianArmResolvesItsCacheResolution:
    def test_metric_cache_resolution_is_the_declared_one(self) -> None:
        """Declared 32; ran at the 0 default, i.e. metric caching disabled."""
        from pathlib import Path

        from spectramr.config.settings import TrainingSettings

        p = Path("experiments/inprogress/novel_2026/idea_5_riemannian_qmap_brain.yaml")
        if not p.exists():
            pytest.skip("arm not present (curated branch?)")
        blk = TrainingSettings.from_yaml(str(p)).training.riemannian_bloch_diffusion
        assert blk is not None and blk.metric_cache_resolution == 32


class TestPreviouslyBypassedSpecsAreNowMounted:
    """The six specs that existed and never ran.

    They were written, reviewed and correct, and zero percent of them executed --
    `BYPASSED_BLOCKS` was their inventory. Mounting them needed a `defer_build`
    plus deferred imports, because four of their modules import back into
    `training/base.py` for a different class in the same file.
    """

    ARMS: ClassVar[dict[str, str]] = {
        "se3_equivariant_navigator": "experiments/inprogress/vf/exp_p3_b3_se3_navigator.yaml",
        "twin_dps": "experiments/inprogress/vf/exp_vf_twin_dps_v2.yaml",
        "ib_vf": "experiments/inprogress/vf/exp_vf_ib_infonce_v2.yaml",
        "hamiltonian_acquisition": "experiments/inprogress/vf/exp_p7_b4_hamiltonian_acquisition.yaml",
        "bloch_manifold_dps": "experiments/inprogress/vf/exp_p2_b2_bloch_manifold_dps.yaml",
        "equivariance_conformal": "experiments/inprogress/vf/exp_p1_b1_equivariance_conformal.yaml",
    }

    @pytest.mark.parametrize("block", sorted(ARMS))
    def test_the_arm_resolves_a_typed_block(self, block: str) -> None:
        from pathlib import Path

        from spectramr.config.settings import TrainingSettings

        p = Path(self.ARMS[block])
        if not p.exists():
            pytest.skip("arm not present (curated branch?)")
        arrived = getattr(TrainingSettings.from_yaml(str(p)).training, block)
        assert arrived is not None and not isinstance(arrived, dict), (
            f"training.{block} still arrives as a raw dict; its spec exists but "
            f"does not run."
        )

    def test_the_deferred_rebuild_precedes_the_stage_rebuild(self) -> None:
        """Ordering is load-bearing, and silently so.

        `StageEnvironmentSchema.model_rebuild()` references
        `TrainingStrategyConfigSchema` and therefore forces its schema to build.
        If the deferred imports run after it, `defer_build` has already been
        defeated and class creation raises `PydanticUndefinedAnnotation`.
        """
        from pathlib import Path

        src = Path("src/spectramr/config/schemas/training/base.py").read_text()
        # Anchor on a newline: the explanatory comment above the deferred block
        # names `StageEnvironmentSchema.model_rebuild()` in prose and would
        # otherwise match first.
        deferred = src.index("\nTrainingStrategyConfigSchema.model_rebuild()")
        stage = src.index("\nStageEnvironmentSchema.model_rebuild(")
        assert deferred < stage, (
            "the deferred sub-block rebuild must precede "
            "StageEnvironmentSchema.model_rebuild(), which forces the schema"
        )
