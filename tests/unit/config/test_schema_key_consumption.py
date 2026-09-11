"""Repo-wide ratchet on unconsumed config keys.

``test_physics_config_execution.py`` pins nine ``SCHEMA_ONLY`` sub-configs by
hand, for the ``enable_*`` flag of one block. This generalises that idea to
every key reachable from :class:`TrainingSettings` -- pitfall #15 ("every
exposed knob is wired") applies to all 2500+ of them, not just physics.

The inventory below is *technical debt*, not a specification. Each entry is a
key that experiment YAMLs set and that no non-schema module references. Wiring
one up is a fix: delete it from the list. Adding a new one fails the ratchet.

Zero references is a candidate, not proof -- aliasing validators (see
a ``mode='before'`` aliasing validator), ``model_dump()`` splatting
and ``getattr`` dispatch all consume a field without naming it. Every entry
here was confirmed by reading its would-be consumer; if you are adding one,
confirm it the same way rather than trusting the scan.

**A reference is not an execution.** The ripgrep index below answers "does any
line mention this key?", never "can that line run?" -- which is how
``logging.tracking.tensorboard_dir`` passed as consumed while its only reader
sat in a class the bootstrap never constructs (#928 / #932). The rg half is kept
because it is what the ratchet above was measured with;
:class:`TestReachabilityAwareConsumption` asks the harder question through
:func:`spectramr.config.key_reachability.is_key_reachable`, and needs no ``rg`` at
all -- which is why the ripgrep guard is now per-class rather than module-wide,
and why it **fails** instead of skipping (see :func:`_require_ripgrep`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from spectramr.config.key_reachability import ReachabilityVerdict, is_key_reachable
from spectramr.config.settings import TrainingSettings
from spectramr.models.losses.weights import accessor_read_paths

TOOLS = Path(__file__).resolve().parents[3] / "tools" / "audit"
sys.path.insert(0, str(TOOLS))

#: Opt out of the loud failure below, for a dev box without ripgrep. Deliberately
#: opt-*out* and env-gated: unlike ``SPECTRAMR_ALLOW_MAMBA_FALLBACK`` --
#: which ``conftest.py`` sets ambient-on for every test session via
#: ``setdefault`` -- this var is NOT set anywhere in the tracked ``conftest*.py``
#: files (see ``test_ripgrep_opt_out_is_not_set_ambient_by_conftest`` below,
#: which pins that). A developer on a box without ``rg`` has to set it
#: themselves; nothing here defaults the degraded mode on.
RIPGREP_OPT_OUT = "SPECTRAMR_ALLOW_MISSING_RIPGREP"


def _require_ripgrep() -> None:
    """Missing ``rg`` must FAIL this gate, not quietly skip it.

    The rg half of this module -- including ``KNOWN_UNCONSUMED``, the repo's
    primary dead-knob ratchet -- cannot run without ripgrep. It used to
    ``skipif``, so on a box without ``rg`` (this one, as it happens) the ratchet
    reported ``s`` and nothing raised. A gate that silently cannot fail is the
    same defect family as #928 itself, and non-negotiable #3 says an unavailable
    dependency is raised, never degraded around.
    """
    if shutil.which("rg") is not None:
        return
    if os.environ.get(RIPGREP_OPT_OUT) == "1":
        pytest.skip(f"ripgrep absent; degraded run accepted via {RIPGREP_OPT_OUT}=1")
    pytest.fail(
        "ripgrep is not installed, so the rg-backed consumer index cannot be "
        "built and KNOWN_UNCONSUMED is not being checked at all. This used to "
        "skip, which made the ratchet unfalsifiable (#928). Install ripgrep, or "
        f"set {RIPGREP_OPT_OUT}=1 to accept the gap knowingly. "
        "TestReachabilityAwareConsumption needs no ripgrep and still runs."
    )


@pytest.fixture
def ripgrep_available() -> None:
    _require_ripgrep()


def test_ripgrep_opt_out_is_not_set_ambient_by_conftest() -> None:
    """``RIPGREP_OPT_OUT`` must stay asked-for, never conftest-defaulted.

    ``SPECTRAMR_ALLOW_MAMBA_FALLBACK`` is NOT the asked-for precedent the comment
    above ``RIPGREP_OPT_OUT`` used to claim: ``conftest.py`` sets it ambient-on
    for every test session via ``os.environ.setdefault(...)``, so it is on by
    default and a developer has to opt *out* of the fallback, not into it. If
    a future edit made ``SPECTRAMR_ALLOW_MISSING_RIPGREP`` do the same -- default
    it on from a ``conftest*.py`` -- the loud failure in ``_require_ripgrep``
    would go quiet exactly the way #928 is about, without this test's own
    assertion message noticing.

    This scans tracked ``conftest*.py`` SOURCE FILES for the variable name --
    not ``os.environ`` at runtime -- because a legitimate developer opt-out
    (``SPECTRAMR_ALLOW_MISSING_RIPGREP=1`` in their own shell) is exactly the
    asked-for use this var exists for and must keep working; asserting the env
    var is unset at runtime would fail for that legitimate case instead of
    only for an ambient conftest default.
    """
    repo_root = Path(__file__).resolve().parents[3]
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", "conftest.py", "**/conftest.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    conftest_files = [repo_root / p for p in tracked if p.endswith(".py")]
    assert conftest_files, "expected at least the root conftest.py to be tracked"

    offenders = [
        path for path in conftest_files if RIPGREP_OPT_OUT in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"{RIPGREP_OPT_OUT} is mentioned in tracked conftest file(s), which "
        "would default the ripgrep opt-out on for every test session -- the "
        "same ambient-default shape SPECTRAMR_ALLOW_MAMBA_FALLBACK actually has, "
        "which this variable is deliberately NOT supposed to copy:\n  "
        + "\n  ".join(str(p) for p in offenders)
    )


#: The rg index is one half of this module, not the whole of it. Applying this
#: per class rather than module-wide is what lets the reachability ratchet --
#: pure-Python AST work -- keep running when ripgrep is gone.
requires_ripgrep = pytest.mark.usefixtures("ripgrep_available")


#: Entries the rg index scores CONSUMED, whose only non-schema mention is a
#: comment.  `build_index` is a raw identifier regex over `*.py` with no comment
#: stripping, so naming a dead knob in prose marks it consumed -- 8 configured
#: keys repo-wide, measured with a token index that keeps string literals (so
#: `getattr(cfg, "x")` still counts) and drops comments and docstrings. -> #1409
#:
#: These must NOT be pruned by `test_the_unconsumed_ratchet_only_shrinks`: the
#: ledger is right and the index is wrong.  The set shrinks to empty when #1409
#: lands.
CONSUMED_ONLY_BY_PROSE: frozenset[str] = frozenset(
    {
        # Sole non-schema mention: a comment at
        # models/generators/convolution_variants.py:151.
        "physics.iterative_refinement.residual_learning",
    }
)


def resolved_unconsumed_entries(
    known: frozenset[str],
    index: dict,
    still_dead: set[str],
    prose_exempt: frozenset[str],
) -> set[str]:
    """Ledger entries that now have a consumer, so must be pruned.

    Module-level and parameterised so the planted violations in
    ``TestTheLedgerDetectorsActuallyFail`` exercise this predicate rather than
    a copy of it (non-negotiable 15).
    """
    return {
        key
        for key in known
        if key in index and index[key]["yaml_uses"] > 0 and key not in still_dead
    } - prose_exempt


def misfiled_as_no_read(section: frozenset[str], unreachable: dict) -> dict[str, tuple]:
    """Entries claiming "no read at all" that in fact have read sites."""
    return {
        key: unreachable[key].sites
        for key in section
        if key in unreachable and unreachable[key].sites
    }


def misfiled_as_unreachable_read(section: frozenset[str], unreachable: dict) -> list[str]:
    """Entries claiming "a read nothing reaches" that have no read at all."""
    return sorted(key for key in section if key in unreachable and not unreachable[key].sites)


# Keys set by at least one experiment YAML with no non-schema consumer.
KNOWN_UNCONSUMED: frozenset[str] = frozenset(
    {
        # The `acceleration:` block was renamed to `undersampling:` (renames.py,
        # posture=fold). These six were pinned under the old spelling, so the
        # rename unpinned them and they re-entered as "new" -- an exemption keyed
        # on a path silently lets go when the path moves. Only the three the
        # corpus still declares need an entry; the other three now count zero
        # arms path-exactly and fall out of `configured_only` on their own.
        "undersampling.mask_types",
        "undersampling.use_compile",  # -> #680
        "undersampling.use_distributed",  # -> #680
        "acquisition.active.query_strategy",
        "audit.ksd.bootstrap_replications",
        "checkpoint.pretrained_path",
        "checkpoint.resume_training",
        "checkpoint.save_on_epoch",
        "data.augmentation.cutmix_alpha",
        # Deferred to the physics lane, not dead by accident: wiring k-space
        # acceleration augmentation needs fft_ops.fft2c (non-negotiable 2) and
        # an applied-vs-declared entry in the snapshot provenance
        # (non-negotiable 14).  112 / 15 arms declare them, none set them true.
        # -> #1407, and tests/utils/augmentation_coverage.NOT_PER_SAMPLE, which
        # fails if the deferral is ever silently dropped.
        "data.augmentation.enable_kspace_undersampling_augmentation",
        "data.augmentation.undersampling_factor_augmentation",
        "data.augmentation.enable_cutmix",
        "data.augmentation.enable_mixup",
        "data.augmentation.mixup_alpha",
        "data.caching.cleanup_trigger",
        "data.caching.staging_behavior",
        # No read anywhere in src/ outside its own declaration; 3 arms declare
        # it, none set it true.  data.py's own prose cites this field as the
        # example of a knob whose tidy schema home implies it works. -> #1406
        "data.use_async_dataloader",
        "data.known_dataset",
        "data.multi_contrast.negative_sampling",
        "data.multi_contrast.positive_sampling",
        "data.test_split",
        # Nothing in src/ names `adaptation_rate` at all. It looked consumed
        # until #1294 only because `TrainingState.initialize_ema` listed it in
        # an `ema_config` dict handed to
        # `models.utils.adaptive_ema.create_adaptive_ema_for_model` -- a module
        # deleted in ff0efff9f, so the import above that dict raised
        # ImportError on every run and a bare `except ImportError` swallowed it
        # (pitfall #16). #1294 removed the dead branch, so the key surfaces
        # here now: the facade was the only thing making it look read.
        # `EMAConfigSchema` is extra="forbid" and 31 reference arms declare it
        # (29 under `experiments/training/`, 2 schema-reference files under
        # `experiments/shared/`; none under `inprogress/`), so deleting the
        # field would fail all of them at load. The schema description says
        # NOT READ, and `validate_adaptive_ema_is_fully_wired` REFUSES the key
        # outright whenever `enable_adaptive_ema` is set -- so the arm that
        # would actually depend on it fails loud rather than silently ignoring
        # it. Wire it or retire it with a corpus migration; do not trust it.
        "ema.adaptation_rate",
        "ema.enable_ema_for_optimizer",
        "ema.enable_momentum_scaling",
        "logging.log_activations",
        "logging.log_validation_graphs",
        "logging.log_weights",
        "logging.progress_bar_enabled",
        "logging.progress_bar_no_progress",
        "logging.progress_bar_on_warning",
        "logging.save_images_per_epoch",
        # Refused, not merely inert (#675): a non-null value now RAISES at
        # construction (LoggingConfigSchema._refuse_deferred_wandb). Still
        # reachable=False / NO_READ_FOUND like the rest of this list -- the
        # census has no "refused" state -- but unlike its neighbours here,
        # this is deliberate and permanent, not debt awaiting a fix.
        "logging.wandb_entity",
        "logging.wandb_project",
        "losses.diffusion.enable_data_consistency",
        "losses.diffusion.enable_diffusion_amp",
        "losses.latent.latent_regularization_weight",
        "losses.latent.n_embeddings",
        "losses.latent.use_latent_regularization",
        "losses.physics.lambda_snr_preserving",
        "losses.pinn.enable_magnitude_tv",
        "losses.pinn.enable_pde",
        "losses.pinn.enable_pinn_dc",
        "losses.pinn.enable_unit_norm_coil",
        "losses.reconstruction.enable_marker_loss",
        # `losses.reconstruction.lambda_snr_preserving` was RETIRED here (#421):
        # it duplicated `losses.physics.lambda_snr_preserving`, both canonicalise
        # to `snr_preserving`, and a materialised config raised on the pair. The
        # `losses.physics.*` entry above STAYS -- the elected owner is itself
        # still unread, which is pre-existing non-negotiable 8 debt tracked
        # separately, not something this election introduced.
        "metrics.best_metric_mode",
        # rg blind-spot, NOT dead: `_extract_metrics_from_config`
        # (infrastructure/training/strategies/mixins/metrics_mixin.py) calls
        # `schema_flag_to_metric()`, which enumerates MetricsConfigSchema
        # model_fields and resolves every `compute_*` through
        # core/metrics/flag_map.metric_for_flag.  That resolver returns
        # `robust_mri_psnr`, which IS in MetricsRegistry -- so the flag is
        # genuinely honoured and names no token the index can see.  This is the
        # sanctioned move the assertion message describes: consumer read and
        # cited, not a name added to quiet a gate.
        "metrics.compute_robust_mri_psnr",
        # Genuinely dead, unlike its `compute_*` siblings above: it does not
        # start with `compute_`, so `schema_flag_to_metric` never sees it and
        # `metric_for_flag` raises on it.  211 arms declare it, 205 set it
        # true. -> #1405
        "metrics.enable_paradigm_specific",
        "metrics.compute_bland_altman_bias",
        "metrics.compute_coefficient_of_variation",
        "metrics.compute_folding_fraction",
        "metrics.compute_icc_3_1",
        "metrics.compute_limits_of_agreement_lower",
        "metrics.compute_limits_of_agreement_upper",
        "metrics.paradigm_specific_metrics",
        "metrics.track_best_metric",
        "model.architecture.bridge.pretrained_source",
        "model.architecture.decoder.pretrained_source",
        "model.architecture.encoder.pretrained_source",
        "model.decoder_component",
        "model.encoder_component",
        "model.trellis.gaussian_latent_dim",
        "model.trellis.gaussian_num_gaussians",
        "model.trellis.gaussian_resolution",
        "model.trellis.mesh_latent_dim",
        "model.trellis.mesh_num_faces",
        "model.trellis.mesh_num_vertices",
        "model.trellis_vae.decoder.attn_mode",
        "model.trellis_vae.decoder.representation_config",
        "model.trellis_vae.encoder.attn_mode",
        "model.trellis_vae.trainer.batch_size_per_gpu",
        "model.trellis_vae.trainer.batch_split",
        "model.trellis_vae.trainer.ema_rate",
        "model.trellis_vae.trainer.fp16_mode",
        "model.trellis_vae.trainer.fp16_scale_growth",
        "model.trellis_vae.trainer.grad_clip",
        "model.trellis_vae.trainer.i_log",
        "model.trellis_vae.trainer.i_sample",
        "model.trellis_vae.trainer.i_save",
        "mrf.phantom_calibration_path",
        "mrf.spiral_rotation_schedule",
        # Unmasked by the mixed-case scanner fix, not newly broken: the old
        # identifier pattern was lowercase-only and the old YAML counter matched
        # `^\s*([a-z_][a-z0-9_]*)\s*:`, so a key holding a capital was invisible
        # on BOTH sides and could never reach this ledger. Verified dead by hand;
        # `cleanup_interval`'s one hit is the unrelated hardcoded literal
        # `"memory_cleanup_interval": 100` at
        # infrastructure/services/memory_optimization_service.py:257.
        "optimization.memory.cleanup_interval",
        "training.bloch_manifold_dps.param_bounds_B1",
        "training.hamiltonian_acquisition.num_trajectories_per_batch",
        "training.se3_equivariant_navigator.bio_harmonic_freqs_Hz",
        "training.se3_equivariant_navigator.dc_navigator_sample_rate_Hz",
        "training.se3_equivariant_navigator.ib_beta_minus",
        "training.se3_equivariant_navigator.ib_beta_plus",
        "optimization.memory.enable_fragmentation_mitigation",
        "optimization.optimizer.generator_learning_rate",
        "optimization.memory.monitoring_interval",
        "optimization.memory.enable_batch_size_optimization",
        "physics.b0_correction.enable_b0_correction",
        "physics.b0_correction.estimate_from_data",
        "physics.b0_correction.num_shim_orders",
        "physics.bloch_simulation.target_te",
        "physics.bloch_simulation.target_tr",
        "physics.coil_sensitivity.calibration_region_size",
        "physics.coil_sensitivity.estimation_type",
        "physics.coil_sensitivity.num_threads",
        "physics.coil_sensitivity.precomputed_type",
        "physics.compressed_sensing.convergence_tolerance",
        "physics.compressed_sensing.reconstruction_algorithm",
        "physics.compressed_sensing.sparsity_transform",
        "physics.compressed_sensing.threshold_scaling",
        "physics.compressed_sensing.use_adaptive_threshold",
        "physics.concomitant.waveform_uri",
        "physics.iterative_refinement",
        "physics.iterative_refinement.residual_learning",
        "physics.kspace.zero_pad_reconstruction",
        "physics.motion_correction.reference_frame",
        "physics.phase_correction.enable_phase_estimation",
        "physics.phase_correction.phase_smoothing",
        "physics.regularization.enable_sparsity",
        "physics.regularization.enable_wavelets",
        "physics.regularization.l2_penalty",
        "physics.regularization.physics_penalty_weight",
        "physics.regularization.sparsity_type",
        "physics.regularization.sparsity_weight",
        "physics.regularization.tv_type",
        "physics.regularization.wavelet_weight",
        "physics.relaxation_priors.t1_table_uri",
        "physics.relaxation_priors.t2_table_uri",
        "training.bloch_synth.opaque_band",
        "training.bloch_synth.target_field_tesla",
        "training.cs_mno.architecture.lift_dim",
        "training.diffusion.degradation_dynamics",
        "training.diffusion.enable_diffusion_amp",
        "training.field_cocycle.triple_sampling",
        "training.gan.progan_gradient_penalty_lambda",
        "training.geomamba_ulf.topology.ph_dimensions",
        "training.geomamba_ulf.unwarp.emit_beltrami_diagnostic",
        "training.geomamba_ulf.unwarp.enable_grad_nonlinearity",
        "training.geomamba_ulf.unwarp.gradient_amplitudes_mt_per_m",
        "training.latent.latent_regularization_weight",
        "training.latent.n_embeddings",
        "training.latent.use_latent_regularization",
        "training.multi.stages.stage_config.loss.diffusion.enable_data_consistency",
        "training.multi.stages.stage_config.loss.diffusion.enable_diffusion_amp",
        "training.multi.stages.stage_config.loss.latent.latent_regularization_weight",
        "training.multi.stages.stage_config.loss.latent.n_embeddings",
        "training.multi.stages.stage_config.loss.latent.use_latent_regularization",
        "training.multi.stages.stage_config.loss.physics.lambda_snr_preserving",
        "training.multi.stages.stage_config.loss.pinn.enable_magnitude_tv",
        "training.multi.stages.stage_config.loss.pinn.enable_pde",
        "training.multi.stages.stage_config.loss.pinn.enable_pinn_dc",
        "training.multi.stages.stage_config.loss.pinn.enable_unit_norm_coil",
        "training.multi.stages.stage_config.loss.reconstruction.enable_marker_loss",
        # stage_config twin of the retired
        # `losses.reconstruction.lambda_snr_preserving` (#421); the
        # `...loss.physics.lambda_snr_preserving` entry above stays.
        "training.multi.stages.stage_config.metrics.best_metric_mode",
        "training.multi.stages.stage_config.metrics.compute_advanced_metrics",
        "training.multi.stages.stage_config.metrics.compute_bland_altman_bias",
        "training.multi.stages.stage_config.metrics.compute_coefficient_of_variation",
        "training.multi.stages.stage_config.metrics.compute_folding_fraction",
        "training.multi.stages.stage_config.metrics.compute_icc_3_1",
        "training.multi.stages.stage_config.metrics.compute_limits_of_agreement_lower",
        "training.multi.stages.stage_config.metrics.compute_limits_of_agreement_upper",
        "training.multi.stages.stage_config.metrics.paradigm_specific_metrics",
        "training.multi.stages.stage_config.metrics.track_best_metric",
        "training.multi.stages.stage_config.optimization.memory.enable_fragmentation_mitigation",
        "training.multi.stages.stage_config.optimization.optimizer.generator_learning_rate",
        "training.multi.stages.stage_config.optimization.memory.monitoring_interval",
        "training.multi.stages.stage_config.optimization.memory.safety_margin",
        "training.multi.stages.stage_config.optimization.memory.enable_batch_size_optimization",
        "training.multi.stages.stage_config.optimization.optimizer.param_groups",
        "training.spin_sde.dt",
        "training.vae.latent_regularization_weight",
        "training.vae.n_embeddings",
        "training.vae.use_latent_regularization",
        "validation.enable_validation_augmentation",
        # `validation.frequency_steps` was here until phase 10a. It did not get
        # wired -- it got MERGED: it and `eval_interval` were one cadence with
        # one default, 58 arms declared both and none disagreed, so the fold
        # sends both to `validation.schedule.interval_steps`, which the training
        # loop already read. The key is gone, so the entry would be stale.
        "validation.num_visualizations",
        "validation.use_training_loss",
        "validation.validation_dir",
        "validation.validation_metric",
        "validation.visualization_dir",
    }
)


@pytest.fixture(scope="module")
def index() -> dict:
    """The rg consumer index. Only the ``@requires_ripgrep`` classes request it.

    Guarded again here because fixture ordering against ``ripgrep_available`` is
    not guaranteed, and ``build_index`` would otherwise die inside ``subprocess``
    with a ``FileNotFoundError`` that names neither the cause nor the opt-out.
    """
    _require_ripgrep()

    from schema_key_consumption import build_index, census

    return build_index(census(TrainingSettings))


@pytest.fixture(scope="module")
def configured_keys() -> list[str]:
    """Every census key at least one experiment arm declares.

    Built from ``census`` + ``count_yaml_declarations`` rather than from the rg
    ``index`` fixture on purpose: the reachability ratchet must be able to run
    where ripgrep is not installed, which is where the rg half of this module
    silently contributes nothing.
    """
    from schema_key_consumption import census, count_yaml_declarations

    counts = count_yaml_declarations(Path(__file__).resolve().parents[3] / "experiments")
    keys = {row["key"] for row in census(TrainingSettings)}
    return sorted(key for key in keys if counts.get(key, 0) > 0)


@pytest.fixture(scope="module")
def unreachable(configured_keys: list[str]) -> dict[str, ReachabilityVerdict]:
    """Configured keys whose every read was positively shown unable to run.

    The loss-weight accessor's read set is injected (#1925). ``is_key_reachable``
    finds a read by its field name as a source *token*, and
    ``build_loss_weight_table`` names none of the 112 paths it reads -- it builds
    every name at runtime from ``LAMBDA_SECTIONS`` and ``model_fields_set``. 52 of
    them therefore looked exactly like "nothing reads this". They are declared
    beside the reader, checked against what the reader executes by
    ``test_weights.py``, and passed in here because ``config/`` may not import
    ``models/`` (NN5). This test file is the one place that legitimately sees both.
    """
    external_reads = accessor_read_paths()
    return {
        key: verdict
        for key in configured_keys
        if not (verdict := is_key_reachable(key, external_reads=external_reads)).reachable
    }


class TestCensusWalksTheLiveTree:
    """The census must come from pydantic, never from a grep or a static list.

    Marked per test, not per class: the third case exercises ``iter_submodels``
    on live annotations and needs no consumer index, so failing it for a missing
    ripgrep would be a false alarm.
    """

    @requires_ripgrep
    def test_census_reaches_nested_blocks(self, index: dict) -> None:
        # A grep-built inventory misses these; the live walk cannot.
        for key in (
            "optimization.optimizer.learning_rate",
            "data.augmentation.enable_flip",
        ):
            assert key in index, f"{key} unreachable -- census stopped descending"

    @requires_ripgrep
    def test_census_covers_every_top_level_block(self, index: dict) -> None:
        blocks = {key.split(".")[0] for key in index}
        assert {
            "data",
            "training",
            "losses",
            "physics",
            "model",
            "optimization",
        } <= blocks

    def test_optional_and_list_annotations_are_unwrapped(self) -> None:
        from schema_key_consumption import iter_submodels

        from spectramr.config.schemas.optimization import OptimizationConfigSchema

        assert iter_submodels(OptimizationConfigSchema | None) == [OptimizationConfigSchema]
        assert iter_submodels(list[OptimizationConfigSchema]) == [OptimizationConfigSchema]
        assert iter_submodels(int) == []


@requires_ripgrep
class TestUnconsumedKeysRatchet:
    """The set of dead-but-configured knobs must not grow."""

    def test_no_new_unconsumed_keys(self, index: dict) -> None:
        from schema_key_consumption import unconsumed

        current = set(unconsumed(index, configured_only=True))
        new = current - KNOWN_UNCONSUMED
        assert not new, (
            f"{len(new)} config key(s) are set by experiment YAMLs but read by "
            f"nothing outside config/schemas/:\n  "
            + "\n  ".join(sorted(new))
            + "\n\nEvery exposed knob must be read, validated and stamped into "
            "provenance (pitfall #15). Either wire it up, or -- if the scan is "
            "wrong because the field is aliased/splatted/getattr-dispatched -- "
            "confirm the real consumer and add the key to KNOWN_UNCONSUMED with "
            "the consumer cited."
        )

    def test_the_unconsumed_ratchet_only_shrinks(self, index: dict) -> None:
        """A key that became consumed must leave the list, not linger.

        The reachability half has carried ``test_the_ratchet_only_shrinks``
        since it was written; the rg half had no counterpart, so an entry here
        only ever had to stop existing in the schema to be pruned -- never to
        stop being dead. Ten entries had drifted by 2026-08-22, six of them
        long before this change: ``data.return_image_domain``,
        ``metrics.compute_advanced_metrics``, ``optimization.memory.safety_``
        ``margin``, ``optimization.optimizer.param_groups``,
        ``physics.multi_acquisition.forward_psf.mu`` and four ``data.``
        ``augmentation.*`` keys. A ratchet that cannot shrink is a list, and a
        list of things that used to be broken is not evidence of anything.
        """
        from schema_key_consumption import unconsumed

        still_dead = set(unconsumed(index, configured_only=True))
        resolved = resolved_unconsumed_entries(
            KNOWN_UNCONSUMED, index, still_dead, CONSUMED_ONLY_BY_PROSE
        )
        assert not resolved, (
            f"{len(resolved)} key(s) in KNOWN_UNCONSUMED now have a non-schema "
            "consumer -- delete them, so the list keeps meaning "
            "'dead' rather than 'was dead once':\n  "
            + "\n  ".join(
                f"{key}  -> {index[key]['consumer_files'][:2]}" for key in sorted(resolved)
            )
            + "\n\nIf the only mention is a comment, the index is wrong rather "
            "than the ledger (#1409): add the key to CONSUMED_ONLY_BY_PROSE "
            "with the comment's file:line, do not delete it here."
        )

    def test_the_prose_exemption_has_no_stale_entries(self, index: dict) -> None:
        """``CONSUMED_ONLY_BY_PROSE`` shrinks to empty when #1409 lands."""
        from schema_key_consumption import unconsumed

        still_dead = set(unconsumed(index, configured_only=True))
        stale = {k for k in CONSUMED_ONLY_BY_PROSE if k in still_dead}
        assert not stale, (
            "these keys are no longer scored consumed by the index, so the "
            f"prose exemption is dead weight -- delete it:\n  {sorted(stale)}"
        )

    def test_ratchet_list_has_no_stale_entries(self, index: dict) -> None:
        """Entries that no longer exist in the schema must be pruned."""
        stale = {key for key in KNOWN_UNCONSUMED if key not in index}
        assert not stale, (
            f"KNOWN_UNCONSUMED names {len(stale)} key(s) that the schema no longer "
            f"declares -- delete them:\n  " + "\n  ".join(sorted(stale))
        )


class TestReachabilityAwareConsumption:
    """The same question, asked of the call graph instead of the text (#928).

    ``KNOWN_UNCONSUMED`` above is the rg answer: a key is consumed when some
    line mentions it. The keys below pass *that* bar and fail this one -- their
    reads either sit in a scope nothing can reach, or are not reads at all
    (a prose comment, a docstring, a module named after the same leaf).

    This is a **ratchet on findings, not a suppression list -- for keys not
    already tracked elsewhere.** ``test_no_new_unreachable_reads`` below also
    exempts anything already in ``KNOWN_UNCONSUMED``: those keys are
    pre-existing rg-tracked debt, and re-litigating each one here at the same
    time the rg gate above is measuring it would just double-count the same
    finding under two ratchets. That exemption is a **fixed, already-measured
    delta list**, not a live escape hatch -- it does not grow when a key is
    unreachable for a *new* reason. A key that is unreachable and NOT already
    in ``KNOWN_UNCONSUMED`` must be wired up, deleted, or added to
    ``KNOWN_UNREACHABLE_READS`` with its consumer read and cited -- moving it
    into ``KNOWN_UNCONSUMED`` instead would convert the finding this gate
    exists to produce back into silence, the exact move #928 is about, and is
    not a legitimate way to satisfy this test. Both lists may only shrink, by
    wiring the knob up, deleting it, or (if the analysis is wrong) proving the
    real consumer and citing it here.

    Two failure shapes are pinned together because both are "text, not
    reachability", and the verdict distinguishes them:

    ``sites=()``
        No read anywhere in ``src/spectramr/`` outside ``config/schemas/``. rg
        counted a comment (``losses.gan.enable_gradient_penalty``), a docstring
        that explicitly says the key is *not* read
        (``physics.coil_sensitivity.estimation_method``), or an unrelated module
        with the same name (``physics.bloch_simulation``).

    ``sites`` non-empty, none live
        A real read in a scope the analysis positively showed cannot run.

    Caveat carried over from the rg index unchanged: a field consumed by
    ``model_dump()`` splatting or by ``getattr`` over ``model_fields`` names no
    token, so it lands here even though it is read. The ``metrics.compute_*``
    entries are the likely instances. That is a triage prompt, not a licence to
    delete -- read the consumer before acting on one.
    """

    #: SHRINK ONLY. See the class docstring before touching it.
    #:
    #: Split into two named sets on 2026-08-22.  The single frozenset carried
    #: both states under comment headings, which meant a key could drift from
    #: "a real read nothing reaches" to "no read at all" -- or back -- by an
    #: invisible move between two comment blocks.  The two sets are verified
    #: against the live reachability index below, so a drifting key now has to
    #: be RE-CLASSIFIED explicitly, in a diff a reviewer can see.
    #:
    #: Keys with no read at all in ``src/spectramr/`` outside ``config/schemas/``.
    #: This is the same category ``KNOWN_UNCONSUMED`` holds, so a key belongs
    #: here only while its reachability verdict is the finding being tracked;
    #: once it is triaged it moves to the rg-debt ledger with a citation.
    NO_READ_AT_ALL: frozenset[str] = frozenset(
        {
            "backend_acceleration.cosim_bridge_addr",
            "data.caching",
            "losses.gan.enable_gradient_penalty",
            "metrics.compute_precision_recall",
            "physics.bloch_simulation",
            "physics.coil_sensitivity.estimation_method",
            "physics.phase_correction.estimation_method",
            "physics.pinn.boundary_condition",
            "physics.relaxation_priors",
            "training.cs_mno",
            "training.multi.stages.stage_config.loss.gan.enable_gradient_penalty",
        }
    )

    #: A real read, in a scope nothing reaches.  These are the entries a
    #: launderer would want to bury -- the key has live text, so moving it to
    #: the rg-debt ledger would convert a positive finding into silence, the
    #: move #928 is about.  ``test_the_unreachable_read_section_is_sealed``
    #: below makes that move cost a second, visible edit.
    UNREACHABLE_READ: frozenset[str] = frozenset(
        {
            # 855 arms declare `metric_interval`. Its only reader is
            # `IterationCounterService.should_compute_metrics`, whose name is
            # referenced from no live scope -- only from tests/. The schema
            # description asserts it "is called" from the training loop.
            "metrics.metric_interval",
            # Named at scheduler_system.py:505 (a signature default), :522 and
            # :546 -- all inside `AdaptiveWarmupScheduler`, an unrelated
            # learning-rate scheduler nothing constructs. The leaf names merely
            # collide; no EMA code reads it. Its sibling `ema.adaptation_rate`
            # sits in KNOWN_UNCONSUMED (no textual read at all); see the note
            # there for why both surfaced with #1294.
            "ema.stability_threshold",
            # Read at memory_optimization_service.py:226 / :357 / :382. Every
            # read is either a signature default of `NoOpMemoryOptimizationService`
            # or inside `MemoryOptimizationService.optimize_batch_size`, whose
            # name is referenced from no live scope -- non-negotiable 16's shape:
            # existing capability, unwired. Surfaced when the entry was pruned
            # from KNOWN_UNCONSUMED, where it had been masking a real read.
            "optimization.memory.safety_margin",
            # The second loss accessor, not yet declared (#1925 covers only the
            # first). The single token site, `loss_recommender.py:523`, is not
            # even a read -- it is a keyword ARGUMENT constructing a
            # `ReconstructionLossesConfig`, inside `LossRecommender`, a class
            # nothing constructs. The real read is
            # `LossConfigSchema.get_enabled_losses`, which builds the field name
            # at runtime (`enable_field = f"enable_{loss_name}"` at loss.py:1735,
            # :1745, :1761, :1773, :1794) and is live from `loss_builder.py:229`
            # and `training_loop.py:1078`. Measured, not inferred: with
            # `enable_l1` ON, moving `lambda_l1` moves the returned dict; with it
            # OFF, nothing moves -- so the flag is read and it gates a weight.
            #
            # It is NOT in `accessor_read_paths()` on purpose. That export is
            # derived from the two constants `build_loss_weight_table` iterates,
            # so it cannot drift from its reader. `get_enabled_losses` has no
            # such constant: its enable-field mapping is ~85 lines of inline
            # `elif` literals inside a nested closure, and deriving a read set
            # from it means hoisting that table out of a live loss resolver --
            # a refactor of what trains, not of what audits. Declared here with
            # its consumer cited instead; delete this entry when that accessor
            # declares its own reads.
            "losses.reconstruction.enable_l1",
        }
    )

    #: The union both ratchet tests below consume. Membership in either half
    #: exempts a key; which half it is in is what the two verification tests
    #: pin.
    KNOWN_UNREACHABLE_READS: frozenset[str] = NO_READ_AT_ALL | UNREACHABLE_READ

    def test_the_two_sections_are_disjoint(self) -> None:
        """A key cannot be both "no read" and "a read nothing reaches"."""
        overlap = self.NO_READ_AT_ALL & self.UNREACHABLE_READ
        assert not overlap, f"double-booked: {sorted(overlap)}"

    def test_no_read_at_all_entries_really_have_no_sites(
        self, unreachable: dict[str, ReachabilityVerdict]
    ) -> None:
        """The classification is checked against the live index, not asserted.

        A key that grows a read must be moved to ``UNREACHABLE_READ`` (or
        wired), explicitly. Before the split this transition was a comment
        move -- invisible in review and unchecked by anything.
        """
        misfiled = misfiled_as_no_read(self.NO_READ_AT_ALL, unreachable)
        assert not misfiled, (
            f"{len(misfiled)} key(s) in NO_READ_AT_ALL now have real read "
            "sites -- re-classify them into UNREACHABLE_READ with the reason "
            "their scope cannot run, or wire them:\n  "
            + "\n  ".join(f"{k}  {v[:2]}" for k, v in sorted(misfiled.items()))
        )

    def test_unreachable_read_entries_really_have_sites(
        self, unreachable: dict[str, ReachabilityVerdict]
    ) -> None:
        """The converse: an entry here claims a read exists. It must."""
        misfiled = misfiled_as_unreachable_read(self.UNREACHABLE_READ, unreachable)
        assert not misfiled, (
            f"{len(misfiled)} key(s) in UNREACHABLE_READ have no read sites at "
            "all -- they are plain rg debt; move them to NO_READ_AT_ALL, or to "
            f"KNOWN_UNCONSUMED with a citation:\n  {misfiled}"
        )

    def test_the_unreachable_read_section_is_sealed(self) -> None:
        """Pins UNREACHABLE_READ membership literally.

        These are the entries whose finding is a positive one: the read exists
        and cannot run. Silencing such a key by moving it to KNOWN_UNCONSUMED
        is the laundering the class docstring forbids, and a test cannot read
        git history to catch it. What this seal buys is review visibility --
        the move now costs two edits, one of them to a list whose docstring
        says why it exists, instead of one quiet deletion.

        Deleting an entry here is legitimate exactly when the key was WIRED.
        Say so in the commit message.

        `losses.reconstruction.enable_l1` was ADDED by #1925 rather than
        deleted, which the seal also makes visible. It is not a key being
        quieted: it moved here from the gate's live red set once the first loss
        accessor declared its reads, because its own reader -- the second
        accessor -- still declares nothing. The entry cites that reader.
        """
        assert (
            frozenset(
                {
                    "ema.stability_threshold",
                    "losses.reconstruction.enable_l1",
                    "metrics.metric_interval",
                    "optimization.memory.safety_margin",
                }
            )
            == self.UNREACHABLE_READ
        ), (
            "UNREACHABLE_READ changed. If a key was wired, update this seal in "
            "the same commit and say so. If it was moved to KNOWN_UNCONSUMED "
            "to quiet a gate, that is the #928 move -- revert it."
        )

    def test_the_sweep_found_keys_to_judge(self, configured_keys: list[str]) -> None:
        """Anti-vacuity: a gate that judged nothing would pass forever."""
        assert len(configured_keys) > 1000

    def test_no_new_unreachable_reads(self, unreachable: dict[str, ReachabilityVerdict]) -> None:
        new = {
            key: verdict
            for key, verdict in unreachable.items()
            if key not in self.KNOWN_UNREACHABLE_READS and key not in KNOWN_UNCONSUMED
        }
        assert not new, (
            f"{len(new)} config key(s) are set by experiment YAMLs and their "
            "only reads cannot execute:\n  "
            + "\n  ".join(
                f"{key}\n      sites: {verdict.sites[:3]}\n      {verdict.reason}"
                for key, verdict in sorted(new.items())
            )
            + "\n\nWire it up or delete it. Do NOT add it to KNOWN_UNCONSUMED "
            "to quiet this gate -- that list is exempted here only for keys "
            "already tracked as pre-existing rg debt, not as a back door for "
            "a new finding this gate produces; using it that way converts the "
            "finding back into silence, the exact move #928 is about. Add it "
            "to KNOWN_UNREACHABLE_READS instead, with its consumer read and "
            "cited, following that list's own rule."
        )

    def test_the_ratchet_only_shrinks(self, unreachable: dict[str, ReachabilityVerdict]) -> None:
        """A key that became reachable must leave the list, not linger.

        Without this the list would drift into a stale ledger and stop being
        evidence of anything -- the same rot ``test_ratchet_list_has_no_stale_
        entries`` guards for the rg half.
        """
        fixed = sorted(self.KNOWN_UNREACHABLE_READS - unreachable.keys())
        assert not fixed, (
            f"{len(fixed)} key(s) are now reachable (or no longer configured / no "
            f"longer in the census) -- delete them from KNOWN_UNREACHABLE_READS:"
            "\n  " + "\n  ".join(fixed)
        )

    def test_the_two_ledgers_do_not_overlap(self) -> None:
        """``KNOWN_UNREACHABLE_READS`` and ``KNOWN_UNCONSUMED`` name disjoint keys.

        ``test_no_new_unreachable_reads`` treats membership in either list as
        "already accounted for". That is only a safe design if no key is
        double-booked -- a key present in both would be indistinguishable from
        one whose reachability entry is dead weight, since the rg-based
        exemption alone would already keep it silent. Measured empty on
        2026-08-12; this pins it so the two ledgers cannot quietly grow into
        each other.
        """
        overlap = self.KNOWN_UNREACHABLE_READS & KNOWN_UNCONSUMED
        assert not overlap, (
            f"{len(overlap)} key(s) are listed in both KNOWN_UNREACHABLE_READS "
            "and KNOWN_UNCONSUMED -- the KNOWN_UNCONSUMED entry alone already "
            "exempts them from test_no_new_unreachable_reads, so the "
            "KNOWN_UNREACHABLE_READS entry is redundant; delete it there:\n  "
            + "\n  ".join(sorted(overlap))
        )


class TestYamlDeclarationCountsArePathExact:
    """``yaml_uses`` must identify a key by its DOTTED PATH, never its leaf name.

    The counter was built from ``rg '^\\s*([a-z_][a-z0-9_]*)\\s*:'`` over
    ``experiments/`` and looked up by ``row["leaf"]``, so every one of the ~40
    distinct ``.enabled`` paths in the schema reported the same 10417 -- the
    number of ``enabled:`` lines in the corpus. That figure is not cosmetic:
    ``unconsumed(..., configured_only=True)`` gates the ratchet below on
    ``yaml_uses > 0``, so a key nothing declares enters the "set by experiment
    YAMLs and read by nothing" list on the strength of a same-named leaf in an
    unrelated block.
    """

    def test_unrelated_paths_sharing_a_leaf_are_counted_separately(self, tmp_path: Path) -> None:
        from schema_key_consumption import count_yaml_declarations

        (tmp_path / "a.yaml").write_text("physics:\n  iterative_refinement:\n    enabled: true\n")
        (tmp_path / "b.yaml").write_text("data:\n  augmentation:\n    enabled: false\n")

        counts = count_yaml_declarations(tmp_path)

        assert counts["physics.iterative_refinement.enabled"] == 1
        assert counts["data.augmentation.enabled"] == 1

    def test_a_path_no_arm_declares_counts_zero(self, tmp_path: Path) -> None:
        """The defect that put six phantom keys into the ratchet."""
        from schema_key_consumption import count_yaml_declarations

        (tmp_path / "a.yaml").write_text("optimization:\n  use_compile: true\n")

        counts = count_yaml_declarations(tmp_path)

        assert counts.get("undersampling.use_compile", 0) == 0

    def test_a_key_is_counted_once_per_arm_not_once_per_occurrence(self, tmp_path: Path) -> None:
        """``yaml_uses`` is read as "arms declaring this"; the token counter
        summed occurrences, so a key repeated in one file inflated it."""
        from schema_key_consumption import count_yaml_declarations

        (tmp_path / "a.yaml").write_text(
            "training:\n"
            "  multi:\n"
            "    stages:\n"
            "      - stage_config:\n"
            "          metrics:\n"
            "            best_metric_mode: max\n"
            "      - stage_config:\n"
            "          metrics:\n"
            "            best_metric_mode: min\n"
        )

        counts = count_yaml_declarations(tmp_path)

        assert counts["training.multi.stages.stage_config.metrics.best_metric_mode"] == 1

    def test_a_legacy_spelling_counts_toward_its_canonical_key(self, tmp_path: Path) -> None:
        """A folded block still reaches the canonical key at runtime, so a
        path-exact count that ignored the fold would under-report it to zero and
        silently drop a live knob out of the ratchet."""
        from schema_key_consumption import count_yaml_declarations

        (tmp_path / "legacy.yaml").write_text("acceleration:\n  use_compile: false\n")

        counts = count_yaml_declarations(tmp_path)

        assert counts["undersampling.use_compile"] == 1


@requires_ripgrep
class TestReferenceScanSeesMixedCaseIdentifiers:
    """The consumer index must not be blind to a name with a capital letter.

    ``build_index`` bucketed references with ``rg '\\b[a-z_][a-z0-9_]{2,}\\b'``.
    That pattern cannot match ANY token containing an uppercase character:
    against ``lambda_M`` it consumes ``ambda_`` and then fails the trailing
    ``\\b`` because ``M`` is still a word character. So every MRI knob named
    after a physical quantity -- ``lambda_M``, ``param_bounds_T1_ms``,
    ``G_max_mT_per_m``, ``bio_harmonic_freqs_Hz`` -- scanned as zero-referenced
    no matter how many places read it, and the ratchet would call them dead.
    """

    def test_a_read_uppercase_key_is_not_reported_unconsumed(self, index: dict) -> None:
        # Read at infrastructure/training/strategies/twin_dps_strategy.py:65
        # (`self.lambda_M = float(td.lambda_M)`).
        entry = index["training.twin_dps.lambda_M"]
        assert entry["n_nonschema_refs"] > 0, (
            "lambda_M has a real consumer; a zero here means the identifier "
            "scan cannot see mixed-case names"
        )

    def test_the_scan_is_not_silently_lowercase_only(self, index: dict) -> None:
        # Read at strategies/bloch_manifold_dps_strategy.py:283.
        assert index["training.bloch_manifold_dps.param_bounds_T1_ms"]["n_nonschema_refs"] > 0


# ---------------------------------------------------------------------------
# An unread knob must SAY it is unread.
#
# The ledger above is a CI-side fact: it keeps the count from growing. It does
# nothing for the person reading the schema, or the Sphinx page generated from
# it, who sees `description="Save best model based on metric"` on a field that
# saves nothing. That reader is the one who declares it in an arm.
#
# Deleting these fields is the real fix and it is blocked on a corpus migration:
# `metrics`/`validation` are extra="forbid" and 922 / 391 / 209 / 136 / 60 arms
# declare them across active/ and validated/, trees CLAUDE.md scopes to separate
# owner decisions. Until then the description is where the truth lives.
# ---------------------------------------------------------------------------


class TestUnconsumedKnobsSaySoInTheirDescription:
    """The subset of the ledger whose dishonesty is most costly.

    Not the whole ledger: many entries are structural (a sub-block key that a
    builder reads reflectively), and this is a ratchet on the ones a user
    plausibly sets expecting an effect.
    """

    _DOCUMENTED = [
        (
            "spectramr.config.schemas.logging",
            "LoggingConfigSchema",
            "log_validation_graphs",
        ),
        (
            "spectramr.config.schemas.logging",
            "LoggingConfigSchema",
            "save_images_per_epoch",
        ),
        ("spectramr.config.schemas.ema", "EMAConfigSchema", "adaptation_rate"),
        ("spectramr.config.schemas.metrics", "MetricsConfigSchema", "best_metric_mode"),
        ("spectramr.config.schemas.metrics", "MetricsConfigSchema", "track_best_metric"),
        (
            "spectramr.config.schemas.validation",
            "ValidationConfigSchema",
            "num_visualizations",
        ),
        (
            "spectramr.config.schemas.validation",
            "ValidationConfigSchema",
            "visualization_dir",
        ),
        (
            "spectramr.config.schemas.validation",
            "ValidationConfigSchema",
            "use_training_loss",
        ),
    ]

    @pytest.mark.parametrize(("module", "cls_name", "field"), _DOCUMENTED)
    def test_the_description_admits_it_is_not_read(self, module, cls_name, field):
        import importlib

        cls = getattr(importlib.import_module(module), cls_name)
        desc = cls.model_fields[field].description or ""

        assert "NOT READ" in desc, (
            f"{cls_name}.{field} is in KNOWN_UNCONSUMED but its description "
            "does not say so — a reader of the schema has no way to know."
        )

    @pytest.mark.parametrize(("module", "cls_name", "field"), _DOCUMENTED)
    def test_it_is_still_in_the_ledger(self, module, cls_name, field):
        """Anti-vacuity, and the removal path.

        If someone WIRES one of these, the ledger entry goes away and this test
        goes red — which is the prompt to delete the 'NOT READ' note rather than
        leave a wired knob documented as inert.
        """
        block = module.rsplit(".", 1)[-1]
        assert f"{block}.{field}" in KNOWN_UNCONSUMED


def test_eval_on_epoch_is_invisible_to_the_scanner():
    """`metrics.eval_on_epoch` is unread, set by 568 arms, and NOT flagged.

    The consumption scan matches key NAMES in source text. `training_loop.py`
    binds a LOCAL variable called `eval_on_epoch` holding
    `validation.schedule.on_epoch` — a different key in a different block — and
    that coincidence is enough to make `metrics.eval_on_epoch` read as consumed.

    So the ratchet has a false-negative class: any unread key whose leaf name
    collides with a local variable anywhere in `src/` is invisible to it. This
    test pins the one instance found, so the blind spot is recorded rather than
    rediscovered. The knob's own description carries the NOT READ note.
    """
    from spectramr.config.schemas.metrics import MetricsConfigSchema

    desc = MetricsConfigSchema.model_fields["eval_on_epoch"].description or ""
    assert "NOT READ" in desc

    # The collision that hides it. If this ever stops being true, the scanner
    # will start flagging the key on its own and the xfail above can go.
    src = (
        Path(__file__).resolve().parents[3] / "src" / "spectramr" / "pipelines" / "training_loop.py"
    ).read_text()
    assert "eval_on_epoch = config.validation.schedule.on_epoch" in src


class TestTheLedgerDetectorsActuallyFail:
    """Non-negotiable 15: every detector added with the ledger split, watched
    failing on the violation shape it claims to catch.

    These need no ripgrep and no live tree -- the predicates above are pure
    functions of (ledger, index), which is why they were extracted. A plant
    that built its own copy of the predicate would prove nothing.
    """

    @staticmethod
    def _verdict(*sites: str):
        return SimpleNamespace(sites=tuple(sites))

    def test_it_catches_a_ledger_entry_that_became_consumed(self) -> None:
        """Shape 1 -- the drift that had gone unchecked for six entries."""
        index = {"a.b": {"yaml_uses": 3, "consumer_files": ["x.py"]}}
        resolved = resolved_unconsumed_entries(
            frozenset({"a.b"}), index, still_dead=set(), prose_exempt=frozenset()
        )
        assert resolved == {"a.b"}

    def test_it_leaves_a_still_dead_entry_alone(self) -> None:
        """Anti-false-positive: the predicate must not demand pruning a key
        that is still genuinely dead."""
        index = {"a.b": {"yaml_uses": 3, "consumer_files": []}}
        assert not resolved_unconsumed_entries(
            frozenset({"a.b"}), index, still_dead={"a.b"}, prose_exempt=frozenset()
        )

    def test_the_prose_exemption_suppresses_the_finding(self) -> None:
        """Shape 2 -- a key the index scores consumed on a comment alone (#1409)
        must be exempt, or the ledger loses a true finding to a false index."""
        index = {"a.b": {"yaml_uses": 3, "consumer_files": ["x.py"]}}
        assert not resolved_unconsumed_entries(
            frozenset({"a.b"}), index, still_dead=set(), prose_exempt=frozenset({"a.b"})
        )

    def test_it_catches_a_no_read_entry_that_grew_a_read(self) -> None:
        """Shape 3 -- the drift the split exists to make explicit."""
        unreachable = {"a.b": self._verdict("src/spectramr/x.py:1")}
        assert misfiled_as_no_read(frozenset({"a.b"}), unreachable) == {
            "a.b": ("src/spectramr/x.py:1",)
        }

    def test_it_passes_a_no_read_entry_with_no_sites(self) -> None:
        unreachable = {"a.b": self._verdict()}
        assert not misfiled_as_no_read(frozenset({"a.b"}), unreachable)

    def test_it_catches_an_unreachable_read_entry_with_no_read(self) -> None:
        """Shape 4 -- the converse drift, which would let plain rg debt sit in
        the sealed section and inherit its protection."""
        unreachable = {"a.b": self._verdict()}
        assert misfiled_as_unreachable_read(frozenset({"a.b"}), unreachable) == ["a.b"]

    def test_it_passes_an_unreachable_read_entry_with_a_read(self) -> None:
        unreachable = {"a.b": self._verdict("src/spectramr/x.py:1")}
        assert not misfiled_as_unreachable_read(frozenset({"a.b"}), unreachable)

    def test_the_seal_is_not_vacuous(self) -> None:
        """A seal over an empty set would pass forever."""
        assert len(TestReachabilityAwareConsumption.UNREACHABLE_READ) >= 3
        assert (
            TestReachabilityAwareConsumption.NO_READ_AT_ALL
            | TestReachabilityAwareConsumption.UNREACHABLE_READ
            == TestReachabilityAwareConsumption.KNOWN_UNREACHABLE_READS
        )
