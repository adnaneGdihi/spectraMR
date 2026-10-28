"""Behavioural wiring tests: does setting a knob actually change what gets built?

Acceptance is the weakest possible signal. A key can validate, survive
``extra="forbid"``, appear in provenance, and still be discarded before it
reaches the object that acts on it. ``optimization.optimizer.betas`` does exactly that: it
has a ``@field_validator``, and ``_create_optimizer`` reads ``beta1``/``beta2``
instead, so a declared ``(0.5, 0.6)`` trains at ``(0.5, 0.999)``.

Static analysis cannot see that. These tests build the real artifact twice, once
per knob value, and assert the artifact differs. The knob is wired if and only
if the thing it configures changes.

Each table below carries its own control group of knobs known to be wired. Those
are not padding: they are what proves the harness can detect wiring at all. A
suite where every case is expected to fail cannot distinguish "knob is dead"
from "harness is broken".

``WIRED`` cases must pass. ``FACADE`` cases are ``xfail(strict=True)``, so when
someone wires one up the suite fails on the unexpected pass and forces the table
to be corrected. The table is therefore the living inventory of which knobs are
known not to work.

Companion to ``test_schema_key_consumption.py``, which asks the cheaper question
(is the key referenced at all). This one asks whether the reference does
anything.
"""

from __future__ import annotations

import types

import pytest
import torch

WIRED = "wired"
FACADE = "facade"


def _optimizer_for(**knobs):
    """Build a real optimizer through the production builder path."""
    from spectramr.config.schemas.optimization import OptimizationConfigSchema
    from spectramr.infrastructure.builders.context import BuilderContext
    from spectramr.infrastructure.training.builders.optimization_builder import (
        OptimizationBuilder,
    )

    config = OptimizationConfigSchema(
        **{"learning_rate": 1e-3, "optimizer_type": "adam", **knobs}
    )
    # _create_optimizer only reads self._config to hand it to OptimizerBuilder;
    # a stub keeps the test off the full TrainingSettings (which needs model+data).
    context = BuilderContext(
        config=types.SimpleNamespace(optimization=config, training=None)
    )
    builder = OptimizationBuilder(context)
    return builder._create_optimizer(torch.nn.Linear(2, 2), config).param_groups[0]


# (knob, value_a, value_b, param_group_key, status)
OPTIMIZER_KNOBS = [
    ("learning_rate", 1e-3, 7e-5, "lr", WIRED),
    ("weight_decay", 0.0, 0.03, "weight_decay", WIRED),
    ("beta1", 0.9, 0.55, "betas", WIRED),
    ("beta2", 0.999, 0.777, "betas", WIRED),
    # Was FACADE: validated by a @field_validator, then dropped because
    # _create_optimizer built betas from beta1/beta2 and never read this field.
    # Wired by the optimizer-SSOT work — resolve_optimizer_spec._resolve_betas
    # now reconciles the two homes (and raises when both are declared and differ).
    ("betas", (0.9, 0.999), (0.5, 0.6), "betas", WIRED),
]


@pytest.mark.parametrize(
    "knob,value_a,value_b,group_key,status",
    OPTIMIZER_KNOBS,
    ids=[f"{k}-{s}" for k, _, _, _, s in OPTIMIZER_KNOBS],
)
def test_optimizer_knob_reaches_the_optimizer(
    knob, value_a, value_b, group_key, status
):
    """Changing the knob must change the optimizer it configures."""
    if status == FACADE:
        pytest.xfail(
            f"optimization.{knob} is declared and validated but never reaches the optimizer"
        )

    before = _optimizer_for(**{knob: value_a})[group_key]
    after = _optimizer_for(**{knob: value_b})[group_key]
    assert before != after, (
        f"optimization.{knob}: setting {value_a!r} and {value_b!r} both produced "
        f"param_group[{group_key!r}]={before!r}. The knob validates but does not "
        f"reach the optimizer, so every arm that sets it trains on the default."
    )


def test_betas_is_not_overridden_by_beta1_beta2_defaults():
    """Regression pin for the exact failure mode, now fixed.

    ``betas`` used to be discarded and the optimizer rebuilt from the
    ``beta1``/``beta2`` *defaults*, so ``betas=(0.5, 0.6)`` silently trained at
    ``(0.5, 0.999)`` — the second element came from ``beta2``'s default. The
    parametrised case above only proves the knob *moves* the optimizer; this one
    proves it arrives **unmodified**, which is what the bug actually broke.
    """
    group = _optimizer_for(betas=(0.5, 0.6))
    assert group["betas"] == (0.5, 0.6), (
        f"optimization.optimizer.betas=(0.5, 0.6) reached the optimizer as {group['betas']!r}. "
        "A second element of 0.999 means beta2's default is overriding the "
        "declared pair again (the pre-SSOT bug)."
    )


def _augmentation_transforms(**knobs) -> list[str]:
    """Class names of the transforms the production factory actually composes."""
    from spectramr.config.schemas.augmentation import AugmentationConfigSchema
    from spectramr.data.transforms.augmentation_factory import TorchIOAugmentationFactory

    built = TorchIOAugmentationFactory.build(
        AugmentationConfigSchema(enabled=True, **knobs)
    )
    if built is None:
        return []
    return [type(t).__name__ for t in getattr(built, "transforms", [built])]


# (flag, transform it claims to add, status)
AUGMENTATION_FLAGS = [
    ("enable_flip", "RandomFlip", WIRED),
    ("enable_ghosting", "RandomGhosting", WIRED),
    # The factory's intensity section used to be a placeholder comment
    # ("# ... (unchanged) ...") and every flag below composed nothing. Wired on
    # branch fix/wire-dead-augmentation-knobs; the characterisation test that
    # pinned the broken behaviour is deleted with this change, as its own
    # message instructed.
    ("enable_gamma", "RandomGamma", WIRED),
    ("enable_noise", "RandomNoise", WIRED),
    ("enable_brightness", "RandomBrightness", WIRED),
    ("enable_contrast", "RandomContrast", WIRED),
    # Section 3 of the same factory was elided the same way.
    ("enable_bias_field", "RandomBiasField", WIRED),
    ("enable_rician_noise", "RandomRicianNoise", WIRED),
    ("enable_motion_blur", "RandomMotionBlur", WIRED),
    ("enable_blur", "RandomBlur", WIRED),
]


@pytest.mark.parametrize(
    "flag,transform,status",
    AUGMENTATION_FLAGS,
    ids=[f"{f}-{s}" for f, _, s in AUGMENTATION_FLAGS],
)
def test_augmentation_flag_adds_a_transform(flag, transform, status):
    """Enabling an augmentation must add a transform to the composed pipeline."""
    if status == FACADE:
        pytest.xfail(
            f"data.augmentation.{flag} composes no transform; the data is never perturbed"
        )

    baseline = _augmentation_transforms()
    enabled = _augmentation_transforms(**{flag: True})
    added = [t for t in enabled if t not in baseline]
    assert added, (
        f"data.augmentation.{flag}=True added no transform (expected something like "
        f"{transform}; pipeline is {enabled}). Arms enabling it train on unperturbed "
        f"data, so any ablation against it is a byte-identical run."
    )
    # Name the transform, not just "something changed": the elision this test
    # now guards was a whole missing SECTION, so a neighbouring flag leaking a
    # transform into the diff would otherwise read as this flag being wired.
    assert transform in added, (
        f"data.augmentation.{flag}=True added {added}, which does not include "
        f"{transform}. Either the flag composes the wrong transform, or another "
        f"flag is defaulting on and masking this one."
    )


def test_harness_detects_wiring():
    """The controls must actually discriminate.

    If this fails, every other result in the file is meaningless: it would mean
    the harness cannot see a change even when one happens.
    """
    assert (
        _optimizer_for(learning_rate=1e-3)["lr"]
        != _optimizer_for(learning_rate=7e-5)["lr"]
    )
    assert "RandomFlip" in _augmentation_transforms(enable_flip=True)
    assert "RandomFlip" not in _augmentation_transforms()


# --------------------------------------------------------------------------
# Paradigm sub-blocks: is the block's schema the thing that validates it?
# --------------------------------------------------------------------------
#
# TrainingStrategyConfigSchema is extra="allow". A sub-block it does not declare
# (or declares loosely) is stored as the raw dict it arrived as, so the block's
# own schema -- its bounds, its extra="forbid", its frozen=True -- never runs.
# The spec is written, reviewed and correct, and zero percent of it executes.
#
# COERCED blocks must keep coercing. BYPASSED blocks are the inventory of specs
# that exist but are not integrated; mounting one flips its case to failing here
# and forces the table to be updated.

COERCED_BLOCKS = [
    "diffusion",
    "gan",
    "vae",
    "latent",
    # 2026-08 batch. These four had NO schema at all, so `extra="allow"` stored
    # them as raw dicts and every getattr(cfg, knob, default) in their strategy
    # returned the DEFAULT. Two were an ablation's only axis: idea_1_spatial_only_sfc
    # declared lambda_t: 0.0 and ran at 0.01, identical to its own baseline.
    "spatiotemporal_adaptive_sfc_recon",
    "beltrami_epi_distortion",
    "adaptive_sfc_hssc",
    "conformal_diffusion_recon",
    "conformal_mrf_dictless_recon",
    "crlb_mrf_pulse_design",
    "cross_scanner_mrf_harmonisation",
    "bloch_equivariant_translation",
    "ib_active_acquisition",
    "riemannian_bloch_diffusion",
    "privileged",
    "dtn2s",
    # Specs that already EXISTED and were merely never mounted -- these seven were
    # in BYPASSED_BLOCKS until 2026-08-02. Mounting them needed a `defer_build`
    # + deferred-import fix, because four of their modules import back into
    # training/base.py for a different class in the same file.
    "se3_equivariant_navigator",
    "twin_dps",
    "ib_vf",
    "hamiltonian_acquisition",
    "bloch_manifold_dps",
    "equivariance_conformal",
    "phys_residual_conformal",
    "motion",
]

BYPASSED_BLOCKS = [
    "ssl",
    "tto",
    "federated",
    "meta_learning",
    "spectra_tta",
    "operator_id",
]


#: Blocks whose schema has REQUIRED fields, so `{}` cannot construct them. The
#: minimum payload is not padding -- it is what the block genuinely needs, and a
#: coercion test that skipped them would silently stop covering these two blocks.
COERCED_BLOCK_MINIMUM: dict[str, dict] = {
    "twin_dps": {"marker_template_path": "/tmp/marker.pt"},
    "phys_residual_conformal": {"tr_ms": 3000.0, "te_ms": 90.0},
}


def _training_block(block: str, payload: dict):
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema

    merged = {**COERCED_BLOCK_MINIMUM.get(block, {}), **payload}
    return getattr(TrainingStrategyConfigSchema(**{block: merged}), block, None)


@pytest.mark.parametrize("block", COERCED_BLOCKS)
def test_declared_block_is_coerced_to_its_schema(block):
    """Control group: these blocks really are validated by their own schema."""
    coerced = _training_block(block, {})
    assert not isinstance(coerced, dict), (
        f"training.{block} used to coerce to its schema and now arrives as a raw "
        f"dict. Its declared constraints have stopped running."
    )


@pytest.mark.parametrize("block", BYPASSED_BLOCKS)
def test_bypassed_block_arrives_as_a_raw_dict(block):
    """Pin the known-bad state: the block's schema does not validate the block.

    Asserting the *defect* rather than xfailing keeps the inventory explicit and
    makes a fix show up as a failure with a clear instruction.
    """
    arrived = _training_block(block, {"an_invented_key_no_schema_declares": True})
    assert isinstance(arrived, dict), (
        f"training.{block} now coerces to a real schema rather than a raw dict. "
        f"That is the desired fix: move it from BYPASSED_BLOCKS to "
        f"COERCED_BLOCKS."
    )


def test_motion_block_rejects_values_its_own_schema_forbids():
    """MotionConfig bounds max_translation to (0, 128] and sets extra="forbid"; both must run."""
    out_of_bounds = {"max_translation": 99999.0, "totally_invented_key": True}

    with pytest.raises(Exception, match="max_translation"):  # pydantic ValidationError
        _training_block("motion", out_of_bounds)


# --------------------------------------------------------------------------
# Acceleration: does the knob survive the schema and change the physics?
# --------------------------------------------------------------------------
#
# AccelerationConfigSchema is extra="ignore". A knob it does not declare is
# dropped at load with no error, no warning and no provenance trace, so the
# runtime silently falls back to a default. acceleration.min_center_fraction
# was dropped exactly that way: 47 arms carried the #534 ladder fix in YAML,
# the process plumbing was correct, the CI gate was green, and every arm still
# ran a ladder that topped out at 12.2x instead of the declared 32x.
#
# Acceptance would not have caught it (extra="ignore" accepts anything) and a
# read-tracer would not either (nothing was there to read). Only building the
# degradation operator twice and comparing the realised ladder catches it.

EXP11_ACCELERATION = {
    "acceleration_type": "equispaced",
    "schedule_type": "step",
    "base_acceleration": 2.0,
    "max_acceleration": 32.0,
    "center_fraction": 0.08,
    "acceleration_range": [2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0],
}


def _realised_ladder(**knobs) -> list[float]:
    """Effective R at each timestep, resolved the way the runtime resolves it."""
    from spectramr.config.schemas.acceleration import AccelerationConfigSchema
    from spectramr.models.diffusion.kspace_process import (
        KSpaceUndersamplingProcess,
        resolve_undersampling_kwargs,
    )

    config = AccelerationConfigSchema(**{**EXP11_ACCELERATION, **knobs})
    process = KSpaceUndersamplingProcess(
        num_timesteps=28, **resolve_undersampling_kwargs(config)
    )
    return [
        effective
        for _t, _nominal, effective, _kept in process.describe_ladder((256, 256))
    ]


def test_min_center_fraction_changes_the_realised_ladder():
    """The knob must reach the degradation operator, not just the config."""
    static = _realised_ladder()
    shrinking = _realised_ladder(min_center_fraction=0.02)
    assert static != shrinking, (
        "acceleration.min_center_fraction=0.02 produced the same acceleration "
        f"ladder as leaving it unset ({static}). The knob is dropped before it "
        "reaches KSpaceUndersamplingProcess, so every arm that sets it trains "
        "and validates on a collapsed ladder while its YAML says otherwise."
    )


def test_min_center_fraction_makes_the_top_rungs_distinct():
    """Pin the physics, not just 'something changed'.

    With a static ACS the 8% centre band IS the whole sampling budget at
    R=12.5, so R=16 and R=32 realise the same mask: 12 of 28 timesteps are
    physically identical, and val_robust_mri_psnr_32x measures 12.2x.
    """
    static = _realised_ladder()
    shrinking = _realised_ladder(min_center_fraction=0.02)

    assert static[20] == pytest.approx(static[24], abs=1e-6)
    assert static[24] == pytest.approx(12.19, abs=0.1)

    assert shrinking[20] == pytest.approx(16.0, rel=0.05)
    assert shrinking[27] == pytest.approx(32.0, rel=0.05)


def test_acceleration_knob_harness_discriminates():
    """Control: the ladder differs when a knob known to be wired changes.

    ``center_fraction`` is the sibling of the knob under test and sets the same
    ACS band, so it exercises the identical path. Note ``base_acceleration``
    would NOT work here: under ``schedule_type: step`` with an explicit
    ``acceleration_range`` the rungs come from the range alone.
    """
    assert _realised_ladder()[24] != _realised_ladder(center_fraction=0.04)[24]
