"""Regression pins for ConcreteMotionMetaTrainingStrategy.

F-kspace-real (smoke audit 2026-06-13): the svd-coil ``experiment_vf_hyper_mamba_
meta`` arm delivers a *k-space* target (``dataset_type: kspace`` + svd coil-
processing → 2-channel real-interleaved k-space). The strategy applies the
kinematic motion operator + the reconstruction loss + cached visuals in image
domain, so the target MUST be IFFT'd to image domain first. The pre-fix code
asserted "target_complex is already an image" and dropped the IFFT — false for
k-space data: the REAL reference then rendered as raw |k-space| and the FAKE
collapsed to black.

Instantiating the strategy needs a full TrainingEnvironment + a CUDA NUFFT
kinematic operator, so the contract is pinned via source-text assertions plus a
behavioural check of the inherited ``_ensure_image_domain_target`` seam on CPU.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import torch

from spectramr.infrastructure.training.strategies.motion_meta_strategy import (
    ConcreteMotionMetaTrainingStrategy,
)

# Anchored to this file, not to the CWD. A bare ``pathlib.Path("src/...")``
# resolves against the process working directory, and this read happens at
# MODULE level -- so launching pytest from anywhere but the repo root raises
# FileNotFoundError during collection, and a collection error is not a test
# failure: pytest discards the whole session. Running from ``tests/`` was
# measured to abort the run with "Interrupted: 2 errors during collection",
# taking every unrelated test with it.
_SRC_PATH = (
    pathlib.Path(__file__).resolve().parents[5]
    / "src/spectramr/infrastructure/training/strategies/motion_meta_strategy.py"
)
_SRC = _SRC_PATH.read_text(encoding="utf-8")


def test_both_paths_convert_target_to_image_domain() -> None:
    """_compute_losses_impl AND validation_step must route the target through the
    k-space->image seam before the kinematic operator / loss / visuals."""
    assert _SRC.count("self._ensure_image_domain_target(target_complex)") >= 2, (
        "both _compute_losses_impl and validation_step must IFFT a k-space target"
    )


def test_stale_already_an_image_claim_is_gone() -> None:
    """The misleading comment that justified dropping the IFFT must not return."""
    assert "target_complex is already an\n        # an image" not in _SRC
    assert "already\n        # an image" not in _SRC
    # The specific false justification string must be absent.
    assert "the prior ifft2c(fft2c(.)) was an identity round-trip" not in _SRC


def test_conversion_precedes_kinematic_op() -> None:
    """In the training path the conversion happens before the corruption."""
    conv = _SRC.index("self._ensure_image_domain_target(target_complex)")
    kin = _SRC.index("self.kinematic_op(target_complex, theta)")
    assert conv < kin, "must convert the target to image domain BEFORE corrupting it"


def test_inherited_seam_iffts_svd_kspace_target() -> None:
    """Behavioural check of the inherited helper under a svd/kspace config."""
    s = ConcreteMotionMetaTrainingStrategy.__new__(ConcreteMotionMetaTrainingStrategy)
    s.config = SimpleNamespace(
        model=SimpleNamespace(model_type="hyper_mamba_unet", target_domain="image"),
        data=SimpleNamespace(
            dataset_type="kspace",
            normalize_kspace=False,
            output_domain="image",
            coil_processing_mode="svd",
        ),
        physics=SimpleNamespace(kspace=SimpleNamespace(enable_kspace_recon=False)),
    )
    from spectramr.infrastructure.physics.fft_ops import ifft2c

    kspace = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
    out = s._ensure_image_domain_target(kspace)
    assert torch.allclose(out, ifft2c(kspace), atol=1e-6)


# ---------------------------------------------------------------------------
# Loss-weight SSOT (issue #1918 comment: "go through the losses, strategies,
# mixins and update them to the new paradigm").
#
# ``_setup_strategy_specific_components`` used to read
# ``self.config.losses.reconstruction.lambda_{l1,ssim,hfen}`` directly. That
# surface is the *category* paradigm; an arm on the *domain* paradigm declares
# ``losses.image_losses: [{name: l1, weight: ...}]`` and no ``reconstruction:``
# block at all -- but the schema still hands back a fully-defaulted
# ``ReconstructionLossesConfig``, so the raw read returned SCHEMA DEFAULTS and
# silently ignored the arm.
#
# The plant below is what gives these tests power: the one live arm
# (``experiment_vf_hyper_mamba_meta_v2``) declares l1 at exactly 10.0, which is
# also ``lambda_l1``'s default, so a config built from the arm agrees on both
# surfaces and cannot discriminate. The config here declares 3.7/0.25/0.5 --
# values no default can produce -- so a regression to the raw read reads
# 10.0/0.0/0.0 and fails loudly.
# ---------------------------------------------------------------------------


def _disagreeing_losses_config():
    """A domain-paradigm losses block whose weights differ from every default."""
    from spectramr.config.schemas.loss import LossComponentConfig, LossConfigSchema

    return LossConfigSchema(
        output_domain="image",
        image_losses=[
            LossComponentConfig(name="l1", weight=3.7, enabled=True),
            LossComponentConfig(name="ssim", weight=0.25, enabled=True),
            LossComponentConfig(name="hfen", weight=0.5, enabled=True),
        ],
    )


def test_the_two_surfaces_actually_disagree() -> None:
    """Guard the guard: if these ever agree, the test below is vacuous.

    A domain-only arm gets a defaulted category block back, so the raw read
    yields lambda_l1's default 10.0 and 0.0 for the two terms that have no
    default -- never the 3.7/0.25/0.5 the arm declared.
    """
    losses = _disagreeing_losses_config()
    assert losses.reconstruction is not None, "schema stopped defaulting the block"
    raw = (
        losses.reconstruction.lambda_l1,
        losses.reconstruction.lambda_ssim,
        losses.reconstruction.lambda_hfen,
    )
    assert raw == (10.0, 0.0, 0.0), f"defaults moved: {raw}"


def test_setup_resolves_weights_through_the_loss_weight_ssot(monkeypatch) -> None:
    """The strategy must apply the arm's DECLARED weights, not category defaults."""
    import spectramr.infrastructure.training.strategies.motion_meta_strategy as mod

    identity = torch.nn.Identity()
    monkeypatch.setattr(mod, "KinematicForwardOperator", lambda **kw: identity)
    monkeypatch.setattr(mod, "VirtualFiducial", lambda **kw: identity)
    monkeypatch.setattr(mod, "create_loss", lambda name: identity)

    s = ConcreteMotionMetaTrainingStrategy.__new__(ConcreteMotionMetaTrainingStrategy)
    # Only ``config.losses`` participates in the assertion and it is a REAL
    # schema object; the rest is inert scaffolding the method happens to touch.
    s.device = torch.device("cpu")
    s.config = SimpleNamespace(
        data=SimpleNamespace(sampling=SimpleNamespace(patch_size=[32, 32])),
        training=SimpleNamespace(),
        losses=_disagreeing_losses_config(),
    )

    s._setup_strategy_specific_components()

    assert (s._lambda_l1, s._lambda_ssim, s._lambda_hfen) == (3.7, 0.25, 0.5), (
        "weights must come from the loss-weight SSOT (declared_loss_weights); "
        f"got {(s._lambda_l1, s._lambda_ssim, s._lambda_hfen)} -- (10.0, 0.0, 0.0) "
        "means the raw losses.reconstruction.lambda_* read is back"
    )


def test_undeclared_term_is_zero_not_a_default() -> None:
    """An arm that declares none of the three gets 0.0, never lambda_l1's 10.0."""
    import spectramr.infrastructure.training.strategies.motion_meta_strategy as mod
    from spectramr.config.schemas.loss import LossComponentConfig, LossConfigSchema

    identity = torch.nn.Identity()
    losses = LossConfigSchema(
        output_domain="image",
        image_losses=[LossComponentConfig(name="l2", weight=1.0, enabled=True)],
    )
    s = ConcreteMotionMetaTrainingStrategy.__new__(ConcreteMotionMetaTrainingStrategy)
    s.device = torch.device("cpu")
    s.config = SimpleNamespace(
        data=SimpleNamespace(sampling=SimpleNamespace(patch_size=[32, 32])),
        training=SimpleNamespace(),
        losses=losses,
    )
    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(mod, "KinematicForwardOperator", lambda **kw: identity)
        mp.setattr(mod, "VirtualFiducial", lambda **kw: identity)
        mp.setattr(mod, "create_loss", lambda name: identity)
        s._setup_strategy_specific_components()

    assert (s._lambda_l1, s._lambda_ssim, s._lambda_hfen) == (0.0, 0.0, 0.0)


def test_raw_category_lambda_read_is_gone_from_source() -> None:
    """Belt-and-braces name pin, so a partial revert is visible in review."""
    assert "self.config.losses.reconstruction" not in _SRC
    assert "declared_loss_weights(self.config)" in _SRC
