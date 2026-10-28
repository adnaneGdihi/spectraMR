"""Regression tests for the dynamic-loss dispatcher in unified loss computers.

Background — the diffusion-validation `val_sense_adjoint_l1 = 0.0` bug
was a symptom of a wider architectural pattern:

* The `losses_dict` dynamic loop in unified GAN / VAE / diffusion-recon
  computers used to call ``loss_fn(pred, target)`` directly, without
  forwarding the strategy's physics/posterior kwargs (``smaps``,
  ``mask``, ``mu`` / ``logvar``, ``z``, ``coords``, …).
* Losses with non-``(pred, target)`` signatures (KL, BetaTCVAE,
  VQ losses, PDE residuals, smoothness regularisers) were either
  silently miscalled with image tensors interpreted as
  ``(mu, log_var)`` — yielding huge bogus values — or skipped via a
  ``try/except: logger.debug``, which CLAUDE.md #9 explicitly forbids.
* Failures were only logged at DEBUG, so misconfigurations were
  invisible during normal training.

These tests pin the new contract:

1. **Smaps forwarding** — `sense_adjoint_l1` registered in a GAN / VAE
   `losses_dict` must receive ``smaps`` and produce a non-zero value.
2. **KL exclusion** — KL / VQ / BetaTCVAE registered in a GAN dynamic
   loop must be SKIPPED with a warning, not silently miscalled.
3. **Failure visibility** — a loss that raises must trigger a
   ``logger.warning`` so the user notices.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
import torch
from torch import nn

from spectramr.models.losses.complex_losses import SENSEAdjointL1Loss
from spectramr.models.losses.computers.base import LossOutput
from spectramr.models.losses.computers.unified_gan import (
    UnifiedGANLossComputer,
    _NON_PRED_TARGET_LOSS_NAMES as GAN_BLOCKED,
)
from spectramr.models.losses.computers.unified_vae import (
    UnifiedVAELossComputer,
    _NON_PRED_TARGET_LOSS_NAMES as VAE_BLOCKED,
)
from spectramr.models.losses.kl_divergence import KLDivergenceLoss

# ---------------------------------------------------------------------------
# Minimal config fixtures (mirrors `tests/unit/test_loss_computers.py`)
# ---------------------------------------------------------------------------


class _Cfg:
    """Minimal config object with `losses` namespace for both GAN and VAE."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.optimization = type("opt", (), {"learning_rate": 1e-4, "use_amp": False})()
        # ``discriminator_component`` is a DECLARED field on
        # ``ModelConfigSchema`` and Pydantic materializes every declared field,
        # so a real config always carries it -- defaulted to None when the arm
        # configures no critic. A double that omits it asserts a schema shape
        # the schema does not have, and the elected critic-name owner raises on
        # it by design (#1921, non-negotiable 3).
        self.model = type(
            "m",
            (),
            {"model_type": "test", "in_channels": 2, "discriminator_component": None},
        )()
        # Deliberately bare: each test path resolves weights via _get_loss_weight
        # which falls through `getattr` and defaults to 1.0 when attributes are
        # missing.
        self.losses = type(
            "losses",
            (),
            {
                "reconstruction": type("r", (), {"lambda_l1": 0.0, "lambda_l2": 0.0})(),
                "adversarial": type("a", (), {"lambda_adv": 0.0})(),
                "perceptual": type("p", (), {"lambda_percep": 0.0})(),
                "kl": type("kl", (), {"lambda_kl": 0.0})(),
            },
        )()


# ---------------------------------------------------------------------------
# Cross-computer name-exclusion contract
# ---------------------------------------------------------------------------


def test_kl_is_in_both_blocked_lists() -> None:
    """The shared blocklist guarantees KL never reaches a (pred, target) loop."""
    for label, blocked in [("GAN", GAN_BLOCKED), ("VAE", VAE_BLOCKED)]:
        for name in ("kl", "kl_divergence", "vq_kl", "beta_tc_vae"):
            assert name in blocked, f"{label} must block {name!r}"


def test_blocked_lists_are_consistent() -> None:
    """GAN and VAE share the same exclusion set so the contract doesn't drift."""
    assert GAN_BLOCKED == VAE_BLOCKED


# ---------------------------------------------------------------------------
# KL via a GAN strategy must be SKIPPED, not miscalled
# ---------------------------------------------------------------------------


def test_unified_gan_skips_kl_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """If a GAN config registers KL, the dynamic loop must skip it with a warning.

    Previously: KL was silently miscalled as ``loss_fn(image_pred, image_target)``,
    treating the images as ``(mu, log_var)`` and producing garbage.
    """
    computer = UnifiedGANLossComputer(_Cfg(), device=torch.device("cpu"))
    pred = torch.randn(2, 4, 16, 16)
    target = torch.randn(2, 4, 16, 16)
    losses_dict = {"kl": KLDivergenceLoss(beta=1.0)}

    with caplog.at_level(
        logging.WARNING, logger="spectramr.models.losses.computers.unified_gan"
    ):
        out = computer.compute(pred, target, losses_dict=losses_dict)

    assert isinstance(out, LossOutput)
    assert "kl" not in out.components, (
        "KL must NOT be present in components — it must be skipped before "
        "reaching the (pred, target) call site."
    )
    assert any(
        "kl" in rec.message.lower() for rec in caplog.records
    ), "Skipping KL in the GAN dynamic loop must emit a warning."


# ---------------------------------------------------------------------------
# Smaps must reach `sense_adjoint_l1` through the GAN dynamic loop
# ---------------------------------------------------------------------------


class _DummyGANComputerCallable(nn.Module):
    """Wrapper that mimics how LossBuilder hands a SENSE-adjoint loss to a computer."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = SENSEAdjointL1Loss(reduction="mean")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        smaps: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.inner(pred, target, smaps=smaps)


def _real_stacked(B: int, C: int, H: int, W: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, 2 * C, H, W)


def _real_stacked_normed_smaps(
    B: int, C: int, H: int, W: int, seed: int
) -> torch.Tensor:
    raw = _real_stacked(B, C, H, W, seed)
    real = raw[:, 0::2]
    imag = raw[:, 1::2]
    rss = torch.sqrt((real**2 + imag**2).sum(dim=1, keepdim=True) + 1e-8)
    raw[:, 0::2] = real / rss
    raw[:, 1::2] = imag / rss
    return raw


def test_unified_gan_forwards_smaps_to_sense_adjoint_l1() -> None:
    """The new `_call_safe_loss` routing must pipe smaps through to the loss."""
    computer = UnifiedGANLossComputer(_Cfg(), device=torch.device("cpu"))
    pred = _real_stacked(2, 4, 32, 32, seed=0)
    target = _real_stacked(2, 4, 32, 32, seed=1)
    smaps = _real_stacked_normed_smaps(2, 4, 32, 32, seed=2)
    loss_fn = SENSEAdjointL1Loss(reduction="mean")

    out = computer.compute(
        pred,
        target,
        losses_dict={"sense_adjoint_l1": loss_fn},
        smaps=smaps,
    )
    assert (
        "sense_adjoint_l1" in out.components
    ), "sense_adjoint_l1 must be present in components when smaps is forwarded."
    assert out.components["sense_adjoint_l1"].detach().abs().item() > 1e-6, (
        "sense_adjoint_l1 with smaps forwarded via _call_safe_loss must produce "
        "a non-zero residual; the previous direct `loss_fn(pred, target)` path "
        "produced 0."
    )


def test_unified_gan_without_smaps_drops_sense_adjoint_l1_entirely() -> None:
    """Documents what actually happens today: the term VANISHES from the objective.

    Not an endorsement — a record. ``SENSEAdjointL1Loss`` raises on ``smaps=None``
    (its anti-DC-blob guard), the computer's ``except Exception`` catches it, warns,
    and SKIPS the loss. So ``sense_adjoint_l1`` is not zero — it is **absent**, and
    the total is computed without the fidelity term at all.

    The old assertion here (``component is not None`` and ``== 0.0``) pinned the
    pre-guard fallback and had been red on ``dev`` ever since the guard landed
    (issue #188).

    See ``test_unified_gan_must_not_silently_drop_a_failing_loss`` below for the
    behaviour this SHOULD have (issue #289).
    """
    computer = UnifiedGANLossComputer(_Cfg(), device=torch.device("cpu"))
    pred = _real_stacked(2, 4, 32, 32, seed=0)
    target = _real_stacked(2, 4, 32, 32, seed=1)
    loss_fn = SENSEAdjointL1Loss(reduction="mean")

    out = computer.compute(
        pred,
        target,
        losses_dict={"sense_adjoint_l1": loss_fn},
        # smaps deliberately absent
    )
    assert "sense_adjoint_l1" not in out.components, (
        "The guard raised and the computer skipped the loss, so the term is "
        "absent — not zero. If this now holds a value, the swallow was fixed: "
        "update this test and drop the xfail below."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "issue #289: the unified loss computers catch ANY loss exception and skip "
        "the term, so SENSEAdjointL1Loss's smaps=None guard is neutralised — the "
        "fidelity term contributes zero to the objective anyway, which is the exact "
        "DC-blob the guard exists to prevent. Remove this marker (strict=True will "
        "force you to) once the swallow re-raises."
    ),
)
def test_unified_gan_must_not_silently_drop_a_failing_loss() -> None:
    """A loss that cannot be computed is a CONFIG ERROR — it must not be skipped.

    Silently dropping the term trains a different objective than the one declared
    in the YAML, for the entire run, and still reports ``success: true``
    (CLAUDE.md #9 / #10 / #16).
    """
    computer = UnifiedGANLossComputer(_Cfg(), device=torch.device("cpu"))
    pred = _real_stacked(2, 4, 32, 32, seed=0)
    target = _real_stacked(2, 4, 32, 32, seed=1)
    loss_fn = SENSEAdjointL1Loss(reduction="mean")

    with pytest.raises(Exception, match="sense_adjoint_l1|smaps=None"):
        computer.compute(
            pred,
            target,
            losses_dict={"sense_adjoint_l1": loss_fn},
            # smaps deliberately absent
        )


# ---------------------------------------------------------------------------
# VAE dynamic loop excludes KL (already true) AND forwards smaps
# ---------------------------------------------------------------------------


class _VAECfg(_Cfg):
    pass


def test_unified_vae_forwards_smaps_to_sense_adjoint_l1() -> None:
    """A VAE config that uses sense-adjoint reconstruction must get smaps."""
    computer = UnifiedVAELossComputer(_VAECfg(), device=torch.device("cpu"))
    pred = _real_stacked(2, 4, 16, 16, seed=0)
    target = _real_stacked(2, 4, 16, 16, seed=1)
    smaps = _real_stacked_normed_smaps(2, 4, 16, 16, seed=2)
    loss_fn = SENSEAdjointL1Loss(reduction="mean")

    out = computer.compute(
        pred,
        target,
        losses_dict={"sense_adjoint_l1": loss_fn},
        smaps=smaps,
    )
    assert "sense_adjoint_l1" in out.components
    assert out.components["sense_adjoint_l1"].detach().abs().item() > 1e-6


def test_unified_vae_skips_kl_without_posterior_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """KL must be excluded by name from the VAE dynamic loop.

    The VAE computer routes KL through its dedicated posterior path; if a
    ``KLDivergenceLoss`` instance leaks into ``losses_dict`` it must be
    skipped, not miscalled.
    """
    computer = UnifiedVAELossComputer(_VAECfg(), device=torch.device("cpu"))
    pred = torch.randn(2, 4, 16, 16)
    target = torch.randn(2, 4, 16, 16)

    with caplog.at_level(
        logging.WARNING, logger="spectramr.models.losses.computers.unified_vae"
    ):
        out = computer.compute(
            pred,
            target,
            losses_dict={"kl": KLDivergenceLoss(beta=1.0)},
        )
    assert "kl" not in out.components, (
        "KL must not be miscalled via (pred, target) — the VAE excludes it "
        "by name and routes through the dedicated posterior path."
    )
