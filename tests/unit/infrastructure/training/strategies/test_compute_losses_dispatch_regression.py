"""Regression: ``_compute_losses_impl`` must survive the base orchestrator's
calling convention.

``BaseTrainingStrategy._compute_losses`` (base.py:907) invokes the hook as
``_compute_losses_impl(input_batch=..., target_batch=..., epoch=..., **kwargs)``
— all keyword. Two audit-found (2026-05-31) crash modes regress here:

* **NameError** — a canonical-signature override whose body forwards
  ``super()._compute_losses_impl(batch, *args, **kwargs)`` while ``*args`` was
  never declared (cross_contrast, field_probe, girf, adversarial).
* **TypeError** — a legacy ``(self, batch, *args)`` signature: the named call
  leaves the positional ``batch`` unbound / collides on ``input_batch``
  (bloch_consistent).

These also read the model from ``self.context.generator`` — a field the
``StrategyContext`` dataclass does not have (always ``None``) — now
``self.env.generator``.
"""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.infrastructure.training.strategies.adversarial_robustness_strategy import (
    AdversarialRobustnessStrategy,
)
from spectramr.infrastructure.training.strategies.bloch_consistent_denoising_strategy import (
    BlochConsistentDenoisingStrategy,
)
from spectramr.infrastructure.training.strategies.cross_contrast_kspace_diffusion_strategy import (
    CrossContrastKspaceDiffusionStrategy,
)
from spectramr.infrastructure.training.strategies.field_probe_coupled_strategy import (
    FieldProbeCoupledStrategy,
)
from spectramr.infrastructure.training.strategies.girf_aware_strategy import (
    GIRFAwareStrategy,
)
from spectramr.infrastructure.training.strategies.pilot_strategy import PILOTStrategy
from spectramr.infrastructure.training.strategies.bald_acquisition_strategy import (
    BALDAcquisitionStrategy,
)
from spectramr.infrastructure.training.strategies.inverse_bloch_phase_strategy import (
    InverseBlochPhaseStrategy,
)
from spectramr.infrastructure.training.strategies.synthetic_pathology_aug_strategy import (
    SyntheticPathologyAugStrategy,
)
from spectramr.infrastructure.training.strategies.edm_training_strategy import (
    EDMTrainingStrategy,
)
from spectramr.infrastructure.training.strategies.kspace_inr_strategy import (
    KSpaceINRStrategy,
)


def _make(cls, **attrs):
    """Construct a strategy bypassing __init__ (no DI/env build needed)."""
    s = object.__new__(cls)
    s.env = types.SimpleNamespace(generator=torch.nn.Identity())
    for k, v in attrs.items():
        setattr(s, k, v)
    return s


def _patch_parent(monkeypatch, s):
    """Stub the immediate-parent _compute_losses_impl so super() returns a
    sentinel dict without building a real loss computer."""
    parent = type(s).__mro__[1]
    monkeypatch.setattr(
        parent,
        "_compute_losses_impl",
        lambda self, input_batch=None, target_batch=None, epoch=0, **kw: {
            "loss_total": torch.zeros(())
        },
        raising=False,
    )


def _dispatch_reaches_the_body(s, **call):
    """Invoke the hook the way the base orchestrator does; return the outcome.

    The contract this FILE tests is narrow: the named call must bind, i.e. it
    must not raise ``NameError`` (an undeclared ``*args`` forward) or
    ``TypeError`` (a legacy positional signature). Anything the strategy raises
    ABOUT ITS INPUT is downstream of that and proves the dispatch worked -- the
    call got far enough to inspect the batch.

    Three strategies have since grown anti-facade guards that raise exactly
    there (pitfall #16: refuse rather than fall back to a plain denoiser under a
    themed arm's name). They used to fall through to ``super()`` and return the
    sentinel dict, which is why these tests asserted ``isinstance(out, dict)``
    and went red on cluster job 8004252 when the guards landed. Asserting the
    dict again would mean asking for the facade back.
    """
    try:
        return s._compute_losses_impl(**call)
    except (NameError, TypeError) as exc:  # the regression this file exists for
        raise AssertionError(
            f"named-convention dispatch broke: {type(exc).__name__}: {exc}"
        ) from exc


_LR = torch.randn(1, 2, 4, 4)
_HR = torch.randn(1, 2, 4, 4)


class TestDispatchNoCrash:
    def test_cross_contrast_guard_forwards_named(self, monkeypatch):
        """Dispatch binds; the paired-kspace guard is what stops it."""
        s = _make(CrossContrastKspaceDiffusionStrategy)
        _patch_parent(monkeypatch, s)
        with pytest.raises(ValueError, match="kspace_source"):
            _dispatch_reaches_the_body(
                s, input_batch=_LR, target_batch=_HR, epoch=0
            )

    def test_adversarial_guard_forwards_named(self, monkeypatch):
        s = _make(AdversarialRobustnessStrategy)
        _patch_parent(monkeypatch, s)
        # dict missing kspace/input/target -> x is None -> guard -> super().
        out = s._compute_losses_impl(
            input_batch=_LR, target_batch=_HR, epoch=0, batch={"foo": _LR}
        )
        assert isinstance(out, dict)

    def test_girf_no_waveform_forwards_named(self, monkeypatch):
        s = _make(GIRFAwareStrategy, _girf=None)
        _patch_parent(monkeypatch, s)
        # dict without gradient_waveform reaches the formerly *args-broken line.
        out = s._compute_losses_impl(
            input_batch=_LR, target_batch=_HR, epoch=0, batch={"x": _LR}
        )
        assert isinstance(out, dict)

    def test_bloch_consistent_migrated_signature(self, monkeypatch):
        """A legacy ``(self, batch, *args)`` signature would TypeError here.

        It does not -- the RuntimeError below comes from the signal-synthesis
        guard, which only runs once the arguments have bound.
        """
        s = _make(BlochConsistentDenoisingStrategy, _bloch=None)
        _patch_parent(monkeypatch, s)
        with pytest.raises(RuntimeError, match="Bloch signal-synthesis"):
            _dispatch_reaches_the_body(
                s, input_batch=_LR, target_batch=_HR, epoch=0
            )

    def test_field_probe_guard_forwards_named(self, monkeypatch):
        s = _make(FieldProbeCoupledStrategy)
        _patch_parent(monkeypatch, s)
        out = s._compute_losses_impl(input_batch=_LR, target_batch=_HR, epoch=0)
        assert isinstance(out, dict)

    def test_pilot_legacy_signature_migrated(self, monkeypatch):
        # The sampled UNet input needs a trajectory and its NUFFT; a PILOT built without
        # them raises rather than passing the raw input through, so both are wired here.
        from spectramr.infrastructure.distributed.owned_modules import OwnedParameter
        from spectramr.infrastructure.physics.implementations.nufft_operator import NUFFTOperator

        traj = torch.nn.Parameter(torch.randn(8, 2) * 0.1)
        s = _make(
            PILOTStrategy,
            _codesign=types.SimpleNamespace(enabled=False),
            _trajectory=traj,
            _owned_modules={"_trajectory": OwnedParameter(traj)},
            _owned_wrappers={},
            _n_shots=1,
            _samples_per_shot=8,
            _im_size=(4, 4),
            _nufft=NUFFTOperator(traj=(2 * torch.pi * traj.detach()).flip(-1).T, im_size=(4, 4)),
        )
        _patch_parent(monkeypatch, s)
        out = s._compute_losses_impl(input_batch=_LR, target_batch=_HR, epoch=0)
        assert isinstance(out, dict)

    def test_bald_legacy_signature_migrated(self, monkeypatch):
        s = _make(
            BALDAcquisitionStrategy,
            _step_count=0,
            _active=types.SimpleNamespace(enabled=False),
        )
        _patch_parent(monkeypatch, s)
        out = s._compute_losses_impl(input_batch=_LR, target_batch=_HR, epoch=0)
        assert isinstance(out, dict)

    def test_inverse_bloch_legacy_signature_migrated(self, monkeypatch):
        """Binds under the named convention; the phase-prior guard then fires."""
        s = _make(InverseBlochPhaseStrategy)
        _patch_parent(monkeypatch, s)
        with pytest.raises(ValueError, match="phase_residual"):
            _dispatch_reaches_the_body(
                s, input_batch=_LR, target_batch=_HR, epoch=0
            )

    def test_synthetic_pathology_else_branch_named(self, monkeypatch):
        s = _make(SyntheticPathologyAugStrategy)
        _patch_parent(monkeypatch, s)
        out = s._compute_losses_impl(input_batch=_LR, target_batch=_HR, epoch=0)
        assert isinstance(out, dict)

    def test_edm_legacy_signature_no_target(self):
        # Legacy (self, batch, *args) would TypeError on the named call.
        # dict without "target" -> early zero-loss return (no parent needed).
        s = _make(EDMTrainingStrategy)
        s.device = torch.device("cpu")
        out = s._compute_losses_impl(input_batch={"x": _LR}, target_batch=None, epoch=0)
        assert isinstance(out, dict) and "loss_total" in out

    def test_kspace_inr_legacy_signature(self):
        # k-space INR legitimately requires batch['kspace'] (pitfall #9 — no
        # silent zero-loss fallback); supply it so the all-keyword dispatch
        # reaches the real loss path. _make gives env.generator = Identity, so
        # pred == kspace and the MAE is 0.
        s = _make(KSpaceINRStrategy)
        s.device = torch.device("cpu")
        out = s._compute_losses_impl(
            input_batch={"kspace": _LR}, target_batch=None, epoch=0
        )
        assert isinstance(out, dict) and "loss_total" in out
