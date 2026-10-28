"""Regression: PrivilegedLearningStrategy must BUILD its teacher and domain
discriminator in ``_setup_strategy_specific_components`` and register the
discriminator's params on ``env.opt_g``.

Pre-fix bug (2026-05-31 audit): the strategy read ``self.env.teacher`` and
``self.env.domain_discriminator`` — neither of which is a field on the frozen
:class:`TrainingEnvironment`, so the first training step raised
``AttributeError`` — and ``_setup_strategy_specific_components`` was a bare
``return``, so nothing was ever built. The fix builds a frozen teacher copy of
the student and a :class:`DomainDiscriminator` (stored on ``self`` because env
is frozen), and adds the discriminator params to ``opt_g``.

These tests drive the setup hook with a stub env (real ``opt_g`` SGD over a
dummy param, a tiny ``nn.Module`` student) and assert:

1. the teacher + domain discriminator exist after setup,
2. the discriminator params land in ``opt_g.param_groups``, and
3. the domain-adversarial loss actually flows gradient to those params.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.privileged_learning_strategy import (
    DomainDiscriminator,
    PrivilegedLearningStrategy,
)


class _TinyStudent(nn.Module):
    """1->1 channel conv so input and output share the channel contract."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _param_ids(opt: torch.optim.Optimizer) -> set[int]:
    return {id(p) for g in opt.param_groups for p in g["params"]}


def _make_strategy() -> PrivilegedLearningStrategy:
    """Build a bare strategy via ``object.__new__`` with only the attributes
    that ``_setup_strategy_specific_components`` reads."""
    s = object.__new__(PrivilegedLearningStrategy)
    s._owned_modules, s._owned_wrappers = {}, {}
    s.device = torch.device("cpu")

    student = _TinyStudent()
    dummy = nn.Parameter(torch.zeros(1))
    config = types.SimpleNamespace(model=types.SimpleNamespace(out_channels=1))
    s.env = types.SimpleNamespace(
        generator=student,
        opt_g=torch.optim.SGD([dummy], lr=0.1),
    )
    s.config = config

    # Attributes the setup hook + loss path touch (normally set in __init__).
    # NOTE: ``_gamma`` is intentionally absent — the hint term has no feature
    # seam on the model forward, so the strategy no longer carries that knob.
    s._disc_hidden_dim = 16
    s._teacher = None
    s._domain_discriminator = None
    s._alpha, s._beta, s._delta = 1.0, 0.5, 0.1
    s._tau0, s._n_min, s._n_max = 5.0, 0.0, 1.0
    return s


class TestPrivilegedTeacherBuilt:
    def test_teacher_and_disc_built_and_registered(self):
        s = _make_strategy()
        before = _param_ids(s.env.opt_g)

        s._setup_strategy_specific_components()

        # 1. Both strategy-owned modules exist (env is frozen → on self).
        assert isinstance(s._teacher, nn.Module), "teacher was not built"
        assert isinstance(
            s._domain_discriminator, DomainDiscriminator
        ), "domain discriminator was not built"

        # Teacher is frozen (never optimized; runs under no_grad in loss path).
        assert all(not p.requires_grad for p in s._teacher.parameters())

        # 2. The discriminator's params were added to opt_g.
        after = _param_ids(s.env.opt_g)
        assert after - before, "domain discriminator params were not added to opt_g"
        assert all(
            id(p) in after for p in s._domain_discriminator.parameters()
        ), "not all discriminator params are registered on opt_g"

    def test_domain_loss_flows_gradient_to_disc(self):
        """KEY property: the domain-adversarial term reaches the discriminator
        params and produces a gradient (the paradigm actually trains them)."""
        s = _make_strategy()
        s._setup_strategy_specific_components()

        x = torch.randn(2, 1, 8, 8)
        target = torch.randn(2, 1, 8, 8)
        marker = torch.randn(2, 1, 8, 8)
        losses = s._compute_losses_impl(
            input_batch=x,
            target_batch=target,
            epoch=0,
            # ``domain_label`` is REQUIRED whenever delta > 0 (A6). The strategy
            # used to fabricate a constant ``1`` here, which trained the
            # gradient-reversal discriminator to separate one class from itself.
            # Two genuine domains, so the term has something to learn.
            batch={"marker": marker, "domain_label": [0.0, 1.0]},
        )

        assert "g_total_loss" in losses
        assert "g_priv_domain" in losses, "domain-adversarial term was not computed"

        losses["g_total_loss"].backward()
        disc_grads = [
            p.grad for p in s._domain_discriminator.parameters() if p.grad is not None
        ]
        assert disc_grads, "no gradient reached the domain discriminator params"
        assert any(
            torch.any(g != 0) for g in disc_grads
        ), "domain discriminator received only zero gradients"


class TestPrivilegedMarkerRequired:
    """F3 regression: a markerless batch must RAISE, not silently zero-fill.

    Pre-fix the strategy did ``marker = pick_present(marker,
    torch.zeros_like(input_batch))`` — a missing marker became a zero tensor,
    so ``teacher_input == input_batch`` and the teacher saw no privileged
    info (the teacher-student gap silently collapsed). The fix raises.
    """

    def test_markerless_batch_raises(self):
        s = _make_strategy()
        s._setup_strategy_specific_components()

        x = torch.randn(2, 1, 8, 8)
        target = torch.randn(2, 1, 8, 8)
        with pytest.raises(ValueError, match="requires batch\\['marker'\\]"):
            s._compute_losses_impl(
                input_batch=x, target_batch=target, epoch=0, batch={}
            )

    def test_marker_present_does_not_raise(self):
        s = _make_strategy()
        # The domain term is a different subject and now demands its own key
        # (A6); switching it off keeps this test about the marker alone.
        s._delta = 0.0
        s._setup_strategy_specific_components()

        x = torch.randn(2, 1, 8, 8)
        target = torch.randn(2, 1, 8, 8)
        marker = torch.randn(2, 1, 8, 8)
        out = s._compute_losses_impl(
            input_batch=x,
            target_batch=target,
            epoch=0,
            batch={"marker": marker},
        )
        assert "g_total_loss" in out

    def test_marker_actually_reaches_teacher(self):
        """The teacher input must differ from the student input when a marker
        is present (otherwise the privileged path is a no-op)."""
        s = _make_strategy()
        s._delta = 0.0  # marker test; the domain term needs its own key (A6)
        s._setup_strategy_specific_components()

        x = torch.zeros(1, 1, 8, 8)
        target = torch.zeros(1, 1, 8, 8)
        # Marker with every entry non-zero at epoch 0 (keep_frac ~ 1.0) so the
        # teacher input is provably perturbed away from the student input.
        marker = torch.ones(1, 1, 8, 8)
        out = s._compute_losses_impl(
            input_batch=x, target_batch=target, epoch=0, batch={"marker": marker}
        )
        # distill = (student - teacher)^2; with x==0 and a non-zero marker the
        # teacher sees a different input than the student → distill is nonzero.
        assert float(out["g_priv_distill"]) > 0.0, (
            "teacher saw no privileged perturbation — marker did not reach it"
        )


class TestPrivilegedGammaKnobRemoved:
    """F4 regression: the unwired ``gamma`` hint knob is gone (pitfall #15).

    The hint term in :func:`privileged_learning_loss` is gated on
    ``student_features``/``teacher_features``, which the strategy never
    supplies (no model exposes intermediate features). A ``gamma`` knob was
    therefore an advertised-but-no-op knob and has been removed.
    """

    def test_strategy_has_no_gamma_attr(self):
        s = _make_strategy()
        s._setup_strategy_specific_components()
        assert not hasattr(s, "_gamma"), (
            "_gamma is an unwired no-op knob and must not be carried"
        )


class TestPrivilegedMarkerCountCurriculum:
    """F6 regression: the curriculum drops a fraction of marker *entries*
    (count), it does not uniformly attenuate the marker amplitude.
    """

    def test_keep_mask_entry_count_decreases_with_epoch(self):
        s = _make_strategy()
        # Strong decay so the kept-entry count visibly shrinks across epochs.
        s._tau0, s._n_min, s._n_max = 1.0, 0.0, 1.0

        marker = torch.ones(1, 1, 8, 8)  # 64 discrete entries
        counts = []
        for epoch in (0, 1, 2, 4, 8):
            keep_frac = s._curriculum_marker_fraction(epoch)
            mask = s._marker_keep_mask(marker, keep_frac)
            # Mask is binary (entries kept = 1.0, dropped = 0.0), NOT a scalar
            # amplitude scale — assert it only contains 0/1 values.
            uniq = set(torch.unique(mask).tolist())
            assert uniq <= {0.0, 1.0}, f"mask must be binary keep/drop, got {uniq}"
            counts.append(int(mask.sum().item()))

        # Count must be non-increasing and strictly decrease overall (drop).
        assert counts == sorted(counts, reverse=True), (
            f"kept marker count must be non-increasing over epochs: {counts}"
        )
        assert counts[0] > counts[-1], (
            f"kept marker count must decrease through training: {counts}"
        )

    def test_curriculum_is_count_based_not_amplitude(self):
        """Distinguishes count-dropout from the pre-fix amplitude scaling:
        applying the mask to a uniform marker yields a tensor whose *nonzero*
        entries keep their original amplitude (1.0), rather than every entry
        scaled by a fraction < 1."""
        s = _make_strategy()
        marker = torch.full((1, 1, 4, 4), 3.0)
        masked = marker * s._marker_keep_mask(marker, 0.5)
        nonzero = masked[masked != 0]
        assert torch.all(nonzero == 3.0), (
            "surviving marker entries must keep full amplitude (count dropout), "
            "not be uniformly attenuated"
        )
