"""Unit tests for :class:`VFConsistencyDistillationStrategy` (PR-5 / idea N-B).

A real few-step consistency-distillation strategy with a virtual-fiducial (VF)
anchor. We avoid the heavyweight ``BaseTrainingStrategy`` DI/AMP init chain by
constructing the strategy via ``object.__new__`` and attaching a minimal
env/generator stub (same lightweight-``self`` convention as
``test_virtual_fiducial_strategy.py``).

Asserts:
- ``_compute_losses_impl`` returns the three documented scalar-tensor keys.
- the EMA teacher state ACTUALLY updates across two consecutive steps.
- the VF anchor is genuinely exercised (a real ``virtual_fiducial`` call) and
  contributes a non-zero term when ``beta0 > 0``.
- ``beta0 = 0`` drops the anchor term (trivial limit = plain consistency
  distillation).
- ``sample`` returns a same-shaped tensor in <= 4 steps.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.vf_consistency_distillation_strategy import (
    VFConsistencyDistillationStrategy,
)


class _TinyGenerator(nn.Module):
    """A minimal consistency student: a single conv that preserves shape.

    Accepts an optional timestep argument (``gen(x, t)``) like real diffusion
    generators, falling back to ``gen(x)``.
    """

    def __init__(self, channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        return self.conv(x)


def _make_strategy(beta0: float = 0.5, im_size: tuple[int, int] = (16, 16)):
    """Build a strategy without the heavyweight base init.

    Returns ``(strategy, generator)``.
    """
    strat = object.__new__(VFConsistencyDistillationStrategy)
    gen = _TinyGenerator()
    strat.env = SimpleNamespace(generator=gen)
    strat.device = torch.device("cpu")
    # Constructor-default knobs (no YAML schema keys).
    strat.beta0 = beta0
    strat.ema_decay = 0.9
    strat.sigma_data = 0.5
    strat.num_consistency_steps = 2
    strat.vf_grid_spacing = 4
    strat.vf_im_size = im_size
    strat._ema_teacher = None
    strat._vf_marker = None
    strat._vf_mask = None
    return strat, gen


def _batch(im_size: tuple[int, int] = (16, 16)):
    H, W = im_size
    target = torch.randn(2, 1, H, W)
    return {"target": target, "source": torch.randn(2, 1, H, W)}


def test_loss_dict_has_three_scalar_keys():
    strat, _ = _make_strategy(beta0=0.5)
    losses = strat._compute_losses_impl(input_batch=_batch())
    for key in ("loss_total", "loss_consistency", "loss_vf_anchor"):
        assert key in losses, f"missing {key}"
        t = losses[key]
        assert isinstance(t, torch.Tensor)
        assert t.dim() == 0, f"{key} must be a scalar tensor"


def test_ema_teacher_state_updates_across_two_steps():
    strat, _ = _make_strategy(beta0=0.5)

    strat._compute_losses_impl(input_batch=_batch())
    assert strat._ema_teacher is not None, "EMA teacher must be created on first step"
    # Snapshot teacher weights after step 1.
    snap = {k: v.clone() for k, v in strat._ema_teacher.state_dict().items()}

    # Mutate the *student* so the EMA blend must move on the next update.
    with torch.no_grad():
        for p in strat.env.generator.parameters():
            p.add_(1.0)

    strat._compute_losses_impl(input_batch=_batch())
    after = strat._ema_teacher.state_dict()

    moved = any(
        not torch.allclose(snap[k], after[k]) for k in snap if after[k].dtype.is_floating_point
    )
    assert moved, "EMA teacher weights must change after the student moved"


def test_ema_teacher_is_detached_from_student():
    """EMA teacher params must not require grad (it is a frozen target)."""
    strat, gen = _make_strategy(beta0=0.5)
    strat._compute_losses_impl(input_batch=_batch())
    assert all(not p.requires_grad for p in strat._ema_teacher.parameters())
    # And it is a distinct module, not the student object.
    assert strat._ema_teacher is not gen


def test_vf_anchor_is_exercised_and_nonzero():
    """beta0 > 0 -> the VF marker is built and the anchor term is real."""
    strat, _ = _make_strategy(beta0=1.0)
    losses = strat._compute_losses_impl(input_batch=_batch())
    assert strat._vf_marker is not None, "VF marker m* must be materialised"
    assert losses["loss_vf_anchor"].item() > 0.0, "anchor must contribute when beta0>0"


def test_vf_marker_is_data_independent():
    """The marker m* must not depend on the batch (target-leak-free)."""
    strat, _ = _make_strategy(beta0=1.0)
    strat._compute_losses_impl(input_batch=_batch())
    m1 = strat._vf_marker.clone()
    # Different batch -> identical marker.
    strat._compute_losses_impl(input_batch={"target": torch.randn(2, 1, 16, 16)})
    assert torch.allclose(m1, strat._vf_marker), "marker must be data-independent"


def test_beta0_zero_drops_anchor_term():
    strat, _ = _make_strategy(beta0=0.0)
    losses = strat._compute_losses_impl(input_batch=_batch())
    assert losses["loss_vf_anchor"].item() == 0.0
    # loss_total reduces to the consistency term (plain consistency distillation).
    assert torch.allclose(losses["loss_total"], losses["loss_consistency"])


def test_sample_returns_same_shape_in_few_steps():
    strat, gen = _make_strategy(beta0=0.5)
    x = torch.randn(2, 1, 16, 16)
    for n_steps in (1, 2, 4):
        out = strat.sample(gen, x, n_steps=n_steps)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


def test_sample_rejects_too_many_steps():
    strat, gen = _make_strategy(beta0=0.5)
    x = torch.randn(1, 1, 16, 16)
    try:
        strat.sample(gen, x, n_steps=5)
    except ValueError:
        return
    raise AssertionError("sample must reject n_steps > 4")


def test_no_generator_returns_zero_total():
    strat, _ = _make_strategy(beta0=0.5)
    strat.env = SimpleNamespace(generator=None)
    losses = strat._compute_losses_impl(input_batch=_batch())
    assert losses["loss_total"].item() == 0.0


# ---------------------------------------------------------------------------
# The model-input contract (non-negotiable 14).
#
# Overriding `_compute_losses_impl` inherits `DiffusionTrainingStrategy`'s
# `snapshot_prepared_is_model_input = False` / `snapshot_model_input_tag =
# "diffusion_step"` while never emitting that tag — a dangling pointer in every
# artifact. Distillation is the case where a single-tensor snapshot would still
# mislead: there are TWO forward inputs at two adjacent trajectory knots, and
# the noise gap between them is what the objective is built on.
# ---------------------------------------------------------------------------


class _RecordingStudent(nn.Module):
    """Records the tensor handed to the network under training."""

    def __init__(self, channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.seen: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        self.seen.append(x)
        return self.conv(x)


def test_declares_the_student_input_not_the_clean_target():
    strat, _ = _make_strategy(beta0=0.0)
    student = _RecordingStudent()
    strat.env = SimpleNamespace(generator=student)
    batch = _batch()

    strat._compute_losses_impl(input_batch=batch)

    assert strat._declared_model_input is not None
    tensors, extra, in_kspace_keys = strat._declared_model_input

    # The student is called first; that call is the one the contract binds to.
    assert torch.equal(tensors["model_input"], student.seen[0])
    # And it is the NOISED knot, not the clean x0 that `input_prepared` holds.
    assert not torch.equal(tensors["model_input"], batch["target"])
    assert torch.equal(tensors["target"], batch["target"])
    assert extra["model_input_key"] == "model_input"
    assert in_kspace_keys == set(), "must be explicit, not None"


def test_teacher_input_rides_alongside_and_differs_from_the_student_input():
    """A one-tensor snapshot cannot show the mechanism.

    Student and teacher get the SAME noise draw at two adjacent knots
    (sigma_next > sigma_now). The gap between them is the whole objective, so an
    artifact showing only one of the two cannot show whether the mechanism fired
    — and `model_input` is the student's because that is the network receiving
    gradient.
    """
    strat, _ = _make_strategy(beta0=0.0)
    # num_consistency_steps=2 makes the two knots distinct for every n_idx draw.
    strat.num_consistency_steps = 2
    student = _RecordingStudent()
    strat.env = SimpleNamespace(generator=student)

    strat._compute_losses_impl(input_batch=_batch())
    tensors, _extra, _keys = strat._declared_model_input

    assert "teacher_input" in tensors
    assert not torch.equal(tensors["model_input"], tensors["teacher_input"]), (
        "the two knots must differ, or the consistency gap is degenerate"
    )
    # Same noise draw at both knots, so within one sample the difference is a
    # pure rescaling of one noise vector: delta_teacher = (sigma_now /
    # sigma_next) * delta_student. The ratio is PER SAMPLE, because the knot
    # index n is drawn per sample -- a batch-wide comparison would be wrong.
    delta_student = tensors["model_input"] - tensors["target"]
    delta_teacher = tensors["teacher_input"] - tensors["target"]
    for i in range(delta_student.shape[0]):
        nonzero = delta_student[i].abs() > 1e-6
        ratios = (delta_teacher[i][nonzero] / delta_student[i][nonzero]).flatten()
        assert torch.allclose(ratios, ratios[0].expand_as(ratios), atol=1e-4), (
            f"sample {i}: student and teacher must share one noise draw, differing only in sigma"
        )


class _ExplodingStudent(nn.Module):
    """Accepts ``(x, t)`` and then fails *inside* the forward."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        raise TypeError("channel mismatch three frames inside the student")


class _TimelessStudent(nn.Module):
    """Genuinely un-time-conditioned: one positional parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TestForwardDispatchIsIntrospectedNotSwallowed:
    """SAQ-001 (#1189): both ``_fwd`` closures dispatch on the signature.

    ``try: module(x, t) except TypeError: module(x)`` answered a genuine
    in-forward failure by re-running the student *un-time-conditioned* against a
    time-conditioned teacher. That converges to something, so nothing ever
    announced the bug -- the distillation simply distilled the wrong function.
    """

    def test_a_typeerror_raised_inside_the_student_propagates(self):
        strat, _ = _make_strategy(beta0=0.5)
        strat.env = SimpleNamespace(generator=_ExplodingStudent())

        try:
            strat._compute_losses_impl(input_batch=_batch())
        except TypeError as exc:
            assert "three frames inside" in str(exc)
        else:
            raise AssertionError(
                "an in-forward TypeError must propagate, not be retried as module(x)"
            )

    def test_an_un_time_conditioned_student_still_trains(self):
        # The dispatch the try/except used to reach by exception is now reached
        # by introspection, and reaches it for the teacher too (a deepcopy of
        # the student, hence the same signature).
        strat, _ = _make_strategy(beta0=0.5)
        strat.env = SimpleNamespace(generator=_TimelessStudent())

        losses = strat._compute_losses_impl(input_batch=_batch())

        assert torch.isfinite(losses["loss_total"])
        assert losses["loss_total"].requires_grad

    def test_sample_resolves_the_signature_once_not_once_per_refinement_step(self):
        # `_generator_accepts_time` runs `inspect.signature` and is uncached --
        # its own docstring says "detect ONCE". Inside the `n_steps` loop it ran
        # per step for a `student_fn` that cannot change mid-call.
        strat, gen = _make_strategy(beta0=0.5)
        calls: list = []
        original = VFConsistencyDistillationStrategy._generator_accepts_time

        def _counting(g):
            calls.append(g)
            return original(g)

        strat._generator_accepts_time = _counting
        out = strat.sample(gen, torch.randn(2, 1, 16, 16), n_steps=4)

        assert len(calls) == 1, f"resolved {len(calls)}x for a 4-step sample"
        assert out.shape == (2, 1, 16, 16)


def test_it_declares_its_own_loss_ownership() -> None:
    """Issue #1918: the term is a distillation loss against the teacher, not the target.

    Read off ``__dict__``, never the inherited value: this class sits under
    ``DiffusionTrainingStrategy``, whose ``folds_image_losses = True`` is truthful for ITSELF and
    becomes a lie the moment a subclass replaces ``_compute_losses_impl``. An
    inherited True makes the audit's ``image_losses_reach_the_objective`` witness
    PASS every declared ``losses.image_losses`` entry on this strategy's arms
    while the training step discards them.
    """
    assert VFConsistencyDistillationStrategy.__dict__["folds_image_losses"] is False
    assert VFConsistencyDistillationStrategy.__dict__["inline_losses"] == frozenset()
