"""PILOT samples the image on its learned trajectory and feeds the UNet with it.

The UNet input is the NUFFT of the image on the learned trajectory, adjointed with
density compensation (cohort5 §2.7). The trajectory therefore shapes the data term. The
torchkbnufft gradient with respect to the trajectory is unavailable, so the feasibility
penalty is the only gradient the trajectory gets; that is reported, not hidden.
"""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.infrastructure.distributed.owned_modules import OwnedParameter
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.physics.implementations.nufft_operator import NUFFTOperator
from spectramr.infrastructure.training.strategies import pilot_strategy as pilot_mod
from spectramr.infrastructure.training.strategies.pilot_strategy import PILOTStrategy


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def _wire_sampling(s, n_shots, spp, im_size):
    """The attributes ``_sampled_input`` reads, set the way ``__init__`` sets them."""
    s._n_shots = n_shots
    s._samples_per_shot = spp
    s._im_size = im_size
    s._nufft = NUFFTOperator(traj=s._omega(), im_size=im_size)
    return s


def _make_pilot(trajectory, *, codesign_enabled=False):
    """Construct PILOTStrategy bypassing __init__ (no DI/env build needed)."""
    s = object.__new__(PILOTStrategy)
    s.env = types.SimpleNamespace(generator=torch.nn.Identity())
    s.device = torch.device("cpu")
    s._trajectory = trajectory
    s._owned_modules = {"_trajectory": OwnedParameter(trajectory)}
    s._owned_wrappers = {}
    s._codesign = types.SimpleNamespace(enabled=codesign_enabled)
    n = trajectory.shape[0]
    return _wire_sampling(s, 1, n, (4, 4))


def _patch_parent_capture(monkeypatch, s, captured):
    """Stub the immediate-parent ``_compute_losses_impl``; record its kwargs and input."""
    parent = type(s).__mro__[1]

    def _stub(self, input_batch=None, target_batch=None, epoch=0, **kw):
        captured.update(kw)
        captured["input_batch"] = input_batch
        return {"loss_total": torch.zeros(())}

    monkeypatch.setattr(parent, "_compute_losses_impl", _stub, raising=False)


_LR = torch.randn(1, 2, 4, 4)
_HR = torch.randn(1, 2, 4, 4)


class TestPILOTSampledInput:
    def test_parent_receives_the_sampled_image_not_the_raw_input(self, monkeypatch):
        s = _make_pilot(torch.nn.Parameter(torch.randn(8, 2) * 0.1))
        captured: dict = {}
        _patch_parent_capture(monkeypatch, s, captured)

        out = s._compute_losses_impl(input_batch=_LR, target_batch=_HR, epoch=0)

        assert isinstance(out, dict)
        assert captured["input_batch"].shape == _LR.shape
        assert not torch.allclose(captured["input_batch"], _LR), "the UNet saw the raw input"

    def test_the_sampled_input_depends_on_the_trajectory(self):
        s = _make_pilot(torch.nn.Parameter(torch.randn(8, 2) * 0.1))
        a = s._sampled_input(_LR)
        with torch.no_grad():
            s._trajectory.add_(0.05)
        b = s._sampled_input(_LR)
        assert not torch.allclose(a, b), "moving the trajectory did not change the UNet input"

    def test_validation_grades_the_same_sampled_input(self, monkeypatch):
        s = _make_pilot(torch.nn.Parameter(torch.randn(8, 2) * 0.1))
        seen: dict = {}
        parent = PILOTStrategy.__mro__[1]

        def _stub(self, input_batch, batch_context, **kw):
            seen["x"] = input_batch
            return input_batch

        monkeypatch.setattr(parent, "_validation_forward", _stub, raising=False)
        s._validation_forward(_LR, {})
        assert torch.allclose(seen["x"], s._sampled_input(_LR))
        assert not torch.allclose(seen["x"], _LR)


class TestPILOTNufftRoundTrip:
    def test_cartesian_trajectory_matches_fft2c(self):
        """On a fully sampled Cartesian grid the NUFFT forward is fft2c (ortho)."""
        n = 16
        grid = torch.arange(-n // 2, n // 2, dtype=torch.float32) / n
        ky, kx = torch.meshgrid(grid, grid, indexing="ij")
        traj = torch.stack([kx.flatten(), ky.flatten()], dim=-1)
        s = _wire_sampling(_make_pilot(torch.nn.Parameter(traj)), 1, n * n, (n, n))
        image = torch.randn(1, 1, n, n, dtype=torch.complex64)

        samples = s._nufft.forward(image) / n
        reference = fft2c(image)[0, 0].flatten()
        rel = (samples[0, 0] - reference).norm() / reference.norm()
        assert rel.item() < 1e-3

    def test_unit_dcf_adjoint_returns_the_image(self):
        n = 16
        grid = torch.arange(-n // 2, n // 2, dtype=torch.float32) / n
        ky, kx = torch.meshgrid(grid, grid, indexing="ij")
        traj = torch.stack([kx.flatten(), ky.flatten()], dim=-1)
        s = _wire_sampling(_make_pilot(torch.nn.Parameter(traj)), 1, n * n, (n, n))
        image = torch.randn(1, 1, n, n, dtype=torch.complex64)
        samples = s._nufft.forward(image) / n
        back = s._nufft.adjoint(samples, dcf=torch.ones(1, 1, n * n)) / n
        rel = (back - image).norm() / image.norm()
        assert rel.item() < 1e-3


# ---------------------------------------------------------------------------
# Regression: model_kwargs.n_shots / samples_per_shot are READ (pitfall #15)
# ---------------------------------------------------------------------------
#
# 2026-05-31 audit bug — ``__init__`` used ``getattr(self.config.model.
# model_kwargs, "n_shots", 16)``. ``model_kwargs`` is a ``dict[str, Any]``, so
# getattr never finds the key as an attribute and always returns the 16/256
# defaults — any YAML override of these knobs was a silent no-op. The fix uses
# dict ``.get()``, validates ``> 0`` (raise, no silent fallback), and stamps
# the resolved values onto ``self._n_shots`` / ``self._samples_per_shot``.
#
# These tests drive the *real* ``__init__`` body that reads the knobs, by
# constructing the strategy bypassing ``object.__init__``-level DI: we stub the
# parent chain's ``__init__`` so only ``PILOTStrategy.__init__`` runs, with a
# hand-built config exposing ``model.model_kwargs`` and ``acquisition.codesign``.


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def _make_pilot_via_init(monkeypatch, *, model_kwargs, codesign_enabled=False, fov_m=None):
    """Run the real ``PILOTStrategy.__init__`` body with a stub parent init.

    Only the attributes ``__init__`` reads are supplied:
    ``config.model.model_kwargs``, ``config.acquisition.codesign``, ``device``,
    and ``env.opt_g`` (None → no param-group registration).
    """
    parent = PILOTStrategy.__mro__[1]
    constraints = _ns(
        gradient_max_mT_per_m=40.0,
        slew_rate_max_T_per_m_per_s=200.0,
        gradient_dwell_us=10.0,
    )
    codesign = _ns(
        enabled=codesign_enabled,
        feasibility_lambda=0.01,
        constraints=constraints,
        field_of_view_m=fov_m,
    )
    config = _ns(
        model=_ns(model_kwargs=model_kwargs),
        acquisition=_ns(codesign=codesign),
        data=_ns(sampling=_ns(patch_size=(256, 256, 1))),
        undersampling=_ns(adaptive=_ns(mask_learning_rate=1e-4)),
    )

    def _stub_parent_init(self, *a, **k):
        # The base __init__ normally wires config/device/env; supply them here.
        self.config = config
        self.device = torch.device("cpu")
        self.env = _ns(opt_g=None)
        self._owned_modules = {}
        self._owned_wrappers = {}

    monkeypatch.setattr(parent, "__init__", _stub_parent_init, raising=False)
    return PILOTStrategy()


class TestPILOTKnobsAreRead:
    def test_n_shots_and_samples_per_shot_override_is_read(self, monkeypatch):
        """YAML override of model_kwargs.{n_shots, samples_per_shot} must drive
        the trajectory size. Pre-fix getattr() ignored these (4096 = 16*256);
        post-fix the trajectory has 8*64 = 512 points."""
        s = _make_pilot_via_init(
            monkeypatch, model_kwargs={"n_shots": 8, "samples_per_shot": 64}
        )
        assert s._n_shots == 8
        assert s._samples_per_shot == 64
        # The flat trajectory has n_shots * samples_per_shot rows of (kx, ky).
        assert s._trajectory.shape[0] == 512, (
            "n_shots/samples_per_shot knobs were not read — the trajectory still "
            "uses the 16/256 defaults (4096 points)"
        )
        assert s._trajectory.shape == (512, 2)

    def test_defaults_when_kwargs_absent(self, monkeypatch):
        """Empty model_kwargs → documented 16/256 defaults (4096 points)."""
        s = _make_pilot_via_init(monkeypatch, model_kwargs={})
        assert s._n_shots == 16
        assert s._samples_per_shot == 256
        assert s._trajectory.shape[0] == 16 * 256

    def test_non_positive_n_shots_raises(self, monkeypatch):
        """No silent fallback (pitfall #9): n_shots <= 0 must raise."""
        import pytest

        with pytest.raises(ValueError, match="n_shots must be > 0"):
            _make_pilot_via_init(
                monkeypatch, model_kwargs={"n_shots": 0, "samples_per_shot": 64}
            )

    def test_non_positive_samples_per_shot_raises(self, monkeypatch):
        """No silent fallback (pitfall #9): samples_per_shot <= 0 must raise."""
        import pytest

        with pytest.raises(ValueError, match="samples_per_shot must be > 0"):
            _make_pilot_via_init(
                monkeypatch, model_kwargs={"n_shots": 4, "samples_per_shot": -3}
            )


# ---------------------------------------------------------------------------
# Regression: feasibility penalty must not difference across shot boundaries
# ---------------------------------------------------------------------------
#
# 2026-05-31 audit bug — the flat ``[n_shots*samples_per_shot, 2]`` trajectory
# was differenced with ``torch.diff(..., dim=0)``, so shot i's last sample was
# differenced against shot i+1's first sample, injecting a spurious slew /
# gradient spike at every shot boundary. The fix reshapes to
# ``[n_shots, samples_per_shot, 2]`` and differences along the per-shot axis.


def _make_pilot_for_penalty(traj, n_shots, samples_per_shot, *, feas_lambda=1.0, limits=None):
    """Build a PILOTStrategy (bypassing __init__) wired for _feasibility_penalty.

    Constraints are set very low so that the inter-shot jump, if it leaked into
    the diff, would dominate g_overshoot / s_overshoot — making the boundary
    bug loud and easy to detect.
    """
    s = object.__new__(PILOTStrategy)
    s.device = torch.device("cpu")
    s._trajectory = traj
    s._n_shots = n_shots
    s._samples_per_shot = samples_per_shot
    constraints = limits or _ns(
        gradient_max_mT_per_m=0.0001,  # ~0 mT/m → any gradient overshoots
        slew_rate_max_T_per_m_per_s=0.0001,
        gradient_dwell_us=10.0,
    )
    s._codesign = _ns(
        enabled=True,
        feasibility_lambda=feas_lambda,
        constraints=constraints,
        field_of_view_m=0.2,
    )
    s._im_size = (256, 256)
    s._fov_m = 0.2
    return s


class TestPILOTFeasibilityPhysicalUnits:
    """Penalty in T/m and T/m/s (cohort5 §2.7): the same trajectory moves in or out of limits."""

    _LIMITS = _ns(
        gradient_max_mT_per_m=40.0,
        slew_rate_max_T_per_m_per_s=200.0,
        gradient_dwell_us=4.0,
    )

    def test_a_static_trajectory_is_within_limits(self):
        traj = torch.nn.Parameter(torch.zeros(2 * 64, 2))
        s = _make_pilot_for_penalty(traj, 2, 64, limits=self._LIMITS)
        assert s._feasibility_penalty().item() == 0.0

    def test_a_trajectory_over_the_gradient_limit_is_penalised(self):
        traj = torch.zeros(64, 2)
        traj[32:, 0] = 0.5  # one jump of half the FOV in one dwell
        s = _make_pilot_for_penalty(torch.nn.Parameter(traj), 1, 64, limits=self._LIMITS)
        assert s._feasibility_penalty().item() > 0.0

    def test_the_penalty_is_normalised_by_the_limits(self):
        """Halving the limits raises the penalty, by a factor set by the limits."""
        traj = torch.zeros(64, 2)
        traj[32:, 0] = 0.5
        loose = _make_pilot_for_penalty(torch.nn.Parameter(traj.clone()), 1, 64, limits=self._LIMITS)
        tight = _make_pilot_for_penalty(
            torch.nn.Parameter(traj.clone()),
            1,
            64,
            limits=_ns(
                gradient_max_mT_per_m=20.0,
                slew_rate_max_T_per_m_per_s=100.0,
                gradient_dwell_us=4.0,
            ),
        )
        assert tight._feasibility_penalty().item() > loose._feasibility_penalty().item()


class TestPILOTKnobs:
    def test_nufft_is_required_and_raises_without_torchkbnufft(self, monkeypatch):
        monkeypatch.setattr(pilot_mod, "TORCHKBNUFFT_AVAILABLE", False)
        with pytest.raises(ImportError, match="torchkbnufft"):
            _make_pilot_via_init(monkeypatch, model_kwargs={"n_shots": 2, "samples_per_shot": 8})

    def test_codesign_without_a_field_of_view_raises(self, monkeypatch):
        with pytest.raises(ValueError, match="field_of_view_m"):
            _make_pilot_via_init(
                monkeypatch,
                model_kwargs={"n_shots": 2, "samples_per_shot": 8},
                codesign_enabled=True,
                fov_m=None,
            )


class TestPILOTFeasibilityShotBoundary:
    def test_reordering_shots_does_not_change_penalty(self):
        """The feasibility penalty must be invariant to the ORDER of shots.

        The two trajectories share their per-shot sample tensors, and only the
        concatenation order changes. A correct per-shot penalty is invariant to that
        order; the pre-fix flat ``dim=0`` diff straddles the inter-shot gap and
        diverges by orders of magnitude.
        """
        spp = 16
        torch.manual_seed(0)
        # Two shots with tiny within-shot motion (so any boundary leak dominates)
        # but endpoints FAR apart (~10 units) so a straddling diff is enormous.
        shot_a = torch.randn(spp, 2) * 0.01 - 5.0
        shot_b = torch.randn(spp, 2) * 0.01 + 5.0

        flat_ab = torch.cat([shot_a, shot_b], dim=0)
        flat_ba = torch.cat([shot_b, shot_a], dim=0)

        s_ab = _make_pilot_for_penalty(torch.nn.Parameter(flat_ab), 2, spp)
        s_ba = _make_pilot_for_penalty(torch.nn.Parameter(flat_ba), 2, spp)
        penalty_ab = s_ab._feasibility_penalty().item()
        penalty_ba = s_ba._feasibility_penalty().item()

        # Bit-exact: within-shot diffs are identical between orderings, and the
        # only difference (the ignored inter-shot pair) must contribute nothing.
        # Relative tolerance: the penalty is now in physical units, so its magnitude is
        # large and only float32 summation order differs between the two orderings.
        assert penalty_ab == pytest.approx(penalty_ba, rel=1e-5), (
            "feasibility penalty changed when shots were reordered — the diff is "
            "still straddling the shot boundary (dim=0 flat bug). "
            f"[A,B]={penalty_ab} vs [B,A]={penalty_ba}"
        )

    def test_single_shot_value_independent_of_split(self):
        """A trajectory's penalty must not depend on whether we DECLARE it as one
        shot or split the SAME flat samples into several shots — except that the
        multi-shot version drops the (physically meaningless) boundary diffs.

        We assert the multi-shot penalty is STRICTLY LESS THAN the single-shot
        penalty when the inter-shot gaps are large: the single-shot (flat)
        computation, the pre-fix behaviour, includes those huge boundary diffs;
        the correct multi-shot computation excludes them.
        """
        spp = 8
        torch.manual_seed(3)
        shot0 = torch.randn(spp, 2) * 0.01
        shot1 = torch.randn(spp, 2) * 0.01 + 50.0  # huge gap at the boundary
        flat = torch.cat([shot0, shot1], dim=0)

        # Declared as ONE flat shot of 2*spp samples → boundary diff IS included
        # (this is exactly the pre-fix flat-diff behaviour).
        s_flat = _make_pilot_for_penalty(
            torch.nn.Parameter(flat.clone()), 1, 2 * spp
        )
        penalty_flat = s_flat._feasibility_penalty().item()

        # Declared as TWO shots → the 50-unit boundary diff is excluded.
        s_split = _make_pilot_for_penalty(torch.nn.Parameter(flat.clone()), 2, spp)
        penalty_split = s_split._feasibility_penalty().item()

        assert penalty_split < penalty_flat, (
            "splitting into shots did not drop the inter-shot boundary diff — "
            f"split={penalty_split} should be < flat={penalty_flat}"
        )

    def test_penalty_shape_reduces_correctly(self):
        """Sanity: penalty is a finite scalar for a multi-shot trajectory."""
        n_shots, spp = 3, 16
        torch.manual_seed(1)
        traj = torch.nn.Parameter(torch.randn(n_shots * spp, 2) * 0.01)
        s = _make_pilot_for_penalty(traj, n_shots, spp)
        out = s._feasibility_penalty()
        assert out.ndim == 0
        assert torch.isfinite(out)


class TestPILOTTrajectoryGradient:
    def test_a_reconstruction_loss_gives_the_trajectory_a_nonzero_gradient(self):
        """The data term reaches the trajectory through the sampled input (§2.7 follow-up)."""
        s = _make_pilot(torch.nn.Parameter(torch.randn(8, 2) * 0.1))
        out = s._sampled_input(_LR)
        loss = (out - _HR).pow(2).mean()
        loss.backward()
        assert s._trajectory.grad is not None
        assert s._trajectory.grad.abs().sum().item() > 0.0
