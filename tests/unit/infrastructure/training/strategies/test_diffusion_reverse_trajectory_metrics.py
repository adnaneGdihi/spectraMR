"""`val_reverse_*` / `val_band_*` reach the cascade row (#2067).

Two halves, because either alone is the facade shape non-negotiable 16 is about:

* the helper PRODUCES the columns -- exercised against a stub generator, with
  the oracle knob off and on;
* the cascade CALLS it -- asserted on the AST of the sweep method, so a helper
  that quietly stopped being invoked fails here rather than silently emitting
  nothing. `last_effective_steps` was computed and discarded for exactly that
  reason before this change.
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy
from spectramr.models.diffusion.kspace_process import KSpaceUndersamplingProcess

pytestmark = pytest.mark.unit

H = W = 32
DIFFUSION_SOURCE = (
    pathlib.Path(__file__).resolve().parents[5]
    / "src"
    / "spectramr"
    / "infrastructure"
    / "training"
    / "strategies"
    / "diffusion.py"
)


def _strategy(*, stats, attribution: bool, target_domain: str = "kspace"):
    """A `self` carrying only what the helper reads.

    ``model`` / ``data`` are here because the helper resolves the prediction's
    DOMAIN through `needs_ifft_for_visualization` -- a stub without them
    resolves to image space and the attribution is correctly declined.
    """
    process = KSpaceUndersamplingProcess(
        num_timesteps=8,
        base_acceleration=1.0,
        max_acceleration=4.0,
        center_fraction=0.08,
        mask_type="equispaced",
        train_identity_rung=True,
    )
    gen = SimpleNamespace(last_reverse_stats=stats, kspace_process=process)
    return SimpleNamespace(
        generator_model=gen,
        config=SimpleNamespace(
            validation=SimpleNamespace(sampling=SimpleNamespace(reveal_attribution=attribution)),
            model=SimpleNamespace(
                model_type="kspace_cold_diffusion",
                input_type="kspace",
                target_domain=target_domain,
                model_kwargs={},
            ),
            data=SimpleNamespace(dataset_type="kspace"),
        ),
        logging_service=SimpleNamespace(
            log_debug=lambda *a, **k: None, log_warning=lambda *a, **k: None
        ),
    )


def _call(strategy, timestep_used=2):
    pred = torch.randn(1, 2, H, W)
    return DiffusionTrainingStrategy._reverse_trajectory_metrics(
        strategy, pred, pred.clone(), timestep_used
    )


STATS = {
    "effective_steps": 2,
    "skipped_steps": 1,
    "terminal_timestep_called": 1,
    "schedule": [2, 1, 0],
    # The partition mirrors a FREEZE loop. `additive` -- the constructor default
    # -- rewrites the whole plane each step, so the mode has to travel with the
    # record rather than be assumed.
    "reverse_mode": "replace_freeze_dc",
}


def test_reverse_counters_are_emitted_without_the_oracle_knob() -> None:
    out = _call(_strategy(stats=STATS, attribution=False))
    assert out["val_reverse_effective_steps"] == 2.0
    assert out["val_reverse_skipped_steps"] == 1.0
    assert out["val_reverse_terminal_timestep"] == 1.0
    assert not any(k.startswith("val_band_") for k in out), "the oracle must stay opt-in"


def test_terminal_timestep_reports_the_executed_step_not_the_schedule_tail() -> None:
    """PLANTED: a schedule ending at 0 whose terminal step was skipped.

    A stamp reading `schedule[-1]` would report 0 here and hide the very skip
    the column exists to surface.
    """
    out = _call(_strategy(stats=STATS, attribution=False))
    assert STATS["schedule"][-1] == 0
    assert out["val_reverse_terminal_timestep"] == 1.0


def test_a_sampler_with_no_record_emits_nothing_rather_than_zeros() -> None:
    """Absent is reported by omission; zeros would read as 'the loop ran no steps'."""
    assert _call(_strategy(stats=None, attribution=False)) == {}


def test_band_columns_appear_when_the_oracle_knob_is_on() -> None:
    out = _call(_strategy(stats=STATS, attribution=True))
    assert out["val_band_count"] >= 1.0
    # prediction == target, so every measurable band is a perfect gain.
    assert out["val_band_gain_modulus_min"] == pytest.approx(1.0, abs=1e-4)
    assert out["val_band_phase_rad_absmax"] == pytest.approx(0.0, abs=1e-4)
    assert out["val_band_observed_gain_modulus"] == pytest.approx(1.0, abs=1e-4)


def test_a_degenerate_schedule_still_emits_the_counters() -> None:
    """One scheduled step has no reveal partition, but the counters are still facts."""
    stats = {**STATS, "schedule": [0]}
    out = _call(_strategy(stats=stats, attribution=True))
    assert out["val_reverse_effective_steps"] == 2.0
    assert not any(k.startswith("val_band_") for k in out)


def test_the_cascade_actually_calls_the_helper() -> None:
    """The non-negotiable-16 half: defined is not delivered; called is.

    Scoped to the function that also calls ``_compute_validation_metrics`` --
    the cascade sweep itself. A file-wide name search would be satisfied by a
    call in a dead branch, a docstring example, or another method entirely,
    which is the shape of every facade this contract exists to catch.
    """
    tree = ast.parse(DIFFUSION_SOURCE.read_text())

    def calls_in(fn: ast.FunctionDef) -> set[str]:
        return {
            node.func.attr
            for node in ast.walk(fn)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }

    sweeps = [
        fn
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef) and "_compute_validation_metrics" in calls_in(fn)
    ]
    assert sweeps, "no function calls _compute_validation_metrics -- the anchor moved"
    assert any("_reverse_trajectory_metrics" in calls_in(fn) for fn in sweeps), (
        "the helper is defined but the cascade sweep no longer calls it -- the "
        "columns would silently stop being emitted while every test above "
        f"stayed green (searched {[fn.name for fn in sweeps]})"
    )


# ---------------------------------------------------------------------------
# Two preconditions the sampler does not guarantee.
#
# Neither is reachable from the corpus today -- `reveal_attribution` has zero
# hits in `experiments/` -- but `additive` is the CONSTRUCTOR default, so the
# first arm to enable the knob without also naming a freeze mode gets an
# attribution describing a partition the loop never produced.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["additive", "", None, "some_future_mode"])
def test_a_non_partitioning_reverse_mode_emits_no_band_columns(mode) -> None:
    """Counters still come out; only the attribution is declined."""
    stats = dict(STATS)
    if mode is None:
        stats.pop("reverse_mode")
    else:
        stats["reverse_mode"] = mode

    out = _call(_strategy(stats=stats, attribution=True))

    assert out["val_reverse_effective_steps"] == 2.0
    assert not [k for k in out if k.startswith("val_band_")], (
        f"reverse_mode={mode!r} does not write-once-and-freeze, so no step "
        "'wrote' a given coefficient -- attributing error to one is fiction"
    )


@pytest.mark.parametrize("mode", ["replace_freeze", "replace_freeze_dc"])
def test_both_freeze_modes_are_attributed(mode) -> None:
    stats = dict(STATS, reverse_mode=mode)

    out = _call(_strategy(stats=stats, attribution=True))

    assert [k for k in out if k.startswith("val_band_")]


def test_an_image_domain_arm_emits_no_band_columns() -> None:
    """The band selector is a k-space line mask; on an image-domain arm it
    would select image PIXELS and hand back a plausible number."""
    out = _call(_strategy(stats=STATS, attribution=True, target_domain="image"))

    assert out["val_reverse_effective_steps"] == 2.0
    assert not [k for k in out if k.startswith("val_band_")]


def test_the_decline_is_reported_not_silent() -> None:
    """An oracle that quietly emits nothing is indistinguishable from one that
    ran and found nothing."""
    warnings: list[str] = []
    strategy = _strategy(stats=dict(STATS, reverse_mode="additive"), attribution=True)
    strategy.logging_service.log_warning = lambda msg, *a, **k: warnings.append(str(msg))

    _call(strategy)

    assert warnings and "additive" in warnings[0]
