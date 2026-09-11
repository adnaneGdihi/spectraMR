"""Regression guard: the train-metric throttle is fed the LIVE loop iteration.

Background (pitfall #16). ``MetricsMixin._compute_training_metrics`` throttles the
per-step reconstruction-quality metrics (SSIM / PSNR / MAE) with::

    if current_step % train_metric_interval != 0:
        return metrics

Fed a frozen ``0`` the throttle is defeated -- ``0 % interval == 0`` for every
interval -- and the (host-syncing) metrics are recomputed and logged on EVERY step.
This module pins the POSITIVE half of the contract: the call site passes the live
``iteration``.

**The NEGATIVE half moved out, and why (non-negotiable 17).** This file used to also
assert ``'getattr(self.env, "step"' not in src``. That pin now lives in
``tests/architecture/test_strategies_read_live_iteration.py``, which scans every
module under ``infrastructure/training/`` by AST for all three frozen-read shapes.
The old pin is deleted rather than kept as defence in depth: it was the weaker of
the two owners and, kept alongside, neither would be audited as the sole line of
defence.

**It was weaker in a way that mattered.** ``_loss_impl_source`` returned on the
FIRST class ``inspect.getmembers`` yielded -- alphabetical -- so for ``vae.py`` it
read ``VAETrainingStrategy`` (lines 118-301) and never reached
``VQVAETrainingStrategy`` (lines 525-620), which carried
``getattr(self.env, "step", 0)`` at line 605. The guard written to forbid that exact
line named the module containing it and passed green. The docstring here compounded
it, asserting ``vae.py`` "was migrated" and "now reuses the live ``iteration``" --
true of one of its two strategies. A detector defect outranks an equal-scoring code
defect (non-negotiable 15), so the walk below yields EVERY class that defines
``_compute_losses_impl``, and the case ids name the class rather than the module.
"""

from __future__ import annotations

import ast
import inspect
import re

import pytest

from spectramr.infrastructure.training.strategies import reconstruction as _recon_mod
from spectramr.infrastructure.training.strategies import vae as _vae_mod


def _loss_impl_sources(module: object) -> list[tuple[str, str]]:
    """``(qualname, source)`` for EVERY class in ``module`` defining the hook.

    Deliberately not ``getmembers`` + first hit: that ordering is alphabetical and
    silently drops every later class (see the module docstring).
    """
    found: list[tuple[str, str]] = []
    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        fn = obj.__dict__.get("_compute_losses_impl")
        if fn is not None:
            found.append(
                (f"{module.__name__.rsplit('.', 1)[-1]}.{obj.__name__}", inspect.getsource(fn))
            )
    if not found:
        raise AssertionError(f"no _compute_losses_impl found in {module.__name__}")
    return found


def _all_hooks() -> list[tuple[str, str]]:
    return [rec for mod in (_recon_mod, _vae_mod) for rec in _loss_impl_sources(mod)]


#: resolved once -- the ids must be the qualnames, never the source bodies
_HOOKS: list[tuple[str, str]] = _all_hooks()


def test_the_walk_reaches_every_class_not_just_the_first() -> None:
    """The blindness that let ``vae.py:605`` survive: pin the walk itself.

    ``vae`` defines two strategies with the hook; a first-hit walk sees one.
    """
    names = [name for name, _ in _loss_impl_sources(_vae_mod)]
    assert names == ["vae.VAETrainingStrategy", "vae.VQVAETrainingStrategy"], names


def test_hook_count_matches_the_source() -> None:
    """Cross-check the runtime walk against a static parse of the same files.

    A class the import machinery does not expose (conditionally defined, renamed on
    export) would make the walk quietly narrower than the file.
    """
    import pathlib

    for mod in (_recon_mod, _vae_mod):
        path = pathlib.Path(inspect.getfile(mod))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        static = [
            cls.name
            for cls in tree.body
            if isinstance(cls, ast.ClassDef)
            and any(
                isinstance(m, ast.FunctionDef | ast.AsyncFunctionDef)
                and m.name == "_compute_losses_impl"
                for m in cls.body
            )
        ]
        runtime = [name.split(".", 1)[1] for name, _ in _loss_impl_sources(mod)]
        assert sorted(runtime) == sorted(static), (
            f"{path.name}: runtime walk saw {runtime}, the file defines {static}"
        )


@pytest.mark.unit
@pytest.mark.parametrize(("qualname", "src"), _HOOKS, ids=[n for n, _ in _HOOKS])
def test_train_metrics_fed_live_iteration(qualname: str, src: str) -> None:
    call = re.search(r"_compute_training_metrics\((.*?)\)", src, re.DOTALL)
    assert call is not None, f"{qualname}: no _compute_training_metrics call site found"
    assert "current_step=iteration" in call.group(1), (
        f"{qualname}: the train-metric throttle is not fed the live loop iteration "
        f"(got `{' '.join(call.group(1).split())}`). A frozen step recomputes "
        "SSIM/PSNR/MAE every step (pitfall #16, #1937)."
    )


@pytest.mark.unit
def test_metrics_throttle_honours_current_step() -> None:
    """The mechanism the fix relies on: a non-multiple step is throttled to {}.

    Constructs the mixin via ``__new__`` (no heavy strategy init) and stitches on
    only what ``_compute_training_metrics`` touches.
    """
    import torch

    from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
        MetricsMixin,
    )

    mixin = MetricsMixin.__new__(MetricsMixin)

    class _SpyComputer:
        def __init__(self) -> None:
            self.calls = 0

        def compute(self, pred, target):
            self.calls += 1
            return {"ssim": 1.0}

    spy = _SpyComputer()
    # ``training_metrics_computer`` is a lazy property backed by ``_training_computer``.
    mixin._training_computer = spy  # type: ignore[attr-defined]
    mixin._apply_metric_transforms = lambda p, t, c: (p, t)  # type: ignore[attr-defined]

    config = type(
        "Cfg",
        (),
        {"metrics": type("M", (), {"enable_tracking": True, "train_metric_interval": 100})()},
    )()
    # The lazy property reads ``self.config`` before returning the cached computer.
    mixin.config = config  # type: ignore[attr-defined]
    pred = torch.zeros(1, 1, 8, 8)
    target = torch.zeros(1, 1, 8, 8)

    # A non-multiple step is throttled (no compute); the frozen-0 bug made this
    # branch unreachable because env.step was always 0.
    assert mixin._compute_training_metrics(pred, target, config, current_step=50) == {}
    assert spy.calls == 0

    # A multiple step computes and prefixes the metric.
    out = mixin._compute_training_metrics(pred, target, config, current_step=100)
    assert out == {"train_ssim": 1.0}
    assert spy.calls == 1
