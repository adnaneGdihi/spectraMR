"""Regression tests for SM-001: ``AdversarialMixin``'s per-step closures must
NOT trigger a GPU->CPU sync.

Previously the ``d_closure`` / ``g_closure`` bodies stored
``float(tensor.detach())`` into ``_last_step_metrics``. ``float()`` on a CUDA
tensor calls ``__float__`` == ``.item()`` == a blocking D2H transfer, executed
on EVERY training step (NN#9 violation). The fix defers that conversion: the
closures store detached *tensors* on-device, and ``get_last_metrics()`` resolves
them to Python floats at the coarser reporting cadence.

These tests are CPU-only and mock the heavy collaborators (generator,
discriminator, loss computer) so the real closures run unchanged.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.mixins import adversarial
from spectramr.infrastructure.training.strategies.mixins.adversarial import (
    AdversarialMixin,
)


class _LossOutput(SimpleNamespace):
    """Duck-typed stand-in for the loss-computer output: exposes ``.total`` and
    ``.components`` exactly like ``UnifiedGANLossComputer`` results."""


class _StubLossComputer:
    """Minimal loss computer matching the attributes the closures probe."""

    def compute_discriminator_loss(self, **_: Any) -> _LossOutput:
        return _LossOutput(
            total=torch.tensor(0.7, requires_grad=True),
            components={"adv": torch.tensor(0.3)},
        )

    def compute_generator_loss(self, **_: Any) -> _LossOutput:
        return _LossOutput(
            total=torch.tensor(0.5, requires_grad=True),
            components={"adv": torch.tensor(0.2)},
        )


class _Strat(AdversarialMixin):
    """Bare adversarial-mixin host wired with CPU-only stubs for everything the
    closures touch, so ``train_step_adversarial`` exercises the real closure
    bodies (and thus the SM-001 metric-storage path)."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self._step_counter = 0
        self.loss_computer = _StubLossComputer()
        gen = torch.nn.Identity()
        disc = torch.nn.Conv2d(1, 1, 1)
        self._gen = gen
        self._disc = disc
        # disc_updates lives at config.losses.gan.disc_updates (v6.0 SSOT)
        cfg = SimpleNamespace(
            losses=SimpleNamespace(gan=SimpleNamespace(disc_updates=1)),
            training=SimpleNamespace(enforce_output_range=False),
        )
        self.config = cfg
        self.env = SimpleNamespace(config=cfg, step=0, criterion_l1=None)
        self.state = SimpleNamespace(opt_d=None, opt_g=None, config=cfg)

    # --- stubbed collaborators the closures read -------------------------
    @property
    def generator_model(self) -> torch.nn.Module:
        return self._gen

    @property
    def discriminator_model(self) -> torch.nn.Module:
        return self._disc

    def _to_device(self, data: Any) -> Any:
        return data

    def _unpack_batch(self, batch: Any) -> tuple[Any, Any]:
        return batch["input"], batch["target"]


def _run_closures() -> _Strat:
    strat = _Strat()
    img = torch.randn(1, 1, 4, 4)
    batch = {"input": img, "target": img.clone()}
    step_configs = strat.train_step_adversarial(batch, epoch=0, iteration=5)
    # Execute every produced closure (d then g) exactly as the Trainer would.
    for cfg in step_configs:
        cfg["closure"]()
    return strat


def test_closures_store_tensors_not_floats() -> None:
    """SM-001 core: after the closures run, stored metrics are detached tensors
    (on-device), proving no per-step D2H float() conversion happened inside the
    closure body."""
    strat = _run_closures()
    metrics = strat._last_step_metrics
    assert metrics, "closures should have populated _last_step_metrics"
    # The loss totals are stored as tensors, NOT Python floats.
    assert isinstance(metrics["d_total_loss"], torch.Tensor)
    assert isinstance(metrics["g_total_loss"], torch.Tensor)
    # And they are detached (no grad history leaks into the metrics dict).
    assert not metrics["d_total_loss"].requires_grad
    assert not metrics["g_total_loss"].requires_grad
    # Component metrics likewise remain tensors.
    assert isinstance(metrics["d_adv"], torch.Tensor)


def test_get_last_metrics_stays_on_device() -> None:
    """The D2H sync is deferred PAST get_last_metrics, not to it (#707).

    This asserted "a dict of pure Python floats", which was the original SM-001
    design: the closures stop syncing per key, and `get_last_metrics` converts.
    The sync audit found the second half was still wrong -- `training_loop`
    calls this on EVERY iteration, outside the `log_interval` gate, so the sync
    count per step never dropped. It only moved. The loop's gated converter now
    owns the single fused transfer.

    The values are still the ones the closures stored, so what this test really
    guarded (the closures publish real metrics) is unchanged.
    """
    strat = _run_closures()
    resolved = strat.get_last_metrics()
    assert resolved, "get_last_metrics should expose the stored metrics"
    assert all(isinstance(v, torch.Tensor) for v in resolved.values())
    # fp32 round-off: the stub stores 0.7/0.5 which widen-then-narrow through float32.
    assert float(resolved["d_total_loss"]) == pytest.approx(0.7, abs=1e-6)
    assert float(resolved["g_total_loss"]) == pytest.approx(0.5, abs=1e-6)


def test_no_bare_float_detach_in_metric_storage() -> None:
    """AST guard: the closure metric-storage sites must not wrap a ``.detach()``
    in ``float(...)`` (the exact D2H-sync pattern SM-001 removed)."""
    source = Path(inspect.getsourcefile(adversarial)).read_text()
    tree = ast.parse(source)

    offenders: list[int] = []
    for node in ast.walk(tree):
        # Match float(<x>.detach())
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "float"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Attribute)
            and node.args[0].func.attr == "detach"
        ):
            offenders.append(node.lineno)

    assert not offenders, (
        "float(x.detach()) (a per-step GPU->CPU sync) reintroduced at "
        f"adversarial.py lines {offenders}; defer to get_last_metrics()."
    )


def test_train_step_reads_loop_state_not_frozen_env_step() -> None:
    """WS-3 follow-up: ``train_step_adversarial``'s ``current_step`` (fed to the
    ``train_metric_interval`` throttle) reads the live ``loop_state`` seam via
    ``resolve_loop_iteration(self)``, not the frozen ``self.env.step`` 0 that
    made the throttle fire every step (pitfall #16).

    The helper degrades to 0 when ``loop_state`` is absent, so the bare
    ``_Strat(AdversarialMixin)`` stub used by the other tests in this module
    (no strategy base, no loop_state) stays safe."""
    code = "\n".join(
        ln
        for ln in inspect.getsource(
            AdversarialMixin.train_step_adversarial
        ).splitlines()
        if not ln.lstrip().startswith("#")
    )
    assert "self.env.step" not in code
    assert "resolve_loop_iteration(self)" in code


# --- the mixin must not score the critic itself ------------------------------
#
# The closures used to guard both loss-computer hooks with
# ``hasattr(...) else None`` and feed the critic themselves on the None branch.
# ``setup_adversarial`` installs ``UnifiedGANLossComputer`` unconditionally and
# it defines both hooks, so that branch was unreachable -- but it was a second,
# divergent critic feed: this file handed a complex fake over raw while
# ``gan.py`` realified it first, and with no critic it silently replaced the
# adversarial objective with plain L1 (NN3). A computer missing a hook must now
# raise, naming it, rather than quietly scoring the critic a different way.


class _ComputerMissingHooks:
    """A computer with neither hook, which used to divert to ``compute``."""

    def __init__(self) -> None:
        self.compute_calls = 0

    def compute(self, **_: Any) -> _LossOutput:
        self.compute_calls += 1
        return _LossOutput(total=torch.tensor(1.0, requires_grad=True), components={})


def test_a_computer_without_the_hooks_raises_instead_of_feeding_the_critic():
    strat = _Strat()
    broken = _ComputerMissingHooks()
    strat.loss_computer = broken

    img = torch.randn(1, 1, 4, 4)
    step_configs = strat.train_step_adversarial(
        {"input": img, "target": img.clone()}, epoch=0, iteration=5
    )
    with pytest.raises(AttributeError) as excinfo:
        for cfg in step_configs:
            cfg["closure"]()

    assert "compute_discriminator_loss" in str(excinfo.value) or (
        "compute_generator_loss" in str(excinfo.value)
    )
    assert broken.compute_calls == 0, (
        "the deleted fallback re-entered `compute` and scored the critic itself"
    )


def test_closures_delegate_the_critic_instead_of_scoring_it():
    """AST, not source text, so a comment naming the critic cannot satisfy it.

    Falsified against ``origin/dev``, where the same walk finds three calls
    (two in ``d_closure``, one in ``g_closure``).
    """
    tree = ast.parse(
        inspect.cleandoc(inspect.getsource(AdversarialMixin.train_step_adversarial))
    )
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in {"discriminator_model", "discriminator"}
    ]
    assert calls == []


def test_the_installed_computer_defines_the_hooks_the_closures_now_call_unguarded():
    """What licenses deleting the guards: the computer always has both hooks."""
    from spectramr.models.losses.computers.unified_gan import UnifiedGANLossComputer

    assert hasattr(UnifiedGANLossComputer, "compute_generator_loss")
    assert hasattr(UnifiedGANLossComputer, "compute_discriminator_loss")
    assert "UnifiedGANLossComputer" in inspect.getsource(
        AdversarialMixin.setup_adversarial
    )


_GAN_ARM = (
    Path(__file__).resolve().parents[6] / "experiments" / "inprogress" / "gans" / "beta_vae_gan.yaml"
)


class _BanneredHost(AdversarialMixin):
    """``AdversarialMixin`` over the real banner machinery.

    ``setup_adversarial`` is annotated ``self: BaseTrainingStrategy`` throughout, so
    borrowing the two members it now depends on states that contract rather than
    re-implementing it: a stubbed banner would pass whether or not the SSOT was read.
    """

    _log_loss_objective = BaseTrainingStrategy._log_loss_objective
    _weight_table = BaseTrainingStrategy._weight_table

    def __init__(self, config: Any) -> None:
        self.device = torch.device("cpu")
        self.config = config
        self.env = SimpleNamespace(config=config)
        self.state = SimpleNamespace(config=config, model_type="beta_vae_gan")
        self.logged: list[str] = []

        def _record(message: str, *_a: Any, **_k: Any) -> None:
            # The helpers on this path log with keyword arguments, so a bare
            # ``list.append`` spy raises before the assertion is ever reached.
            self.logged.append(message)

        self.logging_service = SimpleNamespace(
            log_info=_record, log_warning=_record, log_debug=_record
        )


@pytest.fixture
def gan_arm_config():
    from spectramr.config.settings import TrainingSettings

    assert _GAN_ARM.exists(), f"missing experiment config: {_GAN_ARM}"
    return TrainingSettings.from_yaml(str(_GAN_ARM))


@pytest.mark.unit
class TestSetupAnnouncesTheObjective:
    """#1918/#1919: the adversarial family must announce what it optimizes, read
    from the loss-weight SSOT — the same wiring ``DiffusionTrainingStrategy`` got
    in #1964. Before this change ``setup_adversarial`` printed nothing at all."""

    def test_setup_adversarial_calls_the_banner(self) -> None:
        """Non-negotiable 16: defining the helper is the easy half — the production
        setup path has to call it."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(AdversarialMixin.setup_adversarial)))
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "_log_loss_objective" in calls

    def test_the_banner_reaches_the_logger(self, gan_arm_config) -> None:
        """Observed, not inferred: drive the real ``setup_adversarial`` and read the spy."""
        host = _BanneredHost(gan_arm_config)
        host.setup_adversarial(expected_modes=("gan",))

        assert host.logged, "setup printed no banner — the state this change removes"
        assert host.logged[0].startswith("[_BanneredHost] Configured Losses (")
        assert not host.logged[0].startswith("[_BanneredHost] Configured Losses (0)")

    def test_the_banner_reports_the_declared_weights(self, gan_arm_config) -> None:
        """The arm declares ``losses.gan.lambda_adv: 0.1`` and ``image_losses[mse]: 1.0``;
        a resolver called at iteration 0 would print 0.0000 for the warm-up-gated
        adversarial term, so the banner must read the declared weight instead."""
        host = _BanneredHost(gan_arm_config)
        host.setup_adversarial(expected_modes=("gan",))

        assert any("adversarial" in ln and "0.1000" in ln for ln in host.logged)
        assert any("l2" in ln and "1.0000" in ln for ln in host.logged)

    def test_the_prefix_names_the_concrete_strategy(self, gan_arm_config) -> None:
        """Four strategies reach this one setup, so a fixed prefix would make their
        banners indistinguishable in a single log."""

        class _SecondArm(_BanneredHost):
            pass

        host = _SecondArm(gan_arm_config)
        host.setup_adversarial(expected_modes=("gan",))

        assert host.logged[0].startswith("[_SecondArm] ")
