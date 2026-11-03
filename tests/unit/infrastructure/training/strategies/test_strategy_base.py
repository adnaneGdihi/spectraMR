"""Unit tests for BaseTrainingStrategy public seams.

Scope (TASK IV.E):
  - BaseTrainingStrategy is abstract / cannot be instantiated directly.
  - Public lifecycle hooks (on_epoch_start/end, on_validation_start/end)
    accept the declared signatures without raising on valid calls.
  - _compute_losses_impl raises NotImplementedError — the contract that
    all concrete strategies must implement.
  - apply_adapters is a no-op when no adapter chain is registered for
    the requested hook.
  - get_last_metrics returns a plain dict[str, float].
  - _verify_strategy_config raises ValueError when config is None.
  - LossResult typed container behaves correctly.

Heavy full forward/backward passes and GPU usage are marked @pytest.mark.slow
so they are skipped in the default fast lane.  Structural / import-only checks
run unconditionally.

No real TrainingEnvironment is built here — that requires GPU + full config.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_config() -> MagicMock:
    """Return a MagicMock that satisfies all BaseTrainingStrategy attribute paths."""
    cfg = MagicMock()
    cfg.optimization.optimizer.learning_rate = 1e-3
    cfg.optimization.precision.enabled = False
    cfg.optimization.gradient.clip.enabled = False
    cfg.optimization.gradient.clip.value = 1.0
    # schema defaults; bare MagicMock fails StandardOptimizerStepper's raise-on-unknown
    cfg.optimization.gradient.clip.method = "norm"
    cfg.optimization.gradient.accumulation_steps = 1
    cfg.optimization.memory.enable_monitoring = False
    cfg.model.target_domain = "image"
    cfg.model.in_channels = 1
    cfg.model.out_channels = 1
    cfg.model.model_type = "test_model"
    cfg.physics = MagicMock()
    cfg.physics.kspace = MagicMock()
    cfg.physics.kspace.enable_kspace_recon = False
    cfg.physics.data_consistency = MagicMock()
    cfg.physics.data_consistency.method = "soft"
    cfg.physics.data_consistency.weight = 1.0
    cfg.physics.pinn = MagicMock()
    cfg.physics.pinn.enabled = False
    cfg.physics.compressed_sensing = MagicMock()
    cfg.physics.compressed_sensing.enabled = False
    cfg.physics.digital_twin = MagicMock()
    cfg.physics.digital_twin.enabled = False
    cfg.logging.log_gradients = False
    cfg.logging.log_interval = 100
    # Snapshots OFF. Must be the nested spelling: on a MagicMock the retired
    # flat keys do not merely stop working, they INVERT -- `_resolve_config`
    # reads the auto-vivified `logging.snapshots`, whose leaves are truthy and
    # int-able, so it resolves to `enabled=True, save_images=True`. Pinned by
    # `test_minimal_config_actually_disables_snapshots` below.
    # (`max_calls = 0` is gone with the flat keys: 0 was the old "never
    # snapshot" spelling, and the schema now requires >= 1 because `enabled`
    # is the one way to say it.)
    cfg.logging.snapshots.enabled = False
    cfg.logging.snapshots.max_calls = 1
    cfg.logging.snapshots.save_images = False
    cfg.logging.snapshots.save_json = False
    cfg.logging.snapshots.interval_steps = 0
    cfg.training.output_dir = "/tmp"
    cfg.adapters = None
    return cfg


def _make_minimal_env(cfg: MagicMock | None = None) -> MagicMock:
    """Return a MagicMock that satisfies TrainingEnvironment duck-typing checks."""
    import torch

    from spectramr.infrastructure.training.builders.environment import TrainingEnvironment

    if cfg is None:
        cfg = _make_minimal_config()

    env = MagicMock(spec=TrainingEnvironment)
    env.config = cfg
    env.device = torch.device("cpu")

    gen = MagicMock()
    gen.training = True
    env.models = {"generator": gen}
    env.generator = gen
    env.discriminator = None
    env.opt_g = MagicMock()
    env.opt_d = None
    env.model_type = "reconstruction"
    env.losses = {}
    env.run_output_dir = "/tmp"
    return env


# ---------------------------------------------------------------------------
# 0. The fixture's own off-switch
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_minimal_config_actually_disables_snapshots() -> None:
    """The snapshot off-switch in ``_make_minimal_config`` must really be off.

    Asked of the resolver, not of the attribute we just set -- asserting the
    flag we wrote is what let this rot unnoticed. When the block decomposition
    moved these keys, the old flat spelling stopped being read and
    the resolver fell through to the MagicMock's auto-vivified
    ``logging.snapshots``, whose leaves are truthy: the fixture silently
    resolved to snapshots ENABLED. Nothing failed, because nothing asked.
    """
    from spectramr.infrastructure.training.snapshot.policy import SnapshotSchedule

    schedule = SnapshotSchedule.from_logging(_make_minimal_config().logging)
    assert schedule.enabled is False
    assert not schedule.due("train", 0)


# ---------------------------------------------------------------------------
# 1. Structural: BaseTrainingStrategy is abstract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_base_training_strategy_is_abstract() -> None:
    """BaseTrainingStrategy must not be directly instantiable."""
    from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

    # BaseTrainingStrategy can't mark _compute_losses_impl @abstractmethod
    # (≈14 concrete strategies override _compute_losses / train_step instead),
    # so __init__ guards direct instantiation with an explicit
    # ``type(self) is BaseTrainingStrategy`` check that raises TypeError.
    env = _make_minimal_env()

    with pytest.raises(TypeError):
        # Attempt direct instantiation without overriding _compute_losses_impl
        BaseTrainingStrategy(env=env)


@pytest.mark.unit
def test_base_training_strategy_compute_losses_impl_is_abstract() -> None:
    """_compute_losses_impl on BaseTrainingStrategy raises NotImplementedError."""
    from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

    # Build a concrete subclass that only exists to instantiate the base
    # without triggering the full __init__ chain (heavy DI / AMP setup).
    # We patch _setup_strategy_specific_components and BaseTrainingStrategy.__init__
    # to isolate the method-level check.
    with patch.object(BaseTrainingStrategy, "__init__", lambda self, **kw: None):
        # Create a subclass that provides the required abstractmethod
        class _ConcreteForTest(BaseTrainingStrategy):
            def _compute_losses_impl(self, input_batch, target_batch, epoch, **kw):
                raise NotImplementedError(
                    "Subclasses must implement _compute_losses_impl"
                )

        instance = _ConcreteForTest.__new__(_ConcreteForTest)

    # Directly call the base method to confirm it raises
    import torch

    with pytest.raises(NotImplementedError):
        BaseTrainingStrategy._compute_losses_impl(
            instance,
            input_batch=torch.zeros(1, 1, 4, 4),
            target_batch=torch.zeros(1, 1, 4, 4),
            epoch=0,
        )


# ---------------------------------------------------------------------------
# 2. Public lifecycle hooks accept valid call signatures
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lifecycle_hooks_signatures() -> None:
    """Public lifecycle hooks on BaseTrainingStrategy must accept the declared args."""
    from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

    # Verify the hook signatures — at least check the parameter names exist.
    for method_name, required_params in [
        ("on_epoch_start", {"epoch"}),
        ("on_epoch_end", {"epoch", "metrics"}),
        ("on_validation_start", set()),
        ("on_validation_end", {"metrics"}),
    ]:
        method = getattr(BaseTrainingStrategy, method_name)
        sig = inspect.signature(method)
        param_names = set(sig.parameters.keys()) - {"self"}
        assert required_params.issubset(param_names), (
            f"{method_name}: expected params {required_params}, " f"got {param_names}"
        )


@pytest.mark.unit
def test_lifecycle_hooks_are_callable() -> None:
    """Lifecycle hooks must be callable (not None / property)."""
    from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

    for name in (
        "on_epoch_start",
        "on_epoch_end",
        "on_validation_start",
        "on_validation_end",
    ):
        attr = getattr(BaseTrainingStrategy, name)
        assert callable(attr), f"{name} must be callable"


# ---------------------------------------------------------------------------
# 3. LossResult typed container
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_loss_result_to_dict_and_get_scalar() -> None:
    """LossResult.to_dict returns the losses dict; get_scalar handles missing keys."""
    import torch

    from spectramr.infrastructure.training.strategies.base import LossResult

    t = torch.tensor(0.5)
    result = LossResult(
        losses={"g_total_loss": t, "loss_l1": t * 0.5}, metrics={"psnr": 32.0}
    )

    d = result.to_dict()
    assert "g_total_loss" in d
    assert "loss_l1" in d

    assert result.get_scalar("psnr") == pytest.approx(32.0)
    assert result.get_scalar("missing_key", default=0.0) == pytest.approx(0.0)


@pytest.mark.unit
def test_loss_result_none_metrics_get_scalar() -> None:
    """LossResult.get_scalar returns default when metrics is None."""
    import torch

    from spectramr.infrastructure.training.strategies.base import LossResult

    result = LossResult(losses={"g_total_loss": torch.tensor(1.0)}, metrics=None)
    assert result.get_scalar("anything", default=-1.0) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 4. _verify_strategy_config raises when config is None
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_verify_strategy_config_raises_on_none_config() -> None:
    """_verify_strategy_config raises ValueError when self.config is None."""
    from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

    with patch.object(BaseTrainingStrategy, "__init__", lambda self, **kw: None):

        class _Stub(BaseTrainingStrategy):
            def _compute_losses_impl(self, *a, **kw):  # type: ignore[override]
                raise NotImplementedError

        stub = _Stub.__new__(_Stub)
        stub.config = None  # type: ignore[assignment]

    with pytest.raises(ValueError, match="valid config"):
        stub._verify_strategy_config(expected_modes=())


# ---------------------------------------------------------------------------
# 5. apply_adapters is a no-op when no chain is declared
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_adapters_noop_when_no_chain() -> None:
    """apply_adapters must return input unchanged when no chain is registered."""
    import torch

    from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

    with patch.object(BaseTrainingStrategy, "__init__", lambda self, **kw: None):

        class _Stub(BaseTrainingStrategy):
            def _compute_losses_impl(self, *a, **kw):  # type: ignore[override]
                raise NotImplementedError

        stub = _Stub.__new__(_Stub)
        stub.adapter_chains = {}  # empty — no chains registered

    x = torch.zeros(1, 1, 4, 4)
    out = stub.apply_adapters("pre_model", x)
    assert (
        out is x
    ), "apply_adapters must return the input tensor unchanged when no chain is declared"


# ---------------------------------------------------------------------------
# 6. get_last_metrics returns dict[str, float]
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_last_metrics_returns_on_device_values() -> None:
    """get_last_metrics returns the stored values UNCONVERTED (#707).

    This asserted ``str -> float`` until the sync audit inverted the contract.
    ``float(cuda_tensor)`` IS ``.item()``, and `training_loop` calls this on
    EVERY iteration, outside the ``log_interval`` gate -- so converting here
    cost one host sync per component metric per step and threw the result away
    on every non-logging step. The loop's gated converter now owns the single
    fused transfer; all four implementations return raw values.
    """
    import torch

    from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

    with patch.object(BaseTrainingStrategy, "__init__", lambda self, **kw: None):

        class _Stub(BaseTrainingStrategy):
            def _compute_losses_impl(self, *a, **kw):  # type: ignore[override]
                raise NotImplementedError

        stub = _Stub.__new__(_Stub)
        stub._last_step_metrics = {
            "loss_l1": torch.tensor(0.42),
            "psnr": torch.tensor(28.5),
        }

    result = stub.get_last_metrics()
    assert isinstance(result, dict)
    for k, v in result.items():
        assert isinstance(k, str)
        assert isinstance(
            v, torch.Tensor
        ), "converting here re-pays the per-step sync #707 removed"
    # A copy, so a caller mutating the result cannot corrupt the next step.
    result["loss_l1"] = 0.0
    assert stub._last_step_metrics["loss_l1"] != 0.0


# ---------------------------------------------------------------------------
# 7. HANDLED_TRAINING_ERRORS tuple is exported and contains expected types
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_handled_training_errors_exported() -> None:
    """HANDLED_TRAINING_ERRORS must be a non-empty tuple of exception types."""
    from spectramr.infrastructure.training.strategies.base import HANDLED_TRAINING_ERRORS

    assert isinstance(HANDLED_TRAINING_ERRORS, tuple)
    assert len(HANDLED_TRAINING_ERRORS) > 0
    for exc_type in HANDLED_TRAINING_ERRORS:
        assert issubclass(exc_type, BaseException)


# ---------------------------------------------------------------------------
# 8. WS-3 PR-3: every strategy exposes a mutable loop_state seam, defaulted to 0
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_strategy_exposes_loop_state_seam_defaulted_to_zero() -> None:
    """The env-holding base ``__init__`` must seed ``self.loop_state`` with a
    fresh :class:`LoopState` (iteration/epoch 0).

    This is the seam the training loop advances each step and the diffusion
    curriculum diagnostic reads — replacing the frozen, perpetually-zero
    ``env.step``. The default-0 matters: a strategy constructed outside the loop
    (tests / scripting) must still read a sane iteration rather than crash."""
    from spectramr.infrastructure.training.loop_state import LoopState
    from spectramr.infrastructure.training.strategies.base import (
        BaseTrainingStrategy,
        TrainingStepStrategy,
    )

    env = _make_minimal_env()

    class _Stub(BaseTrainingStrategy):
        def _compute_losses_impl(self, *a, **kw):  # type: ignore[override]
            raise NotImplementedError

    stub = _Stub.__new__(_Stub)
    # Run only the env-holder __init__ (which seeds loop_state) — the full
    # BaseTrainingStrategy.__init__ chain needs GPU + full config scaffolding.
    TrainingStepStrategy.__init__(stub, env=env)

    assert isinstance(stub.loop_state, LoopState)
    assert stub.loop_state.iteration == 0
    assert stub.loop_state.epoch == 0

    # Mutating it is observable on the strategy (the loop writes through it).
    stub.loop_state.iteration = 27_500
    assert stub.loop_state.iteration == 27_500


# ---------------------------------------------------------------------------
# sync_scheduled_loss_weights — the paradigm-agnostic loss_schedule seam
# (regression for the "silent no-op outside reconstruction" critical finding).
# Exercised as an unbound method on a lightweight stub so no GPU / full config
# scaffolding is needed.
# ---------------------------------------------------------------------------


def _sync(loss_computer: Any, loop_state: Any) -> Any:
    from types import SimpleNamespace

    from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

    stub = SimpleNamespace(loss_computer=loss_computer, loop_state=loop_state)
    BaseTrainingStrategy.sync_scheduled_loss_weights(stub)
    return stub


def test_sync_publishes_overrides_to_loss_computer() -> None:
    from types import SimpleNamespace

    lc = SimpleNamespace(scheduled_weights={})
    ls = SimpleNamespace(loss_weight_overrides={"l1": 0.3, "perceptual": 0.0})
    _sync(lc, ls)
    assert lc.scheduled_weights == {"l1": 0.3, "perceptual": 0.0}


def test_sync_empty_overrides_yields_empty_dict() -> None:
    from types import SimpleNamespace

    lc = SimpleNamespace(scheduled_weights={"stale": 1.0})
    ls = SimpleNamespace(loss_weight_overrides={})
    _sync(lc, ls)
    assert lc.scheduled_weights == {}  # static-config behavior, never None


def test_sync_is_noop_without_loss_computer() -> None:
    from types import SimpleNamespace

    # A strategy that owns no loss_computer must not raise (e.g. inline-loss
    # paradigms). The method simply returns.
    _sync(None, SimpleNamespace(loss_weight_overrides={"l1": 0.5}))


def test_sync_handles_missing_loop_state() -> None:
    from types import SimpleNamespace

    lc = SimpleNamespace(scheduled_weights={"stale": 2.0})
    _sync(lc, None)
    assert lc.scheduled_weights == {}


# ---------------------------------------------------------------------------
# The capability contract is declared exactly once (ast-level).
# ---------------------------------------------------------------------------


def test_base_declares_capabilities_exactly_once() -> None:
    """``capabilities`` must be declared once on BaseTrainingStrategy.

    Two workstreams each added the ClassVar (the workflow ledger's regime tags
    and the cached-cascade config contract), separated by an unrelated field so
    neither review caught it. Python silently accepts the re-annotation and the
    second wins — harmless only because both defaults happened to be equal. Had
    either carried a value it would have been dropped without a sound.

    This is an ast assertion because a duplicate declaration leaves NO runtime
    signature: ``BaseTrainingStrategy.capabilities`` reads back fine either way.
    The source is the only place the bug is visible.
    """
    import ast
    import inspect as _inspect

    from spectramr.infrastructure.training.strategies import base as _base

    tree = ast.parse(_inspect.getsource(_base))
    class_def = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "BaseTrainingStrategy"
    )
    declarations = [
        stmt
        for stmt in class_def.body
        if isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
        and stmt.target.id == "capabilities"
    ]
    assert len(declarations) == 1, (
        f"BaseTrainingStrategy declares `capabilities` {len(declarations)} times "
        f"(lines {[d.lineno for d in declarations]}). Declare it exactly once — "
        "a second annotation silently shadows the first."
    )
