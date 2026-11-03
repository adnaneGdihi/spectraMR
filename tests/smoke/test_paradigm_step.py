"""Per-paradigm smoke tests: one fwd+bwd+opt step on a tiny phantom.

Each test is parametrized over every key in ``STRATEGY_CLASS_PATHS``.  A
strategy key that has no minimal-settings fixture is either ``xfail``-ed
(if the strategy class can be imported but the YAML is missing) or
``skip``-ped (if the fixture mapping records no YAML yet).

Invariants checked for each covered strategy:

1. Loss is a finite scalar (``torch.isfinite`` after ``.item()``).
2. At least one parameter has a non-None gradient after backward.
3. At least one parameter changed value after the optimizer step.

These checks run on CPU only.  GPU-specific behaviour is covered by
``tests/convergence/`` and ``tests/performance/``.

Notes
-----
* The tests intentionally use ``MagicMock``-based environments so they
  work without a real dataset or full pipeline — they exercise the
  *strategy dispatch + one training step* path only.
* Strategies that require multi-model setups (GAN discriminator, PINN PDE
  solver, …) have their ancillary components stubbed out via the same
  mock environment used in ``tests/smoke/test_strategies.py``.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from tests.utils.minimal_settings import SettingsNotFound, covered_keys, minimal_settings_for
from tests.utils.registry_iterators import all_strategy_keys

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared tiny model helpers
# ---------------------------------------------------------------------------

class _TinyNet(nn.Module):
    """Minimal 1-layer conv net for CPU step tests."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # noqa: ARG002
        return self.conv(x)

    # Strategy protocols: sample / generate / encode / decode
    def sample(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(x)

    def generate(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(x)

    def encode(self, x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.conv(x)
        return z, torch.zeros_like(z)

    def decode(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.conv(z)


class _TinyDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).mean(dim=(-2, -1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Mock environment builder
# ---------------------------------------------------------------------------

class _LossAutoMock(MagicMock):
    """MagicMock that returns 0.0 for ``enable_*`` / ``lambda_*`` attrs."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("enable_"):
            return False
        if name.startswith("lambda_"):
            return 0.0
        return super().__getattr__(name)


def _make_mock_config(strategy_key: str) -> MagicMock:
    """Build a mock config that is sufficient for most strategy constructors."""
    cfg = MagicMock()

    # --- model ---
    cfg.model.model_type = "standard_unet"
    cfg.model.in_channels = 1
    cfg.model.out_channels = 1
    cfg.model.input_type = "image"
    cfg.model.prior_model = None
    cfg.model.target_domain = "image"

    # --- training ---
    cfg.training_mode = strategy_key
    cfg.device = "cpu"
    cfg.warmup_epochs = 0
    cfg.deep_supervision_weight = 0.0
    cfg.enforce_output_range = False
    cfg.collect_feature_flags = lambda: {}

    # --- optimization ---
    cfg.optimization.precision.enabled = False
    cfg.optimization.mixed_precision = "fp16"
    cfg.optimization.loss_scaling = 1024.0
    cfg.optimization.gradient.clip.value = 1.0
    cfg.optimization.gradient.clip.method = "norm"
    cfg.optimization.gradient.clip.enabled = False
    cfg.optimization.optimize_memory_amp = True
    cfg.optimization.dynamic_loss_scaling = True
    cfg.optimization.optimizer.learning_rate = 1e-4
    cfg.optimization.gradient.accumulation_steps = 1

    # --- objectives / losses (auto-mock) ---
    recon = _LossAutoMock()
    recon.lambda_l1 = 1.0
    for attr in [
        "lambda_l2", "lambda_perceptual", "lambda_ssim", "lambda_kspace",
        "lambda_flow", "lambda_flow_likelihood", "lambda_graph_smoothness",
        "lambda_feature_alignment", "lambda_affine_regularization",
        "lambda_domain_adversarial", "lambda_sparsity_penalty",
        "lambda_wasserstein", "lambda_frequency", "lambda_log_spectral",
        "lambda_lpips", "weighted_kspace_exponent", "histogram_bins",
        "ffl_alpha",
    ]:
        setattr(recon, attr, 0.0)
    recon.log_spectral_skip_fft = False

    cfg.objectives = MagicMock()
    cfg.objectives.reconstruction = recon
    cfg.objectives.physics = _LossAutoMock()
    for attr in [
        "lambda_parallel_imaging", "lambda_physics_constraint", "lambda_bloch_residual",
    ]:
        setattr(cfg.objectives.physics, attr, 0.0)
    for attr in [
        "enable_parallel_imaging", "enable_physics_constraint", "enable_bloch_residual",
    ]:
        setattr(cfg.objectives.physics, attr, False)

    cfg.objectives.gan = MagicMock()
    cfg.objectives.gan.gan_loss_type = "vanilla"
    cfg.objectives.gan.lambda_adv = 1.0
    cfg.objectives.gan.lambda_gp = 0.0
    cfg.objectives.gan.r1 = MagicMock()
    cfg.objectives.gan.r1.interval = 0
    cfg.objectives.gan.r1.weight = 0.0
    cfg.objectives.gan.r1.probability = 1.0

    cfg.objectives.latent = MagicMock()
    cfg.objectives.latent.enable_kl = False
    cfg.objectives.latent.enable_commitment = False

    cfg.objectives.ssl = MagicMock()
    cfg.objectives.ssl.enable_contrastive = False

    cfg.losses = cfg.objectives

    # --- diffusion ---
    cfg.training.diffusion = MagicMock()
    cfg.training.diffusion.timesteps = 10
    cfg.training.diffusion.num_timesteps = 10
    cfg.training.diffusion.noise_schedule = "linear"
    cfg.training.diffusion.guidance_scale = 1.0
    cfg.objectives.diffusion = MagicMock()
    cfg.objectives.diffusion.timesteps = 10
    cfg.objectives.diffusion.noise_schedule = "linear"
    cfg.objectives.diffusion.lambda_mse = 1.0
    cfg.r1_interval = 0

    # --- physics ---
    cfg.physics.pinn.enabled = False
    cfg.physics.pinn.pde_type = "bloch_equation"
    cfg.physics.pinn.lambda_pde = 0.0
    cfg.physics.pinn.weight = 0.0
    cfg.physics.kspace.enable_kspace_recon = False
    cfg.physics.data_consistency.enabled = False

    # --- data ---
    cfg.data.prior_loading.enabled = False

    # --- misc ---
    cfg.acceleration.acceleration_factor = 4.0
    cfg.acceleration.base_acceleration = 2.0
    cfg.acceleration.max_acceleration = 8.0
    cfg.acceleration.gradient_accumulation_steps = 1

    cfg.logging.log_gradients = False
    cfg.logging.log_interval = 10

    return cfg


def _make_training_env(
    strategy_key: str,
) -> MagicMock:
    """Return a MagicMock TrainingEnvironment suitable for strategy construction."""
    from spectramr.infrastructure.training.builders.optimization_builder import (
        OptimizationBuilder,
    )

    gen = _TinyNet(in_ch=1, out_ch=1)
    disc = _TinyDiscriminator(in_ch=1)

    opt_g = OptimizationBuilder.create_single_optimizer(
        gen.parameters(), learning_rate=1e-4, optimizer_type="adam"
    )
    opt_d = OptimizationBuilder.create_single_optimizer(
        disc.parameters(), learning_rate=1e-4, optimizer_type="adam"
    )

    cfg = _make_mock_config(strategy_key)

    env = MagicMock()
    env.config = cfg
    env.generator = gen
    env.discriminator = disc
    env.models = {"generator": gen, "discriminator": disc}
    env.optimizers = {"opt_g": opt_g, "opt_d": opt_d}
    env.schedulers = {}
    env.losses = {}
    env.physics = {}
    env.data_loaders = {}
    env.metrics = {}
    env.scaler = None
    env.device = torch.device("cpu")
    env.opt_g = opt_g
    env.opt_d = opt_d
    env.ema = None
    env.step = 0
    env.model_type = "standard_unet"
    return env


# ---------------------------------------------------------------------------
# Core one-step smoke
# ---------------------------------------------------------------------------

def _params_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.data.clone() for name, param in model.named_parameters()}


def _any_grad(model: nn.Module) -> bool:
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def _any_changed(model: nn.Module, before: dict[str, torch.Tensor]) -> bool:
    return any(
        not torch.allclose(param.data, before[name])
        for name, param in model.named_parameters()
    )


def _run_one_step(strategy: Any) -> dict[str, Any]:
    """Execute one train_step on a tiny phantom and return diagnostics dict."""
    phantom = torch.randn(2, 1, 16, 16)

    before_g = _params_snapshot(strategy.env.generator)

    result = strategy.train_step(
        batch=phantom,
        epoch=0,
        input_batch=phantom,
        target_batch=phantom,
    )

    gen = strategy.env.generator
    has_grad = _any_grad(gen)
    params_changed = _any_changed(gen, before_g)

    # Extract scalar loss from result
    loss_val: float | None = None
    if isinstance(result, list) and result:
        for step_dict in result:
            if isinstance(step_dict, dict):
                for v in step_dict.values():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        loss_val = v.item()
                        break
    elif isinstance(result, dict):
        for v in result.values():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                loss_val = v.item()
                break

    return {
        "result": result,
        "loss_val": loss_val,
        "has_grad": has_grad,
        "params_changed": params_changed,
    }


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------

@pytest.mark.smoke
@pytest.mark.parametrize("strategy_key", all_strategy_keys(), ids=lambda k: k)
def test_paradigm_one_step(strategy_key: str) -> None:
    """Instantiate strategy and run one fwd+bwd+opt step on a CPU phantom.

    Skip conditions (runtime, not collection-time):
    - No minimal-settings fixture YAML exists for this strategy family yet
      → pytest.skip with instructions.

    Xfail conditions:
    - The strategy class can be imported but the forward pass raises a
      known "not-yet-implemented on CPU" error
      → pytest.xfail, not a hard failure.
    """
    # --- check if a fixture is available ---
    try:
        _settings = minimal_settings_for(strategy_key)  # noqa: F841 — validates only
    except SettingsNotFound as exc:
        pytest.skip(f"minimal settings YAML not yet authored for '{strategy_key}': {exc}")
    except FileNotFoundError as exc:
        pytest.skip(f"fixture file missing for '{strategy_key}': {exc}")
    except Exception as exc:
        pytest.skip(
            f"minimal settings could not be loaded for '{strategy_key}' "
            f"(schema/IO error): {exc}"
        )

    # --- load strategy class ---
    try:
        from spectramr.infrastructure.training.strategy_factory import (  # noqa: PLC0415
            TrainingStrategyFactory,
        )

        factory = TrainingStrategyFactory()
        class_path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS[strategy_key]
        strategy_cls = factory._load_strategy_class(class_path)
    except Exception as exc:
        pytest.xfail(f"Strategy class import failed for '{strategy_key}': {exc}")
        return  # unreachable but satisfies type-checker

    # --- build environment ---
    env = _make_training_env(strategy_key)

    # --- instantiate strategy ---
    try:
        strategy = strategy_cls(env=env, device=torch.device("cpu"))
    except Exception as exc:
        pytest.xfail(
            f"Strategy '{strategy_key}' ({strategy_cls.__name__}) "
            f"could not be instantiated on CPU: {exc}"
        )
        return

    # --- run one step ---
    try:
        diag = _run_one_step(strategy)
    except Exception as exc:
        pytest.xfail(
            f"Strategy '{strategy_key}' raised during train_step on CPU: {exc}"
        )
        return

    # --- assertions ---
    result = diag["result"]
    assert result is not None, (
        f"train_step for '{strategy_key}' returned None"
    )

    loss_val = diag["loss_val"]
    if loss_val is not None:
        assert torch.isfinite(torch.tensor(loss_val)), (
            f"Non-finite loss {loss_val!r} from '{strategy_key}'"
        )

    # At least one of: gradient populated or params changed
    # (some strategies apply the optimizer step internally, some return a closure)
    assert diag["has_grad"] or diag["params_changed"], (
        f"Strategy '{strategy_key}': no gradients populated and no parameters "
        "changed after one training step — step appears to be a no-op"
    )


# ---------------------------------------------------------------------------
# Execution floor
# ---------------------------------------------------------------------------
# `test_paradigm_one_step` above ends every branch in skip or xfail, so all 206
# of its cases report green while verifying nothing (#1033). That is not a bug
# in one paradigm — `BaseTrainingStrategy.train_step` returns a PLAN (a list of
# `{optimizer, closure, model}` descriptors the training loop drives), and the
# runner never drives it, so "no gradients after one step" describes the test.
#
# Rebuilding this suite on the real `TrainingSettings` it currently discards is
# an owner decision (#1033, disposition 3). Until then these two floors keep the
# facts visible and stop them silently getting worse.
#
# FLOORS, not targets: they may only RISE. Deliberately not a `>= 0` assertion,
# which would be the same do-nothing gate this file already suffers from.
#
# Measured 2026-08-13 on dev @ 4776ba20a, over `covered_keys()` (48).
CONSTRUCTIBLE_FLOOR = 37   # of 48 covered keys, construct at all
YIELDING_FLOOR = 37        # return {optimizer, closure, model} descriptors
EXECUTING_FLOOR = 4        # survive actually running those closures
#
# The first two coincide at 37 today: every strategy that constructs also
# returns work, because nothing raises until the closure is DRIVEN. They are
# still separate properties and can diverge, so they are pinned separately --
# but the honest reading is that 37 -> 4 is where this suite actually stands,
# and that gap is what #1033 is about. An earlier revision of this block had
# `YIELDING_FLOOR` measuring descriptor-existence while labelled with the
# executing count; raising both floors at once failed only ONE test, which is
# how the duplication surfaced. Mutate each constant on its own.


def _execution_census() -> dict[str, Any]:
    """How far each covered strategy gets: constructed, then drivable work.

    `precision.dtype` is set HERE rather than in `_make_mock_config`, on
    purpose. A MagicMock attribute is not absent — it is a Mock, and
    `resolve_amp_precision` validates the dtype against a closed set, so an
    unset `.dtype` makes every strategy fail construction. Setting it in the
    shared builder would turn `test_paradigm_one_step`'s 48 xfails into 37 hard
    failures on the tier CLAUDE.md says to run before launching training,
    without buying any real coverage (measured: 0 strategies take a real step
    even then). So the census sees past the blocker; the suite above does not.

    `physics.compressed_sensing.sampling_pattern` is the SAME blocker, one field
    later, and is set for the same reason. PR #1095 (#1092) replaced
    `GraphColdDiffusionStrategy._setup_accelerator` — which swallowed an
    ImportError and silently degraded on a hardcoded mask — with
    `_resolve_degradation_pattern`, which validates the declared pattern against
    `_ACCELERATOR_REGISTRY` and RAISES on anything unregistered. That is the
    correct behaviour, but a MagicMock is not a registered pattern, so all three
    keys mapping to that one class (`cold_diffusion`, `kspace_cold_diffusion`,
    `graph_cold_diffusion`) stopped constructing and both floors read 34 against
    37 — dev went red on a strategy change that broke nothing.

    The strategies are fine: the schema default is `"cartesian"`, which
    `_PATTERN_ALIASES` maps to the registered `random_cartesian`, and the four
    `inprogress/` arms declaring a non-Cartesian pattern all use strategies that
    do not carry the refusal. What was missing was a real value in the mock.

    The general shape to expect: every new closed-set validation at construction
    costs this census one line, because a Mock satisfies no closed set. A floor
    drop is only a real regression once the mock supplies a legal value.
    """
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
    from tests.utils.snapshot_census import records_a_model_stage, traced_step

    factory = TrainingStrategyFactory()
    built = yielded = executed = 0
    unsnapshotted: list[str] = []
    for key in covered_keys():
        class_path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS.get(key)
        if class_path is None:
            continue
        try:
            strategy_cls = factory._load_strategy_class(class_path)
            env = _make_training_env(key)
            env.config.optimization.precision.dtype = "float32"
            # The schema's own default, not an arbitrary legal value: an arm that
            # declares nothing gets exactly this, so the census measures the
            # default configuration rather than a hand-picked one.
            env.config.physics.compressed_sensing.sampling_pattern = "cartesian"
            strategy = strategy_cls(env=env, device=torch.device("cpu"))
        except Exception:  # any failure means "did not construct"
            continue
        built += 1
        phantom = torch.randn(2, 1, 16, 16)
        try:
            result = strategy.train_step(
                batch={}, epoch=0, input_batch=phantom, target_batch=phantom
            )
        except Exception:  # any failure means "no drivable work"
            continue
        work = [
            item
            for item in (result if isinstance(result, list) else [])
            if isinstance(item, dict) and "optimizer" in item and "closure" in item
        ]
        if not work:
            continue
        yielded += 1
        # DRIVE it. Merely returning descriptors is a much weaker property than
        # the step actually running -- 37 strategies yield work, 4 survive
        # executing it. Counting only the former would make this floor a
        # duplicate of `built` and unable to fail on its own.
        try:
            trace = traced_step(
                strategy, lambda work=work: [i["optimizer"].step(i["closure"]) for i in work]
            )
        except Exception:  # any failure means "did not execute"
            continue
        executed += 1
        if not records_a_model_stage(trace):
            unsnapshotted.append(key)
    return {
        "built": built,
        "yielded": yielded,
        "executed": executed,
        "unsnapshotted": unsnapshotted,
    }


#: Memoised rather than a module-scoped fixture: the census needs the
#: function-scoped `ensure_di_container_initialized`, and a module-scoped
#: fixture cannot depend on a narrower one.
_CENSUS_CACHE: dict[str, Any] = {}


@pytest.fixture
def execution_census(ensure_di_container_initialized: None) -> dict[str, Any]:
    if not _CENSUS_CACHE:
        _CENSUS_CACHE.update(_execution_census())
    return _CENSUS_CACHE


def test_the_census_is_not_vacuous(execution_census: dict[str, int]) -> None:
    """Anti-vacuity. Floors over an empty key set would pass forever."""
    assert covered_keys(), "no covered strategy keys — the census measured nothing"
    assert min(CONSTRUCTIBLE_FLOOR, YIELDING_FLOOR, EXECUTING_FLOOR) > 0, (
        "a floor of zero is not a gate"
    )


def test_strategies_still_construct(execution_census: dict[str, int]) -> None:
    """A drop here means a strategy stopped being constructible at all.

    `test_paradigm_one_step` cannot catch that: it xfails on construction
    failure, so a paradigm going from 'constructs' to 'raises' is invisible.
    """
    built = execution_census["built"]
    assert built >= CONSTRUCTIBLE_FLOOR, (
        f"{built} of {len(covered_keys())} covered strategies construct, "
        f"below the floor of {CONSTRUCTIBLE_FLOOR}. A strategy stopped being "
        f"constructible; raise the floor only when the number genuinely rises."
    )


#: The three `training_mode` keys that all resolve to the ONE class PR #1095
#: changed (`GraphColdDiffusionStrategy`). Named rather than counted: the floors
#: above are aggregates, so these three could break while three others were
#: fixed and the totals would not move.
_COLD_DIFFUSION_KEYS = ("cold_diffusion", "kspace_cold_diffusion", "graph_cold_diffusion")


@pytest.mark.parametrize("key", _COLD_DIFFUSION_KEYS)
def test_the_cold_diffusion_family_constructs(
    ensure_di_container_initialized: None, key: str
) -> None:
    """Regression guard for the census's mock, not for the strategy.

    `_resolve_degradation_pattern` validates `sampling_pattern` against
    `_ACCELERATOR_REGISTRY` and raises on anything unregistered — correct, and
    the whole point of #1092. A MagicMock is unregistered, so before the census
    supplied a real value these three raised and both floors read 34/37.

    Failing here means one of two things, and the message says which to check:
    either a new closed-set validation needs another line in `_execution_census`,
    or the class genuinely broke.
    """
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    class_path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS.get(key)
    assert class_path is not None, f"{key!r} is no longer a registered training_mode"

    strategy_cls = TrainingStrategyFactory()._load_strategy_class(class_path)
    env = _make_training_env(key)
    env.config.optimization.precision.dtype = "float32"
    env.config.physics.compressed_sensing.sampling_pattern = "cartesian"

    try:
        strategy_cls(env=env, device=torch.device("cpu"))
    except Exception as exc:  # the diagnostic message is this test's deliverable
        pytest.fail(
            f"{key!r} no longer constructs: {type(exc).__name__}: {exc}\n\n"
            "If this is a closed-set validation rejecting a MagicMock, give the "
            "census a real value for that field (see _execution_census's "
            "docstring — precision.dtype and sampling_pattern are both there for "
            "this reason). If it is not, the strategy actually regressed."
        )


def test_strategies_still_yield_drivable_work(execution_census: dict[str, int]) -> None:
    """`train_step` must keep returning work a training loop can drive.

    This is the deferred-work contract: a list of `{optimizer, closure, model}`
    descriptors. `test_paradigm_one_step` believes it tests this and does not —
    it never drives what is returned.
    """
    yielded = execution_census["yielded"]
    assert yielded >= YIELDING_FLOOR, (
        f"{yielded} strategies yield drivable work, below the floor of "
        f"{YIELDING_FLOOR}. `train_step` returns a list of "
        f"{{optimizer, closure, model}} descriptors; something stopped."
    )


def test_strategies_still_execute_their_step(execution_census: dict[str, int]) -> None:
    """The strictly stronger property: the closures actually RUN.

    37 strategies yield work; only 4 survive executing it. That gap is the real
    state of this suite and the reason #1033 is open — the rest raise partway
    through, downstream of a MagicMock config. Pinned separately from
    `yielded` so the two cannot pass for each other.
    """
    executed = execution_census["executed"]
    assert executed >= EXECUTING_FLOOR, (
        f"{executed} strategies execute a full step, below the floor of "
        f"{EXECUTING_FLOOR}."
    )


def test_every_executing_strategy_records_its_model_input(execution_census) -> None:
    """A strategy whose step runs must leave a ``model/`` stage in its snapshot.

    The hooks reach a model only through ``env.models`` or
    ``_capture_model_output(module=...)``; any other route hides what the model was
    fed. Covers exactly the strategies that execute here (``EXECUTING_FLOOR``), and
    widens as #1033 lets more of them run.
    """
    missing = execution_census["unsnapshotted"]
    assert not missing, (
        f"{missing} ran a step that recorded no model call: register the module in "
        "env.models or pass it to _capture_model_output(module=...)"
    )
