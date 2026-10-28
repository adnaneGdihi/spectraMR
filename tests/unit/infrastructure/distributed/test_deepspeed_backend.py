"""DeepSpeed backend: the ds_config generator, the step policy, the guards.

The config-builder tests are the important ones and they are **pure** -- no
torch, no engine, no GPU. That is deliberate: the risk this backend carries is
not "does the engine start", it is "does the ds_config say something different
from the YAML". DeepSpeed will happily accept a
``train_micro_batch_size_per_gpu`` that contradicts ``data.batch_size``, and
nothing anywhere compares them, so the run succeeds at the wrong experiment.

Hence no ``config_path`` field: a hand-edited ds_config.json is a
cluster-critical file with no committed generator, and unlike a data manifest
(which fails loudly with "Index file not found") its divergence is silent.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tests.utils.config_block_stub import block_stub  # noqa: E402
from tests.utils.data_config_stub import DataConfigStub  # noqa: E402

nn = pytest.importorskip("torch.nn")

from spectramr.config.schemas.base import ParallelismConfigSchema  # noqa: E402
from spectramr.infrastructure.distributed.deepspeed_backend import (  # noqa: E402
    DERIVED_KEYS,
    DeepSpeedStepPolicy,
    build_deepspeed_config,
    deepspeed_available,
)
from spectramr.infrastructure.distributed.strategy_registry import (  # noqa: E402
    resolve_parallel_strategy,
)


def _settings(*, batch_size=4, accum=8, clip=1.0, use_amp=False, amp_dtype=None, **ds):
    parallel = ParallelismConfigSchema(
        strategy="deepspeed", deepspeed={"enabled": True, **ds}
    )
    # Shared stubs, not hand-rolled namespaces: the block decomposition moved
    # every leaf below (`data.batch_size` -> `data.loader.batch_size`,
    # `use_amp` -> `optimization.precision.enabled`, `gradient_clip_value` ->
    # `optimization.gradient.clip.value`), and these route from RENAMES so the
    # stub cannot disagree with the loader about where a reader looks.
    return SimpleNamespace(
        parallel=parallel,
        data=DataConfigStub(batch_size=batch_size),
        optimization=block_stub(
            "optimization",
            gradient_accumulation_steps=accum,
            gradient_clip_value=clip,
            enable_gradient_clipping=clip is not None,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        ),
    )


class TestTheDsConfigCannotDisagreeWithTheYaml:
    """The keys this generator owns are NOT declarable in the schema."""

    def test_no_config_path_field_exists(self) -> None:
        from spectramr.config.schemas.base import DeepSpeedConfigSchema

        assert not {"config_path", "ds_config_path", "config_file"} & set(
            DeepSpeedConfigSchema.model_fields
        )

    def test_derived_keys_are_not_also_schema_fields(self) -> None:
        """Two homes for one number is exactly how a ds_config drifts."""
        from spectramr.config.schemas.base import DeepSpeedConfigSchema

        assert not DERIVED_KEYS & set(DeepSpeedConfigSchema.model_fields)

    def test_micro_batch_comes_from_data_batch_size(self) -> None:
        cfg = build_deepspeed_config(_settings(batch_size=7))
        assert cfg["train_micro_batch_size_per_gpu"] == 7

    def test_accumulation_comes_from_optimization(self) -> None:
        cfg = build_deepspeed_config(_settings(accum=8))
        assert cfg["gradient_accumulation_steps"] == 8

    def test_clipping_comes_from_optimization(self) -> None:
        assert build_deepspeed_config(_settings(clip=0.5))["gradient_clipping"] == 0.5

    def test_clipping_is_omitted_when_disabled(self) -> None:
        assert "gradient_clipping" not in build_deepspeed_config(_settings(clip=None))

    def test_optimizer_and_scheduler_are_absent(self) -> None:
        """The engine ADOPTS the objects OptimizationBuilder built. Declaring
        them here would fork the optimizer vocabulary away from TTUR, the LR
        multipliers and resolve_scheduler_spec -- and provenance['lr_schedule']
        would describe a scheduler nothing stepped."""
        cfg = build_deepspeed_config(_settings())
        assert "optimizer" not in cfg
        assert "scheduler" not in cfg


class TestPrecisionTracksResolveAmpPrecision:
    def test_amp_off_declares_neither_block(self) -> None:
        cfg = build_deepspeed_config(_settings(use_amp=False))
        assert "fp16" not in cfg and "bf16" not in cfg

    def test_fp16(self) -> None:
        cfg = build_deepspeed_config(_settings(use_amp=True, amp_dtype="float16"))
        assert cfg["fp16"] == {"enabled": True} and "bf16" not in cfg

    def test_bf16(self) -> None:
        cfg = build_deepspeed_config(_settings(use_amp=True, amp_dtype="bfloat16"))
        assert cfg["bf16"] == {"enabled": True} and "fp16" not in cfg

    def test_float32_means_amp_disabled_not_fp16(self) -> None:
        """``resolve_amp_precision`` returns enabled=False for float32 -- the
        precision string alone would say 'fp16' and be wrong."""
        cfg = build_deepspeed_config(_settings(use_amp=True, amp_dtype="float32"))
        assert "fp16" not in cfg and "bf16" not in cfg


class TestZeroBlock:
    @pytest.mark.parametrize("stage", [0, 1, 2, 3])
    def test_stage_is_forwarded(self, stage: int) -> None:
        cfg = build_deepspeed_config(_settings(zero_stage=stage))
        assert cfg["zero_optimization"]["stage"] == stage

    def test_cpu_offload_is_expressed_as_a_device_block(self) -> None:
        cfg = build_deepspeed_config(_settings(offload_optimizer="cpu"))
        assert cfg["zero_optimization"]["offload_optimizer"] == {"device": "cpu"}

    def test_nvme_offload_carries_the_path(self) -> None:
        cfg = build_deepspeed_config(
            _settings(offload_param="nvme", nvme_path="/scratch/nvme")
        )
        assert cfg["zero_optimization"]["offload_param"]["nvme_path"] == "/scratch/nvme"

    def test_stage3_enables_the_16bit_gather_that_consolidation_needs(self) -> None:
        cfg = build_deepspeed_config(_settings(zero_stage=3))
        assert cfg["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"]

    def test_the_gather_key_is_absent_below_stage3(self) -> None:
        cfg = build_deepspeed_config(_settings(zero_stage=2))
        assert (
            "stage3_gather_16bit_weights_on_model_save" not in cfg["zero_optimization"]
        )


class TestStepPolicyOwnership:
    """Both flags True is the entire point of the class."""

    def test_claims_accumulation_and_zero_grad(self) -> None:
        policy = DeepSpeedStepPolicy()
        assert policy.owns_gradient_accumulation is True
        assert policy.owns_zero_grad is True

    def test_the_executor_honours_that_and_stops_dividing(self) -> None:
        """Both dividing gives 1/N^2 loss scaling and one real step per N^2
        micro-batches -- and it TRAINS, so every single-host test still passes."""
        from spectramr.infrastructure.training.step_executor import StepExecutor

        executor = StepExecutor(
            SimpleNamespace(scaler=None),
            DeepSpeedStepPolicy(),
            gradient_accumulation_steps=8,
        )
        assert executor.gradient_accumulation_steps == 1
        assert executor.requested_gradient_accumulation_steps == 8

    def test_backward_and_step_drives_the_engine_every_micro_batch(self) -> None:
        """``perform_step`` is ignored on purpose: the engine decides boundaries
        itself, so step() is a no-op mid-window. Double-gating would stall it."""
        calls: list[str] = []

        class _FakeEngine:
            def backward(self, loss):
                calls.append("backward")

            def step(self):
                calls.append("step")

        optimizer = object()
        policy = DeepSpeedStepPolicy()
        policy.register(optimizer, _FakeEngine())
        policy.backward_and_step(torch.tensor(1.0), optimizer, perform_step=False)
        assert calls == ["backward", "step"]

    def test_an_unregistered_optimizer_raises(self) -> None:
        """Stepping it directly would bypass ZeRO and update unsharded copies."""
        with pytest.raises(RuntimeError, match="no registered engine"):
            DeepSpeedStepPolicy().backward_and_step(torch.tensor(1.0), object())

    def test_guard_loss_is_a_no_op(self) -> None:
        """An overflowing micro-batch is an EXPECTED event under dynamic loss
        scaling; raising would kill runs the engine is designed to survive."""
        DeepSpeedStepPolicy().guard_loss(
            torch.tensor(float("nan")), name="g", global_step=1, scaler=None
        )

    def test_guard_loss_records_nothing_into_an_attached_monitor(self) -> None:
        """The engine skips the overflowing step itself, so the per-config flag
        must not reach the loop's divergence verdict either."""
        from spectramr.infrastructure.training.nonfinite_monitor import NonFiniteMonitor

        monitor = NonFiniteMonitor(torch.device("cpu"), distributed=False)
        DeepSpeedStepPolicy().guard_loss(
            torch.tensor(float("nan")), name="g", global_step=1, scaler=None, monitor=monitor
        )
        monitor.end_step(1)
        assert monitor.resolve() is None

    def test_clip_gradients_is_a_no_op(self) -> None:
        """Clipping is declared in the ds_config and applied by the engine;
        doing it here too would clip already-clipped gradients."""
        model = nn.Linear(2, 2)
        model.weight.grad = torch.full_like(model.weight, 100.0)
        DeepSpeedStepPolicy().clip_gradients(model, 1.0)
        assert model.weight.grad.abs().max().item() == pytest.approx(100.0)


class TestStrategyGuards:
    def test_deepspeed_is_registered(self) -> None:
        assert resolve_parallel_strategy("deepspeed").name == "deepspeed"

    def test_schema_vocabulary_and_registry_now_match_exactly(self) -> None:
        from typing import get_args

        from spectramr.config.schemas.base import ParallelStrategy
        from spectramr.infrastructure.distributed.strategy_registry import (
            list_parallel_strategies,
        )

        assert set(get_args(ParallelStrategy)) == set(list_parallel_strategies())

    def test_without_a_process_group_it_raises(self) -> None:
        plugin = resolve_parallel_strategy("deepspeed")
        ctx = SimpleNamespace(
            config=_settings(),
            device=torch.device("cpu"),
            parallel=_settings().parallel,
        )
        with pytest.raises(RuntimeError, match="process group"):
            plugin.adopt({"generator": nn.Linear(2, 2)}, {"opt_g": object()}, {}, ctx)

    def test_two_optimizers_raise_rather_than_deadlocking(self, monkeypatch) -> None:
        """engine.step() issues collectives, so a GAN whose discriminator steps
        on a different cadence HANGS rather than erroring -- far more expensive
        to diagnose on a cluster than a refusal here."""
        import spectramr.infrastructure.distributed.strategies as strategies_mod

        monkeypatch.setattr(
            strategies_mod, "_require_process_group", lambda _name: None
        )
        plugin = resolve_parallel_strategy("deepspeed")
        settings = _settings()
        ctx = SimpleNamespace(
            config=settings, device=torch.device("cpu"), parallel=settings.parallel
        )
        with pytest.raises(RuntimeError, match="allow_multi_engine"):
            plugin.adopt(
                {"generator": nn.Linear(2, 2), "discriminator": nn.Linear(2, 2)},
                {"opt_g": object(), "opt_d": object()},
                {},
                ctx,
            )


class TestAvailability:
    def test_reports_the_real_environment(self) -> None:
        import importlib.util

        assert deepspeed_available() is (
            importlib.util.find_spec("deepspeed") is not None
        )

    def test_the_audit_agrees_with_availability(self) -> None:
        from spectramr.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker,
        )

        checker = ConfigHealthChecker.__new__(ConfigHealthChecker)
        result = checker.check_deepspeed_extra_installed(_settings())
        assert result.passed is deepspeed_available()


@pytest.mark.skipif(not deepspeed_available(), reason="needs the [deepspeed] extra")
class TestAgainstTheRealPackage:
    """Proves the generated dict is one DeepSpeed actually accepts.

    The fake-engine tests above prove the WIRING; only this proves the ds_config
    is well-formed. Config validation needs no process group and no GPU.
    """

    def test_the_generated_config_passes_deepspeeds_own_validation(self) -> None:
        from deepspeed.runtime.config import DeepSpeedConfig

        cfg = build_deepspeed_config(
            _settings(zero_stage=2, use_amp=True, amp_dtype="bfloat16")
        )
        parsed = DeepSpeedConfig(cfg)
        assert parsed.train_micro_batch_size_per_gpu == 4
        assert parsed.gradient_accumulation_steps == 8

    @pytest.mark.parametrize("stage", [0, 1, 2, 3])
    def test_every_zero_stage_parses(self, stage: int) -> None:
        from deepspeed.runtime.config import DeepSpeedConfig

        DeepSpeedConfig(build_deepspeed_config(_settings(zero_stage=stage)))

    def test_gradient_clipping_reaches_the_parsed_config(self) -> None:
        from deepspeed.runtime.config import DeepSpeedConfig

        parsed = DeepSpeedConfig(build_deepspeed_config(_settings(clip=0.25)))
        assert parsed.gradient_clipping == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# DeepCompile / ZenFlow / ZeRO-3
# ---------------------------------------------------------------------------


class TestZenFlowRendering:
    """ZenFlow lives under ``zero_optimization.zenflow`` and is None-or-absent.

    DeepSpeed's ``configure_zenflow`` branches on ``zenflow_config == None``,
    and ``ZenFlowConfig`` has NO ``enabled`` field -- so emitting an empty dict
    for a disabled block constructs a full-default ZenFlowConfig and turns
    ZenFlow **on**. Absent vs present is the entire switch.
    """

    def test_disabled_emits_no_key_at_all(self) -> None:
        cfg = build_deepspeed_config(_settings(zero_stage=3))
        assert "zenflow" not in cfg["zero_optimization"]

    def test_enabled_renders_under_zero_optimization_not_top_level(self) -> None:
        cfg = build_deepspeed_config(
            _settings(
                zero_stage=3,
                accum=4,
                offload_optimizer="cpu",
                zenflow={
                    "enabled": True,
                    "select_strategy": "step",
                    "select_interval": 16,
                    "update_interval": 4,
                },
            )
        )
        assert "zenflow" not in cfg
        assert cfg["zero_optimization"]["zenflow"]["update_interval"] == 4

    def test_the_enabled_flag_is_not_forwarded(self) -> None:
        """``enabled`` is ours; ZenFlowConfig has no such field and forbids extras."""
        cfg = build_deepspeed_config(
            _settings(
                zero_stage=3,
                accum=4,
                offload_optimizer="cpu",
                zenflow={
                    "enabled": True,
                    "select_strategy": "step",
                    "select_interval": 16,
                    "update_interval": 4,
                },
            )
        )
        assert "enabled" not in cfg["zero_optimization"]["zenflow"]

    def test_update_interval_overrides_declared_accumulation(self) -> None:
        """configure_zenflow's last statement is
        ``engine._config.gradient_accumulation_steps = engine.update_interval``.
        Render what the engine will USE, so the ds_config on disk is not a lie."""
        cfg = build_deepspeed_config(
            _settings(
                zero_stage=3,
                accum=4,
                offload_optimizer="cpu",
                zenflow={
                    "enabled": True,
                    "select_strategy": "step",
                    "select_interval": 16,
                    "update_interval": 4,
                },
            )
        )
        assert cfg["gradient_accumulation_steps"] == 4


class TestDeepCompileRendering:
    def test_disabled_emits_no_compile_key(self) -> None:
        assert "compile" not in build_deepspeed_config(_settings(zero_stage=3))

    def test_enabled_sets_the_deepcompile_flag(self) -> None:
        cfg = build_deepspeed_config(
            _settings(zero_stage=3, compile={"enabled": True, "passes": ["z3"]})
        )
        assert cfg["compile"]["deepcompile"] is True
        assert cfg["compile"]["passes"] == ["z3"]

    def test_passes_are_omitted_when_not_declared(self) -> None:
        """DeepSpeed's default is None; an empty list is not the same thing."""
        cfg = build_deepspeed_config(
            _settings(zero_stage=3, compile={"enabled": True})
        )
        assert "passes" not in cfg["compile"]

    def test_our_enabled_flag_is_translated_not_forwarded(self) -> None:
        cfg = build_deepspeed_config(
            _settings(zero_stage=3, compile={"enabled": True})
        )
        assert "enabled" not in cfg["compile"]


class TestRealDeepSpeedParserAcceptsIt:
    """The only check that the rendered dict is VALID rather than merely shaped.

    A golden-dict test proves the generator is self-consistent; it cannot catch
    a key DeepSpeed renamed or a nesting level we guessed wrong.
    """

    def test_zero3_deepcompile_zenflow_parses(self) -> None:
        pytest.importorskip("deepspeed")
        from deepspeed.runtime.config import DeepSpeedConfig

        cfg = build_deepspeed_config(
            _settings(
                zero_stage=3,
                accum=4,
                offload_optimizer="cpu",
                compile={"enabled": True, "passes": ["z3"]},
                zenflow={
                    "enabled": True,
                    "select_strategy": "step",
                    "select_interval": 16,
                    "update_interval": 4,
                },
            )
        )
        parsed = DeepSpeedConfig(cfg)
        assert parsed.zero_config.stage == 3
        assert parsed.compile_config.deepcompile is True
        assert parsed.compile_config.passes == ["z3"]
        assert parsed.zero_config.zenflow is not None
        assert parsed.zero_config.zenflow.update_interval == 4

    def test_a_disabled_block_leaves_zenflow_none(self) -> None:
        """The None-vs-{} trap, asserted against the real parser."""
        pytest.importorskip("deepspeed")
        from deepspeed.runtime.config import DeepSpeedConfig

        parsed = DeepSpeedConfig(build_deepspeed_config(_settings(zero_stage=3)))
        assert parsed.zero_config.zenflow is None
        assert parsed.compile_config.deepcompile is False


class TestEngineCompileIsActuallyCalled:
    """Rendering ``deepcompile: true`` is HALF the wiring.

    Without ``engine.compile()`` DeepSpeed emits one ``log_dist_once`` line on
    rank 0 -- "DeepCompile is enabled but engine.compile() has not been called"
    -- and runs eagerly. On a cluster that is indistinguishable from silence, so
    the knob would be declared, accepted, stamped into provenance, and inert.
    """

    class _FakeEngine:
        def __init__(self) -> None:
            self.compile_calls = 0

        def compile(self) -> None:
            self.compile_calls += 1

    def _initialize(self, monkeypatch, config, engine=None):
        import spectramr.infrastructure.distributed.deepspeed_backend.runtime as rt

        engine = engine or self._FakeEngine()
        fake_ds = SimpleNamespace(
            initialize=lambda **kw: (engine, "opt", None, "sched")
        )
        monkeypatch.setattr(rt, "require_deepspeed", lambda: fake_ds)
        rt.initialize_deepspeed_engine(
            model=None, optimizer=None, lr_scheduler=None, config=config
        )
        return engine

    def test_compile_is_called_when_deepcompile_is_on(self, monkeypatch) -> None:
        engine = self._initialize(monkeypatch, {"compile": {"deepcompile": True}})
        assert engine.compile_calls == 1

    def test_compile_is_not_called_otherwise(self, monkeypatch) -> None:
        engine = self._initialize(monkeypatch, {"compile": {"deepcompile": False}})
        assert engine.compile_calls == 0

    def test_compile_is_not_called_without_a_compile_block(self, monkeypatch) -> None:
        engine = self._initialize(monkeypatch, {})
        assert engine.compile_calls == 0

    def test_an_engine_without_compile_raises_rather_than_degrading(
        self, monkeypatch
    ) -> None:
        """Silently skipping would leave the run reporting throughput numbers
        that belong to a different configuration."""

        class _OldEngine:
            pass

        with pytest.raises(RuntimeError, match="no compile"):
            self._initialize(
                monkeypatch, {"compile": {"deepcompile": True}}, engine=_OldEngine()
            )


class TestDeepCompileGpuFamilyGate:
    """Planted violations (LM-16): DeepCompile lowers through Inductor, so a P100
    rank must be refused before ``deepspeed.initialize`` rather than at step 1."""

    def _initialize(self, monkeypatch, config, capability):
        import torch

        import spectramr.core.device_capabilities as dc
        import spectramr.infrastructure.distributed.deepspeed_backend.runtime as rt

        calls = []
        fake_ds = SimpleNamespace(
            initialize=lambda **kw: calls.append(kw)
            or (TestEngineCompileIsActuallyCalled._FakeEngine(), "opt", None, "sched")
        )
        monkeypatch.setattr(rt, "require_deepspeed", lambda: fake_ds)
        monkeypatch.setattr(rt, "_deepcompile_device", lambda model: torch.device("cuda", 0))
        monkeypatch.setattr(dc, "capability_of_device", lambda device: capability)
        import spectramr.core.compile_capability as cc

        monkeypatch.setattr(cc, "triton_available", lambda: True)
        rt.initialize_deepspeed_engine(
            model=None, optimizer=None, lr_scheduler=None, config=config
        )
        return calls

    def test_a_p100_rank_is_refused_before_initialize(self, monkeypatch):
        from spectramr.core.compile_capability import CompileUnsupportedError

        calls = []
        with pytest.raises(CompileUnsupportedError, match=r"6\.0"):
            calls = self._initialize(monkeypatch, {"compile": {"deepcompile": True}}, (6, 0))
        assert calls == []

    def test_an_unreadable_capability_is_refused(self, monkeypatch):
        from spectramr.core.compile_capability import CompileUnsupportedError

        with pytest.raises(CompileUnsupportedError, match="could not be read"):
            self._initialize(monkeypatch, {"compile": {"deepcompile": True}}, None)

    @pytest.mark.parametrize("capability", [(7, 0), (8, 9)])
    def test_volta_and_ada_ranks_initialize(self, monkeypatch, capability):
        calls = self._initialize(monkeypatch, {"compile": {"deepcompile": True}}, capability)
        assert len(calls) == 1

    def test_a_p100_without_deepcompile_is_not_refused(self, monkeypatch):
        """The gate guards compilation, not DeepSpeed itself."""
        calls = self._initialize(monkeypatch, {"compile": {"deepcompile": False}}, (6, 0))
        assert len(calls) == 1


class TestEveryRenderedConfigSurvivesTheRealParser:
    """Regression net for the whole ZenFlow/DeepCompile surface.

    Written after the generator emitted ``gradient_accumulation_steps: 0`` on
    the ZenFlow ``update_interval: 'auto'`` path -- which is what
    ``configure_zenflow`` sets POST-parse, but which DeepSpeed's own parser
    rejects outright ("Train batch size: 0 has to be greater than 0"). Every
    key-level test passed; only feeding the dict to DeepSpeed caught it.
    """

    @pytest.mark.parametrize(
        ("label", "ds"),
        [
            ("zero3 bare", {"zero_stage": 3}),
            ("zero0", {"zero_stage": 0}),
            (
                "zero3 + cpu offload",
                {"zero_stage": 3, "offload_optimizer": "cpu", "offload_param": "cpu"},
            ),
            (
                "deepcompile z3",
                {"zero_stage": 3, "compile": {"enabled": True, "passes": ["z3"]}},
            ),
            (
                "deepcompile z1 @ stage2",
                {"zero_stage": 2, "compile": {"enabled": True, "passes": ["z1"]}},
            ),
            (
                "deepcompile no passes",
                {"zero_stage": 3, "compile": {"enabled": True}},
            ),
            (
                "zenflow auto interval",
                {
                    "zero_stage": 3,
                    "offload_optimizer": "cpu",
                    "zenflow": {"enabled": True},
                },
            ),
            (
                "zenflow explicit interval",
                {
                    "zero_stage": 3,
                    "offload_optimizer": "cpu",
                    "zenflow": {
                        "enabled": True,
                        "select_strategy": "step",
                        "select_interval": 32,
                        "update_interval": 4,
                    },
                },
            ),
            (
                "the full stack",
                {
                    "zero_stage": 3,
                    "offload_optimizer": "cpu",
                    "compile": {"enabled": True, "passes": ["z3"]},
                    "zenflow": {
                        "enabled": True,
                        "select_strategy": "step",
                        "select_interval": 32,
                        "update_interval": 4,
                        "overlap_step": True,
                        "offload": True,
                    },
                },
            ),
        ],
    )
    def test_deepspeed_accepts_it(self, label: str, ds: dict) -> None:
        pytest.importorskip("deepspeed")
        from deepspeed.runtime.config import DeepSpeedConfig

        # accum must match update_interval where one is declared, mirroring what
        # check_zenflow_accumulation_conflict enforces on real arms.
        accum = (ds.get("zenflow") or {}).get("update_interval")
        accum = accum if isinstance(accum, int) else 1
        cfg = build_deepspeed_config(_settings(accum=accum, **ds))
        DeepSpeedConfig(cfg)  # raises on anything malformed

    def test_auto_update_interval_never_renders_a_zero(self) -> None:
        """The specific bug: 0 is what the ENGINE sets post-parse, and what the
        PARSER refuses. Rendering it turned a working arm into a hard crash."""
        cfg = build_deepspeed_config(
            _settings(
                accum=1,
                zero_stage=3,
                offload_optimizer="cpu",
                zenflow={"enabled": True},
            )
        )
        assert cfg["gradient_accumulation_steps"] >= 1


class TestImportFailureIsActionable:
    """The failure message must name the path the OS refused.

    A 4-rank cluster launch on 2026-08-16 reported, four times::

        Training pipeline build failed: parallel.strategy='deepspeed' requires
        the [deepspeed] extra. Install with: pip install -e ".[deepspeed]"
        (import failed: PermissionError(13, 'Permission denied'))

    Both halves were unhelpful. The extra *was* installed, so the advice pointed
    at the one thing that was not wrong; and ``repr()`` on an ``OSError`` drops
    ``filename``, discarding the only fact that identifies the fix.
    """

    def test_repr_really_does_drop_the_path(self) -> None:
        """Anti-vacuity: pin the stdlib behaviour the fix exists to work around."""
        exc = PermissionError(13, "Permission denied", "/home/u/.triton/autotune")
        assert "/home/u/.triton/autotune" not in repr(exc)
        assert "/home/u/.triton/autotune" in str(exc)

    def test_the_description_keeps_the_path(self) -> None:
        from spectramr.infrastructure.distributed.deepspeed_backend.runtime import (
            _describe_import_failure,
        )

        described = _describe_import_failure(
            PermissionError(13, "Permission denied", "/home/u/.triton/autotune")
        )
        assert "/home/u/.triton/autotune" in described
        assert "PermissionError" in described

    def test_a_filename_survives_even_without_a_message(self) -> None:
        """An OSError carrying only a filename still stringifies without it."""
        from spectramr.infrastructure.distributed.deepspeed_backend.runtime import (
            _describe_import_failure,
        )

        exc = OSError()
        exc.filename = "/scratch/torch_extensions/lock"
        assert "/scratch/torch_extensions/lock" in _describe_import_failure(exc)

    def test_a_plain_exception_still_renders(self) -> None:
        from spectramr.infrastructure.distributed.deepspeed_backend.runtime import (
            _describe_import_failure,
        )

        described = _describe_import_failure(RuntimeError("CUDA version mismatch"))
        assert described == "RuntimeError: CUDA version mismatch"

    def test_an_installed_but_broken_deepspeed_is_not_told_to_reinstall(
        self, monkeypatch
    ) -> None:
        """The advice must follow the diagnosis, not the strategy name."""
        from spectramr.infrastructure.distributed.deepspeed_backend import runtime

        monkeypatch.setattr(runtime, "deepspeed_available", lambda: True)
        monkeypatch.setitem(
            __import__("sys").modules, "deepspeed", None
        )  # forces `import deepspeed` to raise ImportError

        with pytest.raises(ImportError) as excinfo:
            runtime.require_deepspeed()

        message = str(excinfo.value)
        assert "IS installed" in message, message
        assert 'pip install -e ".[deepspeed]"' not in message, (
            "an operator whose deepspeed is installed was sent to reinstall it"
        )

    def test_a_genuinely_missing_deepspeed_still_gets_the_install_hint(
        self, monkeypatch
    ) -> None:
        """Anti-vacuity for the test above: the original message must survive."""
        from spectramr.infrastructure.distributed.deepspeed_backend import runtime

        monkeypatch.setattr(runtime, "deepspeed_available", lambda: False)
        monkeypatch.setitem(__import__("sys").modules, "deepspeed", None)

        with pytest.raises(ImportError) as excinfo:
            runtime.require_deepspeed()

        assert 'pip install -e ".[deepspeed]"' in str(excinfo.value)
