"""Regression tests for ModelBuilder.build_discriminator.

F31 (2026-05-22) — the discriminator branch guarded on
``hasattr(self._config.training, "gan")``, which is ALWAYS True because the
training schema declares ``gan: GANSubConfigSchema | None = Field(default=None)``
(``config/schemas/training/base.py``). For every non-GAN paradigm
``training.gan`` is None, so the next clause dereferenced
``None.discriminator`` and the whole training pipeline died with
``'NoneType' object has no attribute 'discriminator'`` — ~60 experiments in
smoke 20260521 (every dummy_*, baseline_*, geomamba_*, exp_hm_*, abl_*,
diffusion/vae/mae/recon arm). The fix replaces the no-op ``hasattr`` with an
explicit ``is not None`` check.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.nn as _nn

from spectramr.models.registry import register_model as _register_model

torch = pytest.importorskip("torch")  # noqa: E402

from spectramr.infrastructure.training.builders.model_builder import (  # noqa: E402
    ModelBuilder,
)
from tests.fixtures.run_state_files import write_run_state  # noqa: E402


def _builder_with_config(config: SimpleNamespace) -> ModelBuilder:
    """Construct a ModelBuilder without running its heavy __init__."""
    mb = ModelBuilder.__new__(ModelBuilder)
    mb._config = config
    mb._models = {}
    return mb


def test_build_discriminator_skips_when_training_gan_is_none() -> None:
    """Non-GAN paradigm: discriminator_component=None, discriminator=None,
    training.gan=None. Must NOT raise and must create no discriminator.
    """
    config = SimpleNamespace(
        model=SimpleNamespace(
            discriminator_component=None, discriminator=None, out_channels=2
        ),
        training=SimpleNamespace(gan=None),
    )
    builder = _builder_with_config(config)

    # Pre-fix this raised AttributeError: 'NoneType' object has no attribute
    # 'discriminator'.
    result = builder.build_discriminator()

    assert result is builder
    assert "discriminator" not in builder._models


def test_build_ema_threads_warmup_flag() -> None:
    """config.ema.warmup must reach ModelEma (pitfall #15).

    Pre-fix ``build_ema`` constructed ``ModelEma(generator, decay=...)`` without
    forwarding ``warmup``, so it defaulted to True regardless of config — the
    cause of the byte-identical EMA-warmup on/off Experiment-11 ablation
    (validation graded the same shadow in both arms).
    """
    gen = torch.nn.Conv2d(2, 2, 3, padding=1)
    for warmup_flag in (True, False):
        config = SimpleNamespace(
            ema=SimpleNamespace(enabled=True, decay=0.99, warmup=warmup_flag)
        )
        builder = _builder_with_config(config)
        builder._device = torch.device("cpu")
        builder._ema = None
        builder._models["generator"] = gen

        result = builder.build_ema()

        assert result is builder
        assert builder._ema is not None
        assert builder._ema.warmup is warmup_flag
        assert builder._ema.decay == pytest.approx(0.99)


def test_build_ema_skipped_when_disabled() -> None:
    """ema.enabled=False -> no shadow built (no warmup access required)."""
    config = SimpleNamespace(
        ema=SimpleNamespace(enabled=False, decay=0.99, warmup=True)
    )
    builder = _builder_with_config(config)
    builder._device = torch.device("cpu")
    builder._ema = None
    builder._models["generator"] = torch.nn.Conv2d(2, 2, 3, padding=1)

    builder.build_ema()

    assert builder._ema is None


def _ema_builder(**ema_kwargs) -> ModelBuilder:
    defaults = dict(enabled=True, decay=0.99, warmup=True)
    defaults.update(ema_kwargs)
    builder = _builder_with_config(SimpleNamespace(ema=SimpleNamespace(**defaults)))
    builder._device = torch.device("cpu")
    builder._ema = None
    builder._models["generator"] = torch.nn.Conv2d(2, 2, 3, padding=1)
    return builder


def test_build_ema_threads_the_adaptive_schedule() -> None:
    """The four deterministic adaptive knobs must reach ModelEma (#1294).

    ``enable_adaptive_ema`` / ``warmup_steps`` / ``initial_decay`` /
    ``final_decay`` were schema-only from ff0efff9f (which deleted
    ``models/utils/adaptive_ema.py``) until this wiring: an arm could declare
    the whole family and silently get plain fixed-decay EMA.
    """
    builder = _ema_builder(
        enable_adaptive_ema=True,
        warmup_steps=500,
        initial_decay=0.8,
        final_decay=0.9995,
    )

    builder.build_ema()

    ema = builder._ema
    assert ema is not None
    assert ema.adaptive is True
    assert ema.warmup_steps == 500
    assert ema.initial_decay == pytest.approx(0.8)
    assert ema.final_decay == pytest.approx(0.9995)


def test_adaptive_schedule_supersedes_decay_and_warmup() -> None:
    """decay/warmup are still forwarded but must not drive the schedule."""
    builder = _ema_builder(
        decay=0.5,
        warmup=True,
        enable_adaptive_ema=True,
        warmup_steps=100,
        initial_decay=0.0,
        final_decay=0.9,
    )

    builder.build_ema()

    builder._ema.num_updates = 50
    assert builder._ema._current_decay() == pytest.approx(0.45)


def test_build_ema_logs_which_schedule_ran(caplog) -> None:
    """A run log that did not name the schedule would leave a reader unable to
    tell an adaptive arm from a fixed-decay one."""
    builder = _ema_builder(
        enable_adaptive_ema=True, warmup_steps=100, initial_decay=0.1, final_decay=0.9
    )
    with caplog.at_level("INFO"):
        builder.build_ema()
    # caplog .message is order-dependent under wide runs; getMessage() is not.
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "ADAPTIVE" in text
    assert "supersedes" in text


def test_unhonourable_ema_declaration_raises_instead_of_disabling_ema() -> None:
    """A zero-length adaptive ramp must not be swallowed into ``_ema = None``.

    The blanket ``except Exception`` here used to turn any construction error
    into a WARNING plus a silently EMA-less run, while validation went on
    grading live weights (non-negotiable 3).
    """
    builder = _ema_builder(enable_adaptive_ema=True, warmup_steps=0)

    with pytest.raises(ValueError, match="warmup_steps"):
        builder.build_ema()


def test_build_discriminator_skips_when_gan_present_but_no_discriminator() -> None:
    """training.gan exists but carries no discriminator → still a no-op."""
    config = SimpleNamespace(
        model=SimpleNamespace(
            discriminator_component=None, discriminator=None, out_channels=2
        ),
        training=SimpleNamespace(gan=SimpleNamespace(discriminator=None)),
    )
    builder = _builder_with_config(config)

    result = builder.build_discriminator()

    assert result is builder
    assert "discriminator" not in builder._models


# ── model.checkpoint_path init seam (sequential-campaign warm-start, option a) ──


def _ckpt_builder(checkpoint_path) -> ModelBuilder:
    config = SimpleNamespace(model=SimpleNamespace(checkpoint_path=checkpoint_path))
    builder = _builder_with_config(config)
    builder._device = torch.device("cpu")
    return builder


def test_warm_start_transfers_weights(tmp_path) -> None:
    """``model.checkpoint_path`` must load another model's weights INTO the freshly
    built generator (transfer-init / conditioning warm-start). This is the
    consumer that makes the sequential-campaign injection real, not just a logged
    no-op."""
    torch.manual_seed(1)
    src = torch.nn.Conv2d(2, 2, 3, padding=1)
    ckpt = write_run_state(tmp_path / "src.safetensors", {"generator": src.state_dict()})

    torch.manual_seed(2)
    tgt = torch.nn.Conv2d(2, 2, 3, padding=1)
    assert not torch.allclose(tgt.weight, src.weight)

    _ckpt_builder(str(ckpt))._warm_start(tgt)
    assert torch.allclose(tgt.weight, src.weight)


def test_warm_start_noop_when_unset() -> None:
    tgt = torch.nn.Conv2d(2, 2, 3, padding=1)
    before = tgt.weight.detach().clone()
    _ckpt_builder(None)._warm_start(tgt)
    assert torch.allclose(tgt.weight, before)


def test_warm_start_raises_on_zero_overlap(tmp_path) -> None:
    """A checkpoint with no overlapping keys must RAISE — a silently-dropped
    warm-start (strict=False swallowing everything) is the Method-C facade bug
    (CLAUDE.md pitfall #16)."""
    ckpt = write_run_state(
        tmp_path / "alien.safetensors", {"generator": {"totally.unrelated.key": torch.zeros(3)}}
    )
    tgt = torch.nn.Conv2d(2, 2, 3, padding=1)
    with pytest.raises(ValueError, match="zero parameter names"):
        _ckpt_builder(str(ckpt))._warm_start(tgt)


def test_warm_start_refuses_a_foreign_state_dict_as_foreign(tmp_path) -> None:
    """A bare ``torch.save`` state dict was not written by a run: it is refused as
    foreign, never guessed at, and the converter (which cannot apply) is not named."""
    ckpt = tmp_path / "pretrained.pt"
    torch.save(torch.nn.Conv2d(2, 2, 3, padding=1).state_dict(), ckpt)
    with pytest.raises(ValueError, match="not a spectramr RunState"):
        _ckpt_builder(str(ckpt))._warm_start(torch.nn.Conv2d(2, 2, 3, padding=1))


class _AutoencoderHost(torch.nn.Module):
    """Stands in for the LDM: the builder hands it the stage-1 generator part."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def load_autoencoder_state(self, state, *, freeze, source):
        self.calls.append({"keys": sorted(state), "freeze": freeze, "source": source})


def _vae_builder(vae_path, **extra) -> ModelBuilder:
    model = SimpleNamespace(
        checkpoint_path=None, model_kwargs={"vae_checkpoint_path": vae_path, **extra}
    )
    builder = _builder_with_config(SimpleNamespace(model=model))
    builder._device = torch.device("cpu")
    return builder


def test_warm_start_hands_the_stage1_generator_part_to_the_autoencoder(tmp_path) -> None:
    stage1 = torch.nn.Linear(2, 2)
    path = write_run_state(
        tmp_path / "best.safetensors",
        {"generator": {f"module.{k}": v for k, v in stage1.state_dict().items()}},
    )
    host = _AutoencoderHost()
    _vae_builder(str(path), freeze_vae=False)._warm_start(host)
    assert host.calls == [{"keys": ["bias", "weight"], "freeze": False, "source": str(path)}]


def test_warm_start_refuses_a_vae_path_the_generator_cannot_take(tmp_path) -> None:
    path = write_run_state(tmp_path / "best.pt", {"generator": torch.nn.Linear(2, 2).state_dict()})
    with pytest.raises(ValueError, match="loads no stage-1 autoencoder"):
        _vae_builder(str(path))._warm_start(torch.nn.Linear(2, 2))


def test_a_missing_stage1_vae_keeps_the_layout_hint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match=r"best\.<format>"):
        _vae_builder(str(tmp_path / "checkpoint_best.pt"))._warm_start(_AutoencoderHost())


def test_warm_start_is_off_unless_the_builder_is_told() -> None:
    """infer/make/export/campaign evaluation build the same model and must not
    warm-start from a predecessor before loading their own checkpoint."""
    import inspect

    assert inspect.signature(ModelBuilder.__init__).parameters["warm_start"].default is False
    src = inspect.getsource(ModelBuilder.build_generator)
    assert "if self._warm_start_enabled:" in src
    assert "self._warm_start(generator)" in src
    # Before train(), so a frozen stage-1 autoencoder ends in train mode as it did when
    # the constructor loaded it.
    assert src.index("self._warm_start(generator)") < src.index("generator.train()")


def test_the_director_is_the_one_warm_starting_caller() -> None:
    """The director always warm-starts. The second caller, ``pipelines/make.py``, does
    so only when it is given no checkpoint of its own (pinned in test_make.py)."""
    import inspect
    import re

    from spectramr.infrastructure.training.builders import director

    src = inspect.getsource(director)
    assert "ModelBuilder(self._config, self._device, warm_start=True)" in src
    pkg = Path(director.__file__).resolve().parents[3]
    others = [
        p for p in pkg.rglob("*.py")
        if p.name != "director.py" and re.search(r"ModelBuilder\([^)]*warm_start=True", p.read_text())
    ]
    assert others == []


# ---------------------------------------------------------------------------
# ModelBuilder no longer compiles.
#
# `compile()` was deleted, not deprecated. Compilation is placed per parallel
# strategy now (`builders/compile_placement.py`) and applied by the director at
# the point that table names -- step 1 was the wrong point for every strategy
# but DeepSpeed. The behaviour the old tests pinned, failure RAISES and never
# degrades to eager (#619 F2), moved with the code to `test_compile_apply.py`.
# ---------------------------------------------------------------------------


def test_model_builder_no_longer_exposes_compile() -> None:
    """Guards against a second compile owner reappearing here.

    Two call sites for one decision is how the placement came to be fixed at
    step 1 in the first place (non-negotiable 17).
    """
    assert not hasattr(ModelBuilder, "compile")




class TestFsdpIsNotWrappedTwice:
    """FSDP wrapping belongs to the parallel plugin (Stage A), not here.

    ``director.build()`` calls ``ModelBuilder.build()`` BEFORE the Stage-A hook,
    so wrapping in both places gave every ``strategy: fsdp`` arm an
    ``FSDP(FSDP(model))``: sharding and activation checkpointing applied twice,
    with the outer root managing zero parameters. ``maybe_wrap_with_fsdp`` has no
    already-wrapped guard, so nothing detected it at runtime.
    """

    @staticmethod
    def _build_ast():
        """Parse ``ModelBuilder.build`` rather than substring-match it.

        The first version of these tests asserted ``"maybe_wrap_with_fsdp" not
        in source`` and failed against the COMMENT explaining why the call was
        removed. Source-grep matches prose; AST matches code.
        """
        import ast
        import inspect
        import textwrap

        from spectramr.infrastructure.training.builders import model_builder

        return ast.parse(
            textwrap.dedent(inspect.getsource(model_builder.ModelBuilder.build))
        )

    def test_model_builder_no_longer_calls_fsdp_wrap(self) -> None:
        import ast

        called = {
            (n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", ""))
            for n in ast.walk(self._build_ast())
            if isinstance(n, ast.Call)
        }
        assert "maybe_wrap_with_fsdp" not in called, (
            "ModelBuilder.build() wraps with FSDP again; the plugin already "
            "does it at Stage A, so this double-wraps every fsdp arm"
        )

    def test_peft_injection_is_retained(self) -> None:
        """PEFT is NOT duplicated by the plugin, so removing the FSDP branch
        must not take PEFT with it.

        AST, not substring: a positive assertion has the mirror-image flaw of
        the negative ones -- it would pass on a COMMENT mentioning inject_peft
        long after the call itself was deleted.
        """
        import ast

        called = {
            (n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", ""))
            for n in ast.walk(self._build_ast())
            if isinstance(n, ast.Call)
        }
        assert "inject_peft" in called

    def test_peft_import_failure_is_not_swallowed(self) -> None:
        """The old block caught ImportError and continued with a warning, so a
        broken install trained every parameter instead of the adapters and
        reported success (CLAUDE.md #9).

        Asserted over the AST: the previous substring form matched the COMMENT
        that documents the removal, so it failed against correct code.
        """
        import ast

        caught = set()
        for node in ast.walk(self._build_ast()):
            if not isinstance(node, ast.ExceptHandler) or node.type is None:
                continue
            for name in ast.walk(node.type):
                if isinstance(name, ast.Name):
                    caught.add(name.id)
        assert "ImportError" not in caught, (
            "ModelBuilder.build() swallows an ImportError again; a missing PEFT "
            "dependency must be loud, not a full-finetune reporting success"
        )


class TestKSpaceLogScaledIsInjectedFromTheDataBlock:
    """``data.processing.enable_log_scaling`` must reach the cold-diffusion
    generator, because the magnitude ceilings enforce a PHYSICAL ratio and
    cannot be built without knowing the domain of the k-space they bound.

    Without this wiring ``experiment_11_attention_none``'s declared
    ``reverse_clip_ratio: 1.3`` realised **29.8x** on real M4Raw data (#1281).
    The knob is read from the EXISTING data block rather than a new YAML key,
    so there is exactly one source of truth and no per-arm migration.
    """

    def test_the_generator_declares_the_kwarg_so_the_contract_can_see_it(self) -> None:
        """``_get_contract`` introspects ``__init__``; a kwarg consumed only via
        ``**kwargs`` would be invisible to it and silently never injected."""
        import inspect

        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            KSpaceColdDiffusionGenerator,
        )

        params = inspect.signature(KSpaceColdDiffusionGenerator.__init__).parameters
        assert "kspace_log_scaled" in params, (
            "KSpaceColdDiffusionGenerator must DECLARE kspace_log_scaled; "
            "ModelBuilder's signature contract cannot inject what it cannot see"
        )

    @staticmethod
    def _config(enable_log_scaling: bool, model_kwargs: dict | None = None):
        """A minimal real ``TrainingSettings`` -- not a stub.

        The injection is contract-gated against the registered generator, so a
        duck-typed config would exercise a different path than training does.
        """
        from spectramr.config.settings import TrainingSettings

        return TrainingSettings(
            model={
                "model_type": "_log_scale_probe_model",
                "in_channels": 1,
                "out_channels": 1,
                "model_kwargs": dict(model_kwargs or {}),
            },
            data={
                "sampling": {"patch_size": [8, 8]},
                "loader": {"batch_size": 1},
                "processing": {
                    "enable_kspace_normalization": enable_log_scaling,
                    "enable_log_scaling": enable_log_scaling,
                },
            },
            optimization={},
            logging={},
        )

    def test_the_builder_reads_the_data_block_knob(self) -> None:
        """End-to-end: the value reaches the constructed model.

        Asserted through a real build rather than over ``build_generator``'s
        AST. The AST form pinned the *location* of the read, so extracting the
        resolution to its SSOT broke it with no behavioural change -- while a
        build that silently stopped injecting would still have passed as long
        as the attribute name appeared somewhere in the method.
        """
        import torch

        from spectramr.infrastructure.training.builders.model_builder import ModelBuilder

        models = (
            ModelBuilder(self._config(True), torch.device("cpu"))
            .build_generator()
            .build()
        )
        assert models["generator"].seen_kspace_log_scaled is True, (
            "ModelBuilder no longer pipes data.processing.enable_log_scaling "
            "to the generator; the cold-diffusion magnitude ceiling would fall "
            "back to guessing the domain"
        )

    def test_the_data_block_value_is_the_one_that_lands(self) -> None:
        """Negative control: a different knob value produces a different model.

        Without this, an injection hardcoded to ``True`` would pass the test
        above.
        """
        import torch

        from spectramr.infrastructure.training.builders.model_builder import ModelBuilder

        models = (
            ModelBuilder(self._config(False), torch.device("cpu"))
            .build_generator()
            .build()
        )
        assert models["generator"].seen_kspace_log_scaled is False

    def test_a_disagreeing_model_kwargs_copy_is_refused(self) -> None:
        """Two declarations of the same fact must not diverge silently -- the
        same _reconcile discipline the DC block already uses."""
        import pytest
        import torch

        from spectramr.infrastructure.training.builders.model_builder import ModelBuilder

        config = self._config(False, {"kspace_log_scaled": True})
        with pytest.raises(ValueError, match="kspace_log_scaled"):
            ModelBuilder(config, torch.device("cpu")).build_generator()



@_register_model("_log_scale_probe_model", "reconstruction")
class _LogScaleProbeModel(_nn.Module):
    """Declares ``kspace_log_scaled`` so the contract gate can inject it."""

    def __init__(self, in_channels=1, out_channels=1, kspace_log_scaled=None, **kwargs):
        super().__init__()
        self.seen_kspace_log_scaled = kspace_log_scaled
        self.conv = _nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


# ── one weight reader (checkpointing.restore) ──────────────────────────────


def test_builder_reads_through_restore_not_its_own_vocabulary() -> None:
    """A RunState has one place for the generator, and ``restore.weights`` /
    ``read_part`` are the only readers of it."""
    import spectramr.infrastructure.training.builders.model_builder as mb

    restore = "spectramr.infrastructure.checkpointing.restore"
    assert mb.weights.__module__ == restore and mb.read_part.__module__ == restore


def test_build_generator_passes_its_device_to_kwarg_resolution(monkeypatch) -> None:
    """The builder's resolved device reaches step 3d (#1508).

    ``ModelBuilder`` is the only place on the training path that holds the run's
    resolved compute device at generator-construction time. Without this
    argument the k-space undersampling process was built device-less and pinned
    its mask generator to CPU, so the device-resident mask table was
    unreachable and every ``q_sample`` paid a host sync -- a capability
    registered and never called (non-negotiable 16).

    Observed by spy rather than by outcome: the assertion is that the call
    happens with the device, not that some downstream model looks right, so it
    cannot be satisfied by a coincidence further down the chain.
    """
    import spectramr.infrastructure.training.builders.model_builder as mb_mod

    seen: dict = {}

    class _StopError(RuntimeError):
        pass

    def _spy(config, **kwargs):
        seen.update(kwargs)
        raise _StopError

    monkeypatch.setattr(mb_mod, "resolve_generator_kwargs", _spy)

    config = SimpleNamespace(model=SimpleNamespace(model_type="stub", model_kwargs={}))
    mb = _builder_with_config(config)
    mb._device = torch.device("cuda")

    # build_generator wraps every failure in ValueError; the spy's _StopError is how
    # we return without constructing a real model.
    with pytest.raises(ValueError):
        mb.build_generator()

    assert seen.get("device") == torch.device("cuda")
