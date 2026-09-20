"""The one live ``torch.compile`` call site.

These behaviours moved here from ``test_model_builder.py`` when
``ModelBuilder.compile()`` was deleted. The policy they pin is unchanged:
**failure raises, never degrades to eager.** The old body wrapped everything in
a blanket ``except Exception`` that logged a warning and continued, so a
``compile_model: true`` arm ran to completion, reported success, and had never
been compiled (#619 F1+F2).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.core.compile_capability import CompileUnsupportedError
from spectramr.infrastructure.training.builders.compile_apply import (
    apply_compile,
    compile_is_enabled,
)
from spectramr.infrastructure.training.builders.compile_regions import (
    NoRegionsToCompileError,
)


def _config(**compile_opts) -> SimpleNamespace:
    opts = {
        "enabled": True,
        "mode": "default",
        "backend": "inductor",
        "fullgraph": False,
        "dynamic": True,
        "regional": False,
        "apply_to": None,
        "recompile_limit": None,
        "fail_on_recompile_limit": True,
    }
    opts.update(compile_opts)
    return SimpleNamespace(optimization=SimpleNamespace(compile=SimpleNamespace(**opts)))


def _models() -> dict[str, torch.nn.Module]:
    return {"generator": torch.nn.Conv2d(2, 2, 3)}


class TestDisabled:
    def test_is_a_noop(self):
        """``compile.enabled: false`` is the supported way to ask for eager."""
        models = _models()
        original = models["generator"]
        result, record = apply_compile(models, _config(enabled=False), stage="after_stage_b")
        assert result["generator"] is original
        assert record == {}

    def test_a_config_without_an_optimization_block_is_tolerated(self):
        """Scripting-API callers pass partial configs."""
        assert compile_is_enabled(SimpleNamespace()) is False


class TestEnabled:
    def test_wraps_the_model(self):
        """``torch.compile`` is lazy, so the wrapper type is the observable."""
        result, _ = apply_compile(_models(), _config(), stage="after_stage_b")
        assert type(result["generator"]).__name__ == "OptimizedModule"

    def test_does_not_mutate_the_input_mapping(self):
        """The director keeps the pre-compile mapping in some paths."""
        models = _models()
        original = models["generator"]
        apply_compile(models, _config(), stage="after_stage_b")
        assert models["generator"] is original

    def test_a_none_entry_is_skipped(self):
        """``discriminator`` is ``None`` on every non-adversarial arm."""
        models = {"generator": torch.nn.Conv2d(2, 2, 3), "discriminator": None}
        result, record = apply_compile(models, _config(), stage="after_stage_b")
        assert result["discriminator"] is None
        assert "discriminator" not in record

    def test_the_record_carries_the_stage(self):
        """So a reader can tell ``DDP(compile(m))`` from ``compile(DDP(m))``
        after the fact, which is the whole point of the placement work."""
        _, record = apply_compile(_models(), _config(), stage="after_stage_a")
        assert record["generator"]["stage"] == "after_stage_a"
        assert record["generator"]["backend"] == "inductor"


class TestFailureRaises:
    """The #619 F2 contract. Do not relax any of these."""

    def test_raises_instead_of_falling_back_to_eager(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise RuntimeError("backend exploded")

        monkeypatch.setattr(torch, "compile", _boom)
        with pytest.raises(RuntimeError, match="Refusing to fall back to eager"):
            apply_compile(_models(), _config(), stage="after_stage_b")

    def test_the_error_names_the_model_and_the_settings(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise RuntimeError("backend exploded")

        monkeypatch.setattr(torch, "compile", _boom)
        with pytest.raises(RuntimeError) as excinfo:
            apply_compile(_models(), _config(mode="max-autotune"), stage="after_stage_b")
        message = str(excinfo.value)
        assert "generator" in message
        assert "max-autotune" in message
        assert "compile.enabled: false" in message, "no opt-out offered"

    def test_raises_when_torch_compile_is_unavailable(self, monkeypatch):
        monkeypatch.delattr(torch, "compile", raising=False)
        with pytest.raises(RuntimeError, match=r"torch\.compile is unavailable"):
            apply_compile(_models(), _config(), stage="after_stage_b")

    def test_an_unavailable_torch_compile_is_not_raised_when_disabled(self, monkeypatch):
        """The availability probe must not fire on an arm that asked for eager."""
        monkeypatch.delattr(torch, "compile", raising=False)
        result, _ = apply_compile(_models(), _config(enabled=False), stage="after_stage_b")
        assert result is not None


class TestTheDeviceIsCrossCheckedFirst:
    """Inductor on CUDA needs Triton to emit kernels.

    ``compile_backend_for`` has always known that and raised
    ``CompileUnsupportedError``; its only other caller catches it immediately
    and files it under ``incomplete`` for provenance, which is reporting rather
    than enforcement. Calling it here is what makes the error reachable -- the
    alternative is finding out at the first forward pass on a cluster node.
    """

    def test_cuda_without_triton_raises_before_anything_is_compiled(self, monkeypatch):
        import spectramr.infrastructure.training.builders.compile_apply as mod

        monkeypatch.setattr(mod, "triton_available", lambda: False)
        compiled = []
        monkeypatch.setattr(torch, "compile", lambda m, **k: compiled.append(m) or m)

        with pytest.raises(CompileUnsupportedError, match="Triton"):
            apply_compile(_models(), _config(), stage="after_stage_b", device="cuda")
        assert compiled == [], "the refusal must precede the first torch.compile call"

    def test_cuda_with_triton_proceeds(self, monkeypatch):
        import spectramr.infrastructure.training.builders.compile_apply as mod

        monkeypatch.setattr(mod, "triton_available", lambda: True)
        result, _ = apply_compile(
            _models(), _config(), stage="after_stage_b", device="cuda"
        )
        assert result["generator"] is not None

    def test_a_backend_that_does_not_need_triton_is_not_refused(self, monkeypatch):
        """Planted: refusing ``eager`` for want of Triton would be a false
        refusal, and a gate that reddens correct arms stops being read."""
        import spectramr.infrastructure.training.builders.compile_apply as mod

        monkeypatch.setattr(mod, "triton_available", lambda: False)
        _, record = apply_compile(
            _models(), _config(backend="eager"), stage="after_stage_b", device="cuda"
        )
        assert record["generator"]["backend"] == "eager"

    def test_a_torch_device_object_is_accepted(self, monkeypatch):
        """The director passes ``torch.device``, not a string."""
        import spectramr.infrastructure.training.builders.compile_apply as mod

        monkeypatch.setattr(mod, "triton_available", lambda: False)
        with pytest.raises(CompileUnsupportedError):
            apply_compile(
                _models(), _config(), stage="after_stage_b", device=torch.device("cuda:0")
            )


class TestTheRecordReportsWhatHappened:
    def test_the_wrap_is_observed_not_assumed(self):
        """A record built from the config alone says the same thing whether or
        not torch.compile did anything."""
        _, record = apply_compile(_models(), _config(), stage="after_stage_b")
        assert record["generator"]["wrapped"] is True

    def test_a_backend_that_returns_the_model_unchanged_is_recorded_as_unwrapped(
        self, monkeypatch
    ):
        """Planted: the point of observing rather than assuming."""
        monkeypatch.setattr(torch, "compile", lambda m, **k: m)
        _, record = apply_compile(_models(), _config(), stage="after_stage_b")
        assert record["generator"]["wrapped"] is False


class TestTheRegionalPath:
    """The branch a `regional: true` arm actually takes.

    It had no test at all: `test_compile_regions.py` exercises
    `compile_regions` directly, which is not the path a user reaches.
    """

    @staticmethod
    def _stacked() -> dict[str, torch.nn.Module]:
        class Block(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 3, padding=1)

            def forward(self, x):
                return self.conv(x)

        class Stacked(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = torch.nn.ModuleList([Block() for _ in range(3)])

        return {"generator": Stacked()}

    def test_the_blocks_are_compiled_and_the_root_is_left_bare(self):
        models = self._stacked()
        result, record = apply_compile(
            models, _config(regional=True), stage="after_stage_b"
        )
        assert record["generator"]["regions_compiled"] == 3
        assert record["generator"]["regional"] is True
        assert record["generator"]["wrapped"] is False, "the root must stay bare"
        assert result["generator"] is models["generator"], "regional rewrites in place"

    def test_a_refusal_is_not_reframed_as_a_compile_failure(self):
        """Planted: the wrapper turned "nothing qualified" into "torch.compile
        failed ... set compile.enabled: false" -- a frame that is wrong twice.
        Nothing failed, and `enabled: false` is not the fix.
        """
        models = {"generator": torch.nn.Sequential(torch.nn.Conv2d(2, 2, 1))}
        with pytest.raises(NoRegionsToCompileError) as excinfo:
            apply_compile(models, _config(regional=True), stage="after_stage_b")
        message = str(excinfo.value)
        assert "regional: false" in message
        assert "torch.compile failed" not in message
        assert "enabled: false" not in message

    def test_nothing_is_mutated_when_one_model_does_not_qualify(self):
        """`compile_regions` rewrites in place, so a per-model check would leave
        the earlier models compiled and the build half-applied."""
        qualifying = self._stacked()["generator"]
        models = {
            "generator": qualifying,
            "discriminator": torch.nn.Sequential(torch.nn.Conv2d(2, 2, 1)),
        }
        with pytest.raises(NoRegionsToCompileError, match="discriminator"):
            apply_compile(models, _config(regional=True), stage="after_stage_b")
        assert all(type(b).__name__ == "Block" for b in qualifying.blocks)

    def test_whole_model_compilation_records_zero_regions(self):
        _, record = apply_compile(_models(), _config(), stage="after_stage_b")
        assert record["generator"]["regions_compiled"] == 0
        assert record["generator"]["regional"] is False


class TestApplyToSelectsWhichModelsCompile:
    """`regional` was unusable by every GAN arm before this existed.

    It refuses an arm whose models do not all expose a uniform block list, and a
    patch critic is a plain `nn.Sequential`. Naming the generator is what makes
    the combination reachable.
    """

    @staticmethod
    def _gan() -> dict[str, torch.nn.Module]:
        class Block(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 3, padding=1)

            def forward(self, x):
                return self.conv(x)

        class Generator(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = torch.nn.ModuleList([Block() for _ in range(4)])

        return {
            "generator": Generator(),
            "discriminator": torch.nn.Sequential(
                torch.nn.Conv2d(2, 8, 3), torch.nn.LeakyReLU(), torch.nn.Conv2d(8, 1, 3)
            ),
        }

    def test_regional_on_a_gan_is_blocked_without_a_selection(self):
        """Planted: this is the state the knob exists to fix, and it must stay
        red so the fix is not quietly reverted."""
        with pytest.raises(NoRegionsToCompileError, match="discriminator"):
            apply_compile(self._gan(), _config(regional=True), stage="after_stage_b")

    def test_naming_the_generator_clears_it(self):
        models = self._gan()
        _, record = apply_compile(
            models, _config(regional=True, apply_to=["generator"]), stage="after_stage_b"
        )
        assert record["generator"]["regions_compiled"] == 4
        assert "discriminator" not in record

    def test_an_unselected_model_is_returned_untouched(self):
        """The director assigns the returned mapping, so dropping a model here
        would delete it from the training environment."""
        models = self._gan()
        result, _ = apply_compile(
            models, _config(regional=True, apply_to=["generator"]), stage="after_stage_b"
        )
        assert set(result) == {"generator", "discriminator"}
        assert type(result["discriminator"]).__name__ == "Sequential"

    def test_naming_a_model_this_arm_does_not_build_raises(self):
        """Planted: the schema closes the vocabulary, but a GAN arm naming
        `encoder` still asks for something absent -- and compiling nothing while
        reporting compilation is the failure this module refuses."""
        with pytest.raises(RuntimeError, match="does not build"):
            apply_compile(
                self._gan(), _config(apply_to=["encoder"]), stage="after_stage_b"
            )

    def test_no_selection_still_compiles_everything(self):
        _, record = apply_compile(self._gan(), _config(), stage="after_stage_b")
        assert set(record) == {"generator", "discriminator"}


class TestTheRecompilationPolicy:
    """Torch drops a frame to eager once it exhausts its recompilation budget,
    and `fail_on_recompile_limit_hit` is OFF by default -- so a run can stop
    being compiled partway through and still report a compiled configuration.
    """

    def test_the_budget_is_applied_to_dynamo(self):
        import torch._dynamo.config as dynamo_config

        apply_compile(_models(), _config(recompile_limit=16), stage="after_stage_b")
        assert dynamo_config.recompile_limit == 16

    def test_failing_on_the_limit_is_applied_and_defaults_on(self):
        """Planted: torch ships this False. Leaving it there keeps the silent
        eager fallback that the build-time raise already refuses."""
        import torch._dynamo.config as dynamo_config

        dynamo_config.fail_on_recompile_limit_hit = False
        apply_compile(_models(), _config(), stage="after_stage_b")
        assert dynamo_config.fail_on_recompile_limit_hit is True

    def test_the_policy_is_recorded(self):
        """Otherwise a run cannot say afterwards what budget it ran under."""
        _, record = apply_compile(
            _models(), _config(recompile_limit=12), stage="after_stage_b"
        )
        assert record["generator"]["recompile_limit"] == 12
        assert record["generator"]["fail_on_recompile_limit"] is True

    def test_an_absent_budget_leaves_torchs_value_alone(self):
        import torch._dynamo.config as dynamo_config

        dynamo_config.recompile_limit = 99
        apply_compile(_models(), _config(), stage="after_stage_b")
        assert dynamo_config.recompile_limit == 99
        dynamo_config.recompile_limit = 8
