"""``AMPPolicy.should_use_amp`` must not be a second AMP resolver (#806).

It read ``optimization.precision.enabled`` and never ``precision.dtype``, so it
could not represent the third state ``PrecisionConfigSchema`` documents --
``dtype: 'float32'`` disables AMP even when ``enabled`` is true. And it
force-disabled AMP whenever the ``model_type`` STRING contained "diffusion",
silently overriding an explicit ``precision.enabled: true``.

It had no production callers, so neither bit had fired yet; but it sits on the
``IAMPPolicy`` ABC, which is an invitation to call it. These tests pin it to
``resolve_amp_precision`` -- the resolver ``BaseTrainingStrategy`` and
``build_deepspeed_config`` already use -- rather than restating its answers,
so the two cannot drift apart again (pitfall #13b).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from spectramr.config.schemas.optimization import (  # noqa: E402
    OptimizationConfigSchema,
)
from spectramr.infrastructure.training.mixed_precision import (  # noqa: E402
    resolve_amp_precision,
)
from spectramr.infrastructure.training.optimizers.amp_policy import (  # noqa: E402
    AMPPolicy,
)

#: Built via the string so the collection-time hard-CUDA scanner in
#: ``tests/conftest.py`` does not auto-mark this module ``gpu``: nothing here
#: allocates on a device, it only reads ``device.type``.
_ACCEL = torch.device("cuda")
_CPU = torch.device("cpu")


def _config(enabled: bool, dtype: str | None) -> SimpleNamespace:
    """A settings-shaped object carrying the REAL precision block."""
    return SimpleNamespace(
        optimization=OptimizationConfigSchema(
            precision={"enabled": enabled, "dtype": dtype}
        )
    )


#: Every declarable precision state. `float32` is the one the old code could
#: not represent.
_STATES = [
    (False, None),
    (False, "bfloat16"),
    (True, None),
    (True, "float16"),
    (True, "bfloat16"),
    (True, "float32"),
]


class TestItTracksTheAmpSsot:
    @pytest.mark.parametrize("enabled,dtype", _STATES)
    def test_it_agrees_with_resolve_amp_precision(self, enabled, dtype) -> None:
        """Parametrised over the whole state space, and compared against the
        SSOT rather than against a restated expectation -- a hand-written
        expected value is a third copy of the same decision."""
        expected, _ = resolve_amp_precision(enabled, dtype)
        assert (
            AMPPolicy().should_use_amp("unet", _ACCEL, _config(enabled, dtype))
            is expected
        )

    def test_float32_disables_amp_even_when_enabled_is_true(self) -> None:
        """The headline #806 defect, stated on its own so a regression names it.

        ``PrecisionConfigSchema``: "dtype='float32' disables AMP even when
        enabled is true -- a third state". The old reader saw only `enabled`
        and returned True here.
        """
        assert resolve_amp_precision(True, "float32")[0] is False
        assert AMPPolicy().should_use_amp("unet", _ACCEL, _config(True, "float32")) is (
            False
        )


class TestNoHardcodedParadigmBranch:
    """A ``model_type`` substring must not override a declared precision."""

    @pytest.mark.parametrize(
        "model_type",
        ["latent_diffusion", "cold_diffusion", "DIFFUSION", "x_diffusion"],
    )
    def test_diffusion_named_models_honour_the_declared_precision(
        self, model_type: str
    ) -> None:
        assert (
            AMPPolicy().should_use_amp(model_type, _ACCEL, _config(True, "bfloat16"))
            is True
        )

    def test_and_still_respect_it_when_it_says_off(self) -> None:
        assert (
            AMPPolicy().should_use_amp("latent_diffusion", _ACCEL, _config(False, None))
            is False
        )

    def test_model_type_changes_nothing(self) -> None:
        """The parameter stays for the IAMPPolicy signature; it must not be read."""
        cfg = _config(True, "bfloat16")
        answers = {
            AMPPolicy().should_use_amp(m, _ACCEL, cfg)
            for m in ("unet", "diffusion", "gan", "", "vae")
        }
        assert answers == {True}


class TestPreservedBehaviour:
    def test_force_disable_wins(self) -> None:
        assert (
            AMPPolicy(force_disable_amp=True).should_use_amp(
                "unet", _ACCEL, _config(True, "bfloat16")
            )
            is False
        )

    def test_cpu_is_never_amp(self) -> None:
        assert (
            AMPPolicy().should_use_amp("unet", _CPU, _config(True, "bfloat16")) is False
        )

    def test_absent_config_does_not_object(self) -> None:
        assert AMPPolicy().should_use_amp("unet", _ACCEL, None) is True

    def test_a_config_without_optimization_raises(self) -> None:
        """Fail loudly rather than read a stale key off an unknown shape."""
        with pytest.raises(TypeError, match="optimization"):
            AMPPolicy().should_use_amp("unet", _ACCEL, SimpleNamespace())


class TestItDelegatesRatherThanReimplements:
    def test_the_source_calls_the_ssot(self) -> None:
        """Anti-drift: agreement today is not the same as one resolver.

        Both spellings returned the same answer for every state EXCEPT
        `float32`, which is why the divergence survived. Pin the delegation
        itself, not just its current output.
        """
        import inspect

        src = inspect.getsource(AMPPolicy.should_use_amp)
        assert "resolve_amp_precision" in src

    def test_it_does_not_read_enabled_without_dtype(self) -> None:
        """The exact shape of the bug: `precision.enabled` consulted alone."""
        import inspect

        src = inspect.getsource(AMPPolicy.should_use_amp)
        assert "bool(config.optimization.precision.enabled)" not in src


class TestAStepThatTrainsNothingRaises:
    """#2309: the pre-backward guard reads only ``loss.grad_fn``, so a severed
    loss that went through one more op reached backward and stepped nothing.
    Each shape below is a planted violation the post-backward check must catch.
    """

    class _Helper:
        scaler = None

        @staticmethod
        def get_autocast_context():
            import contextlib

            return contextlib.nullcontext()

    def _run(self, make_loss, *, gas: int = 1, steps: int = 1):
        import torch

        from spectramr.infrastructure.training.optimizers.amp_policy import AMPPolicy
        from spectramr.infrastructure.training.step_executor import StepExecutor

        model = torch.nn.Linear(3, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        executor = StepExecutor(self._Helper(), AMPPolicy(), gradient_accumulation_steps=gas)
        for step in range(steps):
            executor.execute_step(
                {"optimizer": opt, "closure": lambda: make_loss(model), "model": model, "name": "g"},
                epoch=0,
                global_step=step,
            )
        return model

    @staticmethod
    def _severed_leaf():
        import torch

        return torch.tensor(1.0, requires_grad=True)

    def test_a_weighted_severed_leaf_raises(self) -> None:
        import pytest

        with pytest.raises(RuntimeError, match="received a gradient"):
            self._run(lambda m: self._severed_leaf() * 0.5)

    def test_a_python_sum_over_a_severed_leaf_raises(self) -> None:
        import pytest

        with pytest.raises(RuntimeError, match="received a gradient"):
            self._run(lambda m: sum([self._severed_leaf()]))

    def test_it_raises_at_the_accumulation_boundary(self) -> None:
        import pytest

        with pytest.raises(RuntimeError, match="received a gradient"):
            self._run(lambda m: self._severed_leaf() * 0.5, gas=2, steps=2)

    def test_a_connected_loss_steps(self) -> None:
        import torch

        torch.manual_seed(0)
        start = torch.nn.Linear(3, 1).weight.detach().clone()
        torch.manual_seed(0)
        model = self._run(lambda m: (m(torch.ones(2, 3)) ** 2).mean(), steps=3)
        assert not torch.allclose(model.weight.detach(), start)

    def test_a_zero_weighted_connected_loss_is_not_a_severed_one(self) -> None:
        """A weight of 0 still reaches every parameter: zero gradients, not none."""
        import torch

        self._run(lambda m: 0.0 * (m(torch.ones(2, 3)) ** 2).mean(), steps=2)
