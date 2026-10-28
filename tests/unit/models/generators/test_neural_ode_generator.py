"""``NeuralODEGenerator``: the solver it picks, and what that costs per forward.

Three defects lived in ``forward`` at once, and each one is invisible to a test
that only checks the output shape:

1. A ten-iteration loop computed the four RK4 stages and **never wrote back to**
   ``h``. Its results were rebound and discarded, so the reconstruction was
   correct -- it simply cost twice what it should. Eight arms select this model,
   two of them under ``experiments/validated/``.
2. ``from torchdiffeq import odeint`` sat **inside** ``forward``. Python does not
   negative-cache a failed import, so every step re-ran the whole ``sys.meta_path``
   finder search.
3. ``except ImportError: pass`` chose a different integrator without a word, and
   the two integrators did not even share a grid: ``linspace(0, T, 10)`` is nine
   intervals of ``T/9``, while the built-in solver walks ten of ``T/10``.

Each test below fails on the specific shape it names. None of them reads source
text -- a comment or a docstring must not be able to satisfy them.
"""

from __future__ import annotations

import sys

import pytest
import torch

from spectramr.models.generators import neural_ode_generator as mod
from spectramr.models.generators.neural_ode_generator import NeuralODEGenerator

# Ten RK4 steps, four derivative evaluations each. The dead loop made it 80.
_STEPS = 10
_EVALS_PER_FORWARD = _STEPS * 4


def _counting_model() -> tuple[NeuralODEGenerator, dict[str, int]]:
    """A generator whose ``ode_func`` tallies every call it receives."""
    model = NeuralODEGenerator(in_channels=1, out_channels=1, features=8)
    calls = {"n": 0}
    inner = model.ode_func

    class Counting(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, t, h):
            calls["n"] += 1
            return self.inner(t, h)

    model.ode_func = Counting()
    return model, calls


def test_forward_spends_no_derivative_evaluations_on_a_discarded_loop() -> None:
    """40 evaluations, not 80.

    The dead loop is *output-preserving*, so nothing about the reconstruction
    can detect it. Counting the derivative evaluations is what makes it visible.
    """
    model, calls = _counting_model()
    model(torch.randn(2, 1, 16, 16))
    assert calls["n"] == _EVALS_PER_FORWARD, (
        f"{calls['n']} ode_func evaluations per forward, expected "
        f"{_EVALS_PER_FORWARD} ({_STEPS} RK4 steps x 4 stages). A count of "
        f"{2 * _EVALS_PER_FORWARD} is the discarded stage loop returning."
    )


def test_forward_with_t_eval_also_pays_no_discarded_loop() -> None:
    """The dead loop sat *before* the ``t_eval`` branch, so both paths paid it."""
    model, calls = _counting_model()
    t_eval = torch.tensor([0.5, 1.0])
    out = model(torch.randn(2, 1, 16, 16), t_eval=t_eval)
    assert out.shape[1] == len(t_eval)
    # The t_eval path takes as many steps as it needs to reach each target time;
    # what it must NOT do is spend a fixed 40 evaluations before it starts.
    assert calls["n"] < 2 * _EVALS_PER_FORWARD, (
        f"{calls['n']} evaluations for two target times -- the discarded "
        f"{_EVALS_PER_FORWARD}-evaluation loop appears to run before the branch."
    )


def test_forward_attempts_no_module_import() -> None:
    """The optional import is resolved once at module load, never per step.

    Asserted against the import system itself rather than the source text: a
    ``finder`` that records every name Python tries to resolve while ``forward``
    runs. A hoisted import touches it zero times.
    """

    seen: list[str] = []

    class Recorder:
        def find_module(self, fullname, path=None):  # pragma: no cover - legacy API
            return None

        def find_spec(self, fullname, path=None, target=None):
            seen.append(fullname)
            return None

    recorder = Recorder()
    sys.meta_path.insert(0, recorder)
    try:
        model = NeuralODEGenerator(in_channels=1, out_channels=1, features=8)
        model(torch.randn(1, 1, 16, 16))
    finally:
        sys.meta_path.remove(recorder)

    assert "torchdiffeq" not in seen, (
        "forward() attempted `import torchdiffeq`. Python does not negative-cache "
        "a failed import, so this re-runs the full meta_path search every step. "
        "Resolve it once at module level behind HAVE_TORCHDIFFEQ."
    )


def test_the_absence_of_torchdiffeq_is_a_named_flag_not_a_silent_branch() -> None:
    """Absent is a state to report, never a state to infer (non-negotiable 18)."""
    assert hasattr(mod, "HAVE_TORCHDIFFEQ"), (
        "no HAVE_TORCHDIFFEQ flag: whether the adjoint/adaptive solvers are "
        "reachable is then undiscoverable from outside forward()."
    )
    assert isinstance(mod.HAVE_TORCHDIFFEQ, bool)


def test_both_solvers_walk_the_same_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """``linspace(0, T, steps)`` is ``steps - 1`` intervals, not ``steps``.

    Substitutes a stand-in solver on the module globals so the torchdiffeq branch
    runs on a host where the real package is absent, and reads the grid the
    generator actually hands it. ``importlib.reload`` is deliberately NOT used --
    it re-fires ``@register_model``, which refuses a duplicate name.
    """
    captured: dict[str, torch.Tensor] = {}

    def fake_odeint(func, y0, t, method=None, **kwargs):
        captured["t"] = t.detach().clone()
        return torch.stack([y0, y0])

    monkeypatch.setattr(mod, "HAVE_TORCHDIFFEQ", True)
    monkeypatch.setattr(mod, "odeint", fake_odeint)

    model = NeuralODEGenerator(in_channels=1, out_channels=1, features=8)
    model.integration_time = 1.0
    model(torch.randn(1, 1, 16, 16))

    assert "t" in captured, (
        "HAVE_TORCHDIFFEQ was True and the stand-in solver was never called -- "
        "forward() is not branching on the module-level flag."
    )
    grid = captured["t"]
    assert grid.numel() == _STEPS + 1, (
        f"torchdiffeq is handed {grid.numel()} points, i.e. {grid.numel() - 1} "
        f"intervals of T/{grid.numel() - 1}, while the built-in solver walks "
        f"{_STEPS} of T/{_STEPS}. The two branches disagree on the discretisation, "
        "so a result depends on whether an undeclared package happens to be installed."
    )
    step = torch.diff(grid)
    assert torch.allclose(step, torch.full_like(step, 1.0 / _STEPS)), (
        f"grid is not uniform at T/{_STEPS}: {grid.tolist()}"
    )


def test_the_registered_name_still_resolves_to_this_class() -> None:
    """Guards the fix against being applied to an unreachable class."""
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    entry = MODEL_REGISTRY.get("neural_ode")
    assert entry is not None, "model 'neural_ode' left the registry"


def _depth_model(mode: str) -> NeuralODEGenerator:
    torch.manual_seed(0)
    return NeuralODEGenerator(in_channels=1, out_channels=1, features=8, depth_mode=mode)


def test_an_unknown_depth_mode_raises() -> None:
    with pytest.raises(ValueError, match="depth_mode='ode'"):
        NeuralODEGenerator(features=8, depth_mode="ode")


def test_the_fixed_depth_control_is_a_resnet_of_untied_blocks() -> None:
    fixed = _depth_model("fixed")
    assert fixed.ode_func is None and len(fixed.residual_blocks) == mod._STEPS
    ode = _depth_model("continuous")
    assert ode.residual_blocks is None
    per_block = sum(p.numel() for p in ode.ode_func.parameters())
    extra = sum(p.numel() for p in fixed.parameters()) - sum(p.numel() for p in ode.parameters())
    assert extra == (mod._STEPS - 1) * per_block


def test_the_two_depth_modes_compute_different_maps() -> None:
    x = torch.rand(1, 1, 8, 8)
    with torch.no_grad():
        a, b = _depth_model("continuous")(x), _depth_model("fixed")(x)
    assert a.shape == b.shape == x.shape and not torch.allclose(a, b)


def test_the_fixed_control_refuses_a_time_query() -> None:
    with pytest.raises(ValueError, match="no continuous time"):
        _depth_model("fixed")(torch.rand(1, 1, 8, 8), t_eval=torch.tensor([0.5]))
