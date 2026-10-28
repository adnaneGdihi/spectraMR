"""A strategy that calls a non-forward method on the generator must declare ``parallel_compat``.

A call such as ``generator.encode(x)`` bypasses the parallel wrapper's forward: DDP and ZeRO
never reduce a gradient that flows only through it, and FSDP reads a shard where it expects a
whole tensor. The run trains and reports success. The class-level ``parallel_compat``
declaration is what makes ``TrainingEnvironmentDirector`` refuse the combination (#2696), so a
strategy making such a call without one is the defect this gate names.

The scan is syntactic: a call on an attribute or name called ``generator`` / ``generator_model`` /
``gen`` / ``model`` whose method is not one every ``nn.Module`` has.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.architecture

STRATEGIES = Path("src/spectramr/infrastructure/training/strategies")

#: Methods every nn.Module has, or that the training loop legitimately calls on the wrapper.
_MODULE_METHODS = frozenset(
    {
        "forward",
        "train",
        "eval",
        "parameters",
        "named_parameters",
        "buffers",
        "named_buffers",
        "to",
        "cuda",
        "cpu",
        "state_dict",
        "load_state_dict",
        "zero_grad",
        "requires_grad_",
        "children",
        "modules",
        "named_modules",
        "apply",
        "half",
        "float",
        "double",
        "type",
        "register_forward_hook",
        "register_forward_pre_hook",
        "register_buffer",
        "no_sync",
        "named_children",
        "extra_repr",
        "get_submodule",
        "get_parameter",
        "share_memory",
        "bfloat16",
    }
)
_GENERATOR_NAMES = frozenset({"generator", "generator_model", "gen", "model"})

#: ``class -> methods`` reviewed as safe: no gradient flows through them.
_REVIEWED_SAFE: dict[str, frozenset[str]] = {
    # Sets the progressive-growing phase; changes module state, computes nothing.
    "ProgressiveGANStrategy": frozenset({"set_phase"}),
}


def _is_generator(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr in _GENERATOR_NAMES
    return isinstance(node, ast.Name) and node.id in {"generator", "gen"}


def _non_forward_calls(cls: ast.ClassDef) -> set[str]:
    calls: set[str] = set()
    for node in ast.walk(cls):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and _is_generator(node.func.value)
            and node.func.attr not in _MODULE_METHODS
            # torch.Generator().manual_seed(...) is a random generator, not the model.
            and node.func.attr != "manual_seed"
        ):
            calls.add(node.func.attr)
    return calls


def _declares(cls: ast.ClassDef) -> bool:
    return any(
        isinstance(stmt, ast.Assign)
        and any(getattr(t, "id", "") == "parallel_compat" for t in stmt.targets)
        for stmt in cls.body
    )


def _classes() -> dict[str, tuple[str, ast.ClassDef]]:
    out: dict[str, tuple[str, ast.ClassDef]] = {}
    for path in sorted(STRATEGIES.rglob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef):
                out[node.name] = (str(path.relative_to(STRATEGIES)), node)
    return out


def undeclared_offenders(classes: dict[str, tuple[str, ast.ClassDef]]) -> set[str]:
    """``file::Class (methods)`` for each class that makes the calls without declaring."""

    def declared(name: str, seen: tuple[str, ...] = ()) -> bool:
        if name not in classes or name in seen:
            return False
        node = classes[name][1]
        bases = [b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in node.bases]
        return _declares(node) or any(declared(b, (*seen, name)) for b in bases)

    offenders: set[str] = set()
    for name, (file, node) in classes.items():
        calls = _non_forward_calls(node) - _REVIEWED_SAFE.get(name, frozenset())
        if calls and not declared(name):
            offenders.add(f"{file}::{name} ({', '.join(sorted(calls))})")
    return offenders


def test_no_undeclared_strategy_calls_a_generator_method_outside_the_wrapper() -> None:
    offenders = undeclared_offenders(_classes())
    assert not offenders, (
        "These strategies call a non-forward method on the generator, which a parallel "
        "wrapper does not see. Declare `parallel_compat = frozenset()` on the class (or the "
        "backends it was audited for), or add it to _REVIEWED_SAFE with the reason:\n  "
        + "\n  ".join(sorted(offenders))
    )


def test_the_gate_turns_red_on_a_planted_violation() -> None:
    planted = ast.parse(
        "class Plain:\n"
        "    def step(self):\n"
        "        return self.env.generator.encode(1)\n"
        "class Declared:\n"
        "    parallel_compat = frozenset()\n"
        "    def step(self):\n"
        "        return self.env.generator.encode(1)\n"
        "class Child(Declared):\n"
        "    def step(self):\n"
        "        return self.generator_model.render(1)\n"
        "class Benign:\n"
        "    def step(self):\n"
        "        self.generator_model.eval()\n"
        "        return torch.Generator().manual_seed(0)\n"
    )
    classes = {
        node.name: ("planted.py", node) for node in planted.body if isinstance(node, ast.ClassDef)
    }
    assert undeclared_offenders(classes) == {"planted.py::Plain (encode)"}
