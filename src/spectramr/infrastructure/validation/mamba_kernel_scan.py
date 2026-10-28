"""Static answer to "does this model class construct a ``mamba_ssm`` kernel?".

A class requires the kernel when its definition reaches, through names and
intra-package imports, :class:`MambaBlock` or a ``mamba_ssm`` import on every
path. The scan reads source with :mod:`ast` and never imports or builds a model.

Routes behind one branch of an ``if`` are not counted: they depend on the arm's
config. A route in both branches counts. ``try`` bodies are optional imports
(non-negotiable 18) and are not followed.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
from collections.abc import Iterable
from pathlib import Path

_KERNEL_MODULE = "spectramr.models.blocks.mamba_block"
_KERNEL_SYMBOL = "MambaBlock"
_KERNEL_PACKAGE = "mamba_ssm"
_PACKAGE_ROOT = "spectramr"

Route = tuple[str, ...]
_Seen = set[tuple[str, str]]


class _ModuleIndex:
    """Top-level definitions and the import bindings of one module."""

    def __init__(self, tree: ast.Module) -> None:
        self.defs: dict[str, ast.AST] = {}
        # local name -> (absolute module, original name, or None for ``import x``)
        self.imports: dict[str, tuple[str, str | None]] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.defs[node.name] = node
        for node in _walk_unconditional(tree.body):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                for alias in node.names:
                    self.imports[alias.asname or alias.name] = (node.module, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name.split(".")[0]
                    self.imports[local] = (alias.name, None)


def _walk_unconditional(nodes: Iterable[ast.AST]) -> Iterable[ast.AST]:
    """Nodes reachable without passing through an ``if`` branch or a ``try`` body."""
    stack = list(nodes)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.If, ast.IfExp)):
            # Only the test is unconditional; the branches are judged by _node_route.
            stack.append(node.test)
        elif isinstance(node, ast.Try):
            stack.extend(node.handlers)
            stack.extend(node.orelse)
            stack.extend(node.finalbody)
        else:
            stack.extend(ast.iter_child_nodes(node))


@functools.cache
def _module_index(module: str) -> _ModuleIndex | None:
    spec = importlib.util.find_spec(module)
    origin = getattr(spec, "origin", None)
    if not origin or not origin.endswith(".py"):
        return None
    return _ModuleIndex(ast.parse(Path(origin).read_text(encoding="utf-8")))


def _is_kernel_package(module: str) -> bool:
    return module.split(".")[0] == _KERNEL_PACKAGE


def _resolve(module: str, symbol: str, seen: _Seen) -> Route | None:
    """Route from ``module.symbol`` to the kernel, or None."""
    if (module, symbol) in seen:
        return None
    seen.add((module, symbol))
    if module == _KERNEL_MODULE and symbol == _KERNEL_SYMBOL:
        return (f"{module}.{symbol}",)
    index = _module_index(module)
    if index is None:
        return None
    definition = index.defs.get(symbol)
    if definition is not None:
        route = _nodes_route([definition], module, index, seen)
        return None if route is None else (f"{module}.{symbol}", *route)
    # Re-exported name: follow its import binding.
    binding = index.imports.get(symbol)
    return None if binding is None else _binding_route(binding, seen)


def _binding_route(binding: tuple[str, str | None], seen: _Seen) -> Route | None:
    module, original = binding
    if _is_kernel_package(module):
        return (module,)
    if original is None or not module.startswith(_PACKAGE_ROOT):
        return None
    return _resolve(module, original, seen)


def _nodes_route(
    nodes: Iterable[ast.AST], module: str, index: _ModuleIndex, seen: _Seen
) -> Route | None:
    """First route to the kernel among ``nodes``, following only unconditional code."""
    for node in _walk_unconditional(nodes):
        if isinstance(node, (ast.If, ast.IfExp)):
            # A route counts only when both branches carry it.
            body = [node.body] if isinstance(node, ast.IfExp) else node.body
            orelse = [node.orelse] if isinstance(node, ast.IfExp) else node.orelse
            then_route = _nodes_route(body, module, index, set(seen))
            if then_route is not None and _nodes_route(orelse, module, index, set(seen)):
                return then_route
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                route = _binding_route((node.module, alias.name), seen)
                if route is not None:
                    return route
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_kernel_package(alias.name):
                    return (alias.name,)
        elif isinstance(node, ast.Name):
            if node.id in index.defs:
                route = _resolve(module, node.id, seen)
                if route is not None:
                    return route
            binding = index.imports.get(node.id)
            if binding is not None:
                route = _binding_route(binding, seen)
                if route is not None:
                    return route
    return None


def mamba_kernel_route(cls: type) -> Route | None:
    """The chain of definitions by which ``cls`` reaches the kernel, or None.

    The last element is ``spectramr.models.blocks.mamba_block.MambaBlock`` or a
    ``mamba_ssm`` module name.
    """
    top_level = cls.__qualname__.split(".")[0]
    return _resolve(cls.__module__, top_level, set())


def class_requires_mamba_kernel(cls: type) -> bool:
    """Whether ``cls`` unconditionally constructs a ``MambaBlock`` or imports ``mamba_ssm``."""
    return mamba_kernel_route(cls) is not None
