"""Forward hooks that record what every model was actually fed and what it returned.

One pre-hook and one forward hook per entry of ``env.models``. Hooks are registered on
the module the environment holds and on its unwrapped inner module, because some
strategies call the inner one directly; a per-model depth counter records only the
outermost call, so a wrapper and its inner module never record the same forward twice.
A compiled module's ``_orig_mod`` is never hooked, and a hook traced into a compiled
graph (DeepSpeed's in-place ``engine.compile()``) returns at once, so it never breaks
the graph; such a model's stages are then recorded only by its outer wrapper.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from spectramr.core.snapshot_trace import View, current

logger = logging.getLogger(__name__)

_MAX_DEPTH = 3


def flatten_tensors(obj: Any, prefix: str = "", depth: int = 0) -> dict[str, torch.Tensor]:
    """Name every tensor in a nested args/kwargs/output structure."""
    if isinstance(obj, torch.Tensor):
        return {prefix or "out": obj}
    if depth >= _MAX_DEPTH:
        return {}
    out: dict[str, torch.Tensor] = {}
    if isinstance(obj, Mapping):
        items = ((str(k), v) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        items = ((str(i), v) for i, v in enumerate(obj))
    elif hasattr(obj, "__dataclass_fields__"):
        items = ((k, getattr(obj, k)) for k in obj.__dataclass_fields__)
    else:
        return {}
    for k, v in items:
        out.update(flatten_tensors(v, f"{prefix}.{k}" if prefix else k, depth + 1))
    return out


def _unwrapped(module: nn.Module) -> nn.Module:
    inner = getattr(module, "module", None)  # DDP / FSDP1 / DataParallel
    return inner if isinstance(inner, nn.Module) else module


def _fit(view: View, t: torch.Tensor) -> View:
    """Keep a declared channel split only on the tensor whose width it describes."""
    if not view.segments:
        return view
    width = sum(seg[1] for seg in view.segments)
    if t.dim() >= 2 and t.shape[1] == width:
        return view
    return View(kspace=view.kspace, authoritative=view.authoritative, log_scaled=view.log_scaled)


class ModelHooks:
    """Record ``model/input[<name>]`` and ``model/output[<name>]`` on a snapshot step.

    ``view_for(kind)`` gives the rendering view for ``"model_input"`` or
    ``"model_output"``; the active trace's ``model_views`` override it when a strategy
    declared how its input superposes domains.
    """

    def __init__(self, models: Mapping[str, nn.Module], view_for: Callable[[str], View]) -> None:
        self._view_for = view_for
        self._depth: dict[str, int] = {}
        self._hooked: set[int] = set()
        self.handles: list[Any] = []
        for name, module in models.items():
            self.attach(name, module)

    def attach(self, name: str, module: Any) -> None:
        """Hook ``module`` (and its unwrapped inner module) once, under ``name``.

        Strategies that build their own generator (cut, cyclegan, stargan_v2) register
        it through ``_capture_model_output(module=...)``.
        """
        if not isinstance(module, nn.Module) or id(module) in self._hooked:
            return
        targets = {id(module): module}
        if not hasattr(module, "_orig_mod"):
            inner = _unwrapped(module)
            targets[id(inner)] = inner
        for key, target in targets.items():
            if key not in self._hooked:
                self._hooked.add(key)
                self._attach(name, target)
        self.attach_stages(getattr(module, "_orig_mod", None) or _unwrapped(module))

    def attach_stages(self, root: Any) -> None:
        """Hook every module under ``root`` whose class declares a ``snapshot_stage``.

        Data consistency and projection modules declare ``snapshot_stage = "post"``, so
        each call inside a model's forward -- every cascade of an unrolled network --
        is recorded as ``post/input[<Class>]`` and ``post/output[<Class>]``.
        """
        if not isinstance(root, nn.Module):
            return
        for sub in root.modules():
            group = getattr(type(sub), "snapshot_stage", None)
            if group and id(sub) not in self._hooked:
                self._hooked.add(id(sub))
                self._attach(type(sub).__name__, sub, group=group)

    def _attach(self, name: str, module: nn.Module, group: str = "model") -> None:
        key = f"{group}:{name}"

        def pre(_m: nn.Module, args: tuple, kwargs: dict) -> None:
            if torch.compiler.is_compiling():  # constant under dynamo: no graph break
                return
            depth = self._depth.get(key, 0)
            self._depth[key] = depth + 1
            trace = current()
            if trace is None or depth:
                return
            tensors = flatten_tensors(args, "arg")
            tensors.update(flatten_tensors(kwargs, "kw"))
            self._add(trace, group, "input", name, tensors)

        def post(_m: nn.Module, _args: tuple, _kwargs: dict, output: Any) -> None:
            if torch.compiler.is_compiling():
                return
            depth = self._depth.get(key, 1) - 1
            self._depth[key] = max(depth, 0)
            trace = current()
            if trace is None or depth:
                return
            self._add(trace, group, "output", name, flatten_tensors(output))

        try:
            self.handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
            self.handles.append(
                module.register_forward_hook(post, with_kwargs=True, always_call=True)
            )
        except Exception as exc:  # a module that refuses hooks only loses its model stages
            logger.warning("[snapshot] could not hook model %r: %s", name, exc)

    def _add(
        self, trace: Any, group: str, kind: str, name: str, tensors: dict[str, torch.Tensor]
    ) -> None:
        if not tensors:
            return
        if group == "model":
            view = trace.model_views.get(kind) or self._view_for(f"model_{kind}")
            source = f"forward hook on env.models[{name!r}]"
        else:  # a DC / projection module may take or return either domain
            view, source = View(kspace=True), f"forward hook on {name} (snapshot_stage)"
        trace.add(
            group,
            f"{kind}[{name}]",
            tensors,
            views={k: _fit(view, t) for k, t in tensors.items()},
            params=None,
            source=source,
        )

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self._hooked.clear()


__all__ = ["ModelHooks", "flatten_tensors"]
