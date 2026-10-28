"""Builds runtime adapter chains from a validated ``adapters:`` YAML block.

The chain is a ``dict[hook_name, list[Adapter]]`` that the strategy
applies at the corresponding insertion points. The audit
(``check_data_model_compatibility`` in
``src/infrastructure/validation/config_health_checker.py``) has already
verified the chain is valid and bridges the declared mismatch — this
builder just instantiates it.

Because adapter classes are registered with ``@register_adapter`` and
expose a standard ``__call__(x)`` shape, the builder is a thin wrapper
over the registry. See
``docs/superpowers/specs/2026-05-05-experiment-spec-card-and-adapters-design.md``.
"""

from __future__ import annotations

import logging
from typing import Any

from spectramr.config.schemas.adapters import AdaptersConfigSchema
from spectramr.data.adapters.registry import get_adapter, get_adapter_capabilities
from spectramr.models.capabilities import VALID_INSERTION_POINTS

logger = logging.getLogger(__name__)


HOOKS: tuple[str, ...] = (
    "pre_model",
    "post_model",
    "pre_loss_target",
    "pre_metric",
)


class AdapterChainBuilder:
    """Materialize ``adapters:`` YAML into a hook→[adapter,...] dict.

    Returns an empty dict when the config has no ``adapters:`` block,
    so callers can ``chain.get(hook, [])`` without conditional branches.
    """

    def __init__(self, adapters_cfg: AdaptersConfigSchema | None):
        self._cfg = adapters_cfg

    def build(self) -> dict[str, list[Any]]:
        # Force-import the adapters package so concrete adapters register
        # before the registry lookups below.
        import spectramr.data.adapters  # noqa: F401

        result: dict[str, list[Any]] = {h: [] for h in HOOKS}
        if self._cfg is None:
            return result
        for hook in HOOKS:
            steps = getattr(self._cfg, hook, None) or []
            for step in steps:
                if not getattr(step, "enabled", True):
                    continue
                # Defense-in-depth: re-validate the step is allowed at this
                # hook even though the audit already checked. A live mismatch
                # at strategy setup time is better than a silent no-op.
                caps = get_adapter_capabilities(step.name)
                if caps is not None and caps.insertion_points and hook not in caps.insertion_points:
                    raise ValueError(
                        f"Adapter {step.name!r} is not allowed at hook {hook!r}; "
                        f"declared insertion_points={caps.insertion_points}."
                    )
                cls = get_adapter(step.name)
                # Forward all step kwargs except `name`/`enabled` to the
                # adapter constructor.
                init_kwargs = {
                    k: v for k, v in step.model_dump().items() if k not in ("name", "enabled")
                }
                try:
                    instance = cls(**init_kwargs)
                except TypeError as exc:
                    raise TypeError(
                        f"Adapter {step.name!r} constructor rejected kwargs {init_kwargs}: {exc}"
                    ) from exc
                result[hook].append(instance)
                logger.info(
                    "Adapter built: %s @ %s (kwargs=%s)",
                    step.name,
                    hook,
                    init_kwargs,
                )
        return result


def apply_chain(chain: list[Any], x: Any) -> Any:
    """Sequentially apply each adapter in the chain to ``x``.

    Adapters are callable (the registry's :class:`Adapter` Protocol) —
    plain functional adapters and ``nn.Module`` subclasses both work.
    """
    for adapter in chain:
        x = adapter(x)
    return x


def apply_pre_model(
    chains: dict[str, list[Any]], input_batch: Any, target_batch: Any
) -> tuple[Any, Any]:
    """Apply the ``pre_model`` chain to input and target, the one owner for train and validation.

    Both paths must hand the model the same tensor; a validation that skips this grades
    the model on a different input than it trained on.
    """
    chain = chains.get("pre_model") or []
    if not chain:
        return input_batch, target_batch
    if input_batch is not None:
        input_batch = apply_chain(chain, input_batch)
    if target_batch is not None:
        target_batch = apply_chain(chain, target_batch)
    return input_batch, target_batch


__all__ = ["HOOKS", "AdapterChainBuilder", "apply_chain", "apply_pre_model"]
# VALID_INSERTION_POINTS re-exported for callers that need to validate
# YAML hook names against the canonical set.
assert set(HOOKS) == set(VALID_INSERTION_POINTS), "HOOKS and VALID_INSERTION_POINTS must agree."
