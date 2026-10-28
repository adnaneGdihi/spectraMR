"""Schema for the top-level ``adapters:`` block.

Adapters are *the only declarative way* to bridge a capability mismatch
between data, model, loss, and metric layers. Per CLAUDE.md item #9 they
NEVER fire silently — the YAML must opt in. The audit's
``check_data_model_compatibility`` (Phase 2) walks each declared chain
and verifies it actually bridges the gap claimed.

YAML shape::

    adapters:
      pre_model:
        - name: slice_3d_to_2d
          axis: 2
      post_model:
        - name: gather_2d_to_3d
          axis: 2
      pre_loss_target:
        - name: rss_coils_to_magnitude

See ``docs/superpowers/specs/2026-05-05-experiment-spec-card-and-adapters-design.md``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AdapterStepSchema(BaseModel):
    """One adapter invocation at a hook point.

    ``name`` resolves to ``ADAPTER_REGISTRY[name]`` at audit time and
    must be registered. Any other key is forwarded as a kwarg to the
    adapter's ``__init__``.
    """

    # NN#1: config schemas are frozen. ``extra="allow"`` still forwards adapter
    # kwargs — they're set at construction, then immutable (no code mutates a
    # constructed AdapterStepSchema; grep-verified).
    model_config = ConfigDict(extra="allow", protected_namespaces=(), frozen=True)

    name: str = Field(description="Registered adapter name (see src/data/adapters/).")
    enabled: bool = Field(
        default=True, description="Set False to skip this step without removing it."
    )


class AdaptersConfigSchema(BaseModel):
    """Per-hook adapter chains.

    Each list is *ordered*: the first adapter consumes the upstream
    form, each subsequent adapter consumes the previous one's output.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=(), frozen=True)

    pre_model: list[AdapterStepSchema] = Field(
        default_factory=list,
        description="Applied between dataset and model. Example: 5D→4D slicer.",
    )
    post_model: list[AdapterStepSchema] = Field(
        default_factory=list,
        description="Applied to model output before downstream stages.",
    )
    pre_loss_target: list[AdapterStepSchema] = Field(
        default_factory=list,
        description="Bridges dataset target to loss-side target (e.g. RSS coils→magnitude).",
    )
    pre_metric: list[AdapterStepSchema] = Field(
        default_factory=list,
        description="Applied to (pred, target) before metric computation.",
    )

    @model_validator(mode="before")
    @classmethod
    def _refuse_retired_hooks(cls, data: Any) -> Any:
        """Refuse the retired ``pre_loss_pred`` hook (#1777): declared, never applied."""
        if isinstance(data, dict) and "pre_loss_pred" in data:
            raise ValueError(
                "adapters.pre_loss_pred is retired (#1777): the hook was declared in YAML but "
                "never applied to the loss path. Remove the pre_loss_pred block. A model/loss "
                "domain mismatch must be fixed by changing losses.output_domain or the model, "
                "not bridged by an adapter."
            )
        return data

    def all_steps(self) -> list[tuple[str, AdapterStepSchema]]:
        """Flat ``(hook, step)`` list across all hooks. Order: declaration."""
        out: list[tuple[str, AdapterStepSchema]] = []
        for hook in (
            "pre_model",
            "post_model",
            "pre_loss_target",
            "pre_metric",
        ):
            for step in getattr(self, hook):
                out.append((hook, step))
        return out

    def is_empty(self) -> bool:
        return not any(
            getattr(self, h)
            for h in (
                "pre_model",
                "post_model",
                "pre_loss_target",
                "pre_metric",
            )
        )

    def kwargs_for(self, step: AdapterStepSchema) -> dict[str, Any]:
        """Adapter constructor kwargs (everything except ``name`` and ``enabled``)."""
        return {k: v for k, v in step.model_dump().items() if k not in ("name", "enabled")}


__all__ = ["AdapterStepSchema", "AdaptersConfigSchema"]
