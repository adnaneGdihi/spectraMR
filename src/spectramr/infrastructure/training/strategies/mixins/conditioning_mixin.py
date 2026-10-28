"""Adaptive-conditioning seam shared by all training strategies (2026-05-26).

Lifts the FiLM ``AdaptiveConditioner`` wiring out of
``ConcreteVirtualFiducialStrategy`` so any strategy can adapt to its declared
condition sources (degradation severity, diffusion timestep, contrast, and the
domain axes field-strength / scanner / site). A strategy:

1. declares ``_SUPPORTED_CONDITION_SOURCES`` — the ``ConditioningContext``
   sources it can actually populate;
2. calls ``self._ensure_conditioner(n_channels)`` on first forward to lazily
   build the conditioner and register its params with the generator optimizer;
3. builds a ``ConditioningContext`` — ``_domain_conditioning_context(batch)``
   covers the standard exposed domain axes (``field_strength`` / ``scanner_id``
   / ``site_id`` / ``contrast_id``); strategies add their own (severity,
   diffusion_t, …).

The FiLM head is zero-init, so mixing this in is the identity until the
conditioner learns — no host model is perturbed. Mounted on
``BaseTrainingStrategy`` so every strategy inherits the seam; the empty default
``_SUPPORTED_CONDITION_SOURCES`` means "does not consume conditioning" (the
Tier-1 audit guard rejects an enabled ``model.conditioning`` whose sources are
not covered, per CLAUDE.md #9 — no silent no-op).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch

from spectramr.infrastructure.distributed.owned_modules import OWNED_PARAM_GROUP
from spectramr.infrastructure.training.strategies.owned_param_groups import add_owned_param_group
from spectramr.models.conditioning import AdaptiveConditioner, ConditioningContext

logger = logging.getLogger(__name__)

# Domain axes the data pipeline can expose via ``data.expose_*`` flags.
_DOMAIN_BATCH_KEYS = ("field_strength", "scanner_id", "site_id", "contrast_id")


def conditioning_block_of(config: Any) -> Any | None:
    """``config.model.conditioning``, read one getattr at a time (``config`` may be unbound)."""
    model_cfg = getattr(config, "model", None)
    return getattr(model_cfg, "conditioning", None)


def conditioning_enabled(cfg: Any | None) -> bool:
    """Whether a conditioner is built for ``cfg``: the one predicate build and gate share."""
    return (
        cfg is not None
        and bool(getattr(cfg, "enabled", False))
        and bool(getattr(cfg, "sources", None))
    )


class ConditioningMixin:
    """Shared adaptive-conditioning behaviour for ``BaseTrainingStrategy``."""

    #: Sources the strategy can populate. Subclasses widen this. Empty ⇒ the
    #: strategy does not consume conditioning.
    _SUPPORTED_CONDITION_SOURCES: tuple[str, ...] = ()

    def _conditioning_cfg(self) -> Any | None:
        return conditioning_block_of(getattr(self, "config", None))

    #: Set by a step that calls ``_apply_input_conditioning`` on its forward path. The
    #: conditioner is built only there, so only these classes own it at runtime.
    consumes_input_conditioning: ClassVar[bool] = False

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        """Owned parameters this config adds to the generator optimizer.

        Read by the parallel gate (``strategy_compat``) before any build, so a config the
        build path would refuse is refused at audit time. Subclasses extend it via super().
        """
        if cls.consumes_input_conditioning and conditioning_enabled(conditioning_block_of(config)):
            return (("_conditioner", OWNED_PARAM_GROUP),)
        return ()

    @classmethod
    def _validate_condition_sources(cls, sources: list[str]) -> None:
        """Raise if any requested source isn't in the strategy's supported set.

        Runtime mirror of the Tier-1 audit guard: a requested condition the
        strategy cannot populate must fail loudly, not silently do nothing.
        """
        unsupported = [s for s in sources if s not in cls._SUPPORTED_CONDITION_SOURCES]
        if unsupported:
            raise ValueError(
                f"{cls.__name__} does not support conditioning source(s) "
                f"{unsupported}; supported={list(cls._SUPPORTED_CONDITION_SOURCES)}. "
                "Add them to _SUPPORTED_CONDITION_SOURCES and populate them when "
                "building the ConditioningContext, or remove them from "
                "model.conditioning.sources."
            )

    def _ensure_conditioner(self, n_channels: int) -> AdaptiveConditioner | None:
        """Lazily build the FiLM conditioner and register params with opt_g.

        Built on first forward so ``num_features`` matches the real input
        channel count; its parameters are added to the generator optimizer so
        they train. Returns ``None`` when conditioning is disabled / sourceless.
        """
        cfg = self._conditioning_cfg()
        if not conditioning_enabled(cfg):
            return None
        self._validate_condition_sources(list(cfg.sources))
        if getattr(self, "_conditioner", None) is None:
            cond = AdaptiveConditioner.from_config(cfg, num_features=n_channels)
            if cond is not None:
                cond = cond.to(getattr(self, "device", torch.device("cpu")))
                env = getattr(self, "env", None)
                opt = (
                    getattr(env, "opt_g", None)
                    or getattr(env, "optimizer_g", None)
                    or getattr(env, "optimizer", None)
                )
                if opt is not None:
                    add_owned_param_group(
                        self, opt, {"params": list(cond.parameters())}, owner="its conditioner"
                    )
                else:
                    logger.warning(
                        "[%s] no generator optimizer found; conditioner params "
                        "will NOT be trained.",
                        type(self).__name__,
                    )
            self._conditioner = cond
        return self._conditioner

    @staticmethod
    def _domain_conditioning_context(batch: Any, device: Any) -> ConditioningContext | None:
        """Build a context from the standard exposed domain axes in ``batch``.

        Reads ``field_strength`` / ``scanner_id`` / ``site_id`` / ``contrast_id``
        when the data pipeline exposed them (``data.expose_*``). Returns ``None``
        when none are present, so callers can ``or`` it with their own context.
        """
        if not isinstance(batch, dict):
            return None
        kw: dict[str, torch.Tensor] = {}
        for key in _DOMAIN_BATCH_KEYS:
            v = batch.get(key)
            if isinstance(v, torch.Tensor):
                kw[key] = v.to(device)
        # Alias: m4raw_dataset / slice_dataset / multi_contrast collation emit the
        # contrast index as `contrast_idx`, while the ConditioningContext field (and
        # _DOMAIN_BATCH_KEYS) use `contrast_id`. Without this bridge, conditioning on
        # `contrast_id` silently no-ops on those datasets (pitfall #16). mrixfields_dataset
        # emits `contrast_id` directly and is already captured by the loop above, so an
        # explicit `contrast_id` takes precedence.
        if "contrast_id" not in kw:
            ci = batch.get("contrast_idx")
            if isinstance(ci, torch.Tensor):
                kw["contrast_id"] = ci.to(device)
        return ConditioningContext(**kw) if kw else None

    def _apply_input_conditioning(self, x: torch.Tensor, batch: Any) -> torch.Tensor:
        """FiLM-modulate ``x`` with the domain conditioning context from ``batch``.

        Returns the **same** ``x`` object when conditioning is *disabled*
        (``_ensure_conditioner`` is ``None``), so non-conditioned arms pay
        nothing. When conditioning is *enabled* but the batch carries none of the
        declared sources, this is the mechanism-fires guard (pitfall #16): the
        modulation would silently no-op, so it RAISES instead of quietly passing
        the input through — making the audit's "consumed" claim true at runtime
        rather than merely structural. A declared-but-unpropagated source fails
        at the first forward, never trains a silently-unconditioned model
        (CLAUDE.md #9). A source the strategy cannot populate at all is rejected
        earlier by ``_ensure_conditioner`` → ``_validate_condition_sources``.
        """
        cond = self._ensure_conditioner(int(x.shape[1]))
        if cond is None:
            return x
        ctx = self._domain_conditioning_context(batch, x.device)
        if ctx is None:
            cfg = self._conditioning_cfg()
            srcs = list(getattr(cfg, "sources", []) or [])
            raise ValueError(
                f"model.conditioning is enabled with sources {srcs} but the batch "
                "carried none of them — the FiLM modulation would silently no-op "
                "(pitfall #16). Expose the axis via data.expose_field_strength / "
                "data.expose_scanner_id / data.expose_site_id / "
                "data.multi_contrast.enabled (or use a dataset that emits it), or "
                "remove the source from model.conditioning.sources."
            )
        return cond(x, ctx)


__all__ = ["ConditioningMixin"]
