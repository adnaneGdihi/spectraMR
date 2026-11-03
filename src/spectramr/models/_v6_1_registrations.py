"""v6.1 model registrations — aliases + lightweight wrappers.

Plan: TODO/backlog_paradigm_expansion_roadmap.md (PR-3..PR-25),
      TODO/integration_plan_ulf_cheap_fast_mri.md (ideas 1, 2, 5, 8, 10),
      TODO/backlog_baseline_replication_experiment_11.md,
      TODO/backlog_multi_contrast_remaining.md.

All v6.1 ``model_type`` strings referenced by the new experiment YAMLs
land here. Most are thin wrappers (forward-pass passthroughs) over
already-registered backbones; the strategy classes carry the actual
training-loop logic. The wrappers:

- expose ``model_type`` to the audit and registry
- annotate the right capabilities so the audit ladder reports the
  correct domain / spatial dims
- forward to a configurable backbone (``unet`` by default) so the YAML
  smoke test does not need any new architecture code

When someone needs a *real* learned mask (LOUPE) or trajectory (PILOT)
or vendor embedding etc., they implement it inside the wrapper and
the registration line stays the same.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn

from spectramr.config.schemas.model import ModelConfigSchema
from spectramr.core.component_signature import signature_contract
from spectramr.models.registry import MODEL_REGISTRY, register_model

#: Top-level ``config.model`` fields GeneratorBuilder sweeps into every build on the chance the
#: constructor wants them; anything else reaching a wrapper came from the arm's ``model_kwargs``.
_SWEPT_MODEL_FIELDS = frozenset(ModelConfigSchema.model_fields) - {"model_kwargs"}


def _build_backbone(
    name: str,
    *,
    in_channels: int,
    out_channels: int,
    spatial_dims: int,
    kwargs: dict[str, Any],
    declared: frozenset[str] | None,
    owner: str,
) -> nn.Module:
    """Build ``name`` through the factory seam a top-level build uses; raise on an unread knob.

    ``declared`` is the arm's pre-sweep key set from the factory; without one (a direct
    construction) every key not named like a top-level model field counts as declared.
    """
    from spectramr.models.factories.model_factory import ModelFactory

    info = MODEL_REGISTRY.get(name)
    if info is None:
        raise KeyError(
            f"{owner}: backbone {name!r} is not registered. "
            f"Did you forget to call populate_model_registry()?"
        )
    if declared is None:
        declared = frozenset(kwargs) - _SWEPT_MODEL_FIELDS
    declared = declared & frozenset(kwargs)
    contract = signature_contract(info["class"])
    unread = sorted(k for k in declared if k not in contract.accepted)
    if unread and not contract.accepts_var_kwargs:
        raise TypeError(
            f"{owner}: backbone {name!r} does not accept model_kwargs {unread}; "
            f"accepted: {sorted(contract.accepted)}."
        )
    return ModelFactory().create_generator(
        model_type=name,
        in_channels=in_channels,
        out_channels=out_channels,
        _declared_keys=declared,
        **{"spatial_dims": spatial_dims, **kwargs},
    )


# ============================================================
# Generic v6.1 wrapper — used by most "thin pass-through" models
# ============================================================


class _BackboneAlias(nn.Module):
    """Generic forwarding wrapper. Strategy classes do the real work.

    Per CLAUDE.md #9 (silent fallbacks forbidden), unknown backbones and
    incompatible-kwarg errors raise instead of silently degrading to a
    3x3 conv (which is what the previous ``except (TypeError, KeyError)``
    branch did, contaminating ~17 v6.1 strategies). The forward path
    similarly raises on signature errors so feature flags such as
    ``hypernet_modulation`` aren't silently dropped. See
    TODO/audit/09_models_registry_generators.md §3.5.
    """

    #: ``model_kwargs`` keys the wrapper's strategy reads from the config itself; never forwarded.
    STRATEGY_KEYS: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        spatial_dims: int = 2,
        backbone: str = "configurable_unet",
        _declared_keys: frozenset[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        kwargs = {k: v for k, v in kwargs.items() if k not in self.STRATEGY_KEYS}
        self.backbone = _build_backbone(
            backbone,
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=spatial_dims,
            kwargs=kwargs,
            declared=_declared_keys,
            owner=type(self).__name__,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # Fail loud on signature mismatch instead of silently dropping
        # *args/**kwargs (which masks contrast-conditioning, time-step
        # injection, and other feature flags).
        return self.backbone(x, *args, **kwargs)


# ============================================================
# Learnable / active acquisition (PR-11 / PR-25)
# ============================================================


@register_model(
    name="loupe",
    training_mode="learnable_acquisition",
    spatial_dims=(2,),
    # The backbone sees the pre_model image the strategy undersamples, not k-space.
    input_domain="image",
    output_domain="image",
    accepts_complex=True,
    requires_paired_data=True,
)
class LOUPEModel(_BackboneAlias):
    """LOUPE wrapper — strategy injects the learnable mask layer."""

    STRATEGY_KEYS = frozenset({"share_across_coils"})


@register_model(
    name="pilot",
    training_mode="learnable_acquisition",
    spatial_dims=(2,),
    # The backbone sees the pre_model image the strategy undersamples, not k-space.
    input_domain="image",
    output_domain="image",
    accepts_complex=True,
    requires_paired_data=True,
)
class PILOTModel(_BackboneAlias):
    """PILOT wrapper — strategy injects the learnable trajectory."""

    STRATEGY_KEYS = frozenset({"n_shots", "samples_per_shot"})


@register_model(
    name="bald_active_recon",
    training_mode="active_acquisition",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="image",
    requires_paired_data=True,
)
class BALDActiveReconModel(_BackboneAlias):
    """BALD wrapper: MC dropout on the backbone's feature maps; the strategy drives the loop.

    ``mc_dropout_p`` adds channel dropout to the output of every parametrised backbone
    stage except the last (the output head), active in train mode and while
    ``mc_active`` is set (Gal & Ghahramani 2016), so posterior sampling can run in eval
    mode without touching normalisation statistics. Without it the samples BALD
    compares are identical and every line scores zero.
    """

    STRATEGY_KEYS = frozenset({"n_posterior_samples"})

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        spatial_dims: int = 2,
        backbone: str = "configurable_unet",
        _declared_keys: frozenset[str] | None = None,
        mc_dropout_p: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_channels, out_channels, spatial_dims, backbone, _declared_keys, **kwargs
        )
        if not 0.0 <= float(mc_dropout_p) < 1.0:
            raise ValueError(f"mc_dropout_p must lie in [0, 1), got {mc_dropout_p}")
        self.mc_dropout_p = float(mc_dropout_p)
        self.mc_active = False
        self.mc_dropout_stages: list[str] = []
        if self.mc_dropout_p == 0.0:
            return
        stages = [
            (n, m)
            for n, m in self.backbone.named_children()
            if next(m.parameters(), None) is not None
        ]
        for name, module in stages[:-1]:
            module.register_forward_hook(self._feature_dropout)
            self.mc_dropout_stages.append(name)
        if not self.mc_dropout_stages:
            raise ValueError(f"backbone {backbone!r} has no stage before its head to drop out")

    def _feature_dropout(self, _module: nn.Module, _inputs: Any, output: Any) -> Any:
        if not isinstance(output, torch.Tensor) or output.dim() < 3:
            return output
        drop = nn.functional.dropout2d if output.dim() == 4 else nn.functional.dropout
        return drop(output, self.mc_dropout_p, self.training or self.mc_active)


# ============================================================
# Multi-contrast wrappers (Items 1-4)
# ============================================================


# contrast_agnostic_encoder lives in models/generators/contrast_agnostic_encoder.py.


# frozen_backbone_with_head lives in models/generators/frozen_backbone_with_head.py.


# NOTE (audit 2026-05-28): latent_diffusion_residual_head is NOT stubbed here.
# The real implementation lives in
# spectramr/models/generators/latent_diffusion_with_residual_head.py
# (LatentDiffusionWithResidualHead) and registers the same name; it is
# auto-imported via the `spectramr.models.generators` walk, which runs before
# this module. Keeping the old _BackboneAlias placeholder double-registered the
# name and tripped the fail-loud collision guard — and because that guard
# RAISES, it aborted the import of this entire module, silently dropping every
# registration below it (latent_diffusion_multi_axis_cfg, vendor_aware_translation,
# multi_param_mapping, cross_field_fm, ...). Same fix as the cdiffmr/shen2024/fdb
# baselines noted further down.


# latent_diffusion_multi_axis_cfg lives in models/generators/latent_diffusion_multi_axis_cfg.py.


# vendor_aware_translation lives in models/generators/vendor_aware_translation.py.


# cdiffmr_baseline / shen2024_baseline / fdb_baseline are registered by the out-of-tree
# ``spectramr_baselines`` plugin; a placeholder here would trip the duplicate guard.


# ============================================================
# Multi-parameter mapping + cross-field FM (ideas 2, 5, 10)
# ============================================================


# multi_param_mapping lives in models/generators/multi_param_mapping.py.


# cross_field_fm lives in models/generators/cross_field_fm.py.


@register_model(
    name="scas",
    training_mode="learnable_acquisition",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="image",
)
class SCASModel(_BackboneAlias):
    """Alias for the scout-conditioned adaptive sampler."""


@register_model(
    name="diffusion",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
)
class GenericDiffusion(_BackboneAlias):
    """Generic diffusion alias for EDM and similar variants."""


__all__ = [
    "BALDActiveReconModel",
    "GenericDiffusion",
    "LOUPEModel",
    "PILOTModel",
    "SCASModel",
]
