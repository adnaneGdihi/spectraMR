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

from typing import Any

import torch
import torch.nn as nn

from spectramr.models.registry import MODEL_REGISTRY, register_model


def _resolve_backbone(name: str | None, **kwargs: Any) -> nn.Module:
    """Instantiate a registered backbone by name with sensible defaults."""
    name = name or "configurable_unet"
    info = MODEL_REGISTRY.get(name)
    if info is None:
        raise KeyError(
            f"Cannot resolve backbone {name!r}: not registered. "
            f"Did you forget to call populate_model_registry()?"
        )
    return info["class"](**kwargs)


# ============================================================
# Generic v6.1 wrapper — used by most "thin pass-through" models
# ============================================================


class _BackboneAlias(nn.Module):
    """Generic forwarding wrapper. Strategy classes do the real work.

    Per CLAUDE.md #9 (silent fallbacks forbidden), unknown backbones and
    incompatible-kwarg errors raise instead of silently degrading to a
    3×3 conv (which is what the previous ``except (TypeError, KeyError)``
    branch did, contaminating ~17 v6.1 strategies). The forward path
    similarly raises on signature errors so feature flags such as
    ``hypernet_modulation`` aren't silently dropped. See
    TODO/audit/09_models_registry_generators.md §3.5.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        spatial_dims: int = 2,
        model_kwargs: dict[str, Any] | None = None,
        **legacy_kwargs: Any,
    ) -> None:
        super().__init__()
        kwargs = dict(model_kwargs or {})
        backbone_name = kwargs.pop("backbone", "configurable_unet")
        kwargs.setdefault("in_channels", in_channels)
        kwargs.setdefault("out_channels", out_channels)
        kwargs.setdefault("spatial_dims", spatial_dims)
        # Fail loud — the audit ladder needs to reject illegal YAML at
        # load time, not silently produce a no-op model at training time.
        self.backbone = _resolve_backbone(backbone_name, **kwargs)

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
    input_domain="kspace",
    output_domain="image",
    accepts_complex=True,
    requires_paired_data=True,
)
class LOUPEModel(_BackboneAlias):
    """LOUPE wrapper — strategy injects the learnable mask layer."""


@register_model(
    name="pilot",
    training_mode="learnable_acquisition",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="image",
    accepts_complex=True,
    requires_paired_data=True,
)
class PILOTModel(_BackboneAlias):
    """PILOT wrapper — strategy injects the learnable trajectory."""


@register_model(
    name="bald_active_recon",
    training_mode="active_acquisition",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="image",
    requires_paired_data=True,
)
class BALDActiveReconModel(_BackboneAlias):
    """BALD wrapper with MC-dropout — strategy drives the active loop."""


# ============================================================
# Multi-contrast wrappers (Items 1-4)
# ============================================================


@register_model(
    name="contrast_agnostic_encoder",
    training_mode="ssl",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    requires_paired_data=False,
)
class ContrastAgnosticEncoder(_BackboneAlias):
    """SSL backbone — projection head emerges from ``model_kwargs``."""


@register_model(
    name="frozen_backbone_with_head",
    training_mode="gan",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    supports_contrast_conditioning=True,
    requires_paired_data=False,
)
class FrozenBackboneWithHead(_BackboneAlias):
    """Stage-2 frozen-backbone wrapper. Strategy + adapter freeze logic
    live in :mod:`spectramr.models.blocks.frozen_backbone_adapter` (TBD)."""


# NOTE (audit 2026-05-28): latent_diffusion_residual_head is NOT stubbed here.
# The real implementation lives in
# spectramr/models/generators/latent_diffusion_with_residual_head.py
# (LatentDiffusionWithResidualHead) and registers the same name; it is
# auto-imported via the `spectramr.models.generators` walk, which runs before
# this module. Keeping the old _BackboneAlias placeholder double-registered the
# name and tripped the fail-loud collision guard — and because that guard
# RAISES, it aborted the import of this entire module, silently dropping every
# registration below it (latent_diffusion_multi_axis_cfg, vendor_aware_translation,
# the baselines, mrf_dictionary_matching, federated_dp_recon, ...). Same fix as
# the cdiffmr/shen2024/fdb baselines noted further down.


@register_model(
    name="latent_diffusion_multi_axis_cfg",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    supports_contrast_conditioning=True,
)
class LatentDiffusionMultiAxisCFG(_BackboneAlias):
    """Latent diffusion with multi-axis CFG schedule."""


@register_model(
    name="vendor_aware_translation",
    training_mode="gan",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    supports_contrast_conditioning=True,
)
class VendorAwareTranslation(_BackboneAlias):
    """Vendor-conditioned generator with soft-prompt embedding."""


# ============================================================
# Cold-diffusion baseline replications (Phases B-D)
# ============================================================


# NOTE: cdiffmr_baseline / shen2024_baseline / fdb_baseline are NOT stubbed
# here: the out-of-tree ``spectramr_baselines`` plugin registers them, and a
# placeholder would collide with it on the fail-loud duplicate guard.
# filterdiff_baseline / amk_cdiffnet_baseline below remain placeholders
# (no upstream implementation yet).


@register_model(
    name="filterdiff_baseline",
    training_mode="cold_diffusion",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="image",
    accepts_complex=True,
)
class FilterDiffBaseline(_BackboneAlias):
    """FilterDiff (Song 2025) baseline placeholder — Phase E. Conditional on upstream code release."""


@register_model(
    name="amk_cdiffnet_baseline",
    training_mode="cold_diffusion",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="image",
    accepts_complex=True,
)
class AMKCDiffNetBaseline(_BackboneAlias):
    """AMK-CDiffNet (Dong 2026) baseline placeholder — Phase F. Conditional on upstream code availability."""


# ============================================================
# Multi-parameter mapping + cross-field FM (ideas 2, 5, 10)
# ============================================================


@register_model(
    name="multi_param_mapping",
    training_mode="multi_param_mapping",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
)
class MultiParamMapping(_BackboneAlias):
    """One-shot multi-parameter (T1/T2/PD/seg) mapping head."""

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        feat = super().forward(x, *args, **kwargs)
        # Lazy split into per-parameter heads — channels are interpreted
        # as (t1, t2, pd, seg). Real implementations will replace this
        # with a proper MultiParameterHead under spectramr.models.multi_param/.
        if feat.shape[1] >= 4:
            return {
                "t1": feat[:, 0:1],
                "t2": feat[:, 1:2],
                "pd": feat[:, 2:3],
                "segmentation": feat[:, 3:],
            }
        return {"t1": feat}


@register_model(
    name="mrf_dictionary_matching",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
)
class MRFDictionaryMatching(_BackboneAlias):
    """Gold-standard MRF lookup. Real impl uses
    ``spectramr.infrastructure.physics.mrf_dictionary``; this wrapper just
    keeps the registry honest for campaigns referencing the model."""


@register_model(
    name="cross_field_hypernet",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="image",
)
class CrossFieldHypernet(_BackboneAlias):
    """X-Field-FM hypernetwork wrapper."""


@register_model(
    name="cross_field_fm",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="image",
)
class CrossFieldFM(_BackboneAlias):
    """Alias for the X-Field-FM model."""


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


@register_model(
    name="scout_conditioned_sampler_hypernet",
    training_mode="learnable_acquisition",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="image",
)
class ScoutConditionedSamplerHypernet(_BackboneAlias):
    """SCAS hypernetwork — predicts per-sample mask logits from a scout."""


# ============================================================
# Federated wrapper for idea 8
# ============================================================


@register_model(
    name="federated_dp_recon",
    training_mode="federated",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="image",
)
class FederatedDPRecon(_BackboneAlias):
    """Federated DP-conformal wrapper (the strategy carries the DP-SGD logic)."""


__all__ = [
    "BALDActiveReconModel",
    "ContrastAgnosticEncoder",
    "CrossFieldHypernet",
    "FederatedDPRecon",
    "FrozenBackboneWithHead",
    "LOUPEModel",
    "LatentDiffusionMultiAxisCFG",
    "MRFDictionaryMatching",
    "MultiParamMapping",
    "PILOTModel",
    "ScoutConditionedSamplerHypernet",
    "VendorAwareTranslation",
]
