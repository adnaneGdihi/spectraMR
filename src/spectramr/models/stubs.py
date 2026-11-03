"""Model Alias Registry
======================

Registers secondary names for models that already have real implementations
but are referenced in experiment YAMLs under a different ``model_type`` key.

This module is imported as the **final step** of ``populate_model_registry()``
in ``init_registry.py``.  It never creates stub/placeholder classes — every
entry here points to a genuine, architecturally-correct ``nn.Module``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register_aliases() -> None:
    """Add secondary registration names for existing model implementations."""
    from spectramr.models.registry import MODEL_REGISTRY

    # -----------------------------------------------------------------
    # Generator filename → registered-as → experiment-expects
    # Each setdefault ensures we never overwrite a real registration.
    # -----------------------------------------------------------------

    # --- Generators registered under a shorter canonical name ----------

    # coil_sensitivity_network.py → "coil_sensitivity"
    if "coil_sensitivity" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("coil_sensitivity_network", MODEL_REGISTRY["coil_sensitivity"])

    # conditional_diffusion_generator.py → "conditional_diffusion"
    if "conditional_diffusion" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault(
            "conditional_diffusion_generator", MODEL_REGISTRY["conditional_diffusion"]
        )

    # continual_learning_unet.py → "continual_learning_unet"
    if "continual_learning_unet" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("continual_learning", MODEL_REGISTRY["continual_learning_unet"])

    # cycle_gan.py → "cyclegan_generator"
    if "cyclegan_generator" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("cycle_gan", MODEL_REGISTRY["cyclegan_generator"])

    # edsr_generator.py → "edsr"
    if "edsr" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("edsr_generator", MODEL_REGISTRY["edsr"])

    # elan_generator.py → "elan"
    if "elan" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("elan_generator", MODEL_REGISTRY["elan"])

    # equivariant_diffusion_generator.py → "equivariant_diffusion"
    if "equivariant_diffusion" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault(
            "equivariant_diffusion_generator", MODEL_REGISTRY["equivariant_diffusion"]
        )

    # foundation_model.py → "mamba" (also needs "foundation_model" name)
    if "mamba" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("foundation_model", MODEL_REGISTRY["mamba"])

    # gflownet_generator.py → "gflownet"
    if "gflownet" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("gflownet_generator", MODEL_REGISTRY["gflownet"])

    # han_generator.py → "han"
    if "han" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("han_generator", MODEL_REGISTRY["han"])

    # hobs_generator.py → "hobs"
    if "hobs" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("hobs_generator", MODEL_REGISTRY["hobs"])

    # hypernetwork_generator.py → "hypernetwork"
    if "hypernetwork" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("hypernetwork_generator", MODEL_REGISTRY["hypernetwork"])

    # kspace_cold_diffusion_generator.py → "kspace_cold_diffusion"
    if "kspace_cold_diffusion" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault(
            "kspace_cold_diffusion_generator", MODEL_REGISTRY["kspace_cold_diffusion"]
        )

    # lans_generator.py → "lans"
    if "lans" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("lans_generator", MODEL_REGISTRY["lans"])

    # latent_diffusion_generator.py → "latent_gan_generator"
    if "latent_gan_generator" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("latent_diffusion", MODEL_REGISTRY["latent_gan_generator"])

    # multicontrast_fusion_generator.py → "multicontrast_fusion"
    if "multicontrast_fusion" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault(
            "multi_contrast_fusion_generator", MODEL_REGISTRY["multicontrast_fusion"]
        )

    # nafnet_generator.py → "nafnet"
    if "nafnet" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("nafnet_generator", MODEL_REGISTRY["nafnet"])

    # nca_generator.py → "nca"
    if "nca" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("nca_generator", MODEL_REGISTRY["nca"])

    # neural_ode_generator.py → "neural_ode"
    if "neural_ode" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("neural_ode_generator", MODEL_REGISTRY["neural_ode"])

    # rcan_generator.py → "rcan"
    if "rcan" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("rcan_generator", MODEL_REGISTRY["rcan"])

    # rician_diffusion_generator.py → "rician_diffusion"
    if "rician_diffusion" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("rician_diffusion_generator", MODEL_REGISTRY["rician_diffusion"])

    # score_based_diffusion_generator.py → "score_based_diffusion"
    if "score_based_diffusion" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault(
            "score_based_diffusion_generator", MODEL_REGISTRY["score_based_diffusion"]
        )
        MODEL_REGISTRY.setdefault("score_based_unet", MODEL_REGISTRY["score_based_diffusion"])

    # slice_to_volume_generator.py → "slice_to_volume"
    if "slice_to_volume" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("slice_to_volume_generator", MODEL_REGISTRY["slice_to_volume"])

    # swin_transformer_generator.py → "transformer_unet"
    if "transformer_unet" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("swin_transformer", MODEL_REGISTRY["transformer_unet"])

    # swinir_generator.py → "swinir"
    if "swinir" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("swinir_generator", MODEL_REGISTRY["swinir"])
        MODEL_REGISTRY.setdefault("swin_ir", MODEL_REGISTRY["swinir"])

    # vnet_generator.py → "vnet"
    if "vnet" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("vnet_generator", MODEL_REGISTRY["vnet"])

    # world_model.py → "generative_world_model"
    if "generative_world_model" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("world_model", MODEL_REGISTRY["generative_world_model"])

    # --- Non-generator modules that need IGenerator wrappers -----------

    # uncertainty_wrapper, pin_nerf, monai_swin_unetr, qmri_network, dvs, diffusion_recon,
    # implicit_representation and latent_diffusion_hierarchical are NOT aliased here: each
    # has its own @register_model class, so a setdefault would never bind.

    # `rl_scanner` is the YAML alias for `rl_mri` (ActiveScanner) — same
    # active-acquisition family.
    #
    # NB: `rl_acquisition_agent` and `surgery_loop_agent` are NOT aliases
    # for `rl_mri`. They are distinct classes that self-register via
    # `@register_model` from `src/models/generators/` (relocated
    # 2026-05-15 from `src/application/agents/` per D14 to close the
    # last layer-direction violation). The previous
    # `setdefault(..., rl_mri)` aliases silently remapped the missing
    # agents to an unrelated `ActiveScanner` model — see
    # `TODO/audit/02_application.md` F2 (CLAUDE.md pitfall #9).
    if "rl_mri" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("rl_scanner", MODEL_REGISTRY["rl_mri"])

    # physics_driven → pinn. UnrolledPhysicsNetwork is an unrolled network, not a PINN; the
    # name is kept because VALID_MODEL_TYPES and a campaign arm still use it.
    if "physics_driven" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("pinn", MODEL_REGISTRY["physics_driven"])

    # neural_cache → neural_ca. A neural cache, not a neural cellular automaton; kept because
    # experiment_88_neural_ca and VALID_MODEL_TYPES name it.
    if "neural_cache" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("neural_ca", MODEL_REGISTRY["neural_cache"])

    # configurable_unet → standard_unet, enhanced_deep_unet  (2026-05-04 cleanup)
    # Previously these were two empty ``class X(UNet): pass`` subclasses with
    # their own ``@register_model`` decorators in
    # ``src/models/reconstruction/unet.py``. Consolidated here so the alias
    # mechanism is uniform; the Python class names live as module-level
    # ``X = UNet`` assignments in unet.py for the 41+ direct importers.
    if "configurable_unet" in MODEL_REGISTRY:
        MODEL_REGISTRY.setdefault("standard_unet", MODEL_REGISTRY["configurable_unet"])
        MODEL_REGISTRY.setdefault("enhanced_deep_unet", MODEL_REGISTRY["configurable_unet"])
        # `enhanced_unet` joins them, and is a PLAIN ASSIGNMENT, not setdefault
        # (#977). Until this change the squeeze-excitation `EnhancedUNet` in
        # generators/convolution_variants.py claimed the name here via
        # @register_model, while ModelFactory -- the only build path -- bound it
        # to the configurable UNet. 11 arms declared it and got the configurable
        # UNet; MODEL_REGISTRY told every other consumer, including
        # infer_output_domain's capability lookup, that they had the SE variant.
        #
        # setdefault would reintroduce exactly that: it makes the binding
        # conditional on import order, so a future @register_model claim on this
        # name would silently win here and still lose in the factory. The two
        # registries now agree on `enhanced_unet` unconditionally.
        MODEL_REGISTRY["enhanced_unet"] = MODEL_REGISTRY["configurable_unet"]

    # -----------------------------------------------------------------
    # Wave-2 ledger restoration (2026-05-22) — re-add the 20 name-variant
    # aliases that were purged from VALID_MODEL_TYPES in commit d8ccb8452.
    # Each was a duplicate string for an already-registered canonical
    # model; restoring the alias makes the advertised name resolve again
    # (closing the "audit-surface lie" without re-introducing a stub).
    # See TODO/deleted_model_types_reimplementation_plan.md Phase 1.
    # `kan` and `trellis_structured_vae` are intentionally NOT restored
    # (bare-ambiguous token / abstract base — see plan §1 footnotes).
    # ALIAS_TABLE in tests/unit/models/test_registry_aliases.py mirrors
    # this block and regresses if an entry is dropped.
    # -----------------------------------------------------------------
    _wave2_aliases = {
        "MambaReconstruction": "mamba_reconstruction",
        "capsule_networks": "capsule_network",
        "kspace_gpt_foundation": "kspace_gpt",
        "mixture_density": "mixture_density_network",
        "moe_kan": "moe_kan_generator",
        "nafnets": "nafnet",
        "normalizing_flow_image": "normalizing_flow",
        "reversible_network": "invertible_network",
        "standard_vit": "vision_transformer",
        "stylegan": "stylegan2",
        "swin_transformer_kan": "swin_kan_transformer",
        "trellis_gaussian": "trellis_gaussian_vae",
        "trellis_mesh": "trellis_mesh_vae",
        "unrolled_list": "unrolled_reconstruction",
        "vit": "vision_transformer",
        "vit_with_kan": "vit_kan",
        "vq_vae": "vqvae",
        # DELETED 2026-08-12 (diffusion cleanup, phase 1.3):
        #   "conditional_diffusion_cfg": "conditional_diffusion",
        #   "multimodal_conditional_diffusion": "conditional_diffusion",
        # Exact dict-entry copies of ``conditional_diffusion`` (same class, same
        # mode) with zero arms in all 1508 committed YAMLs. Both names also
        # promise capability the alias cannot deliver — CFG and multimodal
        # conditioning are knobs on ConditionalDiffusionGenerator, not separate
        # models — so the name read as a selector and selected nothing.
        # ``conditional_diffusion`` and ``conditional_diffusion_generator`` remain.
        "neural_ode_adaptive": "neural_ode",
    }
    for _alias, _canonical in _wave2_aliases.items():
        if _canonical in MODEL_REGISTRY:
            MODEL_REGISTRY.setdefault(_alias, MODEL_REGISTRY[_canonical])

    logger.debug(
        "Model alias registration complete. Total registry size: %d",
        len(MODEL_REGISTRY),
    )


# NOT run on import -- ``populate_model_registry`` calls this directly.
#
# Every alias above is guarded by ``if <canonical> in MODEL_REGISTRY``, so the
# pass is order-dependent: run it before the generators have registered and the
# guards simply fail, skipping the aliases in silence. ``sys.modules`` caching
# then makes it unrepairable -- the import in ``init_registry`` becomes a no-op,
# so the pass never gets a second chance.
#
# Measured on this tree: a single early ``import spectramr.models.stubs`` dropped
# 53 of 586 model names -- ``standard_unet``, ``vit``, ``cycle_gan``, ``pinn``,
# ``vq_vae`` and ``latent_diffusion`` among them -- while
# ``config/validation_constants.py`` went on accepting every one of them, so a
# config naming one validated clean and then could not be built.
#
# The pass is idempotent (guarded ``setdefault`` throughout), so calling it
# again is harmless; calling it EARLY is not. That is why the import-time call
# is removed rather than kept alongside the explicit one -- two owners of the
# same invariant is non-negotiable 17, and here the weaker one is the defect.
