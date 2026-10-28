"""Model Discovery Initialization
==============================

This module is responsible for eagerly importing model modules to ensure
they are registered in the global MODEL_REGISTRY via the @register_model decorator.
This avoids circular dependencies in spectramr.models.registry by moving the discovery
logic to a separate entry point.
"""

import importlib
import logging
import pkgutil
import sys

logger = logging.getLogger(__name__)

# Idempotency guard for the full pkgutil discovery walk. The walk is
# expensive on shared cluster filesystems (metadata-latency tax over the
# whole model tree) and is invoked on every config load via
# ``TrainingSettings.from_yaml``. It is also *not* safely replaceable by
# a "registry is non-empty" check: ``@register_model`` decorators fire
# eagerly as model packages are imported (populating a partial subset),
# while ~200 models AND all the compatibility aliases below register
# ONLY through this walk. So we gate on a flag the walk itself owns:
# the walk runs at most once, but always at least once.
_REGISTRY_POPULATED = False

# module path -> repr of the exception that stopped it importing during the walk.
# A module that fails to import contributes ZERO models, so every @register_model
# inside it silently vanishes from the catalog. That used to be logged at DEBUG,
# which made the downstream symptom unreadable: `TrainingSettings.from_yaml` would
# reject a perfectly valid `model_type` with "Invalid model_type: ... Must be one of
# the registered models", naming a model that exists, is decorated, and is simply
# absent because an optional dependency (torchvision) was missing from the env.
# Keep the failures so the model_type error can say what actually happened.
_IMPORT_FAILURES: dict[str, str] = {}


def _record_import_failure(module_name: str, exc: BaseException) -> None:
    """Record + WARN. A dropped module is a silently shrunken catalog, not a detail.

    A ``PluginImportError`` is re-raised instead of recorded. The discovery seams
    (``core/metrics/__init__``, ``models/losses/__init__``) call
    ``discover_plugins`` at module-import time, so a user's broken
    ``SPECTRAMR_PLUGINS`` token surfaces INSIDE this walk's ``except`` -- and got
    reported as "every @register_model in spectramr.models.generators is MISSING",
    naming an in-tree package that is perfectly fine. Two different faults with
    two different fixes: a dropped in-tree module means a missing optional
    dependency; a PluginImportError means the user's own plugin is broken. The
    re-raise lives here rather than at the five call sites so the rule has one
    owner (CLAUDE.md non-negotiable 17).
    """
    from spectramr.plugins import PluginImportError

    if isinstance(exc, PluginImportError):
        raise exc

    _IMPORT_FAILURES[module_name] = f"{type(exc).__name__}: {exc}"
    logger.warning(
        "Model discovery could not import %s (%s: %s) — every @register_model in it "
        "is MISSING from the catalog. If a config names one of those models, it will "
        "be rejected as an unknown model_type.",
        module_name,
        type(exc).__name__,
        exc,
    )


def _on_walk_package_error(package_name: str) -> None:
    """Record a sub-package that ``pkgutil`` could not import while recursing.

    A named function rather than a lambda so the error path can actually be
    executed by a test. pkgutil calls this with the name only -- a callback
    taking two arguments would ``TypeError`` here and nowhere else, i.e. a crash
    no green run would ever surface.
    """
    _record_import_failure(package_name, sys.exc_info()[1] or RuntimeError("unknown walk error"))


def get_registry_import_failures() -> dict[str, str]:
    """Modules that failed to import during the discovery walk, module -> error.

    Non-empty means the model catalog is INCOMPLETE: any ``@register_model`` in a
    listed module did not fire, so a valid ``model_type`` may look unregistered.
    """
    return dict(_IMPORT_FAILURES)


def _discover_model_plugins() -> None:
    """Fire ``@register_model`` decorators from out-of-tree plugin modules.

    The ONLY seam by which a model defined outside the spectramr tree joins
    ``MODEL_REGISTRY``. Called on every ``populate_model_registry`` -- including
    the early-return path -- because whether a plugin is discoverable must not
    depend on whether something else already triggered the walk.
    """
    from spectramr.plugins import discover_plugins

    discover_plugins("spectramr.models")


def populate_model_registry(*, force: bool = False) -> None:
    """Import core model modules to trigger registration dynamically using pkgutil.

    Idempotent: the underlying ``pkgutil.walk_packages`` discovery walk
    (plus alias/stub registration) runs only on the first call and is a
    cheap no-op thereafter. Pass ``force=True`` to re-run the walk (e.g.
    in tests that reset the registry).
    """
    global _REGISTRY_POPULATED
    if _REGISTRY_POPULATED and not force:
        # The expensive pkgutil WALK is done -- but plugin discovery is not part
        # of what "done" means here, and must still run.
        #
        # Discovery used to sit only at the END of this function, below this
        # early return, so it was reachable on the FIRST call and never again.
        # Anything that loaded a config first (``settings_from_dict`` calls this
        # to validate ``model_type``) marked the registry populated, and a later
        # ``SPECTRAMR_PLUGINS`` was then never read: a long-lived process -- a
        # notebook, a server, a campaign runner -- silently never discovered the
        # plugin. It also made `test_env_plugin_model_resolves_in_config` pass
        # alone and fail in the suite, which is how the ordering surfaced (#1637).
        #
        # ``discover_plugins`` carries its own idempotency (``_discovered_groups``
        # per entry-point group, ``_imported_paths`` per module), so calling it
        # every time costs a set lookup per token.
        _discover_model_plugins()
        return

    # List of packages to scan for model implementations
    model_packages = [
        "spectramr.models.geometric",
        "spectramr.models.vae",
        "spectramr.models.discriminators",
        "spectramr.models.generators",
        "spectramr.models.inr",
        "spectramr.models.latent",
        "spectramr.models.reconstruction",
        "spectramr.models.mae",
        "spectramr.models.trellis",
        "spectramr.models.diffusion",
        "spectramr.models.experimental",
        # Added 2026-05-04: these packages have @register_model decorators that
        # were never firing because they weren't in the discovery list.
        # `vq` registers `vqvae`/`vqgan` (referenced by experiments/inprogress/dummy/dummy_vqvae.yaml,
        # vae_latent/experiment_54_hierarchical_vq_vae.yaml, training/generation/vqvae.yaml, ...).
        # `meta_learning` registers `meta_varnet` (training/umr/exp_meta_varnet.yaml, fewshot/exp_meta_varnet_1shot.yaml).
        # NB: `specialized` was here too (registered `low_field_augmented`)
        # but had zero YAML/test/code references — deleted 2026-05-15 per
        # TODO/audit/18b_exotic_subdirs.md C2.
        "spectramr.models.vq",
        "spectramr.models.meta_learning",
        # transport: registers `velocity_network`, `kidot_transport` (optimal-transport models for unpaired MRI translation)
        "spectramr.models.transport",
        # Added 2026-05-09 per TODO/audit/10_models_blocks_layers.md F1+F2:
        # `transformer` registers `recon_former` (decorator was dead).
        # `encoders` registers `kan_encoder` and `medical_dino` (encoder
        # variant). Without these in the walk list those decorators never
        # fired and the registry surface lied about what was available.
        "spectramr.models.transformer",
        "spectramr.models.encoders",
        # Added 2026-08-07 (contrast/field-agnostic bundle M4): `physics_ae`
        # registers `disp_bloch_ae`. Without this entry the decorator never
        # fires during population and the arm is rejected at load time as an
        # unknown model_type — advertised but un-resolvable, the same failure
        # the entries below were added to fix.
        "spectramr.models.physics_ae",
        # Added 2026-05-15 per TODO/audit/18a_meta_unrolled_gans_etc.md F3:
        # `gans` registers 9 models (stylegan2, stylegan_discriminator,
        # stylegan_kan_gen, swin_kan_transformer, swin_kan_generator,
        # style_kan_gan, kan_gan, kan_discriminator, vit_kan_generator).
        # Before this entry, only the last 3 fired (transitively via
        # generators/__init__.py); the other 6 decorators were dead.
        # NB: `standard_gan.py::PhysicsInformedUNet` had a broken import
        # (referenced a long-removed `PhysicsInformedDataConsistency`)
        # and was deleted as part of the same triage — the
        # `physics_informed_unet` name had no consumers in experiments/.
        "spectramr.models.gans",
        # Added 2026-05-22: packages whose @register_model decorators never
        # fired during population, so the names were advertised (schemas /
        # validation_constants) but un-resolvable at build time. The framework
        # dispatches models dynamically by name from YAML config, so an
        # un-registered-but-advertised model is a latent KeyError, not dead
        # code. Wiring the packages in makes the registry surface honest.
        #   gaussian_splatting → sugar_sdf, gs_mri
        #   adaptation         → latent_aligner (federated.py)
        #   components         → conditioning_encoder (listed in validation_constants)
        #   specialized        → low_field_augmented (re-added; the 2026-05-15
        #                        eviction used static-zero-reference as the
        #                        criterion, which does not hold for a
        #                        config-dispatched registry).
        "spectramr.models.gaussian_splatting",
        "spectramr.models.adaptation",
        "spectramr.models.components",
        "spectramr.models.specialized",
        # Added 2026-05-28 per audit finding H6: `latent_diffusion`
        # (latent_discriminator.py) carries three @register_model
        # decorators — `ldm_latent_discriminator`,
        # `ldm_patch_latent_discriminator`,
        # `ldm_multiscale_latent_discriminator` — but the package was
        # never in this walk list, so the decorators never fired even
        # though `ldm_latent_discriminator` is advertised as a valid
        # model_type in config/validation_constants.py. The framework
        # dispatches models dynamically by name from YAML, so an
        # advertised-but-unregistered name is a latent KeyError at build
        # time, not dead code. NB: latent_diffusion/__init__.py keeps
        # latent_training_strategies.py commented out, so only the
        # discriminator + latent_unet modules carry live decorators.
        "spectramr.models.latent_diffusion",
        # VF campaign Phase 2 breakthrough-method model homes.
        "spectramr.models.physics",
        "spectramr.models.navigation",
        "spectramr.models.acquisition",
        # [Proposal 1] operator-identification conditioner
        # (`bch_operator_conditioner`). Without this entry the
        # @register_model decorator in operator_id/bch_conditioner.py never
        # fires and the YAML's model_type validation would reject it.
        "spectramr.models.operator_id",
        # Added 2026-07-02 (sota_registry retirement): these packages used to
        # be imported explicitly by models/sota_registry.py, which then fired
        # ~41 dynamic ``register_model(name, mode)(Class)`` calls. That module
        # is now a deprecated no-op shim; every SOTA model carries its own
        # per-class ``@register_model`` decorator and registers through this
        # walk instead. ``tests/unit/models/test_registry_census.py`` pins the
        # full roster with exact (name, mode), so dropping any package below
        # is a hard CI failure, not a silent catalog loss.
        "spectramr.models.edge",
        "spectramr.models.equilibrium",
        "spectramr.models.fingerprinting",
        "spectramr.models.foundation",
        "spectramr.models.generative",
        "spectramr.models.mamba",
        "spectramr.models.motion",
        "spectramr.models.reasoning",
        "spectramr.models.spectral",
        "spectramr.models.synthesis",
        "spectramr.models.topological",
        # ``unrolled`` was imported ONLY by sota_registry (neural_ode_recon);
        # without this entry its decorator would never fire post-retirement.
        "spectramr.models.unrolled",
        "spectramr.models.volumetric",
    ]

    if force:
        _IMPORT_FAILURES.clear()

    for package_name in model_packages:
        try:
            package = importlib.import_module(package_name)
            if hasattr(package, "__path__"):
                # ``onerror`` is not optional. walk_packages recurses into a
                # sub-package by IMPORTING it, inside pkgutil -- so the
                # ``except`` below never sees that failure, and pkgutil's
                # default discards it and abandons the whole sub-tree. Every
                # listed package is flat today, which makes the hole latent
                # rather than absent: the first nested package added here
                # would drop its models with no error and exit 0.
                for _, module_name, is_pkg in pkgutil.walk_packages(
                    package.__path__,
                    package.__name__ + ".",
                    onerror=_on_walk_package_error,
                ):
                    if not is_pkg:
                        try:
                            importlib.import_module(module_name)
                        except Exception as e:
                            _record_import_failure(module_name, e)
        except ImportError as e:
            _record_import_failure(package_name, e)

    # Top-level model modules (single files directly under spectramr.models, not
    # inside any walked sub-package) whose @register_model decorators must fire
    # for the names to be resolvable from config. Added 2026-05-22.
    #   model_compression    → compression_efficient_unet
    #   latent_input_wrapper → latent_input_wrapper
    #   domain_adaptation    → domain_discriminator, cross_scanner_adaptation,
    #                          multi_domain_extractor
    for top_level_module in (
        "spectramr.models.model_compression",
        "spectramr.models.latent_input_wrapper",
        "spectramr.models.domain_adaptation",
    ):
        try:
            importlib.import_module(top_level_module)
        except Exception as e:
            _record_import_failure(top_level_module, e)

    # v6.1 — explicit single-module registration of the
    # paradigm-expansion wrappers (LOUPE / PILOT / BALD / multi-contrast
    # SSL / cold-diffusion baselines / multi-parameter mapping / etc.).
    try:
        importlib.import_module("spectramr.models._v6_1_registrations")
    except Exception as e:  # pragma: no cover - fail-loud at audit time
        _record_import_failure("spectramr.models._v6_1_registrations", e)

    # NOTE (2026-07-02): the ``import spectramr.models.sota_registry`` step that
    # used to live here is gone. sota_registry is a deprecated no-op shim; all
    # former SOTA registrations are per-class @register_model decorators fired
    # by the package walk above (see the dated walk-list block). The registry
    # census test pins the roster so a regression here fails CI loudly.

    # Register aliases for cluster configuration compatibility
    from spectramr.models.registry import MODEL_REGISTRY

    if "mae_mri" in MODEL_REGISTRY and "mae" not in MODEL_REGISTRY:
        MODEL_REGISTRY["mae"] = MODEL_REGISTRY["mae_mri"]

    # DELETED 2026-08-12 (diffusion cleanup, phase 1.3) — two ZeroRF names were
    # aliased onto RectifiedFlowGenerator, which is a different model:
    #
    #   MODEL_REGISTRY["zerorf"]             = MODEL_REGISTRY["rectified_flow"]
    #   MODEL_REGISTRY["zero_shot_transfer"] = MODEL_REGISTRY["rectified_flow"]
    #
    # ``zerorf`` was dead-but-armed: ``model_factory`` registers it to the real
    # ``ZeroRFReconstructor`` first, so the ``not in MODEL_REGISTRY`` guard always
    # short-circuited — but any change to import order would have silently swapped
    # the model under ``experiment_88_zero_shot_transfer.yaml`` (which declares
    # ``model_type: zerorf``) with no error. ``zero_shot_transfer`` DID fire, so
    # the same name meant ZeroRFReconstructor in the factory registry and
    # RectifiedFlowGenerator here. Zero arms declared it; ``zerorf`` and
    # ``rectified_flow`` both keep their own correct bindings.
    #
    # ``sde_diffusion`` (alias of ``score_based_diffusion``, same class, same mode)
    # went with them: zero arms in all 1508 committed YAMLs, and
    # ``experiment_96_sde_diffusion.yaml`` — the one arm the name was presumably
    # coined for — declares the canonical ``score_based_diffusion``.

    if "physics_driven" in MODEL_REGISTRY and "pinn_mri" not in MODEL_REGISTRY:
        # Alias generic PINN request to Physics Driven Network (Experiment 3.1)
        MODEL_REGISTRY["pinn_mri"] = MODEL_REGISTRY["physics_driven"]
        MODEL_REGISTRY["pinn"] = MODEL_REGISTRY["physics_driven"]

    if "kspace_gpt" in MODEL_REGISTRY and "causal_transformer" not in MODEL_REGISTRY:
        MODEL_REGISTRY["causal_transformer"] = MODEL_REGISTRY["kspace_gpt"]

    # Alias for consistency
    if "hybrid_qmap_generator" in MODEL_REGISTRY and "hybrid_pinn" not in MODEL_REGISTRY:
        MODEL_REGISTRY["hybrid_pinn"] = MODEL_REGISTRY["hybrid_qmap_generator"]

    if "patchgan_discriminator" in MODEL_REGISTRY and "patch_gan" not in MODEL_REGISTRY:
        MODEL_REGISTRY["patch_gan"] = MODEL_REGISTRY["patchgan_discriminator"]

    if "physics_vae" in MODEL_REGISTRY and "disentangled_vae" not in MODEL_REGISTRY:
        MODEL_REGISTRY["disentangled_vae"] = MODEL_REGISTRY["physics_vae"]

    # SOTA aliases (2026-07-02, sota_registry retirement) — one-class-many-names
    # cases from the retired sota_registry:
    #   vision_mamba → D2Mamba, canonically ``d2_mamba`` (mamba/d2_mamba.py)
    #
    # The stated rationale for this block used to be "a class may carry only ONE
    # @register_model decorator". That is FALSE and was never true: decorators
    # stack, and ``GraphUNetGenerator`` carries two on purpose
    # (``graph_unet``/reconstruction and ``graph_unet_diffusion``/diffusion, which
    # is why those two are NOT interchangeable). A second decorator is the better
    # home for any name that needs its own capability declaration; this block is
    # only for names that must be an exact dict-entry copy of another key.
    if "d2_mamba" in MODEL_REGISTRY and "vision_mamba" not in MODEL_REGISTRY:
        MODEL_REGISTRY["vision_mamba"] = MODEL_REGISTRY["d2_mamba"]

    # DELETED 2026-08-12 (diffusion cleanup, phase 1.3):
    #   MODEL_REGISTRY["metal_diffusion"] = MODEL_REGISTRY["cold_diffusion_process"]
    # A facade (pitfall #16): the name promises metal-artefact degradation, but an
    # alias only renames the class — the comment here conceded "config must set
    # degradation='metal'", and nothing did. Zero arms in all 1508 committed YAMLs.
    # ``cold_diffusion_process`` keeps ColdDiffusion reachable under its canonical
    # name. NB ``cold_diffusion`` (no suffix) is a DIFFERENT class —
    # ColdDiffusionGenerator in generators/ — do not conflate.

    # Final step: apply all stub aliases & fill remaining gaps.
    #
    # Called, not merely imported. The aliases are guarded on the canonical name
    # already being present, so the pass must run AFTER the generators register.
    # An import cannot promise that: if anything imported ``stubs`` earlier the
    # pass already ran against a partial registry, and this import would be a
    # ``sys.modules`` no-op that silently leaves 53 names missing. Calling the
    # function makes the ordering this comment claims actually enforced.
    try:
        from spectramr.models.stubs import register_aliases

        register_aliases()
    except ImportError as e:
        logger.warning("Failed to import stubs module: %s", e)

    # Out-of-tree models: the ONLY seam by which a @register_model defined
    # OUTSIDE the spectramr tree (a pip-installed plugin or a SPECTRAMR_PLUGINS
    # module) fires its decorator and joins MODEL_REGISTRY. Idempotent — see
    # spectramr.plugins.discover_plugins.
    # In-tree names are registered by the walk ABOVE before plugins load, so a
    # plugin that re-registers an existing name collides visibly instead of
    # winning by import order.
    _discover_model_plugins()

    _REGISTRY_POPULATED = True
