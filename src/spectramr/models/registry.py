"""
Model Registry Module.

Provides the central registry and decorator for registering model architectures.
This registry enables O(1) lookup of model classes by name and supports
different training modes (e.g., 'gan', 'diffusion', 'reconstruction').

Capability flags (e.g. ``supports_contrast_conditioning``) let the
audit ladder fail loudly when YAML opts into a feature the chosen
model does not implement — preventing the silent-fallback pitfall
documented in CLAUDE.md.
"""

from typing import Any, Literal

from spectramr.config.schemas.enums import Regime, Task
from spectramr.models.capabilities import Domain, ModelCapabilities

MODEL_REGISTRY: dict[str, dict[str, Any]] = {}


def register_model(
    name: str,
    training_mode: str,
    *,
    role: "Literal['generator', 'discriminator']" = "generator",
    # Capability metadata (Phase 1 of experiment-spec-card design).
    # All default to None ("unannotated"); the audit skips checks for
    # unannotated fields so existing registrations stay green.
    #
    # The two conditioning flags below defaulted to ``False`` until #1916.
    # ``None`` is the file's own stated convention and the distinction is
    # real: ``False`` is a positive claim that the model ignores the id,
    # ``None`` means nobody has said. Defaulting them to ``False`` would
    # also give every one of the 586 registrations a non-None capability
    # field, collapsing the "unannotated" sentinel that
    # ``get_model_capabilities`` and the audit's default-deny bucket are
    # both built on (measured: 407 models would flip).
    supports_contrast_conditioning: bool | None = None,
    supports_vendor_conditioning: bool | None = None,
    # Domain fields accept either a single Domain literal or a tuple
    # of literals for models that genuinely handle multiple domains.
    spatial_dims: tuple[int, ...] | None = None,
    input_domain: "Domain | tuple[Domain, ...] | None" = None,
    output_domain: "Domain | tuple[Domain, ...] | None" = None,
    accepts_complex: bool | None = None,
    expects_real_imag_interleaved: bool | None = None,
    requires_paired_data: bool | None = None,
    output_field_units: str | None = None,
    trajectory_parametrization: str | None = None,
    override: bool = False,
    workflows: "frozenset[Regime] | None" = None,
    tasks: "frozenset[Task] | None" = None,
):
    """Decorator to register a model class.

    Args:
        name: Unique name of the model.
        training_mode: The training paradigm this model belongs to
            (e.g., 'gan', 'diffusion', 'reconstruction').
        role: Whether this registration is a generator or a discriminator.
            ``ModelFactory`` keeps a separate bucket for each, and this is
            the declaration it buckets on. It used to *guess*, with
            ``issubclass(cls, IDiscriminator)`` and a silent
            "default to generator for backward compatibility" branch --
            which put 9 discriminator classes that do not subclass the
            interface into the generator bucket, so ``create_discriminator``
            raised "Discriminator type '<x>' not registered" for every one
            of them (#1932).

            The default is deliberate, and is not the guess being removed.
            There are ~600 registration sites; requiring all of them to
            declare a role is a mechanical rewrite across the model tree,
            while only the ~19 discriminators have anything to say. The
            difference from the deleted code is that a default *states* the
            role in source where it can be read and overridden, whereas the
            guess *derived* it from an unrelated property at classification
            time. To keep the default from going silently wrong,
            registration raises when a class subclasses ``IDiscriminator``
            but declares (or defaults to) ``role="generator"``.
        supports_contrast_conditioning: True if the model's ``forward``
            accepts a ``contrast_idx`` (or ``contrast_id``) tensor and
            uses it for FiLM-style conditioning. Set this on every
            generator **and every discriminator** that participates in
            Pattern C (multi-contrast) training. The Tier-1 audit
            ``check_multi_contrast_model_support`` uses this flag to fail
            loudly when YAML enables ``data.multi_contrast`` against a model
            that ignores the id -- it reads the flag for the generator named
            by ``model.model_type`` and for the critic named by
            ``model.discriminator_component.name`` (#1931).

            **The flag is load-bearing at runtime, not just in the audit.**
            ``DiffusionTrainingStrategy._critic_conditioning`` reads it -- via
            ``model_supports``, the same top-level reader the audit uses -- to
            decide whether to send ``{timesteps, contrast_idx}`` to the critic
            on both the D and the G step. It never introspects the critic's
            signature, so declaring the flag on a class whose ``forward``
            cannot accept those kwargs raises on step 1 rather than training a
            run unconditioned. Declare it only where it is true.

            No count is quoted here on purpose. The census is recomputed per
            run (``config_health_checker._contrast_aware_critics``); a constant
            in this docstring went stale the day the first critic declared the
            flag, which is exactly what happened between the two halves of
            #1931.
        spatial_dims: Tuple of spatial-dim ranks the model supports
            (e.g. ``(2,)``, ``(3,)``, ``(2, 3)``). When set, the audit
            blocks YAMLs whose data block declares a different rank
            unless an explicit adapter bridges it.
        input_domain: Domain of the input tensor (``image``, ``kspace``,
            ``complex_image``, ``latent``, ``pde_grid``, ``mesh``).
        output_domain: Domain of the output tensor.
        accepts_complex: True if the forward path accepts
            ``torch.complex`` tensors directly.
        expects_real_imag_interleaved: True if the model expects 2C
            real channels representing C complex coils.
        requires_paired_data: True if training requires paired (input,
            target) examples. Cycle/SSL models set False explicitly.
        output_field_units: Physical units of a field-valued output
            (e.g. ``"Hz"`` for a B0/off-resonance map). The field-domain
            metrics + Tier-1 audit read this to enforce the parametrization
            guard (pitfall #16) so a Hz-RMSE metric refuses to grade a model
            whose declared output is an image.
        trajectory_parametrization: Coordinate system of a trajectory-valued
            output (``"spiral"`` / ``"cartesian"`` / ``"radial"``). Blocks a
            spiral-trajectory metric from grading a Cartesian-per-line estimate.
        override: Permit a same-class re-registration to REPLACE a
            capabilities-bearing entry with an empty (all-None) one. Off by
            default so a second bare ``register_model(name, mode)(Cls)`` cannot
            silently clobber a decorator's full ``ModelCapabilities`` (the
            bloch_mamba_v2 scar) — which would disable the audit's
            compatibility checks for that model without any diagnostic.
    """

    capabilities = ModelCapabilities(
        spatial_dims=spatial_dims,
        input_domain=input_domain,
        output_domain=output_domain,
        accepts_complex=accepts_complex,
        expects_real_imag_interleaved=expects_real_imag_interleaved,
        requires_paired_data=requires_paired_data,
        supports_contrast_conditioning=supports_contrast_conditioning,
        supports_vendor_conditioning=supports_vendor_conditioning,
        output_field_units=output_field_units,
        trajectory_parametrization=trajectory_parametrization,
        workflows=workflows,
        tasks=tasks,
    )

    def decorator(cls: type[Any]):
        # Per CLAUDE.md #9 and TODO/audit/09_models_registry_generators.md
        # §3.18, refuse to silently overwrite an existing registration
        # with a *different* class. Same-class re-registration (test
        # reloads / idempotent imports) is allowed.
        existing = MODEL_REGISTRY.get(name)
        if existing is not None:
            existing_cls = existing.get("class")
            if existing_cls is not cls:
                raise ValueError(
                    f"Model '{name}' already registered to "
                    f"{existing_cls.__module__}.{existing_cls.__qualname__} "
                    f"(mode={existing.get('mode')!r}); refusing to overwrite "
                    f"with {cls.__module__}.{cls.__qualname__} "
                    f"(mode={training_mode!r}). Rename one of the two registrations."
                )
            # Same class re-registration: refuse a capability DOWNGRADE.
            # A second registration that declares LESS than the first
            # silently replaces the decorator's declared caps and disables
            # the audit's data/model compatibility checks (the bloch_mamba_v2
            # scar).
            #
            # This used to test ``capabilities == ModelCapabilities()`` --
            # "the new registration declares nothing at all". That was a
            # PROXY for a downgrade, and it held only while the dataclass
            # carried nothing but contract fields: any field set made the
            # caps non-empty, and every field was one you would be sorry to
            # lose. #1916 adds two conditioning flags, which breaks the
            # proxy in the worst direction -- ``register_model(name, mode,
            # supports_contrast_conditioning=True)`` on an already-registered
            # class is now non-empty, so the old condition waves through the
            # single most plausible partial re-registration ("just add the
            # flag") and drops spatial_dims / input_domain / output_domain to
            # None in silence. Measured on this branch before the fix: the
            # re-registration raised on origin/dev and overwrote here.
            #
            # So test the thing the comment always claimed to test -- a field
            # going declared -> undeclared -- rather than a stand-in for it.
            # Adding flags is still free; only dropping one raises.
            existing_caps = existing.get("capabilities")
            if not override and isinstance(existing_caps, ModelCapabilities):
                dropped = sorted(
                    f
                    for f in existing_caps.__dataclass_fields__
                    if getattr(existing_caps, f) is not None
                    and getattr(capabilities, f, None) is None
                )
                if dropped:
                    raise ValueError(
                        f"Refusing to re-register model '{name}' "
                        f"({cls.__module__}.{cls.__qualname__}): it would DROP "
                        f"already-declared capabilities {dropped}. The existing "
                        f"registration declares {existing_caps}. Silently "
                        f"un-declaring them disables the audit's compatibility "
                        f"checks for this model. Re-state the dropped "
                        f"capabilities in this registration, remove the "
                        f"redundant registration, or pass override=True to force."
                    )

        # One owner for capability flags: the nested ``ModelCapabilities``.
        # Until #1916 this literal ALSO fanned the two conditioning flags out
        # to ad-hoc top-level keys, which is how one registry came to hold two
        # disagreeing answers to the same question -- ``model_supports`` read
        # the top level and returned False for all 18 models that declare
        # ``accepts_complex`` nested, while ``get_model_capabilities`` read the
        # nested half and returned None for all 27 that declared contrast
        # support at the top. Zero overlap on every flag, no error either way
        # (non-negotiable 17).
        # A class that implements the discriminator interface but is filed as
        # a generator is always a mistake, and a silent one: it lands in the
        # generator bucket and ``create_discriminator`` then reports the name
        # as "not registered". Refuse it at registration, where the fix is
        # one keyword away -- same shape as the EMPTY-capabilities raise above.
        if role == "generator":
            # Imported here, not at module scope, to keep this module's import
            # weight off torch (the interfaces package pulls it in). NOT
            # wrapped in try/except: a swallowed ImportError would set the
            # name to None and silently disable the very guard that exists to
            # stop a silent misfiling -- the same non-negotiable 3 shape this
            # change deletes from ModelRegistry. The interfaces package
            # imports only abc/typing/torch and its own siblings, so there is
            # no cycle back to this module; if it ever fails to import, that
            # is a real breakage and must surface here.
            from spectramr.models.interfaces import IDiscriminator

            if isinstance(cls, type) and issubclass(cls, IDiscriminator):
                raise ValueError(
                    f"Model '{name}' ({cls.__module__}.{cls.__qualname__}) "
                    f"subclasses IDiscriminator but is registered with "
                    f"role='generator' (the default). It would land in the "
                    f"generator bucket and create_discriminator('{name}') "
                    f"would report it as not registered. Pass "
                    f"role='discriminator' to @register_model."
                )

        MODEL_REGISTRY[name] = {
            "class": cls,
            "mode": training_mode,
            "role": role,
            "capabilities": capabilities,
        }
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Rejected aspirational model names (Phase 4 of
# TODO/deleted_model_types_reimplementation_plan.md).
#
# These 24 names appeared in the d8ccb8452 deletion ledger with NO implementing
# class anywhere. Unlike the Phase-3 set (glow, blurring_diffusion, ... — now
# implemented) and the Phase-1 aliases (e.g. reversible_network → invertible_
# network), these are rejected: each is either an under-specified umbrella name
# with no anchor paper, a misfiled non-model (operator / procedure), or a
# speculative variant already subsumed by an implemented family. Re-adding any
# of them to VALID_MODEL_TYPES re-creates the "audit-surface lie" the deletion
# fixed. The namespace-axis audit check consults this set to emit the rejection
# rationale as the fix hint instead of a generic "NOT FOUND in registry".
# ---------------------------------------------------------------------------
REJECTED_NAMES: dict[str, str] = {
    # Under-specified umbrella names — no anchor paper, so the name is
    # marketing not specification. File a real impl under the chosen paper's
    # specific name instead.
    "advanced_vae_latent": "under-specified umbrella name; no anchor paper",
    "attention_dense": "under-specified umbrella name; no anchor paper",
    "autoregressive_decoder": "under-specified umbrella name; no anchor paper",
    "contrastive_network": "under-specified umbrella name; no anchor paper",
    "dense_prediction_unet": "under-specified umbrella name; no anchor paper",
    "dynamic_unet": "under-specified umbrella name; no anchor paper",
    "hyperspherical_network": "under-specified; use hyperspherical_vae instead",
    "mesh_mri": "under-specified umbrella name; no anchor paper",
    "uncertain_pyramid": "under-specified umbrella name; no anchor paper",
    "universal_adapter": "under-specified umbrella name; no anchor paper",
    "student_unet": "under-specified distillation stub; no anchor paper",
    # Misfiled non-models — an operator or a procedure, not an architecture.
    "max_pool": "a pooling operator, not a model architecture",
    "pareto_optimization": "a multi-objective procedure, not a model",
    # Subsumed by existing primitives / losses.
    "epistemic_aleatoric": "covered by evidential + uncertainty losses on heads",
    # Speculative GAN/VAE variants — no anchor paper; implemented families
    # (glow, progressive_gan, hierarchical_vq_vae, hyperspherical_vae, moe_vae,
    # beta_vae_gan) cover the scientifically-grounded cases.
    "score_based_gan": "speculative variant; no anchor paper",
    "sn_gan": "spectral-norm is a layer flag, not a distinct model",
    "unet_gan": "speculative variant; use a discriminator + UNet generator",
    "topographic_vae": "speculative variant; no anchor paper",
    "recursive_cascade": "speculative variant; no anchor paper",
    "recursive_residual": "speculative variant; no anchor paper",
    "wasserstein_vae": "speculative variant; no anchor paper",
    # TRELLIS speculative variants — the implemented trellis_*_vae set covers
    # the structured-latent family; these have no formulation.
    "trellis_diffusion": "speculative TRELLIS variant; no formulation",
    "trellis_image_large": "speculative TRELLIS variant; no formulation",
    "trellis_volume": "speculative TRELLIS variant; no formulation",
}


def get_model_class(name: str) -> type[Any]:
    """get_model_class.

    Args:
        name (str): Description.
    Returns:
        type[Any]: Description.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]["class"]


def get_model_mode(name: str) -> str:
    """get_model_mode.

    Args:
        name (str): Description.
    Returns:
        str: Description.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry.")
    return MODEL_REGISTRY[name]["mode"]


def _boolean_capability_fields() -> frozenset[str]:
    """The ``ModelCapabilities`` fields that are boolean flags.

    Keyed on the annotation rather than on ``__dataclass_fields__`` wholesale,
    because a truthiness read of a non-boolean field answers a question nobody
    asked: ``spatial_dims=(2, 3)`` would make
    ``model_supports(name, "spatial_dims")`` return True, which is neither
    wrong-and-loud nor right.
    """
    return frozenset(
        fname
        for fname, f in ModelCapabilities.__dataclass_fields__.items()
        if "bool" in str(f.type)
    )


def model_supports(name: str, capability: str) -> bool:
    """Return True if the registered model declares the given boolean capability.

    Reads the nested :class:`ModelCapabilities` -- the single owner of every
    capability flag since #1916.

    Raises on an unknown *capability*, and this replaces a documented silent
    fallback. The old implementation was ``entry.get(capability, False)``
    against a dict whose only keys were ``class``/``mode``/``capabilities``
    plus two ad-hoc flags, so every nested flag name answered False without
    erroring -- ``model_supports(m, "accepts_complex")`` was False for all 586
    models while 18 genuinely declared it. A typo'd flag name was
    indistinguishable from a model that does not have the capability, and both
    read as a confident "no" (non-negotiable 3).

    Returns False for an unknown *model*, which is deliberate and is not the
    same fallback. A model name arriving here has already passed the registry
    lookup that raises on an unknown name; the audit layer owns the loud
    failure for a bad ``model_type``, and answering False keeps this helper
    usable for "is this optional component present and capable?" probes.

    Args:
        name: Registered model name.
        capability: A boolean field of :class:`ModelCapabilities`.

    Raises:
        ValueError: If ``capability`` is not a boolean capability field.
    """
    valid = _boolean_capability_fields()
    if capability not in valid:
        raise ValueError(
            f"Unknown model capability {capability!r}. "
            f"Valid boolean capabilities: {sorted(valid)}. "
            f"Capability flags live on ModelCapabilities "
            f"(spectramr/models/capabilities.py); declare new ones there and "
            f"pass them through register_model()."
        )
    caps = get_model_capabilities(name)
    if caps is None:
        return False
    return getattr(caps, capability) is True


def get_model_capabilities(name: str) -> ModelCapabilities | None:
    """Return ``ModelCapabilities`` for a registered model, or None.

    Returns None for both unknown models and models whose decorator
    did not set any capability fields. Callers MUST distinguish between
    "unannotated" and "annotated as not-supported" — the latter has a
    populated dataclass, the former returns None.
    """
    entry = MODEL_REGISTRY.get(name)
    if entry is None:
        return None
    caps = entry.get("capabilities")
    if not isinstance(caps, ModelCapabilities):
        return None
    # Treat fully-default dataclass as "unannotated" so the audit skips it.
    if all(getattr(caps, f) is None for f in caps.__dataclass_fields__):
        return None
    return caps


def list_models() -> dict[str, dict[str, Any]]:
    """Return dictionary of all registered models."""
    return MODEL_REGISTRY.copy()


def list_models_with_capability(capability: str) -> list[str]:
    """Return all registered model names that declare ``capability=True``.

    Reads the nested :class:`ModelCapabilities` through
    :func:`model_supports`, so this list and that predicate cannot disagree.
    Before #1916 they agreed only by accident: both read the ad-hoc top-level
    keys, so both were blind to the same nested flags, and this function fed
    the audit's "compatible models" fix hint -- which rendered
    ``<no models declare this capability>`` for a capability 18 models declare.

    Raises ValueError on an unknown capability, via :func:`model_supports`.
    """
    return [n for n in MODEL_REGISTRY if model_supports(n, capability)]


# Note: Model discovery is handled in src/models/init_registry.py to avoid circular imports.
