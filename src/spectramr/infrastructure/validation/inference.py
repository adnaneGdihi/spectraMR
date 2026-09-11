"""Inference layer: derive Tags + Recommendations from a config.

Each function takes the loaded ``TrainingSettings`` and returns either a
list of :class:`Tag` (for property derivation) or a list of
:class:`Recommendation` (for design suggestions). All are pure: no
mutation of the config, no IO.

The CLI's ``compile_report`` function aggregates these alongside the
existing ``HealthCheckReport`` results into a unified
:class:`CompilerReport`. See
``docs/superpowers/specs/2026-05-05-experiment-spec-card-and-adapters-design.md``.
"""

from __future__ import annotations

from typing import Any

from spectramr.config.schemas.loss import LOSS_LIST_DOMAINS
from spectramr.infrastructure.validation.recommendations import (
    Recommendation,
    RecommendationLevel,
    Tag,
)
from spectramr.models.capabilities import ModelCapabilities

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Walk a chain of attribute (or mapping) names, defaulting on any miss.

    The dict branch used to ``return`` on the first name instead of descending,
    so a multi-name chain against a raw dict yielded the intermediate BLOCK.
    Invisible while every caller passed a single leaf; the block decomposition
    made every such call a chain.
    """
    for n in names:
        if obj is None:
            return default
        v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
        if v is None:
            return default
        obj = v
    return obj


def _model_capabilities(config: Any) -> ModelCapabilities | None:
    """The configured model's declared capabilities, or None.

    Delegates to :func:`get_model_capabilities`. This function used to carry
    its own copy of that body -- the entry lookup, the ``isinstance`` check
    and the all-fields-None "unannotated" sentinel -- which made it a second
    owner of the sentinel's definition (non-negotiable 17). The two agreed
    only because the copy iterated ``__dataclass_fields__`` dynamically; a
    sentinel keyed on an explicit field list, in either copy, would have
    silently diverged the moment a field was added.
    """
    from spectramr.models.registry import get_model_capabilities

    model = getattr(config, "model", None)
    if model is None:
        return None
    mt = getattr(model, "model_type", None)
    if not mt:
        return None
    return get_model_capabilities(str(mt))


# ---------------------------------------------------------------------------
# Tag derivation
# ---------------------------------------------------------------------------


def derive_tags(config: Any) -> list[Tag]:
    """Tag the experiment with derived properties.

    Tags are pure functions of the YAML — no side effects, deterministic.
    Downstream tooling reads ``CompilerReport.tags`` to filter / route.
    """
    tags: list[Tag] = []

    data = getattr(config, "data", None)
    model = getattr(config, "model", None)
    losses = getattr(config, "losses", None)
    training = getattr(config, "training", None)
    metadata = getattr(config, "metadata", None)

    # ── multi_contrast ───────────────────────────────────────────────
    mc_data = _get(data, "multi_contrast", "enabled", default=False) if data is not None else False
    mc_model = bool(_get(model, "use_contrast_guidance", default=False))
    mc_kwargs_n = _get(model, "model_kwargs", default={}) or {}
    mc_kwargs = isinstance(mc_kwargs_n, dict) and bool(
        mc_kwargs_n.get("use_contrast_guidance") or mc_kwargs_n.get("num_contrasts")
    )
    mc_meta = False
    if metadata is not None:
        meta_tags = (
            metadata.get("tags") if isinstance(metadata, dict) else getattr(metadata, "tags", None)
        ) or {}
        mc_meta = isinstance(meta_tags, dict) and meta_tags.get("paradigm") == "multi_contrast"
    is_mc = bool(mc_data or mc_model or mc_kwargs or mc_meta)
    if is_mc:
        sources = []
        if mc_data:
            sources.append("data.multi_contrast.enabled")
        if mc_model:
            sources.append("model.use_contrast_guidance")
        if mc_kwargs:
            sources.append("model.model_kwargs.{use_contrast_guidance,num_contrasts}")
        if mc_meta:
            sources.append("metadata.tags.paradigm")
        tags.append(
            Tag(
                name="multi_contrast",
                value=True,
                derived_from=tuple(sources),
                note="Experiment treats multiple acquisition contrasts (T1w/T2w/FLAIR/...) as conditioning input.",
            )
        )

    # ── paradigm ─────────────────────────────────────────────────────
    if training is not None:
        tm = getattr(training, "training_mode", None)
        sc = getattr(training, "strategy_class", None) or ""
        paradigm = tm or None
        if not paradigm and "diffusion" in str(sc).lower():
            paradigm = "diffusion"
        if not paradigm and "vae" in str(sc).lower():
            paradigm = "vae"
        if not paradigm and "gan" in str(sc).lower():
            paradigm = "gan"
        if paradigm:
            tags.append(
                Tag(
                    name="paradigm",
                    value=paradigm,
                    derived_from=("training.training_mode", "training.strategy_class"),
                )
            )

    # ── task ─────────────────────────────────────────────────────────
    task = _get(training, "task")
    if task:
        tags.append(Tag(name="task", value=str(task), derived_from=("training.task",)))

    # ── operates_on ──────────────────────────────────────────────────
    caps = _model_capabilities(config)
    if caps is not None and caps.input_domain is not None and caps.output_domain is not None:
        if caps.input_domain == caps.output_domain:
            tags.append(
                Tag(
                    name="operates_on",
                    value=caps.input_domain,
                    derived_from=("model.model_type@registered_capabilities",),
                    note=f"Model consumes and produces {caps.input_domain}.",
                )
            )
        else:
            tags.append(
                Tag(
                    name="operates_on",
                    value=f"{caps.input_domain}_to_{caps.output_domain}",
                    derived_from=("model.model_type@registered_capabilities",),
                    note=f"Cross-domain model: {caps.input_domain} → {caps.output_domain}.",
                )
            )

    # ── spatial_dimensionality ───────────────────────────────────────
    if data is not None:
        # `data.sampling.patch_size` since the block decomposition. The old flat
        # read returned None on every arm, so this Tag was never emitted.
        ps = _get(data, "sampling", "patch_size") or ()
        sd = sum(1 for d in ps if isinstance(d, int) and d > 1) if ps else None
        if sd is not None:
            tags.append(
                Tag(
                    name="spatial_dimensionality",
                    value=f"{sd}D",
                    derived_from=("data.patch_size",),
                )
            )

    # ── uses_adapters ────────────────────────────────────────────────
    adapters = getattr(config, "adapters", None)
    if adapters is not None:
        any_adapter = any(
            (getattr(adapters, h, None) or [])
            for h in ("pre_model", "post_model", "pre_loss_pred", "pre_loss_target", "pre_metric")
        )
        if any_adapter:
            tags.append(
                Tag(
                    name="uses_adapters",
                    value=True,
                    derived_from=("adapters.*",),
                )
            )

    # ── physics_aware ────────────────────────────────────────────────
    physics = getattr(config, "physics", None)
    if physics is not None:
        dc_enabled = _get(physics, "data_consistency", "enabled", default=False)
        kspace_enabled = _get(physics, "kspace", "enable_kspace_recon", default=False)
        if dc_enabled or kspace_enabled:
            tags.append(
                Tag(
                    name="physics_aware",
                    value=True,
                    derived_from=tuple(
                        p
                        for p, v in [
                            ("physics.data_consistency.enabled", dc_enabled),
                            ("physics.kspace.enable_kspace_recon", kspace_enabled),
                        ]
                        if v
                    ),
                )
            )

    # ── scan_mode (Phase 8) ──────────────────────────────────────────
    mk = _get(model, "model_kwargs", default={}) or {}
    if isinstance(mk, dict) and mk.get("scan_mode"):
        tags.append(
            Tag(
                name="scan_mode",
                value=mk.get("scan_mode"),
                derived_from=("model.model_kwargs.scan_mode",),
                note="SFC linearizer used to flatten the spatial grid into a 1-D Mamba sequence.",
            )
        )

    # ── has_bloch_simulator (Phase 8) ────────────────────────────────
    bloch_models = {
        "bloch_cycle",
        "bloch_cycle_network",
        "cycle_physio_diff",
        "tissue_diffusion_mri",
        "epg_bloch_ode",
    }
    if model is not None:
        mt = _get(model, "model_type")
        if mt in bloch_models or (isinstance(mk, dict) and "bloch" in str(mk).lower()):
            tags.append(
                Tag(
                    name="has_bloch_simulator",
                    value=True,
                    derived_from=("model.model_type",),
                    note="Model embeds a Bloch / EPG signal simulator. Acquisition_params (TR/TE/TI/FA) drive the forward model.",
                )
            )

    # ── requires_sense_maps (Phase 8) ────────────────────────────────
    sense_loss_names = {"sense_adjoint_l1", "sense_adjoint_consistency"}
    if losses is not None:
        all_loss_entries = []
        for k in LOSS_LIST_DOMAINS:
            all_loss_entries.extend(getattr(losses, k, None) or [])
        names = {getattr(e, "name", None) for e in all_loss_entries}
        if names & sense_loss_names:
            matched = sorted(names & sense_loss_names)
            tags.append(
                Tag(
                    name="requires_sense_maps",
                    value=True,
                    derived_from=(f"losses.*.name in {matched}",),
                    note="At least one declared loss requires per-coil sensitivity maps. Verify ESPIRiT / coil_maps source.",
                )
            )

    # ── acceleration_factor (Phase 8) ────────────────────────────────
    accel = getattr(config, "undersampling", None)
    if accel is not None:
        ma = _get(accel, "max_acceleration") or _get(accel, "base_acceleration")
        if ma:
            tags.append(
                Tag(
                    name="acceleration",
                    value=f"{float(ma):.0f}x",
                    derived_from=(
                        "acceleration.max_acceleration",
                        "acceleration.base_acceleration",
                    ),
                )
            )

    return tags


# ---------------------------------------------------------------------------
# Recommendation generators
# ---------------------------------------------------------------------------


def recommend_inverse_adapter(config: Any) -> list[Recommendation]:
    """AUDIT-R001 — invertible pre_model adapter without post_model inverse.

    A 5D→4D slicer on ``pre_model`` without a 4D→5D gather on ``post_model``
    means the strategy emits 4D output where the dataset target is 5D.
    The audit's check_data_model_compatibility may not catch this if it
    only verifies ``pre_model``.
    """
    from spectramr.data.adapters.registry import get_adapter_capabilities

    out: list[Recommendation] = []
    adapters = getattr(config, "adapters", None)
    if adapters is None:
        return out

    pre = getattr(adapters, "pre_model", None) or []
    post = getattr(adapters, "post_model", None) or []
    post_names = {s.name for s in post if getattr(s, "enabled", True)}

    for step in pre:
        if not getattr(step, "enabled", True):
            continue
        caps = get_adapter_capabilities(step.name)
        if caps is None or not caps.invertible:
            continue
        # Heuristic: if the pre adapter's bridges_from has a key that
        # bridges_to does NOT, the post-model arm needs to undo that
        # transformation.
        changed_keys = {
            k for k in caps.bridges_from if caps.bridges_to.get(k) != caps.bridges_from[k]
        }
        if not changed_keys:
            continue
        # Look for any post_model adapter that bridges back.
        has_inverse = False
        for pname in post_names:
            pc = get_adapter_capabilities(pname)
            if pc is None:
                continue
            if any(pc.bridges_to.get(k) == caps.bridges_from.get(k) for k in changed_keys):
                has_inverse = True
                break
        if not has_inverse:
            out.append(
                Recommendation(
                    code="AUDIT-R001",
                    title=f"Declare a post_model adapter inverse to {step.name!r}",
                    detail=(
                        f"adapters.pre_model declares {step.name!r} which transforms "
                        f"{dict(caps.bridges_from)} → {dict(caps.bridges_to)}. The model "
                        "output will then be in the bridged-to form; downstream stages "
                        "(loss, metrics, validation visualization) usually expect the "
                        "original form. Declare a post_model adapter that undoes this."
                    ),
                    level=RecommendationLevel.SUGGESTION,
                    yaml_keys=("adapters.pre_model", "adapters.post_model"),
                    suggested_yaml=(
                        f"adapters:\n  post_model:\n    - name: <inverse_of_{step.name}>\n"
                    ),
                    references=(
                        "docs/superpowers/specs/2026-05-05-experiment-spec-card-and-adapters-design.md",
                    ),
                )
            )
    return out


def recommend_multi_contrast_alignment(config: Any) -> list[Recommendation]:
    """AUDIT-R002 / R003 — multi-contrast data and model alignment.

    Three patterns, each emits a different recommendation:
    - data multi-contrast but model has no contrast conditioning capability
    - data multi-contrast but no acquisition_params declared
    - model accepts contrast but no multi-contrast data declared
    """
    out: list[Recommendation] = []
    data = getattr(config, "data", None)
    model = getattr(config, "model", None)
    if data is None or model is None:
        return out

    mc_enabled = _get(data, "multi_contrast", "enabled", default=False)
    n_contrasts = _get(data, "multi_contrast", "n_contrasts")
    acq_params = _get(data, "multi_contrast", "acquisition_params")

    # Model side
    mt = getattr(model, "model_type", None)
    from spectramr.models.registry import MODEL_REGISTRY, model_supports

    # Read through ``model_supports``, not ``entry.get(...)``. This site used
    # to reach past the helper into the registry entry's top-level keys, which
    # is why a grep for the helper's callers did not find it -- and once the
    # flag moved onto the nested ``ModelCapabilities`` (#1916), a top-level
    # ``.get`` would have answered False for every model that declares it and
    # raised this recommendation against 27 correctly-annotated models.
    supports_contrast = (
        bool(mt)
        and mt in MODEL_REGISTRY
        and model_supports(str(mt), "supports_contrast_conditioning")
    )

    if mc_enabled and not supports_contrast:
        out.append(
            Recommendation(
                code="AUDIT-R002",
                title=f"Model {mt!r} does not declare contrast conditioning support",
                detail=(
                    f"data.multi_contrast.enabled=True (n_contrasts={n_contrasts}) "
                    f"but {mt!r} was registered without `supports_contrast_conditioning=True`. "
                    "The model will receive multi-contrast batches but ignore the contrast id, "
                    "collapsing distinct contrasts into a single mean. Either pick a "
                    "contrast-aware model variant or annotate this model's capability "
                    "if its forward actually does consume the contrast id."
                ),
                level=RecommendationLevel.SUGGESTION,
                yaml_keys=("data.multi_contrast.enabled", "model.model_type"),
                references=(
                    "src/models/registry.py::register_model::supports_contrast_conditioning",
                ),
            )
        )

    if mc_enabled and not acq_params:
        # `acquisition_params` feeds Bloch/PMPS *physics* conditioning
        # (`BlochSignalEncoder`, reached via `multi_contrast.bloch_grounded` --
        # which already RAISES at load time when they are absent, see
        # `data.py::_validate_bloch_grounded`). Contrast-*id* conditioning
        # (Pattern C: `contrast_idx` -> `nn.Embedding(n_contrasts, embed_dim)`,
        # e.g. `kspace_cold_diffusion`) embeds an INDEX and never reads
        # TR/TE/TI/FA, so for an id-conditioned arm the blanket form of this hint
        # asserted a fallback that does not happen and handed the user a
        # copy-paste block that would add a knob nothing reads (pitfall #15).
        #
        # Deliberately still EMITTED for both cases rather than gated off: the
        # registry has `supports_contrast_conditioning` but no "consumes
        # acquisition_params" flag, so a hard gate would have to guess across 590
        # models and could silence a genuine hint. What changes is that
        # applicability is now *computed* from the config instead of asserted,
        # and the paste-ready YAML is offered only when it would be read.
        bloch_grounded = bool(_get(data, "multi_contrast", "bloch_grounded", default=False))
        embed_dim = _get(data, "multi_contrast", "embed_dim")
        id_conditioned = bool(supports_contrast and n_contrasts and not bloch_grounded)
        if id_conditioned:
            detail = (
                "data.multi_contrast.enabled=True with no acquisition_params list. "
                f"This arm looks contrast-ID conditioned: {mt!r} is registered "
                "supports_contrast_conditioning=True, the arm declares "
                f"n_contrasts={n_contrasts}"
                + (f" / embed_dim={embed_dim}" if embed_dim else "")
                + ", and multi_contrast.bloch_grounded is not set. That path "
                "embeds a contrast INDEX and reads no TR/TE/TI/FA, so contrast "
                "conditioning is already fully specified and declaring "
                "acquisition_params here would add a knob nothing reads. "
                "No action needed unless you switch this arm to physics "
                "(Bloch/EPG) conditioning, which is what consumes them."
            )
            suggested = None
        else:
            detail = (
                "data.multi_contrast.enabled=True but no acquisition_params list "
                "is declared. Physics-conditioned models (Bloch/EPG) read "
                "(TE, TR, TI, FA, B0) per contrast from this block and cannot "
                "recover them from anywhere else. Declare one entry per contrast, "
                "using this dataset's real protocol values -- do not copy the "
                "example numbers below, they are shape-only placeholders."
            )
            suggested = (
                "data:\n"
                "  multi_contrast:\n"
                "    acquisition_params:\n"
                "      - {name: T1w, TE: 12.0, TR: 600.0, FA: 70.0, B0: 3.0, contrast_type: spin_echo}\n"
                "      - {name: T2w, TE: 80.0, TR: 4000.0, FA: 90.0, B0: 3.0, contrast_type: spin_echo}\n"
            )
        out.append(
            Recommendation(
                code="AUDIT-R003",
                title="Multi-contrast data without acquisition_params",
                detail=detail,
                level=RecommendationLevel.HINT,
                yaml_keys=("data.multi_contrast.acquisition_params",),
                suggested_yaml=suggested,
            )
        )

    # ── R010: model wants contrast but data has no multi_contrast ────
    # Detected via model_kwargs even when data.multi_contrast is absent.
    mk = _get(model, "model_kwargs", default={}) or {}
    if isinstance(mk, dict):
        model_wants_contrast = (
            bool(mk.get("use_contrast_guidance"))
            or bool(mk.get("num_contrasts"))
            or bool(mk.get("contrast_embed_dim"))
        )
    else:
        model_wants_contrast = False
    # Datasets that emit a per-sample contrast_id natively (mrixfields) satisfy
    # the model's contrast conditioning WITHOUT a data.multi_contrast block, so
    # R010 would be a false positive there — worse, its suggested fix would wire
    # an unused, *different* conditioning path. Exempt them.
    dataset_type = _get(data, "dataset_type", default=None)
    native_contrast = dataset_type == "mrixfields"
    if model_wants_contrast and not mc_enabled and not native_contrast:
        nc = mk.get("num_contrasts") if isinstance(mk, dict) else None
        out.append(
            Recommendation(
                code="AUDIT-R010",
                title="Model wired for contrast conditioning but data has no multi_contrast block",
                detail=(
                    f"model.model_kwargs declares {'/'.join(k for k in ('use_contrast_guidance', 'num_contrasts', 'contrast_embed_dim') if mk.get(k))}, "
                    f"but data.multi_contrast.enabled is False/absent. The model "
                    "will allocate the contrast embedding table at init but the "
                    "data pipeline will never provide a contrast_idx tensor — the "
                    "embedding is dead weight. Either drop the model-side contrast "
                    "kwargs or enable multi-contrast data."
                ),
                level=RecommendationLevel.SUGGESTION,
                yaml_keys=(
                    "data.multi_contrast.enabled",
                    "model.model_kwargs.num_contrasts",
                    "model.model_kwargs.use_contrast_guidance",
                ),
                suggested_yaml=(
                    "data:\n"
                    "  multi_contrast:\n"
                    "    enabled: true\n"
                    f"    n_contrasts: {nc or 4}\n"
                    "    embed_dim: 128\n"
                ),
            )
        )

    return out


def recommend_redundant_adapter(config: Any) -> list[Recommendation]:
    """AUDIT-R004 — adapter declared whose bridge is a no-op given capabilities.

    If the model's input_domain already matches the data's inferred
    domain, declaring a domain-bridging adapter on pre_model is dead
    config — flag for cleanup.
    """
    from spectramr.data.adapters.registry import get_adapter_capabilities
    from spectramr.infrastructure.validation.spec_card import (
        _derive_data_form,
        _derive_model_form,
    )

    out: list[Recommendation] = []
    adapters = getattr(config, "adapters", None)
    if adapters is None:
        return out

    data_form = _derive_data_form(config)
    model_form = _derive_model_form(config)
    caps = model_form.get("capabilities") if model_form else None
    if caps is None or not data_form.get("present"):
        return out

    pre = getattr(adapters, "pre_model", None) or []
    for step in pre:
        if not getattr(step, "enabled", True):
            continue
        ac = get_adapter_capabilities(step.name)
        if ac is None:
            continue
        # The adapter bridges from X to Y. If the data already produces
        # Y *and* the model accepts Y, the adapter is redundant.
        # Compare for each key the adapter changes.
        changes_match_data = all(
            data_form.get(k.replace("domain", "domain_inferred")) == v
            or v is None
            or (k == "domain" and data_form.get("domain_inferred") == v)
            for k, v in ac.bridges_to.items()
        )
        # Crude check: if the data already produces what the adapter produces,
        # it's likely a no-op.
        data_already_produces = (
            ac.bridges_to.get("domain") == data_form.get("domain_inferred")
        ) and (ac.bridges_to.get("spatial_dims") in (None, data_form.get("spatial_dims")))
        if data_already_produces and changes_match_data:
            out.append(
                Recommendation(
                    code="AUDIT-R004",
                    title=f"Adapter {step.name!r} on pre_model is redundant",
                    detail=(
                        f"Data already produces {ac.bridges_to}. The adapter "
                        "would re-apply a no-op transformation. Remove for clarity "
                        "or change the adapter."
                    ),
                    level=RecommendationLevel.LINT,
                    yaml_keys=(f"adapters.pre_model[*].name={step.name!r}",),
                )
            )
    return out


def recommend_diffusion_visualization(config: Any) -> list[Recommendation]:
    """AUDIT-R005 — diffusion paradigm should enable validation visualization.

    Diffusion models without periodic image-domain validation samples are
    very hard to debug — you only see scalar losses, never the actual
    reconstructions. Recommend enabling.
    """
    out: list[Recommendation] = []
    training = getattr(config, "training", None)
    if training is None:
        return out
    sc = (getattr(training, "strategy_class", None) or "").lower()
    tm = (getattr(training, "training_mode", None) or "").lower()
    if "diffusion" not in sc and "diffusion" not in tm:
        return out

    logging_cfg = getattr(config, "logging", None)
    # `logging.images.{save,log}_validation` since the block decomposition. The
    # old flat reads returned False on EVERY arm, so AUDIT-R005 fired on all 58
    # drained kspace_filling arms while they were saving images -- and its
    # suggestion taught the retired spelling.
    save_imgs = _get(logging_cfg, "images", "save_validation", default=False)
    log_imgs = _get(logging_cfg, "images", "log_validation", default=False)
    if not (save_imgs or log_imgs):
        out.append(
            Recommendation(
                code="AUDIT-R005",
                title="Diffusion experiment without validation image saving",
                detail=(
                    "Diffusion training is hard to interpret from loss curves alone. "
                    "Enable `logging.images.save_validation: true` and "
                    "`logging.images.log_validation: true` so the validation "
                    "loop emits PNGs that reveal mode collapse, identity collapse, "
                    "and reconstruction quality regressions."
                ),
                level=RecommendationLevel.SUGGESTION,
                yaml_keys=("logging.images.save_validation", "logging.images.log_validation"),
                suggested_yaml=(
                    "logging:\n"
                    "  images:\n"
                    "    save_validation: true\n"
                    "    log_validation: true\n"
                    "  intervals:\n"
                    "    validation_images: 10\n"
                ),
            )
        )
    return out


def recommend_explicit_kspace_image_adapter(config: Any) -> list[Recommendation]:
    """AUDIT-R006 — k-space data + image-domain losses without explicit chain.

    Catches the experiment_11_kspace_cold_diffusion class of bug: the
    LossBuilder used to silently pass image-domain reals to k-space losses
    (zeroes). Even now with the corrected output_domain, declaring the
    adapter explicitly makes the chain visible at audit time.
    """
    out: list[Recommendation] = []
    losses = getattr(config, "losses", None)
    data = getattr(config, "data", None)
    if losses is None or data is None:
        return out

    from spectramr.infrastructure.validation.spec_card import _derive_data_form

    data_form = _derive_data_form(config)
    domain_inferred = data_form.get("domain_inferred")
    output_domain = losses.policy.output_domain if losses else None

    # Pattern: data is k-space, losses output_domain is image, no adapter
    has_image_losses = bool(getattr(losses, "image_losses", None) or [])
    adapters = getattr(config, "adapters", None)
    has_explicit_chain = adapters is not None and (
        getattr(adapters, "pre_loss_pred", None) or getattr(adapters, "pre_loss_target", None)
    )
    if (
        domain_inferred == "kspace"
        and output_domain == "image"
        and has_image_losses
        and not has_explicit_chain
    ):
        out.append(
            Recommendation(
                code="AUDIT-R006",
                title="K-space data + image losses: declare explicit pre_loss adapter",
                detail=(
                    "Data is k-space, losses.output_domain='image', and image_losses "
                    "are declared. The LossBuilder applies an implicit iFFT-magnitude "
                    "bridge — but that bridge is invisible to anyone reading the "
                    "YAML. Declare the chain explicitly so the audit's adapter "
                    "tracking reflects reality and the Spec Card surfaces the "
                    "transformation's side-effects."
                ),
                level=RecommendationLevel.HINT,
                yaml_keys=("adapters.pre_loss_pred", "adapters.pre_loss_target"),
                suggested_yaml=(
                    "adapters:\n"
                    "  pre_loss_pred:\n"
                    "    - name: ifft_kspace_to_image\n"
                    "    - name: magnitude_from_complex\n"
                    "  pre_loss_target:\n"
                    "    - name: ifft_kspace_to_image\n"
                    "    - name: magnitude_from_complex\n"
                ),
            )
        )
    return out


def recommend_strategy_paradigm_consistency(config: Any) -> list[Recommendation]:
    """AUDIT-R008 — training_mode disagrees with the strategy class's paradigm."""
    out: list[Recommendation] = []
    training = getattr(config, "training", None)
    if training is None:
        return out
    tm = (getattr(training, "training_mode", None) or "").lower()
    sc = (getattr(training, "strategy_class", None) or "").lower()
    if not tm or not sc:
        return out

    pairs = [
        ("diffusion", "diffusion"),
        ("vae", "vae"),
        ("gan", "gan"),
        ("reconstruction", "reconstruction"),
        ("ssl", "pretraining"),  # MAEPretrainingStrategy
        ("mae", "pretraining"),
    ]
    expected_in_class = next((cls_marker for tag, cls_marker in pairs if tm == tag), None)
    if expected_in_class is None:
        return out
    if expected_in_class not in sc:
        out.append(
            Recommendation(
                code="AUDIT-R008",
                title=f"training_mode={tm!r} but strategy_class doesn't reference it",
                detail=(
                    f"Declared training_mode='{tm}' but strategy_class is '{sc}'. "
                    "Either training_mode is stale (legacy field) or strategy_class "
                    "points at the wrong implementation. The strategy_class is the "
                    "actual dispatch; training_mode is decorative — but inconsistency "
                    "confuses downstream readers."
                ),
                level=RecommendationLevel.LINT,
                yaml_keys=("training.training_mode", "training.strategy_class"),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def recommend_sense_loss_needs_sense_maps(config: Any) -> list[Recommendation]:
    """AUDIT-R020 — sense_adjoint_l1 in losses but no coil_sensitivity declared.

    SENSE-style losses need per-coil sensitivity maps. If the YAML
    doesn't declare a source (ESPIRiT, file path, or PINN-estimated),
    the loss either silently no-ops or uses an identity coil (single
    virtual coil), defeating the parallel-imaging point.
    """
    out: list[Recommendation] = []
    losses = getattr(config, "losses", None)
    if losses is None:
        return out
    sense_loss_names = {"sense_adjoint_l1", "sense_adjoint_consistency"}
    has_sense = False
    for k in LOSS_LIST_DOMAINS:
        for entry in getattr(losses, k, None) or []:
            if getattr(entry, "name", None) in sense_loss_names:
                has_sense = True
                break
    if not has_sense:
        return out
    # Coil-sensitivity sources we recognise (mirror the data/physics builders).
    # NOTE: ``physics.coil_sensitivity`` exposes ``enable_estimation`` (+
    # ``precomputed_path``), NOT ``enabled``; ``physics.coil_processing.estimation.method
    # != "none"`` is also a source. The previous check looked for the non-existent
    # ``physics.coil_sensitivity.enabled`` key, so it false-fired on every arm that
    # enabled ESPIRiT / power-iter coil estimation (46 cohort arms, 2026-06-20).
    has_source = (
        bool(_get(config, "data", "load_coil_sensitivity", default=False))
        or bool(_get(config, "data", "coil_sensitivity_path", default=None))
        or bool(_get(config, "data", "use_espirit_calibration", default=False))
        or bool(_get(config, "physics", "coil_sensitivity", "enable_estimation", default=False))
        or bool(_get(config, "physics", "coil_sensitivity", "precomputed_path", default=None))
        or (
            _get(config, "physics", "coil_processing", "estimation", "method", default="none")
            not in (None, "none")
        )
    )
    if not has_source:
        out.append(
            Recommendation(
                code="AUDIT-R020",
                title="SENSE-style loss declared but no coil_sensitivity source",
                detail=(
                    "losses.* contains sense_adjoint_l1 (or sibling) but the "
                    "YAML doesn't declare where coil sensitivity maps come from. "
                    "Without sense maps, the loss reduces to a per-coil L1 with "
                    "no parallel-imaging benefit. Declare ESPIRiT calibration, "
                    "a precomputed file path, or PINN-estimated sense maps."
                ),
                level=RecommendationLevel.HINT,
                yaml_keys=(
                    "physics.coil_sensitivity.enable_estimation",
                    "physics.coil_sensitivity.precomputed_path",
                    "physics.coil_processing.estimation.method",
                ),
                suggested_yaml=(
                    "physics:\n"
                    "  coil_sensitivity:\n"
                    "    enable_estimation: true\n"
                    "    estimation_method: espirit\n"
                ),
                references=("src/infrastructure/physics/coil_sensitivity.py",),
            )
        )
    return out


def recommend_bloch_acquisition_params(config: Any) -> list[Recommendation]:
    """AUDIT-R021 — Bloch model + multi_contrast but acquisition_params missing/incomplete.

    Triggers when the model embeds a Bloch / EPG simulator and the
    data block declares multiple contrasts but the per-contrast
    TR/TE/(TI/FA) vectors are absent or sparse.
    """
    out: list[Recommendation] = []
    model = getattr(config, "model", None)
    data = getattr(config, "data", None)
    if model is None or data is None:
        return out

    bloch_models = {
        "bloch_cycle",
        "bloch_cycle_network",
        "cycle_physio_diff",
        "tissue_diffusion_mri",
        "epg_bloch_ode",
    }
    mt = getattr(model, "model_type", None)
    is_bloch = mt in bloch_models
    if not is_bloch:
        return out

    mc_enabled = _get(data, "multi_contrast", "enabled", default=False)
    if not mc_enabled:
        return out

    n = _get(data, "multi_contrast", "n_contrasts")
    acq = _get(data, "multi_contrast", "acquisition_params") or []
    required_keys = {"TR", "TE"}
    missing_per_contrast: list[str] = []
    if not acq:
        out.append(
            Recommendation(
                code="AUDIT-R021",
                title=f"Bloch model {mt!r} multi-contrast without acquisition_params",
                detail=(
                    f"The Bloch / EPG simulator inside {mt!r} consumes per-contrast "
                    "TR/TE (and ideally TI/FA/B0) to predict the signal magnitude. "
                    "data.multi_contrast.enabled=True but no acquisition_params list "
                    "is declared — the simulator will fall back to identity/zero "
                    "parameters and the contrast embedding becomes a black-box id "
                    "with no physical anchor."
                ),
                level=RecommendationLevel.SUGGESTION,
                yaml_keys=("data.multi_contrast.acquisition_params",),
                suggested_yaml=(
                    "data:\n"
                    "  multi_contrast:\n"
                    "    acquisition_params:\n"
                    "      - {name: T1w, TE: 12.0, TR: 600.0, FA: 70.0, B0: 3.0, contrast_type: spin_echo}\n"
                    "      - {name: T2w, TE: 80.0, TR: 4000.0, FA: 90.0, B0: 3.0, contrast_type: spin_echo}\n"
                ),
            )
        )
        return out
    # Check completeness for each declared contrast.
    if isinstance(acq, list):
        for i, p in enumerate(acq):
            keys_present = set(p.keys()) if isinstance(p, dict) else set()
            missing = required_keys - keys_present
            if missing:
                missing_per_contrast.append(
                    f"contrast[{i}]={p.get('name', '?') if isinstance(p, dict) else '?'} missing {sorted(missing)}"
                )
    if missing_per_contrast:
        out.append(
            Recommendation(
                code="AUDIT-R021",
                title="Bloch model acquisition_params incomplete",
                detail=(
                    f"acquisition_params is declared but: {'; '.join(missing_per_contrast)}. "
                    "Each entry needs at minimum TR and TE for a Bloch / EPG forward."
                ),
                level=RecommendationLevel.HINT,
                yaml_keys=("data.multi_contrast.acquisition_params",),
            )
        )
    return out


def recommend_mamba_scan_mode(config: Any) -> list[Recommendation]:
    """AUDIT-R024 — Mamba-family model without explicit ``scan_mode``.

    Many Mamba variants here (FE/CRM/HW/MM/MDI/HLD/INR/HilbertMamba/
    HDSF/CTMamba) use a space-filling-curve linearizer to feed the 1-D
    SSM. The linearizer falls back to a default if ``scan_mode`` is
    omitted — a *silent* configuration choice that affects which
    locality assumption the model makes (Hilbert vs raster vs Morton).
    Recommend declaring it explicitly so the audit's
    ``check_sfc_scan_mode_matches_spatial_dims`` can verify rank, and
    so reviewers see which scan order the experiment commits to.
    """
    out: list[Recommendation] = []
    model = getattr(config, "model", None)
    if model is None:
        return out
    mt = getattr(model, "model_type", None)
    if not mt:
        return out
    # Mamba variants known to take an SFC linearizer kwarg.
    mamba_with_sfc = {
        "fe_mamba",
        "crm_mamba",
        "hw_mamba",
        "mm_mamba",
        "mdi_mamba",
        "two_half_d_mamba",
        "hld_mamba",
        "inr_mamba",
        "hilbert_mamba_unet",
        "ct_mamba",
        "hyper_mamba_unet",
        "geo_mamba_unet",
        "cs_mno_operator",
    }
    if mt not in mamba_with_sfc:
        return out
    mk = getattr(model, "model_kwargs", None) or {}
    if not isinstance(mk, dict):
        return out
    # Either scan_mode or linearization_mode (some classes use the
    # longer name) is acceptable.
    has_scan = bool(
        mk.get("scan_mode") or mk.get("linearization_mode") or mk.get("linearization_modes")
    )
    if has_scan:
        return out
    out.append(
        Recommendation(
            code="AUDIT-R024",
            title=f"{mt!r} is Mamba-family but no scan_mode declared",
            detail=(
                f"{mt!r} uses a space-filling-curve linearizer to flatten the "
                "input grid into a 1-D Mamba sequence. Without an explicit "
                "scan_mode in model_kwargs, the model falls back to its "
                "default (usually hilbert_2d), which may not match the "
                "experiment's intent. Declaring it makes the audit's "
                "check_sfc_scan_mode_matches_spatial_dims verify rank and "
                "makes the locality assumption visible to reviewers."
            ),
            level=RecommendationLevel.HINT,
            yaml_keys=("model.model_kwargs.scan_mode",),
            suggested_yaml=(
                "model:\n"
                "  model_kwargs:\n"
                "    scan_mode: hilbert_2d   # or: morton_2d / snake_2d / raster_2d / raster_1d / hilbert_3d\n"
            ),
        )
    )
    return out


def recommend_dc_without_acceleration(config: Any) -> list[Recommendation]:
    """AUDIT-R023 — DC enabled but no acceleration block.

    Without an acceleration mask, the "measured k-space" the DC
    layer projects against is the *full* k-space (no missing lines)
    and the projection is a no-op. This usually means the user
    forgot to declare the undersampling pattern.
    """
    out: list[Recommendation] = []
    physics = getattr(config, "physics", None)
    if physics is None:
        return out
    dc = getattr(physics, "data_consistency", None)
    if dc is None or not getattr(dc, "enabled", False):
        return out
    # The digital twin masks inside the step and hands its mask to the DC
    # layers (``undersampling_mask_kwargs``); on a twin arm the top-level block
    # is not the mask's owner, ``physics.digital_twin.enable_undersampling`` is
    # (VF review 2026-09-03).
    twin = getattr(physics, "digital_twin", None)
    if (
        twin is not None
        and getattr(twin, "enabled", False)
        and getattr(twin, "enable_undersampling", False)
    ):
        return out
    accel = getattr(config, "undersampling", None)
    if accel is None:
        out.append(
            Recommendation(
                code="AUDIT-R023",
                title="physics.data_consistency.enabled but no acceleration block",
                detail=(
                    "DC layers project model output onto measured k-space at the "
                    "*sampled* lines. Without an acceleration block, every line "
                    "is sampled and DC reduces to identity — the layer fires but "
                    "does no work. Declare an `acceleration:` block (or disable DC)."
                ),
                level=RecommendationLevel.SUGGESTION,
                yaml_keys=("physics.data_consistency.enabled", "acceleration"),
                suggested_yaml=(
                    "acceleration:\n"
                    "  base_acceleration: 4.0\n"
                    "  max_acceleration: 4.0\n"
                    "  center_fraction: 0.08\n"
                    "  acceleration_type: equispaced\n"
                ),
            )
        )
    return out


_RECOMMENDATION_GENERATORS = (
    recommend_inverse_adapter,
    recommend_multi_contrast_alignment,
    recommend_redundant_adapter,
    recommend_diffusion_visualization,
    recommend_explicit_kspace_image_adapter,
    recommend_strategy_paradigm_consistency,
    # Phase 8 — Transforms + Physics
    recommend_sense_loss_needs_sense_maps,
    recommend_bloch_acquisition_params,
    recommend_dc_without_acceleration,
    recommend_mamba_scan_mode,
)


def collect_all_recommendations(config: Any) -> list[Recommendation]:
    out: list[Recommendation] = []
    for gen in _RECOMMENDATION_GENERATORS:
        try:
            out.extend(gen(config) or [])
        except Exception as e:  # never let a recommender crash the audit
            from spectramr.infrastructure.validation.recommendations import RecommendationLevel

            out.append(
                Recommendation(
                    code="AUDIT-R999",
                    title=f"Recommender {gen.__name__} crashed: {e}",
                    detail="This is a bug in the audit, not in your YAML.",
                    level=RecommendationLevel.HINT,
                )
            )
    return out


__all__ = [
    "collect_all_recommendations",
    "derive_tags",
    "recommend_bloch_acquisition_params",
    "recommend_dc_without_acceleration",
    "recommend_diffusion_visualization",
    "recommend_explicit_kspace_image_adapter",
    "recommend_inverse_adapter",
    "recommend_multi_contrast_alignment",
    "recommend_redundant_adapter",
    "recommend_sense_loss_needs_sense_maps",
    "recommend_strategy_paradigm_consistency",
]
