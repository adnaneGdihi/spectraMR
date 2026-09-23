"""Domain Inference Utility — Single Source of Truth for output domain detection.

Determines the true output domain of any generator+data combination using
a priority cascade:

    Priority 1: Explicit config declaration (model.target_domain)
    Priority 2: Model-type registry lookup (known k-space output models)
    Priority 3: Heuristic from data pipeline configuration
    Priority 4: Default to "image" (safest fallback)

This module exists because `model.input_type` does NOT determine output domain.
A generator with `input_type: image` may still output k-space data (it just
uses real-stacked channels for nn.Conv2d compatibility).

See docs/DOMAIN_HANDLING_RULES.md, Rule 2.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Models whose forward() output is in k-space domain.
# These models reconstruct / operate in Fourier space and their raw output
# needs IFFT before image-domain visualization or perceptual metrics.
#
# This set is intentionally conservative — only models KNOWN to output k-space.
# Models not listed here will rely on config.model.target_domain or default
# to "image" (see priority cascade in infer_output_domain).
KNOWN_KSPACE_OUTPUT_MODELS: frozenset[str] = frozenset(
    {
        # VarNet family
        "varnet",
        "jfe_varnet",
        "pma_varnet",
        "vcc_varnet",
        # MoDL / unrolled optimization
        "modl_sr",
        # K-space specific generators
        "kspace_cold_diffusion_generator",
        "kspace_cold_diffusion",
        "kspace_gpt",
        "kspace_gpt_foundation",
        "kspace_phase_ramp",
        # GRAPPA-style k-space interpolation
        "marker_grappa_conv",
        # Physics-based k-space models
        "diff_nufft",
        "diff_trajectory_opt",
        "hard_dc_forcing",
        "ls_ista",
        "pdhg_recon",
        "svd_projection",
        # Coil-related k-space models. (neural_complex_sum + dirichlet_espirit were
        # MOVED to KNOWN_IMAGE below — both are residual image reconstructors; the
        # two kept here genuinely operate on coil k-space / sensitivities and still
        # need a per-model forward read before any move.)
        "siren_sens_net",
        "pre_whitening_layer",
        # Phase / navigator models operating in k-space
        "fourier_shift_op",
        # VF physics / proof / k-space-operator models that genuinely operate in
        # k-space / Fourier domain. NB: the VF residual backbones that reconstruct in
        # IMAGE domain were MOVED to KNOWN_IMAGE_OUTPUT_MODELS below (smoke 2026-06-16,
        # E-VIZ2) — each forward is a residual / phase transform on the strategy's
        # already-IFFT'd image input (output domain == input domain), so listing them
        # here forced a spurious viz IFFT → DC-blob:
        #   vf_field_generators.py:       reciprocity_divisor, phase_tracking_lstm,
        #     resnet_unwrap, bloch_siegert_algebraic, afi_ratio_cnn, dam_trigfit,
        #     siamese_epi_tracker, mrf_dict_matcher, bssfp_peak_finder, graph_cut_unwrap
        #   vf_reconstruction_generators: kalman_pll, wiener_unet, neural_complex_sum,
        #     dirichlet_espirit
        #   vf_kspace_operators:          mtf_deapodization, phase_navigator_lock
        # The ones kept here still need per-model verification before any further move
        # (tracked follow-up).
        "sh_fitter",  # Spherical harmonic B0 fitting
        "stn_warp",  # Spatial transformer (k-space warps)
        "eddy_predictor_1d",  # Eddy current prediction
        "richardson_lucy",  # Fourier-domain deconvolution
        "pinn_helmholtz",  # PINN Helmholtz PDE
        "rigid_affine_kabsch",  # Rigid body motion correction
    }
)

# Models KNOWN to output image domain even when data is k-space.
# These models perform internal IFFT or operate directly on magnitude images.
KNOWN_IMAGE_OUTPUT_MODELS: frozenset[str] = frozenset(
    {
        "image_cold_diffusion",
        "dip_unet",
        "siren",
        "dynamic_mr_nerf",
        "standard_unet",
        "swin_unet",
        "vision_mamba",
        # VF marker backbones. The VF / motion-meta / distillation strategies
        # IFFT k-space → image *before* the backbone runs (see the
        # ``hyper_mamba_unet`` @register_model note: input_domain/output_domain
        # are declared ("image", "kspace") only to keep the spatial-dims and
        # data-model-compatibility checks permissive). Functionally these
        # backbones reconstruct in image domain in every VF arm. They were
        # previously listed in KNOWN_KSPACE_OUTPUT_MODELS, which forced a
        # spurious IFFT on the already-image prediction → DC-blob fakes across
        # 10 of 12 "PASS" VF arms (smoke audit 2026-06-03). The tuple
        # output_domain makes P2 abstain, so this P3 entry is the SSOT for
        # their default domain; an arm that genuinely emits k-space must set
        # ``model.target_domain: kspace`` (P1 overrides this set).
        "cross_attention_oracle_unet",
        "hyper_mamba_unet",
        # vf_field_generators.py residual field/recon backbones (smoke 2026-06-16,
        # E-VIZ2). Run by ConcreteVirtualFiducialStrategy (and motion-meta /
        # distillation), which IFFTs k-space → image *before* the backbone; each
        # forward is ``<spatial op>(x) + x`` → image output (internal FFTs in
        # siamese_epi_tracker / bssfp_peak_finder are intermediate, not output).
        # Moved here from KNOWN_KSPACE_OUTPUT_MODELS, where they forced a spurious
        # viz IFFT on the already-image prediction → DC-blob fakes. The decorators
        # stay UNANNOTATED by design (they consume the strategy's *synthesised*
        # stack, not the raw dataset domain — see the bssfp_b0_regressor
        # @register_model note), so this P3 entry is the SSOT for their default
        # domain; an arm that genuinely emits k-space sets ``model.target_domain:
        # kspace`` (P1 overrides this set).
        "reciprocity_divisor",
        "phase_tracking_lstm",
        "resnet_unwrap",
        "bloch_siegert_algebraic",
        "afi_ratio_cnn",
        "dam_trigfit",
        "siamese_epi_tracker",
        "mrf_dict_matcher",
        "bssfp_peak_finder",
        "graph_cut_unwrap",
        # Second batch (same E-VIZ2 reasoning, all in active VF use): the
        # vf_reconstruction_generators.py residual image reconstructors and the
        # vf_kspace_operators.py marker operators whose forward is a residual /
        # global-phase transform of the strategy's image input (the internal fft2c
        # is only for marker MTF / phase estimation, not the output path).
        "kalman_pll",
        "wiener_unet",
        "neural_complex_sum",
        "dirichlet_espirit",
        "mtf_deapodization",
        "phase_navigator_lock",
    }
)


#: Coil-processing modes that emit IMAGE-domain tensors regardless of the declared
#: ``dataset_type``: both apply an IFFT inside the dataset's TorchIO pipeline.
#: Listed exhaustively so adding a mode is a deliberate decision, not a fall-through.
# Re-exported from the data layer, which owns what a dataset produces. Two
# hand-maintained copies of "which modes emit images" is how the audit and the
# runtime start disagreeing about the same arm (#1010).
from spectramr.data.datasets.axis_exposure import (  # noqa: E402
    IMAGE_DOMAIN_COIL_MODES,
)


def _get_attr_safe(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get attribute from object or dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _data_leaf(data_cfg: Any, block: str, leaf: str, legacy: str, default: Any = None) -> Any:
    """Read a decomposed ``data:`` leaf, canonical location FIRST.

    ``data.normalize_kspace`` and ``data.coil_processing_mode`` moved into the
    ``processing:`` / ``coils:`` sub-blocks. Both reads below used the flat
    spelling, so on any real (Pydantic) config ``_get_attr_safe`` fell through to
    its default and the guard was **inert**: an ``rss_image`` / ``magnitude``
    target -- already an image -- was reported as needing an IFFT, which is
    precisely the mirror-image regression this module documents itself as
    preventing.

    Measured on ``exp_prcc_bloch_field`` before the fix::

        cfg.data.coils.processing_mode                    -> 'rss_image'
        _get_attr_safe(cfg.data, 'coil_processing_mode')  -> ''
        needs_ifft_for_visualization(cfg)                 -> (True, True)

    The legacy name is still read as a fallback so the duck-typed stubs this
    module is deliberately tolerant of keep working; a real config never reaches
    it, because the loader folds the legacy spelling into the block.
    """
    block_cfg = _get_attr_safe(data_cfg, block)
    value = _get_attr_safe(block_cfg, leaf) if block_cfg is not None else None
    if value is None:
        value = _get_attr_safe(data_cfg, legacy, default)
    return default if value is None else value


def _get_declared_path(obj: Any, path: str) -> Any:
    """Value at ``path``, but only when it was DECLARED rather than defaulted.

    The retired flat ``data.output_domain`` defaulted to ``None``, so P4 fell
    through to the heuristics whenever an arm stayed silent. Its canonical home
    ``data.domain.output`` defaults to ``'image'``: read plainly, P4 would fire on
    every config and leave P5/P6/P7 unreachable -- forcing every k-space arm to
    image, a worse regression than the stale read it replaced. Pydantic's
    ``model_fields_set`` is the only thing that separates a declaration from a
    default; for dicts and namespaces, presence already means declaration.
    """
    *blocks, leaf = path.split(".")
    owner = obj
    for segment in blocks:
        owner = _get_attr_safe(owner, segment)
        if owner is None:
            return None
    fields_set = getattr(owner, "model_fields_set", None)
    if fields_set is not None and leaf not in fields_set:
        return None
    return _get_attr_safe(owner, leaf)


def _registry_output_domain(model_type: str) -> str | None:
    """Look up ``output_domain`` capability from ``@register_model`` metadata.

    The ``@register_model`` decorator stores per-model capability flags on
    ``MODEL_REGISTRY`` (see ``src/models/registry.py``). When a generator
    declares its true output domain via the decorator (e.g.
    ``output_domain="kspace"``), that declaration is the single source of
    truth — domain inference must consult it before the hardcoded sets
    below, which exist only as a legacy fallback for unannotated models.

    Returns ``None`` when the model is not in the registry or has no
    declared output domain. Returns ``"image"``/``"kspace"`` otherwise.
    Tuple-valued declarations (multi-domain models) currently return
    ``None`` — those need explicit ``model.target_domain`` in the YAML.
    """
    try:
        from spectramr.models.registry import MODEL_REGISTRY
    except Exception:  # pragma: no cover - registry import should never fail
        return None
    entry = MODEL_REGISTRY.get(model_type)
    if entry is None:
        return None
    caps = entry.get("capabilities")
    if caps is None:
        return None
    domain = getattr(caps, "output_domain", None)
    if domain is None:
        return None
    if isinstance(domain, tuple):
        # Multi-domain model — cannot disambiguate without explicit YAML decl
        return None
    domain_str = str(domain).lower()
    if domain_str in ("image", "kspace"):
        return domain_str
    return None


def infer_output_domain(
    config: Any,
) -> Literal["image", "kspace"]:
    """Determine the output domain of the generator using a priority cascade.

    This is the SINGLE authoritative function for domain inference.
    All image logging, metric computation, and visualization code should
    call this instead of implementing ad-hoc heuristics.

    Priority cascade — **declarations first, then tables, then heuristics**:
        1. config.model.target_domain (explicit YAML declaration)
        2. ``@register_model`` ``output_domain`` capability (decorator metadata SSOT)
        3. config.data.domain.output (data pipeline declaration)
        4. Hardcoded model-type sets (KNOWN_KSPACE_/_IMAGE_OUTPUT_MODELS — legacy)
        5. Heuristic from physics config (enable_kspace_recon)
        6. Heuristic from data config (dataset_type == "kspace" OR normalize_kspace)
        7. Default: "image" (safest fallback — prevents IFFT on image data)

    3 and 4 were the other way round until #986. Measured over all 725 loadable
    arms in ``experiments/``, the swap changes **0** verdicts.

    .. warning::

       ``losses.policy.output_domain`` is **not a tier**, although it is an
       explicit declaration of exactly this quantity. Arms exist that declare it
       ``kspace`` while this function answers ``image`` from tier 4 -- three of
       the arms held by #937 are precisely that shape. Adding it is not a free
       change (it would move live arms), so it is tracked on #986 rather than
       done here; what is recorded here is that the omission is known, not an
       oversight to be "fixed" by whoever notices it next.

    Args:
        config: TrainingSettings or any object with model/data/physics attributes.

    Returns:
        "image" or "kspace" indicating the generator's output domain.
    """
    model_cfg = _get_attr_safe(config, "model")
    data_cfg = _get_attr_safe(config, "data")
    physics_cfg = _get_attr_safe(config, "physics")

    # ── Priority 1: Explicit target_domain declaration in YAML ──
    target_domain = _get_attr_safe(model_cfg, "target_domain")
    if target_domain is not None and str(target_domain).lower() in ("image", "kspace"):
        result = str(target_domain).lower()
        logger.debug("[DomainInference] P1 explicit target_domain='%s'", result)
        return result  # type: ignore[return-value]

    # ── Priority 2: ``@register_model`` decorator metadata (SSOT) ──
    model_type = _get_attr_safe(model_cfg, "model_type", "")
    model_type_lower = str(model_type).lower() if model_type else ""
    if model_type_lower:
        registry_domain = _registry_output_domain(model_type_lower)
        if registry_domain is not None:
            logger.debug(
                "[DomainInference] P2 model_type='%s' → %s (capability metadata)",
                model_type,
                registry_domain,
            )
            return registry_domain  # type: ignore[return-value]

    # ── Priority 3: Data output_domain declaration ──
    #
    # This ran BELOW the hardcoded legacy sets until #986, which put a table of
    # model names ahead of an explicit statement the arm makes about itself. A
    # fallback that outranks a declaration is not a fallback. P1 and P2 are
    # declarations *about the model*; this is a declaration *about the run*;
    # the legacy sets below are neither, and are documented as covering
    # unannotated models.
    #
    # Measured over all 725 loadable arms in ``experiments/`` at the time of
    # the swap: **0 arms change verdict**, because no arm declares
    # ``data.domain.output`` AND carries a model_type in a legacy set that
    # disagrees. So this is a correctness ordering, not a migration -- it
    # decides the next arm that declares both, not any arm that exists.
    data_output_domain = _get_declared_path(data_cfg, "domain.output")
    if data_output_domain is not None and str(data_output_domain).lower() in (
        "image",
        "kspace",
    ):
        result = str(data_output_domain).lower()
        logger.debug("[DomainInference] P3 data.domain.output='%s'", result)
        return result  # type: ignore[return-value]

    # ── Priority 4: Hardcoded model-type sets (legacy fallback) ──
    #
    # These sets encode a *usage assumption*, not an architectural fact: the
    # entries are largely generic backbones, and the same backbone trained on
    # k-space outputs k-space. They are kept because removing entries is NOT
    # behaviour-neutral -- dropping just ``siren``, ``hyper_mamba_unet`` and
    # ``cross_attention_oracle_unet`` flips 13 arms from image to kspace, and
    # spot-checking three of those shows the flips are not uniformly right:
    # one agrees with the arm's own ``losses.policy.output_domain: kspace``,
    # one contradicts an arm declaring ``dataset_type: image`` *and*
    # ``losses.policy.output_domain: image``, and one lands on an arm whose
    # own declarations disagree with each other. See #986.
    if model_type_lower:
        if model_type_lower in KNOWN_KSPACE_OUTPUT_MODELS:
            logger.debug(
                "[DomainInference] P4 model_type='%s' → kspace (legacy set)",
                model_type,
            )
            return "kspace"
        if model_type_lower in KNOWN_IMAGE_OUTPUT_MODELS:
            logger.debug(
                "[DomainInference] P4 model_type='%s' → image (legacy set)",
                model_type,
            )
            return "image"

    # ── Priority 5: Physics config heuristic ──
    if physics_cfg is not None:
        kspace_cfg = _get_attr_safe(physics_cfg, "kspace")
        if kspace_cfg is not None:
            enable_kspace_recon = _get_attr_safe(kspace_cfg, "enable_kspace_recon", False)
            if enable_kspace_recon:
                logger.debug(
                    "[DomainInference] P5 physics.kspace.enable_kspace_recon=True → kspace"
                )
                return "kspace"

    # ── Priority 6: Data pipeline heuristic ──
    # If dataset is k-space, the model likely outputs k-space too.
    # NOTE: input_type="image" means real-stacked channels for Conv2d
    # compatibility, NOT that the output is image domain. Image-output
    # models already short-circuit at P2/P3.
    #
    # Audit-2026-05-14 §1 round-6: ``coil_processing_mode`` overrides
    # the kspace-by-dataset_type heuristic. Modes ``rss_image`` and
    # ``magnitude`` apply an IFFT inside the dataset's TorchIO
    # transforms so the model sees image-domain data — declaring
    # ``dataset_type: kspace`` alongside ``rss_image`` is the m4raw
    # cross-contrast / experiment_113 / experiment_114 pattern. Without
    # this override the model output_domain falls through to "kspace"
    # and downstream visualisation IFFTs an already-image tensor.
    dataset_type = _get_attr_safe(data_cfg, "dataset_type", "")
    coil_processing_mode = str(
        _data_leaf(data_cfg, "coils", "processing_mode", "coil_processing_mode", "") or ""
    ).lower()

    if loader_serves_kspace(config):
        logger.debug(
            "[DomainInference] P6 dataset_type='%s' coil_mode='%s' → kspace",
            dataset_type,
            coil_processing_mode or "none",
        )
        return "kspace"

    if coil_processing_mode in IMAGE_DOMAIN_COIL_MODES:
        logger.debug(
            "[DomainInference] P6 coil_processing_mode='%s' forces image",
            coil_processing_mode,
        )
        return "image"

    # ── Priority 7: Default fallback ──
    logger.debug("[DomainInference] P7 default → image")
    return "image"


def loader_serves_kspace(config: Any) -> bool:
    """Does the dataloader hand the strategy a k-space tensor?

    The ONE owner of that question (non-negotiable 17). ``infer_output_domain``'s
    Priority-6 heuristic and :func:`needs_ifft_for_visualization` both used to
    recompute it inline from the same four leaves, and
    :func:`needs_kspace_to_image_bridge` is a third consumer — three copies of a
    predicate is how they start to disagree.

    ``coil_processing_mode`` in ``rss_image`` / ``magnitude`` wins over
    ``dataset_type: kspace``: those modes IFFT inside the dataset's TorchIO chain,
    so the tensor that reaches the model is already an image.

    Args:
        config: the resolved training config.

    Returns:
        ``True`` when the batch the strategy receives is k-space.
    """
    data_cfg = _get_attr_safe(config, "data")
    dataset_type = str(_get_attr_safe(data_cfg, "dataset_type", "") or "").lower()
    normalize_kspace = _data_leaf(
        data_cfg,
        "processing",
        "enable_kspace_normalization",
        "normalize_kspace",
        False,
    )
    coil_processing_mode = str(
        _data_leaf(data_cfg, "coils", "processing_mode", "coil_processing_mode", "") or ""
    ).lower()
    return (
        dataset_type in ("kspace", "kspace_paired") or bool(normalize_kspace)
    ) and coil_processing_mode not in IMAGE_DOMAIN_COIL_MODES


def needs_kspace_to_image_bridge(config: Any) -> bool:
    r"""Must the batch be ``ifft2c``'d before it reaches the network's ``forward``?

    True when the loader serves k-space and the model is registered as consuming
    the **image** domain — i.e. the adjoint :math:`A^H` is the strategy's job, not
    the network's.

    This decision used to have no owner at all. It rode on ``use_dc``, the
    data-consistency flag: ``_prepare_generator_inputs`` applied ``ifft2c`` only
    ``if use_dc``, and ``use_dc`` is set from ``batch['use_dc']``, which is
    ``strategy.dc_layer is not None``. ``initialize_data_consistency`` sets
    ``dc_layer = None`` whenever the generator has no built-in DC module — with a
    log *warning*, not a raise — so an arm declaring
    ``physics.data_consistency.enabled: true`` over a plain U-Net got **no DC and
    no adjoint**, and the network was handed raw k-space while every consumer
    downstream (the loss's Fourier bridge, the metrics, the previewer) read its
    output as an image. Observed on ``ei_unet_m4raw_r4`` / ``ei_control_mc_only``
    (cluster run 2026-09-20): the reported reconstruction is the DC-spike blob,
    because it *is* k-space.

    Reading the registry rather than ``model.input_type`` is deliberate: that
    field defaults to ``"image"`` and 165 of 672 corpus arms never set it, so a
    config-level read would flip the bridge on for arms that never asked. The
    ``@register_model`` capability is a positive declaration by the model itself
    — ``standard_unet`` says ``input_domain='image'``, ``complex_unet`` says
    ``'kspace'`` — and only 117 of 593 registered models claim ``image``.

    Args:
        config: the resolved training config.

    Returns:
        ``True`` when the strategy must apply the k-space → image adjoint.
    """
    if not loader_serves_kspace(config):
        return False

    model_cfg = _get_attr_safe(config, "model")
    model_type = _get_attr_safe(model_cfg, "model_type", None)
    if not model_type:
        return False

    try:
        from spectramr.models.registry import get_model_capabilities

        caps = get_model_capabilities(str(model_type))
    except Exception:  # pragma: no cover - registry import is best effort
        return False

    declared = getattr(caps, "input_domain", None) if caps is not None else None
    if declared is None:
        # Unannotated: say nothing rather than guess. 443 of 593 models are in
        # this bucket, and flipping a domain bridge on a guess is the failure
        # this function exists to end.
        return False
    domains = declared if isinstance(declared, tuple) else (declared,)
    return "image" in domains


def needs_ifft_for_visualization(
    config: Any,
) -> tuple[bool, bool]:
    """Determine whether predictions and/or targets need IFFT before visualization.

    Returns:
        Tuple of (needs_ifft_predictions, needs_ifft_targets).
        - needs_ifft_predictions: True if generator output is k-space → apply IFFT
        - needs_ifft_targets: True if targets from dataloader are k-space → apply IFFT

    .. note::

        Audit-2026-05-14 §1 round-6: ``data.coil_processing_mode`` can
        override ``data.dataset_type=kspace`` into image-domain output.
        ``"rss_image"`` (and ``"magnitude"``) apply an IFFT inside the
        dataset's TorchIO transform pipeline so the model sees image-
        domain data, even though ``dataset_type`` still reads "kspace".
        Pre-2026-05-15 ``needs_ifft_for_visualization`` ignored
        ``coil_processing_mode`` and returned ``needs_ifft_targets=True``
        on these configs — the validation writer then IFFT'd an
        already-image tensor, producing the spectral / tiled-noise
        aliasing pattern visible in 10 of the smoke fakes
        (``experiment_113_graph_cuts_neural``,
        ``experiment_114_mamba``, etc.). The fix below treats
        ``rss_image`` / ``magnitude`` as image-domain coil-processing
        modes; the dataset emits image data and the visualizer must
        not re-transform it.
    """
    output_domain = infer_output_domain(config)
    targets_are_kspace = loader_serves_kspace(config)

    needs_ifft_preds = output_domain == "kspace"
    needs_ifft_targets = targets_are_kspace

    logger.debug(
        "[DomainInference] Visualization: IFFT preds=%s, IFFT targets=%s "
        "(output_domain=%s)",
        needs_ifft_preds,
        needs_ifft_targets,
        output_domain,
    )

    return needs_ifft_preds, needs_ifft_targets


def looks_like_kspace(mag: Any) -> bool:
    """Does a (pre-IFFT) magnitude map carry k-space's DC-energy signature?

    Tensor-level companion to the config-level :func:`needs_ifft_for_visualization`.
    k-space concentrates almost all energy in the zero-frequency (DC) pixel at the
    centre, so the central magnitude is orders of magnitude above the mean
    (empirically ≈ 170 for a 64² phantom, into the thousands at 256²); an image —
    even a centred brain with a dark ventricle — sits at ≈ 1-3. The two-orders-of-
    magnitude gap is scale-stable, so a threshold of 10 separates them with margin.

    Used as a *veto* on a spurious IFFT: when a tensor is flagged k-space by the
    config (``dataset_type: kspace``) but a strategy already handed an
    image-domain target/prediction to the visualiser, IFFT'ing it would produce a
    black DC-spike / blob PNG (smoke 2026-06-12 debug snapshots; 2026-06-15
    validation real-images for exp_p7 / exp_p3). Returns ``True`` only when the
    magnitude genuinely looks like k-space, so image-domain tensors are left alone.

    Args:
        mag: a real-valued magnitude tensor ``[B, C, H, W]`` (or ``[H, W]``).

    Returns:
        ``True`` if the DC-energy signature is present (IFFT it), else ``False``.
    """
    import torch

    x = mag.detach().float() if hasattr(mag, "detach") else torch.as_tensor(mag).float()
    while x.dim() > 2:
        x = x.mean(dim=0)
    if x.dim() != 2:
        return True  # cannot assess → preserve prior (IFFT) behaviour
    gmean = float(x.mean())
    if gmean <= 0:
        return False
    h, w = x.shape
    cy, cx = h // 2, w // 2
    centre = float(x[max(0, cy - 1) : cy + 2, max(0, cx - 1) : cx + 2].max())
    return (centre / gmean) > 10.0


def metric_transform_produced_image(before: Any, after: Any) -> bool:
    """Did a metric transform actually convert *before* into an image tensor?

    ``_apply_metric_transforms`` has four paths that return their INPUT
    unchanged -- ``metrics.domain == "none"``, ``validation.domain == "none"``,
    no resolvable ``transform_name``, and a non-complex non-2-channel prediction
    with nothing configured. The diffusion validation path asserted
    ``is_preds_image = True`` after calling it regardless, so on a no-op the flag
    was a lie about the domain (#927).

    What that cost, measured on ``experiment_11_attention_none`` (cluster run
    2026-08-08): the transform handed back ``(36, 8, 256, 256)`` float32 -- still
    k-space -- and all 135 validation-image writes hit the
    ``kspace_to_image(already_image=True)`` guard and raised, so the run emitted
    ZERO images, while PSNR computed over k-space read 58 dB and
    ``robust_mri_psnr`` went ``NaN``. The guard was doing its job; the assertion
    upstream was the defect.

    The postcondition this checks: every metric transform in this codebase ends
    in a magnitude, so it necessarily moves the tensor's **shape** (the coil axis
    collapses under RSS / SENSE-combine, or halves when interleaved real/imag
    pairs into complex coils) or its **dtype** (complex -> real). A result that
    moves neither did not change domain.

    Deliberately structural, not statistical: :func:`looks_like_kspace` answers a
    different question (does this magnitude carry the DC signature) and needs a
    host sync to do it. This one is free and cannot be fooled by an unusually
    flat spectrum.

    Args:
        before: the tensor handed to the transform.
        after: what it returned.

    Returns:
        ``True`` when the transform demonstrably produced an image-domain
        tensor. Never raises -- it runs inside validation and must not be the
        thing that fails.
    """
    if after is before:
        return False
    before_shape = getattr(before, "shape", None)
    after_shape = getattr(after, "shape", None)
    if before_shape is None or after_shape is None:
        # Not tensors (a stub, a stand-in). Fall back to the identity answer
        # above rather than guessing a domain from something with no shape.
        return after is not before
    if tuple(before_shape) != tuple(after_shape):
        return True
    return getattr(before, "dtype", None) != getattr(after, "dtype", None)


__all__ = [
    "KNOWN_IMAGE_OUTPUT_MODELS",
    "KNOWN_KSPACE_OUTPUT_MODELS",
    "infer_output_domain",
    "loader_serves_kspace",
    "looks_like_kspace",
    "metric_transform_produced_image",
    "needs_ifft_for_visualization",
    "needs_kspace_to_image_bridge",
]
