"""Experiment Spec Card — synthesized human-readable reference.

Printed at the start of every audit run, info-only severity, never
blocks. Gives the user (and any reviewer) a single legible reference
for *what shape, domain, and normalization the experiment actually
operates on at each layer*. See
``docs/superpowers/specs/2026-05-05-experiment-spec-card-and-adapters-design.md``.

Phase 1: capabilities are read but never enforced. Phase 2 will add
``check_data_model_compatibility`` that uses the same derivation logic
to actually fail the audit on mismatch.
"""

from __future__ import annotations

from typing import Any

from spectramr.config.schemas.loss import LOSS_LIST_DOMAINS
from spectramr.models.capabilities import ModelCapabilities


def _safe_get(obj: Any, *names: str, default: Any = None) -> Any:
    """First non-``None`` of ``names``, each a flat attribute or a dotted path.

    Dotted support exists because the phase 8-12 decomposition moved most of
    these leaves into sub-blocks. This reads by STRING, so a stale name does not
    raise -- it returns ``default``, and the card renders ``None`` for a fact the
    YAML plainly declares. Callers pass the canonical path first and the legacy
    spelling second, which keeps unmigrated configs readable while the fold
    posture lasts.
    """
    for n in names:
        cur: Any = obj
        for part in n.split("."):
            if not hasattr(cur, part):
                cur = None
                break
            cur = getattr(cur, part)
        if cur is not None:
            return cur
    return default


def derive_loader_output_domain(config: Any) -> str:
    """The signal domain the loader hands the model, before any adapter.

    First pass: ``data.dataset_type`` — the k-space family (m4raw, fastmri,
    ``kspace``, ``fastmri_kspace``) is ``kspace``, ``pde_synthetic`` is
    ``pde_grid``, everything else is ``image``. One home for the family
    (``spectramr.data.signal_domain``): the data layer needs the same fact and
    cannot import ``infrastructure/`` (#5). Second pass: a coil-processing mode
    of ``rss_image`` or ``magnitude`` turns k-space into a 1-channel magnitude
    image before the model sees it, whatever the dataset type.

    The spec card and the compatibility-matrix resolver both read this; the
    resolver used to proxy the data domain from ``model.target_domain`` (the
    model OUTPUT), which rejected five physics arms whose loader emits images
    under ``rss_image`` (2026-09-03).
    """
    from spectramr.data.signal_domain import is_kspace_dataset_type

    data = getattr(config, "data", None)
    dataset_type = _safe_get(data, "dataset_type")
    coil_mode = _safe_get(data, "coils.processing_mode", "coil_processing_mode")
    domain = "image"
    if is_kspace_dataset_type(dataset_type):
        domain = "kspace"
    if dataset_type and dataset_type == "pde_synthetic":
        domain = "pde_grid"
    if coil_mode in ("rss_image", "magnitude"):
        domain = "image"
    return domain


def _derive_data_form(config: Any) -> dict[str, Any]:
    """Infer a coarse DataForm from ``config.data``.

    Defensive: anything missing or odd resolves to ``None`` rather
    than raising, so the card always renders.
    """
    data = getattr(config, "data", None)
    if data is None:
        return {"present": False}

    patch_size = _safe_get(data, "sampling.patch_size", "patch_size") or ()
    # Spatial dims = number of >1 entries in patch_size. patch_size is
    # normalized to (W, H, D) by the schema; D=1 means 2D, D>1 means 3D.
    spatial_dims: int | None = None
    if patch_size:
        spatial_dims = sum(1 for d in patch_size if isinstance(d, int) and d > 1)
        # Edge case: 2D-from-3D patch (W,H,1) is still operationally 2D
        if len(patch_size) == 3 and patch_size[2] == 1 and spatial_dims == 2:
            spatial_dims = 2

    dataset_type = _safe_get(data, "dataset_type")
    domain_inferred = derive_loader_output_domain(config)

    # Surface the unified physics.coil_processing block (compression / estimation
    # / combine) alongside the legacy coil_processing_mode so the spec card
    # reflects the new config SSOT.
    coil_proc = getattr(getattr(config, "physics", None), "coil_processing", None)
    coil_processing_block = (
        {
            "compression": getattr(getattr(coil_proc, "compression", None), "method", None),
            "estimation": getattr(getattr(coil_proc, "estimation", None), "method", None),
            "combine": getattr(getattr(coil_proc, "combine", None), "method", None),
            "output": (
                getattr(getattr(coil_proc, "output", None), "domain", None),
                getattr(getattr(coil_proc, "output", None), "channels", None),
            ),
        }
        if coil_proc is not None
        else None
    )

    return {
        "present": True,
        "dataset_type": dataset_type,
        "coil_processing_block": coil_processing_block,
        "patch_size": tuple(patch_size) if patch_size else None,
        "spatial_dims": spatial_dims,
        "domain_inferred": domain_inferred,
        "coil_processing_mode": _safe_get(data, "coils.processing_mode", "coil_processing_mode"),
        "normalization_type": _safe_get(
            data, "processing.normalization_type", "normalization_type"
        ),
        "target_channels": _safe_get(data, "domain.target_channels", "target_channels"),
        "in_channels": _safe_get(data, "in_channels"),
        "voxel_mm": _safe_get(data, "voxel_mm", "voxel_size"),
        "multi_contrast_enabled": (
            _safe_get(getattr(data, "multi_contrast", None), "enabled", default=False)
            if hasattr(data, "multi_contrast")
            else False
        ),
    }


def _derive_model_form(config: Any) -> dict[str, Any]:
    from spectramr.models.registry import MODEL_REGISTRY

    model = getattr(config, "model", None)
    if model is None:
        return {"present": False}

    model_type = _safe_get(model, "model_type")
    entry = MODEL_REGISTRY.get(model_type) if model_type else None
    caps: ModelCapabilities | None = None
    if entry:
        c = entry.get("capabilities")
        if isinstance(c, ModelCapabilities) and any(
            getattr(c, f) is not None for f in c.__dataclass_fields__
        ):
            caps = c

    return {
        "present": True,
        "model_type": model_type,
        "in_channels": _safe_get(model, "in_channels"),
        "out_channels": _safe_get(model, "out_channels"),
        "spatial_dims_declared": _safe_get(model, "spatial_dims"),
        "input_type": _safe_get(model, "input_type"),
        "registered": entry is not None,
        "training_mode": entry.get("mode") if entry else None,
        "capabilities": caps,
    }


def _derive_loss_form(config: Any) -> dict[str, Any]:
    losses = getattr(config, "losses", None)
    if losses is None:
        return {"present": False}

    # The seed keys and the walk below both come from LOSS_LIST_DOMAINS, so they
    # cannot drift apart. Seeding by hand is how ``latent_losses`` went missing from
    # the card while the schema declared it (#1924) -- and note that seeding matters
    # independently of the walk: ``out[k].append`` below would KeyError on a list the
    # seed omits.
    out: dict[str, Any] = {
        "present": True,
        "output_domain": _safe_get(losses, "policy.output_domain", "output_domain"),
    }
    out.update({name: [] for name in LOSS_LIST_DOMAINS})
    for k in LOSS_LIST_DOMAINS:
        for entry in getattr(losses, k, None) or []:
            if getattr(entry, "enabled", True) is False:
                continue
            out[k].append(
                {
                    "name": getattr(entry, "name", "<unknown>"),
                    "weight": getattr(entry, "weight", None),
                }
            )
    return out


def _format_card(
    data: dict[str, Any],
    model: dict[str, Any],
    loss: dict[str, Any],
    adapters: dict[str, Any] | None = None,
    name: str | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"[SPEC] {name or '<unnamed>'}")

    # Data
    if data.get("present"):
        ds = data.get("dataset_type") or "?"
        ps = data.get("patch_size")
        sd = data.get("spatial_dims")
        spatial_str = f"{sd}D" if sd else "?D"
        lines.append("  data:")
        lines.append(f"    dataset_type:        {ds}")
        lines.append(
            f"    spatial:             {spatial_str} {('volume' if sd == 3 else 'slice') if sd else ''}, patch={ps}"
        )
        lines.append(f"    domain (inferred):   {data.get('domain_inferred')}")
        lines.append(f"    target_channels:     {data.get('target_channels')}")
        if data.get("coil_processing_mode") and data["coil_processing_mode"] != "none":
            lines.append(f"    coil_processing:     {data['coil_processing_mode']}")
        cp = data.get("coil_processing_block")
        if cp is not None:
            out = cp.get("output") or (None, None)
            lines.append(
                "    coil_pipeline:       "
                f"compress={cp.get('compression')}, "
                f"estimate={cp.get('estimation')}, "
                f"combine={cp.get('combine')}, "
                f"output={out[0]}/{out[1]}"
            )
        if data.get("normalization_type") and data["normalization_type"] != "none":
            lines.append(f"    normalization:       {data['normalization_type']}")
        if data.get("multi_contrast_enabled"):
            lines.append("    multi_contrast:      enabled")
    else:
        lines.append("  data:                <no data section>")

    # Model
    if model.get("present"):
        lines.append("  model:")
        lines.append(f"    model_type:          {model.get('model_type')}")
        sd_decl = model.get("spatial_dims_declared")
        lines.append(f"    spatial_dims (decl): {sd_decl}")
        lines.append(
            f"    channels:            in={model.get('in_channels')} out={model.get('out_channels')}"
        )
        caps = model.get("capabilities")
        if caps is None:
            lines.append("    capabilities:        unannotated (audit cross-checks skipped)")
        else:
            lines.append(
                f"    capabilities:        spatial_dims={caps.spatial_dims} "
                f"input_domain={caps.input_domain} output_domain={caps.output_domain}"
            )
            # Surface mismatches inline so they're visible without running the
            # Phase 2 enforcement checks.
            data_sd = data.get("spatial_dims")
            if (
                caps.spatial_dims is not None
                and data_sd is not None
                and data_sd not in caps.spatial_dims
            ):
                lines.append(
                    f"    ⚠ MISMATCH:          data is {data_sd}D, "
                    f"model declares spatial_dims={caps.spatial_dims}"
                )
    else:
        lines.append("  model:               <no model section>")

    # Losses
    if loss.get("present"):
        lines.append("  losses:")
        lines.append(f"    output_domain:       {loss.get('output_domain') or '<unset>'}")
        for k in LOSS_LIST_DOMAINS:
            for e in loss.get(k, []):
                lines.append(f"    {k:<19} {e['name']} (weight={e['weight']})")

    # Adapters: walk declared chains, surface side-effects from registry.
    from spectramr.data.adapters.registry import get_adapter_capabilities

    adapter_hooks = (
        "pre_model",
        "post_model",
        "pre_loss_pred",
        "pre_loss_target",
        "pre_metric",
    )
    declared_any = False
    if adapters is not None:
        adapter_lines: list[str] = []
        all_side_effects: list[str] = []
        for hook in adapter_hooks:
            steps = getattr(adapters, hook, None) or []
            steps = [s for s in steps if getattr(s, "enabled", True)]
            if not steps:
                continue
            declared_any = True
            names = [s.name for s in steps]
            adapter_lines.append(f"    {hook:<19} [{', '.join(names)}]")
            for s in steps:
                caps = get_adapter_capabilities(s.name)
                if caps is not None:
                    all_side_effects.extend(caps.side_effects)
        if declared_any:
            lines.append("  adapters:")
            lines.extend(adapter_lines)
            if all_side_effects:
                lines.append(f"    side_effects:        {', '.join(sorted(set(all_side_effects)))}")
    if not declared_any:
        lines.append("  adapters:            none declared")

    return "\n".join(lines)


def synthesize_spec_card(config: Any) -> str:
    """Build the Spec Card string for ``config``.

    Pure: never raises (defensive _safe_get everywhere) and never
    mutates ``config``. Caller emits to stderr / writes into the
    audit JSON under the ``spec_card`` key.
    """
    metadata = getattr(config, "metadata", None)
    if isinstance(metadata, dict):
        name = metadata.get("name")
    else:
        name = getattr(metadata, "name", None) if metadata else None
    data = _derive_data_form(config)
    model = _derive_model_form(config)
    loss = _derive_loss_form(config)
    adapters = getattr(config, "adapters", None)
    card = _format_card(data, model, loss, adapters=adapters, name=name)

    # Reuse the IR (WS-X "reuse the IR everywhere"): append the resolved-context
    # view the compatibility matrix evaluates, so the reviewer reads ONE coherent
    # picture (data / model.io / physics / paradigms / adapter-chain) and can see
    # why a combination passed or was rejected. Fail-soft — resolution must never
    # break card synthesis (this function's contract is "never raises").
    try:
        from spectramr.infrastructure.validation.context_resolver import resolve

        rendered = resolve(config).render()
        card += "\n\n  [RESOLVED CONTEXT — what the compatibility matrix checks]\n"
        card += "\n".join("    " + ln for ln in rendered.splitlines())
    except Exception:  # pragma: no cover - defensive; keep the card pure
        pass
    return card


__all__ = ["synthesize_spec_card"]
