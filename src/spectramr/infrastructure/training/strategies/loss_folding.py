"""Shared builder-image-loss folding (the loss-SSOT seam, factored out).

The declarative ``losses.image_losses`` block is composed by ``LossBuilder`` into
``env.losses`` as bare ``name -> nn.Module`` (per-entry weight stripped). Strategies
that compute their objective *inline* (the "type-B" field strategies, and the GAN-
based cross-field arms) would otherwise leave that block an inert decoy (pitfall
#16). These two pure functions are the single home for folding it back on:

* :func:`declared_loss_weights` re-reads the stripped per-entry weights.
* :func:`fold_builder_image_losses` applies each builder module to the strategy's
  grad-carrying ``(pred, target)`` pair, honouring the ``loss_schedule`` curriculum
  overrides (``loop_state.loss_weight_overrides``) OVER the static weight, and skips
  the universal inline fidelity terms ``{l1, l2, mse}`` so the ubiquitous
  ``image_losses: [{l1, 1.0}]`` placeholder never double-counts.

Both ``ReconstructionTrainingStrategy`` (single-optimizer recon family) and
``FieldCocycleTranslationStrategy`` (two-optimizer GAN family) delegate here, so the
curriculum-aware folding behaviour is defined exactly once (SSOT, pitfall #12).
"""

from __future__ import annotations

from typing import Any

import torch

from spectramr.models.losses.weights import build_loss_weight_table, canonical_loss_name

# The universal inline fidelity terms every type-B strategy computes directly (and
# every mrixfields arm carries as a ``losses.image_losses: [{l1, 1.0}]`` placeholder).
# Skipped so adopting the seam never double-counts the inline term — only ADDED terms
# (hfen / ms_ssim / lpips / sobel_edge / ...) fold in.
#
# CANONICAL names only. This was previously matched against the RAW yaml name, so an arm
# spelling the same loss ``mae`` (a registry alias of ``l1``) sailed past the skip-list
# and got folded ON TOP of the L1 the strategy already computed inline — a silent
# double-count. Canonicalising closes that for every present and future alias.
_INLINE_MANAGED = frozenset({"l1", "l2"})


def _is_inline_managed(name: str, inline_managed: frozenset[str]) -> bool:
    """True when ``name`` (under any alias) is a term the strategy computes inline."""
    return canonical_loss_name(name) in inline_managed


def inline_managed_with(*extra: str) -> frozenset[str]:
    """:data:`_INLINE_MANAGED` widened by a strategy's own inline term names.

    A strategy that computes a *registered* loss inline (the cocycle residual, the
    Bloch seg-consistency term, ...) must declare it here before that term may also
    appear in ``losses.image_losses`` — otherwise the fold would apply the module on
    top of the inline computation and double-count it.

    Declaring the term on the losses surface is what makes it curriculum-targetable:
    ``LossScheduleController`` resolves a rule's base weight through the loss-weight
    SSOT, which only sees ``losses.*``. A ``loss_schedule`` rule aimed at a term that
    lives solely in a ``training.<block>.*_weight`` field cannot be resolved and
    raises at the trigger iteration (CLAUDE.md #13b).
    """
    return _INLINE_MANAGED | frozenset(canonical_loss_name(n) for n in extra)


def declared_inline_losses(cls: type, *, below: type | None = None) -> frozenset[str] | None:
    """The ``inline_losses`` a strategy class declares for itself, canonicalised.

    Walks the MRO and returns the first declaration found in a class's OWN
    ``__dict__`` (``None`` values are "undeclared" and skipped). ``below`` stops
    the walk before that class: the fold passes its own base so a subclass that
    has not declared keeps the universal ``{l1, l2}`` skip set instead of
    inheriting the base's (empty) declaration and double-counting its inline L1.
    The witness passes no ``below`` and so sees the base's declaration for the
    plain reconstruction arms.
    """
    for klass in cls.__mro__:
        if klass is below:
            return None
        declared = klass.__dict__.get("inline_losses")
        if declared is not None:
            return frozenset(canonical_loss_name(n) for n in declared)
    return None


def declared_folds_image_losses(cls: type) -> bool | None:
    """The nearest ``folds_image_losses`` declaration along the MRO, or None."""
    for klass in cls.__mro__:
        declared = klass.__dict__.get("folds_image_losses")
        if declared is not None:
            return bool(declared)
    return None


def declares_inline_objective(cls: type) -> bool:
    """True when an EMPTY built loss stack is this strategy's declared design.

    The two declarations answer this only together: a non-None ``inline_losses``
    says the strategy has stated which terms it computes itself, and a False
    ``folds_image_losses`` says it consumes none of the builder's modules. A
    strategy asserting both owns its whole objective, so a builder that produced
    nothing has not failed. Either reader returning None leaves the strategy
    UNDECLARED and therefore not exempt -- silence is not a claim of ownership
    (#1918), which is the distinction ``LossBuilder.validate()`` rests on.
    """
    return declared_inline_losses(cls) is not None and declared_folds_image_losses(cls) is False


def unreachable_image_losses(
    names: Any,
    inline: frozenset[str],
    folds: bool,
    is_registered: Any,
) -> list[str]:
    """Declared image-loss names that reach the objective by neither route.

    A name reaches when the strategy computes it inline, or when the strategy
    folds the builder's modules and the registry can build it. Everything else
    is silently dropped at runtime -- the facade shape this rule exists to name.
    """
    out: list[str] = []
    for raw in names:
        canon = canonical_loss_name(str(raw))
        if canon in inline:
            continue
        if folds and is_registered(str(raw)):
            continue
        out.append(str(raw))
    return out


def scheduled_overrides(obj: Any) -> dict[str, float]:
    """The live ``loss_schedule`` curriculum weights (``loop_state.loss_weight_overrides``).

    Single reader for the per-step ``{term: weight}`` map the ``LossScheduleController``
    publishes; empty ``{}`` when no schedule is active. Shared by every inline-folding
    strategy so the accessor is defined once (a rename of the SSOT field changes one
    place, not three).
    """
    return getattr(getattr(obj, "loop_state", None), "loss_weight_overrides", None) or {}


def declared_loss_weights(config: Any) -> dict[str, float]:
    """Map declarative loss name -> weight, via the loss-weight SSOT.

    This used to read the per-entry ``weight:`` straight off the YAML list — which made
    it the ONLY resolver treating that field as the real lambda while every other path
    treated it as a mere enable-gate and took the weight from ``lambda_<name>``. Same
    config, two different objectives depending on which strategy you landed in. It now
    delegates to :func:`~spectramr.models.losses.weights.build_loss_weight_table`, so the
    fold applies exactly the weight the rest of the system applies.

    Keys are canonical (``mse`` collapses onto ``l2``), which is also what closes the
    ``mae``-alias double-count hole in :data:`_INLINE_MANAGED`.
    """
    table = build_loss_weight_table(getattr(config, "losses", None))
    return {name: spec.weight for name, spec in table.items()}


def fold_builder_image_losses(
    env_losses: dict[str, Any] | None,
    declared: dict[str, float],
    scheduled: dict[str, float],
    pred: torch.Tensor,
    target: torch.Tensor,
    components: dict[str, torch.Tensor],
    inline_managed: frozenset[str] | None = None,
) -> torch.Tensor | None:
    """Fold the builder-composed declarative image losses onto an inline total.

    Args:
        env_losses: the ``env.losses`` map (``name -> nn.Module``); ``None``/empty
            returns ``None`` so inline-only arms stay byte-identical.
        declared: static per-term weights from :func:`declared_loss_weights`.
        scheduled: dynamic ``loop_state.loss_weight_overrides`` (curriculum); a
            scheduled weight supersedes the static one, empty map = static behaviour.
        pred, target: the strategy's grad-carrying image pair.
        components: mutated in place — each folded term is written as
            ``loss_<name>`` (detached) and the weighted sum as ``loss_builder_aux``.
        inline_managed: canonical names this strategy computes inline, skipped so
            declaring them never double-counts. Defaults to :data:`_INLINE_MANAGED`
            (``l1``/``l2``); a strategy with extra inline terms passes
            ``_INLINE_MANAGED | {...}``. Declaring such a term on
            ``losses.image_losses`` is what lets the ``loss_schedule`` controller
            resolve its base weight through the SSOT (see
            :func:`~spectramr.models.losses.weights.resolve_loss_weight`) instead of
            raising mid-run — the term's value stays the strategy's to compute.

    Returns:
        The weighted sum of the folded terms (grad-carrying), or ``None`` when
        nothing folds. A builder loss that errors is raised, never silently dropped.
    """
    if not env_losses:
        return None
    skip = _INLINE_MANAGED if inline_managed is None else inline_managed
    extra: torch.Tensor | None = None
    for name, module in env_losses.items():
        if _is_inline_managed(name, skip):
            continue
        # `declared` is keyed canonically (it comes from the weight table), while
        # `env.losses` is keyed by the raw YAML name — so canonicalise before lookup.
        canon = canonical_loss_name(name)
        # Scheduled (curriculum) override supersedes the static declared weight.
        weight = scheduled.get(canon, scheduled.get(name, declared.get(canon, 1.0)))
        if not weight:
            continue
        value = module(pred, target)
        if isinstance(value, dict):
            # Explicit None checks, NOT `a or b`: a legitimately-zero loss tensor is
            # falsy, so `value.get("loss") or next(...)` would fold an arbitrary
            # diagnostic entry (e.g. a PSNR of 40) in place of the real 0.0 loss.
            picked = value.get("loss")
            if picked is None:
                picked = value.get("loss_total")
            if picked is None:
                picked = next(iter(value.values()))
            value = picked
        elif isinstance(value, (tuple, list)):
            value = value[0]
        components[f"loss_{name}"] = value.detach()
        term = weight * value
        extra = term if extra is None else extra + term
    if extra is not None:
        components["loss_builder_aux"] = extra.detach()
    return extra
