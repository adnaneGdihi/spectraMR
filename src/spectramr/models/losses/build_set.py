"""The loss build set: which declared terms the builder instantiates.

Owner of the skip sets (terms a strategy or computer computes inline) and of
:func:`declared_build_set`, the projection of the loss-weight table onto the
modules ``LossBuilder`` must build. ``LossConfigSchema`` no longer keeps a second
walk over the ``enable_*`` flags (#1966); the table is the one reader.
"""

from __future__ import annotations

import re
from typing import Any

from spectramr.models.losses.reporting import is_active
from spectramr.models.losses.weights import (
    LossWeightSpec,
    build_loss_weight_table,
    canonical_loss_name,
)

#: Losses that ``LossBuilder`` must NOT instantiate as standalone modules
#: because a training strategy computes them inline (they need gradients,
#: encoder features, an injected ``context`` dict, or are bundled inside a
#: composite GAN loss). This is the SSOT for the skip set — the loss-coverage
#: auditor (``infrastructure/loss_audit.py``) imports it so the two cannot
#: drift (the F41 2026-05-23 incident, re-broken when this set grew but the
#: auditor's private copy did not — re-derived 2026-06-11).
STRATEGY_MANAGED_LOSSES: frozenset[str] = frozenset(
    {
        # Strategy-managed (built by the training strategy)
        "calibrated_gain_bloch",  # CalibratedQMapStrategy: needs the render and gain context
        "commitment",
        "codebook",
        "r1",
        "patch_nce",
        "reconstruction",
        "marker",
        "prior",
        "distill",
        "sim",
        "smooth",
        "padnet_l2",
        "padnet_dc",
        "padnet_reg",
        "bloch",
        "anat",
        "style",
        "content",
        "recon",
        "bloch_residual",
        "physics_constraint",
        "parallel_imaging_kspace",
        "snr_preserving",
        "biophysical_flow",
        "physics_informed",
        # Composite-bundled GAN sub-losses (built by ``_build_composite_gan``;
        # live inside ``losses.gan`` as ``enable_gradient_penalty`` /
        # ``lambda_gp`` / ``lambda_feat_match`` rather than declarative entries).
        "gradient_penalty",
        "feature_matching",
        # Strategy-inline, read straight off ``losses.*`` with NO ``enable_*``
        # gate and no list entry possible. Each is exempt because its read site
        # was verified, not because it looked unbuilt (#9: a declared exemption,
        # never a silent fallback).
        "pre_dc_kspace",  # strategies/diffusion.py:2593 — `lam > 0.0` then inline
        "pre_dc_acquired",  # same site, the acquired-bin half — `lam > 0.0` then inline
        "cycle_adv",  # strategies/cycle_bloch_strategy.py:212 — losses.gan.lambda_cycle_adv
        "cycle_bloch",  # strategies/cycle_bloch_strategy.py:211 — losses.gan.lambda_cycle_bloch
        "nce",  # strategies/cut_strategy.py:336/384 — losses.gan.lambda_nce, adv_g + lambda_nce * nce
    }
)

#: Declaration *sources* whose weight a loss **computer** resolves from the weight
#: table directly, rather than naming a module ``LossBuilder`` must instantiate.
#: Each entry cites the read site that consumes the weight and computes the term
#: inline, so refusing the declaration would refuse a term that does train.
#:
#: Keyed on the exact ``LossWeightSpec.source`` string and NOT on the loss name.
#: Several sections canonicalise onto one name -- ``losses.diffusion.lambda_mse``,
#: ``losses.reconstruction.lambda_l2`` and a future third field would all become
#: ``l2`` -- and only the fields listed here have had their read site read.
#: Exempting the name would carry the next such field in unexamined.
#:
#: **The exemptions are strategy-blind.** ``udr.py:751/757`` run under
#: ``UnifiedReconstructionLossComputer`` and ``unified_vae.py:281`` under the VAE
#: computer; under a different strategy the same lambdas are inert and this guard
#: now passes them. That bound is shared by every skip set in this module
#: (``STRATEGY_MANAGED_LOSSES`` included) and is not narrowed here.
COMPUTER_RESOLVED_LAMBDA_SOURCES: dict[str, str] = {
    # 4-source resolution chain, ending at the weight table.
    "losses.diffusion.lambda_mse": (
        "models/losses/computers/unified_diffusion_reconstruction.py:381-413"
    ),
    # Weight-gated only, term computed inline by ``_complex_safe_l1`` /
    # ``_complex_safe_mse``. ``self.l1_loss_fn`` / ``self.l2_loss_fn`` are assigned
    # ``None`` at udr.py:716-717 and never read, and
    # ``UnifiedReconstructionLossComputer._initialize_losses`` is ``pass`` --
    # no computer asks the builder for an l1 or l2 module.
    "losses.reconstruction.lambda_l1": (
        "models/losses/computers/unified_diffusion_reconstruction.py:751-754"
    ),
    "losses.reconstruction.lambda_l2": (
        "models/losses/computers/unified_diffusion_reconstruction.py:757-760"
    ),
    # ``lambda_l1`` exists on LatentLossesConfig too and canonicalises to the same
    # ``l1`` the read site above consumes.
    "losses.latent.lambda_l1": (
        "models/losses/computers/unified_diffusion_reconstruction.py:751-754"
    ),
    # Weight-gated only; the closed-form KL is computed in the branch itself.
    "losses.latent.lambda_kl": ("models/losses/computers/unified_vae.py:281-301"),
    # CycleGAN computes the cycle and identity L1 terms inline and weights them with
    # these two knobs (strategies/cyclegan_strategy.py:154).
    "losses.gan.lambda_cycle": "strategies/cyclegan_strategy.py:154 (read at :378)",
    "losses.gan.lambda_identity": "strategies/cyclegan_strategy.py:154 (read at :378)",
    # The GRL domain head is built inline with this weight (cyclegan_strategy.py:265).
    "losses.gan.lambda_domain": "strategies/cyclegan_strategy.py:265",
    # StarGAN v2 computes the diversity-sensitive term inline (stargan_v2_strategy.py:382).
    "losses.gan.lambda_diversity": "strategies/stargan_v2_strategy.py:321 (read at :382)",
    # The VQ-VAE computer weights the quantiser term the strategy hands it (unified_vae.py:482).
    "losses.latent.lambda_vq": "models/losses/computers/unified_vae.py:482",
}

#: The subset of the above whose computer reads the lambda as a *different knob*
#: from the table key it canonicalises to. ``losses.diffusion.lambda_mse`` is step
#: 3 of ``UnifiedDiffusionReconstructionLossComputer._resolve_diffusion_weight``
#: and means the diffusion-term weight, while the table aliases ``lambda_mse`` to
#: ``l2`` and joins it to any image or k-space ``mse`` entry. One table key, two
#: knobs, so such a lambda beside a list entry of the same name is not a duplicate.
#:
#: A separate constant because the two questions differ. The set above answers
#: "would refusing this lambda-only declaration stop a run that trains?" and the
#: set below answers "is this lambda redundant beside a list entry?". The four
#: reconstruction and latent sources are in the first and not the second:
#: ``_get_loss_weight("l1")`` reads the table value a list entry already supplies,
#: so the lambda beside it adds nothing and ``dual_surface_loss_declarations``
#: should report it (non-negotiable 17 — one owner per invariant, not one
#: constant for two).
REINTERPRETED_LAMBDA_SOURCES: dict[str, str] = {
    "losses.diffusion.lambda_mse": COMPUTER_RESOLVED_LAMBDA_SOURCES["losses.diffusion.lambda_mse"],
}

#: Matches ``losses.<section>.lambda_<field>`` or ``losses.<list>[<raw>].weight``.
_SOURCE_NAME_RE = re.compile(r"lambda_(?P<field>\w+)$|\[(?P<raw>[^\]]+)\]\.weight$")


def declared_names_in(source: str) -> set[str]:
    """The RAW names an author wrote, recovered from a ``LossWeightSpec.source``.

    ``LossWeightSpec.name`` is canonical, while the skip sets in this module are
    keyed on the spelling the schema uses. Three of the 29
    ``STRATEGY_MANAGED_LOSSES`` entries canonicalise to something else
    (``marker`` -> ``marker_corruption``, ``content`` -> ``perceptual``,
    ``patch_nce`` -> ``cut_patch_nce``), so a canonical-only membership test
    silently loses those three exemptions — measured as a false raise on
    ``pillars/exp_pillar_07_vf_fourier_shift.yaml``, whose ``lambda_marker`` the
    VF strategy applies inline.

    Canonicalising the skip set instead would be worse: it would also exempt
    ``perceptual``, a real buildable loss that #421 established is a DIFFERENT
    term from ``content``. Recovering the raw name keeps both spellings exact.
    """
    names: set[str] = set()
    for segment in source.split("+"):
        m = _SOURCE_NAME_RE.search(segment)
        if m:
            names.add(m.group("field") or m.group("raw"))
    return names


def strategy_resolved(spec: LossWeightSpec) -> bool:
    """True when every declaration behind ``spec`` is computed inline, not built here."""
    for segment in spec.source.split("+"):
        if declared_names_in(segment) & STRATEGY_MANAGED_LOSSES:
            continue
        if segment in COMPUTER_RESOLVED_LAMBDA_SOURCES:
            continue
        return False
    return True


def declared_build_set(loss_config: Any) -> dict[str, float]:
    """``{canonical loss -> weight}`` for every term the builder must instantiate.

    The one owner of the build set (#1966). A term is built when the author declared
    it active (:func:`is_active`), it is not computed inline by a strategy or computer
    (:func:`strategy_resolved`), and ``policy.exclude_defaults`` does not name it. An
    unregistered name that reaches this set is refused by the builder, not dropped here.
    """
    if loss_config is None:
        return {}
    policy = getattr(loss_config, "policy", None)
    excluded = set(getattr(policy, "exclude_defaults", None) or ())
    table = build_loss_weight_table(loss_config)
    build: dict[str, float] = {}
    for spec in table.values():
        if not is_active(spec) or strategy_resolved(spec):
            continue
        if spec.name in excluded or declared_names_in(spec.source) & excluded:
            continue
        build[canonical_loss_name(spec.name)] = spec.weight
    return build
