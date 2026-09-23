"""Loss Builder

Creates loss functions based on objectives configuration.
Supports reconstruction, adversarial, physics, and regularization losses.
SSOT for loss instantiation.
"""

import logging
import re
from typing import Any

import torch
import torch.nn as nn

from spectramr.config.settings import TrainingSettings
from spectramr.core.component_signature import signature_contract
from spectramr.domain.exceptions import ConfigurationError
from spectramr.models.losses.registry import LossRegistry, create_loss, get_loss_capabilities

from .base import Builder

logger = logging.getLogger(__name__)

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


class LossBuilder(Builder):
    """Builds loss functions dynamically from objectives config.

    Creates all loss functions needed for training by automatically parsing
    the enabled losses from the LossConfigSchema via `get_enabled_losses()`.
    Config-specific parameters are injected automatically based on loss type.
    """

    def __init__(self, config: TrainingSettings, device: Any):
        """__init__.

        Args:
            config (TrainingSettings): Description.
            device (Any): Description.
        """
        self._config = config
        self._device = device
        self._losses: dict[str, nn.Module] = {}
        self._already_built_dynamic = False
        self._weight_table: Any = None

    def _loss_weight_table(self):
        """The arm's DECLARED loss weights — built once, shared by every reader.

        One owner (NN17): ``build_loss_weight_table`` keys on written-ness
        (``model_fields_set``) plus the domain lists, so it answers *what did the
        author declare*, which is a different question from *what does the schema
        default to*. A builder reading ``recon_config.lambda_<name>`` raw cannot
        tell the two apart — see :meth:`_build_composite_gan`.
        """
        if self._weight_table is None:
            from spectramr.models.losses.weights import build_loss_weight_table

            self._weight_table = build_loss_weight_table(self._config.losses)
        return self._weight_table

    def _declared_weight(self, name: str) -> float:
        """The weight the author DECLARED for ``name``, or ``0.0`` if they did not.

        The construction-time counterpart to
        :func:`~spectramr.models.losses.weights.resolve_loss_weight`, and
        deliberately NOT that function: ``resolve_loss_weight`` applies the
        warm-up gate against an ``iteration``, which only has an answer inside
        the training loop. A composite built once, before step 0, that asked it
        would receive the warm-up ``0.0`` and freeze it in for the whole run.

        ``enabled: false`` already forces ``weight`` to ``0.0`` in the table
        (``weights.py`` — ``weight=0.0 if not enabled else weight``); the
        ``enabled`` test here is kept for the case the table admits and that
        assignment does not cover: an entry that is *present* but disabled by a
        later list entry.
        """
        spec = self._loss_weight_table().get(name)
        if spec is None or not spec.enabled:
            return 0.0
        return spec.weight

    def get_enabled_losses(self) -> dict[str, float]:
        """get_enabled_losses.

        Returns:
            dict[str, float]: Description.
        """
        if hasattr(self._config.losses, "get_enabled_losses"):
            return self._config.losses.get_enabled_losses()
        return {}

    def get_loss_weights(self) -> dict[str, float]:
        """get_loss_weights.

        Returns:
            dict[str, float]: Description.
        """
        return self.get_enabled_losses()

    def validate_loss_configuration(self) -> bool:
        """validate_loss_configuration.

        Returns:
            bool: Description.
        """
        return True

    def _build_all_dynamic(self):
        """Build all enabled losses dynamically exactly once."""
        if self._already_built_dynamic:
            return
        self._already_built_dynamic = True

        enabled_losses = self.get_enabled_losses()
        if not enabled_losses:
            logger.debug("No enabled losses found in configuration.")
            return

        recon_config = self._config.losses.reconstruction
        gan_config = self._config.losses.gan
        # ``physics``/``evidential`` are read by _schema_loss_kwargs (the SSOT this
        # path now consumes), so they are no longer locals here.

        # ==================== LIST-BASED DOMAIN-AWARE PATH ====================
        # When kspace_losses / image_losses / complex_losses are populated,
        # auto-bridge based on the declared output_domain.
        if self._config.losses.uses_list_based_losses:
            self._build_list_based_losses()
            # After building list-based losses, continue to paradigm-specific
            # losses (GAN, diffusion, latent, etc.) that use the old path.
            # Only build paradigm losses that aren't already in the list-based set.
            # Derived from LOSS_LIST_DOMAINS, not a hand-written tuple. A list
            # missing here is not seen as already-built, so the paradigm
            # fallback below flags its entries as "unmigrated" and refuses the
            # run -- for losses that were in fact built moments earlier.
            from spectramr.config.schemas.loss import LOSS_LIST_DOMAINS
            from spectramr.models.losses.weights import canonical_loss_name

            # Raw names AND their canonical twins. The declared entry is what the
            # author wrote (``mse``), the guard below compares against canonical
            # names (``l2``); keeping both spellings in one set means the
            # membership test cannot reject a list entry for being spelled with a
            # registered alias. Raw names are retained so the ``"adversarial" not
            # in list_loss_names`` bypass just below keeps its exact meaning.
            _list_names_raw = {
                c.name
                for list_name in LOSS_LIST_DOMAINS
                for c in getattr(self._config.losses, list_name)
            }
            list_loss_names = _list_names_raw | {canonical_loss_name(n) for n in _list_names_raw}
            # Build GAN composite if enabled and not in list-based
            if "adversarial" not in list_loss_names and gan_config:
                if enabled_losses.get("adversarial", 0) > 0:
                    self._build_composite_gan(gan_config, recon_config)

            # Deep Supervision special case
            ds_weight = self._config.losses.lambda_deep_supervision
            if ds_weight > 0:
                try:
                    self._losses["deep_supervision"] = create_loss(
                        "deep_supervision", weight=ds_weight
                    ).to(self._device, non_blocking=True)
                    logger.info("Created Deep Supervision loss")
                except Exception as e:
                    raise ConfigurationError(f"Failed to create Deep Supervision loss: {e}") from e

            # Catch any remaining paradigm-specific losses (diffusion, latent, SSL, etc.)
            # that are enabled but were not processed by the declarative lists or GAN bypass.
            #
            # Strategy-managed losses are computed inline by training strategies
            # (e.g., R1 needs discriminator gradients, PatchNCE needs encoder
            # features) or bundled into a composite GAN loss — see the
            # module-level SSOT constant.
            strategy_managed = STRATEGY_MANAGED_LOSSES

            # Registry name translation (same map as legacy flag-based path)
            _fallback_registry_map = {
                "l2": "mse",
                "r1": "r1_regularization",
                "hist": "histogram_consistency",
                "ffl": "focal_frequency",
                "edge": "sobel_edge",
                "dc": "data_consistency",
                "frequency_domain": "frequency_domain_consistency",
                "ms_ssim": "ms_ssim",
                "pde": "helmholtz_pde",
                "pinn_dc": "data_consistency",
                "bloch": "bloch_residual",
            }

            # Losses computed by the reconstruction loss computer resolve their
            # weight from ``reconstruction.lambda_*`` (resolve_static_loss_weight)
            # and are intentionally kept OUT of the declarative lists — they are
            # NOT silently skipped, so the guard must not flag them (the
            # direct_ulf_to_hf_sr / pma_02 pattern). Same idea as strategy_managed.
            recon_managed = self._config.losses.reconstruction_managed_losses()

            unmigrated: list[tuple[str, float]] = []
            flagged: set[str] = set()

            def _is_known(loss_name: str) -> bool:
                """Membership test shared by both passes below."""
                effective = _fallback_registry_map.get(loss_name, loss_name)
                return (
                    loss_name in self._losses
                    or effective in self._losses
                    or loss_name in list_loss_names
                    or effective in list_loss_names
                    or loss_name in recon_managed
                    or effective in recon_managed
                )

            for loss_name, weight in enabled_losses.items():
                if loss_name in strategy_managed:
                    logger.debug(
                        "Skipping '%s' (weight=%s) — strategy-managed loss",
                        loss_name,
                        weight,
                    )
                    continue
                # Translate legacy flag names (``l2``/``edge``/``ffl``/``dc`` …)
                # to their registry twins before the membership test — otherwise
                # a migrated config that uses the legacy alias while its
                # registry-name twin sits in the declarative lists is falsely
                # rejected as "unmigrated". (The map was previously dead.)
                if not _is_known(loss_name) and weight > 0:
                    unmigrated.append((loss_name, weight))
                    flagged.add(canonical_loss_name(loss_name))

            # ---- second pass: the weight table -----------------------------
            # ``get_enabled_losses()`` keeps a ``lambda_<name>`` only while its
            # sibling ``enable_<name>`` is true, and every ``enable_*`` defaults
            # False -- so a lambda-only declaration never reached the loop above
            # and this guard could not fire on the very shape its message
            # describes. ``build_loss_weight_table`` keys on written-ness
            # (``model_fields_set``), which is the declaration the author made.
            #
            # A union, not a replacement: the pass above still owns the
            # ``enable_x: true`` + defaulted-lambda shape, which the table
            # reports at its schema default rather than as an author decision.
            for spec in self._loss_weight_table().values():
                if not spec.enabled or spec.weight <= 0:
                    continue
                raw_names = declared_names_in(spec.source)
                if spec.name in flagged:
                    continue
                if spec.name in strategy_managed or raw_names & strategy_managed:
                    continue
                # ``source`` is "+"-joined when a name is declared on more than
                # one surface. Exempt only when EVERY declaration is
                # computer-resolved -- one non-exempt surface means a module was
                # expected and none was built.
                if all(src in COMPUTER_RESOLVED_LAMBDA_SOURCES for src in spec.source.split("+")):
                    continue
                if not _is_known(spec.name) and not any(_is_known(raw) for raw in raw_names):
                    unmigrated.append((spec.name, spec.weight))
                    flagged.add(spec.name)

            if unmigrated:
                # Per CLAUDE.md #9 (silent fallbacks are forbidden) and #10
                # (warnings are not OK): a non-zero loss weight for a key
                # that isn't in the declarative kspace/image/complex lists
                # is an unambiguous config bug — it would silently train
                # without that loss. Refuse instead of warn-and-skip.
                lines = [
                    f"  • '{n}' (weight={w}) → migrate into losses.image_losses, "
                    f"losses.kspace_losses, losses.complex_losses or "
                    f"losses.latent_losses"
                    for n, w in unmigrated
                ]
                raise ConfigurationError(
                    "Loss configuration references key(s) that are not in the v6.0 "
                    "declarative list-based losses (kspace_losses / image_losses / "
                    "complex_losses / latent_losses). Silently skipping them would "
                    "train without the "
                    "advertised supervision — refusing per CLAUDE.md #9. Unmigrated "
                    "key(s):\n" + "\n".join(lines)
                )

            return

        # ==================== LEGACY FLAG-BASED PATH ====================
        use_universal_bridge = (
            recon_config.spatial_losses_use_fourier_bridge if recon_config else False
        )

        class FourierBridgeLossWrapper(nn.Module):
            """Wraps any spatial loss with a Fourier Bridge for k-space compatibility."""

            def __init__(self, inner_loss: nn.Module):
                """__init__.

                Args:
                    inner_loss (nn.Module): Description.
                """
                super().__init__()
                from spectramr.models.losses.physics_losses import (
                    DifferentiableFourierBridge,
                )

                self.bridge = DifferentiableFourierBridge(
                    spatial_loss_fn=inner_loss, return_complex=False
                )

            def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
                """forward.

                        Args:
                            pred (torch.Tensor): Description.
                            target (torch.Tensor): Description.
                        Returns:
                            torch.Tensor: Description.

                forward method for FourierBridgeLossWrapper.

                Executes PyTorch tensor operations.

                Args:
                    pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
                    target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

                Returns:
                    torch.Tensor: Output tensor.

                Hardware/Device Context:
                    Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
                """
                return self.bridge(pred, target, **kwargs)

        def wrap_spatial_loss(loss_module: nn.Module) -> nn.Module:
            """wrap_spatial_loss.

            Args:
                loss_module (nn.Module): Description.
            Returns:
                nn.Module: Description.
            """
            if use_universal_bridge:
                return FourierBridgeLossWrapper(loss_module)
            return loss_module

        spatial_losses = {
            "l1",
            "smooth_l1",
            "mse",
            "perceptual",
            "ssim",
            "ms_ssim",
            "dists",
            "lpips",
            "sobel_edge",
            "explicit_gradient",
            "graph_consistency",
            "spectral_graph",
        }

        registry_map = {
            "l2": "mse",
            "hist": "histogram_consistency",
            "ffl": "focal_frequency",
            "edge": "sobel_edge",
            "explicit_gradient": "explicit_gradient",
            "graph_consistency": "graph_consistency",
            "spectral_graph": "spectral_graph",
            "dc": "data_consistency",
            "r1": "r1_regularization",
            "frequency_domain": "frequency_domain_consistency",
            "ms_ssim": "ms_ssim",
            "pde": "helmholtz_pde",
            "pinn_dc": "data_consistency",
        }

        if recon_config and hasattr(recon_config, "loss_type"):
            registry_map["reconstruction"] = recon_config.loss_type
            registry_map["recon"] = recon_config.loss_type
        else:
            registry_map["reconstruction"] = "l1"
            registry_map["recon"] = "l1"

        # Kwargs Map Extraction (SSOT shared with the list-based path;
        # see _schema_loss_kwargs -- issue #467 follow-up).
        kwargs_map = self._schema_loss_kwargs()

        # Main Instantiation Loop
        for cfg_name, weight in enabled_losses.items():
            if weight <= 0 and cfg_name not in ("adversarial",):
                continue

            # Special cases for GAN/Adversarial
            if cfg_name == "adversarial" and gan_config:
                self._build_composite_gan(gan_config, recon_config)
                continue

            # Strategy-managed losses: commitment and codebook are computed
            # internally by VQ-VAE/VQ-GAN strategies, not standalone modules.
            # Skip them here; the strategy injects them at training time.
            strategy_managed = STRATEGY_MANAGED_LOSSES
            if cfg_name in strategy_managed:
                logger.debug(
                    "Skipping '%s' loss (weight=%s) — managed by training strategy",
                    cfg_name,
                    weight,
                )
                continue

            # Truly unimplemented losses — fail fast
            if cfg_name == "tissue_bounds":
                raise ConfigurationError(
                    f"Loss '{cfg_name}' is not implemented. "
                    "Remove it from the loss config or provide a registered implementation."
                )

            # Resolve mapped names and load parameters
            reg_name = registry_map.get(cfg_name, cfg_name)
            kwargs = kwargs_map.get(reg_name, {})

            try:
                loss_fn = create_loss(reg_name, **kwargs).to(self._device, non_blocking=True)
                if use_universal_bridge and reg_name in spatial_losses:
                    loss_fn = wrap_spatial_loss(loss_fn)
                self._losses[cfg_name] = loss_fn
                logger.info(f"Created {cfg_name} loss (weight={weight})")
            except Exception as e:
                raise ConfigurationError(f"Failed to create {cfg_name} loss: {e}") from e

        # Deep Supervision special case
        ds_weight = self._config.losses.lambda_deep_supervision
        if ds_weight > 0:
            try:
                self._losses["deep_supervision"] = create_loss(
                    "deep_supervision", weight=ds_weight
                ).to(self._device, non_blocking=True)
                logger.info("Created Deep Supervision loss")
            except Exception as e:
                raise ConfigurationError(f"Failed to create Deep Supervision loss: {e}") from e

    def _schema_loss_kwargs(self) -> dict[str, Any]:
        """Per-loss constructor kwargs declared on the loss SCHEMA (not per-entry).

        SSOT for both build paths. ``_build_all_dynamic`` always honoured these; the
        list-based path did not, so a knob like
        ``losses.reconstruction.log_spectral_skip_fft: true`` was read by the schema,
        logged as configured, and then silently dropped -- ``log_spectral`` ran with
        ``skip_fft=False`` and applied a forward ``fft2c`` to an already-k-space
        tensor, turning a log-SPECTRAL penalty into an image-domain log-magnitude
        one. 17 arms carried exactly that (issue #467 follow-up).

        A per-entry ``kwargs:`` on the list item still wins -- this is the default,
        not an override.
        """
        recon_config = self._config.losses.reconstruction
        physics_config = self._config.losses.physics
        gan_config = self._config.losses.gan
        # ``evidential`` is an optional sub-schema (default None on the v6.0
        # LossConfigSchema). Use getattr so unit-test mocks that don't spec
        # it (and any older configs that predate the field) don't raise.
        evidential_config = getattr(self._config.losses, "evidential", None)

        kwargs_map: dict[str, Any] = {}
        if recon_config:
            kwargs_map["ms_ssim"] = {"data_range": 1.0}
            kwargs_map["ssim"] = {"data_range": 1.0}
            kwargs_map["perceptual"] = {}
            kwargs_map["mind_ssc"] = {}
            kwargs_map["explicit_gradient"] = {}
            kwargs_map["log_spectral"] = {"skip_fft": recon_config.log_spectral_skip_fft}
            kwargs_map["focal_frequency"] = {"alpha": recon_config.ffl_alpha}
            kwargs_map["histogram_consistency"] = {"bins": recon_config.histogram_bins}
            kwargs_map["background_suppression"] = {
                "threshold_ratio": recon_config.background_suppression_threshold_ratio,
                "use_fourier_bridge": recon_config.background_suppression_use_fourier_bridge,
            }
            kwargs_map["rician_consistency"] = {
                "sigma": recon_config.rician_noise_sigma,
                "use_fourier_bridge": recon_config.rician_use_fourier_bridge,
            }
            kwargs_map["frequency_weighted_l1_kspace"] = {
                "alpha": recon_config.frequency_weighted_l1_kspace_alpha
            }
            kwargs_map["weighted_kspace_l1"] = {"exponent": recon_config.weighted_kspace_exponent}
            kwargs_map["sobolev_kspace"] = {}  # Parameterless: weight injected at compute time
            kwargs_map["sense_adjoint_l1"] = {}  # smaps injected dynamically via _call_safe_loss
            kwargs_map["spectral_graph"] = {
                "k": recon_config.spectral_graph_k,
                "patch_size": recon_config.spectral_graph_patch_size,
            }
            try:
                patch_size = self._config.data.sampling.patch_size
                kwargs_map["frequency_weighted_l1_kspace"]["height"] = patch_size[0]
                kwargs_map["frequency_weighted_l1_kspace"]["width"] = patch_size[1]
            except Exception as _exc:
                logger.debug("Suppressed exception: %s", _exc)

        if physics_config:
            kwargs_map["data_consistency"] = {"lambda_dc": physics_config.lambda_physics_constraint}
            kwargs_map["complex_spatial_gradient"] = {
                "use_fourier_bridge": physics_config.complex_spatial_gradient_use_fourier_bridge
            }

        if gan_config:
            kwargs_map["r1_regularization"] = {"weight": gan_config.lambda_r1}

        if evidential_config:
            kwargs_map["evidential"] = {"coeff": evidential_config.lambda_evidential}

        return kwargs_map

    def _build_list_based_losses(self):
        """Build losses from domain-aware kspace/image/complex loss lists.

        Automatically wraps losses with DifferentiableFourierBridge when the
        loss's expected domain differs from the model's output_domain.

        Bridge matrix (output_domain → loss list → action), matching the code below:
          output_domain=kspace:
            kspace_losses  → no bridge (native k-space)
            image_losses   → iFFT bridge (magnitude)
            complex_losses → iFFT bridge (return_complex=True)
          output_domain=image:
            kspace_losses  → no bridge (self-FFT losses only; see the note below)
            image_losses   → no bridge (native)
            complex_losses → no bridge (cast to complex if needed)
          output_domain=complex_image:
            kspace_losses  → no bridge
            image_losses   → no bridge (loss extracts magnitude itself)
            complex_losses → no bridge (native)

        A loss that bridges *internally* (``use_fourier_bridge=True``, e.g.
        ``complex_spatial_gradient`` / ``sense_adjoint_l1`` / ``rician_consistency``)
        must be declared under the list whose bridge mode is ``none``, otherwise the
        tensor is inverse-transformed TWICE. That is silent — both transforms are
        individually valid and the composite is finite — so it is rejected here
        rather than left to produce a meaningless objective (issue #467).
        """
        from spectramr.models.losses.physics_losses import DifferentiableFourierBridge

        output_domain = self._config.losses.policy.output_domain

        class _BridgedLoss(nn.Module):
            """Wraps a spatial loss with DifferentiableFourierBridge for auto domain conversion."""

            def __init__(self, inner_loss: nn.Module, return_complex: bool = False):
                super().__init__()
                self.bridge = DifferentiableFourierBridge(
                    spatial_loss_fn=inner_loss, return_complex=return_complex
                )

            def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
                return self.bridge(pred, target, **kwargs)

        def _create_and_register(name: str, weight: float, kwargs: dict, bridge_mode: str):
            """Create a loss, optionally wrap it, and register.

            Args:
                name: Registry name for the loss.
                weight: Loss weight (for logging only; actual weighting done by the computer).
                kwargs: Extra kwargs for the loss constructor.
                bridge_mode: One of 'none', 'ifft_magnitude', 'ifft_complex'.
            """
            try:
                loss_fn = create_loss(name, **kwargs).to(self._device, non_blocking=True)

                # Reject the double bridge (issue #467). A loss that carries its
                # own DifferentiableFourierBridge would be inverse-transformed a
                # second time by the wrapper below: the outer bridge emits an
                # image (magnitude or complex), the inner one re-reads it as
                # k-space, halves the coil count by re-pairing channels as
                # (real, imag), and iFFTs again. Nothing raises and the value is
                # finite, so the run stays green while the term measures nothing
                # it advertises. Fail loud instead (pitfalls #9 / #16).
                if bridge_mode != "none" and getattr(loss_fn, "use_fourier_bridge", False):
                    raise ConfigurationError(
                        f"Loss '{name}' bridges from k-space internally "
                        f"(use_fourier_bridge=True) but is declared under a loss "
                        f"list that adds a '{bridge_mode}' bridge for "
                        f"output_domain='{output_domain}' — the tensor would be "
                        f"inverse-transformed twice. Declare '{name}' under "
                        f"losses.kspace_losses (bridge 'none', so its own bridge "
                        f"does the single iFFT), or keep the current list and set "
                        f"kwargs: {{use_fourier_bridge: false}} on the entry so "
                        f"the outer bridge is the only one. Note it is `kwargs:`, "
                        f"not `config:` — LossComponentConfig is extra='ignore', "
                        f"so a `config:` block is silently dropped (issue #468)."
                    )

                # Reject the ZERO bridge, the complement of the guard above. That
                # one catches a tensor transformed twice; this one catches a loss
                # that expects an image, sits in a list the builder does not
                # bridge, and does not bridge itself — so nothing transforms
                # anything and it reads raw k-space as if it were anatomy. The
                # symptom is identical to the double bridge: finite values, no
                # raise, a green run measuring nothing it advertises.
                #
                # Only under ``output_domain: kspace``, because that is the one
                # combination where ``bridge_mode == "none"`` leaves a k-space
                # tensor; with an image or complex-image output the unbridged
                # tensor is already what the loss wants.
                if bridge_mode == "none" and output_domain == "kspace":
                    caps = get_loss_capabilities(name)
                    registered_domain = getattr(
                        getattr(caps, "domain", None), "value", getattr(caps, "domain", None)
                    )
                    declared_image = kwargs.get("input_domain") == "image"
                    if (
                        declared_image or registered_domain == "image"
                    ) and not getattr(loss_fn, "use_fourier_bridge", False):
                        why = (
                            "kwargs declare input_domain: image"
                            if declared_image
                            else f"it is registered domain='{registered_domain}'"
                        )
                        raise ConfigurationError(
                            f"Loss '{name}' expects an IMAGE ({why}) but is declared "
                            f"under a loss list that adds no bridge for "
                            f"output_domain='kspace', and it does not bridge itself "
                            f"(use_fourier_bridge is False) — it would receive raw "
                            f"k-space and score it as an image. Declare '{name}' under "
                            f"losses.image_losses (bridge 'ifft_magnitude') or "
                            f"losses.complex_losses (bridge 'ifft_complex'), or set "
                            f"kwargs: {{input_domain: kspace}} if it really is meant to "
                            f"read k-space and bridge itself."
                        )

                if bridge_mode == "ifft_magnitude":
                    loss_fn = _BridgedLoss(loss_fn, return_complex=False)
                    logger.info(f"Created {name} loss (weight={weight}) with iFFT→magnitude bridge")
                elif bridge_mode == "ifft_complex":
                    loss_fn = _BridgedLoss(loss_fn, return_complex=True)
                    logger.info(f"Created {name} loss (weight={weight}) with iFFT→complex bridge")
                else:
                    logger.info(f"Created {name} loss (weight={weight}) [native domain]")

                self._losses[name] = loss_fn
            except ConfigurationError:
                # Already a precise, actionable message (e.g. the double-bridge
                # rejection above) — do not bury it under "Failed to create".
                raise
            except Exception as e:
                raise ConfigurationError(f"Failed to create {name} loss: {e}") from e

        # Determine bridge modes based on output_domain
        if output_domain == "kspace":
            kspace_bridge = "none"
            image_bridge = "ifft_magnitude"
            complex_bridge = "ifft_complex"
        elif output_domain == "image":
            # Image-output model: kspace losses receive image-domain inputs
            # as-is. Some kspace losses self-FFT (e.g. focal_frequency,
            # log_spectral), others (complex_l1) do NOT — those will
            # silently produce wrong values. The right declarative fix is
            # to declare an explicit `adapters.pre_loss_pred:
            # [fft_image_to_kspace]` chain when the model truly outputs
            # image and kspace losses are wanted.
            kspace_bridge = "none"
            image_bridge = "none"
            complex_bridge = "none"
        elif output_domain == "complex_image":
            kspace_bridge = "none"
            image_bridge = "none"  # Loss must handle magnitude extraction itself
            complex_bridge = "none"
        elif output_domain == "latent":
            # The model emits a latent. No Fourier bridge is meaningful here --
            # a latent has no k-space -- so every list is native and the schema
            # forbids the combinations that would need one.
            kspace_bridge = "none"
            image_bridge = "none"
            complex_bridge = "none"
        else:
            # The message used to omit 'latent', which the branch directly
            # above handles -- so a correct latent arm that reached here by some
            # other route was told to use a domain it was already not using.
            raise ValueError(
                f"Invalid output_domain='{output_domain}'. "
                "Must be 'kspace', 'image', 'complex_image' or 'latent'. "
                "(The schema now refuses the other SignalDomain members at load, "
                "so reaching this branch means the config bypassed validation.)"
            )

        # Schema-declared per-loss kwargs (SSOT with the legacy path). A knob like
        # ``losses.reconstruction.log_spectral_skip_fft`` used to be honoured ONLY by
        # ``_build_all_dynamic``, so a list-based arm read it, logged it, and then built
        # the loss with its constructor default -- ``log_spectral`` applied a forward
        # ``fft2c`` to an already-k-space tensor, i.e. a log-magnitude penalty on the
        # IMAGE where the YAML asked for one on the spectrum. A per-entry ``kwargs:``
        # still wins; these are defaults, not overrides.
        schema_kwargs = self._schema_loss_kwargs()

        def _merged(component) -> dict:
            merged = {
                **schema_kwargs.get(component.name, {}),
                **(component.kwargs or {}),
            }
            if not merged:
                return merged

            # A kwarg that never reaches the constructor changes the OBJECTIVE
            # silently. `sobolev_order: 1` was declared by 56 arms, swallowed by
            # `extra="ignore"`, and only found by a manual audit weeks later
            # (#560, #615) -- `SobolevKSpaceLoss` had no `order` parameter at
            # all. Raising here rather than letting `create` fail with a bare
            # TypeError is what makes the message actionable.
            #
            # Posture is RAISE, and that is a measured choice: across the
            # loadable corpus all 33 declared `kwargs:` keys and all 19
            # schema-derived `kwargs_map` entries already reach their ctor, so
            # this rejects nothing that exists today. Same rung as
            # `scheduler_resolution`, which raises on an unroutable knob.
            loss_cls = LossRegistry.get_loss_class(component.name)
            if loss_cls is None:
                return merged  # an unknown loss name is `create`'s error to report
            contract = signature_contract(loss_cls)
            # A **kwargs ctor accepts everything (it may read keys via
            # `kwargs.get`), and a class with no owned `__init__` declares no
            # contract. Neither can be checked here.
            if contract.accepts_var_kwargs or not contract.owner:
                return merged
            unroutable = sorted(k for k in merged if k not in contract.accepted)
            if unroutable:
                raise ConfigurationError(
                    f"Loss {component.name!r} cannot consume {unroutable}. "
                    f"{contract.owner}.__init__ accepts "
                    f"{sorted(contract.accepted)}. A kwarg that never reaches "
                    f"the constructor changes the objective silently."
                )
            return merged

        # Build each list
        for component in self._config.losses.kspace_losses:
            if component.enabled and component.weight > 0:
                _create_and_register(
                    component.name, component.weight, _merged(component), kspace_bridge
                )

        for component in self._config.losses.image_losses:
            if component.enabled and component.weight > 0:
                _create_and_register(
                    component.name, component.weight, _merged(component), image_bridge
                )

        for component in self._config.losses.complex_losses:
            if component.enabled and component.weight > 0:
                _create_and_register(
                    component.name, component.weight, _merged(component), complex_bridge
                )

        # Latent losses are native by construction: the schema only admits them
        # with output_domain='latent', so there is never a bridge to insert.
        for component in self._config.losses.latent_losses:
            if component.enabled and component.weight > 0:
                _create_and_register(component.name, component.weight, _merged(component), "none")

    def _build_composite_gan(self, gan_config, recon_config):
        """Helper to build CompositeGANLoss."""
        try:
            gan_loss_type = gan_config.gan_loss_type
            adv_registry_map = {
                "vanilla": "gan_standard",
                "standard": "gan_standard",
                "bce": "gan_standard",
                "lsgan": "gan_lsgan",
                "ralsgan": "gan_ralsgan",
                "wgan": "gan_wgan",
                "wgan-gp": "gan_wgan",
                "hinge": "gan_hinge",
            }
            adv_name = adv_registry_map.get(gan_loss_type.lower())
            if adv_name is None:
                # NN#3: an unknown registered-option value must raise, never
                # silently degrade to the standard GAN loss.
                raise ConfigurationError(
                    f"Unknown gan_loss_type: {gan_loss_type!r}. "
                    f"Valid values: {sorted(adv_registry_map)}"
                )
            adv_strategy = create_loss(adv_name, label_smoothing=gan_config.label_smoothing)

            # Perceptual weight comes from the DECLARED table, never from the
            # schema default (#1923).
            #
            # ``recon_config.lambda_perceptual`` defaults to 10.0 while its
            # sibling ``enable_perceptual`` defaults False, so reading the field
            # raw builds — and *trains*, via ``CompositeGANLoss``
            # ``.compute_generator_loss`` — a VGG perceptual term at weight 10.0
            # on every GAN arm that never mentioned perceptual. The table keys on
            # written-ness, so an undeclared name is simply absent from it.
            #
            # ``resolve_loss_weight`` is deliberately NOT used here: it applies
            # the warm-up gate against an ``iteration``, and this composite is
            # constructed ONCE with a scalar. Resolving at iteration 0 would bake
            # a warm-up 0.0 in permanently and silently disable perceptual for
            # the whole run. The static declared weight is the correct value for
            # a construction-time argument.
            perceptual_spec = self._loss_weight_table().get("perceptual")
            if perceptual_spec is None or not perceptual_spec.enabled:
                perceptual = None
                lambda_perceptual = 0.0
            else:
                lambda_perceptual = perceptual_spec.weight
                # Reuse the module an earlier pass built. Measured: this is
                # always the live path -- perceptual can only reach the table via
                # a domain list (built by ``_build_list_based_losses``) or via a
                # written ``lambda_perceptual`` (built by the main instantiation
                # loop, which orders ``perceptual`` before ``adversarial``), so
                # the fallback below does not fire for any arm that loads today.
                perceptual = self._losses.get("perceptual", None)
                if perceptual is None and lambda_perceptual > 0:
                    # Kept as a build-order safety net, but WITHOUT the
                    # ``except Exception: logger.debug`` that used to wrap it
                    # (NN3): a perceptual loss the author DECLARED must fail the
                    # run when it cannot be constructed, never train at zero.
                    perceptual = create_loss("perceptual").to(self._device, non_blocking=True)

            # The four reconstruction-side weights come from the same DECLARED
            # table as perceptual (#1949). They were the last raw readers of
            # ``recon_config`` here, and ``lambda_l1`` was the damaging one:
            # its schema default is **10.0** and nothing gates it, so every arm
            # that never mentioned an L1 was handed one at 10.0 -- and unlike
            # its six siblings in ``CompositeGANLoss.compute_generator_loss``,
            # the ``l1_loss`` term carries no ``if self.lambda_l1 > 0`` guard,
            # so it was added to the generator objective unconditionally.
            #
            # That is not a paired-reconstruction default landing in a paired
            # arm: it lands hardest on the unpaired ones (CycleGAN, CUT,
            # StarGAN-v2), where ``fake_images`` and ``real_images`` are not the
            # same subject and an L1 between them is a term nobody chose.
            #
            # ``recon_config`` is consequently no longer read for any weight in
            # this method; it is kept in the signature only because three test
            # modules call the method positionally.
            # The last two raw readers, and the same defect as ``lambda_l1``
            # above wearing an enable flag. ``get_enabled_losses`` pairs
            # ``lambda_gp`` with ``enable_gradient_penalty`` and
            # ``feature_matching`` with ``enable_feature_matching``; this builder
            # -- the surface that actually CONSTRUCTS the term -- read the weight
            # and ignored the flag, so ``enable_gradient_penalty: false`` bought a
            # penalty at the schema default of 10.0. That is what killed
            # ``experiment_11_sense_bridge_critic`` on 2026-09-16: the penalty's
            # ``autograd.grad`` double-backward is not reducible under DeepSpeed
            # ZeRO-2 and the run died in ``reduce_ipg_grads``, for a term the arm
            # had switched off in writing (non-negotiables 8 and 17).
            lambda_gp = gan_config.lambda_gp if gan_config.enable_gradient_penalty else 0.0
            lambda_feat_match = (
                gan_config.feature_matching if gan_config.enable_feature_matching else 0.0
            )
            gan_loss = create_loss(
                "gan_composite",
                adv_strategy=adv_strategy,
                perceptual_loss=perceptual,
                lambda_l1=self._declared_weight("l1"),
                lambda_perceptual=lambda_perceptual,
                lambda_adv=gan_config.lambda_adv,
                lambda_feat_match=lambda_feat_match,
                lambda_gp=lambda_gp,
                lambda_ssim=self._declared_weight("ssim"),
                lambda_ms_ssim=self._declared_weight("ms_ssim"),
                lambda_lpips=self._declared_weight("lpips"),
            ).to(self._device, non_blocking=True)

            self._losses["adversarial"] = gan_loss
            logger.info(f"Created adversarial loss (type={gan_loss_type})")
        except ConfigurationError:
            # Already a descriptive config error (e.g. unknown gan_loss_type) —
            # propagate unchanged rather than re-wrapping it.
            raise
        except Exception as e:
            raise ConfigurationError(f"Failed to create adversarial loss: {e}") from e

    # Aliases to keep API compatibility with pipelines calling build_X_losses()
    def build_reconstruction_losses(self) -> "LossBuilder":
        """build_reconstruction_losses.

        Returns:
            'LossBuilder': Description.
        """
        self._build_all_dynamic()
        return self

    def build_adversarial_losses(self) -> "LossBuilder":
        """build_adversarial_losses.

        Returns:
            'LossBuilder': Description.
        """
        self._build_all_dynamic()
        return self

    def build_physics_losses(self) -> "LossBuilder":
        """build_physics_losses.

        Returns:
            'LossBuilder': Description.
        """
        self._build_all_dynamic()
        return self

    def build_regularization_losses(self) -> "LossBuilder":
        """build_regularization_losses.

        Returns:
            'LossBuilder': Description.
        """
        self._build_all_dynamic()
        return self

    def build_structural_losses(self) -> "LossBuilder":
        """build_structural_losses.

        Returns:
            'LossBuilder': Description.
        """
        self._build_all_dynamic()
        return self

    def build_diffusion_losses(self) -> "LossBuilder":
        """build_diffusion_losses.

        Returns:
            'LossBuilder': Description.
        """
        self._build_all_dynamic()
        return self

    def build_latent_losses(self) -> "LossBuilder":
        """build_latent_losses.

        Returns:
            'LossBuilder': Description.
        """
        self._build_all_dynamic()
        return self

    def build_ssl_losses(self) -> "LossBuilder":
        """build_ssl_losses.

        Returns:
            'LossBuilder': Description.
        """
        self._build_all_dynamic()
        return self

    def _strategy_class(self) -> type:
        """The arm's strategy class, resolved through the framework's own dispatcher.

        One owner (NN17): ``TrainingStrategyFactory.get_strategy_class`` is the
        two-rung resolver the run itself uses (``training.strategy_class`` first,
        then ``training_mode``), so reading it here cannot disagree with the class
        the pipeline goes on to instantiate. Imported lazily to match this file's
        other cross-package readers.
        """
        from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

        return TrainingStrategyFactory().get_strategy_class(self._config)

    def validate(self) -> "LossBuilder":
        """Refuse an empty loss stack unless the strategy declares it owns the objective.

        An empty stack is a defect for the arms that expect the builder to feed
        them and the DESIGN for the arms that compute their objective inline. The
        two are told apart by the strategy's own declaration, never by guessing.

        Returns:
            'LossBuilder': self, for chaining.

        Raises:
            ConfigurationError: nothing was built and the strategy has not declared
                inline ownership, or it could not be resolved to ask.
        """
        if not self._losses:
            from spectramr.infrastructure.training.strategies.loss_folding import (
                declares_inline_objective,
            )

            try:
                strategy_cls = self._strategy_class()
            except (ConfigurationError, ValueError) as exc:
                raise ConfigurationError(
                    "No losses were built by LossBuilder, and the strategy could not be "
                    f"resolved to ask whether that is by design: {exc}"
                ) from exc

            if declares_inline_objective(strategy_cls):
                logger.info(
                    f"Loss validation passed with an empty stack: {strategy_cls.__name__} "
                    "declares inline_losses and folds_image_losses=False, so it computes "
                    "its own objective."
                )
                return self

            raise ConfigurationError(
                f"No losses were built by LossBuilder and {strategy_cls.__name__} does not "
                "declare that it computes its objective inline. Training cannot proceed. "
                "Either declare a loss under 'losses:' (image_losses / kspace_losses / "
                "complex_losses / latent_losses), or set 'inline_losses' and "
                "'folds_image_losses' on the strategy if it owns its objective."
            )
        logger.info(f"Loss validation passed ({len(self._losses)} losses created)")
        return self

    def build(self) -> dict[str, nn.Module]:
        """build.

        Returns:
            dict[str, nn.Module]: Description.
        """
        return dict(self._losses)
