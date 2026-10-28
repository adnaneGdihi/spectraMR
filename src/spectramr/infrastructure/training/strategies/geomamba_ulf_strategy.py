"""GeoMamba-ULF training strategy for ULF→HF MRI super-resolution.

Orchestrates the full forward path described in the GeoMamba-ULF
plan §3:

.. math::

    \\hat I_{HF}^{(c)} = R\\!\\left(
        \\pi^{-1} \\circ f_\\theta^{(c)} \\circ \\pi
        \\circ \\varphi^{-1}(I_{LF}^{(c)})
    \\right)\\Big|_F

Per-step responsibilities:

1. Pull ``input``, ``target`` and per-volume contrast id from the batch.
2. (Optional) Apply :class:`PhysicalUnwarp` (B0 + grad-NL + Maxwell)
   to the input when field maps are present in the batch metadata.
3. Forward through the generator (whose own forward path handles
   the foreground SFC, contrast-conditioned Mamba and background
   re-embedding via the brain mask).
4. Compute composite loss: foreground-masked L1 (or per-contrast
   uncertainty-weighted L1) + optional cubical PH-W2 + optional
   Beltrami diagnostic.
5. Return the loss dict to the trainer.

Per CLAUDE.md, all services are resolved via
:func:`spectramr.infrastructure.di.resolve_service`, never instantiated
directly. The three composite-loss weights (``l1``, ``cubical_ph_w2``,
``beltrami_diagnostic``) come from the loss-weight table
(:func:`~spectramr.infrastructure.training.strategies.loss_folding.declared_loss_weights`):
the ``losses.image_losses`` entries, or an author-written
``losses.<section>.lambda_<name>``; the table refuses a term declared at two
different values, and an undeclared term is an absent one. Per-paradigm
parameters live in ``config.training.geomamba_ulf``.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from spectramr.core.module_utils import unwrap_model
from spectramr.data.batch_types import read_batch_field
from spectramr.infrastructure.distributed.owned_modules import OWNED_MODULE
from spectramr.infrastructure.physics.ulf_unwarp import UlfUnwarp
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.loss_folding import declared_loss_weights

logger = logging.getLogger(__name__)


class GeoMambaULFStrategy(BaseTrainingStrategy):
    """Training strategy for paired ULF→HF MRI super-resolution.

    See module docstring for the per-step responsibilities. The
    strategy is registered in
    :attr:`spectramr.infrastructure.training.strategy_factory.TrainingStrategyFactory.STRATEGY_CLASS_PATHS`
    under the short name ``"geomamba_ulf"``.
    """

    # parallel_compat: The uncertainty weights are an owned module, built in __init__ when
    # enabled and run via call_owned. The SFC linearizer broadcasts rank 0's curve itself.
    parallel_compat = frozenset({"replicated"})

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        geo = getattr(getattr(config, "training", None), "geomamba_ulf", None)
        on = bool(getattr(getattr(geo, "uncertainty", None), "enabled", False))
        own = (("_uncertainty_loss", OWNED_MODULE),) if on else ()
        return (*super().owned_module_declarations(config), *own)

    #: Loss ownership (cohort review 2026-09-03). The three terms are computed
    #: here, and nothing else declared on ``losses.image_losses`` reaches the
    #: objective: this strategy folds none of the builder's modules. The pair
    #: is the contract the loss-ownership witness reads off a strategy class
    #: (``declared_inline_losses`` / ``declared_folds_image_losses`` in
    #: ``loss_folding``, landing with the mrixfields review); a declared entry
    #: outside ``inline_losses`` is a dropped term, not an objective.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset(
        {"l1", "cubical_ph_w2", "beltrami_diagnostic"}
    )
    folds_image_losses: ClassVar[bool | None] = False

    def _setup_strategy_specific_components(self) -> None:
        """Read paradigm-specific parameters from `config.training.geomamba_ulf`."""
        super()._setup_strategy_specific_components()

        geo_cfg = getattr(self.config.training, "geomamba_ulf", None)
        # All three weights come from the loss-weight table: the
        # ``losses.image_losses`` entries or an author-written
        # ``losses.<section>.lambda_<name>`` (the table refuses the two at
        # different values). The strategy used to read ``l1`` from
        # ``losses.reconstruction.lambda_l1`` -- a schema default, never
        # author-written on this cohort -- and the two topology terms through
        # its own resolver, so the same YAML weighed differently here than in
        # every other strategy (cohort review 2026-09-03). An absent entry is
        # an absent term (0.0), never a silent default.
        weights = declared_loss_weights(self.config)
        self._lambda_l1 = float(weights.get("l1", 0.0))
        self._lambda_ph_w2 = float(weights.get("cubical_ph_w2", 0.0))
        self._lambda_beltrami = float(weights.get("beltrami_diagnostic", 0.0))
        if self._lambda_l1 <= 0.0:
            raise ValueError(
                "GeoMambaULFStrategy computes an L1 fidelity term whose weight lives on "
                "losses.image_losses: declare `- {name: l1, weight: <w>}` there."
            )

        # P3 warmup: ramp PH and Beltrami weights linearly from 0 to their
        # target value over the first `warmup_epochs` epochs, then hold.
        topo_cfg = getattr(geo_cfg, "topology", None) if geo_cfg is not None else None
        self._topology_warmup_epochs = (
            int(getattr(topo_cfg, "warmup_epochs", 0)) if topo_cfg is not None else 0
        )
        # The declared PH knobs; _get_ph_loss builds the loss from them.
        self._topology_cfg = topo_cfg
        self._topology_provenance: dict[str, Any] = {}
        # Physical unwarp of the input. Built only when the block is declared, so an
        # arm without it trains as before.
        self._ulf_unwarp = self._build_ulf_unwarp(geo_cfg)

        # Lazy-construct the topology losses on first use to defer any
        # gudhi/POT import cost (and to make the strategy importable
        # without the [topology] extra installed).
        self._ph_loss: torch.nn.Module | None = None
        self._beltrami_loss: torch.nn.Module | None = None

        # FiLM contrast vocabulary — defaults to T1/T2/FLAIR via
        # config.training.geomamba_ulf.film.n_contrasts.
        self._film_n_contrasts = geo_cfg.film.n_contrasts if geo_cfg is not None else 3

        # Bridge the metric-SFC SSOT (config.training.geomamba_ulf.metric_sfc)
        # onto the generator. The model is built from config.model by the
        # factory, which never sees the geomamba_ulf training block, so without
        # this the entire MetricSFCConfig (beta / smoothing_sigma / connectivity
        # / cache_dir) was inert and the model's lazy linearizer always used the
        # hard-coded defaults — making abl_no_metric_sfc (beta=0) byte-identical
        # to v0 and the beta-HPO sweep six identical runs. Applied at setup,
        # before any forward, so it lands before the lazy linearizer is built.
        self._apply_metric_sfc_config(geo_cfg)

        # P2 uncertainty-weighted multi-contrast L1. When enabled it
        # *replaces* the foreground-masked L1 above (Kendall & Gal 2018).
        unc_cfg = getattr(geo_cfg, "uncertainty", None) if geo_cfg is not None else None
        self._uncertainty_enabled = bool(getattr(unc_cfg, "enabled", False))
        self._uncertainty_loss: torch.nn.Module | None = None
        if self._uncertainty_enabled:
            from spectramr.models.losses.masked_uncertainty_multi_contrast_loss import (
                MaskedUncertaintyMultiContrastLoss,
            )

            self._uncertainty_loss = MaskedUncertaintyMultiContrastLoss(
                n_contrasts=self._film_n_contrasts,
                init_log_sigma=float(getattr(unc_cfg, "init_log_sigma", 0.0)),
                eps=float(getattr(unc_cfg, "eps", 1.0e-6)),
            )
            if self.device is not None:
                self._uncertainty_loss = self._uncertainty_loss.to(self.device)
            self.register_owned_module("_uncertainty_loss", self._uncertainty_loss)
            # Register the learnable per-contrast log-sigma weights with opt_g
            # so the P2 uncertainty weighting actually trains (it was created
            # but never moved to device nor handed to any optimizer).
            opt = getattr(self.env, "opt_g", None)
            if opt is not None:
                existing = {p for g in opt.param_groups for p in g["params"]}
                new_params = [p for p in self._uncertainty_loss.parameters() if p not in existing]
                if new_params:
                    opt.add_param_group({"params": new_params, "lr": 1e-3})

    @staticmethod
    def _build_ulf_unwarp(geo_cfg: Any) -> UlfUnwarp | None:
        """Build the unwarp from ``training.geomamba_ulf.unwarp`` when it is declared."""
        if geo_cfg is None or "unwarp" not in getattr(geo_cfg, "model_fields_set", ()):
            return None
        return UlfUnwarp(geo_cfg.unwarp)

    def _prepare_input(self, input_batch: torch.Tensor, batch: Any) -> torch.Tensor:
        """The model's input after the physical unwarp.

        The one place the unwarp is applied. Validation must call it too, so
        training and validation see the same correction.
        """
        if self._ulf_unwarp is None:
            return input_batch
        return self._ulf_unwarp.apply(input_batch, batch)

    def _apply_metric_sfc_config(self, geo_cfg: Any) -> None:
        """Push ``config.training.geomamba_ulf.metric_sfc`` onto the generator.

        The generator is built from ``config.model`` by the model factory,
        which never sees the ``geomamba_ulf`` training block. Without this
        bridge the ``MetricSFCConfig`` knobs are inert and the model's lazy
        ``MetricSFCLinearizer`` always uses its hard-coded defaults. We set the
        values on the (unwrapped) model before any forward runs, so they land
        before the linearizer is lazily built on the first step.
        """
        metric_cfg = getattr(geo_cfg, "metric_sfc", None) if geo_cfg is not None else None
        if metric_cfg is None:
            return
        model = getattr(self.env, "generator", None)
        model = unwrap_model(model)
        if model is None or not hasattr(model, "sfc_beta"):
            # The block is default-filled; only an arm that wrote it expects it to act.
            if "metric_sfc" in getattr(geo_cfg, "model_fields_set", ()):
                raise ValueError(
                    "training.geomamba_ulf.metric_sfc is declared, but the generator "
                    f"{type(model).__name__} has no metric space-filling curve "
                    "(no `sfc_beta`), so nothing would apply it (#1752). Remove the "
                    "block or use a model with a metric SFC linearizer."
                )
            return
        model.sfc_beta = float(getattr(metric_cfg, "beta", model.sfc_beta))
        model.sfc_smoothing_sigma = float(
            getattr(metric_cfg, "smoothing_sigma", model.sfc_smoothing_sigma)
        )
        model.sfc_connectivity = int(getattr(metric_cfg, "connectivity", model.sfc_connectivity))
        model.sfc_cache_dir = getattr(metric_cfg, "cache_dir", model.sfc_cache_dir)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        model = kwargs.get("model") or getattr(self.env, "generator", None)
        if model is None:
            raise RuntimeError(
                "GeoMambaULFStrategy: no generator available — pass "
                "`model=` kwarg or ensure `env.generator` is set."
            )

        batch = kwargs.get("batch", {}) or {}
        mask = self._extract_mask(batch, input_batch)
        contrast_id = self._extract_contrast_id(batch, input_batch)

        prediction = model(
            self._prepare_input(input_batch, batch),
            mask=mask,
            contrast_id=contrast_id,
        )

        # Reconstruction loss: per-contrast uncertainty-weighted L1
        # (P2) or vanilla foreground-masked L1 (v0/P0+P1 default).
        losses: dict[str, torch.Tensor] = {}
        if self._uncertainty_loss is not None:
            loss_recon = self.call_owned(
                "_uncertainty_loss",
                prediction,
                target_batch,
                mask=mask,
                contrast_id=contrast_id,
            )
            losses["loss_uncertainty_multi_contrast"] = loss_recon
        elif mask is not None:
            mask_5d = self._broadcast_mask(mask, prediction)
            n_fg = mask_5d.sum().clamp_min(1.0)
            loss_recon = (mask_5d * (prediction - target_batch).abs()).sum() / n_fg
            losses["loss_l1"] = loss_recon
        else:
            loss_recon = F.l1_loss(prediction, target_batch)
            losses["loss_l1"] = loss_recon

        g_total_loss = self._lambda_l1 * loss_recon

        # P3 warmup factor for the topology losses. Linearly ramps both
        # PH and Beltrami weights from 0 -> target over `warmup_epochs`.
        warmup = self._topology_warmup_factor(epoch)

        # Optional: cubical PH-W2 loss. Lazy-constructed so the strategy
        # can be loaded without the [topology] extra. Skipped silently
        # when weight is 0.
        effective_ph_w2 = self._lambda_ph_w2 * warmup
        if effective_ph_w2 > 0:
            # _get_ph_loss raises if the [topology] extra is missing — no silent
            # skip (the PH term is this arm's headline claim).
            ph_loss_module = self._get_ph_loss()
            loss_ph = ph_loss_module(prediction, target_batch, mask=mask)
            losses["loss_cubical_ph_w2"] = loss_ph
            g_total_loss = g_total_loss + effective_ph_w2 * loss_ph

        # Optional: Beltrami on the model's displacement field. The head must be on:
        # the old check on a 2-channel intensity output was never reachable.
        effective_beltrami = self._lambda_beltrami * warmup
        if effective_beltrami > 0:
            displacement = getattr(unwrap_model(model), "last_displacement", None)
            if displacement is None:
                raise ValueError(
                    f"beltrami_diagnostic is weighted {self._lambda_beltrami} but the model "
                    "produced no displacement field. Set model.model_kwargs.displacement_head: "
                    "true, or delete the beltrami_diagnostic entry."
                )
            loss_bel = self._get_beltrami_loss()(prediction, displacement=displacement)
            losses["loss_beltrami_diagnostic"] = loss_bel
            g_total_loss = g_total_loss + effective_beltrami * loss_bel

        losses["g_total_loss"] = g_total_loss
        return losses

    @torch.no_grad()
    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        *,
        batch_data: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Score the volume with its own contrast and brain mask, as training does.

        FiLM and the foreground SFC both read the contrast and the mask, so the
        default validation (which passes neither) scores a different model.
        """
        computer = self.validation_metrics_computer
        if not computer.get_metric_names():
            return {}
        batch = batch_data if batch_data is not None else {}
        input_batch = input_batch.to(self.device, non_blocking=True)
        target_batch = target_batch.to(self.device, non_blocking=True)
        mask = self._extract_mask(batch, input_batch)
        contrast_id = self._extract_contrast_id(batch, input_batch)
        if contrast_id is None and self._film_n_contrasts > 1:
            raise ValueError(
                "GeoMambaULF validation needs each volume's contrast_id/contrast_idx: "
                f"film.n_contrasts={self._film_n_contrasts} and the batch carries neither. "
                "Defaulting to contrast 0 would score the wrong FiLM parameters."
            )
        generator = self.env.generator
        generator.eval()
        output = generator(
            self._prepare_input(input_batch, batch), mask=mask, contrast_id=contrast_id
        )
        metrics = self._score_validation_output(output, target_batch)
        generator.train()
        return metrics

    def _topology_warmup_factor(self, epoch: int) -> float:
        """Linear 0 → 1 ramp over the first ``warmup_epochs`` epochs."""
        warmup_epochs = self._topology_warmup_epochs
        if warmup_epochs <= 0:
            return 1.0
        return float(min(max(epoch + 1, 0) / warmup_epochs, 1.0))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_ph_loss(self) -> torch.nn.Module:
        if self._ph_loss is not None:
            return self._ph_loss
        try:
            from spectramr.config.schemas.training.geomamba_ulf import TopologyConfig
            from spectramr.models.losses.cubical_ph_w2_loss import (
                CubicalPHWassersteinLoss,
            )

            topo = self._topology_cfg if self._topology_cfg is not None else TopologyConfig()
            self._ph_loss = CubicalPHWassersteinLoss(
                homology_dims=tuple(topo.ph_dimensions),
                wasserstein_p=topo.wasserstein_p,
                stratify_subvolumes=topo.stratify_subvolumes,
            )
            # Declared and applied side by side, so a reader can see a dropped knob.
            self._topology_provenance = {
                "cubical_ph_w2": {
                    "declared": {
                        "ph_dimensions": list(topo.ph_dimensions),
                        "wasserstein_p": topo.wasserstein_p,
                        "stratify_subvolumes": topo.stratify_subvolumes,
                    },
                    "applied": {
                        "homology_dims": list(self._ph_loss.homology_dims),
                        "wasserstein_p": self._ph_loss.wasserstein_p,
                        "stratify_subvolumes": self._ph_loss.stratify_subvolumes,
                    },
                }
            }
            return self._ph_loss
        except ImportError as exc:
            # Fail loud, do NOT warn-and-skip. This method is reached only when
            # cubical_ph_w2 weight > 0, i.e. the arm's headline claim IS the
            # persistent-homology term. Silently returning None degraded the arm
            # to plain L1 while it smoke-PASSed — pitfall #16 (inert facade) +
            # #10 (silent degrade) at the strategy layer, after the loss class
            # itself correctly raises (#9). Install with:
            #     pip install -e ".[topology]"   (or: make install-topology)
            raise RuntimeError(
                "GeoMambaULFStrategy: cubical_ph_w2 weight is non-zero but the "
                "[topology] extra (gudhi/POT) is not installed. Install it with "
                '`pip install -e ".[topology]"` or `make install-topology`, or '
                "set the cubical_ph_w2 weight to 0 to disable the PH term."
            ) from exc

    def _first_steps_extra(self, epoch: int, batch: Any = None) -> dict[str, Any]:
        """Stamp the resolved PH knobs (pitfall 15) into the first-steps snapshot."""
        extra = super()._first_steps_extra(epoch, batch=batch)
        extra.update(self._topology_provenance)
        if self._ulf_unwarp is not None:
            extra["unwarp"] = self._ulf_unwarp.provenance()
        return extra

    def _get_beltrami_loss(self) -> torch.nn.Module | None:
        if self._beltrami_loss is not None:
            return self._beltrami_loss
        from spectramr.models.losses.beltrami_diagnostic_loss import (
            BeltramiDiagnosticLoss,
        )

        self._beltrami_loss = BeltramiDiagnosticLoss(reduction="mean")
        return self._beltrami_loss

    @staticmethod
    def _extract_mask(batch: dict[str, Any], input_batch: torch.Tensor) -> torch.Tensor | None:
        for key in ("foreground_mask", "mask", "brain_mask"):
            mask = read_batch_field(batch, key)
            if mask is None:
                continue
            if hasattr(mask, "data"):  # TorchIO Image-like
                mask = mask.data
            if isinstance(mask, torch.Tensor):
                return mask.to(device=input_batch.device, dtype=input_batch.dtype)
        return None

    @staticmethod
    def _extract_contrast_id(
        batch: dict[str, Any], input_batch: torch.Tensor
    ) -> torch.Tensor | None:
        # Datasets disagree on the key: m4raw / slice_dataset and the default
        # collation emit ``contrast_idx``; mrixfields emits ``contrast_id``.
        # Read BOTH so FiLM contrast-conditioning actually fires — reading only
        # ``contrast_id`` silently dropped the ``contrast_idx`` key the loaders
        # do emit, collapsing the model to the same (gamma, beta) for every
        # contrast. (nifti_paired emits neither yet — backlog item.)
        cid = read_batch_field(batch, "contrast_id", "contrast_idx")
        if cid is None:
            return None
        if isinstance(cid, torch.Tensor):
            return cid.to(device=input_batch.device, dtype=torch.long)
        return torch.tensor(cid, device=input_batch.device, dtype=torch.long)

    @staticmethod
    def _broadcast_mask(mask: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        if mask.shape == prediction.shape:
            return mask
        # Common cases: (B, 1, D, H, W) → broadcast to channels;
        # (B, D, H, W) → unsqueeze(1) and broadcast;
        # (D, H, W) → expand to (B, 1, D, H, W).
        if mask.ndim == 5:
            return mask.expand_as(prediction)
        if mask.ndim == 4:  # (B, D, H, W)
            return mask.unsqueeze(1).expand_as(prediction)
        if mask.ndim == 3:  # (D, H, W)
            return mask.unsqueeze(0).unsqueeze(0).expand_as(prediction)
        raise ValueError(
            f"Cannot broadcast mask of shape {tuple(mask.shape)} to "
            f"prediction shape {tuple(prediction.shape)}"
        )


__all__ = ["GeoMambaULFStrategy"]
