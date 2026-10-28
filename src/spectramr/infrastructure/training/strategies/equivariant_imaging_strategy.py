r"""Equivariant Imaging (EI) reconstruction strategy — Phase A step 3 (un-trap P1.2).

Equivariant Imaging (Chen et al., ICCV 2021) trains a reconstruction network
from undersampled measurements **with no fully-sampled target**. This strategy
emits the reference branch :math:`T_g f(A^* y)` (``context["transformed_recon"]``)
that the registered ``EquivariantSSLReconLoss`` needs to fire (pitfall #16).

Each step computes, for a forward operator :math:`A = M F` built from the
acquisition mask through the complex physics SSOT (``fft2c`` / ``ifft2c``):

* the reconstruction :math:`\hat{x} = f(A^* y)`;
* a measurement-consistency anchor :math:`\lVert A\hat{x} - y\rVert^2`
  (or the nc-χ-aware GSURE risk of the T1 keystone when
  ``robust_correction`` is set — that is the *Robust EI* / ``robust_ei`` arm,
  Chen et al., CVPR 2022); and
* the equivariance term :math:`\lVert f(A T_g \hat{x}) - T_g \hat{x}\rVert`
  for a group element :math:`g` sampled from a brain-appropriate group
  (small-rotation / dihedral, **not** full ``SO(2)``).

The anchor excludes the trivial :math:`f \equiv \text{const}`. Identifiability
(Tachella et al.) requires :math:`A` to break the group symmetry, which the
``ei_sensing_margin`` audit enforces; ``alpha_equivariance == 0`` reduces this to a
measurement-consistency reconstruction, and the step then skips the second
forward and its backward while still drawing the group elements, so the random
stream stays paired with the ``alpha = 1`` arm.

The same ``alpha`` is not the same weight across arms. The measured anchor
averages over the :math:`B \cdot C \cdot H \cdot W` complex k-space entries,
unacquired zeros included (about 78 % at R4), while the equivariance term
averages over the :math:`B \cdot 2 \cdot H \cdot W` reals of the image. The
ratio is fixed within a one-coil or a four-coil pair but differs between them
(about 4x as measured on the M4Raw R4 arms), so ``alpha`` is comparable inside a
pair, not across.

:math:`A = M F` by default, :math:`A = M F S` under ``multicoil_operator`` --
:mod:`spectramr.infrastructure.physics.multicoil_ei_operator` owns that choice
and the identifiability measurement behind it.

``measurement_target: alternate_repetition`` is Rep-EI: the anchor is the declared
``masked_kspace_l2`` against a second repetition, and a k-space network's output is
SENSE-combined before the group acts (``equivariant_imaging_ops``).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch

from spectramr.core.metrics.coil_manifold import as_interleaved
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.physics.group_actions import get_group_action
from spectramr.infrastructure.physics.multicoil_ei_operator import (
    MulticoilEIOperator,
    interleaved_to_complex,
)
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.equivariant_imaging_band import (
    band_project,
    batch_support,
    scored_anchor,
)
from spectramr.infrastructure.training.strategies.equivariant_imaging_ops import (
    alternate_repetition_anchor,
    context_metric_kspace,
    equivariant_imaging_branches,
    measured_anchor,
    model_domains,
    network_image,
    scoring_transform,
    zero_filled_context_metrics,
    zero_filled_metrics,
)
from spectramr.infrastructure.training.strategies.loss_folding import declared_loss_weights
from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)
from spectramr.infrastructure.training.strategies.rng_streams import rank_seeded_generator
from spectramr.infrastructure.training.strategies.validation_chunking import reconstruction_output
from spectramr.models.losses.equivariant_recon_loss import EquivariantSSLReconLoss
from spectramr.models.losses.sure_n2self_losses import GSUREKspaceLoss

logger = logging.getLogger(__name__)


class EquivariantImagingStrategy(ReconstructionTrainingStrategy):
    r"""Self-supervised reconstruction via Equivariant Imaging.

    Reuses the reconstruction plumbing (k-space → image conversion, generator
    forward) and overrides only the loss: a measurement-consistency anchor plus
    the equivariance penalty routed through the registered
    ``EquivariantSSLReconLoss``. No fully-sampled target is read.

    The ``robust_ei`` registry key selects the *Robust EI* arm; it requires
    ``equivariant_imaging.robust_correction = true`` (the key is a promise, the
    knob must honour it — a contradiction raises rather than silently degrading).
    """

    #: Loss ownership (issue #1918). Self-supervised equivariance/measurement-consistency objective -- no ground
    #: truth is used, so no declared image loss can be or is computed.
    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = False
    _ZF_BASELINE_METRICS: ClassVar[tuple[str, ...]] = (
        *ReconstructionTrainingStrategy._ZF_BASELINE_METRICS,
        *ReconstructionTrainingStrategy._ZF_OBJECT_METRICS,
    )
    #: The multi-coil operator raises without the batch's coil maps (#2326).
    validates_on_batch_context: ClassVar[bool] = True
    checkpointed_streams: ClassVar[tuple[str, ...]] = ("_rng",)
    #: The network is fed ``A^H y`` (or ``y``) built inside the step (#2508).
    snapshot_prepared_is_model_input: bool = False
    snapshot_model_input_tag: str | None = "ei_model_input"
    # The validation forward reads y from the context (MulticoilEIOperator.prepare).
    validation_reads_context_measurement = True

    #: The band knobs at their schema defaults, on the class so an instance built
    #: without ``__init__`` (the step tests) reads today's behaviour.
    anchor_mask: str = "input"
    anchor_per_selected: bool = False
    band_limited_target: bool = False

    def __init__(
        self,
        env: TrainingEnvironment | None = None,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env=env, device=device, **kwargs)
        training = self.config.training
        cfg = getattr(training, "equivariant_imaging", None) if training else None
        if cfg is None:
            raise ValueError(
                "EquivariantImagingStrategy requires a "
                "training.equivariant_imaging config block (group, "
                "alpha_equivariance, ...). None was provided."
            )
        self.group_name = str(cfg.group)
        self.alpha = float(cfg.alpha_equivariance)
        self.n_group_samples = int(cfg.n_group_samples)
        self.robust = bool(cfg.robust_correction)
        self.noise_model = str(cfg.noise_model)
        self.n_coils = int(cfg.n_coils)
        self.multicoil_operator = bool(cfg.multicoil_operator)
        self.noise_std = cfg.noise_std_estimate
        self.measurement_target = str(cfg.measurement_target)
        self.anchor_mask = str(cfg.anchor_mask)
        self.anchor_per_selected = str(cfg.anchor_normalization) == "selected"
        self.band_limited_target = str(cfg.equivariance_target) == "band_limited"
        self._consumes_kspace, self._emits_kspace = model_domains(self.config)

        group_kwargs: dict[str, Any] = {}
        if self.group_name == "small_rotation":
            group_kwargs = {"max_angle_deg": float(cfg.max_angle_deg)}
            group_kwargs["num_angles"] = int(cfg.num_angles)
        self._group = get_group_action(self.group_name, **group_kwargs)

        # ``robust_ei`` promises the Jensen/SURE correction (pitfall #9); both keys
        # share one strategy_class, so training_mode tells them apart.
        training_mode = str(getattr(self.config.training, "training_mode", "") or "").lower()
        if training_mode == "robust_ei" and not self.robust:
            raise ValueError(
                "training_mode 'robust_ei' requires "
                "training.equivariant_imaging.robust_correction=true; "
                "use 'equivariant_imaging' for the non-robust arm."
            )

        # Robust EI consumes the T1 keystone GSURE risk as its data-consistency
        # term; build it now so a missing noise estimate fails at construction.
        self._gsure: GSUREKspaceLoss | None = None
        if self.robust:
            if self.noise_std is None:
                raise ValueError(
                    "equivariant_imaging.robust_correction=true requires "
                    "noise_std_estimate (the k-space noise σ); none was set "
                    "(pitfall #15: the knob must be wired)."
                )
            self._gsure = GSUREKspaceLoss(
                sigma=float(self.noise_std),
                noise_model=self.noise_model,
                n_coils=self.n_coils,
            )

        self._owned_ei_loss: EquivariantSSLReconLoss | None = None
        self._rng = rank_seeded_generator(int(cfg.split_seed))
        logger.info(
            "EquivariantImagingStrategy: group=%s, alpha=%.3g, n_samples=%d, robust=%s, "
            "noise_model=%s, multicoil_operator=%s, measurement_target=%s, emits_kspace=%s, "
            "anchor_mask=%s, anchor_per_selected=%s, band_limited_target=%s, image_support=%s",
            *(self.group_name, self.alpha, self.n_group_samples, self.robust, self.noise_model),
            *(self.multicoil_operator, self.measurement_target, self._emits_kspace),
            *(self.anchor_mask, self.anchor_per_selected, self.band_limited_target),
            "coil_maps" if self.multicoil_operator else "none",
        )

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(
            expected_modes=(
                "reconstruction",
                "self_supervised_reconstruction",
                "equivariant_imaging",
                "robust_ei",
            )
        )
        if self.logging_service:
            self._log_config_features(self.logging_service)

    def _resolve_equivariant_loss(self) -> EquivariantSSLReconLoss:
        """The EI loss built from config, else the strategy's own registered instance."""
        env_losses = (getattr(self.env, "losses", None) or {}) if self.env else {}
        for fn in env_losses.values():
            if isinstance(fn, EquivariantSSLReconLoss):
                return fn
        if self._owned_ei_loss is None:
            self._owned_ei_loss = EquivariantSSLReconLoss(norm="l2")
        return self._owned_ei_loss

    def builds_own_model_input(self) -> bool:
        """The multi-coil operator SENSE-combines the coils before an image network (#2240)."""
        return self.multicoil_operator and not getattr(self, "_consumes_kspace", False)

    @classmethod
    def standalone_infer_blocker(cls, config: object) -> str | None:
        from spectramr.infrastructure.training.utils.domain_inference import (
            needs_kspace_to_image_bridge,
        )

        cfg = config.training.equivariant_imaging
        consumes_kspace, _ = model_domains(config)
        if (cfg.multicoil_operator and not consumes_kspace) or needs_kspace_to_image_bridge(config):
            return (
                "this EI arm feeds an image network A^H y built inside the training step "
                "(coil maps, mask, measurement); infer has no such builder and would feed "
                "the file to the network as is"
            )
        return None

    @property
    def _owns_zero_filled_baseline(self) -> bool:  # an image network's val_zf_* (#2325)
        return not getattr(self, "_emits_kspace", False)

    def declared_metric_keys(self) -> frozenset[str]:
        """The terms every training step stamps, so the CSV header has their columns (#1682)."""
        return frozenset({"loss_consistency", "loss_equivariance", "ei_group_element"})

    def _prepare_generator_inputs(
        self, batch_context: dict[str, Any], input_batch: torch.Tensor, *, validation: bool = False
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply A^H = S^H F^H M when an image network inverts the multi-coil operator."""
        if not getattr(self, "_consumes_kspace", False):
            mc = MulticoilEIOperator.resolve(
                self.multicoil_operator, batch_context, batch_context.get("mask")
            )
            if mc is not None:
                input_batch, batch_context = mc.prepare(batch_context, input_batch)
        return super()._prepare_generator_inputs(batch_context, input_batch, validation=validation)

    def _anchor_weight(self) -> float:
        """The declared weight of ``masked_kspace_l2``, resolved once."""
        weight = getattr(self, "_anchor_weight_cache", None)
        if weight is None:
            declared = declared_loss_weights(self.config)
            if "masked_kspace_l2" not in declared:
                raise ValueError(
                    "measurement_target: alternate_repetition scores the declared "
                    "masked_kspace_l2, and the loss-weight table has no entry for it."
                )
            weight = self._anchor_weight_cache = float(declared["masked_kspace_l2"])
        return weight

    def _validation_forward(
        self, input_batch: torch.Tensor, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """The image the training step reconstructs: zero where the maps measure nothing."""
        output = super()._validation_forward(input_batch, batch_context, **kwargs)
        if self._emits_kspace:
            return output
        mc = MulticoilEIOperator.resolve(
            self.multicoil_operator, batch_context, batch_context.get("mask")
        )
        return output if mc is None else mc.restrict(reconstruction_output(output))

    def _context_metric_prediction(self, raw: torch.Tensor, batch_context: dict[str, Any]):
        return context_metric_kspace(raw, batch_context, self.multicoil_operator, self._emits_kspace)

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        *,
        batch_data: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validate on the batch's own measurement, mask and maps: the mixin's bare
        context left the multi-coil operator without maps on every batch (#2326)."""
        context = kwargs.pop("batch_context", None)
        if context is None:
            context = self.validation_batch_context(input_batch, target_batch, batch_data)
        metrics = super().validation_step(
            input_batch, target_batch, batch_data=batch_data, batch_context=context, **kwargs
        )
        if metrics and self._owns_zero_filled_baseline and context is not None:
            # The mixin would grade |k_masked| with the arm's asymmetric transform (#2325).
            mc = MulticoilEIOperator.resolve(self.multicoil_operator, context, context.get("mask"))
            computer, names = self.validation_metrics_computer, self._ZF_BASELINE_METRICS
            transform = scoring_transform(self.config)
            metrics.update(
                zero_filled_metrics(
                    computer, context, target_batch, mc, names, transform=transform, batch_data=batch_data
                )
            )
            metrics.update(zero_filled_context_metrics(computer, context, target_batch, mc, batch_data))
        return metrics

    def _consistency_term(
        self,
        raw_output: torch.Tensor,
        x_hat: torch.Tensor,
        batch_context: dict[str, Any],
        target_batch: torch.Tensor,
        support: torch.Tensor | None,
    ) -> torch.Tensor:
        """The measurement anchor, one owner for the training step and validation's ``val_loss``."""
        y = batch_context["measured_kspace"]
        mask = batch_context["mask"]
        mask_c = mask if mask.is_complex() else mask.to(torch.complex64)
        mc = MulticoilEIOperator.resolve(self.multicoil_operator, batch_context, mask_c)
        forward_op = mc.forward if mc else (lambda x: mask_c * fft2c(interleaved_to_complex(x)))
        if self.measurement_target == "alternate_repetition":
            # The anchor is the declared masked_kspace_l2, at its declared weight
            # (non-negotiable 8): a YAML weight other than 1 used to be ignored.
            consistency = self._anchor_weight() * alternate_repetition_anchor(
                raw_output, target_batch, batch_context, mc, self._emits_kspace, self.env
            )
        elif self._gsure is None and (
            self.anchor_mask == "with_structural_zeros" or self.anchor_per_selected
        ):
            # The structural zeros carry y = 0 by construction, so scoring them needs no reference.
            select = mask.real if mask.is_complex() else mask
            if self.anchor_mask == "with_structural_zeros":
                assert support is not None
                select = ((select > 0) | ~support).to(select.dtype)
            select_c = select.to(torch.complex64)
            if self._emits_kspace:
                a_xhat = select_c * interleaved_to_complex(raw_output)
            else:
                sel_mc = MulticoilEIOperator.resolve(self.multicoil_operator, batch_context, select_c)
                a_xhat = (
                    sel_mc.forward(x_hat)
                    if sel_mc
                    else select_c * fft2c(interleaved_to_complex(x_hat))
                )
            consistency = scored_anchor(
                a_xhat, interleaved_to_complex(y), select, self.anchor_per_selected
            )
        else:
            a_xhat = (
                mask_c * interleaved_to_complex(raw_output)
                if self._emits_kspace
                else forward_op(x_hat)
            )
            consistency = measured_anchor(a_xhat, interleaved_to_complex(y), self._gsure)
        return consistency

    def _validation_loss(
        self,
        raw_prediction: torch.Tensor,
        target_batch: torch.Tensor,
        batch_context: dict[str, Any],
        batch_data: Any,
    ) -> torch.Tensor | None:
        """The training anchor on the validation measurement, not an image-vs-k-space L1."""
        if batch_context.get("measured_kspace") is None or batch_context.get("mask") is None:
            return None
        mask = batch_context["mask"]
        needs_support = self.band_limited_target or self.anchor_mask == "with_structural_zeros"
        support = batch_support(batch_data, mask) if needs_support else None
        x_hat = network_image(raw_prediction, self._validation_operator(batch_context), self._emits_kspace)
        return self._consistency_term(raw_prediction, x_hat, batch_context, target_batch, support)

    def _validation_operator(self, batch_context: dict[str, Any]) -> MulticoilEIOperator | None:
        mask = batch_context["mask"]
        mask_c = mask if mask.is_complex() else mask.to(torch.complex64)
        return MulticoilEIOperator.resolve(self.multicoil_operator, batch_context, mask_c)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch  # EI is self-supervised; no epoch-gated schedule here.
        batch_context = self._prepare_batch_context(input_batch, target_batch, **kwargs)
        y = batch_context.get("measured_kspace")
        mask = batch_context.get("mask")
        if y is None or mask is None:
            raise ValueError(
                "EquivariantImagingStrategy requires batch_context['measured_kspace'] (the "
                "undersampled y) and batch_context['mask']; EI has no fully-sampled target."
            )

        mask_c = mask if mask.is_complex() else mask.to(torch.complex64)

        mc = MulticoilEIOperator.resolve(self.multicoil_operator, batch_context, mask_c)
        forward_op = mc.forward if mc else (lambda x: mask_c * fft2c(interleaved_to_complex(x)))
        raw_outputs: list[torch.Tensor] = []

        def reconstruct(kspace: torch.Tensor) -> torch.Tensor:
            if self._consumes_kspace and torch.is_complex(kspace):
                kspace = as_interleaved(kspace)
            bc = {**batch_context, "measured_kspace": kspace, "mask": mask}
            lr_image, forward_kwargs = self._prepare_generator_inputs(bc, kspace)
            if not raw_outputs:  # x_hat's input, once per step
                keys = {"model_input"} if self._consumes_kspace else set()
                self._declare_model_input(
                    {"model_input": lr_image},
                    extra={"model_input_key": "model_input"},
                    in_kspace_keys=keys,
                    domain_exact=not self._consumes_kspace,
                )
            recon, _ = self._generate_predictions(lr_image, forward_kwargs, bc)
            raw_outputs.append(recon)
            return network_image(recon, mc, self._emits_kspace)

        ei_loss_fn = self._resolve_equivariant_loss()
        support = (
            batch_support(kwargs.get("batch"), mask)
            if self.band_limited_target or self.anchor_mask == "with_structural_zeros"
            else None
        )

        x_hat_ref: torch.Tensor | None = None
        ei_terms: list[torch.Tensor] = []
        last_g = 0
        for _ in range(max(1, self.n_group_samples)):
            # Drawn even when unused, so an alpha = 0 control consumes the stream
            # exactly as its alpha = 1 partner does.
            g = self._group.sample(self._rng)
            last_g = g
            if self.alpha == 0.0:
                continue
            x_hat, prediction, transformed = equivariant_imaging_branches(
                reconstruct, forward_op, self._group, y, g
            )
            if x_hat_ref is None:
                x_hat_ref = x_hat
            if support is not None and self.band_limited_target:
                transformed = band_project(transformed, support)
            if mc is not None:
                # T_g carries anatomy off the maps' support, where f's output is zero
                # by construction; scoring it there is a target no network can reach.
                transformed = mc.restrict(transformed)
            context = {"transformed_recon": transformed}
            ei_terms.append(ei_loss_fn(prediction=prediction, target=target_batch, context=context))
        if ei_terms:
            equivariance = torch.stack(ei_terms).mean()
        else:
            # A zero weight must not cost a second forward, nor let 0 * NaN reach the total.
            x_hat_ref = reconstruct(y)
            equivariance = x_hat_ref.new_zeros(())

        assert x_hat_ref is not None  # loop runs at least once
        consistency = self._consistency_term(
            raw_outputs[0], x_hat_ref, batch_context, target_batch, support
        )

        total = consistency + self.alpha * equivariance if ei_terms else consistency
        return {
            "g_total_loss": total,
            "loss": total,
            "loss_consistency": consistency.detach(),
            "loss_equivariance": equivariance.detach(),
            # A host int already; copying it to the device blocked on every step.
            "ei_group_element": float(last_g),
        }


__all__ = ["EquivariantImagingStrategy", "equivariant_imaging_branches"]
