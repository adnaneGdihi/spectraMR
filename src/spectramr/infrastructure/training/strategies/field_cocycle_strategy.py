"""Cocycle-consistent unified cross-field operator (MICCAI MRIxFields2026, idea 4.2).

Task 3 (any-to-any) with a single conditioned generator whose anti-ensemble property
is *guaranteed* by an algebraic composition (cocycle) law rather than argued. The arm
extends the two-optimizer GAN machinery (:class:`GANTrainingStrategy`) — reusing its
closure-list ``train_step``, discriminator/optimizer wiring and metric reporting — and
swaps only the two closure bodies for the field-conditioned ``encode``/``render`` path
of :class:`FieldCocycleGenerator`.

The generator objective composes five grad-carrying terms (paired L1, latent-cycle,
cocycle, field-identity, hinge-adversarial) plus the builder-folded sharpening losses
(hfen/ms_ssim/lpips). Every term's weight is read from ``loop_state.loss_weight_overrides``
each step, so a top-level ``loss_schedule:`` block drives the curriculum
(``target: adversarial`` warm-up, then ``target: cocycle_consistency`` ramp). The
cocycle residual is stamped as ``last_cocycle_residual`` / ``loss_cocycle`` so the
mechanism is measured (Corollary 5, pitfall #16), and the ``field_cocycle_*`` Tier-1
guards keep the trivial ``G=Id`` solution and per-field routing out.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.training.loop_state import resolve_loop_iteration
from spectramr.infrastructure.training.strategies.gan import GANTrainingStrategy
from spectramr.infrastructure.training.strategies.loss_folding import (
    declared_loss_weights,
    fold_builder_image_losses,
    scheduled_overrides,
)
from spectramr.infrastructure.training.utils.training_utils import clamp_to_range
from spectramr.models.losses.cross_field_losses import (
    CocycleConsistencyLoss,
    FieldIdentityLoss,
    LatentCycleLoss,
)

_N_CONTRAST = 3  # T1w, T2w, T2-FLAIR (mirrors AnatomyFieldRenderer / _CONTRAST_INDEX)
# Discriminator conditioning planes = 1 normalised log-field + one-hot contrast, i.e.
# ``1 + _N_CONTRAST`` (the discriminator's ``cond_channels`` in the YAML). Built inline
# in ``_field_contrast_cond`` from ``_N_CONTRAST``.
_SUPPORTED_GAN_LOSSES = ("hinge", "lsgan", "vanilla", "gan", "bce")


def _disc_adv_loss(loss_type: str, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Discriminator adversarial loss (real should score high, fake low)."""
    if loss_type == "hinge":
        return F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
    if loss_type == "lsgan":
        return 0.5 * (real - 1.0).pow(2).mean() + 0.5 * fake.pow(2).mean()
    if loss_type in ("vanilla", "gan", "bce"):
        return F.binary_cross_entropy_with_logits(
            real, torch.ones_like(real)
        ) + F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
    raise ValueError(  # pitfall #9/#15: unimplemented type must fail loudly
        f"field_cocycle: unsupported losses.gan.gan_loss_type={loss_type!r}; "
        f"supported: {_SUPPORTED_GAN_LOSSES} (wgan-gp needs a gradient penalty this "
        "arm does not wire)."
    )


def _gen_adv_loss(loss_type: str, fake: torch.Tensor) -> torch.Tensor:
    """Generator adversarial loss (fake should fool the discriminator)."""
    if loss_type == "hinge":
        return -fake.mean()
    if loss_type == "lsgan":
        return 0.5 * (fake - 1.0).pow(2).mean()
    if loss_type in ("vanilla", "gan", "bce"):
        return F.binary_cross_entropy_with_logits(fake, torch.ones_like(fake))
    raise ValueError(
        f"field_cocycle: unsupported losses.gan.gan_loss_type={loss_type!r}; "
        f"supported: {_SUPPORTED_GAN_LOSSES}."
    )


class FieldCocycleTranslationStrategy(GANTrainingStrategy):
    """Adversarial cocycle-consistent unified cross-field translator (idea 4.2)."""

    #: Loss ownership (mrixfields review 2026-09-03): folds the builder's other declared entries (calls the fold or the parent path).
    inline_losses: ClassVar[frozenset[str] | None] = frozenset(
        {"cocycle_consistency", "field_identity", "l1", "latent_cycle"}
    )
    folds_image_losses: ClassVar[bool | None] = True

    # AnatomyFieldRenderer conditions the FiLM on these axes; declared so the
    # conditioning audit passes (mirrors ReconstructionTrainingStrategy).
    _SUPPORTED_CONDITION_SOURCES = ("field_strength", "contrast_id")

    #: Registered losses computed INLINE by ``_build_generator_closure`` above. They
    #: are declared on ``losses.image_losses`` so a ``loss_schedule`` rule can resolve
    #: a base weight through the loss-weight SSOT (which sees only ``losses.*``); the
    #: fold must therefore skip them or every term would be counted twice.

    def _setup_strategy_specific_components(self) -> None:
        # Reuse the GAN adversarial setup (UnifiedGANLossComputer, metric reporters,
        # _step_counter) but accept the field_cocycle mode.
        self.setup_adversarial(expected_modes=("field_cocycle",))
        self._step_counter = 0

        cfg = getattr(self.config.training, "field_cocycle", None)

        def _get(name: str, default: float) -> float:
            return float(getattr(cfg, name, default)) if cfg is not None else default

        self._reference_field = _get("reference_field_tesla", 3.0)
        self._field_min = _get("field_min_tesla", 0.1)
        self._field_max = _get("field_max_tesla", 7.0)
        self._cocycle_weight = _get("cocycle_weight", 1.0)
        self._identity_weight = _get("identity_weight", 0.5)
        self._latent_cycle_weight = _get("latent_cycle_weight", 0.1)
        self._adversarial_weight = _get("adversarial_weight", 1.0)
        self._detach_inner = bool(getattr(cfg, "detach_inner", False)) if cfg else False
        self._contrast_conditioning = (
            bool(getattr(cfg, "contrast_conditioning", True)) if cfg else True
        )

        # gan_loss_type is a real, validated knob (#15): read it and RAISE on an
        # unimplemented value rather than silently computing hinge.
        gan_cfg = self.config.losses.gan if self.config.losses is not None else None
        self._gan_loss_type = str(getattr(gan_cfg, "gan_loss_type", "hinge") or "hinge").lower()
        if self._gan_loss_type not in _SUPPORTED_GAN_LOSSES:
            raise ValueError(
                f"field_cocycle: unsupported losses.gan.gan_loss_type="
                f"{self._gan_loss_type!r}; supported: {_SUPPORTED_GAN_LOSSES}."
            )
        # R1 is NOT wired into the field-conditioned hinge closures (the base
        # UnifiedGANLossComputer R1 path is bypassed). Reading r1_interval and doing
        # nothing would be a silent no-op (#15) that lets a copied arm expect
        # discriminator regularisation it never gets, inviting adversarial collapse —
        # so fail loud instead of degrading.
        r1_interval = int(getattr(gan_cfg, "r1_interval", 0) or 0)
        if r1_interval > 0:
            raise ValueError(
                "field_cocycle does not implement R1 discriminator regularisation "
                f"(its closures compute the raw {self._gan_loss_type} term directly); "
                f"got losses.gan.r1_interval={r1_interval}. Set r1_interval: 0."
            )

        # Registered loss modules ARE the runtime path (weight applied by the strategy
        # from the curriculum; module weight stays 1.0). Keeps them non-facade.
        self._cocycle_loss = CocycleConsistencyLoss(weight=1.0)
        self._identity_loss = FieldIdentityLoss(weight=1.0)
        self._latent_cycle_loss = LatentCycleLoss(weight=1.0)

        # Exposure contract (Corollary 5): the last measured epsilon_coc.
        self.last_cocycle_residual: torch.Tensor | None = None
        # The residual that actually discriminates the method from the G=Id facade:
        # ||G(x;s,t) - G(x;s,s)||. See docs/lean/MRIxFields/MRIxFields/Cocycle.lean.
        self.last_field_sensitivity: torch.Tensor | None = None

        # Mapping-like batch captured by ``train_step`` so the closures can read the
        # field/contrast scalars the GAN unpack path drops. The training loop wraps the
        # dict batch in a TrainingBatch dataclass before train_step, so this is a
        # ``.get()``-able TrainingBatch at runtime, not a dict.
        self._current_batch: Any = None

        if getattr(self, "logging_service", None):
            self.logging_service.log_info(
                "FieldCocycleTranslationStrategy: ref="
                f"{self._reference_field}T range=[{self._field_min},{self._field_max}]T "
                f"w(coc={self._cocycle_weight},id={self._identity_weight},"
                f"lat={self._latent_cycle_weight},adv={self._adversarial_weight}) "
                f"gan={self._gan_loss_type} detach_inner={self._detach_inner}"
            )

    # ------------------------------------------------------------------ helpers

    def _scheduled(self) -> dict[str, float]:
        """Curriculum overrides published to loop_state each step (empty if none)."""
        return scheduled_overrides(self)

    def _field_contrast_cond(
        self, ref: torch.Tensor, b_field: torch.Tensor, cid: torch.Tensor | None
    ) -> torch.Tensor:
        """``[B, 1+_N_CONTRAST, H, W]`` discriminator conditioning planes.

        A normalised log-field plane (in ``[0, 1]`` over the configured range) plus
        the contrast one-hot, broadcast spatially and concatenated to the image.
        """
        b, _, h, w = ref.shape
        lf = torch.log(b_field.reshape(-1, 1).float().clamp_min(1e-6))
        lo, hi = math.log(self._field_min), math.log(self._field_max)
        t = ((lf - lo) / (hi - lo)).clamp(0.0, 1.0)  # [B, 1]
        field_plane = t.view(b, 1, 1, 1).expand(b, 1, h, w)
        if self._contrast_conditioning and cid is not None:
            seq = F.one_hot(cid.long().reshape(-1), _N_CONTRAST).to(ref.dtype)
        else:
            seq = torch.zeros(b, _N_CONTRAST, device=ref.device, dtype=ref.dtype)
        contrast_planes = seq.view(b, _N_CONTRAST, 1, 1).expand(b, _N_CONTRAST, h, w)
        return torch.cat([field_plane, contrast_planes], dim=1)

    def _sample_intermediate_field(self, b_t: torch.Tensor) -> torch.Tensor:
        """Log-uniform intermediate field ``m`` per sample (distinct from ``t`` a.s.)."""
        n = b_t.reshape(-1).shape[0]
        lo, hi = math.log(self._field_min), math.log(self._field_max)
        u = torch.rand(n, device=b_t.device, dtype=torch.float32)
        return torch.exp(lo + u * (hi - lo))

    def _batch_fields(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Resolve (source field, target field, contrast id) from the stashed batch."""
        batch = self._current_batch or {}
        b_t = batch.get("field_strength_target", batch.get("field_strength"))
        b_s = batch.get("field_strength", b_t)
        if b_t is None:
            raise ValueError(
                "FieldCocycleTranslationStrategy needs 'field_strength_target' in the "
                "batch (set data.expose_field_strength_target on the mrixfields "
                "dataset)."
            )
        b_t = b_t.to(device)
        b_s = b_s.to(device)
        cid = batch.get("contrast_id")
        cid = cid.to(device) if (cid is not None and self._contrast_conditioning) else None
        return b_s, b_t, cid

    # ------------------------------------------------------------------ train_step

    def train_step(
        self,
        batch: Any,
        epoch: int,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        iteration: int = 0,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        # Capture the mapping-like batch so the closures can read the field/contrast
        # scalars the GAN unpack path discards (it returns only input/target tensors).
        # The loop passes a TrainingBatch (a @dataclass with .get(), NOT a dict), so
        # guard on ``.get`` rather than ``isinstance(dict)`` — the dict-only guard left
        # _current_batch=None and crashed every field_cocycle step at iteration 0.
        self._current_batch = batch if hasattr(batch, "get") else None
        return super().train_step(
            batch,
            epoch,
            input_batch=input_batch,
            target_batch=target_batch,
            iteration=iteration,
            **kwargs,
        )

    def _train_discriminator_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        discriminator: nn.Module,
        epoch: int,
        iteration: int,
        losses_dict: dict[str, Any] | None = None,
    ) -> Any:
        gen = self.env.generator
        _b_s, b_t, cid = self._batch_fields(target_batch.device)

        def d_closure() -> torch.Tensor:
            if isinstance(discriminator, nn.Module):
                discriminator.requires_grad_(True)
            with torch.no_grad():
                x_hat = gen.render(gen.encode(input_batch), b_t, cid)
            cond_t = self._field_contrast_cond(target_batch, b_t, cid)
            real_logits = discriminator(torch.cat([target_batch, cond_t], dim=1))
            fake_logits = discriminator(torch.cat([x_hat, cond_t], dim=1))
            d_total = _disc_adv_loss(self._gan_loss_type, real_logits, fake_logits)
            with torch.no_grad():
                self._last_step_metrics["d_total_loss"] = d_total.detach()
            return d_total

        return d_closure

    def _train_generator_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        discriminator: nn.Module,
        epoch: int,
        iteration: int,
        losses_dict: dict[str, Any] | None = None,
    ) -> Any:
        gen = self.env.generator
        b_s, b_t, cid = self._batch_fields(target_batch.device)

        def g_closure() -> torch.Tensor:
            if isinstance(discriminator, nn.Module):
                discriminator.requires_grad_(False)
            components: dict[str, torch.Tensor] = {}

            q = gen.encode(input_batch)  # Phi(x)
            x_hat = gen.render(q, b_t, cid)  # G(x; s, t) direct
            if getattr(self.config.training, "enforce_output_range", False):
                x_hat = clamp_to_range(x_hat, enable=True, telemetry=False)

            # Paired fidelity (always on — pins away from G=Id collapse).
            loss_fid = F.l1_loss(x_hat, target_batch)
            # Latent-cycle (coboundary consistency of the encode-once architecture).
            loss_latent = self._latent_cycle_loss(gen.encode(x_hat), q)
            # Field-identity: render back at the source field reproduces the source.
            x_id = gen.render(q, b_s, cid)  # G(x; s, s)
            loss_identity = self._identity_loss(x_id, input_batch)
            # Cocycle: compose through a freely-sampled intermediate field m and match
            # the direct render (detached target -> composite is pulled to the
            # fidelity-supervised branch, which is the stable direction).
            b_m = self._sample_intermediate_field(b_t)
            x_m = gen.render(q, b_m, cid)
            inner = x_m.detach() if self._detach_inner else x_m
            x_mt = gen.render(gen.encode(inner), b_t, cid)
            loss_cocycle = self._cocycle_loss(x_mt, x_hat.detach())
            # Adversarial (hinge/lsgan/vanilla against the field-conditioned D).
            cond_t = self._field_contrast_cond(x_hat, b_t, cid)
            fake_logits = discriminator(torch.cat([x_hat, cond_t], dim=1))
            loss_adv = _gen_adv_loss(self._gan_loss_type, fake_logits)

            # Curriculum: scheduled weight supersedes the static config weight.
            sched = self._scheduled()
            w_lat = sched.get("latent_cycle", self._latent_cycle_weight)
            w_coc = sched.get("cocycle_consistency", self._cocycle_weight)
            w_id = sched.get("field_identity", self._identity_weight)
            w_adv = sched.get("adversarial", self._adversarial_weight)

            g_total = (
                loss_fid
                + w_lat * loss_latent
                + w_coc * loss_cocycle
                + w_id * loss_identity
                + w_adv * loss_adv
            )

            # Fold the declarative sharpening losses (hfen/ms_ssim/lpips) with the
            # same curriculum map; the inline l1 placeholder and the four terms
            # computed above are skipped (no double count).
            aux = fold_builder_image_losses(
                self.env.losses,
                declared_loss_weights(self.config),
                sched,
                x_hat,
                target_batch,
                components,
                inline_managed=self.inline_losses,
            )
            if aux is not None:
                g_total = g_total + aux

            with torch.no_grad():
                # Anti-facade instrument (MRIxFields/Cocycle.lean). A zero cocycle
                # residual certifies NOTHING: the field-blind autoencoder R(q,b) = D(q)
                # with D∘E = id is exactly G = Id, and it drives cocycle, latent-cycle
                # AND field-identity to zero (`IsFacade.all_residuals_zero`). What
                # separates the method from that facade is whether the render actually
                # MOVES when only the target field changes:
                #     field_sensitivity = ||G(x; s,t) - G(x; s,s)|| = ||x_hat - x_id||
                # and `fieldSensitivity_ge_of_fidelity` certifies the floor
                #     delta - eps_fid - eps_id   with delta = ||y_t - x||,
                # so a run whose measured sensitivity sits at ~0 while delta is large is
                # collapsed, whatever its cocycle_residual says. Both tensors are already
                # in hand — this costs no extra forward pass.
                field_sens = torch.mean(torch.abs(x_hat - x_id))
                delta = torch.mean(torch.abs(target_batch - input_batch))
                self.last_field_sensitivity = field_sens
                self._last_step_metrics["field_sensitivity"] = field_sens
                self._last_step_metrics["field_gap_delta"] = delta.detach()
                self._last_step_metrics["field_sensitivity_floor"] = (
                    delta - loss_fid.detach() - loss_identity.detach()
                )
                self.last_cocycle_residual = loss_cocycle.detach()
                self._last_step_metrics["g_total_loss"] = g_total.detach()
                self._last_step_metrics["loss_fidelity"] = loss_fid.detach()
                self._last_step_metrics["loss_latent_cycle"] = loss_latent.detach()
                self._last_step_metrics["loss_cocycle"] = loss_cocycle.detach()
                self._last_step_metrics["cocycle_residual"] = loss_cocycle.detach()
                self._last_step_metrics["loss_field_identity"] = loss_identity.detach()
                self._last_step_metrics["g_adversarial"] = loss_adv.detach()
                for k, v in components.items():
                    self._last_step_metrics[k] = v
                if hasattr(self, "_compute_training_metrics"):
                    current_step = resolve_loop_iteration(self)
                    train_metrics = self._compute_training_metrics(
                        pred=x_hat,
                        target=target_batch,
                        config=self.config,
                        current_step=current_step,
                    )
                    self._last_step_metrics.update(
                        {
                            k: v.detach() if isinstance(v, torch.Tensor) else v
                            for k, v in train_metrics.items()
                        }
                    )
            return g_total

        return g_closure

    # ------------------------------------------------------------------ validation

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation-image prediction = render at the sample's target field.

        Without this the pipeline's visual-capture seam falls through to its
        unconditional fallback ``generator(input_batch)``, which
        :class:`FieldCocycleGenerator` rejects (``field_strength`` is keyword-only and
        required). That raise is caught and logged as a warning, so the run keeps
        going and simply saves NO validation image — the 2026-07-23 sweep produced 49
        such warnings and zero pictures across an 8-hour ``ablate_cocycle`` run while
        reporting ``status: OK``. Mirrors
        :meth:`CrossFieldTranslationStrategy._validation_forward` (pitfall #16: the
        image must show the actually-translated output, not a fieldless call).
        """
        x0 = batch_context.get("input", input_batch)
        b_t = batch_context.get("field_strength_target", kwargs.get("field_strength_target"))
        if b_t is None:
            raise ValueError(
                "FieldCocycleTranslationStrategy validation requires per-sample "
                "'field_strength_target'. Set data.expose_field_strength_target on "
                "the mrixfields dataset. Got field_strength_target=None."
            )
        cid = (
            batch_context.get("contrast_id", kwargs.get("contrast_id"))
            if getattr(self, "_contrast_conditioning", True)
            else None
        )
        gen = self.env.generator
        out = gen.render(gen.encode(x0), b_t, cid)
        if getattr(self.config.training, "enforce_output_range", False):
            out = clamp_to_range(out, enable=True, telemetry=False)
        return out

    @torch.no_grad()
    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        field_strength_target: Any = None,
        field_strength: Any = None,
        contrast_id: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Field-conditioned validation: render at the sample's target field.

        Declaring ``field_strength_target`` / ``field_strength`` / ``contrast_id``
        makes the pipeline's gated validation seam forward them (it only passes a batch
        key whose name literally appears in this signature — see
        ``_VALIDATION_FORWARD_FIELDS`` in ``pipelines/train.py``). Reuses the inherited
        GAN metric helpers so no metric math is duplicated (pitfall #18: the metric
        scores the translated image, not a fieldless generator call).

        ``field_strength`` (the SOURCE field) is what makes the anti-facade certificate
        computable at validation: it lets us render ``G(x; s, s)`` alongside
        ``G(x; s, t)`` and report how far apart they are.
        """
        if input_batch is None or target_batch is None:
            return {}
        input_batch = self._to_device(input_batch)
        target_batch = self._to_device(target_batch)
        if field_strength_target is None:
            raise ValueError(
                "FieldCocycleTranslationStrategy validation requires per-sample "
                "'field_strength_target'. Set data.expose_field_strength_target on "
                "the mrixfields dataset."
            )
        b_t = field_strength_target.to(input_batch.device)
        cid = (
            contrast_id.to(input_batch.device)
            if (contrast_id is not None and self._contrast_conditioning)
            else None
        )

        self.generator_model.eval()
        gen = self.env.generator
        hr_fakes = gen.render(gen.encode(input_batch), b_t, cid)
        if getattr(self.config.training, "enforce_output_range", False):
            hr_fakes = clamp_to_range(hr_fakes, enable=True, telemetry=False)

        val_config = getattr(self.config, "validation", None)
        compute_img_metrics = bool(
            (val_config.scoring.enable_image_metrics if val_config else True)
            if val_config is not None
            else True
        )
        metrics: dict[str, float] = {}
        if compute_img_metrics:
            try:
                out_t, tgt_t = self._apply_metric_transforms(hr_fakes, target_batch, val_config)
                metrics.update(self.validation_metrics_computer.compute(out_t, tgt_t))
            except Exception as e:  # pragma: no cover - logged, non-fatal
                if getattr(self, "logging_service", None):
                    self.logging_service.log_warning(f"Validation metrics failed: {e}")

        # Anti-facade certificate at validation (MRIxFields/Cocycle.lean). A field-blind
        # G = Id scores a PERFECT cocycle_residual, so that residual cannot certify the
        # single-model claim. What separates method from facade is whether the render
        # MOVES when only the target field moves:
        #     field_sensitivity = ||G(x; s,t) - G(x; s,s)||
        # whose certified floor is delta - eps_fid - eps_id, with delta = ||y_t - x||
        # (`fieldSensitivity_ge_of_fidelity`). A run reporting field_sensitivity ~ 0
        # against a large field_gap_delta has collapsed, whatever its cocycle_residual
        # says.
        if field_strength is not None:
            b_s = field_strength.to(input_batch.device)
            x_id = gen.render(gen.encode(input_batch), b_s, cid)
            eps_fid = torch.mean(torch.abs(hr_fakes - target_batch))
            eps_id = torch.mean(torch.abs(x_id - input_batch))
            delta = torch.mean(torch.abs(target_batch - input_batch))
            metrics["field_sensitivity"] = float(torch.mean(torch.abs(hr_fakes - x_id)).item())
            metrics["field_gap_delta"] = float(delta.item())
            metrics["field_sensitivity_floor"] = float((delta - eps_fid - eps_id).item())

        disc_scores: dict[str, float] = {}
        if self.env.discriminator is not None:
            disc = cast(nn.Module, self.env.discriminator)
            cond_t = self._field_contrast_cond(target_batch, b_t, cid)
            real = disc(torch.cat([target_batch, cond_t], dim=1))
            fake = disc(torch.cat([hr_fakes, cond_t], dim=1))
            disc_scores = {
                "real_score": real.mean().detach().item(),
                "fake_score": fake.mean().detach().item(),
            }

        validation_results = {**metrics, **disc_scores}
        self._log_validation_images_to_tensorboard(
            predictions=hr_fakes,
            targets=target_batch,
            inputs=input_batch,
            metrics=validation_results,
        )
        return validation_results
