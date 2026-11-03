r"""PrivilegedLearningStrategy — Method III full training harness.

Implements ``synthetic_marker_research_plan.md`` §4.3: a teacher-student
training harness where the teacher receives privileged information (the
synthetic marker channel) at training time and the student deploys
without it.

Composes three pre-existing pieces with one new component (the
gradient-reversal domain discriminator):

1. ``privileged_learning_loss`` from
   :mod:`spectramr.infrastructure.physics.marker_estimators`
2. A student network supplied via ``env.generator`` and a frozen
   **teacher** of the same architecture, loaded from the stage-1 checkpoint
   named by ``training.privileged.teacher_checkpoint``. Stage 1
   (``training.privileged.stage: teacher``) trains that teacher on
   ``input + marker``. ``TrainingEnvironment`` is frozen, so the teacher and
   the domain discriminator are stored on the strategy instance
   (``self._teacher`` / ``self._domain_discriminator``), not on ``env``.
3. A gradient-reversal layer + small domain discriminator MLP for the
   synthetic↔real distribution-shift closing step. Its parameters are
   registered on ``env.opt_g`` so the generator optimizer trains them.

Curriculum (Spec §4.3): the marker count visible to the teacher decays
through training:

.. math::
    N_m(\tau) = \max(N_{\min}, N_{\max}\,e^{-\tau/\tau_0})

with the final 20% of epochs at ``N_m = 0`` to bridge the synthetic-real
distribution gap.

References
----------
* V. Vapnik, R. Izmailov, "Learning using privileged information," JMLR
  16, 2015.
* Y. Ganin et al., "Domain-adversarial training of neural networks,"
  JMLR 17, 2016.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any

import torch
import torch.nn as nn

from spectramr.core.module_utils import unwrap_model
from spectramr.data.batch_types import read_batch_field
from spectramr.infrastructure.checkpointing.restore import weights
from spectramr.infrastructure.distributed.owned_modules import OWNED_MODULE
from spectramr.infrastructure.physics.marker_estimators import (
    privileged_learning_loss,
)
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.models.blocks.domain_adaptation import grad_reverse

logger = logging.getLogger(__name__)

PRIVILEGED_STAGES = ("teacher", "student")
_CURRICULUM_KNOBS = ("curriculum_tau0", "min_keep_fraction", "initial_keep_fraction")

# ──────────────────────────────────────────────────────────────────────
# Domain discriminator, fed through the shared reversal (Ganin & Lempitsky 2015)
# ──────────────────────────────────────────────────────────────────────


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    r"""Identity in the forward pass, ``-λ ∇`` in the backward pass.

    Used to make a feature extractor *adversarial* to a downstream
    classifier without splitting the model into two phases.
    """
    return grad_reverse(x, float(lambda_))


class DomainDiscriminator(nn.Module):
    """Tiny MLP that classifies features as ``synthetic`` (1) vs ``real`` (0).

    The discriminator is paired with :func:`gradient_reversal` on its
    input so that minimising its loss simultaneously *removes* domain
    information from the upstream feature extractor.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        return self.net(gradient_reversal(x, lambda_)).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────
# Strategy
# ──────────────────────────────────────────────────────────────────────


class PrivilegedLearningStrategy(BaseTrainingStrategy):
    r"""Privileged-marker training strategy (Method III).

    Per-step pipeline:

    1. Pull the (input, target, marker) triple from the batch. The marker
       channel is **required** — a markerless run is misconfigured and raises.
    2. Apply the **marker-count curriculum** — drop a fraction of the
       marker's discrete entries based on epoch (decay schedule), so the
       *count* of visible markers decays, not the marker amplitude.
    3. Teacher forward: ``f_T(input + curriculum-masked marker)``.
    4. Student forward: ``f_S(input)``.
    5. Compute :func:`privileged_learning_loss` (recon + distill). The
       FitNet hint term is not wired (no model exposes intermediate
       features), so no ``gamma`` knob is advertised.
    6. Compute the gradient-reversal domain-adversarial loss
       (``self._domain_discriminator``) on the student's prediction.

    Config keys read:

    * ``training.privileged.alpha`` (default 1.0) — recon weight
    * ``training.privileged.beta`` (default 0.5) — distill weight
    * ``training.privileged.delta`` (default 0.1) — domain-adv weight
    * ``training.privileged.curriculum_tau0`` (default 5) — exp time-constant
    * ``training.privileged.min_keep_fraction`` (default 0) — final marker keep-fraction
    * ``training.privileged.initial_keep_fraction`` (default 1) — initial marker keep-fraction
    * ``training.privileged.disc_hidden_dim`` (default 128) — disc MLP width
    * ``training.privileged.stage`` (default ``student``) — ``teacher`` trains
      ``f(input + marker)`` against the target (stage 1); ``student`` distils
    * ``training.privileged.teacher_checkpoint`` — stage-1 checkpoint the student's
      teacher is loaded from; required when ``beta > 0``
    """

    # parallel_compat: The domain discriminator is an owned module, wrapped in __init__ and called
    # through call_owned. The teacher is a frozen copy taken after the env broadcast, never trained.
    parallel_compat = frozenset({"replicated"})

    # Strategy-owned modules (env is frozen, so they live on self).
    _teacher: nn.Module | None = None
    _domain_discriminator: DomainDiscriminator | None = None

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        own = (("_domain_discriminator", OWNED_MODULE),)
        return (*super().owned_module_declarations(config), *own)

    def __init__(self, env=None, **kwargs: Any) -> None:
        # Read before the base __init__, which runs _setup_strategy_specific_components.
        cfg = getattr(getattr(getattr(env, "config", None), "training", None), "privileged", None)
        self._alpha = float(getattr(cfg, "alpha", 1.0)) if cfg else 1.0
        self._beta = float(getattr(cfg, "beta", 0.5)) if cfg else 0.5
        # NOTE: no ``gamma``/hint knob — the FitNet hint term has no feature
        # seam on the model forward, so it would be an identically-zero no-op.
        self._delta = float(getattr(cfg, "delta", 0.1)) if cfg else 0.1
        self._tau0 = float(getattr(cfg, "curriculum_tau0", 5.0)) if cfg else 5.0
        self._n_min = float(getattr(cfg, "min_keep_fraction", 0.0)) if cfg else 0.0
        self._n_max = float(getattr(cfg, "initial_keep_fraction", 1.0)) if cfg else 1.0
        self._disc_hidden_dim = int(getattr(cfg, "disc_hidden_dim", 128)) if cfg else 128
        self._stage = str(getattr(cfg, "stage", "student")) if cfg else "student"
        self._teacher_checkpoint = getattr(cfg, "teacher_checkpoint", None) if cfg else None
        self._declared_knobs = frozenset(getattr(cfg, "model_fields_set", ()) or ())
        super().__init__(env=env, **kwargs)

    def _validate_stage_knobs(self) -> None:
        """Raise on a stage/knob combination that would make a declared knob inert."""
        if self._stage not in PRIVILEGED_STAGES:
            raise ValueError(
                f"training.privileged.stage={self._stage!r} is not one of {PRIVILEGED_STAGES}."
            )
        ckpt = self._teacher_checkpoint
        if self._stage == "teacher":
            inert = [k for k in _CURRICULUM_KNOBS if k in self._declared_knobs]
            if self._beta > 0 or self._delta > 0 or ckpt or inert:
                raise ValueError(
                    "training.privileged.stage=teacher trains the marker-fed teacher on the "
                    "reconstruction term alone; set beta: 0.0 and delta: 0.0 and drop "
                    f"teacher_checkpoint and the curriculum knobs (got beta={self._beta}, "
                    f"delta={self._delta}, teacher_checkpoint={ckpt!r}, declared={inert})."
                )
        elif self._beta > 0 and not ckpt:
            raise ValueError(
                f"training.privileged.beta={self._beta} distils from a teacher, but "
                "training.privileged.teacher_checkpoint is unset. Train the stage-1 arm "
                "(stage: teacher) and point teacher_checkpoint at its checkpoint, or set "
                "beta: 0.0 to drop the distillation term."
            )
        elif self._beta == 0 and ckpt:
            raise ValueError(
                f"training.privileged.teacher_checkpoint={ckpt!r} is set but beta=0.0, so "
                "the teacher would never enter the loss. Drop the checkpoint or set beta > 0."
            )

    def _load_teacher(self, student: nn.Module) -> nn.Module:
        """The frozen teacher: the student architecture with the stage-1 weights loaded exactly."""
        teacher = copy.deepcopy(unwrap_model(student))
        report = weights(self._teacher_checkpoint, "generator", teacher, policy="exact")
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        message = (
            f"[privileged] teacher loaded from {report.source} "
            f"({report.matched}/{report.total} tensors, policy=exact)"
        )
        logger.info(message)
        if getattr(self, "logging_service", None):
            self.logging_service.log_info(message)
        return teacher.to(self.device) if self.device is not None else teacher

    def _setup_strategy_specific_components(self) -> None:
        """Load the stage-1 teacher and build the domain discriminator.

        ``TrainingEnvironment`` is a frozen dataclass and exposes neither a
        ``teacher`` nor a ``domain_discriminator`` field, so both live on the
        strategy instance. The teacher (student mode with ``beta > 0`` only) is
        the student architecture with the ``teacher_checkpoint`` weights; it is
        only ever run under ``torch.no_grad`` in the loss path. The domain
        discriminator is a small MLP whose parameters are added to
        ``env.opt_g`` so the generator optimizer trains them jointly with the
        student (the gradient-reversal layer makes that step adversarial).
        """
        self._validate_stage_knobs()
        student = getattr(self.env, "generator", None)
        if student is None:
            raise ValueError(
                "PrivilegedLearningStrategy requires env.generator (the "
                "student network) to build its teacher copy."
            )

        # No teacher in stage 1 or when beta is 0: it would never enter the loss.
        distils = self._stage == "student" and self._beta > 0
        self._teacher = self._load_teacher(student) if distils else None

        # Domain discriminator operates on the student's prediction;
        # feature_dim is the model's output channels.
        feature_dim = int(getattr(self.config.model, "out_channels", 1))
        disc = DomainDiscriminator(feature_dim=feature_dim, hidden_dim=self._disc_hidden_dim)
        if self.device is not None:
            disc = disc.to(self.device)
        self._domain_discriminator = self.register_owned_module("_domain_discriminator", disc)

        # Register the discriminator's params on the generator optimizer so
        # they are actually trained (env is frozen — opt_g is the SSOT here).
        opt = getattr(self.env, "opt_g", None)
        if opt is not None:
            existing = {p for g in opt.param_groups for p in g["params"]}
            new = [p for p in disc.parameters() if p not in existing]
            if new:
                opt.add_param_group({"params": new, "name": "domain_discriminator"})

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        *,
        batch_data: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Score the stage-1 teacher on ``input + marker``, the input it trains on.

        Best-checkpoint selection must rank the teacher the student will load, not a
        markerless forward it never trained for. The student stage scores markerless.
        """
        if self._stage == "teacher":
            marker = read_batch_field(batch_data, "marker")
            if marker is None:
                raise ValueError(
                    "PrivilegedLearningStrategy stage=teacher validates on input + marker, "
                    "but the validation batch carries no 'marker'; list the synthetic_marker "
                    "transform for the validation loader too."
                )
            input_batch = input_batch + marker.to(input_batch.device)
        return super().validation_step(input_batch, target_batch, **kwargs)

    def _curriculum_marker_fraction(self, epoch: int) -> float:
        r"""Fraction of marker entries kept at epoch ``τ``.

        ``keep_frac(τ) = max(N_min, N_max · exp(-τ / τ_0))`` clamped to
        ``[0, 1]``. Interpreted as the *fraction of discrete marker
        components to retain* (Spec §4.3 ``N_m(τ)``), not an amplitude scale.
        """
        decayed = self._n_max * math.exp(-float(epoch) / self._tau0)
        return float(min(1.0, max(self._n_min, decayed)))

    def _marker_keep_mask(self, marker: torch.Tensor, keep_frac: float) -> torch.Tensor:
        r"""Per-entry keep-mask dropping ``1 - keep_frac`` of marker entries.

        Spec §4.3 defines the curriculum on the *count* of visible markers
        :math:`N_m(\tau)` (``x_m(r) = \sum_{j=1}^{N_m} \rho_j \Phi_j``), so the
        curriculum must drop a fraction of the marker's discrete components,
        not uniformly attenuate the whole tensor's amplitude. We build a
        deterministic broadcastable mask over the flattened spatial/channel
        entries of the (per-sample) marker: the first ``round(keep_frac · N)``
        entries by flat index are kept, the rest zeroed. Determinism (no ad-hoc
        seeding) keeps the curriculum reproducible across runs.

        Returns a mask of the same shape as ``marker`` (1.0 = keep, 0.0 = drop).
        """
        # Flatten everything except the batch dim → one "entry budget" per
        # sample. Markers are sparse discrete components, so masking flat
        # entries reduces the visible marker *count*.
        flat = marker.reshape(marker.shape[0], -1)
        n_entries = flat.shape[1]
        keep_count = int(round(float(keep_frac) * n_entries))
        keep_count = max(0, min(n_entries, keep_count))
        mask_flat = torch.zeros_like(flat)
        if keep_count > 0:
            mask_flat[:, :keep_count] = 1.0
        return mask_flat.reshape(marker.shape)

    @staticmethod
    def _resolve_domain_label(raw_label: Any, input_batch: torch.Tensor) -> torch.Tensor | None:
        """Per-sample ``(B,)`` float label, or ``None`` when the batch carries none."""
        if raw_label is None:
            return None
        label = torch.as_tensor(raw_label, device=input_batch.device, dtype=torch.float32).reshape(-1)
        if label.numel() == 1:
            # A scalar is a legitimate single-domain declaration, broadcast explicitly.
            return label.expand(input_batch.shape[0])
        if label.numel() != input_batch.shape[0]:
            raise ValueError(
                "PrivilegedLearningStrategy: batch['domain_label'] carries "
                f"{label.numel()} entries for a batch of {input_batch.shape[0]}. "
                "The label is per-SAMPLE; a length mismatch would broadcast and "
                "mislabel the batch."
            )
        return label

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        student = getattr(self.env, "generator", None)
        teacher = self._teacher
        if teacher is None and self._stage == "student" and self._beta > 0:
            raise ValueError(
                "PrivilegedLearningStrategy.teacher was not loaded; "
                "_setup_strategy_specific_components must run before the "
                "first training step."
            )

        # A dict or a TrainingBatch; read_batch_field reaches TrainingBatch metadata too.
        batch = self._resolve_legacy_batch(input_batch, kwargs)

        # Marker channel — the dataset MUST attach the synthetic privileged
        # markers under batch['marker'] (shape matching input). A run with no
        # markers is misconfigured: the teacher would then see exactly the
        # student's input, collapsing the teacher-student privileged gap to
        # zero (silent no-op). Raise rather than silently degrade (pitfall #9).
        marker = read_batch_field(batch, "marker")
        if marker is None:
            raise ValueError(
                "PrivilegedLearningStrategy requires batch['marker'] (the "
                "synthetic privileged channel); a privileged-learning run with "
                "no markers is misconfigured — the teacher would receive no "
                "privileged information and collapse to the student."
            )
        marker = marker.to(input_batch.device)

        if self._stage == "teacher":
            # Stage 1: the teacher sees every marker, constant, and fits the target.
            pred = student(input_batch + marker)
            recon = (pred - target_batch).pow(2).mean()
            return {
                "g_total_loss": self._alpha * recon,
                "g_priv_recon": recon.detach(),
                "g_priv_marker_frac": torch.ones((), device=input_batch.device),
            }

        # Curriculum: drop a fraction of the marker's discrete entries at the
        # teacher's input. ``keep_mask`` zeroes ``1 - keep_frac`` of the marker
        # components so the *count* of visible markers decays with epoch
        # (Spec §4.3 N_m(τ)), rather than uniformly attenuating amplitude.
        keep_frac = self._curriculum_marker_fraction(epoch)
        keep_mask = self._marker_keep_mask(marker, keep_frac)
        teacher_input = input_batch + marker * keep_mask

        # Student (markerless deployment)
        student_pred = student(input_batch)

        # Teacher (no grad, frozen). With beta 0 there is none and the distill term is zero.
        if teacher is None:
            teacher_pred = student_pred.detach()
        else:
            with torch.no_grad():
                teacher_pred = teacher(teacher_input)

        # Per-sample domain label: 1.0 = synthetic (source, supervised), 0.0 = real
        # (target, unlabelled). ``DomainLabelledBalancer`` (data.multi_domain with
        # balancing: domain_labelled) emits it. A missing label is never fabricated:
        # a constant one trains the discriminator to separate one class from itself.
        domain_label = self._resolve_domain_label(
            read_batch_field(batch, "domain_label"), input_batch
        )

        # Privileged-learning loss. The FitNet-style hint term
        # (``gamma`` / ``student_features`` / ``teacher_features``) is NOT
        # wired here: no generator in ``models/`` exposes intermediate
        # features through its forward contract, so a hint weight would be an
        # identically-zero no-op (pitfall #15). It is therefore left at the
        # loss default and not advertised as a strategy knob — only the
        # recon (alpha) and distillation (beta) terms are active. With domain
        # labels, recon and distill are restricted to the synthetic samples
        # (spec §4.3 defines L_S on the twin's x_a only).
        out = privileged_learning_loss(
            student_pred=student_pred,
            teacher_pred=teacher_pred,
            target=target_batch,
            alpha=self._alpha,
            beta=self._beta,
            sample_weight=domain_label,
        )

        # Domain-adversarial loss (gradient-reversal vs. the strategy-owned
        # discriminator).
        disc = self._domain_discriminator
        if disc is not None and self._delta > 0:
            if domain_label is None:
                raise ValueError(
                    "PrivilegedLearningStrategy: training.privileged.delta="
                    f"{self._delta} enables the domain-adversarial term, but "
                    "batch['domain_label'] is absent. Declare a two-domain "
                    "data.multi_domain block with balancing: domain_labelled "
                    "(DomainLabelledBalancer stamps the label per sample), or set "
                    "training.privileged.delta: 0.0 to drop the term explicitly. "
                    "A constant label trains the gradient-reversal discriminator "
                    "to separate one class from itself."
                )
            # Discriminate on the student's prediction. (A hidden-feature seam
            # would need a ``forward(..., return_features=True)`` contract that
            # no model in ``models/`` implements; the prior ``getattr(...,
            # "_features", ...)`` always fell through to the prediction anyway,
            # so we discriminate on it explicitly rather than mask the dead
            # branch behind an attribute that never exists.)
            disc_logits = self.call_owned("_domain_discriminator", student_pred, lambda_=self._delta)
            disc_loss = nn.functional.binary_cross_entropy_with_logits(
                disc_logits,
                domain_label,
            )
            out["g_priv_domain"] = disc_loss.detach()
            out["g_total_loss"] = out["g_total_loss"] + self._delta * disc_loss

        out["g_priv_marker_frac"] = torch.tensor(keep_frac, device=input_batch.device)
        return out


__all__ = [
    "DomainDiscriminator",
    "PrivilegedLearningStrategy",
    "gradient_reversal",
]
