r"""PrivilegedLearningStrategy — Method III full training harness.

Implements ``synthetic_marker_research_plan.md`` §4.3: a teacher-student
training harness where the teacher receives privileged information (the
synthetic marker channel) at training time and the student deploys
without it.

Composes three pre-existing pieces with one new component (the
gradient-reversal domain discriminator):

1. ``privileged_learning_loss`` from
   :mod:`spectramr.infrastructure.physics.marker_estimators`
2. A student network supplied via ``env.generator`` and a marker-
   supervised **teacher** built by the strategy itself (a frozen copy of
   the student architecture). ``TrainingEnvironment`` is frozen, so the
   teacher and the domain discriminator are stored on the strategy
   instance (``self._teacher`` / ``self._domain_discriminator``), not on
   ``env``.
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
import math
from typing import Any

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.marker_estimators import (
    privileged_learning_loss,
)
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.models.blocks.domain_adaptation import grad_reverse

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
    * ``training.privileged.n_min`` (default 0) — final marker keep-fraction
    * ``training.privileged.n_max`` (default 1) — initial marker keep-fraction
    * ``training.privileged.disc_hidden_dim`` (default 128) — disc MLP width
    """

    def __init__(self, env=None, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        # Resolve curriculum parameters from config (with defaults)
        cfg = getattr(self.config.training, "privileged", None)
        self._alpha = float(getattr(cfg, "alpha", 1.0)) if cfg else 1.0
        self._beta = float(getattr(cfg, "beta", 0.5)) if cfg else 0.5
        # NOTE: no ``gamma``/hint knob — the FitNet hint term has no feature
        # seam on the model forward, so it would be an identically-zero no-op.
        self._delta = float(getattr(cfg, "delta", 0.1)) if cfg else 0.1
        self._tau0 = float(getattr(cfg, "curriculum_tau0", 5.0)) if cfg else 5.0
        self._n_min = float(getattr(cfg, "n_min", 0.0)) if cfg else 0.0
        self._n_max = float(getattr(cfg, "n_max", 1.0)) if cfg else 1.0
        self._disc_hidden_dim = int(getattr(cfg, "disc_hidden_dim", 128)) if cfg else 128
        # Strategy-owned modules (env is frozen → store on self, not env).
        self._teacher: nn.Module | None = None
        self._domain_discriminator: DomainDiscriminator | None = None

    def _setup_strategy_specific_components(self) -> None:
        """Build the marker-supervised teacher and the domain discriminator.

        ``TrainingEnvironment`` is a frozen dataclass and exposes neither a
        ``teacher`` nor a ``domain_discriminator`` field, so both live on the
        strategy instance. The teacher is a frozen copy of the student
        generator (identical architecture → identical input contract); it is
        only ever run under ``torch.no_grad`` in the loss path. The domain
        discriminator is a small MLP whose parameters are added to
        ``env.opt_g`` so the generator optimizer trains them jointly with the
        student (the gradient-reversal layer makes that step adversarial).
        """
        student = getattr(self.env, "generator", None)
        if student is None:
            raise ValueError(
                "PrivilegedLearningStrategy requires env.generator (the "
                "student network) to build its teacher copy."
            )

        # Teacher: frozen deep-copy of the student architecture. Marker
        # supervision flows through its privileged input at train time; it
        # is never optimized by opt_g (the loss path runs it under no_grad).
        teacher = copy.deepcopy(student)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        if self.device is not None:
            teacher = teacher.to(self.device)
        self._teacher = teacher

        # Domain discriminator operates on the student's prediction;
        # feature_dim is the model's output channels.
        feature_dim = int(getattr(self.config.model, "out_channels", 1))
        disc = DomainDiscriminator(feature_dim=feature_dim, hidden_dim=self._disc_hidden_dim)
        if self.device is not None:
            disc = disc.to(self.device)
        self._domain_discriminator = disc

        # Register the discriminator's params on the generator optimizer so
        # they are actually trained (env is frozen — opt_g is the SSOT here).
        opt = getattr(self.env, "opt_g", None)
        if opt is not None:
            existing = {p for g in opt.param_groups for p in g["params"]}
            new = [p for p in disc.parameters() if p not in existing]
            if new:
                opt.add_param_group({"params": new, "name": "domain_discriminator"})

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

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        student = getattr(self.env, "generator", None)
        teacher = self._teacher
        if teacher is None:
            raise ValueError(
                "PrivilegedLearningStrategy.teacher was not built; "
                "_setup_strategy_specific_components must run before the "
                "first training step (it constructs the teacher copy)."
            )

        # The dataloader collate may travel as a dict in ``batch`` (legacy
        # callers) or as the orchestrator's ``input_batch`` kwarg. Resolve
        # it before reading the marker channel; fall back to the positional
        # tensor for the actual student/teacher input.
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if not isinstance(batch, dict):
            batch = {}

        # Marker channel — the dataset MUST attach the synthetic privileged
        # markers under batch['marker'] (shape matching input). A run with no
        # markers is misconfigured: the teacher would then see exactly the
        # student's input, collapsing the teacher-student privileged gap to
        # zero (silent no-op). Raise rather than silently degrade (pitfall #9).
        marker = batch.get("marker")
        if marker is None:
            raise ValueError(
                "PrivilegedLearningStrategy requires batch['marker'] (the "
                "synthetic privileged channel); a privileged-learning run with "
                "no markers is misconfigured — the teacher would receive no "
                "privileged information and collapse to the student."
            )
        marker = marker.to(input_batch.device)

        # Curriculum: drop a fraction of the marker's discrete entries at the
        # teacher's input. ``keep_mask`` zeroes ``1 - keep_frac`` of the marker
        # components so the *count* of visible markers decays with epoch
        # (Spec §4.3 N_m(τ)), rather than uniformly attenuating amplitude.
        keep_frac = self._curriculum_marker_fraction(epoch)
        keep_mask = self._marker_keep_mask(marker, keep_frac)
        teacher_input = input_batch + marker * keep_mask

        # Teacher (no grad — privileged path is frozen at this step)
        with torch.no_grad():
            teacher_pred = teacher(teacher_input)

        # Student (markerless deployment)
        student_pred = student(input_batch)

        # Privileged-learning loss. The FitNet-style hint term
        # (``gamma`` / ``student_features`` / ``teacher_features``) is NOT
        # wired here: no generator in ``models/`` exposes intermediate
        # features through its forward contract, so a hint weight would be an
        # identically-zero no-op (pitfall #15). It is therefore left at the
        # loss default and not advertised as a strategy knob — only the
        # recon (alpha) and distillation (beta) terms are active.
        out = privileged_learning_loss(
            student_pred=student_pred,
            teacher_pred=teacher_pred,
            target=target_batch,
            alpha=self._alpha,
            beta=self._beta,
        )

        # Domain-adversarial loss (gradient-reversal vs. the strategy-owned
        # discriminator). ``batch`` is already a dict (possibly empty).
        disc = self._domain_discriminator
        if disc is not None and self._delta > 0:
            # A6: no loader under ``data/`` emits ``domain_label``, and the
            # default here was ``1`` -- so EVERY sample in EVERY batch carried
            # the same label and the gradient-reversal discriminator was trained
            # to separate one class from itself. The loss stayed finite and
            # plotted a plausible curve while the domain term it exists to
            # supply carried no domain information at all.
            #
            # That is worse than the 33 orphan keys whose absence merely skips a
            # term: the mechanism is present, running, and wrong. See
            # TODO/inprogress/a6_orphan_batch_key_triage_2026_08_06.md, which
            # classes this as the census's single "substitutes" case. Same
            # disposition as ``batch['marker']`` above -- raise rather than
            # fabricate (pitfall #9; #16, an advertised mechanism that cannot
            # work as advertised).
            raw_label = batch.get("domain_label")
            if raw_label is None:
                raise ValueError(
                    "PrivilegedLearningStrategy: training.privileged.delta="
                    f"{self._delta} enables the domain-adversarial term, but "
                    "batch['domain_label'] is absent and no dataset under "
                    "spectramr.data emits it. A constant label trains the "
                    "gradient-reversal discriminator to separate one class "
                    "from itself -- a confounded objective, not a weaker one. "
                    "Supply per-sample domain labels, or set "
                    "training.privileged.delta: 0.0 to drop the term "
                    "explicitly."
                )
            domain_label = torch.as_tensor(
                raw_label, device=input_batch.device, dtype=torch.float32
            ).reshape(-1)
            if domain_label.numel() == 1:
                # A scalar is a legitimate single-domain declaration -- but an
                # EXPLICIT one, which the fabricated default never was.
                domain_label = domain_label.expand(input_batch.shape[0])
            elif domain_label.numel() != input_batch.shape[0]:
                raise ValueError(
                    "PrivilegedLearningStrategy: batch['domain_label'] carries "
                    f"{domain_label.numel()} entries for a batch of "
                    f"{input_batch.shape[0]}. The label is per-SAMPLE; a "
                    "length mismatch would broadcast and mislabel the batch."
                )
            # Discriminate on the student's prediction. (A hidden-feature seam
            # would need a ``forward(..., return_features=True)`` contract that
            # no model in ``models/`` implements; the prior ``getattr(...,
            # "_features", ...)`` always fell through to the prediction anyway,
            # so we discriminate on it explicitly rather than mask the dead
            # branch behind an attribute that never exists.)
            disc_logits = disc(student_pred, lambda_=self._delta)
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
