r"""Knowledge distillation + EMA self-distillation losses.

* **Knowledge Distillation** (Hinton, Vinyals, Dean 2015): the student
  matches the *softened* output distribution of a frozen teacher,

  .. math::
      \mathcal{L}_{\text{KD}} = T^2\,\mathrm{KL}\!\left(
          \mathrm{softmax}(\mathbf{z}_T/T)\,\|\,\mathrm{softmax}(\mathbf{z}_S/T)\right).

  For dense regression (e.g. recon images) we additionally support a
  feature-MSE mode where the student matches teacher features rather
  than logits.

* **EMA self-distillation** (BYOL, DINO): the student matches the
  output of an *exponential-moving-average* copy of itself on a
  different augmentation / view of the same input. The EMA target is
  expected to be supplied via context (the strategy maintains it).

References:
    [57] G. Hinton, O. Vinyals, J. Dean, *arXiv:1503.02531*, 2015.
    [54] J.-B. Grill *et al.*, "Bootstrap your own latent," NeurIPS, 2020.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


@register_loss(
    name="knowledge_distillation",
    aliases=["KDLoss", "kd", "hinton_distillation"],
    domain="agnostic",
)
class KnowledgeDistillationLoss(nn.Module):
    r"""Hinton-style soft-target knowledge distillation.

    Forward expects:

    * ``prediction``: student output (logits in classification mode,
      features in feature-MSE mode).
    * ``target``: ignored — the teacher signal must come via context.
    * ``context["teacher_output"]``: same shape as ``prediction``;
      teacher-side logits or features (call ``teacher.eval()`` and
      ``no_grad`` upstream).

    Args:
        mode: ``"kl"`` (logits with softmax temperature) or
            ``"feature_mse"`` (regression-style MSE between features).
        temperature: Softmax temperature :math:`T`. Only used in ``"kl"``.
            Default 4.0.
    """

    def __init__(
        self,
        mode: str = "kl",
        temperature: float = 4.0,
    ) -> None:
        super().__init__()
        if mode not in ("kl", "feature_mse"):
            raise ValueError(f"mode must be kl or feature_mse, got {mode!r}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.mode = mode
        self.temperature = temperature

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if context is None or "teacher_output" not in context:
            raise ValueError("KnowledgeDistillationLoss requires context['teacher_output'].")
        teacher = context["teacher_output"]
        if teacher.shape != prediction.shape:
            raise ValueError(
                f"teacher shape {tuple(teacher.shape)} != student shape {tuple(prediction.shape)}"
            )
        if self.mode == "feature_mse":
            return F.mse_loss(prediction, teacher.detach())
        T = self.temperature
        log_p_student = F.log_softmax(prediction / T, dim=1)
        p_teacher = F.softmax(teacher / T, dim=1).detach()
        return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)


@register_loss(
    name="ema_self_distillation",
    aliases=["EMADistillationLoss", "byol_loss", "self_distill_ema"],
    domain="agnostic",
)
class EMASelfDistillationLoss(nn.Module):
    r"""BYOL/DINO-style EMA-target self-distillation.

    Forward expects:

    * ``prediction``: online-network output on the first augmentation.
    * ``target``: ignored.
    * ``context["ema_output"]``: EMA-target network output on a
      *second* augmentation of the same input. Strategies maintain the
      EMA copy and call it under ``no_grad``.

    The loss is a normalised L2 distance equivalent to BYOL's
    cosine-similarity formulation up to a constant:

    .. math::
        \mathcal{L} = 2 - 2\,\langle \tilde{p}, \tilde{t}\rangle,
        \quad \tilde{p}=p/\|p\|,\ \tilde{t}=t/\|t\|.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if context is None or "ema_output" not in context:
            raise ValueError("EMASelfDistillationLoss requires context['ema_output'].")
        ema = context["ema_output"].detach()
        if ema.shape != prediction.shape:
            raise ValueError(
                f"ema shape {tuple(ema.shape)} != student shape {tuple(prediction.shape)}"
            )
        # Per-example flatten then cosine-style distance
        p = F.normalize(prediction.flatten(1), dim=1)
        t = F.normalize(ema.flatten(1), dim=1)
        return (2.0 - 2.0 * (p * t).sum(dim=1)).mean()
