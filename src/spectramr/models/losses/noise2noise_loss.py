r"""Noise2Noise (N2N) — supervised denoising **without** clean targets.

Lehtinen *et al.* (2018) showed that, for zero-mean noise, the
expected-MSE-minimising denoiser

.. math::
    \hat{f} = \arg\min_f\,\mathbb{E}\,\|f(\mathbf{x}+\boldsymbol{\eta}_1)
        -(\mathbf{x}+\boldsymbol{\eta}_2)\|^2

equals the optimal supervised denoiser :math:`\arg\min_f
\mathbb{E}\,\|f(\mathbf{x}+\boldsymbol{\eta})-\mathbf{x}\|^2`. In other
words: regressing one noisy realisation against a *second independent*
noisy realisation is statistically as good as regressing against the
clean signal — provided the noise is independent and zero-mean.

The loss form is just MSE / L1 between ``prediction`` (the network
output on the *first* noisy realisation) and ``target`` (the *second*
noisy realisation). The work of producing two independent realisations
lives in the data pipeline / strategy; this loss is intentionally a
thin wrapper so it appears explicitly in YAML and so the registry can
stamp the right citation onto it.

Reference:
    J. Lehtinen *et al.*, "Noise2Noise: Learning image restoration
    without clean data," *Proc. ICML*, 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


@register_loss(
    name="noise2noise",
    aliases=["Noise2NoiseLoss", "n2n"],
    domain="image",
)
class Noise2NoiseLoss(nn.Module):
    r"""Pair-wise MSE / L1 between two independent noisy realisations.

    Forward expects:

    * ``prediction`` — network output on the *first* noisy view
      :math:`\mathbf{x}+\boldsymbol{\eta}_1`.
    * ``target`` — the *second* noisy view :math:`\mathbf{x}+\boldsymbol{\eta}_2`
      (NOT a clean signal). The strategy / dataloader is responsible
      for producing the two independent realisations.

    Args:
        norm: ``"l2"`` (default, paper) or ``"l1"``.
        require_independent: If True (default) emit a ``RuntimeWarning``
            when ``prediction`` and ``target`` are bit-identical (a
            common bug — caller forgot to swap noise seed).
    """

    def __init__(self, norm: str = "l2", require_independent: bool = True) -> None:
        super().__init__()
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be l1 or l2, got {norm!r}")
        self.norm = norm
        self.require_independent = require_independent
        self._checked_independence = False

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        # torch.equal syncs the host, so look once, not every step.
        check = self.require_independent and not self._checked_independence
        self._checked_independence = True
        if (
            check
            and prediction.shape == target.shape
            and torch.equal(prediction.detach(), target.detach())
        ):
            import warnings

            warnings.warn(
                "Noise2NoiseLoss received bit-identical prediction and "
                "target — strategy probably forgot to draw a second "
                "independent noisy view.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.norm == "l1":
            return F.l1_loss(prediction, target)
        return F.mse_loss(prediction, target)
