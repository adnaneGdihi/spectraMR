r"""Kernelised-Stein-Discrepancy federated-consistency loss.

The ``stein_federated`` strategy advertises kernelised-Stein-discrepancy (KSD)
mathematics for federated posterior aggregation, but historically routed to a
generic reconstruction strategy with vanilla L1/L2 losses — a *façade*. This
module closes that gap with a loss that actually penalises the KSD between the
aggregated prediction's sample distribution and the target distribution.

Kernelised Stein discrepancy [Liu, Lee & Jordan, ICML 2016; Chwialkowski et al.
2016; Gorham & Mackey, NeurIPS 2017] measures how far an empirical sample set
``{x_i}`` is from a target distribution ``p`` using the Stein operator and a
(here inverse-multi-quadric) kernel, **without** needing ``p``'s normalising
constant — only its score :math:`s_p(x)=\nabla_x\log p(x)`:

.. math::

    \mathrm{KSD}^2(P\,\|\,Q_n)=\frac{1}{n(n-1)}\sum_{i\neq j} k_p(x_i,x_j),
    \quad k_p(x,y)=s_p(x)^\top k(x,y)\,s_p(y)+\dots

Federated-aggregation reading: the target site samples induce an (unnormalised)
target distribution; we estimate its score from the empirical target batch under
a Gaussian model :math:`p=\mathcal N(\mu_t,\Sigma_t)` so that
:math:`s_p(x)=-\Sigma_t^{-1}(x-\mu_t)`, then evaluate the KSD U-statistic on the
*prediction* samples. When the aggregated prediction matches the target
distribution the discrepancy vanishes; when it drifts (mode collapse, biased
aggregation, DP-noise corruption) the discrepancy grows — exactly the federated
consistency the strategy claims to enforce.

The KSD U-statistic itself is reused verbatim from
:func:`spectramr.core.metrics.stein.kernelised_stein_discrepancy`; this module only
supplies the target score function and the flattening / batching glue.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn

from spectramr.core.metrics.stein import kernelised_stein_discrepancy
from spectramr.models.losses.registry import register_loss


@register_loss(
    name="stein_discrepancy",
    aliases=["ksd", "kernelized_stein_discrepancy", "stein_federated_consistency"],
    domain="image",
)
class SteinDiscrepancyLoss(nn.Module):
    r"""Penalise the KSD between prediction and target sample distributions.

    Each spatial sample (one item of the batch, flattened to a feature vector)
    is treated as a draw from the underlying distribution. The target batch
    defines an empirical Gaussian whose score drives the Stein operator; the
    prediction batch is the sample set whose fit is scored by the KSD
    U-statistic. Differentiable end-to-end through the prediction.

    Args:
        target_score_reg: Diagonal shrinkage :math:`\gamma` added to the target
            covariance before inversion (numerical stability + handles the
            ``identical-samples`` degenerate covariance). Default 1e-2.
        bandwidth: Optional fixed IMQ kernel bandwidth; ``None`` uses the
            median heuristic inside the KSD estimator.
        batch_chunks: Number of contiguous chunks the batch is split into,
            each scored independently before reduction. Lets a large batch act
            as several federated mini-cohorts. Default 1.
        reduction: ``"mean"`` or ``"sum"`` over the per-chunk KSD² values.

    Raises:
        ValueError: on bad ``reduction``, non-positive ``batch_chunks``,
            shape mismatch, or fewer than 2 samples per chunk.
    """

    def __init__(
        self,
        target_score_reg: float = 1e-2,
        bandwidth: float | None = None,
        batch_chunks: int = 1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if target_score_reg <= 0:
            raise ValueError("target_score_reg must be positive")
        if bandwidth is not None and bandwidth <= 0:
            raise ValueError("bandwidth must be positive when provided")
        if batch_chunks < 1:
            raise ValueError("batch_chunks must be >= 1")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be mean/sum, got {reduction!r}")
        self.target_score_reg = float(target_score_reg)
        self.bandwidth = bandwidth
        self.batch_chunks = int(batch_chunks)
        self.reduction = reduction

    def _build_target_score_fn(
        self, target_flat: torch.Tensor
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        r"""Empirical-Gaussian score :math:`s_p(x)=-\Sigma_t^{-1}(x-\mu_t)`.

        :math:`\Sigma_t = U^\top U + \gamma I` with :math:`U` the ``(n, d)``
        Bessel-scaled centred targets; the shrinkage keeps it invertible for
        identical/degenerate samples. It is inverted through the Woodbury
        identity :math:`\Sigma_t^{-1} = \gamma^{-1}(I - U^\top(\gamma I_n +
        UU^\top)^{-1}U)`, an ``(n, n)`` solve: the dense ``(d, d)`` inverse
        needs ~42 GB per chunk at ``d = 320^2``.

        Args:
            target_flat: ``(n, d)`` flattened target samples.

        Returns:
            Callable ``score_fn(x: (m, d)) -> (m, d)``.
        """
        n = target_flat.shape[0]
        mu = target_flat.mean(dim=0, keepdim=True)  # (1, d)
        u = (target_flat - mu) / math.sqrt(max(n - 1, 1))  # (n, d)
        gamma = self.target_score_reg
        eye_n = torch.eye(n, device=target_flat.device, dtype=target_flat.dtype)
        small = u @ u.t() + gamma * eye_n  # (n, n)

        def score_fn(x: torch.Tensor) -> torch.Tensor:
            xc = x - mu  # (m, d)
            coeff: torch.Tensor = torch.linalg.solve(small, (xc @ u.t()).t()).t()  # (m, n)
            return -(xc - coeff @ u) / gamma

        return score_fn

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must share shape; got "
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}."
            )
        if prediction.ndim < 2:
            raise ValueError(
                "expected a batched tensor (n, ...) with n samples on dim 0; "
                f"got shape {tuple(prediction.shape)}."
            )

        n = prediction.shape[0]
        pred_flat = prediction.reshape(n, -1)
        target_flat = target.reshape(n, -1)

        pred_chunks = torch.chunk(pred_flat, self.batch_chunks, dim=0)
        target_chunks = torch.chunk(target_flat, self.batch_chunks, dim=0)

        total = prediction.new_zeros(())
        n_valid = 0
        for pred_c, target_c in zip(pred_chunks, target_chunks, strict=True):
            if pred_c.shape[0] < 2:
                raise ValueError(
                    "KSD needs >=2 samples per chunk; got "
                    f"{pred_c.shape[0]} (n={n}, batch_chunks={self.batch_chunks})."
                )
            score_fn = self._build_target_score_fn(target_c.detach())
            # Authoritative numeric reference: the exact U-statistic estimator
            # from the metric package (returns a detached float via .item()).
            # We invoke it both to reuse the validated KSD machinery and to
            # surface its NaN/inf guards before the differentiable twin runs.
            ref_ksd2, _ = kernelised_stein_discrepancy(
                score_fn, pred_c.detach(), bandwidth=self.bandwidth
            )
            if ref_ksd2 != ref_ksd2:  # NaN guard (no .item() on the loss path)
                raise ValueError("KSD estimator returned NaN; check inputs/score.")
            # The estimator is non-differentiable; recompute the identical
            # U-statistic in-graph so the loss flows gradients to the prediction.
            total = total + self._differentiable_ksd2(pred_c, score_fn)
            n_valid += 1

        if self.reduction == "mean":
            return total / max(n_valid, 1)
        return total

    def _differentiable_ksd2(
        self, samples: torch.Tensor, score_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        r"""In-graph KSD² mirroring the metric's IMQ Stein-kernel U-statistic.

        Identical math to
        :func:`spectramr.core.metrics.stein.kernelised_stein_discrepancy` (IMQ
        kernel, median bandwidth, off-diagonal U-statistic) but kept on the
        autograd graph. As in the metric, the unbiased U-statistic is returned
        **un-clamped** — it can dip slightly negative on a near-perfect fit — so
        gradients flow at every operating point. The float-valued metric call in
        :meth:`forward` is the authoritative numeric reference; this is its
        differentiable twin.
        """
        n, d = samples.shape
        s = score_fn(samples)
        diff = samples.unsqueeze(1) - samples.unsqueeze(0)  # (n, n, d)
        sq = (diff * diff).sum(dim=-1)  # (n, n)
        if self.bandwidth is not None:
            c = float(self.bandwidth)
        else:
            iu = torch.triu_indices(n, n, offset=1, device=samples.device)
            dists = sq[iu[0], iu[1]].sqrt()
            c = dists.median().detach().item() if dists.numel() > 0 else 1.0
        beta = -0.5
        c2 = c * c
        base = (c2 + sq).pow(beta)
        base_dx = 2.0 * beta * (c2 + sq).pow(beta - 1)
        grad_x_k = base_dx.unsqueeze(-1) * diff  # ∇_x k, (n, n, d)
        grad_y_k = -grad_x_k
        trace_term = -2.0 * beta * d * (c2 + sq).pow(beta - 1) - 4.0 * beta * (beta - 1) * sq * (
            c2 + sq
        ).pow(beta - 2)
        term_ss = (s.unsqueeze(1) * s.unsqueeze(0)).sum(dim=-1) * base
        term_sgy = (s.unsqueeze(1) * grad_y_k).sum(dim=-1)
        term_sxg = (grad_x_k * s.unsqueeze(0)).sum(dim=-1)
        stein_kernel = term_ss + term_sgy + term_sxg + trace_term
        off_diag = stein_kernel - torch.diag(torch.diagonal(stein_kernel))
        ksd2 = off_diag.sum() / (n * (n - 1))
        return ksd2
