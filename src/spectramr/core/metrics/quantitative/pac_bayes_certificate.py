r"""PAC-Bayes generalisation certificates for unrolled MRI reconstruction.

The McAllester PAC-Bayes bound [McAllester 1999, COLT] gives, with
probability :math:`1-\delta` over a training set of size :math:`m`,

.. math::

    \mathbb{E}_{\theta\sim Q}\,\mathcal{R}(\theta)
    \le \mathbb{E}_{\theta\sim Q}\,\hat L(\theta)
       + \sqrt{\frac{\mathrm{KL}(Q\|P) + \log(2\sqrt m/\delta)}{2m}},

where :math:`P` is a data-independent prior over network weights and
:math:`Q` a data-dependent posterior. The **PAC-Bayes-Bernstein**
refinement [Tolstikhin-Seldin 2013, NeurIPS] adapts to low-variance
regimes and is the tighter bound to use for unrolled reconstruction
networks per Pérez-Ortiz et al. (2021).

For an unrolled reconstructor :math:`R_\theta`, calibration slices of
size :math:`n_{\rm cal}` give the per-volume certificate

.. math::

    \Pr\Big[\ell(x_*, R_\theta(y_*)) \le \hat L_{\rm cal}
       + \sqrt{2\hat V_{\rm cal}\,\frac{\mathrm{KL}+\log(1/\delta)}{n_{\rm cal}}}
       + \frac{c(\mathrm{KL}+\log(1/\delta))}{n_{\rm cal}}\Big] \ge 1-\delta.

Exports:

* :func:`kl_isotropic_gaussian` — :math:`\mathrm{KL}(\mathcal{N}(\mu,\sigma^2 I)\|\mathcal{N}(0,\sigma_0^2 I))`.
* :func:`mcallester_bound` — closed-form McAllester PAC-Bayes upper bound.
* :func:`pac_bayes_bernstein_bound` — Tolstikhin-Seldin Bernstein refinement.
* :class:`PACBayesCertificate` — registered per-batch metric that returns
  the Bernstein bound on per-image MSE in ``[0, loss_bound]``.
"""

from __future__ import annotations

import math

import torch

from spectramr.core.metrics.registry import register_metric

__all__ = [
    "PACBayesCertificate",
    "kl_isotropic_gaussian",
    "mcallester_bound",
    "pac_bayes_bernstein_bound",
]


def kl_isotropic_gaussian(
    posterior_mean: torch.Tensor,
    posterior_log_var: torch.Tensor,
    prior_var: float = 1.0,
) -> torch.Tensor:
    r"""Closed-form
    :math:`\mathrm{KL}(\mathcal{N}(\mu,\sigma^2 I)\,\|\,\mathcal{N}(0,\sigma_0^2 I))`.

    .. math::

        \mathrm{KL} = \tfrac{1}{2}\sum_i\!\Big[\tfrac{\sigma_i^2+\mu_i^2}{\sigma_0^2}
                       - 1 - \log\tfrac{\sigma_i^2}{\sigma_0^2}\Big].

    Args:
        posterior_mean: ``(D,)`` posterior mean :math:`\mu`.
        posterior_log_var: ``(D,)`` posterior log-variance :math:`\log\sigma^2`.
        prior_var: Prior variance :math:`\sigma_0^2`.

    Returns:
        Scalar KL divergence (non-negative).
    """
    if posterior_mean.shape != posterior_log_var.shape:
        raise ValueError("mean and log_var must have identical shape.")
    if prior_var <= 0:
        raise ValueError("prior_var must be positive.")
    log_prior_var = math.log(prior_var)
    var_ratio = torch.exp(posterior_log_var - log_prior_var)
    mean_term = posterior_mean.pow(2) / prior_var
    kl_per_dim = 0.5 * (var_ratio + mean_term - 1.0 - (posterior_log_var - log_prior_var))
    return kl_per_dim.sum()


def mcallester_bound(
    empirical_risk: float,
    kl_divergence: float,
    n_samples: int,
    delta: float = 0.05,
) -> float:
    r"""Closed-form McAllester PAC-Bayes upper bound.

    .. math::

        \mathcal{R}(Q) \le \hat L(Q) + \sqrt{\frac{\mathrm{KL}+\log(2\sqrt m/\delta)}{2m}}.

    Args:
        empirical_risk: :math:`\hat L(Q)` average bounded loss
            (e.g. clipped NMSE in :math:`[0,1]`).
        kl_divergence: :math:`\mathrm{KL}(Q\|P)\ge 0`.
        n_samples: Training-set size :math:`m`.
        delta: Confidence level (default 5%).

    Returns:
        Upper bound on the expected test risk.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1).")
    if kl_divergence < 0:
        raise ValueError("kl_divergence must be non-negative.")
    log_term = math.log(2 * math.sqrt(n_samples) / delta)
    slack = math.sqrt((kl_divergence + log_term) / (2 * n_samples))
    return empirical_risk + slack


def pac_bayes_bernstein_bound(
    empirical_risk: float,
    empirical_variance: float,
    kl_divergence: float,
    n_calibration: int,
    delta: float = 0.05,
    bernstein_constant: float = 2.0,
) -> float:
    r"""Tolstikhin-Seldin PAC-Bayes-Bernstein refinement (tighter at low variance).

    .. math::

        \mathcal{R}(Q) \le \hat L + \sqrt{2\hat V\,\tfrac{\mathrm{KL}+\log(1/\delta)}{n_{\rm cal}}}
                       + \tfrac{c(\mathrm{KL}+\log(1/\delta))}{n_{\rm cal}}.

    Args:
        empirical_risk: :math:`\hat L_{\rm cal}`.
        empirical_variance: :math:`\hat V_{\rm cal}\ge 0`.
        kl_divergence: :math:`\mathrm{KL}\ge 0`.
        n_calibration: Calibration-set size :math:`n_{\rm cal}`.
        delta: Confidence level.
        bernstein_constant: :math:`c` in the higher-order term.

    Returns:
        Upper bound on the test risk.
    """
    if n_calibration <= 0:
        raise ValueError("n_calibration must be positive.")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1).")
    if empirical_variance < 0 or kl_divergence < 0:
        raise ValueError("variance and kl_divergence must be non-negative.")
    if bernstein_constant <= 0:
        raise ValueError("bernstein_constant must be positive.")
    log_term = math.log(1.0 / delta)
    sqrt_term = math.sqrt(2.0 * empirical_variance * (kl_divergence + log_term) / n_calibration)
    higher_order = bernstein_constant * (kl_divergence + log_term) / n_calibration
    return empirical_risk + sqrt_term + higher_order


@register_metric("pac_bayes_certificate", aliases=["pac_bayes", "pb_bound"])
class PACBayesCertificate:
    r"""Per-batch PAC-Bayes-Bernstein risk-bound metric.

    Treats the batch of ``prediction``/``target`` pairs as a calibration
    sample of per-image MSE losses and returns
    :func:`pac_bayes_bernstein_bound` in the loss's own units. The bound is
    valid only for losses in ``[0, loss_bound]``, so the losses are divided by
    ``loss_bound`` before bounding and any loss above it raises. Because the
    batch is also the validation sample used for model selection, the value
    is a diagnostic, not a held-out certificate.

    Args:
        kl_divergence: Pre-computed :math:`\mathrm{KL}(Q\|P)` for the
            trained network. The metric does not re-estimate it.
        delta: Confidence level.
        loss_bound: Upper end :math:`B` of the per-image loss range.
    """

    def __init__(
        self, kl_divergence: float = 0.0, delta: float = 0.05, loss_bound: float = 1.0
    ) -> None:
        if kl_divergence < 0:
            raise ValueError("kl_divergence must be non-negative.")
        if not 0 < delta < 1:
            raise ValueError("delta must lie in (0, 1).")
        if loss_bound <= 0:
            raise ValueError("loss_bound must be positive.")
        self.kl_divergence = kl_divergence
        self.delta = delta
        self.loss_bound = loss_bound

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **_: object,
    ) -> float:
        per_sample = (prediction - target).pow(2).flatten(start_dim=1).mean(dim=-1)
        if per_sample.numel() == 0:
            raise ValueError("prediction must contain at least one sample.")
        worst = per_sample.max().item()
        if worst > self.loss_bound:
            raise ValueError(
                f"pac_bayes_certificate: per-image MSE {worst:.4g} exceeds loss_bound "
                f"{self.loss_bound:.4g}; the Bernstein bound holds only for losses in "
                "[0, loss_bound]. Rescale the intensities or raise loss_bound."
            )
        scaled = per_sample / self.loss_bound
        bound = pac_bayes_bernstein_bound(
            empirical_risk=scaled.mean().item(),
            empirical_variance=scaled.var(unbiased=False).item(),
            kl_divergence=self.kl_divergence,
            n_calibration=int(per_sample.numel()),
            delta=self.delta,
        )
        return bound * self.loss_bound

    @property
    def name(self) -> str:
        return "pac_bayes_certificate"

    @property
    def higher_is_better(self) -> bool:
        return False
