r"""Self-supervised denoising losses: SURE and Noise2Self / J-invariant.

Both rely on noise statistics rather than ground truth.

* **SURE** (Stein 1981, Soltanayev–Chun 2018 for deep nets): when the
  noise variance :math:`\sigma^2` is known, the unbiased risk estimator

  .. math::
      \widehat{\mathrm{SURE}}(\hat{\mathbf{x}}) = \|\hat{\mathbf{x}}-\mathbf{y}\|^2
          - N\sigma^2 + 2\sigma^2\,\mathrm{div}_{\mathbf{y}}f_{\boldsymbol{\theta}}(\mathbf{y})

  is an unbiased estimate of the true MSE
  :math:`\mathbb{E}\|\hat{\mathbf{x}}-\mathbf{x}\|^2`. The divergence
  term is approximated by Monte-Carlo Hutchinson:
  :math:`2\sigma^2 \mathbb{E}_{\mathbf{b}}\,\mathbf{b}^\top
  (f(\mathbf{y}+\varepsilon\mathbf{b})-f(\mathbf{y}))/\varepsilon` with
  :math:`\mathbf{b}\sim\mathcal{N}(0,\mathbf{I})`. The loss therefore
  needs both the network and the noisy input — passed via
  ``context["model"]`` and ``context["noisy_input"]``.

* **Noise2Self / J-invariant** (Batson & Royer 2019): mask a random
  partition :math:`J` of pixels in the input, ask the network (which
  must be blind to :math:`J`) to predict only those positions, and
  measure the MSE there. ``context`` must supply the mask
  ``"j_mask"``.

References:
    [29] C. M. Stein, *Ann. Stat.*, 9(6), 1981.
    [27 / 26] J. Batson, L. Royer, "Noise2Self: Blind denoising by
        self-supervision," *ICML*, 2019.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss


@register_loss(name="sure", aliases=["SURELoss", "stein_unbiased_risk"], domain="image")
class SURELoss(nn.Module):
    r"""Stein's Unbiased Risk Estimator.

    Args:
        sigma: Noise standard deviation :math:`\sigma`. Required.
        epsilon: Hutchinson finite-difference step. Default 1e-3 of
            ``sigma``.
        n_samples: Number of Hutchinson samples (Rademacher /
            Gaussian). Default 1 (paper uses 1 for speed).
        rademacher: If True (default) use Rademacher probes
            :math:`\pm 1`; if False use standard Gaussian.
    """

    def __init__(
        self,
        sigma: float,
        epsilon: float | None = None,
        n_samples: int = 1,
        rademacher: bool = True,
    ) -> None:
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        self.sigma = sigma
        self.epsilon = epsilon if epsilon is not None else 1e-3 * sigma
        self.n_samples = n_samples
        self.rademacher = rademacher

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if context is None:
            raise ValueError("SURELoss requires context['model'] and context['noisy_input'].")
        model = context.get("model")
        y = context.get("noisy_input")
        if model is None or y is None:
            raise ValueError(
                "context must supply 'model' (the denoiser) and 'noisy_input' (its actual input)."
            )
        if not callable(model):
            raise TypeError("context['model'] must be a callable / nn.Module.")
        sigma2 = self.sigma * self.sigma
        # Data-fit term
        residual = (prediction - y).pow(2).mean()
        residual_total = residual - sigma2  # subtract Nσ²/N = σ² (since we mean over N)

        # Hutchinson divergence estimator
        #   div_y f(y) ~= (1/eps) * b^T (f(y + eps*b) - f(y))
        # Both forward passes must carry gradient: the MC-SURE training gradient
        # is the finite difference of Jacobian-vector products,
        #   (1/eps) * b^T (grad_theta f(y+eps*b) - grad_theta f(y)).
        # Detaching ``pert`` (no_grad) AND ``prediction`` makes the whole term a
        # constant, so the divergence penalty contributes ZERO gradient and SURE
        # collapses to ||f(y)-y||^2 - sigma^2 -- whose minimiser is the identity.
        div = prediction.new_zeros(())
        for _ in range(self.n_samples):
            if self.rademacher:
                b = torch.randint_like(y, 0, 2).mul_(2).sub_(1).to(y.dtype)
            else:
                b = torch.randn_like(y)
            pert: Any = model(y + self.epsilon * b)
            if isinstance(pert, dict):
                pert = pert.get("output", pert.get("reconstruction"))
                if pert is None:
                    raise KeyError(
                        "SURE model dict output must contain 'output' or 'reconstruction'."
                    )
            div_sample = (b * (pert - prediction)).mean() / self.epsilon
            div = div + div_sample
        div = div / self.n_samples
        out: torch.Tensor = residual_total + 2.0 * sigma2 * div
        return out


@register_loss(
    name="noise2self",
    aliases=["Noise2SelfLoss", "j_invariant", "n2self"],
    domain="image",
)
class Noise2SelfLoss(nn.Module):
    r"""Noise2Self / J-invariant masked-pixel loss.

    Forward expects ``prediction`` to be the network output evaluated on
    a *masked* input where the positions in ``context["j_mask"]`` were
    replaced (by zero or local-mean — strategy's choice). The loss
    measures fit to the *original* values at those positions only,
    against ``target`` (which equals the unmasked noisy input).

    Args:
        norm: ``"l1"`` or ``"l2"`` (default).
    """

    def __init__(self, norm: str = "l2") -> None:
        super().__init__()
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be l1 or l2, got {norm!r}")
        self.norm = norm

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if context is None or "j_mask" not in context:
            raise ValueError("Noise2SelfLoss requires context['j_mask'].")
        real_dtype = prediction.real.dtype if prediction.is_complex() else prediction.dtype
        mask = context["j_mask"].to(real_dtype)
        while mask.dim() < prediction.dim():
            mask = mask.unsqueeze(0)
        diff = prediction - target
        per_element = diff.abs() if self.norm == "l1" else diff.abs().pow(2)
        # Count the masked ELEMENTS, not the mask's own pixels: a shared [H, W]
        # J-mask broadcasts over B and C, and dividing by its pixel count alone
        # scales the loss with the batch size.
        denom = mask.expand_as(per_element).sum().clamp_min(1.0)
        out: torch.Tensor = (mask * per_element).sum() / denom
        return out


@register_loss(
    name="gsure_kspace",
    aliases=["ncchi_gsure", "GSUREKspaceLoss"],
    domain="kspace",
)
class GSUREKspaceLoss(nn.Module):
    r"""Noise-model-aware k-space measurement fidelity (SOTA plan T1 keystone).

    A single data-fidelity term, selectable by a **dataset-driven** noise model,
    consumed by the self-supervised reconstruction arms (Robust-SSDU, Equivariant
    Imaging, Ambient/A-DPS, DDS). The framework is not M4Raw-magnitude-only, so
    the noise model is a config knob, never hardcoded:

    * ``"gaussian_kspace"`` — complex k-space Gaussian negative-log-likelihood
      :math:`\tfrac{1}{2\sigma^2}\|\hat{y} - y\|_2^2`. Correct where the per-coil
      k-space noise is approximately complex-Gaussian (the regime in which SURE /
      Noisier2Noise / Tweedie are unbiased), i.e. complex multi-coil raw k-space
      datasets (calgary_campinas, fastmri, ocmr) and the M4Raw complex pathway.

    * ``"ncchi_magnitude"`` — **moment-corrected** non-central-:math:`\chi`
      magnitude fidelity for the RSS-magnitude regime (e.g. magnitude-only
      low-field). It uses the exact second-moment bias
      :math:`\mathbb{E}[m^2] = \nu^2 + 2L\sigma^2` (Gudbjartsson & Patz Rician
      debiasing [1], generalised to :math:`L` coils) to debias the measured
      magnitude, :math:`\hat{\nu} = \sqrt{\max(m^2 - 2L\sigma^2, 0)}`, and scores
      :math:`\tfrac{1}{2\sigma^2}(|\hat{x}| - \hat{\nu})^2`. This is a standard
      low-SNR correction, NOT the full arbitrary-order Bessel nc-:math:`\chi`
      NLL (``torch.special`` has no ``iv``); it captures the dominant noise-floor
      bias without claiming the full likelihood.

    **Corner-case reduction.** As :math:`\sigma \to 0` (SNR :math:`\to \infty`)
    the noise floor :math:`2L\sigma^2 \to 0`, so :math:`\hat{\nu} \to m` and the
    nc-:math:`\chi` branch reduces to the Gaussian magnitude fidelity
    :math:`\tfrac{1}{2\sigma^2}(|\hat{x}| - m)^2`. The repair therefore provably
    generalises the published Gaussian objective rather than replacing it.

    Args:
        sigma: Noise standard deviation :math:`\sigma > 0` (per-coil k-space std,
            or the magnitude-domain std for the nc-:math:`\chi` branch).
        noise_model: ``"gaussian_kspace"`` (default) or ``"ncchi_magnitude"``.
            An unadvertised value raises (pitfall #9).
        n_coils: Coil count :math:`L \ge 1` (the nc-:math:`\chi` degrees of
            freedom are :math:`2L`). Only used by the nc-:math:`\chi` branch.

    References:
        [1] H. Gudbjartsson and S. Patz, "The Rician distribution of noisy MRI
            data," *Magn. Reson. Med.*, 34(6), 1995.
    """

    _NOISE_MODELS = frozenset({"gaussian_kspace", "ncchi_magnitude"})

    def __init__(
        self,
        sigma: float,
        noise_model: str = "gaussian_kspace",
        n_coils: int = 1,
    ) -> None:
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if noise_model not in self._NOISE_MODELS:
            raise ValueError(
                f"Unknown noise_model={noise_model!r}; "
                f"supported: {sorted(self._NOISE_MODELS)} (pitfall #9: no silent fallback)."
            )
        if n_coils < 1:
            raise ValueError(f"n_coils must be >= 1, got {n_coils}")
        self.sigma = float(sigma)
        self.noise_model = noise_model
        self.n_coils = int(n_coils)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        sigma2 = self.sigma * self.sigma
        if self.noise_model == "gaussian_kspace":
            diff = prediction - target
            sq = (diff.conj() * diff).real if torch.is_complex(diff) else diff.pow(2)
            return 0.5 * sq.mean() / sigma2
        # ncchi_magnitude: debias the measured magnitude by the 2Lσ² noise floor.
        # ``.abs()`` yields the magnitude for both complex and real inputs.
        nu = prediction.abs()
        m = target.abs()
        floor = 2.0 * self.n_coils * sigma2
        nu_hat = torch.sqrt(torch.relu(m * m - floor))
        return 0.5 * (nu - nu_hat).pow(2).mean() / sigma2
