"""Attack-robustness and out-of-distribution reconstruction-stability metrics.

PR-6 of ``TODO/backlog_frontier_gap_audit_2026_05.md``. The framework
already carries ``robust_mri_psnr`` (an ROI-based fidelity score) and an
``adversarial_robustness`` *training* strategy, but it has had **no
perturbation-induced robustness METRIC**. MRI reconstruction networks are
known to be unstable under small worst-case input perturbations
(Antun et al. [1]) and to degrade sharply off their training
acceleration. Both failure modes are invisible to PSNR/SSIM on the
nominal in-distribution input. This module exposes them as scalar
metrics.

Two metrics are registered:

``adversarial_psnr_drop`` (``higher_is_better=False``)
    Runs a small bounded projected-gradient-descent (PGD) attack with an
    L-infinity budget on the *input* that produced ``prediction`` to
    maximize the reconstruction MSE, then reports
    ``PSNR(clean) - PSNR(adversarial)`` — the PSNR the network loses to a
    worst-case epsilon perturbation. A larger drop is worse.

    This metric needs the forward model. It is called with the trained
    ``model`` (any callable mapping ``model_input -> reconstruction``) and
    the ``model_input`` (the tensor that produced ``prediction``) via
    ``kwargs``. **Without ``model`` and ``model_input`` it is a documented
    no-op returning ``0.0``** — there is no way to perturb an input we do
    not have, and silently fabricating one would violate the no-silent-
    fallback rule. Reporting layers that lack a live forward model simply
    see ``0.0`` (no instability measured) rather than a fabricated value.

``ood_acceleration_shift`` (``higher_is_better=False``)
    Given the nominal ``prediction``/``target`` AND a held-out-
    acceleration pair (``prediction_ood``/``target_ood`` directly, or an
    ``ood_fn`` callable that returns the pair) via ``kwargs``, reports
    ``PSNR(in-distribution) - PSNR(OOD)`` — the PSNR the network loses
    when evaluated at an acceleration outside its training distribution.
    **Without the OOD pair it is a documented no-op returning ``0.0``.**

Both metrics are pure-CPU torch (no ``.cuda``, no GPU-only ops) so they
run in the synthetic Tier-2 audit probe and on CPU-only CI.

References
----------
[1] V. Antun, F. Renna, C. Poon, B. Adcock, and A. C. Hansen, "On
    instabilities of deep learning in image reconstruction and the
    potential costs of AI," *PNAS*, vol. 117, no. 48, 2020.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from typing import Any

import torch

from spectramr.core.metrics.context import MetricContext, resolve_context
from spectramr.core.metrics.registry import register_metric


def _psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Peak signal-to-noise ratio ``20 log10(max / sqrt(mse))`` in dB.

    The peak is taken as the target's dynamic range (``max - min``),
    falling back to ``max(|target|)`` for zero-range inputs. A perfect
    match returns a large finite cap (``120.0`` dB) rather than ``+inf``
    so downstream subtraction stays finite.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(prediction.shape)} vs target {tuple(target.shape)}"
        )
    pred = prediction.detach().to(torch.float32)
    tgt = target.detach().to(torch.float32)
    mse = torch.mean((pred - tgt) ** 2).item()
    rng = (tgt.max() - tgt.min()).item()
    peak = rng if rng > 0.0 else max(abs(tgt.max().item()), 1e-12)
    if mse <= 0.0:
        return 120.0
    return float(20.0 * math.log10(peak / math.sqrt(mse)))


def _pgd_linf_attack(
    model: Callable[[torch.Tensor], torch.Tensor],
    model_input: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
    steps: int,
) -> torch.Tensor:
    """Bounded L-infinity PGD attack maximizing reconstruction MSE.

    Returns the adversarial reconstruction ``model(x + delta)`` where
    ``delta`` is found by ``steps`` gradient-ascent steps on the output
    MSE, projected to ``||delta||_inf <= epsilon`` each step. Self-
    contained and CPU-safe (no GPU-only ops).
    """
    x0 = model_input.detach().to(torch.float32)
    step_size = epsilon / max(steps, 1) * 2.0  # cover the ball in `steps`
    # Random start inside the L-inf ball. A zero start can be a stationary
    # point of a symmetric reconstruction loss (e.g. a quadratic minimized
    # at delta=0 has zero gradient there), which would trap vanilla PGD.
    delta = torch.empty_like(x0).uniform_(-epsilon, epsilon)
    # Validation runs under no_grad; the ascent needs a graph through the model.
    with torch.enable_grad():
        for _ in range(steps):
            delta = delta.clone().requires_grad_(True)
            out = model(x0 + delta)
            loss = torch.mean((out - target.detach().to(torch.float32)) ** 2)
            (grad,) = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta = delta + step_size * grad.sign()
                delta = torch.clamp(delta, min=-epsilon, max=epsilon)
    with torch.no_grad():
        return model(x0 + delta)


@register_metric("adversarial_psnr_drop", aliases=["adv_psnr_drop", "pgd_psnr_drop"])
class AdversarialPSNRDropMetric:
    """PSNR lost to a bounded worst-case input perturbation (lower is better).

    No-op (returns ``0.0``) when ``model`` / ``model_input`` are absent —
    see the module docstring. Constructor knobs:

    Args:
        epsilon: L-infinity perturbation budget (default ``1e-2``).
        steps: Number of PGD ascent steps (default ``5``).
    """

    def __init__(self, epsilon: float = 1e-2, steps: int = 5) -> None:
        self.epsilon = float(epsilon)
        self.steps = int(steps)

    @property
    def name(self) -> str:
        return "adversarial_psnr_drop"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        if prediction.shape != target.shape:
            raise ValueError(
                f"shape mismatch: pred {tuple(prediction.shape)} vs target {tuple(target.shape)}"
            )
        model = kwargs.get("model")
        model_input = kwargs.get("model_input")
        if model is None or model_input is None:
            # Documented no-op: cannot perturb an input we do not have.
            return 0.0

        clean_psnr = _psnr(prediction, target)
        adv = _pgd_linf_attack(
            model,
            model_input,
            target,
            epsilon=self.epsilon,
            steps=self.steps,
        )
        adv_psnr = _psnr(adv, target)
        return float(clean_psnr - adv_psnr)


@register_metric(
    "certified_radius",
    aliases=["smoothing_certified_radius"],
    requires_reference=False,
    needs=("reconstructor",),
    direction="higher",
)
class CertifiedRadiusMetric:
    """Randomized-smoothing radius lower bound (Cohen et al., ICML 2019), higher is better.

    The smoothed predictor's soft vote is the mean output response over
    ``n_samples`` Gaussian draws ``x + sigma * n``; the radius is
    ``sigma * Phi^{-1}(p)`` with ``p = sigmoid(vote)``. Needs the reconstructor
    (``context.reconstructor``) and the tensor it consumes (``model_input``), and
    raises without them: a radius of an input we do not have cannot be reported.

    Args:
        sigma: Smoothing noise standard deviation.
        n_samples: Monte-Carlo draws per input.
    """

    def __init__(self, sigma: float = 0.25, n_samples: int = 32) -> None:
        if sigma <= 0.0 or n_samples < 1:
            raise ValueError(
                f"certified_radius needs sigma > 0 and n_samples >= 1; got {sigma}, {n_samples}"
            )
        self.sigma = float(sigma)
        self.n_samples = int(n_samples)

    @property
    def name(self) -> str:
        return "certified_radius"

    @property
    def higher_is_better(self) -> bool:
        return True

    def per_sample(
        self, predict: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
    ) -> torch.Tensor:
        """The radius for each row of ``x``, shape ``[B]``."""
        with torch.no_grad():
            votes = torch.zeros(x.shape[0], device=x.device)
            for _ in range(self.n_samples):
                noisy = x + self.sigma * torch.randn_like(x)
                votes = votes + predict(noisy).flatten(1).mean(dim=1)
            p = (votes / self.n_samples).sigmoid().clamp(1e-3, 1 - 1e-3)
            return self.sigma * math.sqrt(2.0) * torch.erfinv(2 * p - 1)

    def __call__(
        self,
        prediction: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        del prediction, target
        recon = resolve_context(context, kwargs).reconstructor
        model_input = kwargs.get("model_input")
        if recon is None or not isinstance(model_input, torch.Tensor):
            raise ValueError(
                "certified_radius needs context.reconstructor and model_input=; the "
                "validation metrics computer supplies neither, so the strategy that "
                "owns the forward must call it (AdversarialRobustnessStrategy does)."
            )
        return float(self.per_sample(recon, model_input).mean().item())


@register_metric("ood_acceleration_shift", aliases=["ood_psnr_shift", "acceleration_ood_shift"])
class OODAccelerationShiftMetric:
    """PSNR degradation between nominal and held-out acceleration (lower better).

    No-op (returns ``0.0``) when no OOD pair is supplied — see the module
    docstring. The OOD pair may be passed directly as
    ``prediction_ood`` / ``target_ood`` kwargs, or produced lazily by an
    ``ood_fn`` callable returning ``(prediction_ood, target_ood)``.
    """

    @property
    def name(self) -> str:
        return "ood_acceleration_shift"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        if prediction.shape != target.shape:
            raise ValueError(
                f"shape mismatch: pred {tuple(prediction.shape)} vs target {tuple(target.shape)}"
            )

        prediction_ood = kwargs.get("prediction_ood")
        target_ood = kwargs.get("target_ood")
        ood_fn = kwargs.get("ood_fn")

        if (prediction_ood is None or target_ood is None) and ood_fn is not None:
            prediction_ood, target_ood = ood_fn()

        if prediction_ood is None or target_ood is None:
            # Documented no-op: no held-out-acceleration pair to compare.
            return 0.0

        if prediction_ood.shape != target_ood.shape:
            raise ValueError(
                f"OOD shape mismatch: pred {tuple(prediction_ood.shape)} vs "
                f"target {tuple(target_ood.shape)}"
            )

        in_dist_psnr = _psnr(prediction, target)
        ood_psnr = _psnr(prediction_ood, target_ood)
        return float(in_dist_psnr - ood_psnr)


@register_metric(
    name="input_dependence_spread",
    aliases=["measurement_sensitivity", "accel_pred_spread"],
)
class InputDependenceSpreadMetric:
    r"""Measurement-sensitivity gate (Experiment-11 DC-blob, L4).

    Quantifies whether a model's prediction varies with the measurement
    (i.e. across an acceleration cascade). The motivating failure is the
    k-space cold-diffusion **DC blob**: the network collapses to a
    measurement-INDEPENDENT contrast-mean that inverse-FFTs to a bright
    central blob, emitting an essentially fixed output regardless of the
    undersampling acceleration (``pred_mean ~ 0.1369 / 0.1366 / 0.1364``
    across R = 2x / 8x / 32x). Fidelity-drop metrics (PSNR/SSIM gap across
    accelerations, e.g. :class:`OODAccelerationShiftMetric`) *cannot* catch
    this: a constant-bad output has roughly constant-bad PSNR at every R, so
    the drop reads ~0 and the model looks deceptively "robust".

    Two evidence sources are supported, in order of preference:

    1. ``cascade_predictions`` — a list of per-acceleration prediction
       tensors (each ``[B, C, H, W]`` or higher rank). The metric computes a
       **per-pixel** standard deviation across the cascade axis, takes its
       mean magnitude, and normalises by the mean absolute prediction:

       .. math::

           S = \frac{\overline{\operatorname{std}_R\big(\hat{x}_R\big)}}
                    {\overline{|\hat{x}|} + \epsilon}

       This catches a *structural* measurement-independence that holds the
       scalar mean constant (a false negative for the scalar path below).

    2. ``cascade_pred_means`` — a list of the already-stamped per-level
       scalar prediction means (``val_pred_mean_<R>x``). The metric returns
       :math:`\sigma / (|\mu| + \epsilon)`. The cheap fallback that fires on
       the exact DC-blob signature.

    When neither kwarg is supplied the metric is a documented no-op (returns
    ``0.0``), mirroring the sibling robustness metrics so it never raises in
    a context that simply does not pass cascade evidence. A collapsed output
    yields spread ~0; ``higher_is_better=True`` because we *want* the output
    to depend on the input.

    Example::

        from spectramr.core.metrics import get_metric
        m = get_metric("input_dependence_spread")
        # collapsed (DC blob) -> ~0
        m(None, None, cascade_pred_means=[0.1369, 0.1366, 0.1364])
    """

    def __init__(self, eps: float = 1e-8, **kwargs: Any) -> None:
        """Initialise the metric.

        Args:
            eps: Numerical floor for the normalising denominator.
            **kwargs: Absorbed (e.g. ``device=`` injected by builders).
        """
        self.eps = float(eps)

    @property
    def name(self) -> str:
        """Metric canonical name."""
        return "input_dependence_spread"

    @property
    def higher_is_better(self) -> bool:
        """Higher spread -> output depends on the measurement (good)."""
        return True

    def __call__(
        self,
        prediction: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> float:
        """Compute the cross-acceleration spread.

        Args:
            prediction: Unused (kept for the metric call signature).
            target: Unused (kept for the metric call signature).
            **kwargs: Must contain ``cascade_predictions`` (preferred) or
                ``cascade_pred_means`` to produce a non-trivial value.

        Returns:
            Relative cross-acceleration spread (``0.0`` for a collapsed,
            measurement-independent output; ``0.0`` when no cascade
            evidence is provided).
        """
        preds = kwargs.get("cascade_predictions")
        if preds is not None and len(preds) >= 2:
            # Per-pixel std across the cascade axis, then mean magnitude,
            # normalised by the mean |prediction| -> relative structural spread.
            with torch.no_grad():
                stack = torch.stack(
                    [(p.abs() if torch.is_complex(p) else p).float() for p in preds],
                    dim=0,
                )
                per_pixel_std = stack.std(dim=0)
                denom = stack.abs().mean() + self.eps
                return float((per_pixel_std.mean() / denom).item())

        means = kwargs.get("cascade_pred_means")
        if means is not None and len(means) >= 2:
            mu = statistics.fmean(means)
            sd = statistics.pstdev(means)
            return float(sd / (abs(mu) + self.eps))

        return 0.0


# ``core/metrics/__init__`` walks every submodule on package import
# (pkgutil.walk_packages), so the ``@register_metric`` decorators above
# fire without any edit to ``__init__.py``.

__all__ = [
    "AdversarialPSNRDropMetric",
    "CertifiedRadiusMetric",
    "InputDependenceSpreadMetric",
    "OODAccelerationShiftMetric",
    "_psnr",
]
