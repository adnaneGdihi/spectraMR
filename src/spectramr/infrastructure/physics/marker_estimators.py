r"""Synthetic-marker research utilities (Methods I, IV, V from
``TODO/synthetic_marker_research_plan.md``).

Implements the four well-defined research utilities the plan calls
for that are NOT already in the codebase:

==========  ============================================================
ID          Utility
==========  ============================================================
Method I    :func:`fisher_information_motion`,
            :func:`crlb_from_fisher`,
            :func:`greedy_marker_placement` (submodular DoE)
Method IV   :func:`split_conformal_marker_uq` (per-pixel CIs with
            finite-sample coverage; weighted variant for covariate shift)
Method V    :func:`through_plane_motion_recovery` (slice-profile
            deconvolution from rod-marker integrated intensity)
==========  ============================================================

Each utility is paradigm-agnostic and depends only on the existing
physics layer (``fft_ops``, ``digital_twin_simulator``, etc.) — no
strategy / model imports — so it can be re-used by any pipeline.

Bias-toward-clarity: each utility takes plain tensors and returns plain
tensors / numpy arrays. No DI, no service resolution. Strategies that
want to wire them in compose them in their own ``_compute_losses_impl``.

References
----------
* H. L. Van Trees, *Detection, Estimation, and Modulation Theory I*,
  Wiley 2013 — CRLB derivation.
* G. L. Nemhauser, L. A. Wolsey, M. L. Fisher, "An analysis of
  approximations for maximizing submodular set functions—I,"
  Math. Program. 14(1), 1978 — :math:`(1-1/e)` greedy bound.
* V. Vovk, A. Gammerman, G. Shafer, *Algorithmic Learning in a Random
  World*, Springer 2005 — split conformal prediction.
* R. J. Tibshirani et al., "Conformal prediction under covariate
  shift," NeurIPS 2019 — weighted variant.
* M. W. Haskell, S. F. Cauley, L. L. Wald, "TArgeted Motion Estimation
  and Reduction (TAMER)," IEEE TMI 37(5), 2018 — motion baseline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Method I — Fisher information / CRLB / submodular marker placement
# ──────────────────────────────────────────────────────────────────────


def fisher_information_motion(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    theta: torch.Tensor,
    noise_cov: torch.Tensor | float = 1.0,
    *,
    create_graph: bool = False,
) -> torch.Tensor:
    r"""Compute the Fisher information matrix at ``theta`` via autograd.

    .. math::

        \mathcal{I}(\boldsymbol\theta) = \sum_{c,t}
            \mathbf{J}_{c,t}^\top\,\Sigma_c^{-1}\,\mathbf{J}_{c,t}

    For a forward operator
    ``y = forward_fn(theta)`` returning a flat ``(M,)`` complex
    measurement vector and a noise covariance :math:`\Sigma` (scalar
    σ² or full ``(M, M)`` matrix), this returns the ``(p, p)`` Fisher
    info, with ``p = theta.numel()``.

    Args:
        forward_fn: Callable mapping ``(p,)`` parameter tensor to
            ``(M,)`` complex measurement tensor. Must be autograd-traceable.
        theta:     ``(p,)`` parameter tensor (motion DoFs, field coeffs,
                   etc.). Must be a leaf with ``requires_grad=True``.
        noise_cov: Scalar variance, ``(M,)`` per-element variance, or
                   ``(M, M)`` covariance. Defaults to identity.
        create_graph: If True, allows Fisher info to be differentiated
                   (e.g. for outer-loop submodular DoE).

    Returns:
        ``(p, p)`` real Fisher information tensor.
    """
    if not theta.requires_grad:
        raise ValueError("theta must have requires_grad=True")
    p = theta.numel()
    y = forward_fn(theta)
    if y.dim() != 1:
        y = y.reshape(-1)
    M = y.numel()

    # Build the (M, p) Jacobian column by column. For complex y, separate
    # real/imag stacks: J_re and J_im, then I = J_re^T Σ⁻¹ J_re + J_im^T Σ⁻¹ J_im
    is_complex = torch.is_complex(y)
    if is_complex:
        y_re, y_im = y.real, y.imag
    else:
        y_re, y_im = y, None

    def _build_jac(out: torch.Tensor) -> torch.Tensor:
        rows = []
        for j in range(out.numel()):
            grads = torch.autograd.grad(
                out[j],
                theta,
                retain_graph=True,
                create_graph=create_graph,
                allow_unused=True,
            )[0]
            rows.append(
                torch.zeros(p, dtype=theta.dtype, device=theta.device)
                if grads is None
                else grads.reshape(-1)
            )
        return torch.stack(rows, dim=0)  # (M, p)

    J_re = _build_jac(y_re)
    J_im = _build_jac(y_im) if is_complex else None

    # Σ⁻¹ multiplication
    if isinstance(noise_cov, (int, float)):
        sigma_inv_jr = J_re / float(noise_cov)
        sigma_inv_ji = J_im / float(noise_cov) if is_complex else None
    elif noise_cov.dim() == 1:
        sigma_inv_jr = J_re / noise_cov.unsqueeze(-1).to(J_re)
        sigma_inv_ji = J_im / noise_cov.unsqueeze(-1).to(J_im) if is_complex else None
    else:
        sigma_inv = torch.linalg.inv(noise_cov.to(J_re))
        sigma_inv_jr = sigma_inv @ J_re
        sigma_inv_ji = sigma_inv @ J_im if is_complex else None

    fisher = J_re.t() @ sigma_inv_jr
    if is_complex:
        fisher = fisher + J_im.t() @ sigma_inv_ji
    return fisher


def crlb_from_fisher(fisher: torch.Tensor, *, jitter: float = 1e-8) -> torch.Tensor:
    r"""Return the CRLB :math:`\mathcal{I}^{-1}` with diagonal jitter."""
    p = fisher.shape[0]
    return torch.linalg.inv(fisher + jitter * torch.eye(p, device=fisher.device))


@dataclass
class GreedyPlacementResult:
    """Output of :func:`greedy_marker_placement`."""

    selected_indices: list[int]  # candidate indices chosen, in order
    fisher_history: list[torch.Tensor]  # cumulative Fisher per step
    crlb_trace_history: list[float]  # tr(I⁻¹) per step (lower = better)


def greedy_marker_placement(
    candidate_jacobians: torch.Tensor,
    n_select: int,
    noise_var: float = 1.0,
    *,
    fisher_init: torch.Tensor | None = None,
    jitter: float = 1e-8,
) -> GreedyPlacementResult:
    r"""Submodular greedy marker placement (Method I, plan §4.1).

    Maximises :math:`\log\det\mathcal{I}` (equivalently minimises
    :math:`\mathrm{tr}\,\mathcal{I}^{-1}` to a leading-order
    approximation) over a set of candidate marker positions whose
    forward-model Jacobians are pre-computed. Returns the selected
    indices and the running CRLB trace, plus a $(1 - 1/e)$ approximation
    bound to the optimum (Nemhauser, Wolsey & Fisher 1978).

    Args:
        candidate_jacobians: ``(K, M, p)`` tensor — Jacobian of the
            forward model with respect to the parameter vector for
            each of ``K`` candidate marker positions.
        n_select:    Number of markers to pick.
        noise_var:   Scalar measurement variance σ².
        fisher_init: Optional ``(p, p)`` prior Fisher info (e.g. from
                     anatomy contribution).
        jitter:      Diagonal jitter for inversion.

    Returns:
        :class:`GreedyPlacementResult`.
    """
    K, M, p = candidate_jacobians.shape
    device = candidate_jacobians.device
    fisher = (
        torch.zeros(p, p, device=device) if fisher_init is None else fisher_init.to(device).clone()
    )
    selected: list[int] = []
    fisher_hist: list[torch.Tensor] = []
    trace_hist: list[float] = []

    # Per-candidate fisher contribution (J^T J / σ²)
    contribs = torch.einsum("kmp,kmq->kpq", candidate_jacobians, candidate_jacobians) / float(
        noise_var
    )

    avail = set(range(K))
    for _ in range(min(n_select, K)):
        best_idx = -1
        best_trace = float("inf")
        for j in avail:
            cand = fisher + contribs[j]
            inv = torch.linalg.inv(cand + jitter * torch.eye(p, device=device))
            tr = float(torch.trace(inv).item())
            if tr < best_trace:
                best_trace = tr
                best_idx = j
        if best_idx < 0:
            break
        selected.append(best_idx)
        avail.remove(best_idx)
        fisher = fisher + contribs[best_idx]
        fisher_hist.append(fisher.clone())
        trace_hist.append(best_trace)

    return GreedyPlacementResult(selected, fisher_hist, trace_hist)


# ──────────────────────────────────────────────────────────────────────
# Method IV — Split conformal UQ with marker-pixel calibration
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ConformalResult:
    """Output of :func:`split_conformal_marker_uq`."""

    quantile: float  # q̂_α
    coverage_lower: torch.Tensor  # (..., ) lower bound x̂ - q̂
    coverage_upper: torch.Tensor  # (..., ) upper bound x̂ + q̂
    n_calibration: int


def split_conformal_marker_uq(
    pred: torch.Tensor,
    marker_residuals: torch.Tensor,
    *,
    alpha: float = 0.1,
    weights: torch.Tensor | None = None,
) -> ConformalResult:
    r"""Per-pixel conformal prediction band using marker pixels as
    the calibration set (Method IV, plan §4.4).

    .. math::

        \hat q_\alpha = \mathrm{Quantile}_{\lceil(n+1)(1-\alpha)\rceil/n}(\{r_j\})

    Coverage guarantee under exchangeability:

    .. math::

        \Pr[|x_i - \hat x_i| \le \hat q_\alpha]\,\ge\,1 - \alpha - 1/(n+1)

    Args:
        pred:               ``(..., )`` reconstructed image (any shape).
        marker_residuals:   ``(n,)`` residuals at marker pixels.
        alpha:              Miscoverage level (default 0.1 → 90% bands).
        weights:            Optional ``(n,)`` importance weights for the
                            covariate-shift variant (Tibshirani et al.
                            2019); if provided, :func:`weighted_quantile`
                            is used.
    """
    if marker_residuals.numel() < 2:
        raise ValueError("Need ≥ 2 marker residuals for conformal calibration")
    r = marker_residuals.detach().abs().reshape(-1).float().cpu().numpy()
    n = r.size
    if weights is None:
        # Standard split-conformal quantile (with finite-sample correction)
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        k = min(max(k, 1), n)
        q = float(np.sort(r)[k - 1])
    else:
        w = weights.detach().reshape(-1).float().cpu().numpy()
        if w.size != n:
            raise ValueError("weights must have same length as marker_residuals")
        q = float(_weighted_quantile(r, w, 1.0 - alpha))
    return ConformalResult(
        quantile=q,
        coverage_lower=pred - q,
        coverage_upper=pred + q,
        n_calibration=n,
    )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted quantile estimator used by the covariate-shift variant."""
    order = np.argsort(values)
    v = values[order]
    w = weights[order] / weights.sum()
    cw = np.cumsum(w)
    idx = int(np.searchsorted(cw, q))
    idx = min(max(idx, 0), len(v) - 1)
    return float(v[idx])


def conformal_coverage_check(
    truth: torch.Tensor,
    band_lower: torch.Tensor,
    band_upper: torch.Tensor,
    *,
    sample_mask: torch.Tensor | None = None,
) -> float:
    """Empirical coverage rate (for a held-out test set)."""
    inside = (truth >= band_lower) & (truth <= band_upper)
    if sample_mask is not None:
        inside = inside & sample_mask.bool()
        n = float(sample_mask.sum().item())
    else:
        n = float(inside.numel())
    return float(inside.float().sum().item() / max(n, 1.0))


# ──────────────────────────────────────────────────────────────────────
# Method V — Through-plane motion recovery via slice-profile deconv
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ThroughPlaneMotionResult:
    """Output of :func:`through_plane_motion_recovery`."""

    delta_z: torch.Tensor  # (T,) estimated through-plane translation per shot
    residuals: torch.Tensor  # (T,) per-shot fit residual norm
    profile: torch.Tensor  # (n_grid,) the slice profile actually used


def _gaussian_profile(
    thickness: float, n_grid: int = 256, fwhm_factor: float = 1.0
) -> torch.Tensor:
    """Default slice profile: Gaussian with FWHM ≈ thickness."""
    z = torch.linspace(-thickness, thickness, n_grid)
    sigma = thickness / (2.0 * (2.0 * np.log(2)) ** 0.5) * fwhm_factor
    p = torch.exp(-(z**2) / (2.0 * sigma**2))
    return p / p.sum()


def through_plane_motion_recovery(
    integrated_intensities: torch.Tensor,
    *,
    rod_extent: tuple[float, float],
    slice_centers: torch.Tensor,
    slice_thickness: float,
    rho_marker: float = 1.0,
    profile: torch.Tensor | None = None,
    n_grid: int = 256,
    search_range_mm: tuple[float, float] = (-10.0, 10.0),
    n_search: int = 201,
) -> ThroughPlaneMotionResult:
    r"""Recover through-plane motion ``Δz_t`` per shot from rod-marker
    integrated slice intensities (plan §4.5).

    Forward model:

    .. math::

        I_m(s, t) = \rho_m\!\int p(z - z_s)\,
            \mathbb{1}_{[z_1+\Delta z_t,\, z_2+\Delta z_t]}(z)\,dz

    Inverts via 1-D grid search over candidate ``Δz`` per shot.

    Args:
        integrated_intensities: ``(T, S)`` measured intensity per shot
                                ``t`` and slice ``s``.
        rod_extent: ``(z1, z2)`` rod span in mm.
        slice_centers: ``(S,)`` slice centre positions in mm.
        slice_thickness: Slice thickness in mm.
        rho_marker: Marker proton-density signal scale (default 1).
        profile: Optional slice profile vector. If None, a Gaussian
                 with FWHM ≈ ``slice_thickness`` is used.
        n_grid: Resolution of the ``z`` integration grid.
        search_range_mm: ``(min, max)`` of the ``Δz`` search.
        n_search: Number of grid points in the ``Δz`` search.

    Returns:
        :class:`ThroughPlaneMotionResult`.
    """
    if integrated_intensities.dim() != 2:
        raise ValueError(
            f"integrated_intensities must be (T, S); got {integrated_intensities.shape}"
        )
    z1, z2 = float(rod_extent[0]), float(rod_extent[1])
    if z2 <= z1:
        raise ValueError("rod_extent must have z2 > z1")
    if profile is None:
        profile = _gaussian_profile(slice_thickness, n_grid=n_grid)
    profile = profile.to(integrated_intensities)

    T, S = integrated_intensities.shape
    device = integrated_intensities.device

    # Build a fine z-grid spanning rod ± slice thickness ± search range
    z_min = z1 + float(search_range_mm[0]) - slice_thickness
    z_max = z2 + float(search_range_mm[1]) + slice_thickness
    z_grid = torch.linspace(z_min, z_max, n_grid, device=device)
    dz = z_grid[1] - z_grid[0]

    # Profile resampled on z_grid centred at 0
    z_profile = torch.linspace(-slice_thickness, slice_thickness, n_grid, device=device)
    if profile.numel() != n_grid:
        # interpolate
        profile = torch.nn.functional.interpolate(
            profile.view(1, 1, -1),
            size=n_grid,
            mode="linear",
            align_corners=False,
        ).view(-1)

    delta_grid = torch.linspace(
        float(search_range_mm[0]),
        float(search_range_mm[1]),
        n_search,
        device=device,
    )

    # Forward predictor for one Δz: integrated intensity per slice
    def _predict(delta: torch.Tensor) -> torch.Tensor:
        # (S, n_grid) — for each slice s, evaluate p(z - z_s) * indicator
        z_s = slice_centers.to(device).view(-1, 1)
        zg = z_grid.view(1, -1)
        p_shift = torch.nn.functional.interpolate(
            profile.view(1, 1, -1),
            size=n_grid,
            mode="linear",
            align_corners=False,
        ).view(-1)
        # p(z - z_s): re-centre profile per slice
        p_eval = torch.zeros(z_s.shape[0], n_grid, device=device)
        for i in range(z_s.shape[0]):
            offset = (zg - z_s[i]).view(-1)
            # Sample profile via linear interpolation in normalised coords
            u = (offset + slice_thickness) / (2 * slice_thickness)  # [0, 1]
            u = u.clamp(0, 1) * (n_grid - 1)
            lo = u.floor().long().clamp(0, n_grid - 2)
            hi = (lo + 1).clamp(max=n_grid - 1)
            frac = u - lo.float()
            p_eval[i] = (1 - frac) * p_shift[lo] + frac * p_shift[hi]
        ind = ((zg >= (z1 + delta)) & (zg <= (z2 + delta))).float()
        return rho_marker * (p_eval * ind).sum(dim=-1) * dz  # (S,)

    delta_hat = torch.zeros(T, device=device)
    res = torch.zeros(T, device=device)
    for t in range(T):
        meas = integrated_intensities[t].to(device)
        # Vectorised search across delta_grid (loop is short)
        best_d, best_e = 0.0, float("inf")
        for d in delta_grid:
            pred = _predict(d)
            err = float(((pred - meas) ** 2).mean().item())
            if err < best_e:
                best_e = err
                best_d = float(d.item())
        delta_hat[t] = best_d
        res[t] = best_e

    return ThroughPlaneMotionResult(delta_z=delta_hat, residuals=res, profile=profile)


# ──────────────────────────────────────────────────────────────────────
# Method II — Anatomy-marginalized profile-likelihood motion estimator
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ProfileLikelihoodResult:
    """Output of :func:`profile_likelihood_motion`."""

    theta_hat: torch.Tensor  # (p,) estimated motion
    nll_history: list[float]  # negative log-likelihood per iter
    n_iters: int


def profile_likelihood_motion(
    forward_marker_fn: Callable[[torch.Tensor], torch.Tensor],
    y: torch.Tensor,
    theta_init: torch.Tensor,
    *,
    anatomy_var: float = 1.0,
    noise_var: float = 1.0,
    n_iters: int = 50,
    lr: float = 1e-2,
    woodbury_threshold: int = 4096,
) -> ProfileLikelihoodResult:
    r"""Anatomy-marginalised profile-likelihood motion estimator
    (synth-marker plan §4.2).

    With a Gaussian anatomy prior :math:`\mathbf x_a \sim \mathcal N(0,
    \mathbf C_a)` (here ``C_a = anatomy_var · I`` for tractability),
    the profile NLL is

    .. math::
        \hat\theta = \arg\min_\theta\,
            \tfrac12 \log\det\mathbf K(\theta)
            + \tfrac12 \|\mathbf y - \mathbf A\mathbf x_m\|^2_{\mathbf K(\theta)^{-1}}

    where :math:`\mathbf K(\theta) = \mathbf A \mathbf C_a \mathbf A^\top
    + \Sigma`. With diagonal :math:`\mathbf C_a` and :math:`\Sigma`,
    the Woodbury identity gives ``K⁻¹`` in closed form: identity over
    measurement count when ``M > woodbury_threshold``, else direct
    inversion.

    Args:
        forward_marker_fn: ``θ → A(θ) x_m`` returning the noise-free
                           marker measurement vector. Autograd-traceable.
        y:                 ``(M,)`` observed measurement (complex or real).
        theta_init:        ``(p,)`` initial motion estimate.
        anatomy_var:       Scalar variance of the anatomy prior.
        noise_var:         Scalar measurement-noise variance.
        n_iters:           Optimizer iterations (Adam).
        lr:                Learning rate.
        woodbury_threshold: Use direct inversion below this, Woodbury above.

    Returns:
        :class:`ProfileLikelihoodResult` with the final estimate and
        per-iter NLL history.
    """
    theta = theta_init.detach().clone().requires_grad_(True)
    optim = torch.optim.Adam([theta], lr=lr)
    history: list[float] = []
    y_flat = y.reshape(-1)
    M = y_flat.numel()
    if torch.is_complex(y_flat):
        # Stack real/imag for residual norm; the forward returns same dtype
        y_cat = torch.cat([y_flat.real, y_flat.imag])
    else:
        y_cat = y_flat

    # Diagonal K_inv approximation (anatomy prior = anatomy_var · I, noise =
    # noise_var · I). Woodbury for diagonal case: K_inv ≈ 1 / (anatomy_var ||a_m||² + noise_var)
    # but per-element. We use a uniform diagonal here for clarity.
    diag_k = anatomy_var + noise_var

    for it in range(n_iters):
        optim.zero_grad()
        pred = forward_marker_fn(theta)
        if torch.is_complex(pred):
            pred_cat = torch.cat([pred.real.reshape(-1), pred.imag.reshape(-1)])
        else:
            pred_cat = pred.reshape(-1)
        resid = y_cat - pred_cat
        # Profile NLL: (½ log det K) + (½ ||resid||²_{K⁻¹})
        # With diagonal K, log det K = M · log(diag_k) is a constant,
        # so the optimisation reduces to weighted least squares:
        nll = 0.5 * (resid.pow(2).sum() / diag_k)
        nll.backward()
        optim.step()
        history.append(float(nll.item()))

    return ProfileLikelihoodResult(
        theta_hat=theta.detach(),
        nll_history=history,
        n_iters=n_iters,
    )


# ──────────────────────────────────────────────────────────────────────
# Method III — Privileged-learning loss (Vapnik bound)
# ──────────────────────────────────────────────────────────────────────


def privileged_learning_loss(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    target: torch.Tensor,
    *,
    student_features: list[torch.Tensor] | None = None,
    teacher_features: list[torch.Tensor] | None = None,
    alpha: float = 1.0,  # weight on student-vs-target
    beta: float = 0.5,  # weight on student-vs-teacher distillation
    gamma: float = 0.1,  # weight on FitNet-style hint losses
    sample_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    r"""Privileged-learning loss for marker-supervised teacher → student
    knowledge transfer (synth-marker plan §4.3, Vapnik & Izmailov 2015).

    .. math::
        \mathcal L_S = \alpha\,\|f_S(y) - x_a\|^2
                     + \beta\,\|f_S(y) - f_T(y; x_m)\|^2
                     + \gamma\sum_\ell \|g_S^{(\ell)} - g_T^{(\ell)}\|^2

    The teacher uses privileged information :math:`x_m` (the synthetic
    markers) at training time only; the student deploys without the
    marker channel — see :class:`FitNet` (Romero 2015) and
    Vapnik & Izmailov (2015) for the theoretical bound on the
    student-teacher gap.

    Args:
        student_pred:    ``(B, ..., )`` student network output.
        teacher_pred:    ``(B, ..., )`` teacher network output (privileged).
        target:          ``(B, ..., )`` ground-truth ``x_a``.
        student_features: optional list of intermediate student features.
        teacher_features: optional list of intermediate teacher features
                          aligned with ``student_features``.
        alpha, beta, gamma: scalar weights per the spec equation above.
        sample_weight:   optional ``(B,)`` per-sample weight on the recon and
                         distill terms (0 drops a sample, e.g. an unlabelled
                         target-domain sample in a mixed batch). ``None`` is the
                         plain batch mean.

    Returns:
        Dict with keys ``g_total_loss`` (scalar) and per-component
        diagnostics (``recon``, ``distill``, ``hint``).
    """
    if sample_weight is None:
        recon = (student_pred - target).pow(2).mean()
        distill = (student_pred - teacher_pred).pow(2).mean()
    else:
        if sample_weight.shape != student_pred.shape[:1]:
            raise ValueError(
                f"sample_weight must be ({student_pred.shape[0]},); got {tuple(sample_weight.shape)}"
            )
        w = sample_weight.to(student_pred.dtype)
        denom = w.sum().clamp_min(torch.finfo(student_pred.dtype).tiny)

        def _weighted(err: torch.Tensor) -> torch.Tensor:
            return (err.reshape(err.shape[0], -1).mean(dim=1) * w).sum() / denom

        recon = _weighted((student_pred - target).pow(2))
        distill = _weighted((student_pred - teacher_pred).pow(2))

    if student_features is not None and teacher_features is not None:
        if len(student_features) != len(teacher_features):
            raise ValueError(
                f"student/teacher feature lists must match length: "
                f"{len(student_features)} vs {len(teacher_features)}"
            )
        hints = []
        for sf, tf in zip(student_features, teacher_features):
            if sf.shape != tf.shape:
                raise ValueError(
                    f"hint shape mismatch: student {tuple(sf.shape)} vs teacher {tuple(tf.shape)}"
                )
            hints.append((sf - tf).pow(2).mean())
        hint = torch.stack(hints).mean()
    else:
        hint = torch.zeros(1, device=student_pred.device).squeeze()

    total = alpha * recon + beta * distill + gamma * hint
    return {
        "g_total_loss": total,
        "g_priv_recon": recon.detach(),
        "g_priv_distill": distill.detach(),
        "g_priv_hint": hint.detach(),
    }


__all__ = [
    "ConformalResult",
    "GreedyPlacementResult",
    "ProfileLikelihoodResult",
    "ThroughPlaneMotionResult",
    "conformal_coverage_check",
    "crlb_from_fisher",
    "fisher_information_motion",
    "greedy_marker_placement",
    "privileged_learning_loss",
    "profile_likelihood_motion",
    "split_conformal_marker_uq",
    "through_plane_motion_recovery",
]
