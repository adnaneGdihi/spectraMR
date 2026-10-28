"""Model-Based Image Reconstruction (MBIR) Solver.

Implements iterative inversion of the MRI forward operator using
Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).
"""

import math
from typing import Protocol

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.wavelet_ops import (
    OrthogonalWavelet2D,
    complex_soft_threshold,
)

#: The regularizers the solver implements; anything else raises.
REGULARIZATIONS = ("identity", "tv", "wavelet")


class LinearOperator(Protocol):
    """What FISTA needs from ``A``: the forward map and its adjoint."""

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def adjoint(self, y: torch.Tensor) -> torch.Tensor: ...


class FISTAMBIRSolver(nn.Module):
    """FISTA solver for L1-regularized Model-Based Image Reconstruction.

    Solves: min_x 0.5 * ||A(x) - y||_2^2 + lambda * R(x)

    Where A is any forward/adjoint pair (``MultiCoilForwardOperator``,
    ``MulticoilEIOperator``) and R is one of ``REGULARIZATIONS``:

    * ``identity``: ``||x||_1``, exact prox (complex soft threshold);
    * ``wavelet``: ``||D Psi x||_1`` with ``Psi`` an orthonormal periodized wavelet
      and ``D`` selecting the detail coefficients (the coarsest approximation is not
      penalised). Exact prox, ``Psi^H soft(Psi x)``, because ``Psi`` is orthonormal;
    * ``tv``: a surrogate proximal step (see ``_apply_tv``), not the exact TV prox,
      so FISTA's convergence guarantees do not cover it.

    Args:
        forward_operator: the ``A`` to invert.
        num_iterations: FISTA iterations.
        lambda_reg: regularization weight.
        step_size: upper bound on the gradient step, or ``None`` for exactly ``1/L``.
        regularization: one of ``REGULARIZATIONS``; anything else raises.
        wavelet: the transform ``Psi``; required by, and only by, ``"wavelet"``.
        lipschitz: a known upper bound on ``||A^H A||``; ``None`` estimates it by
            power iteration, which approaches it from below.
    """

    def __init__(
        self,
        forward_operator: LinearOperator,
        num_iterations: int = 50,
        lambda_reg: float = 0.01,
        step_size: float | None = 0.5,
        regularization: str = "tv",
        wavelet: OrthogonalWavelet2D | None = None,
        lipschitz: float | None = None,
    ):
        super().__init__()
        if regularization not in REGULARIZATIONS:
            raise ValueError(
                f"Unknown regularization {regularization!r}; choose one of {REGULARIZATIONS}."
            )
        if (regularization == "wavelet") != (wavelet is not None):
            raise ValueError(
                "regularization='wavelet' requires a wavelet transform, and no other "
                f"regularization reads one (got regularization={regularization!r}, "
                f"wavelet={wavelet!r})."
            )
        if lambda_reg < 0 or num_iterations < 0:
            raise ValueError(
                f"lambda_reg and num_iterations must be >= 0, got {lambda_reg}, {num_iterations}."
            )
        if lipschitz is not None and not lipschitz > 0:
            raise ValueError(f"lipschitz must be a positive bound, got {lipschitz}.")
        self.A = forward_operator
        self.num_iterations = num_iterations
        self.lambda_reg = lambda_reg
        self.alpha = step_size
        self.regularization = regularization
        self.wavelet = wavelet
        self.lipschitz = lipschitz

    def _soft_threshold(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Apply soft thresholding to a complex tensor (exact prox of the l1 norm)."""
        return complex_soft_threshold(x, threshold)

    def _wavelet_details(self, coeffs: torch.Tensor) -> torch.Tensor:
        assert self.wavelet is not None
        return self.wavelet.detail_mask(*coeffs.shape[-2:], device=coeffs.device)

    def _apply_wavelet(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        r"""Exact :math:`\operatorname{prox}_{\tau\lVert D\Psi\cdot\rVert_1}`."""
        assert self.wavelet is not None
        coeffs = self.wavelet.forward(x)
        shrunk = torch.where(
            self._wavelet_details(coeffs), self._soft_threshold(coeffs, threshold), coeffs
        )
        return self.wavelet.inverse(shrunk)

    def regularizer(self, x: torch.Tensor) -> torch.Tensor:
        """``R(x)`` summed over the batch, for the objective (not used by the iteration)."""
        if self.regularization == "wavelet":
            assert self.wavelet is not None
            coeffs = self.wavelet.forward(x)
            return coeffs.abs()[..., self._wavelet_details(coeffs)].sum()
        if self.regularization == "tv":
            dh = x - torch.roll(x, shifts=1, dims=-2)
            dw = x - torch.roll(x, shifts=1, dims=-1)
            return dh.abs().sum() + dw.abs().sum()
        return x.abs().sum()

    def objective(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """``0.5 ||A x - y||^2 + lambda R(x)``, summed over the batch."""
        residual = self.A.forward(x) - y
        return 0.5 * residual.abs().pow(2).sum() + self.lambda_reg * self.regularizer(x)

    def _prox(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        if self.regularization == "tv":
            return self._apply_tv(x, threshold)
        if self.regularization == "wavelet":
            return self._apply_wavelet(x, threshold)
        return self._soft_threshold(x, threshold)

    def _apply_tv(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """One-step TV gradient surrogate (NOT the exact TV proximal operator).

        Computes ``x - D^T(soft_threshold(D x))`` where D is the circular forward-
        difference (D^T is exact, including the wrap term). This is a single
        Landweber/Jacobi-style descent step on the TV-smoothing objective, not
        ``prox_{lambda*TV}(x)``: in the small-threshold regime it reduces to
        ``(I - D^T D) x``, and ``I - D^T D`` is expansive (Laplacian eigenvalues up
        to 4), so it can overshoot/oscillate and does not satisfy the proximal-map
        contract FISTA's convergence theory assumes. For a rigorous TV prox, replace
        with Chambolle dual projected-gradient iterations (tracked as a follow-up).
        """
        # Calculate gradients (differences)
        dh = x - torch.roll(x, shifts=1, dims=-2)
        dw = x - torch.roll(x, shifts=1, dims=-1)

        # Soft threshold gradients
        dh_thresh = self._soft_threshold(dh, threshold)
        dw_thresh = self._soft_threshold(dw, threshold)

        # Integrate back (adjoint of difference)
        div = (dh_thresh - torch.roll(dh_thresh, shifts=-1, dims=-2)) + (
            dw_thresh - torch.roll(dw_thresh, shifts=-1, dims=-1)
        )

        return x - div

    def forward(
        self,
        y: torch.Tensor | None = None,
        x_init: torch.Tensor | None = None,
        *,
        adjoint_measurement: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Run FISTA from the measurement ``y`` or from :math:`b=A^Hy` alone.

        The data-term gradient is :math:`A^H(Az-y)=A^HAz-b`, so the iteration needs
        :math:`y` only through :math:`b`. A caller that already holds :math:`b` (the
        SENSE-combined zero-filled image a reconstruction strategy feeds its
        network) passes it as ``adjoint_measurement`` and the minimiser is the same.

        Args:
            y: acquired k-space ``[B, C, H, W]``; exclusive with ``adjoint_measurement``.
            x_init: initial image ``[B, 1, H, W]``; defaults to :math:`b`.
            adjoint_measurement: :math:`b=A^Hy`, ``[B, 1, H, W]`` complex.

        Returns:
            The reconstructed image ``[B, 1, H, W]``.
        """
        if (y is None) == (adjoint_measurement is None):
            raise ValueError("FISTAMBIRSolver needs exactly one of y and adjoint_measurement.")
        b = self.A.adjoint(y) if adjoint_measurement is None else adjoint_measurement
        x = b.clone() if x_init is None else x_init.clone()

        # FISTA's O(1/k^2) guarantee needs the gradient step alpha <= 1/L with
        # L = ||A*A||_2. A declared bound is used as is; otherwise L is estimated by
        # power iteration, which approaches it from below. ``step_size`` caps the
        # step; ``None`` takes exactly 1/L.
        if self.lipschitz is not None:
            lam = x.real.new_tensor(self.lipschitz)
        else:
            with torch.no_grad():
                v = torch.randn_like(x)
                v = v / (v.norm() + 1e-12)
                lam = v.norm().real
                for _ in range(8):
                    av = self.A.adjoint(self.A.forward(v))
                    lam = av.norm()  # ||A*A v|| / ||v|| with ||v|| = 1
                    v = av / (lam + 1e-12)
        inverse_l = 1.0 / lam.clamp_min(1e-8)
        alpha = inverse_l if self.alpha is None else torch.minimum(lam.new_tensor(self.alpha), inverse_l)

        z = x.clone()  # Nesterov momentum state
        t = 1.0  # Momentum scaling

        for _ in range(self.num_iterations):
            # 1. Evaluate gradient at z
            # With y in hand the residual form keeps precision near the solution,
            # where A^H A z and b agree to most of their digits.
            if y is None:
                gradient = self.A.adjoint(self.A.forward(z)) - b
            else:
                gradient = self.A.adjoint(self.A.forward(z) - y)

            # 2. Gradient Descent Step (Lipschitz-safe step size)
            x_next_unproxed = z - alpha * gradient

            # 3. Proximal Step (Regularization)
            x_next = self._prox(x_next_unproxed, self.lambda_reg * alpha)

            # 4. Nesterov Momentum Update
            t_next = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
            momentum = (t - 1.0) / t_next
            z = x_next + momentum * (x_next - x)

            # Update state
            x = x_next
            t = t_next

        return x


def positive_contrast_demodulation(
    kspace: torch.Tensor, delta_omega_hz: float, te_sec: float | torch.Tensor
) -> torch.Tensor:
    """Pre-process K-space to shift target frequency to DC for positive contrast.

    Standard susceptibility markers cause signal voids due to violent phase dispersion
    (Δω). By pre-demodulating the k-space, we artificially refocus the marker spins
    so they reconstruct brightly (Positive Contrast), while intentionally blurring out
    the background anatomy.

    Note:
        Positive-contrast refocusing requires a *per-readout-sample* ``te_sec`` tensor
        whose length equals the readout (last) axis of ``kspace`` — only a time-varying
        demodulation refocuses the marker. A scalar ``te_sec`` applies a single uniform
        phase to all of k-space, which is mathematically a global image phase: it
        preserves magnitude and produces NO contrast shift. Pass a scalar only when a
        global phase is intended.

    Args:
        kspace: Acquired k-space [B, C, H, W] or [B, C, K]
        delta_omega_hz: Expected off-resonance frequency of the marker
        te_sec: Echo time in seconds. Scalar float (uniform global phase) or 1-D tensor
            whose length matches the k-space readout (last) axis (time-segmented).

    Returns:
        Demodulated k-space matching input shape

    Raises:
        ValueError: if ``te_sec`` is a tensor whose shape does not match the k-space
            readout (last) axis (silent mis-broadcast guard).
    """
    if isinstance(te_sec, torch.Tensor):
        te = te_sec.to(kspace.device).squeeze()
        if te.ndim > 1:
            raise ValueError(
                f"te_sec tensor must be scalar or 1-D, got shape {tuple(te_sec.shape)}"
            )
        if te.ndim == 1 and te.shape[0] != kspace.shape[-1]:
            raise ValueError(
                "te_sec length must match the k-space readout (last) axis "
                f"({kspace.shape[-1]}), got {te.shape[0]}. A 1-D te_sec is "
                "demodulated along the trailing axis only."
            )
        phase_shift_rad = 2.0 * math.pi * delta_omega_hz * te
    else:
        phase_shift_rad = torch.tensor(
            2.0 * math.pi * delta_omega_hz * te_sec, device=kspace.device
        )

    demod_factor = torch.exp(1j * phase_shift_rad)
    return kspace * demod_factor
