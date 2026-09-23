r"""Fit the coil noise covariance from the null space of one scan.

``coil_subspace`` SCORES a reconstruction against a known :math:`\Sigma_n`.
This fits one. They are separate because their consumers are: the budget is a
per-pixel metric, this is a per-scan calibration that runs once.

The premise is the same asymmetry. A physical image is rank one in coil space,
so :math:`P_q^{\perp}y_q` is pure noise however bright the anatomy is --
measured on the committed M4Raw covariance, about 75% of the noise energy lands
in the 3-D null space and none of the signal does. That makes every pixel an
observation of the noise, and there are ~65,000 of them in a 256x256 slice.

**Why this matters here.** ``m4raw_noise.py`` ships a covariance measured on 7
subjects from one study series and scopes itself explicitly: "Receiver gain may
differ elsewhere in the corpus." R2R's recorruption is exact only where
:math:`\Sigma_z=\Sigma_n`, so every arm using it inherits that scope. This
replaces the constant with a quantity each scan supplies about itself.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.coil_subspace import _as_coil_vectors

__all__ = [
    "UnidentifiableCovarianceError",
    "estimate_covariance_from_nullspace",
    "sigma_from_coil_nullspace",
    "sigma_from_kspace_batch",
]


class UnidentifiableCovarianceError(RuntimeError):
    """The null spaces carry too little coil diversity to determine Sigma_n."""


def estimate_covariance_from_nullspace(
    coil_images: torch.Tensor,
    null_projector: torch.Tensor,
    *,
    support: torch.Tensor | None = None,
    max_condition: float = 1e6,
) -> torch.Tensor:
    r"""Estimate :math:`\Sigma_n` per scan from the coil null space alone.

    A physical image is rank one in coil space, so :math:`P_q^{\perp}y_q` is
    pure noise no matter how bright the anatomy is. Fitting

    .. math::

        \hat{\Sigma}_n=\arg\min_{\Sigma}\sum_q\bigl\lVert P_q^{\perp}
        \bigl(y_qy_q^{H}-\Sigma\bigr)P_q^{\perp}\bigr\rVert_F^2

    gives normal equations that are linear in :math:`\Sigma`. Because
    :math:`P^{\perp}` is Hermitian and idempotent, :math:`P\otimes\bar P` is
    itself a projector and they collapse to

    .. math::

        \Bigl[\sum_q P_q^{\perp}\otimes\overline{P_q^{\perp}}\Bigr]
        \operatorname{vec}(\Sigma)=\sum_q\operatorname{vec}
        \bigl(P_q^{\perp}y_qy_q^{H}P_q^{\perp}\bigr),

    a :math:`C^2\times C^2` solve -- 16x16 for a 4-coil array.

    **Why bother.** ``m4raw_noise.py`` ships a covariance measured on 7 subjects
    from one study series and says so: "Receiver gain may differ elsewhere in
    the corpus." R2R's recorruption is exact only where ``Sigma_z == Sigma_n``,
    so every arm using it inherits that scope. This replaces the constant with a
    quantity each scan supplies about itself.

    **Accuracy.** The estimator is consistent and the error falls as
    :math:`1/\sqrt{N}`: against the committed covariance, with exact projectors,
    the maximum relative entry error is 18% at 100 pixels, 4.5% at 2,000 and
    1.1% at 32,000. A 256x256 slice offers ~65,000.

    **Its signal rejection is only as good as the projector.** ESPIRiT's
    projector is *estimated*, so a little signal survives it and that residue
    scales with amplitude. Measured on a 4-coil phantom, the error is flat at
    ~5.6% while the signal sits within an order of magnitude of the level the
    maps were estimated at, then degrades: 10% at 20x that level and 41% at 50x.
    At realistic MRI SNR this is comfortably inside the flat region, but the
    estimate is not amplitude-independent and should not be quoted as if it
    were.

    Args:
        coil_images: ``(B, C, H, W)`` complex, uncombined, in the units you want
            the covariance in.
        null_projector: ``(B, H, W, C, C)`` from ``estimate_csm_espirit(...,
            return_subspace=True)``.
        support: optional ``(B, H, W)`` bool. Outside the ESPIRiT support the
            projector is the identity and the pixel carries no rank-one
            constraint, so it contributes nothing to identifiability while still
            contributing noise -- pass the support to exclude it.
        max_condition: refuse above this condition number on the normal matrix.

    Returns:
        ``(C, C)`` complex Hermitian.

    Raises:
        UnidentifiableCovarianceError: the sampled null spaces do not span
            enough of the coil space. This is a real state, not a numerical
            nuisance: if every pixel shared one signal direction the fit would
            be blind to that direction entirely (condition number ~1e15 in that
            limit), and returning a plausible matrix would be worse than saying
            so.
    """
    vectors = _as_coil_vectors(coil_images)
    if null_projector.shape[:3] != vectors.shape[:3]:
        raise ValueError(
            f"projector {tuple(null_projector.shape)} and images "
            f"{tuple(coil_images.shape)} disagree on (B, H, W)."
        )
    n_coils = null_projector.shape[-1]
    projectors = null_projector.reshape(-1, n_coils, n_coils)
    measured = vectors.reshape(-1, n_coils, 1)
    if support is not None:
        keep = support.reshape(-1)
        projectors, measured = projectors[keep], measured[keep]
    if projectors.shape[0] < n_coils * n_coils:
        raise UnidentifiableCovarianceError(
            f"{projectors.shape[0]} pixel(s) cannot determine a {n_coils}x{n_coils} "
            f"covariance ({n_coils**2} unknowns)."
        )

    # Accumulate in double: the normal matrix sums over every pixel, and the
    # signal this projects away is orders of magnitude above the noise it keeps.
    projectors = projectors.to(torch.complex128)
    projected = (projectors @ measured.to(torch.complex128)).squeeze(-1)
    normal = torch.einsum("nij,nkl->ikjl", projectors, projectors.conj()).reshape(
        n_coils**2, n_coils**2
    )
    rhs = (projected.unsqueeze(2) @ projected.conj().unsqueeze(1)).sum(0).reshape(-1)

    singular = torch.linalg.svdvals(normal)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    if condition > max_condition:
        raise UnidentifiableCovarianceError(
            f"normal matrix condition number {condition:.3e} exceeds {max_condition:.0e}: "
            "the sampled null spaces do not span the coil space, so at least one "
            "direction of the covariance is unconstrained. This happens when the coil "
            "sensitivities are near-parallel over the scored region -- widen the "
            "support, or keep the measured constant for this scan."
        )

    estimate = torch.linalg.solve(normal, rhs).reshape(n_coils, n_coils)
    # Cosmetic, not corrective: `normal` is built from P (x) conj(P) with P
    # Hermitian, so the solve is already Hermitian to ~1e-15. Kept so the
    # returned matrix is exactly symmetric for consumers that assume it --
    # `expected_null_residual` evaluates tr(Sigma P) on that assumption.
    estimate = (estimate + estimate.conj().transpose(-2, -1)) / 2
    return estimate.to(coil_images.dtype)


def sigma_from_coil_nullspace(
    coil_images: torch.Tensor,
    maps: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""The scalar magnitude :math:`\sigma` of the noise, from the coil null space.

    :func:`estimate_covariance_from_nullspace` fits the whole matrix and needs a
    per-pixel projector, which is a ``(B, H, W, C, C)`` tensor. A consumer that
    only wants a scalar -- Robust SSDU's ``noise_std``, which is one number --
    does not need it: for an idempotent :math:`P^{\perp}`,

    .. math::

        \mathbb{E}\lVert P_q^{\perp}y_q\rVert^2
            = \operatorname{trace}(\Sigma_nP_q^{\perp})
            \approx \sigma^2(C-k),

    and :math:`\lVert P^{\perp}y\rVert^2=\lVert y\rVert^2-\lvert\langle\hat
    s,y\rangle\rvert^2`, so the projector is never formed. Cost is
    :math:`O(BCHW)` -- measured at 2-4 ms for a ``(1, 4, 256, 256)`` slice,
    against ~0.8 s for the eigendecomposition -- which is what makes it callable
    from a training step at all (non-negotiable 9).

    **Accuracy.** Against a known :math:`\Sigma_n` on synthetic 4-coil data,
    across 0.5x-4x the committed M4Raw noise level: 0.4% with exact maps, and a
    **1.5-7% HIGH** bias with ESPIRiT-estimated ones, because map error leaks a
    little signal into the measured residual. The bias direction is known and
    stable, so a consumer that must not over-inject can scale it down.

    Args:
        coil_images: ``[B, C, H, W]`` complex, in the IMAGE domain -- the rank-one
            coil model this rests on does not hold in k-space.
        maps: ``[B, C, H, W]`` complex sensitivities.
        eps: floor on the map norm before normalising.

    Returns:
        A 0-dim tensor: the magnitude :math:`\sigma`, i.e. per-component
        :math:`\sigma/\sqrt2` for a circular complex draw.
    """
    if not torch.is_complex(maps):
        raise ValueError(
            "sigma_from_coil_nullspace needs COMPLEX sensitivities, got "
            f"{maps.dtype}. A magnitude map makes the projector real, which "
            "leaves the phase half of the null space in the residual and biases "
            "sigma high by a factor this correction cannot see."
        )
    if coil_images.shape != maps.shape:
        raise ValueError(
            f"coil images {tuple(coil_images.shape)} and maps {tuple(maps.shape)} "
            "disagree; the projector is per-pixel and per-coil."
        )
    num_coils = coil_images.shape[1]
    if num_coils < 2:
        raise ValueError(
            f"sigma_from_coil_nullspace needs at least 2 coils, got {num_coils}. "
            "With one coil the null space is empty and there is no pure-noise "
            "component to measure."
        )
    norm = maps.abs().pow(2).sum(1, keepdim=True).sqrt().clamp(min=eps)
    unit = maps / norm
    projection = (unit.conj() * coil_images).sum(1)
    residual = (coil_images.abs().pow(2).sum(1) - projection.abs().pow(2)).clamp(min=0.0)
    return (residual.mean() / (num_coils - 1)).sqrt()


def sigma_from_kspace_batch(
    kspace: torch.Tensor,
    maps: torch.Tensor | None,
) -> torch.Tensor:
    r"""Per-batch magnitude :math:`\sigma`, from k-space and the coil maps.

    The batch-shaped wrapper around :func:`sigma_from_coil_nullspace`: it pairs
    real-interleaved channels, moves to the image domain through the physics
    SSOT (the rank-one coil model does not hold in k-space), and refuses the
    inputs that would make the answer quietly wrong.

    Raises rather than falling back to a declared constant. A consumer asks for
    this because the constant is what it does not trust; substituting it on a
    missing map would answer a question nobody asked (pitfall #9).
    """
    from spectramr.infrastructure.physics.fft_ops import ifft2c

    if maps is None:
        raise ValueError(
            "a self-calibrated sigma needs complex coil sensitivities in the "
            "batch, and none arrived. Declare the `espirit_sensitivity` transform "
            "under data.processing.transforms and serve the coils uncompressed "
            "(data.coils.processing_mode: none)."
        )
    if not torch.is_complex(kspace):
        channels = kspace.shape[1]
        if channels % 2 != 0:
            raise ValueError(
                f"a self-calibrated sigma needs complex or real/imag-interleaved "
                f"coil k-space, got {channels} channels. An odd count means the "
                "coils were already combined, and a combination destroys the coil "
                "vector the null space is measured in."
            )
        paired = kspace.view(kspace.shape[0], channels // 2, 2, *kspace.shape[-2:])
        kspace = torch.complex(paired[:, :, 0], paired[:, :, 1])
    return sigma_from_coil_nullspace(ifft2c(kspace), maps)
