"""Measured M4Raw thermal-noise model and the R2R recorruption it feeds.

Recorrupted-to-Recorrupted (Pang et al., CVPR 2021) trains a denoiser from a
SINGLE noisy acquisition ``y = x + n`` by splitting it into two halves that are
statistically independent given the clean image::

    input  = y + alpha * z
    target = y - z / alpha            z ~ CN(0, Sigma_n), fresh per sample

    Cov(input, target) = Sigma_n - (alpha / alpha) * Sigma_n = 0

Jointly Gaussian and uncorrelated implies independent, so ``E[target | input]``
is ``x`` and the L1/L2 minimiser is the CLEAN image -- from one excitation and
no clean reference.

The construction is exact only when ``Sigma_z == Sigma_n``. That is why the
covariance below is *measured* rather than assumed, and why it is stored as the
quantity that was actually observed (a difference of two repetitions) with the
conversion to a single repetition spelled out rather than folded in.

Scope, stated honestly: the constant was measured on 7 subjects from one M4Raw
study series (20220610xx). Receiver gain may differ elsewhere in the corpus.
Re-measure with ``estimate_covariance_from_repetitions`` before extending any
claim beyond that series.
"""

from __future__ import annotations

import torch

#: Coil count this covariance describes. M4Raw is a 4-channel head coil.
M4RAW_NUM_COILS = 4

# Pooled Hermitian covariance E[d d^H] of the PHASE-ALIGNED difference of two
# repetitions, d = rep_j * exp(-i*phi) - rep_0, measured over 32 repetition
# pairs (13 T1 + 12 T2 + 7 FLAIR, 7 subjects) on the outer k-space band with the
# centre excluded. Units are raw scanner units, matching the k-space the M4Raw
# dataset serves before any normalization.
#
# Three properties of this matrix are load-bearing and were each verified:
#
# * CIRCULARITY. Per coil, var(re)/var(im) in [0.9998, 1.0009] and
#   corr(re, im) in [-0.0005, 0.0002]. The noise is proper complex Gaussian, so
#   one complex covariance describes it completely.
# * CONTRAST INDEPENDENCE. Per-contrast sigmas deviate from the pooled value by
#   at most 1.0 %, below the 2.5 % within-contrast scatter -- because
#   samplingBandwidth is 31.25 kHz for all three contrasts and thermal noise is
#   sigma^2 ~ 4 k_B T R df. TR/TE/flip/ETL shape the SIGNAL, never the variance.
#   One matrix therefore serves T1, T2 and FLAIR alike.
# * COMPLEX INTER-COIL COUPLING. Max off-diagonal |rho| = 0.250 (coils 0-3),
#   carrying ~55 deg of phase. A real-only covariance would reproduce just 57 %
#   of that coupling, so the imaginary part is not decorative and a per-coil
#   independent draw would be the wrong noise model.
_DIFF_COV_REAL: tuple[tuple[float, ...], ...] = (
    (57.068, -2.463, -2.728, 8.401),
    (-2.463, 42.632, 12.897, -5.519),
    (-2.728, 12.897, 65.074, -0.415),
    (8.401, -5.519, -0.415, 60.121),
)
_DIFF_COV_IMAG: tuple[tuple[float, ...], ...] = (
    (0.000, -3.159, -2.788, 12.005),
    (3.159, 0.000, -1.055, -3.945),
    (2.788, 1.055, 0.000, 1.239),
    (-12.005, 3.945, -1.239, 0.000),
)

#: ``Sigma_diff = 2 * Sigma_1rep``.
#:
#: A single repetition has per-component variance ``sigma^2``, hence complex
#: variance ``E|n|^2 = 2 sigma^2``. The difference of two INDEPENDENT
#: repetitions doubles that: ``E|d|^2 = 4 sigma^2``. So the single-repetition
#: covariance is the measured difference covariance halved -- and the per-coil
#: per-component sigma is ``sqrt(Sigma_diff[k, k]) / 2``, which reproduces the
#: measured (3.777, 3.265, 4.033, 3.877).
#:
#: This factor is the one number in the module that cannot be got wrong
#: quietly. Halving it twice would give ``Sigma_z = Sigma_n / 2``, leaving
#: ``Cov(input, target) = Sigma_n / 2 != 0``; training would still converge, to
#: a fixed point part-way to the identity map. ``test_m4raw_noise.py`` pins it
#: against the measured per-coil sigmas AND against a closed loop (differencing
#: two sampled repetitions must reproduce ``Sigma_diff``).
_DIFF_TO_SINGLE_REP = 0.5


def measured_difference_covariance(
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """The covariance as measured: ``E[d d^H]`` for a phase-aligned rep pair."""
    real = torch.tensor(_DIFF_COV_REAL, dtype=torch.float64)
    imag = torch.tensor(_DIFF_COV_IMAG, dtype=torch.float64)
    return torch.complex(real, imag).to(device=device, dtype=dtype)


def single_repetition_covariance(
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """``Sigma_n`` for ONE repetition -- the covariance R2R must match."""
    return measured_difference_covariance(device=device, dtype=dtype) * _DIFF_TO_SINGLE_REP


def coil_sigmas(device: torch.device | str | None = None) -> torch.Tensor:
    """Per-coil, per-component sigma of one repetition (the reported numbers)."""
    diag = measured_difference_covariance(device=device, dtype=torch.complex128).diagonal()
    return (diag.real.sqrt() / 2.0).float()


class M4RawNoiseSampler:
    """Draws correlated complex Gaussian k-space noise and applies R2R.

    The draw is a Cholesky transform of standard circular complex noise:
    ``z = L w`` with ``L L^H = Sigma`` and ``E[w w^H] = I``, giving
    ``E[z z^H] = L I L^H = Sigma`` exactly, including the inter-coil coupling.

    Args:
        covariance: ``Sigma_n`` for one repetition. Defaults to the measured
            M4Raw matrix. Supply a re-measured one for a different series.
        alpha: R2R recorruption strength. Cancels from the decorrelation, so it
            trades variance between the halves rather than introducing bias.
    """

    def __init__(
        self,
        covariance: torch.Tensor | None = None,
        alpha: float = 1.0,
    ) -> None:
        if alpha <= 0.0:
            raise ValueError(f"R2R alpha must be > 0 (it divides the target), got {alpha}.")
        cov = single_repetition_covariance() if covariance is None else covariance
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError(f"covariance must be a square matrix, got shape {tuple(cov.shape)}.")
        if not torch.is_complex(cov):
            raise TypeError(
                "covariance must be complex: the measured inter-coil coupling carries "
                "~55 deg of phase, and a real matrix silently drops 43 % of its magnitude."
            )
        self.alpha = float(alpha)
        self._cov = cov
        # Cholesky in float64 -- the matrix is near-singular in no direction, but
        # the factor is computed once and reused for every sample, so precision
        # here is free and a wrong factor is undetectable downstream.
        self._chol = torch.linalg.cholesky(cov.to(torch.complex128))

    @property
    def num_coils(self) -> int:
        """Coil count this sampler's covariance describes."""
        return int(self._cov.shape[0])

    def sample(
        self,
        shape: tuple[int, ...],
        support_mask: torch.Tensor,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.complex64,
    ) -> torch.Tensor:
        """Draw ``z ~ CN(0, Sigma)`` shaped ``(..., C, H, W)``, masked to support.

        Args:
            shape: Target shape. ``shape[-3]`` must equal :attr:`num_coils`.
            support_mask: Boolean mask over the SAMPLED k-space, broadcastable to
                ``shape[-2:]``. Required, never defaulted: roughly 24 % of M4Raw
                phase-encode columns are exact zeros (partial phase FOV), and
                writing noise into one fabricates a line the scanner never
                acquired.
            generator: Torch generator, for reproducible draws.
            device: Device for the returned tensor.
            dtype: Complex dtype for the returned tensor.
        """
        if len(shape) < 3:
            raise ValueError(f"shape must be at least (C, H, W), got {shape}.")
        n_coils = shape[-3]
        if n_coils != self.num_coils:
            raise ValueError(
                f"coil dimension shape[-3]={n_coils} does not match the covariance "
                f"({self.num_coils} coils). The R2R identity Sigma_z = Sigma_n holds "
                "per coil; mismatched coils mean the wrong noise model, not a "
                "broadcastable one. Serve the coils uncombined "
                "(data.coils.processing_mode: none)."
            )
        if support_mask.dtype != torch.bool:
            raise TypeError(f"support_mask must be a bool tensor, got {support_mask.dtype}.")

        # Draw standard circular complex noise with E|w|^2 = 1, coil-last so the
        # Cholesky mixes along the coil axis.
        lead = tuple(shape[:-3]) + tuple(shape[-2:])
        w_shape = (*lead, n_coils)
        real = torch.randn(w_shape, generator=generator, device=device, dtype=torch.float64)
        imag = torch.randn(w_shape, generator=generator, device=device, dtype=torch.float64)
        w = torch.complex(real, imag) / (2.0**0.5)

        # z_i = sum_j L_ij w_j  ==  w @ L^T for coil-last row vectors.
        chol = self._chol.to(device=w.device)
        z = w @ chol.transpose(-1, -2)

        # (..., H, W, C) -> (..., C, H, W)
        z = z.movedim(-1, -3).to(dtype)
        return z * support_mask.to(z.device)

    def recorrupt(
        self,
        kspace: torch.Tensor,
        support_mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split one noisy acquisition into an R2R ``(input, target)`` pair.

        Returns ``(y + alpha*z, y - z/alpha)``. The halves are uncorrelated by
        construction, so an L1/L2 loss between the network's output on the first
        and the second is minimised by the clean image.

        NOTE: this is a second-moment identity. A per-acquisition artefact (an
        interference spike, a gradient glitch) sits in ``y`` and therefore in
        BOTH halves with the SAME sign, so the minimiser becomes ``x + s`` and
        the network is trained to keep it. Only genuinely independent
        acquisitions decorrelate that -- see ``data.target_mode: rep_pair``.
        """
        if not torch.is_complex(kspace):
            raise TypeError(
                f"recorrupt expects complex k-space, got {kspace.dtype}. The noise model "
                "is circular complex; splitting real/imag channels first would apply it "
                "to half the signal."
            )
        z = self.sample(
            tuple(kspace.shape),
            support_mask=support_mask,
            generator=generator,
            device=kspace.device,
            dtype=kspace.dtype,
        )
        return kspace + self.alpha * z, kspace - z / self.alpha


def estimate_covariance_from_repetitions(
    rep_a: torch.Tensor,
    rep_b: torch.Tensor,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    """Re-measure ``Sigma_n`` from a phase-aligned repetition pair.

    Both repetitions carry the same signal, so their difference is pure noise
    once a single global phase offset (B0/frequency drift between excitations,
    which permanent low-field magnets show) is removed. The result is halved to
    undo the variance doubling of the difference.

    Args:
        rep_a: Complex k-space ``(..., C, H, W)``.
        rep_b: A second repetition of the same slice, same shape.
        support_mask: Sampled-support mask, broadcastable to ``(H, W)``.

    Returns:
        The ``(C, C)`` complex Hermitian single-repetition covariance.
    """
    if rep_a.shape != rep_b.shape:
        raise ValueError(f"repetition shapes differ: {tuple(rep_a.shape)} vs {tuple(rep_b.shape)}")
    a = rep_a.to(torch.complex128)
    b = rep_b.to(torch.complex128)
    mask = support_mask.to(a.device)
    phi = torch.angle(torch.sum(torch.conj(a[..., mask]) * b[..., mask]))
    diff = b * torch.exp(-1j * phi) - a
    return _coil_covariance(diff, mask) * _DIFF_TO_SINGLE_REP


def _coil_covariance(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sample covariance ``E[x x^H]`` across coils, over the masked points.

    Boolean fancy-indexing collapses a different number of trailing axes for a
    1-D mask (over the phase-encode columns) than for a 2-D one (over ``H, W``),
    so the coil axis lands in a different place for each. Moving coils to the
    front FIRST makes the reduction independent of the mask's rank -- indexing
    after the move was an off-by-one that silently mixed the readout axis into
    the coil axis and returned a near-uniform, wrong matrix.
    """
    c = x.shape[-3]
    flat = x.movedim(-3, 0).reshape(c, -1, *x.shape[-2:])
    sel = flat[..., mask].reshape(c, -1)
    return sel @ sel.conj().T / sel.shape[1]


__all__ = [
    "M4RAW_NUM_COILS",
    "M4RawNoiseSampler",
    "coil_sigmas",
    "estimate_covariance_from_repetitions",
    "measured_difference_covariance",
    "single_repetition_covariance",
]
