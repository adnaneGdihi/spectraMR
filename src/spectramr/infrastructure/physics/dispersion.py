r"""Bloembergen-Purcell-Pound relaxation dispersion (DL-BAE physics core).

A *microscopic* generative law for the field dependence of relaxation, distinct
from the empirical Bottomley power law in
:mod:`spectramr.infrastructure.physics.relaxation_priors` (which this dovetails
with as a prior). The BPP spectral density

.. math::

   J(\omega;\tau_c) = \frac{\tau_c}{1 + \omega^2\tau_c^2}

generates the relaxation rates from a field-invariant tissue latent
:math:`(a_0, c_0, \{b_k, \tau_{c,k}\})`:

.. math::

   \frac1{T_1(B_0)} &= a_0 + \sum_k b_k\bigl[J(\gamma B_0;\tau_k)
       + 4 J(2\gamma B_0;\tau_k)\bigr], \\
   \frac1{T_2(B_0)} &= c_0 + \sum_k b_k\bigl[\tfrac32 J(0;\tau_k)
       + \tfrac52 J(\gamma B_0;\tau_k) + J(2\gamma B_0;\tau_k)\bigr].

Because the latent parametrises the dispersion *law* and not its value at one
field, it is field-invariant **by construction**. For a single pool the three
constants :math:`(a_0, b, \tau_c)` are identifiable from *R1 values* at
:math:`M \ge 2P+1 = 3` distinct fields (a :math:`P`-pool model needs
:math:`M \ge 2P+1`). A magnitude stack with one image per field observes
neither rate directly: :func:`magnitude_stack_unknowns` counts what it must
determine instead.

References
----------
N. Bloembergen, E. M. Purcell, R. V. Pound, "Relaxation effects in nuclear
magnetic resonance absorption," *Phys. Rev.* 73(7):679-712, 1948.
"""

from __future__ import annotations

import torch

# Proton gyromagnetic ratio (rad s^-1 T^-1).
GAMMA_H: float = 2.6752218744e8


def bpp_spectral_density(omega: torch.Tensor, tau_c: torch.Tensor) -> torch.Tensor:
    r"""BPP spectral density :math:`J(\omega;\tau_c)=\tau_c/(1+\omega^2\tau_c^2)`.

    Args:
        omega: Angular frequency (rad/s), any broadcastable shape.
        tau_c: Rotational correlation time (s), broadcastable with ``omega``.

    Returns:
        :math:`J(\omega;\tau_c)`.
    """
    return tau_c / (1.0 + (omega * tau_c) ** 2)


def _rate_from_pools(
    b0: torch.Tensor,
    baseline: float | torch.Tensor,
    b: torch.Tensor,
    tau_c: torch.Tensor,
    weights: tuple[tuple[float, float], ...],
) -> torch.Tensor:
    """Sum ``baseline + sum_k b_k * sum_terms w * J(mult * gamma * B0; tau_k)``.

    ``weights`` is a tuple of ``(spectral_multiplier, coefficient)`` pairs: the
    multiplier scales ``gamma*B0`` inside ``J`` (0 -> the zero-frequency term),
    the coefficient weights that term.
    """
    b0 = b0.reshape(-1, 1)  # [M, 1]
    b = b.reshape(1, -1)  # [1, P]
    tau = tau_c.reshape(1, -1)  # [1, P]
    omega0 = GAMMA_H * b0  # [M, 1]
    acc = torch.zeros(b0.shape[0], b.shape[1], dtype=b0.dtype, device=b0.device)
    for multiplier, coeff in weights:
        acc = acc + coeff * bpp_spectral_density(multiplier * omega0, tau)
    rate = (b * acc).sum(dim=1)  # [M]
    return baseline + rate


def dispersion_r1(
    b0: torch.Tensor,
    *,
    a0: float | torch.Tensor,
    b: torch.Tensor,
    tau_c: torch.Tensor,
) -> torch.Tensor:
    r"""Longitudinal rate :math:`1/T_1(B_0)` from the dispersion law."""
    return _rate_from_pools(b0, a0, b, tau_c, ((1.0, 1.0), (2.0, 4.0)))


def dispersion_r2(
    b0: torch.Tensor,
    *,
    c0: float | torch.Tensor,
    b: torch.Tensor,
    tau_c: torch.Tensor,
) -> torch.Tensor:
    r"""Transverse rate :math:`1/T_2(B_0)` from the dispersion law."""
    return _rate_from_pools(b0, c0, b, tau_c, ((0.0, 1.5), (1.0, 2.5), (2.0, 1.0)))


def dispersion_t1(
    b0: torch.Tensor,
    *,
    a0: float | torch.Tensor,
    b: torch.Tensor,
    tau_c: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Longitudinal relaxation time :math:`T_1(B_0)` (seconds)."""
    return 1.0 / (dispersion_r1(b0, a0=a0, b=b, tau_c=tau_c) + eps)


def dispersion_t2(
    b0: torch.Tensor,
    *,
    c0: float | torch.Tensor,
    b: torch.Tensor,
    tau_c: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Transverse relaxation time :math:`T_2(B_0)` (seconds)."""
    return 1.0 / (dispersion_r2(b0, c0=c0, b=b, tau_c=tau_c) + eps)


def dispersion_rates_voxelwise(
    b0: torch.Tensor,
    *,
    a0: torch.Tensor,
    c0: torch.Tensor,
    b: torch.Tensor,
    tau_c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Per-voxel :math:`(R_1, R_2)` over a field grid, for the DL-BAE decoder.

    :func:`dispersion_r1` / :func:`dispersion_r2` take *one* global latent and
    return a per-field vector. DL-BAE needs the transpose of that: a latent map
    per voxel, evaluated at every field, so the decoder can render the whole
    multi-field stack in one shot.

    Args:
        b0: Field strengths (T), shape ``[M]``.
        a0: Longitudinal baseline rate per voxel, shape ``[B, 1, H, W]``.
        c0: Transverse baseline rate per voxel, shape ``[B, 1, H, W]``.
        b: Pool couplings per voxel, shape ``[B, P, H, W]``.
        tau_c: Correlation times per voxel, shape ``[B, P, H, W]``.

    Returns:
        ``(r1, r2)``, each of shape ``[B, M, H, W]``.

    Raises:
        ValueError: when ``b`` and ``tau_c`` disagree on the pool count.
    """
    if b.shape != tau_c.shape:
        raise ValueError(
            f"b and tau_c must share shape [B, P, H, W]; got {tuple(b.shape)} vs "
            f"{tuple(tau_c.shape)}."
        )
    # [1, M, 1, 1, 1] broadcasts against [B, 1, P, H, W] -> [B, M, P, H, W].
    omega0 = (GAMMA_H * b0).reshape(1, -1, 1, 1, 1)
    b_e = b.unsqueeze(1)  # [B, 1, P, H, W]
    tau_e = tau_c.unsqueeze(1)  # [B, 1, P, H, W]

    def _accumulate(weights: tuple[tuple[float, float], ...]) -> torch.Tensor:
        acc = torch.zeros_like(b_e * omega0)
        for multiplier, coeff in weights:
            acc = acc + coeff * bpp_spectral_density(multiplier * omega0, tau_e)
        return (b_e * acc).sum(dim=2)  # sum over pools -> [B, M, H, W]

    r1 = a0 + _accumulate(((1.0, 1.0), (2.0, 4.0)))
    r2 = c0 + _accumulate(((0.0, 1.5), (1.0, 2.5), (2.0, 1.0)))
    return r1, r2


def fit_single_pool_dispersion(
    b0: torch.Tensor,
    r1: torch.Tensor,
    *,
    tau_grid: torch.Tensor | None = None,
) -> tuple[float, float, float]:
    r"""Recover a single-pool :math:`(a_0, b, \tau_c)` from :math:`R_1(B_0)`.

    For each candidate :math:`\tau_c` the model is *linear* in :math:`(a_0, b)`,
    so we sweep a :math:`\tau_c` grid, solve the 2-parameter least squares at
    each, and return the best fit. Identifiability requires :math:`M\ge 3`
    distinct fields (a single pool has :math:`2P+1=3` free constants).

    Args:
        b0: Field strengths (T), shape ``[M]``, ``M >= 3``.
        r1: Measured :math:`1/T_1` at those fields, shape ``[M]``.
        tau_grid: Optional candidate correlation times (s); defaults to a
            log-spaced sweep over the physiological range.

    Returns:
        ``(a0, b, tau_c)``.

    Raises:
        ValueError: when fewer than 3 distinct fields are provided.
    """
    b0 = b0.flatten().double()
    r1 = r1.flatten().double()
    if b0.numel() < 3:
        raise ValueError(
            "Single-pool dispersion needs M >= 2P+1 = 3 distinct fields to be "
            f"identifiable; got {b0.numel()}."
        )
    if tau_grid is None:
        tau_grid = torch.logspace(-11, -6, 400, dtype=torch.float64)

    omega0 = GAMMA_H * b0  # [M]
    best = None
    for tau in tau_grid:
        # feature for b: J(gamma B0; tau) + 4 J(2 gamma B0; tau)
        feat = bpp_spectral_density(omega0, tau) + 4.0 * bpp_spectral_density(2.0 * omega0, tau)
        design = torch.stack([torch.ones_like(feat), feat], dim=1)  # [M, 2]
        sol, *_ = torch.linalg.lstsq(design, r1.unsqueeze(1))
        coeffs = sol[:2, 0]
        resid = (design @ coeffs - r1).pow(2).sum().item()
        if best is None or resid < best[0]:
            best = (resid, float(coeffs[0]), float(coeffs[1]), float(tau))

    assert best is not None
    _, a0_hat, b_hat, tau_hat = best
    return a0_hat, b_hat, tau_hat


def is_t1_monotone_in_b0(
    b0: torch.Tensor,
    *,
    a0: float | torch.Tensor,
    b: torch.Tensor,
    tau_c: torch.Tensor,
    tol: float = 1e-6,
) -> bool:
    r"""Check :math:`\partial T_1/\partial B_0 \ge 0` on the given field grid."""
    sorted_b0, _ = torch.sort(b0.flatten())
    t1 = dispersion_t1(sorted_b0, a0=a0, b=b, tau_c=tau_c)
    return bool(torch.all(t1[1:] - t1[:-1] >= -tol))


def magnitude_stack_unknowns(n_pools: int) -> int:
    r"""Per-voxel unknowns of a :math:`P`-pool latent seen through one magnitude per field.

    The latent :math:`(\rho, a_0, c_0, \{b_k, \tau_{c,k}\}_{k=1}^P)` has
    :math:`2P+3` constants, and a stack of :math:`M` magnitudes supplies :math:`M`
    numbers per voxel, so :math:`M \ge 2P+3` is necessary for T1 to be determined
    at the acquired fields (#2417). It is not sufficient: with one TE,
    :math:`(\rho, c_0) \to (\rho e^{TE\,t}, c_0 + t)` leaves every magnitude unchanged.

    Raises:
        ValueError: when ``n_pools < 1``.
    """
    n_pools = int(n_pools)
    if n_pools < 1:
        raise ValueError(f"n_pools must be >= 1; got {n_pools}.")
    return 2 * n_pools + 3


def power_law_t1_transport(
    t1_ref: torch.Tensor,
    b0_src: float | torch.Tensor,
    b0_tgt: float | torch.Tensor,
    beta: float | torch.Tensor,
) -> torch.Tensor:
    r"""Transport an estimated :math:`T_1` across field by the empirical power law.

    Implements the Rooney/Bottomley power-law field dependence used by the
    ``bloch_synth`` relaxometry arm (MICCAI MRIxFields2026, idea 2.1):

    .. math::

       T_1(B_0^{\text{tgt}}) = \hat T_1^{\text{ref}}
           \left(\frac{B_0^{\text{tgt}}}{B_0^{\text{src}}}\right)^{\beta},
       \qquad \beta \in [0.3, 0.4]\ \text{(tissue-dependent)} .

    This is the differentiable extrapolation factor whose exponent error drives
    the ``ln(B0_tgt/B0_src)|Δβ|`` extrapolation term of Proposition 3: the map is
    linear in ``ln`` of the field ratio, so the sensitivity grows only
    logarithmically in the field ratio (a 0.1→7 T step is ~5x a 3→7 T step, not
    30x). Two consistency limits hold by construction and are asserted in the
    physics tests: ``b0_tgt == b0_src`` (ratio 1) reproduces ``t1_ref`` and
    ``beta == 0`` yields field-invariant :math:`T_1`.

    Args:
        t1_ref: Estimated reference-field :math:`T_1` (any positive units,
            broadcastable), the encoder output. Kept in its input units.
        b0_src: Source field strength (Tesla), scalar or broadcastable tensor.
        b0_tgt: Target field strength (Tesla), scalar or broadcastable tensor.
        beta: Dispersion exponent (per-voxel/per-tissue, broadcastable with
            ``t1_ref``).

    Returns:
        :math:`T_1` transported to ``b0_tgt`` (same units as ``t1_ref``).

    Raises:
        ValueError: when a scalar source/target field is non-positive (the power law
            is undefined there; fail loudly rather than emit NaN, pitfall #9).
    """
    # Positivity guard WITHOUT a device sync: only Python scalars are validated here
    # (the common misuse, e.g. ref_field=0). ``bool(torch.any(...))`` on a device
    # tensor forces a per-call GPU->CPU sync, and this runs twice per training step in
    # the bloch_synth loop (CLAUDE.md perf non-negotiable). The render path already
    # clamps b0 > 0 upstream; a stray non-positive tensor entry surfaces via NaN/finite
    # guards, not a hot-loop synchronisation.
    for _v in (b0_src, b0_tgt):
        if isinstance(_v, (int, float)) and _v <= 0:
            raise ValueError(
                "power_law_t1_transport requires positive field strengths (Tesla); "
                f"got b0_src={b0_src!r}, b0_tgt={b0_tgt!r}."
            )
    b0_src_t = torch.as_tensor(b0_src, dtype=t1_ref.dtype, device=t1_ref.device)
    b0_tgt_t = torch.as_tensor(b0_tgt, dtype=t1_ref.dtype, device=t1_ref.device)
    ratio = b0_tgt_t / b0_src_t
    return t1_ref * ratio.pow(beta)


def power_law_t2_transport(
    t2_ref: torch.Tensor,
    b0_src: float | torch.Tensor,
    b0_tgt: float | torch.Tensor,
    gamma: float | torch.Tensor,
) -> torch.Tensor:
    r"""Transport an estimated :math:`T_2` across field by an empirical power law.

    .. math::

       T_2(B_0^{\text{tgt}}) = \hat T_2^{\text{ref}}
           \left(\frac{B_0^{\text{tgt}}}{B_0^{\text{src}}}\right)^{\gamma},
       \qquad \gamma \approx 0.05\ \text{(weak, tissue-dependent)} .

    **This is not** :func:`dispersion_t2`, and the two are kept apart on purpose
    (non-negotiable 17: a copy retained for a real difference is named for that
    difference). They are different models of the same physical quantity:

    - :func:`dispersion_t2` inverts the BPP rate :func:`dispersion_r2`, built from
      the spectral density :func:`bpp_spectral_density`. It is parametrised by pool
      amplitudes and correlation times and it is what the dispersion autoencoder
      estimates.
    - This function is a two-parameter empirical fit that transports an *already
      estimated* :math:`T_2` from one field to another, the transverse companion of
      :func:`power_law_t1_transport`. It has no pool structure.

    Executed on the same transport, the two disagree in magnitude while agreeing in
    sign -- both shorten :math:`T_2` toward low field, since :math:`J(\omega;\tau)`
    is maximal at :math:`\omega = 0`. The size of that disagreement depends on the
    pool parameters handed to the BPP form, so it is pinned in
    ``tests/physics/test_dispersion.py`` rather than quoted here, where nothing
    would re-measure it.

    Args:
        t2_ref: Estimated reference-field :math:`T_2` (any positive units,
            broadcastable). Kept in its input units.
        b0_src: Source field strength (Tesla), scalar or broadcastable tensor.
        b0_tgt: Target field strength (Tesla), scalar or broadcastable tensor.
        gamma: Transverse dispersion exponent, broadcastable with ``t2_ref``.

    Returns:
        :math:`T_2` transported to ``b0_tgt`` (same units as ``t2_ref``).

    Raises:
        ValueError: when a scalar source/target field is non-positive.
    """
    # Same sync-free guard as power_law_t1_transport: Python scalars only, because
    # ``bool(torch.any(...))`` on a device tensor forces a GPU->CPU sync per call
    # (non-negotiable 9).
    for _v in (b0_src, b0_tgt):
        if isinstance(_v, (int, float)) and _v <= 0:
            raise ValueError(
                "power_law_t2_transport requires positive field strengths (Tesla); "
                f"got b0_src={b0_src!r}, b0_tgt={b0_tgt!r}."
            )
    b0_src_t = torch.as_tensor(b0_src, dtype=t2_ref.dtype, device=t2_ref.device)
    b0_tgt_t = torch.as_tensor(b0_tgt, dtype=t2_ref.dtype, device=t2_ref.device)
    ratio = b0_tgt_t / b0_src_t
    return t2_ref * ratio.pow(gamma)


__all__ = [
    "GAMMA_H",
    "bpp_spectral_density",
    "dispersion_r1",
    "dispersion_r2",
    "dispersion_rates_voxelwise",
    "dispersion_t1",
    "dispersion_t2",
    "fit_single_pool_dispersion",
    "is_t1_monotone_in_b0",
    "magnitude_stack_unknowns",
    "power_law_t1_transport",
    "power_law_t2_transport",
]
