r"""Acquisition-stage physics losses for Hamiltonian-NODE trajectory learning.

Two losses, both registered with ``domain="physics"`` (Phase-0.5 F4: the
loss-domain enum is ``{image, kspace, complex, latent, physics, agnostic}`` —
``"trajectory"`` is **not** a valid domain, so the trajectory-physics losses are
registered as ``physics``).

* :class:`HamiltonianEnergyConservationLoss` — penalises drift of the conserved
  Hamiltonian along the integrated trajectory. With a symplectic integrator the
  energy oscillates within a bounded band but should not drift secularly; this
  loss keeps that band tight and, more importantly, supplies a differentiable
  signal that rewards trajectories the integrator represents faithfully.
* :class:`GradientHardwareComplianceLoss` — soft (ReLU-hinge) penalty that
  activates only when the gradient amplitude exceeds :math:`G_{\max}` or the
  slew rate exceeds :math:`S_{\max}`, expressed in clinical units (mT/m and
  T/m/s).

Both operate on the trajectory dict emitted by
:class:`~spectramr.models.acquisition.hamiltonian_trajectory_generator.HamiltonianTrajectoryGenerator`
``forward`` — ``{"k", "g", "slew", "energy"}`` — but also accept the individual
tensors directly so they are usable outside the strategy.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from spectramr.models.losses.registry import register_loss

__all__ = [
    "GradientHardwareComplianceLoss",
    "HamiltonianEnergyConservationLoss",
]


def _energy_from_inputs(
    prediction: Tensor | Mapping[str, Tensor],
    energy_kw: Tensor | None,
) -> Tensor:
    """Resolve a per-step energy tensor ``[B, T]`` from flexible inputs."""
    if energy_kw is not None:
        return energy_kw
    if isinstance(prediction, Mapping):
        if "energy" not in prediction:
            raise KeyError(
                "HamiltonianEnergyConservationLoss requires an 'energy' tensor; "
                "call the trajectory generator with return_energy=True, or pass "
                "energy=... explicitly."
            )
        return prediction["energy"]
    # A bare tensor is assumed to already be the per-step energy.
    return prediction


@register_loss("hamiltonian_energy_conservation", domain="physics")
class HamiltonianEnergyConservationLoss(nn.Module):
    r"""Penalise deviation of the Hamiltonian from its initial value.

    For a trajectory with per-step energy :math:`H_n` (:math:`n=0\dots T`) the
    loss is the mean-squared relative drift from the initial energy:

    .. math::

        \mathcal{L}_{\text{energy}}
        = \frac{1}{T+1}\sum_{n=0}^{T}
          \left(\frac{H_n - H_0}{\lvert H_0\rvert + \epsilon}\right)^2 .

    Args:
        epsilon: Small constant guarding the relative-drift denominator.
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(
        self,
        prediction: Tensor | Mapping[str, Tensor],
        target: object = None,
        *,
        energy: Tensor | None = None,
        **_kwargs: object,
    ) -> Tensor:
        """Compute the energy-conservation penalty.

        Args:
            prediction: Either the trajectory dict (must hold ``"energy"``) or a
                bare per-step energy tensor ``[B, T+1]``.
            target: Unused (protocol compliance).
            energy: Optional explicit per-step energy ``[B, T+1]``.

        Returns:
            Scalar loss tensor.
        """
        e = _energy_from_inputs(prediction, energy)
        if e.ndim == 1:
            e = e.unsqueeze(0)
        e0 = e[:, :1]
        rel_drift = (e - e0) / (e0.abs() + self.epsilon)
        return (rel_drift**2).mean()


@register_loss("gradient_hardware_compliance", domain="physics")
class GradientHardwareComplianceLoss(nn.Module):
    r"""Soft hinge penalty on gradient-amplitude and slew-rate violations.

    Trajectories live in normalised k-space (:math:`k\in[-0.5, 0.5]`, cycles per
    PIXEL, so :math:`k_{\max} = 1/(2\Delta x)`). To express the hardware bounds
    in physical units we convert the normalised gradient velocity
    :math:`\dot{\mathbf k}` and acceleration :math:`\ddot{\mathbf k}` to
    physical gradient amplitude / slew via

    .. math::

        \mathbf G_{\text{phys}} = \frac{N\,\dot{\mathbf k}}{\gamma\,\mathrm{FOV}},
        \qquad
        \dot{\mathbf G}_{\text{phys}} = \frac{N\,\ddot{\mathbf k}}{\gamma\,\mathrm{FOV}},

    with :math:`N` the matrix size (:math:`\mathrm{FOV}/N = \Delta x`),
    :math:`\gamma=\gamma_{\text{Hz/T}}` and FOV in metres; the result is in
    T/m and T/m/s, scaled to mT/m and T/m/s for comparison with the bounds. The
    penalty is

    .. math::

        \mathcal{L}_{\text{hw}}
        = \operatorname{mean}\big[\operatorname{ReLU}(\lVert\mathbf G\rVert - G_{\max})^2\big]
        + \operatorname{mean}\big[\operatorname{ReLU}(\lVert\dot{\mathbf G}\rVert - S_{\max})^2\big],

    i.e. exactly zero inside the feasible region and quadratic outside.

    Args:
        g_max_mT_per_m: Gradient-amplitude bound :math:`G_{\max}` (mT/m).
        s_max_T_per_m_per_s: Slew-rate bound :math:`S_{\max}` (T/m/s).
        dt_seconds: Step used for finite-difference slew when ``slew`` is not
            provided.
        gamma_hz_per_t: Gyromagnetic ratio :math:`\gamma/2\pi` (Hz/T).
        fov_m: Field of view (m) for the normalised→physical conversion.
        matrix_size: Matrix size :math:`N` of the reconstructed grid; required,
            since cycles/pixel → cycles/m needs it (#2276).
    """

    def __init__(
        self,
        g_max_mT_per_m: float = 40.0,  # noqa: N803 — unit suffix mT/m
        s_max_T_per_m_per_s: float = 200.0,  # noqa: N803 — unit suffix T/m/s
        dt_seconds: float = 4e-6,
        gamma_hz_per_t: float = 42.577478e6,
        fov_m: float = 0.22,
        *,
        matrix_size: int,
    ) -> None:
        super().__init__()
        if g_max_mT_per_m <= 0 or s_max_T_per_m_per_s <= 0:
            raise ValueError("Hardware bounds must be positive.")
        if isinstance(matrix_size, bool) or not isinstance(matrix_size, int) or matrix_size <= 0:
            raise ValueError(f"matrix_size must be a positive int; got {matrix_size!r}.")
        self.matrix_size = matrix_size
        self.g_max = float(g_max_mT_per_m)
        self.s_max = float(s_max_T_per_m_per_s)
        self.dt = float(dt_seconds)
        self.gamma = float(gamma_hz_per_t)
        self.fov = float(fov_m)

    def _unpack(self, prediction: Tensor | Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Return ``(g, slew)`` tensors ``[B, T, d]`` from flexible inputs."""
        if isinstance(prediction, Mapping):
            if "g" not in prediction:
                raise KeyError("GradientHardwareComplianceLoss needs a 'g' (gradient) tensor.")
            g = prediction["g"]
            slew = prediction.get("slew")
            if slew is None:
                slew = self._derive_slew(g)
            return g, slew
        # Bare tensor: treat as gradient waveform; derive slew by diff.
        g = prediction
        return g, self._derive_slew(g)

    def _derive_slew(self, g: Tensor) -> Tensor:
        """Finite-difference slew from a ``[B, T, d]`` gradient waveform.

        The guard is here rather than at the call sites because both of them
        reach it. Without it a ``[B, C, H, W]`` image slips through: ``dim=1``
        has length 1, ``torch.diff`` returns an EMPTY tensor, and the
        ``.mean()`` in ``forward`` is nan — reported as a loss value, with a
        zero gradient, instead of the contract violation it is. A trajectory
        waveform is rank-3 and needs at least two samples to have a slew rate
        at all (non-negotiable #3: raise, never degrade).
        """
        if g.dim() != 3:
            raise ValueError(
                f"GradientHardwareComplianceLoss expects a gradient waveform "
                f"[B, T, d]; got a {g.dim()}-D tensor of shape {tuple(g.shape)}. "
                f"This loss scores a k-space trajectory, not an image."
            )
        if g.shape[1] < 2:
            raise ValueError(
                f"Slew rate needs at least 2 time samples to finite-difference; "
                f"got T={g.shape[1]} in shape {tuple(g.shape)}. Pass an explicit "
                f"'slew' tensor for a single-sample waveform."
            )
        return torch.diff(g, dim=1) / self.dt

    def forward(
        self,
        prediction: Tensor | Mapping[str, Tensor],
        target: object = None,
        **_kwargs: object,
    ) -> Tensor:
        """Compute the hardware-compliance penalty.

        Args:
            prediction: Trajectory dict (``"g"``, optional ``"slew"``) or a bare
                gradient waveform ``[B, T, d]``.
            target: Unused (protocol compliance).

        Returns:
            Scalar loss tensor (zero inside the feasible region).
        """
        g_norm, slew = self._unpack(prediction)

        # Convert normalised k-space velocity/acceleration to physical units.
        # G_phys [T/m] = N * (dk/dt) / (gamma * FOV); x1e3 -> mT/m.
        scale = self.matrix_size / (self.gamma * self.fov)
        g_phys_mt = g_norm * scale * 1e3  # mT/m
        slew_phys = slew * scale  # T/m/s

        g_amp = torch.linalg.vector_norm(g_phys_mt, dim=-1)
        slew_amp = torch.linalg.vector_norm(slew_phys, dim=-1)

        g_violation = torch.relu(g_amp - self.g_max)
        s_violation = torch.relu(slew_amp - self.s_max)

        return (g_violation**2).mean() + (s_violation**2).mean()
