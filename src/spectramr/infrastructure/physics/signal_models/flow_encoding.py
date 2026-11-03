"""Phase-contrast (PC) / 4D-flow signal model.

Phase-contrast MRI encodes velocity in the *phase* of the complex signal via a
bipolar gradient. With a chosen velocity-encoding value ``venc`` the phase is

.. math::

    \\phi = \\pi \\, v / v_{enc}

so a spin moving at ``v = venc`` accrues exactly :math:`\\pi` of phase and
``|v| > venc`` aliases (phase wraps). These are pure, differentiable tensor
ops with an exact analytic round-trip (``v -> phi -> v`` at known ``venc``),
which is what the physics tests pin.

A *referenced* scheme acquires one flow-compensated reference and one encoding
per direction; velocity is read from each encoding's phase relative to the
reference. ``venc`` may be a float or a tensor broadcastable over the direction
axis (one venc per direction); a tensor venc is validated once by its owner,
not per call, so the hot path never synchronises on it.

The gradient first moment realising a given ``venc`` is
:math:`M_1 = \\pi / (\\gamma\\, v_{enc})` with ``gamma`` the gyromagnetic ratio
in rad·s⁻¹·T⁻¹ (¹H: ``2.675e8``).
"""

from __future__ import annotations

import math

import torch

from spectramr.config.schemas.enums import Regime
from spectramr.infrastructure.physics.signal_models.registry import register_signal_model

#: Gyromagnetic ratio of ¹H, rad·s⁻¹·T⁻¹.
GAMMA_H: float = 2.6752218744e8

#: Referenced schemes: encoding 0 is the reference, encoding ``d + 1`` carries
#: direction ``d``. ``two_point`` is the through-plane 2D PC acquisition.
ENCODING_DIRECTIONS: dict[str, int] = {"four_point_reference": 3, "two_point": 1}


def _check_venc(venc: float | torch.Tensor) -> None:
    if not isinstance(venc, torch.Tensor) and venc <= 0:
        raise ValueError(f"venc must be positive, got {venc}")


def venc_on_axis(venc: float | torch.Tensor, ndim: int, axis: int) -> float | torch.Tensor:
    """Place a per-direction venc on the direction ``axis`` of an ``ndim`` tensor.

    A float or 0-d venc is returned as is. A 1-D ``[n_dir]`` venc is reshaped to
    broadcast only along ``axis``: left to plain broadcasting it would align with
    the trailing axis (frames or width) and scale the wrong velocities.
    """
    if not isinstance(venc, torch.Tensor) or venc.ndim == 0:
        return venc
    if venc.ndim != 1:
        raise ValueError(
            f"venc must be a float or a 1-D [n_dir] tensor; got shape {tuple(venc.shape)}"
        )
    shape = [1] * ndim
    shape[axis] = venc.numel()
    return venc.reshape(shape)


def encoding_directions(scheme: str) -> int:
    """Number of encoded directions of a referenced scheme; raises on an unknown one."""
    try:
        return ENCODING_DIRECTIONS[scheme]
    except KeyError:
        raise ValueError(
            f"unknown velocity-encoding scheme {scheme!r}; implemented: "
            f"{sorted(ENCODING_DIRECTIONS)}"
        ) from None


def wrap_phase(phase: torch.Tensor) -> torch.Tensor:
    """Wrap a phase into ``(-pi, pi]``, the range ``torch.angle`` returns."""
    return torch.atan2(torch.sin(phase), torch.cos(phase))


def velocity_to_phase(velocity: torch.Tensor, venc: float | torch.Tensor) -> torch.Tensor:
    """Map velocity to encoded phase: ``phi = pi * v / venc`` (radians)."""
    _check_venc(venc)
    return math.pi * velocity / venc


def phase_to_velocity(phase: torch.Tensor, venc: float | torch.Tensor) -> torch.Tensor:
    """Map encoded phase back to velocity: ``v = phi * venc / pi``."""
    _check_venc(venc)
    return phase * venc / math.pi


def venc_to_first_moment(venc: float, gamma: float = GAMMA_H) -> float:
    """Gradient first moment ``M1 = pi / (gamma * venc)`` for a target ``venc``."""
    _check_venc(venc)
    return math.pi / (gamma * venc)


@register_signal_model(
    name="phase_contrast",
    regime=Regime.FLOW,
    parameters=("velocity", "venc", "magnitude"),
    signal="flow_encoded_complex_signal",
    reference="Pelc NJ et al., Phase contrast cine MRI, Magn Reson Q 7(4):229-254, 1991",
)
def pc_forward(
    velocity: torch.Tensor,
    venc: float | torch.Tensor,
    magnitude: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Forward PC model: velocity -> complex flow-encoded signal.

    ``signal = magnitude * exp(i * pi * v / venc)``.
    """
    phase = velocity_to_phase(velocity, venc)
    mag = (
        magnitude
        if isinstance(magnitude, torch.Tensor)
        else torch.as_tensor(magnitude, dtype=phase.dtype, device=phase.device)
    )
    return mag * torch.exp(1j * phase.to(torch.complex64))


def pc_adjoint(signal: torch.Tensor, venc: float | torch.Tensor) -> torch.Tensor:
    """Inverse PC model: complex signal -> velocity (via the wrapped phase)."""
    return phase_to_velocity(torch.angle(signal), venc)


def phase_difference(x_enc: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    """Phase of ``x_enc`` relative to ``x_ref``: ``angle(x_enc * conj(x_ref))``.

    Exact and already wrapped to ``(-pi, pi]``, unlike subtracting two wrapped
    phases. Undefined (and its gradient NaN) where either magnitude is zero, so
    a training loss must weight by magnitude before this is differentiated.
    """
    if not (x_enc.is_complex() and x_ref.is_complex()):
        raise TypeError("phase_difference needs complex images; got real tensors")
    return torch.angle(x_enc * x_ref.conj())


def referenced_encode(
    velocity: torch.Tensor, venc: float | torch.Tensor, scheme: str, dim: int = -1
) -> torch.Tensor:
    """``[..., n_dir]`` velocity -> ``[..., n_dir + 1]`` phases along ``dim``.

    The reference phase is 0 and each encoding is ``pi * v_d / venc``, unwrapped:
    wrapping is what the decode undoes, so the encode keeps the true phase.
    """
    n_dir = encoding_directions(scheme)
    v = velocity.movedim(dim, -1)
    if v.shape[-1] != n_dir:
        raise ValueError(
            f"{scheme} encodes {n_dir} direction(s); got {v.shape[-1]} along dim {dim}"
        )
    ref = torch.zeros_like(v[..., :1])
    phase = velocity_to_phase(v, venc_on_axis(venc, v.ndim, -1))
    return torch.cat([ref, phase], dim=-1).movedim(-1, dim)


def referenced_decode_phases(
    phases: torch.Tensor, venc: float | torch.Tensor, scheme: str, dim: int = -1
) -> torch.Tensor:
    """``[..., n_dir + 1]`` phases -> ``[..., n_dir]`` velocity along ``dim``.

    The encoding-minus-reference difference is re-wrapped: two phases in
    ``(-pi, pi]`` differ by up to ``2 pi``, which would otherwise decode to as much
    as ``2 venc``.
    """
    n_dir = encoding_directions(scheme)
    p = phases.movedim(dim, -1)
    if p.shape[-1] != n_dir + 1:
        raise ValueError(f"{scheme} has {n_dir + 1} encodings; got {p.shape[-1]} along dim {dim}")
    delta = wrap_phase(p[..., 1:] - p[..., :1])
    return phase_to_velocity(delta, venc_on_axis(venc, p.ndim, -1)).movedim(-1, dim)


def referenced_decode_images(
    images: torch.Tensor, venc: float | torch.Tensor, scheme: str, dim: int = -1
) -> torch.Tensor:
    """Complex ``[..., n_dir + 1]`` images -> ``[..., n_dir]`` velocity along ``dim``."""
    n_dir = encoding_directions(scheme)
    x = images.movedim(dim, -1)
    if x.shape[-1] != n_dir + 1:
        raise ValueError(f"{scheme} has {n_dir + 1} encodings; got {x.shape[-1]} along dim {dim}")
    delta = phase_difference(x[..., 1:], x[..., :1])
    return phase_to_velocity(delta, venc_on_axis(venc, x.ndim, -1)).movedim(-1, dim)


def four_point_reference_encode(velocity: torch.Tensor, venc: float | torch.Tensor) -> torch.Tensor:
    """``[..., 3]`` (vx, vy, vz) -> ``[..., 4]`` phases (reference, x, y, z)."""
    return referenced_encode(velocity, venc, "four_point_reference")


def four_point_reference_decode(phases: torch.Tensor, venc: float | torch.Tensor) -> torch.Tensor:
    """``[..., 4]`` phases (reference, x, y, z) -> ``[..., 3]`` velocity, re-wrapped."""
    return referenced_decode_phases(phases, venc, "four_point_reference")


__all__ = [
    "ENCODING_DIRECTIONS",
    "GAMMA_H",
    "encoding_directions",
    "four_point_reference_decode",
    "four_point_reference_encode",
    "pc_adjoint",
    "pc_forward",
    "phase_difference",
    "phase_to_velocity",
    "referenced_decode_images",
    "referenced_decode_phases",
    "referenced_encode",
    "velocity_to_phase",
    "venc_on_axis",
    "venc_to_first_moment",
    "wrap_phase",
]
