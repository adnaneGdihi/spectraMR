"""Critic resolution for CycleBlochStrategy: ``losses.gan`` -> the adversarial loss spec."""

from __future__ import annotations

from typing import Any, NamedTuple

#: losses.gan.gan_loss_type -> (registered adversarial loss, D-loss scale, uses WGAN-GP).
#: The 0.5 keeps lsgan identical to the historical 0.5 * (mse(real, 1) + mse(fake, 0)).
_CRITICS: dict[str, tuple[str, float, bool]] = {
    "lsgan": ("gan_lsgan", 0.5, False),
    "wgan-gp": ("gan_wgan", 1.0, True),
}


class CriticSpec(NamedTuple):
    name: str
    loss_name: str
    d_scale: float
    uses_gp: bool
    lambda_gp: float
    label_smoothing: float


def resolve_critic(gan_cfg: Any) -> CriticSpec:
    """``losses.gan`` -> critic spec; an absent block keeps the historical LSGAN critic."""
    name = str(gan_cfg.gan_loss_type).lower() if gan_cfg is not None else "lsgan"
    if name not in _CRITICS:
        raise ValueError(
            f"cycle_bloch: losses.gan.gan_loss_type={name!r} is not implemented; "
            f"supported: {sorted(_CRITICS)}."
        )
    loss_name, d_scale, uses_gp = _CRITICS[name]
    gp_declared = bool(gan_cfg.enable_gradient_penalty) if gan_cfg is not None else False
    if gp_declared != uses_gp:
        raise ValueError(
            f"cycle_bloch: gan_loss_type={name!r} requires "
            f"losses.gan.enable_gradient_penalty={uses_gp}, got {gp_declared}."
        )
    return CriticSpec(
        name=name,
        loss_name=loss_name,
        d_scale=d_scale,
        uses_gp=uses_gp,
        lambda_gp=float(gan_cfg.lambda_gp) if uses_gp else 0.0,
        label_smoothing=float(gan_cfg.label_smoothing) if gan_cfg is not None else 0.0,
    )
