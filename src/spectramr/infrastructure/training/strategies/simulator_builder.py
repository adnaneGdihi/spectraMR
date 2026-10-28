"""Shared simulator builder for Virtual Fiducial strategies.

Provides a single function that constructs a :class:`DigitalTwinSimulator`
from ``config.physics.digital_twin`` (SSOT).  All VF strategies must use
this builder instead of hardcoding simulator parameters.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from spectramr.infrastructure.physics.digital_twin_simulator import DigitalTwinSimulator

logger = logging.getLogger(__name__)


def _resolve_image_size(data_cfg: Any) -> tuple[int, int]:
    """Extract ``(H, W)`` image dimensions from data config.

    Priority:
        1. ``data_cfg.sampling.patch_size`` (list/tuple ≥ 2 elements)
        2. ``data_cfg.image_size`` (int or list)
        3. Default ``(256, 256)``
    """
    ps = data_cfg.sampling.patch_size
    if ps and len(ps) >= 2:
        return (int(ps[0]), int(ps[1]))

    ims = getattr(data_cfg, "image_size", None)
    if ims is not None:
        if isinstance(ims, int):
            return (ims, ims)
        if hasattr(ims, "__len__") and len(ims) >= 2:
            return (int(ims[0]), int(ims[1]))

    return (256, 256)


def build_simulator_from_config(
    config: Any,
    device: torch.device,
) -> DigitalTwinSimulator:
    """Build a :class:`DigitalTwinSimulator` from ``config.physics.digital_twin``.

    This is the **single entry point** for all VF strategies.  It reads
    every parameter from the Pydantic ``DigitalTwinConfig`` schema, so
    experiments are fully controlled by YAML — no hardcoded defaults.

    Args:
        config: ``TrainingSettings`` instance (must have ``.physics.digital_twin``).
        device: Target device for the simulator buffers.

    Returns:
        Fully configured :class:`DigitalTwinSimulator` on *device*.

    Raises:
        AttributeError: If ``config.physics.digital_twin`` is missing.
        ValueError: If ``physics.digital_twin.enabled`` is ``False``. The four
            VF strategies call this unconditionally and the twin *is* their
            method, so a disabled twin is a misconfiguration, not a
            preference — the callers for which the twin is genuinely optional
            (``strategies/base.py``, ``models/diffusion/cold_diffusion.py``)
            already test the flag before calling.
    """
    dt = config.physics.digital_twin
    if not dt.enabled:
        raise ValueError(
            "physics.digital_twin.enabled is False but a Digital Twin "
            "simulator was requested. The twin is the method for VF-family "
            "strategies (virtual_fiducial, vf_admm, pma_varnet, "
            "distillation), so building one the config never asked for would "
            "make the flag decorative. Declare `physics.digital_twin.enabled: "
            "true` in the arm's YAML, or route through a strategy that does "
            "not require a twin."
        )
    img_size = _resolve_image_size(config.data)

    # SSOT config→simulator mapping lives on the simulator (also used by the
    # transversal DigitalTwinDegradation data transform).
    simulator = DigitalTwinSimulator.from_config(
        dt,
        img_size,
        field_strength_t=config.physics.field_strength,
        phase_encode_axis=getattr(config.data.acquisition, "phase_encode_axis", None),
    ).to(device)

    logger.info(
        "[build_simulator] marker=%s, motion=%s, im_size=%s on %s",
        dt.marker_type,
        dt.motion_type,
        img_size,
        device,
    )
    return simulator


def require_epi_profile(simulator: Any) -> None:
    """Raise if the twin's scanner profile is not EPI.

    The VF field score grades the twin's phase-encode B0 shift, which only EPI produces: a
    spin-warp profile makes that shift identically zero, so the score would be silently dead.
    """
    profile = getattr(simulator, "scanner_profile", None)
    if profile is not None and str(profile.sequence) != "epi":
        raise ValueError(
            "The virtual-fiducial field score needs the phase-encode B0 shift that only EPI "
            f"produces, but physics.digital_twin.acquisition declares sequence "
            f"{str(profile.sequence)!r} (profile {profile.name!r}). Declare sequence: epi, "
            "or drop the acquisition block."
        )


def undersampling_mask_kwargs(simulator: Any) -> dict[str, torch.Tensor]:
    """Generator kwargs carrying the simulator's last k-space sampling mask.

    Returns ``{"mask": <tensor>}`` when the most recent simulator forward pass
    applied undersampling (so the model's data-consistency layers can enforce
    consistency against the measured lines), or ``{}`` otherwise — in which
    case the generator is called exactly as before (no mask kwarg), so non-
    undersampling arms and mask-agnostic models are unaffected.

    The mask is drawn randomly per call inside the simulator and cannot be
    reconstructed afterwards, so it must be read straight off the simulator
    (``DigitalTwinSimulator.last_undersampling_mask``) right after its forward.
    """
    mask = getattr(simulator, "last_undersampling_mask", None)
    return {"mask": mask} if mask is not None else {}


__all__ = ["build_simulator_from_config", "require_epi_profile", "undersampling_mask_kwargs"]
