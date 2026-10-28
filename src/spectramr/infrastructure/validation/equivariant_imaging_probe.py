r"""Tier-2 mechanism-fires probe for the Equivariant Imaging paradigm (Phase A step 3).

The generic :func:`spectramr.infrastructure.validation.forward_probe.synthetic_forward_probe`
validates the model's forward/backward but does **not** exercise the EI
objective: the equivariance term is strategy-owned and needs
``context["transformed_recon"]``, which a generic loss computer never supplies.
This probe closes that gap. On a synthetic phantom, with the **configured model,
group, and acceleration mask**, it asserts the two properties that distinguish a
firing mechanism from the P1.2 inert facade (pitfall #16):

* **identifiability** — ``sensing_margin`` of the configured operator/group is
  strictly positive (the operator breaks the group symmetry; Tachella sensing
  theorems). A fully-sampled arm fails here.
* **mechanism fires** — the equivariance term (the registered
  :class:`~spectramr.models.losses.equivariant_recon_loss.EquivariantSSLReconLoss`)
  is finite, non-zero, and its gradient reaches the network — proving the
  strategy emits ``transformed_recon`` and the loss is no longer inert.

Running the *real* model as the reconstruction function (twice — once on ``y``,
once on the re-measured transformed recon) also covers the model's
forward/backward, so this probe subsumes the generic check for EI arms.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.group_actions import (
    get_group_action,
    sensing_margin,
)
from spectramr.infrastructure.validation.forward_probe import ProbeResult
from spectramr.models.losses.equivariant_recon_loss import EquivariantSSLReconLoss

__all__ = ["ei_forward_probe"]

# Cap the probe grid so the two forward passes stay cheap on CI.
_MAX_PROBE_SIDE = 32
# A genuinely-undersampled operator must expose at least this much null-space
# energy under the group, or EI has nothing to learn.
_MIN_SENSING_MARGIN = 1.0e-3


def _probe_side(config: Any) -> int:
    patch = list(
        getattr(getattr(getattr(config, "data", None), "sampling", None), "patch_size", None) or []
    )
    while len(patch) > 2 and patch[-1] == 1:
        patch = patch[:-1]
    side = patch[0] if patch and isinstance(patch[0], int) and patch[0] > 0 else 16
    return min(int(side), _MAX_PROBE_SIDE)


def _acceleration(config: Any) -> float:
    physics = getattr(config, "physics", None)
    cs = getattr(physics, "compressed_sensing", None)
    accel = getattr(config, "undersampling", None)
    for r in (getattr(cs, "acceleration_factor", None), getattr(accel, "base_acceleration", None)):
        if isinstance(r, (int, float)) and r > 0:
            return float(r)
    return 1.0


def _cartesian_mask(h: int, w: int, acceleration: float) -> torch.Tensor:
    """1-D Cartesian phase-encode mask at the given acceleration (+ small ACS)."""
    mask = torch.zeros(1, 1, h, w)
    if acceleration <= 1.0:
        mask[...] = 1.0
        return mask
    step = max(2, round(acceleration))
    mask[..., :, ::step] = 1.0
    acs = max(2, w // 16)
    c0 = w // 2 - acs // 2
    mask[..., :, c0 : c0 + acs] = 1.0
    return mask


def _rep_mask(config: Any) -> torch.Tensor | None:
    """``(1, 1, H, W)`` M1 of a declared ``complementary_rep_masking``, else ``None``.

    The arm's own mask family on its own patch grid: a 1-D stand-in would certify
    an operator a Poisson-disc or pseudo-radial arm never trains with, and the
    R fallback reads the compressed-sensing default, not the transform's R. The
    support is the whole axis; the partial-FOV band is data, not configuration.
    """
    from spectramr.data.transforms.complementary_rep_masking import draw_rep_masks
    from spectramr.data.transforms.registry import build_transform
    from spectramr.data.transforms.rep_mask_patterns import CARTESIAN_1D, draw_rep_masks_2d

    data = getattr(config, "data", None)
    entries = getattr(getattr(data, "processing", None), "transforms", None) or []
    entry = next((e for e in entries if getattr(e, "name", None) == "complementary_rep_masking"), None)
    if entry is None:
        return None
    t = build_transform("complementary_rep_masking", **entry.resolved_kwargs())
    patch = list(getattr(getattr(data, "sampling", None), "patch_size", None) or [256, 256])
    height, width = int(patch[0]), int(patch[1])
    support = torch.ones(width, dtype=torch.bool)
    kwargs = {
        "acceleration": t.acceleration,
        "center_fraction": t.center_fraction,
        "mode": "same",
        "keys": ("ei_probe", 0, t.seed),
    }
    if t.pattern == CARTESIAN_1D:
        line, _ = draw_rep_masks(support, **kwargs)
        mask = line[None, :].expand(height, width)
    else:
        mask, _, _ = draw_rep_masks_2d(t.pattern, support, height, **kwargs)
    return mask.to(torch.float32)[None, None]


def _build_model(config: Any) -> torch.nn.Module:
    """Instantiate the configured recon model (signature-filtered kwargs)."""
    import inspect

    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import get_model_class

    populate_model_registry()
    model_type = str(getattr(getattr(config, "model", None), "model_type", "") or "")
    cls = get_model_class(model_type)
    model_kwargs = dict(getattr(config.model, "model_kwargs", {}) or {})
    candidate = {
        "in_channels": getattr(config.model, "in_channels", 2),
        "out_channels": getattr(config.model, "out_channels", 2),
        **model_kwargs,
    }
    try:
        sig = inspect.signature(cls.__init__)
        accepts_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        if not accepts_var_kw:
            candidate = {k: v for k, v in candidate.items() if k in sig.parameters}
    except (TypeError, ValueError):
        pass
    return cls(**candidate)


def ei_forward_probe(config: Any, device: str = "cpu") -> ProbeResult:
    """Run the Equivariant Imaging Tier-2 mechanism-fires probe.

    Args:
        config: a loaded ``TrainingSettings`` whose ``training.equivariant_imaging``
            block is present.
        device: ``"cpu"`` (default, CI-safe) or ``"cuda"``.
    """
    t0 = time.perf_counter()

    def _fail(message: str, fix_hint: str | None = None, tb: str | None = None) -> ProbeResult:
        return ProbeResult(
            passed=False,
            category="ei_mechanism_fires",
            message=message,
            severity="error",
            yaml_keys=[
                "training.equivariant_imaging",
                "physics.compressed_sensing.acceleration_factor",
            ],
            fix_hint=fix_hint,
            device=device,
            elapsed_seconds=time.perf_counter() - t0,
            traceback=tb,
        )

    ei_cfg = getattr(getattr(config, "training", None), "equivariant_imaging", None)
    if ei_cfg is None:
        return _fail("No training.equivariant_imaging block — not an EI arm.")

    try:
        dev = torch.device(device)
        side = _probe_side(config)
        accel = _acceleration(config)
        in_ch = int(getattr(config.model, "in_channels", 2))
        group_name = str(getattr(ei_cfg, "group", "dihedral"))
        group_kwargs: dict[str, Any] = {}
        if group_name == "small_rotation":
            group_kwargs = {
                "max_angle_deg": float(getattr(ei_cfg, "max_angle_deg", 10.0)),
                "num_angles": int(getattr(ei_cfg, "num_angles", 4)),
            }
        group = get_group_action(group_name, **group_kwargs)

        rep_mask = _rep_mask(config)
        mask = _cartesian_mask(side, side, accel) if rep_mask is None else rep_mask
        mask = mask.to(dev)
        height, width = mask.shape[-2:]
        mask_c = mask.to(torch.complex64)

        def _to_complex(x_model: torch.Tensor) -> torch.Tensor:
            if in_ch == 2:
                return torch.complex(x_model[:, :1], x_model[:, 1:2])
            return torch.complex(x_model, torch.zeros_like(x_model))

        def _to_model_input(x_complex: torch.Tensor) -> torch.Tensor:
            if in_ch == 2:
                return torch.cat([x_complex.real, x_complex.imag], dim=1)
            return x_complex.abs()

        def forward_op(x_model: torch.Tensor) -> torch.Tensor:
            return mask_c * fft2c(_to_complex(x_model))

        def adjoint_to_model(k: torch.Tensor) -> torch.Tensor:
            return _to_model_input(ifft2c(mask_c * k))

        # ── (i) numeric identifiability ────────────────────────────────
        phantom = torch.randn(1, in_ch, height, width, device=dev)
        margin = sensing_margin(forward_op, adjoint_to_model, group, phantom, n_samples=4)
        if margin <= _MIN_SENSING_MARGIN:
            return _fail(
                f"sensing_margin={margin:.2e} <= {_MIN_SENSING_MARGIN:.0e}: the "
                f"operator is (near-)equivariant to group '{group_name}' (likely "
                "fully sampled) — Equivariant Imaging has no null-space to learn.",
                fix_hint="Undersample (acceleration_factor > 1) so A breaks the group symmetry.",
            )

        # ── (ii) mechanism fires on the configured model ───────────────
        model = _build_model(config).to(dev)

        def reconstruct(kspace: torch.Tensor) -> torch.Tensor:
            return model(adjoint_to_model(kspace))

        y = forward_op(phantom)
        x_hat = reconstruct(y)
        transformed = group.apply(x_hat, group.sample())
        re_measured = forward_op(transformed)
        prediction = reconstruct(re_measured)

        ei_loss = EquivariantSSLReconLoss(norm="l2")(
            prediction=prediction,
            target=torch.zeros_like(prediction),
            context={"transformed_recon": transformed},
        )
        if not torch.isfinite(ei_loss).all():
            return _fail("equivariance loss is non-finite on the synthetic batch.")
        if float(ei_loss.detach()) == 0.0:
            return _fail(
                "equivariance loss is exactly 0 — the EI mechanism did not fire "
                "(transformed_recon == prediction); the loss is inert (pitfall #16)."
            )
        model.zero_grad(set_to_none=True)
        ei_loss.backward()
        grad_sq = sum(
            float(p.grad.detach().pow(2).sum()) for p in model.parameters() if p.grad is not None
        )
        if grad_sq <= 0.0:
            return _fail(
                "equivariance loss produced no gradient on the recon network — "
                "the EI term does not train the model (dead mechanism)."
            )
    except Exception as exc:  # never raise out of a probe
        return _fail(f"EI probe raised {type(exc).__name__}: {exc}", tb=traceback.format_exc())

    return ProbeResult(
        passed=True,
        category="ei_mechanism_fires",
        message=(
            f"equivariant_imaging: group='{group_name}', sensing_margin={margin:.2e}, "
            f"equivariance fires (loss={float(ei_loss.detach()):.3e}, "
            f"grad-norm={grad_sq**0.5:.2e}) on {height}x{width}."
        ),
        severity="info",
        device=device,
        elapsed_seconds=time.perf_counter() - t0,
    )
