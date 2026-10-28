"""Physics-equivariant SSL pretraining strategy — Idea 6 of audit_plan_novel.

The encoder ``f_θ`` and a small ensemble of latent operators
``g_φ^{(k)}`` are jointly trained so that, for every operator
``O_k`` in the default pool
``{fft_shift, kspace_phase_shift, rigid_rotation_small}``,

    f_θ(O_k(x)) ≈ g_φ^{(k)} ( f_θ(x) ).

The two k-space operators (``fft_shift`` / ``kspace_phase_shift``)
round-trip through the physics SSOT (``fft_ops.fft2c`` / ``ifft2c``)
rather than approximating the transform in the image domain.
A VICReg-style variance regulariser prevents encoder collapse.

This strategy reuses ``BaseTrainingStrategy``'s template-method
machinery — only ``_compute_losses_impl`` is overridden.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.distributed.owned_modules import OWNED_PARAM_GROUP
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.owned_param_groups import add_owned_param_group
from spectramr.infrastructure.training.strategies.rng_streams import rank_seeded_generator

logger = logging.getLogger(__name__)

# Operators that ``_apply_operator`` can dispatch on. Any ``operator_pool``
# entry outside this set is rejected at construction (pitfall #9).
_SUPPORTED_OPERATORS = frozenset({"fft_shift", "kspace_phase_shift", "rigid_rotation_small"})

# Latent-operator families ``__init__`` knows how to build. Only the linear
# family is implemented; an unknown value must raise, not degrade.
_SUPPORTED_LATENT_OPERATOR_TYPES = frozenset({"linear"})


class _LinearLatentOperator(nn.Module):
    """Per-operator learnable matrix g_φ^{(k)} ∈ R^{d × d}."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.matrix = nn.Parameter(torch.eye(latent_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.matrix.T


def _make_operator_pool(device: torch.device) -> list[str]:
    """Return the default operator pool.

    Each entry is a *string identifier* that ``_apply_operator`` knows
    how to dispatch on. The two k-space operators are implemented via the
    physics SSOT (``spectramr.infrastructure.physics.fft_ops``); the rigid
    rotation is an image-domain affine. Only operators that are actually
    implemented are advertised here — no silent unimplemented entries.
    """
    return ["fft_shift", "kspace_phase_shift", "rigid_rotation_small"]


def _apply_operator(
    x: torch.Tensor, op: str, *, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Apply one of the physics-pool operators to a batch.

    The default operators are deliberately cheap so the SSL pass runs
    well within Tier-2 audit budget; richer operators (random NUFFT
    trajectories, motion fields) can be plugged in later.
    """
    if op == "fft_shift":
        # Circularly shift the k-space lines (NOT the image) by a random
        # integer offset, round-tripping through the physics SSOT so the
        # operator lives in the frequency domain as advertised.
        b, _, h, w = x.shape
        sh = int(torch.randint(0, h, (1,), generator=generator).item()) if generator else h // 4
        sw = int(torch.randint(0, w, (1,), generator=generator).item()) if generator else w // 4
        k = fft2c(x)
        k = torch.roll(k, shifts=(sh, sw), dims=(-2, -1))
        out = ifft2c(k)
        return torch.real(out).to(x.dtype)
    if op == "kspace_phase_shift":
        # Multiply k-space by the linear phase ramp exp(-i·2π·(δ_h·k_h + δ_w·k_w)),
        # which by the Fourier-shift theorem translates the image by (δ_h, δ_w)
        # sub-pixel offsets. Implemented in the frequency domain via fft_ops.
        b, _, h, w = x.shape
        if generator is not None:
            delta_h = 0.05 * torch.randn(b, generator=generator).to(x.device)
            delta_w = 0.05 * torch.randn(b, generator=generator).to(x.device)
        else:
            delta_h = 0.05 * torch.randn(b, device=x.device)
            delta_w = 0.05 * torch.randn(b, device=x.device)
        kh = torch.fft.fftshift(torch.fft.fftfreq(h, device=x.device))
        kw = torch.fft.fftshift(torch.fft.fftfreq(w, device=x.device))
        # phase[b, h, w] = -2π (δ_h[b] k_h + δ_w[b] k_w)
        phase = (
            -2.0
            * torch.pi
            * (delta_h.view(b, 1, 1) * kh.view(1, h, 1) + delta_w.view(b, 1, 1) * kw.view(1, 1, w))
        )
        ramp = torch.exp(1j * phase).unsqueeze(1)  # [b, 1, h, w] complex
        k = fft2c(x)
        k = k * ramp
        out = ifft2c(k)
        return torch.real(out).to(x.dtype)
    if op == "rigid_rotation_small":
        b, _, _, _ = x.shape
        angles = 0.1 * torch.randn(b, device=x.device)  # ~6 degrees std
        theta = torch.zeros(b, 2, 3, device=x.device)
        theta[:, 0, 0] = torch.cos(angles)
        theta[:, 0, 1] = -torch.sin(angles)
        theta[:, 1, 0] = torch.sin(angles)
        theta[:, 1, 1] = torch.cos(angles)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)
    raise ValueError(f"Unknown operator: {op}")


class PhysicsEquivariantSSLStrategy(BaseTrainingStrategy):
    """SSL strategy where the encoder must commute with physics operators.

    The strategy owns one ``nn.ModuleDict`` of latent operators
    (one per pool entry) plus the existing generator (used as the
    encoder ``f_θ``). The encoder must be a registered SSL backbone
    producing a flat latent of size ``latent_dim``.
    """

    # parallel_compat: The latent operators and projection are bare groups on opt_g, refused under
    # ddp. Each rank draws its operator from a rank-seeded stream, so ranks train different
    # operators each step, and the projection is sized by the first encoder output.
    parallel_compat = frozenset({"replicated"})
    checkpointed_streams: ClassVar[tuple[str, ...]] = ("_rng",)

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        own = (("latent_ops", OWNED_PARAM_GROUP), ("_projection", OWNED_PARAM_GROUP))
        return (*super().owned_module_declarations(config), *own)

    def __init__(
        self,
        env: TrainingEnvironment,
        **kwargs: Any,
    ) -> None:
        super().__init__(env=env, **kwargs)
        # ``config.training`` uses ``extra="allow"``, so the YAML
        # ``physics_eq_ssl:`` block is parsed as a plain ``dict`` (not a
        # typed sub-model). Read it via dict keys — ``getattr(dict, ...)``
        # would silently fall through to every default (pitfall #15).
        pe = getattr(self.config.training, "physics_eq_ssl", None)
        if pe is not None and not isinstance(pe, dict):
            raise TypeError(
                "training.physics_eq_ssl must be a mapping of knobs "
                f"(latent_dim/lambda_var/gamma_var/...); got {type(pe).__name__}."
            )
        pe = pe or {}
        # extra="forbid"-equivalent typo rejection, strategy-side: an unknown
        # key (e.g. a typo'd "latent_dimm") would otherwise be silently dropped
        # by ``pe.get(..., default)`` — an unread knob (pitfall #15). Validate
        # the supplied keys against the advertised set and raise on any extra.
        # (A typed training.physics_eq_ssl schema block is the fuller fix but
        # would require mounting on TrainingStrategyConfigSchema; this guard
        # gives the same protection without the schema edit.)
        _known = {
            "latent_dim",
            "lambda_var",
            "gamma_var",
            "latent_operator_type",
            "operator_pool",
        }
        unknown = set(pe) - _known
        if unknown:
            raise ValueError(
                f"Unknown physics_eq_ssl knob(s) {sorted(unknown)}; supported: {sorted(_known)}."
            )
        self.latent_dim = int(pe.get("latent_dim", 128))
        self.lambda_var = float(pe.get("lambda_var", 0.1))
        self.gamma_var = float(pe.get("gamma_var", 1.0))
        # ``latent_operator_type`` is an advertised knob: validate it against
        # the implemented set and raise on an unknown value rather than
        # silently ignoring the config (pitfall #9 — no silent fallbacks).
        operator_type = str(pe.get("latent_operator_type", "linear"))
        if operator_type not in _SUPPORTED_LATENT_OPERATOR_TYPES:
            raise ValueError(
                f"Unknown latent_operator_type={operator_type!r}; "
                f"supported: {sorted(_SUPPORTED_LATENT_OPERATOR_TYPES)}."
            )
        self.latent_operator_type = operator_type
        # Honor a configured operator pool when present, validating every
        # entry against the dispatchable set so a typo'd operator raises at
        # construction instead of degrading to the default pool.
        configured_pool = pe.get("operator_pool")
        if configured_pool is not None:
            pool = [str(op) for op in configured_pool]
            if not pool:
                raise ValueError("physics_eq_ssl.operator_pool must be non-empty.")
            unknown = [op for op in pool if op not in _SUPPORTED_OPERATORS]
            if unknown:
                raise ValueError(
                    f"Unknown physics_eq_ssl operator(s) {unknown}; "
                    f"supported: {sorted(_SUPPORTED_OPERATORS)}."
                )
            self.operator_pool = pool
        else:
            self.operator_pool = _make_operator_pool(self.device)
        self.latent_ops = nn.ModuleDict(
            {op: _LinearLatentOperator(self.latent_dim) for op in self.operator_pool}
        ).to(self.device)
        # Register the latent operators on opt_g or they never train.
        self._register_on_opt_g(self.latent_ops)
        # The operator-draw stream: one per rank, carried through the checkpoint.
        self._rng = rank_seeded_generator(int(self.config.run.seed))

    def _register_on_opt_g(self, module: nn.Module) -> None:
        """Add ``module``'s parameters to ``opt_g`` as one new param group."""
        _opt = getattr(self.env, "opt_g", None)
        if _opt is not None:
            _existing = {p for g in _opt.param_groups for p in g["params"]}
            _new = [p for p in module.parameters() if p not in _existing]
            if _new:
                add_owned_param_group(self, _opt, {"params": _new}, owner="its latent operators")

    def _build_projection(self, in_features: int, device: torch.device) -> None:
        # The encoder's flattened output width is out_channels x H x W, so it is
        # known only once a batch of the run's spatial size has passed through.
        self._projection = nn.Linear(in_features, self.latent_dim).to(device)
        self._register_on_opt_g(self._projection)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.generator_model(x)
        if z.dim() > 2:
            z = z.flatten(start_dim=1)
        if z.shape[-1] != self.latent_dim:
            # The encoder may produce a richer latent; project to latent_dim.
            if getattr(self, "_projection", None) is None:
                self._build_projection(z.shape[-1], z.device)
            z = self._projection(z)
        return z

    def _materialize_owned_state(self, state: dict[str, Any]) -> None:
        """Rebuild a saved projection from its weight shape and register it on ``opt_g``.

        ``checkpointing.resume`` restores strategy state before optimizer state,
        so the param groups match the saved ``opt_g`` when that loads.
        """
        super()._materialize_owned_state(state)
        saved = state.get("modules", {}).get("_projection")
        if saved is not None and getattr(self, "_projection", None) is None:
            out_features, in_features = saved["weight"].shape
            if out_features != self.latent_dim:
                raise ValueError(
                    f"The checkpoint's PE-SSL projection maps to {out_features} "
                    f"features, but physics_eq_ssl.latent_dim is {self.latent_dim}."
                )
            self._build_projection(int(in_features), self.device)

    def _vicreg_variance(self, z: torch.Tensor) -> torch.Tensor:
        std = z.std(dim=0)
        return torch.mean(torch.clamp(self.gamma_var - std, min=0.0) ** 2)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Equivariance loss + VICReg variance regulariser."""
        del target_batch  # SSL — no supervised target.
        del epoch
        idx = int(torch.randint(0, len(self.operator_pool), (1,), generator=self._rng))
        op_name = self.operator_pool[idx]
        x = input_batch
        with torch.no_grad():
            x_aug = _apply_operator(x.detach(), op_name)
        z = self._encode(x)
        z_aug = self._encode(x_aug)
        z_predicted = self.latent_ops[op_name](z)
        equivariance = F.mse_loss(z_aug, z_predicted)
        variance_reg = self._vicreg_variance(z) + self._vicreg_variance(z_aug)
        total = equivariance + self.lambda_var * variance_reg
        return {
            "g_total_loss": total,
            "phys_eq_residual": equivariance.detach(),
            "phys_eq_variance_reg": variance_reg.detach(),
            "phys_eq_operator_idx": torch.tensor(float(idx), device=total.device),
        }


__all__ = ["PhysicsEquivariantSSLStrategy"]
