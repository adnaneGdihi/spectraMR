r"""SE(3)-equivariant DC-navigator training strategy (B-3 of the VF campaign).

Trains :class:`spectramr.models.navigation.se3_equivariant_dc_navigator.SE3EquivariantDCNavigator`
to undo synthetic rigid-body motion. Mirrors the motion-meta / VF pipeline:

1. take a clean fully-sampled anatomy target;
2. synthesise a random rigid motion trajectory and corrupt the anatomy with the
   :class:`KinematicForwardOperator` (producing motion-corrupted k-space);
3. forward the corrupted image + measured k-space through the navigator;
4. evaluate the YAML-declared reconstruction losses (routed by tensor domain,
   exactly like the VF strategy) plus the SE(3) equivariance-defect *monitoring*
   loss on the navigator's motion latent.

Composes :class:`BaseTrainingStrategy` (already mixes in :class:`KspaceMixin`)
with :class:`ReconstructionMixin` and :class:`AdversarialMixin`. The adversarial
mixin is inherited so an optional discriminator can be enabled via the config;
when no discriminator is present the strategy runs as a pure reconstruction loop.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.kinematic_forward import KinematicForwardOperator
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.mixins.adversarial import AdversarialMixin
from spectramr.infrastructure.training.strategies.mixins.reconstruction import (
    ReconstructionMixin,
)
from spectramr.models.losses.registry import create_loss

logger = logging.getLogger(__name__)


class SE3EquivariantNavigatorStrategy(BaseTrainingStrategy, ReconstructionMixin, AdversarialMixin):
    """Training strategy for the SE(3)-equivariant DC-navigator.

    Attributes:
        kinematic_op: Differentiable rigid-motion forward operator.
        _loss_specs: ``(name, loss_fn, weight, domain)`` reconstruction losses.
        _eq_defect: SE(3) equivariance-defect monitoring loss (or None).
        _lambda_eq: weight on the equivariance-defect monitoring term.
    """

    # parallel_compat: Calls generator.estimate_motion_latent outside the wrapped forward, so
    # DDP/ZeRO reduce no gradient for it and FSDP reads a shard.
    parallel_compat = frozenset()

    def __init__(
        self,
        env: Any,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env=env, device=device, **kwargs)
        # No second ``self._setup_strategy_specific_components()``: the base __init__
        # already runs it, and it is not idempotent once it adds optimizer param groups.

    # ------------------------------------------------------------------
    def _setup_strategy_specific_components(self) -> None:
        """Build the kinematic operator, the losses, and read SE(3) knobs."""
        # Image size from data config (SSOT).
        _ps = self.config.data.sampling.patch_size
        _is = getattr(self.config.data, "image_size", None)
        if _ps and len(_ps) >= 2:
            img_size = (int(_ps[0]), int(_ps[1]))
        elif isinstance(_is, (list, tuple)) and len(_is) >= 2:
            img_size = (int(_is[0]), int(_is[1]))
        elif isinstance(_is, int):
            img_size = (int(_is), int(_is))
        else:
            img_size = (256, 256)

        self.kinematic_op = KinematicForwardOperator(im_size=img_size).to(self.device)

        # ── Reconstruction losses routed by tensor domain (VF convention) ──
        self._loss_specs: list[tuple[str, torch.nn.Module, float, str]] = []
        seen: set[str] = set()
        for domain in ("kspace", "image", "complex"):
            for comp in getattr(self.config.losses, f"{domain}_losses", []) or []:
                if comp.enabled and comp.weight > 0:
                    self._loss_specs.append(
                        (comp.name, create_loss(comp.name).to(self.device), comp.weight, domain)
                    )
                    seen.add(comp.name)

        # No silent L2 fallback (CLAUDE.md #9).
        if not self._loss_specs:
            raise ValueError(
                "[SE3EquivariantNavigatorStrategy] No enabled reconstruction losses "
                "found. Declare at least one entry under "
                "losses.{image,complex,kspace}_losses. Refusing to fall back to an "
                "implicit L2 — silent fallbacks are forbidden."
            )

        # ── SE(3) equivariance-defect monitoring loss ──
        nav_cfg = getattr(self.config.training, "se3_equivariant_navigator", None)
        group = getattr(nav_cfg, "group", "SE3") if nav_cfg is not None else "SE3"
        # The defect loss samples rotations in the same plane the warp uses;
        # SE3 navigator → full SO3 probe, SO3-only → SO3 probe too.
        probe_group = "SO3" if str(group) in ("SO3", "SE3") else "SO2"
        self._eq_defect = create_loss(
            "se3_equivariance_defect", group=probe_group, num_rotations=4, seed=0
        ).to(self.device)
        # SE(3)-equivariance weight lives on the se3 nav block (extra="forbid",
        # validated). The old read from losses.reconstruction was silently
        # dropped — that schema is extra="ignore" with no such field, so the
        # YAML value never reached the strategy (the objective was monitoring
        # at weight 0.0 regardless of config).
        self._lambda_eq = (
            float(getattr(nav_cfg, "lambda_se3_equivariance", 0.0)) if nav_cfg is not None else 0.0
        )

        logger.info(
            "[SE3EquivariantNavigatorStrategy] losses=%s, group=%s, lambda_eq=%.3g",
            {f"{n}@{d}": f"{w:.2f}" for n, _, w, d in self._loss_specs},
            group,
            self._lambda_eq,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _to_complex(batch: torch.Tensor) -> torch.Tensor:
        """Real-stacked ``[B, 2C, H, W]`` → complex64 ``[B, C, H, W]``."""
        if torch.is_complex(batch):
            return batch
        c = batch.shape[1]
        if c >= 2 and c % 2 == 0:
            return torch.complex(batch[:, 0::2], batch[:, 1::2])
        return batch.to(torch.complex64)

    @staticmethod
    def _complex_to_real(t: torch.Tensor) -> torch.Tensor:
        """Complex ``[B, C, H, W]`` → real-interleaved ``[B, 2C, H, W]``."""
        if not torch.is_complex(t):
            return t
        b, c, h, w = t.shape
        out = torch.empty(b, c * 2, h, w, device=t.device, dtype=torch.float32)
        out[:, 0::2] = t.real
        out[:, 1::2] = t.imag
        return out

    def _apply_domain_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate each loss on its declared tensor domain (VF convention)."""
        pred_c = self._to_complex(pred)
        target_c = self._to_complex(target)
        pred_mag = pred_c.abs()
        target_mag = target_c.abs()

        g_total = torch.zeros((), device=self.device)
        loss_dict: dict[str, torch.Tensor] = {}
        kspace_cache: tuple[torch.Tensor, torch.Tensor] | None = None

        for name, loss_fn, weight, domain in self._loss_specs:
            if domain == "complex":
                p, t = pred_c, target_c
            elif domain == "kspace":
                if kspace_cache is None:
                    kspace_cache = (fft2c(pred_c), fft2c(target_c))
                p, t = kspace_cache
            else:  # image / magnitude
                p, t = pred_mag, target_mag
            loss_val = loss_fn(p, t)
            g_total = g_total + weight * loss_val
            loss_dict[f"loss_{name}"] = loss_val.detach()
        return g_total, loss_dict

    # ------------------------------------------------------------------
    def _corrupt_and_forward(
        self,
        target_batch: torch.Tensor,
        *,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Synthesise motion, corrupt the target, and run the navigator.

        Returns ``(pred_ri, target_ri, corrupted_kspace_ri)`` all real-interleaved.
        """
        target_complex = self._to_complex(target_batch)
        # The data is k-space (``dataset_type: kspace``); the kinematic operator,
        # the loss and the cached visuals are all image-domain, so bring the
        # target into image domain once. Without this the REAL reference rendered
        # as raw |k-space| and the corruption/forward produced a black FAKE
        # (exp_p3, smoke audit 2026-06-13). Idempotent for image-domain data.
        target_complex = self._ensure_image_domain_target(target_complex)
        b = target_complex.shape[0]

        # Random rigid motion (deterministic when ``seed`` given, for val).
        # Uses a SCOPED torch.Generator (repo convention) instead of the old
        # global torch.manual_seed + CPU/CUDA RNG save/restore bracket — the
        # bracket copied the full CUDA RNG state on every seeded val call and
        # still mutated global state mid-call. The scoped generator touches
        # no global stream at all.
        gen: torch.Generator | None = None
        if seed is not None:
            if getattr(self, "_val_motion_generator", None) is None:
                self._val_motion_generator = torch.Generator(device="cpu")
            gen = self._val_motion_generator
            gen.manual_seed(seed)
        theta = self.kinematic_op.generate_random_motion(
            batch_size=b,
            max_translation=5.0,
            max_rotation=0.05,
            motion_type="smooth",
            device=self.device,
            generator=gen,
        )

        corrupted_kspace = self.kinematic_op(target_complex, theta)
        corrupted_image = ifft2c(corrupted_kspace)

        corrupted_ri = self._complex_to_real(corrupted_image)
        corrupted_ksp_ri = self._complex_to_real(corrupted_kspace)

        pred = self.generator_model(corrupted_ri, kspace_measured=corrupted_ksp_ri)
        target_ri = self._complex_to_real(target_complex)
        return pred, target_ri, corrupted_ksp_ri

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute reconstruction + equivariance-defect losses."""
        if target_batch.ndim == 5:
            mid = target_batch.shape[-1] // 2
            target_batch = target_batch[..., mid]

        pred, target_ri, corrupted_ksp_ri = self._corrupt_and_forward(target_batch)

        g_total, loss_dict = self._apply_domain_losses(pred, target_ri)

        # SE(3) equivariance-defect monitoring term on the navigator latent.
        gen = self.generator_model
        if self._eq_defect is not None and hasattr(gen, "estimate_motion_latent"):
            xi = gen.estimate_motion_latent(corrupted_ksp_ri)  # [B, L, 6]
            # The SE3 defect loss tests a *latent-space* map G: [B,L,6]->[B,L,6]
            # (it applies the adjoint action Ad_R to its single [B,L,6] argument
            # ``u``). The previous call passed ``lie_input=corrupted_ksp_ri``
            # (k-space [B,2C,H,W]), which overrode ``u`` and tripped the loss's
            # shape guard — "SE3 equivariance defect expects [B, L, 6]; got
            # (4, 2, 256, 256)" (exp_p3, dispatch 20260603_200125 task 21).
            # ``estimate_motion_latent`` (k-space->latent) is also NOT a valid
            # latent->latent ``G`` for the loss's internal ``g(u)`` calls. Score
            # the trajectory latent ``xi`` via the loss's documented contract.
            # (A full navigator *input-space* equivariance test — rotate the
            # k-space, compare Ad_R of the latent — needs a richer loss API and
            # is tracked as a follow-up.)
            eq_defect = self._eq_defect(xi)
            if self._lambda_eq > 0:
                g_total = g_total + self._lambda_eq * eq_defect
            loss_dict["loss_se3_equivariance"] = eq_defect.detach()

        loss_dict["g_total_loss"] = g_total
        return loss_dict

    # ------------------------------------------------------------------
    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validate with a fixed motion seed and report PSNR + configured metrics."""
        target = self._to_device(target_batch)
        if target.ndim == 5:
            mid = target.shape[-1] // 2
            target = target[..., mid]

        with torch.no_grad():
            pred, target_ri, _ = self._corrupt_and_forward(target, seed=1234)

            pred_mag = self._to_complex(pred).abs()
            target_mag = self._to_complex(target_ri).abs()
            mse = torch.nn.functional.mse_loss(pred_mag, target_mag)
            data_range = target_mag.max() - target_mag.min() + 1e-8
            psnr = 10.0 * torch.log10(data_range**2 / (mse + 1e-10))

            self._last_visual_pred = pred_mag.detach().cpu()
            self._last_visual_target = target_mag.detach().cpu()

        result: dict[str, float] = {"val_psnr": float(psnr)}

        computer = getattr(self, "validation_metrics_computer", None)
        if computer is not None:
            try:
                computed = computer.compute(pred_mag, target_mag)
                result.update({k: float(v) for k, v in computed.items()})
            except Exception as exc:  # log, don't mask (CLAUDE.md #9)
                logger.warning(
                    "[SE3EquivariantNavigatorStrategy] validation_metrics_computer "
                    "failed; only val_psnr available this step: %s",
                    exc,
                )
        return result


__all__ = ["SE3EquivariantNavigatorStrategy"]
