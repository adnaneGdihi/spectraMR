r"""Operator-identification training strategy (Proposal 1).

Fits the composite degradation operator

.. math::

    \mathcal D(\cdot\,;\theta) = \mathcal N_{\Sigma(u)}\circ
        \exp\!\big(\Omega(s(u))\big)

of an MRI scanner from data. Per batch the strategy:

1. runs the conditioner :math:`\psi` (``bch_operator_conditioner``) to map the
   contrast/field coordinate :math:`u=(c,\beta)` to per-mode severities
   :math:`s`, the noise PSD :math:`\rho`, and the low-rank covariance gains;
2. assembles the Magnus/BCH partial sum :math:`\Omega(s)` as a matrix-free
   :class:`~spectramr.infrastructure.physics.magnus_exponential.LinearOperator`;
3. applies the Krylov action :math:`\exp(\Omega)x` to the clean reference;
4. evaluates the structured-Gaussian NLL on paired samples and the
   pushforward-MMD on unpaired samples, combined as
   :math:`\mathcal L = \mathcal L_{\mathrm{pair}} + \lambda\,
   \mathcal L_{\mathrm{unpair}}`.

The deterministic generators and the structured covariance are the single
sources of truth under ``infrastructure/physics/``; this strategy only
orchestrates them.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch

from spectramr.infrastructure.physics.degradation_generators import build_generators
from spectramr.infrastructure.physics.magnus_exponential import (
    assemble_omega,
    expm_multiply,
)
from spectramr.infrastructure.physics.structured_covariance import radial_to_grid
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.mixins.kspace import KspaceMixin
from spectramr.models.losses.complex.structured_gaussian_nll import StructuredGaussianNLL
from spectramr.models.losses.image.pushforward_mmd import PushforwardMMD

logger = logging.getLogger(__name__)


class OperatorIdBCHTrainingStrategy(BaseTrainingStrategy, KspaceMixin):
    """Lie-algebraic BCH effective-generator identification strategy.

    Declared ``training_mode = "operator_id"``.
    """

    training_mode = "operator_id"

    # parallel_compat: Calls generator.build_lowrank outside the wrapped forward, so DDP/ZeRO
    # reduce no gradient for it and FSDP reads a shard.
    parallel_compat = frozenset()

    #: Both terms are built and weighted here (``mmd_weight`` in
    #: ``training.operator_id``); no builder module is folded (#1956).
    inline_losses: ClassVar[frozenset[str] | None] = frozenset(
        {"structured_gaussian_nll", "pushforward_mmd"}
    )
    folds_image_losses: ClassVar[bool | None] = False

    def __init__(self, env: Any = None, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        self._op_cfg = self._resolve_operator_config()
        self.nll_loss = StructuredGaussianNLL()
        self.mmd_loss = PushforwardMMD()
        # Generator basis is built lazily once the image size is known.
        self._generators: list | None = None
        self._det_modes: list[str] | None = None
        self._noise_modes: list[str] | None = None
        self._gen_shape: tuple[int, int, int] | None = None

    # ── configuration ────────────────────────────────────────────────
    def _resolve_operator_config(self):
        """Coerce ``config.training.operator_id`` to a typed ``OperatorIDConfig``."""
        from spectramr.config.schemas.training.operator_id import OperatorIDConfig

        raw = getattr(self.config.training, "operator_id", None)
        if raw is None:
            logger.info("[operator_id] no training.operator_id block found; using defaults.")
            return OperatorIDConfig()
        if isinstance(raw, OperatorIDConfig):
            return raw
        if isinstance(raw, dict):
            return OperatorIDConfig.model_validate(raw)
        # Pydantic sub-model or namespace: round-trip through a dict.
        if hasattr(raw, "model_dump"):
            return OperatorIDConfig.model_validate(raw.model_dump())
        return OperatorIDConfig.model_validate(dict(raw))

    # ── input coercion ───────────────────────────────────────────────
    @staticmethod
    def _to_complex_image(x: torch.Tensor) -> torch.Tensor:
        """Coerce ``[B, C, H, W]`` complex or ``[B, 2C, H, W]`` real-interleaved → complex."""
        if torch.is_complex(x):
            return x
        if x.dim() == 4 and x.shape[1] % 2 == 0:
            real = x[:, 0::2]
            imag = x[:, 1::2]
            return torch.complex(real.float(), imag.float())
        # Pure real single-channel: treat as zero-phase complex.
        return torch.complex(x.float(), torch.zeros_like(x.float()))

    def _ensure_generators(self, channels: int, height: int, width: int, device, dtype):
        shape = (channels, height, width)
        if self._generators is not None and self._gen_shape == shape:
            return
        gens, det, noise = build_generators(
            self._op_cfg.mode_dictionary,
            height,
            width,
            device=device,
            dtype=dtype,
        )
        if not gens:
            raise ValueError(
                "operator_id mode_dictionary has no deterministic generators; "
                "at least one non-noise mode is required to form exp(Ω)."
            )
        self._generators = gens
        self._det_modes = det
        self._noise_modes = noise
        self._gen_shape = shape

    def _conditioning(self, batch_size: int, device, **kwargs):
        """Resolve the (contrast, field) coordinate u from the batch / kwargs."""
        contrast_idx = kwargs.get("contrast_idx")
        field_coord = kwargs.get("field_coord")
        if contrast_idx is None:
            contrast_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            contrast_idx = contrast_idx.to(device).reshape(-1).long()
        if field_coord is None:
            beta = float(torch.log10(torch.tensor(0.3)))
            field_coord = torch.full((batch_size,), beta, device=device)
        else:
            field_coord = field_coord.to(device).reshape(-1).float()
        return contrast_idx, field_coord

    # ── core loss ─────────────────────────────────────────────────────
    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        r"""Compute :math:`\mathcal L = \mathcal L_{\mathrm{pair}} + \lambda \mathcal L_{\mathrm{unpair}}`.

        Args:
            input_batch: Degraded observation :math:`y` (a single repetition), the
                framework's input role and what the M4Raw route delivers there.
            target_batch: Clean reference :math:`x` (the repetition mean).
            epoch: Current epoch.
            kwargs: Optional ``contrast_idx`` / ``field_coord`` conditioning,
                and ``unpaired`` (observed degraded pool for the MMD term).
        """
        if input_batch is None or target_batch is None:
            raise ValueError("operator_id requires paired input (degraded) and target (clean).")

        # The operator maps the clean reference onto the observation; reading the
        # roles the other way fitted the inverse direction (#1773).
        x = self._to_complex_image(target_batch)
        y = self._to_complex_image(input_batch)
        b, c, h, w = x.shape
        self._ensure_generators(c, h, w, x.device, x.dtype)

        contrast_idx, field_coord = self._conditioning(b, x.device, **kwargs)
        out = self.env.generator(contrast_idx, field_coord)
        severities = out["severities"]  # [B, K]
        rho_radial = out["rho_radial"]  # [B, bins]
        gains = out["gains"]  # [B, r] complex

        # Deterministic operator action exp(Ω(s)) x.
        omega = assemble_omega(self._generators, severities, order=int(self._op_cfg.bch_order))
        x_deg = expm_multiply(omega, x, krylov_dim=int(self._op_cfg.krylov_dim))
        residual = y - x_deg

        rho_grid = radial_to_grid(rho_radial.to(torch.float32), h, w)
        u = self.env.generator.build_lowrank(gains)

        nll = self.nll_loss(residual, rho_grid, u, channels=c, height=h, width=w)

        # Unpaired pushforward-MMD: match the twin-degraded clean marginal to
        # the observed degraded marginal. Falls back to the paired degraded
        # batch when no dedicated unpaired pool is supplied.
        observed = kwargs.get("unpaired")
        observed = self._to_complex_image(observed) if observed is not None else y
        mmd = self.mmd_loss(x_deg, observed)

        lam = float(self._op_cfg.mmd_weight)
        total = nll + lam * mmd

        result = {
            "g_total_loss": total,
            "nll_loss": nll.detach(),
            "mmd_loss": mmd.detach(),
            "mean_severity": severities.mean().detach(),
        }

        # SOTA plan T3: calibrated operator posterior. When enabled, fit a Laplace
        # posterior over the per-mode severities (Hessian of the structured-Gaussian
        # NLL at the MAP, K×K) and stamp the mean posterior std into the metrics —
        # the uncertainty that distinguishes T3 from the point-estimate incumbent
        # (and from blind JSMoCo). λ₀→∞ collapses it to the point estimate.
        if getattr(self._op_cfg, "operator_posterior", "none") == "laplace":
            from spectramr.models.operator_id.operator_posterior import (
                laplace_posterior,
                operator_posterior_std,
            )

            s_map = severities.mean(dim=0).detach()  # [K]
            order = int(self._op_cfg.bch_order)
            kdim = int(self._op_cfg.krylov_dim)

            def _nll_of_severity(s_vec: torch.Tensor) -> torch.Tensor:
                s_b = s_vec.unsqueeze(0).expand(b, -1)
                omega_s = assemble_omega(self._generators, s_b, order=order)
                resid_s = y - expm_multiply(omega_s, x, krylov_dim=kdim)
                return self.nll_loss(resid_s, rho_grid, u, channels=c, height=h, width=w)

            try:
                _, cov = laplace_posterior(
                    s_map,
                    _nll_of_severity,
                    prior_precision=float(self._op_cfg.posterior_prior_precision),
                )
                result["operator_posterior_std"] = operator_posterior_std(cov).mean().detach()
            except Exception as exc:  # never fail training on the diagnostic
                logger.warning("operator posterior (Laplace) failed: %s", exc)

        return result
