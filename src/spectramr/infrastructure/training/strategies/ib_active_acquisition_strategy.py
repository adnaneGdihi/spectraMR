"""Information-Bottleneck active acquisition strategy — Idea 4.

Training is reconstruction over random phase-encode masks at the arm's line
budget: the strategy masks the fully sampled k-space before the ``pre_model``
adapters, so the reconstructor learns ``f_theta(M*y) -> x`` under the declared losses.
While it trains, it learns a linear-Gaussian prior over the lines from the same
k-space (:class:`~.ib_acquisition_policy.LinePrior`).

Every validation pass then rolls out the line-selection policy
``select_next_line``, which maximises

    L_IB(l) = 1/2 log det(I + Sigma_X|Y_obs a_l^T Sigma_N,l^-1 a_l) - beta · 1/2 log det(I + Sigma_N,l^-1),

at the same budget as three baselines, and grades the reconstructor on each:
``val_*`` under random masks (the training distribution, which early stopping
reads), ``val_ib_*`` under the IB policy, ``val_variance_*`` under greedy
posterior variance and ``val_equispaced_*`` under equispaced lines. beta enters
through the per-line noise term, so it reorders lines whose noise differs.
The nonlinear (Hutchinson) scorer needs a score model this strategy does not
build, so ``mode: nonlinear`` raises at construction.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, ClassVar

import torch

from spectramr.config.schemas.training.strategy_knobs_2026_08 import (
    IBActiveAcquisitionTrainingConfigSchema,
)
from spectramr.infrastructure.builders.leaf.adapter_builders import apply_pre_model
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.loop_state import resolve_loop_iteration
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.ib_acquisition_policy import (
    LinePrior,
    apply_line_mask,
    centre_mask,
    equispaced_line_mask,
    ib_rollout,
    line_budget,
    noise_signal_correlation,
    random_line_masks,
    variance_rollout,
)
from spectramr.infrastructure.training.strategies.ib_active_acquisition_nonlinear import (
    ib_score_nonlinear,
    posterior_covariance_woodbury_update,
)
from spectramr.infrastructure.training.strategies.rng_streams import rank_seeded_generator
from spectramr.models.diffusion.acquired_line_budget import acquired_lines_of, check_declared_lines
from spectramr.models.losses.computers import UnifiedReconstructionLossComputer

logger = logging.getLogger(__name__)

#: The rollouts graded at every validation pass besides the random-mask ``val_*`` pass.
POLICY_ROLLOUTS: tuple[str, ...] = ("ib", "variance", "equispaced")
#: Validation random masks reseed from ``run.seed`` plus this offset at every pass.
_VAL_SEED_OFFSET = 7919


class IBActiveAcquisitionStrategy(BaseTrainingStrategy):
    """Random-mask reconstruction training with IB policy rollouts at validation."""

    #: Advertised line-selection regimes. An unknown ``mode`` must raise
    #: rather than silently default to the linear-Gaussian path (pitfall #9).
    VALID_MODES: frozenset[str] = frozenset({"linear_gaussian", "nonlinear"})

    inline_losses: ClassVar[frozenset[str] | None] = frozenset()
    folds_image_losses: ClassVar[bool | None] = True
    forwards_loss_components: ClassVar[bool] = True
    checkpointed_streams: ClassVar[tuple[str, ...]] = ("_ib_mask_generator",)

    @classmethod
    def standalone_infer_blocker(cls, config: object) -> str | None:
        return (
            "ib_active_acquisition masks the full k-space inside its step and the policy "
            "rollout runs in validation; infer would feed the reconstructor fully sampled input."
        )

    def __init__(
        self,
        env: TrainingEnvironment,
        **kwargs: Any,
    ) -> None:
        super().__init__(env=env, **kwargs)
        self._configure_ib(self.config)
        self.loss_computer = UnifiedReconstructionLossComputer(
            config=self.config, device=self.device
        )
        self._ib_mask_generator = rank_seeded_generator(int(self.config.run.seed))

    def _configure_ib(self, config: Any) -> None:
        """Read, validate and announce the IB knobs and the phase-encode axis."""
        ib = config.training.ib_active_acquisition or IBActiveAcquisitionTrainingConfigSchema()
        self.beta = float(ib.beta)
        self.mode = str(ib.mode)
        if self.mode not in self.VALID_MODES:
            msg = (
                f"IBActiveAcquisitionStrategy: unknown mode {self.mode!r}; "
                f"expected one of {sorted(self.VALID_MODES)}."
            )
            raise ValueError(msg)
        if self.mode == "nonlinear":
            raise ValueError(
                "IBActiveAcquisitionStrategy: mode 'nonlinear' scores lines with a diffusion "
                "score model, and this strategy builds none for its validation rollout. Use "
                "mode: linear_gaussian."
            )
        self.acceleration = float(ib.acceleration)
        self.center_fraction = float(ib.center_fraction)
        acquisition = getattr(config.data, "acquisition", None)
        if acquisition is None:
            raise ValueError(
                "IBActiveAcquisitionStrategy masks phase-encode lines and needs "
                "data.acquisition.phase_encode_axis; the arm declares no data.acquisition block."
            )
        self.phase_encode_axis = int(acquisition.phase_encode_axis)
        self.acquired_pe_lines = acquisition.acquired_pe_lines
        self.acquired_lines_record: dict[str, object] = {}
        self.line_prior = LinePrior()
        self._policy_masks: dict[str, torch.Tensor] | None = None
        self._policy_key: tuple[int, int] | None = None
        self._val_generator: torch.Generator | None = None
        self._missing_prior_reported = False
        self._noise_signal_corr: float | None = None
        logger.info(
            "[ib_active_acquisition] mode=%s beta=%g acceleration=%g center_fraction=%g "
            "phase_encode_axis=%d; training masks are drawn in-strategy before pre_model.",
            self.mode,
            self.beta,
            self.acceleration,
            self.center_fraction,
            self.phase_encode_axis,
        )

    def ib_provenance(self) -> dict[str, Any]:
        """The knobs the masks and rollouts were built from, stamped on the snapshots."""
        return {
            "mode": self.mode,
            "beta": self.beta,
            "acceleration": self.acceleration,
            "center_fraction": self.center_fraction,
            "phase_encode_axis": self.phase_encode_axis,
            "acquired_pe_lines": self.acquired_lines_record,
            "train_masks": "random lines + centre band, applied before pre_model",
            "validation_rollouts": ["random", *POLICY_ROLLOUTS],
        }

    def _first_steps_extra(self, epoch: int, batch: Any = None) -> dict[str, Any]:
        extra = super()._first_steps_extra(epoch, batch=batch)
        extra["ib_acquisition"] = self.ib_provenance()
        return extra

    def _plane_dims(self, kspace: torch.Tensor) -> tuple[int, int]:
        """``(phase-encode dim, readout dim)`` of a ``[B, C, H, W]`` or ``[B, C, H, W, D]`` batch."""
        if kspace.ndim not in (4, 5):
            raise ValueError(
                f"IBActiveAcquisitionStrategy expects a [B, C, H, W(, D)] k-space batch; got "
                f"shape {tuple(kspace.shape)}."
            )
        return (3, 2) if self.phase_encode_axis == -1 else (2, 3)

    def _acquired_lines(self, kspace: torch.Tensor, split: str) -> torch.Tensor:
        """``[B, W]`` acquired phase-encode lines; checked against the declaration once per split."""
        plane_last = kspace.movedim(-1, 1) if kspace.ndim == 5 else kspace
        acquired = acquired_lines_of(plane_last, self.phase_encode_axis)
        if split not in self.acquired_lines_record:
            self.acquired_lines_record[split] = check_declared_lines(
                self.acquired_pe_lines, acquired, None
            )
        return acquired

    def strategy_state_dict(self) -> dict[str, Any]:
        state = super().strategy_state_dict()
        state["ib_line_prior"] = self.line_prior.state_dict()
        return state

    def load_strategy_state_dict(
        self, state: dict[str, Any] | None, *, load_optimizer: bool = True
    ) -> None:
        super().load_strategy_state_dict(state, load_optimizer=load_optimizer)
        if state and state.get("ib_line_prior") is not None:
            self.line_prior.load_state_dict(state["ib_line_prior"], self.device)

    def _prepare_model_input(self, data: Any) -> torch.Tensor:
        """Learn the line prior from the full k-space, then mask it to the training budget."""
        if not isinstance(data, torch.Tensor):
            raise TypeError(
                "IBActiveAcquisitionStrategy masks the raw k-space tensor; got "
                f"{type(data).__name__}."
            )
        pe_dim, ro_dim = self._plane_dims(data)
        self.line_prior.update(data, pe_dim, ro_dim)
        width = data.shape[pe_dim]
        total, centre = line_budget(width, self.acceleration, self.center_fraction)
        support = self._acquired_lines(data, "train")
        masks = random_line_masks(
            data.shape[0], width, total, centre, self._ib_mask_generator, support
        )
        masked = apply_line_mask(data, masks, pe_dim)
        return super()._prepare_model_input(masked)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """The declared reconstruction losses on the masked-input prediction."""
        pred = self.generator_model(input_batch)
        env_losses = (self.env.losses or {}) if self.env else {}
        loss_output = self.loss_computer.compute(
            pred=pred,
            target=target_batch,
            epoch=epoch,
            iteration=resolve_loop_iteration(self),
            losses_dict=env_losses,
        )
        out: dict[str, torch.Tensor] = {"g_total_loss": loss_output.total}
        out.update(loss_output.components)
        return out

    # ---- validation rollouts ----------------------------------------- #

    def on_validation_start(self) -> None:
        super().on_validation_start()
        self._policy_masks = None
        self._val_generator = None

    def _rollout_masks(self, width: int) -> dict[str, torch.Tensor]:
        """The prior-driven policy masks; rebuilt whenever the prior or the width changes."""
        key = (width, self.line_prior.rows)
        if self._policy_masks is not None and self._policy_key == key:
            return self._policy_masks
        total, centre = line_budget(width, self.acceleration, self.center_fraction)
        masks: dict[str, torch.Tensor] = {}
        if self.line_prior.seen:
            cov, noise = self.line_prior.covariance_and_noise()
            if cov.shape[-1] != width:
                raise ValueError(
                    f"IB line prior was learned on {cov.shape[-1]} phase-encode lines; the "
                    f"validation batch has {width}."
                )
            seed = centre_mask(width, centre)
            masks["ib"] = ib_rollout(cov, noise, seed, total, self.beta, self.select_next_line)
            masks["variance"] = variance_rollout(cov, noise, seed, total)
            self._noise_signal_corr = noise_signal_correlation(cov, noise)
        elif not self._missing_prior_reported:
            self._missing_prior_reported = True
            logger.warning(
                "[ib_active_acquisition] validation before any training step: the line prior "
                "is empty, so val_ib_* and val_variance_* are not reported for this pass."
            )
        self._policy_masks, self._policy_key = masks, key
        return masks

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Grade the reconstructor under random masks (``val_*``) and every policy rollout."""
        if input_batch is None or target_batch is None:
            raise ValueError(
                "IBActiveAcquisitionStrategy.validation_step needs the k-space input and the "
                "target; it masks the input itself."
            )
        pe_dim, _ = self._plane_dims(input_batch)
        width = input_batch.shape[pe_dim]
        total, centre = line_budget(width, self.acceleration, self.center_fraction)
        if self._val_generator is None:
            self._val_generator = torch.Generator().manual_seed(
                int(self.config.run.seed) + _VAL_SEED_OFFSET
            )
        support = self._acquired_lines(input_batch, "val")
        random_masks = random_line_masks(
            input_batch.shape[0], width, total, centre, self._val_generator, support
        )
        rollouts = dict(self._rollout_masks(width))
        rollouts["equispaced"] = equispaced_line_mask(width, total, centre, support.all(0))

        def grade(mask: torch.Tensor) -> dict[str, float]:
            # The inherited metric pass skips pre_model; apply it so validation sees the training input.
            inp, tgt = apply_pre_model(
                self.adapter_chains, apply_line_mask(input_batch, mask, pe_dim), target_batch
            )
            return dict(super(IBActiveAcquisitionStrategy, self).validation_step(inp, tgt, **kwargs))

        metrics = {f"val_{k.removeprefix('val_')}": v for k, v in grade(random_masks).items()}
        for policy, mask in rollouts.items():
            metrics.update(_prefixed(grade(mask), policy))
        reference = metrics.get("val_psnr")
        for policy in POLICY_ROLLOUTS:
            value = metrics.get(f"val_{policy}_psnr")
            if value is not None and reference is not None:
                metrics[f"val_{policy}_psnr_gain"] = value - reference
        if self._noise_signal_corr is not None:
            # Checks the noise-dominated outer-band assumption that beta's per-line cost rests on.
            metrics["val_ib_noise_signal_corr"] = self._noise_signal_corr
        return metrics

    # ---- IB policy -------------------------------------------------- #

    @staticmethod
    def select_next_line(
        observed_mask: torch.Tensor,
        current_estimate: torch.Tensor,
        *,
        posterior_cov_inv: torch.Tensor | None = None,
        noise_var: float | torch.Tensor = 1.0,
        beta: float = 1.0,
        mode: str = "linear_gaussian",
        score_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> int:
        """Score every unobserved line and return the argmax.

        This is a *fast* implementation of the linear-Gaussian closed form
        L_IB(ℓ) = ½ log det(I + Σ_X|Y_obs A_ℓᵀ Σ_N,ℓ⁻¹ A_ℓ)
                − β · ½ log det(I + Σ_N,ℓ⁻¹).

        Args:
            observed_mask: 1-D bool tensor of shape [H] marking acquired
                phase-encode lines.
            current_estimate: Last reconstruction (its last dimension sizes
                the default identity prior in the linear-Gaussian path).
            posterior_cov_inv: Posterior precision Σ_X|Y_obs⁻¹ for the
                current observed state. If None, defaults to the
                identity — the score then reduces to a no-prior baseline.
            noise_var: Noise variance sigma_N^2: a scalar, or one value per line
                (``[H]``), which makes the β term differ between lines.
            beta: IB tradeoff.

        Returns:
            Integer index ``ℓ★`` of the unobserved line to acquire.
        """
        unobserved_mask = ~observed_mask.bool()
        unobserved = unobserved_mask.nonzero(as_tuple=False).squeeze(-1)
        if unobserved.numel() == 0:
            return 0
        d = (
            current_estimate.shape[-1] if posterior_cov_inv is None else posterior_cov_inv.shape[-1]
        )
        # Default to identity prior for tractability.
        if posterior_cov_inv is None:
            posterior_cov_inv = torch.eye(d, device=current_estimate.device)

        # No silent fallthrough: an unknown mode must raise here rather than
        # quietly running the linear-Gaussian branch (pitfall #9).
        if mode not in IBActiveAcquisitionStrategy.VALID_MODES:
            msg = (
                f"select_next_line: unknown mode {mode!r}; "
                f"expected one of {sorted(IBActiveAcquisitionStrategy.VALID_MODES)}."
            )
            raise ValueError(msg)

        if mode == "nonlinear":
            if score_fn is None:
                raise ValueError(
                    "IB nonlinear-mode line selection requires a callable score_fn "
                    "(typically the diffusion model's ∇log p_σ(x))."
                )
            # Solve sigma_inv x = I to obtain the posterior covariance
            # diagonal; for diagonal-dominant precision matrices the linear
            # solve is well-conditioned.
            try:
                sigma_post = torch.linalg.solve(
                    posterior_cov_inv,
                    torch.eye(d, device=posterior_cov_inv.device),
                )
            except RuntimeError as exc:
                logger.warning(
                    "IB nonlinear posterior solve failed (%s); falling back to "
                    "identity posterior covariance — the IB score loses its "
                    "prior signal for this step.",
                    exc,
                )
                sigma_post = torch.eye(d, device=posterior_cov_inv.device)
            scores = ib_score_nonlinear(
                score_fn=score_fn,
                current_estimate=current_estimate,
                candidate_lines=unobserved_mask,
                sigma_post=sigma_post,
                noise_var=noise_var,
                beta=beta,
            )
            best_idx = int(unobserved[scores.argmax()].item())
            return best_idx

        # ---- linear-Gaussian path (default) ---------------------- #
        # Diagonal of Σ_post is needed for the per-line determinant
        # update. We get it by one linear solve (Woodbury-equivalent on
        # the dense form); for very large d the caller should pass a
        # pre-computed factorisation via ``posterior_cov_inv``.
        try:
            diag = torch.linalg.solve(
                posterior_cov_inv,
                torch.eye(d, dtype=posterior_cov_inv.dtype, device=posterior_cov_inv.device),
            ).diagonal()
        except RuntimeError as exc:
            logger.warning(
                "IB linear-Gaussian posterior solve failed (%s); falling back "
                "to a no-prior identity diagonal — the IB score loses its prior "
                "signal for this step.",
                exc,
            )
            diag = torch.ones(d, device=current_estimate.device)
        noise = torch.as_tensor(noise_var, dtype=diag.dtype, device=diag.device)
        log_det = 0.5 * torch.log(torch.clamp(1.0 + diag / (noise + 1e-12), min=1e-12))
        # beta-weighted line cost 1/2 log(1 + 1/sigma_N,l^2): constant unless the noise varies per line.
        cost = beta * 0.5 * torch.log1p(1.0 / (noise + 1e-12))
        if cost.ndim:
            cost = cost[unobserved]
        scores = log_det[unobserved] - cost
        best_idx = int(unobserved[scores.argmax()].item())
        return best_idx

    @staticmethod
    def update_posterior_precision(
        sigma_inv: torch.Tensor,
        A_ell: torch.Tensor,
        noise_var: float = 1.0,
    ) -> torch.Tensor:
        """Convenience wrapper around :func:`posterior_covariance_woodbury_update`."""
        return posterior_covariance_woodbury_update(sigma_inv, A_ell, noise_var)


def _prefixed(metrics: dict[str, float], policy: str) -> dict[str, float]:
    """``<m>`` or ``val_<m>`` -> ``val_<policy>_<m>``."""
    return {f"val_{policy}_{k.removeprefix('val_')}": v for k, v in metrics.items()}


__all__ = ["POLICY_ROLLOUTS", "IBActiveAcquisitionStrategy"]
