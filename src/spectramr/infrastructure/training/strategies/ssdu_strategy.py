"""Self-Supervised learning via Data Undersampling (SSDU) and Robust SSDU.

SSDU (Yaman et al., MRM 2020) splits the acquired lines into an input set Lambda
and a loss set Theta. Under ``noisier2noise_correction`` it is Robust SSDU
(arXiv:2210.01696): the input is corrupted before the split, the loss covers every
acquired line, and validation applies the Noisier2Noise correction; the estimator
and its derivation live in :mod:`...strategies.robust_ssdu`. The strategy owns the
operator: the network gets ``A^H y~``, its DC gets ``y~``, Lambda and the maps, and
the loss scores ``F S x_hat`` against the multi-coil measurement.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch

from spectramr.infrastructure.physics.coil_noise_fit import sigma_from_kspace_batch
from spectramr.infrastructure.physics.coil_sensitivity import coil_support
from spectramr.infrastructure.training.strategies.loss_folding import declared_loss_weights
from spectramr.infrastructure.physics.multicoil_ei_operator import (
    MulticoilEIOperator,
    interleaved_to_complex,
)
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.equivariant_imaging_band import batch_support
from spectramr.infrastructure.training.strategies.equivariant_imaging_ops import (
    scoring_transform,
    zero_filled_context_metrics,
    zero_filled_metrics,
)
from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)
from spectramr.infrastructure.training.strategies.rng_streams import rank_seeded_generator
from spectramr.infrastructure.training.strategies.robust_ssdu import (
    context_metric_kspace,
    image_of,
    inject_noisier_kspace,
    kspace_of,
    noisier2noise_correct,
    robust_loss_weights,
    split_acquired_mask,
)

logger = logging.getLogger(__name__)

__all__ = ["SSDUReconstructionStrategy", "inject_noisier_kspace", "split_acquired_mask"]



def _committed_coil_factor() -> torch.Tensor:
    """Cholesky factor of the committed M4Raw coil covariance, scaled to trace C.

    Only the shape is used: the level comes from sigma (declared or fitted), and
    the trace normalisation keeps the mean per-coil injected power at sigma^2.
    """
    from spectramr.infrastructure.physics.m4raw_noise import single_repetition_covariance

    cov = single_repetition_covariance(dtype=torch.complex128)
    coils = cov.shape[0]
    shape = cov * (coils / cov.diagonal().real.sum())
    return torch.linalg.cholesky(shape).to(torch.complex64)

class SSDUReconstructionStrategy(ReconstructionTrainingStrategy):
    """Reference-free reconstruction via SSDU k-space self-supervision."""

    #: Loss ownership (issue #1918). Iterates env.losses but the loop body is `if "ssdu" in name`:
    #: every declared image loss whose name lacks that substring is SKIPPED. The
    #: textual route marker cannot see the filter, so the flag -- not the pin --
    #: is what records that this strategy folds only its own ssdu terms.
    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = False
    #: The network predicts an image against a k-space target; the mixin's baseline
    #: would grade the k-space input with the arm's asymmetric transform.
    _owns_zero_filled_baseline: ClassVar[bool] = True
    _ZF_BASELINE_METRICS: ClassVar[tuple[str, ...]] = (
        *ReconstructionTrainingStrategy._ZF_BASELINE_METRICS,
        *ReconstructionTrainingStrategy._ZF_OBJECT_METRICS,
    )
    #: ``_measurement`` raises without the batch's mask and measurement.
    validates_on_batch_context: ClassVar[bool] = True
    checkpointed_streams: ClassVar[tuple[str, ...]] = ("_rng",)
    #: The network is fed ``A^H y~`` built inside the step, not the prepared k-space (#2508).
    snapshot_prepared_is_model_input: bool = False
    snapshot_model_input_tag: str | None = "ssdu_model_input"
    #: Schema default, on the class so an instance built without ``__init__`` reads it.
    score_structural_zeros: bool = False
    _split_seed: int = 0
    # The validation forward reads y from the context (``_measurement``).
    validation_reads_context_measurement = True

    def __init__(
        self,
        env: TrainingEnvironment | None = None,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env=env, device=device, **kwargs)
        # Typed SSDUTrainingConfigSchema (frozen, extra="forbid"); read the
        # validated fields directly so a renamed/missing field raises instead
        # of silently reverting to the default (pitfall #15).
        cfg = getattr(self.config.training, "ssdu", None) if self.config.training else None
        if cfg is None:
            raise ValueError(
                "SSDUReconstructionStrategy requires a training.ssdu block; the split "
                "fraction, seed and the Robust SSDU knobs have no defaults to fall back to."
            )
        self.theta_fraction = float(cfg.theta_fraction)
        self.num_masks = int(cfg.num_masks)
        if self.num_masks != 1:
            raise ValueError(
                f"ssdu.num_masks={self.num_masks}: multi-mask SSDU needs one network forward "
                "per (Lambda, Theta) split; this step runs one, so every later Theta overlaps "
                "the input Lambda and the loss rewards copying input lines. Use num_masks: 1."
            )
        seed = self._split_seed = int(cfg.split_seed)
        self.noisier2noise = bool(cfg.noisier2noise_correction)
        self.noise_std = cfg.noise_std_estimate
        self.noise_model = str(cfg.noise_model)
        self.noise_covariance = str(cfg.noise_covariance)
        self._coil_factor = _committed_coil_factor() if self.noise_covariance == "committed_m4raw" else None
        self.noise_std_source = str(cfg.noise_std_source)
        self.n_coils = int(cfg.n_coils)
        self.alpha = float(cfg.alpha)
        self.score_structural_zeros = bool(cfg.score_structural_zeros)

        # The 'robust_ssdu' key advertises the Noisier2Noise correction; refuse
        # the contradiction of selecting it with the correction off (pitfall #16).
        training_mode = str(getattr(self.config.training, "training_mode", "") or "").lower()
        if training_mode == "robust_ssdu" and not self.noisier2noise:
            raise ValueError(
                "training_mode 'robust_ssdu' requires "
                "ssdu.noisier2noise_correction=true; use 'ssdu' for vanilla SSDU."
            )

        # One stream per rank, carried through the checkpoint (rng_streams).
        self._rng = rank_seeded_generator(seed)
        logger.info(
            "SSDUReconstructionStrategy: theta_fraction=%.3g, num_masks=%d, split_seed=%d, "
            "robust=%s, alpha=%.3g, sigma_source=%s, n_coils=%d, score_structural_zeros=%s",
            *(self.theta_fraction, self.num_masks, seed, self.noisier2noise, self.alpha),
            *(self.noise_std_source, self.n_coils, self.score_structural_zeros),
        )

    def _setup_strategy_specific_components(self) -> None:
        # Accept the SSDU mode in addition to the recon-family modes.
        self._verify_strategy_config(
            expected_modes=(
                "reconstruction",
                "mae_pretraining",
                "volumetric",
                "self_supervised_reconstruction",
                "ssdu",
                "robust_ssdu",
            )
        )
        if self.logging_service:
            self._log_config_features(self.logging_service)

    def builds_own_model_input(self) -> bool:
        """The network is handed ``A^H y``, never the loader's coil channels (#2240)."""
        return True

    @classmethod
    def standalone_infer_blocker(cls, config: object) -> str | None:
        return (
            "SSDU builds A^H y from the batch's mask, measurement and coil maps inside "
            "the training step; infer has no such builder and would feed the file to the "
            "network as is"
        )

    def _measurement(self, context: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """``(y complex [B, C, H, W], acquired mask, complex maps or None)`` off the context."""
        acquired = context.get("mask")
        measured = context.get("measured_kspace")
        if acquired is None or measured is None:
            raise ValueError(
                "SSDU requires batch_context['mask'] and ['measured_kspace'] (the "
                "measurement y); it has no fully-sampled target."
            )
        y = interleaved_to_complex(measured)
        if y.shape[1] != self.n_coils:
            raise ValueError(
                f"ssdu.n_coils={self.n_coils} but the measurement carries {y.shape[1]} coils."
            )
        maps = context.get("coil_sensitivities")
        if maps is not None and not torch.is_complex(maps):
            raise ValueError("SSDU's operator needs COMPLEX coil maps (`sensitivity_complex`).")
        if maps is None and y.shape[1] > 1:
            raise ValueError(
                f"{y.shape[1]}-coil SSDU needs complex coil maps to form A = M F S: "
                "declare the `espirit_sensitivity` transform."
            )
        return y, acquired, maps

    def _network(
        self,
        context: dict[str, Any],
        kspace: torch.Tensor,
        mask: torch.Tensor,
        maps: torch.Tensor | None,
        *,
        validation: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        """``(A^H k, forward kwargs, context)`` with DC anchored to ``k`` on ``mask``."""
        bc = {**context, "measured_kspace": kspace, "mask": mask, "coil_sensitivities": maps}
        bc.update(use_dc=False, kspace_adjoint_applied=True)
        lr_image, forward_kwargs = super()._prepare_generator_inputs(
            bc, image_of(kspace, maps, mask), validation=validation
        )
        return lr_image, forward_kwargs, bc

    def _ssdu_weights(self) -> dict[str, float]:
        """Declared loss weights, resolved once (the loss-weight SSOT)."""
        weights = getattr(self, "_ssdu_weights_cache", None)
        if weights is None:
            weights = self._ssdu_weights_cache = declared_loss_weights(self.config)
        return weights

    def declared_metric_keys(self) -> frozenset[str]:
        """The ``loss_<name>`` components the step stamps, so the CSV header has their columns."""
        return frozenset(f"loss_{name}" for name in self._ssdu_weights() if "ssdu" in name)

    def _sigma(
        self, y: torch.Tensor, maps: torch.Tensor | None, acquired: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``sigma_n``: declared, or fitted per slice from the coil null space."""
        if self.noise_std_source == "self_calibrated":
            # The measurement is undersampled, so the fit is cut to the ACS band
            # and corrected by its sampled fraction (PAPER.md 6.3 of the n2n
            # cohort); the legacy whole-support read is low by sqrt(f) and high
            # from aliasing. One value per slice, broadcast over coils and pixels.
            sigma = sigma_from_kspace_batch(y, maps, support_policy="acs", acquired=acquired)
            return sigma.view(-1, *([1] * (y.dim() - 1)))
        return torch.as_tensor(float(self.noise_std), dtype=y.real.dtype, device=y.device)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        batch_context = self._prepare_batch_context(input_batch, target_batch, **kwargs)
        return self._ssdu_objective(
            batch_context, target_batch, self._rng, kwargs.get("batch"), declare_input=True
        )

    def _validation_loss(
        self,
        raw_prediction: torch.Tensor,
        target_batch: torch.Tensor,
        batch_context: dict[str, Any],
        batch_data: Any,
    ) -> torch.Tensor | None:
        """The training objective on the validation measurement: a held-out split of Omega.

        A fresh generator per batch draws the same split on every pass, so the curve
        moves only with the weights. The scored forward is the split one, not the
        full-Omega prediction the metrics grade.
        """
        if batch_context.get("measured_kspace") is None or batch_context.get("mask") is None:
            return None
        generator = torch.Generator(device=self._rng.device).manual_seed(self._split_seed)
        return self._ssdu_objective(batch_context, target_batch, generator, batch_data)["loss"]

    def _ssdu_objective(
        self,
        batch_context: dict[str, Any],
        target_batch: torch.Tensor,
        generator: torch.Generator,
        batch: Any,
        *,
        declare_input: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Split, (corrupt,) reconstruct and score: the one owner of the SSDU loss."""
        y, acquired, maps = self._measurement(batch_context)

        # ``num_masks`` independent splits; the network reconstructs from the first
        # Lambda and ``multi_mask_ssdu`` scores every Theta.
        splits = [
            split_acquired_mask(acquired, theta_fraction=self.theta_fraction, generator=generator)
            for _ in range(max(1, self.num_masks))
        ]
        lam, theta = splits[0]
        mask_dtype = y.real.dtype
        if self.noisier2noise:
            # Eq. 25: corrupt, then split. The loss covers all of Omega (Eq. 26),
            # weighted (1+a^2)/a^2 on Lambda (Eq. 35).
            noisy = inject_noisier_kspace(
                y,
                acquired,
                self.alpha * self._sigma(y, maps, acquired),
                generator=generator,
                coil_factor=self._coil_factor,
            )
            network_kspace = noisy * lam.to(mask_dtype)
            loss_mask = acquired
            loss_weights: torch.Tensor | None = robust_loss_weights(lam, acquired, self.alpha)
        else:
            network_kspace = y * lam.to(mask_dtype)
            loss_mask, loss_weights = theta, None
            if self.score_structural_zeros:
                # y is zero off the acquired band by construction, so these are scored without a reference.
                zeros = ~batch_support(batch, acquired)
                splits = [(lm, t.bool() | zeros) for lm, t in splits]
                loss_mask = splits[0][1]

        lr_image, forward_kwargs, bc = self._network(batch_context, network_kspace, lam, maps)
        if declare_input:
            self._declare_model_input(
                {"model_input_image": lr_image},
                extra={"model_input_key": "model_input_image"},
                in_kspace_keys=set(),
            )
        hr_fakes, _ = self._generate_predictions(lr_image, forward_kwargs, bc)

        env_losses: dict[str, Any] = (getattr(self.env, "losses", None) or {}) if self.env else {}
        theta_masks = [loss_mask] if self.noisier2noise else [t for (_, t) in splits]
        context = {
            "target_kspace": y,
            "theta_mask": loss_mask,
            "theta_masks": theta_masks,
            "loss_weights": loss_weights,
            "coil_sensitivities": maps,
        }
        prediction = interleaved_to_complex(hr_fakes)
        total = hr_fakes.new_zeros(())
        components: dict[str, torch.Tensor] = {}
        ran_ssdu = False
        weights = self._ssdu_weights()
        for name, loss_fn in env_losses.items():
            if "ssdu" in name:
                if name not in weights:
                    raise ValueError(
                        f"the SSDU loss {name!r} was built but the loss-weight table has "
                        f"no entry for it ({sorted(weights)})."
                    )
                # At its declared weight (non-negotiable 8); it used to be ignored.
                val = weights[name] * loss_fn(prediction, target_batch, context=context)
                total = total + val
                components[f"loss_{name}"] = val
                ran_ssdu = True
        if not ran_ssdu:
            raise ValueError(
                "SSDU strategy requires an 'ssdu' or 'multi_mask_ssdu' loss in "
                "config.objectives -- none was built. (pitfall #9: no silent fallback.)"
            )

        result: dict[str, torch.Tensor] = {"g_total_loss": total, "loss": total}
        result.update(components)
        return result

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        *,
        batch_data: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validate on the batch's own measurement, mask and maps, then add ``val_zf_*``.

        The mixin's bare context carried no mask and no measurement, so data
        consistency was silently off at validation (#2327).
        """
        context = kwargs.pop("batch_context", None)
        if context is None:
            context = self.validation_batch_context(input_batch, target_batch, batch_data)
        metrics = super().validation_step(
            input_batch, target_batch, batch_data=batch_data, batch_context=context, **kwargs
        )
        if metrics and context is not None:
            _, acquired, maps = self._measurement(context)
            operator = (
                MulticoilEIOperator(maps, acquired, where="in the SSDU validation batch")
                if maps is not None
                else None
            )
            metrics.update(
                zero_filled_metrics(
                    self.validation_metrics_computer,
                    context,
                    target_batch,
                    operator,
                    self._ZF_BASELINE_METRICS,
                    transform=scoring_transform(self.config),
                    batch_data=batch_data,
                )
            )
            metrics.update(
                zero_filled_context_metrics(
                    self.validation_metrics_computer, context, target_batch, operator, batch_data
                )
            )
        return metrics

    def _context_metric_prediction(self, raw: torch.Tensor, batch_context: dict[str, Any]):
        return context_metric_kspace(raw, self._measurement(batch_context)[2]), "kspace"

    def _validation_forward(
        self,
        input_batch: torch.Tensor,
        batch_context: dict[str, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""``f(A^H y)`` with DC on the acquired lines, Noisier2Noise-corrected on
        :math:`\Omega` under Robust SSDU (Eq. 36) and brought back to an image."""
        y, acquired, maps = self._measurement(batch_context)
        lr_image, forward_kwargs, bc = self._network(
            batch_context, y, acquired, maps, validation=True
        )
        lr_image = self._apply_input_conditioning(lr_image, bc)
        image = self.env.generator(lr_image, **forward_kwargs)
        if self.noisier2noise:
            corrected = noisier2noise_correct(kspace_of(image, maps), y, acquired, self.alpha)
            image = image_of(corrected, maps, None)
        # F S x_hat is blind where S = 0, so the network's image there is unidentified.
        return image if maps is None else image * coil_support(maps).to(image.real.dtype)
