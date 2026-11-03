"""MRF embedding / acquisition-design / harmonisation strategies —
§§3, 4, 5.

* :class:`ConformalMRFDictlessReconStrategy`
  (``conformal_mrf_dictless_recon``) — trains a 5-D conformal
  embedding of MRF fingerprints (MRF §3). Couples a contrastive
  cosine-agreement loss with the
  :class:`ConformalityJacobianLoss`.
* :class:`CRLBMRFPulseDesignStrategy`
  (``crlb_mrf_pulse_design``) — bilevel optimisation that designs a
  Beltrami-parameterised MRF pulse sequence by minimising the
  expected relative Cramér-Rao bound of an IR-FISP fingerprint over a T1/T2
  prior, inside the declared SAR/slew proxies (MRF §4).
* :class:`CrossScannerMRFHarmonisationStrategy`
  (``cross_scanner_mrf_harmonisation``) — learns a per-scanner
  square-root-velocity time reparameterisation τ_s to align
  fingerprints across vendors (MRF §5).

References:
    [14] McGivney et al., "SVD compression for MRF in the time
         domain", *IEEE Trans. Med. Imag.*, 33(12), 2014.
    [18] Zhao et al., "Optimal experiment design for MRF: Cramér-Rao
         bound meets spin dynamics", *IEEE TMI*, 38(3), 2019.
    [27] Srivastava et al., "Shape analysis of elastic curves",
         *IEEE TPAMI*, 33(7), 2011.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from spectramr.config.schemas.enums import Regime, Task
from spectramr.core.module_utils import unwrap_model
from spectramr.domain.services.mrf_dictless_matching import (
    MRFDictlessMatcher,
)
from spectramr.infrastructure.physics.acquisition.mrf_acquisition_manifold import (
    AcquisitionBound,
    project_pulse_to_bound,
    sar_estimate,
    slew_estimate,
)
from spectramr.infrastructure.physics.acquisition.mrf_crlb import (
    expected_relative_crlb,
    log_grid_prior,
    sample_log_uniform_prior,
)
from spectramr.infrastructure.physics.manifolds.time_reparameterisation import (
    apply_time_reparam,
)
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.rng_streams import rank_seeded_generator
from spectramr.models.capabilities import StrategyCapabilities
from spectramr.models.generators.mrf_pulse_design import MRFPulseDesign


class ConformalMRFDictlessReconStrategy(BaseTrainingStrategy):
    """Trains a conformal fingerprint embedding for dictionary-free MRF.

    Loss = cosine-agreement between ``ψ(f)`` and ``f`` pairwise
    cosine matrices, plus an optional Jacobian-conformality penalty
    when the embedding model exposes ``jacobian_at``.
    """

    # Computes its own objective and reads no `losses:` block, so an empty builder stack
    # is the design (LossBuilder.validate) and any declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset()
    folds_image_losses: ClassVar[bool | None] = False

    # parallel_compat: Calls model.jacobian_at outside the wrapped forward, so DDP/ZeRO reduce
    # no gradient for it and FSDP reads a shard.
    parallel_compat = frozenset()

    #: Fingerprinting: dictionary-free fingerprint matching (MRFDictlessMatcher,
    #: a KD-tree over the learned 5-D Bloch-parameter embedding) IS the defining
    #: MRF operation. Its two siblings here are deliberately UNTAGGED --
    #: CRLBMRFPulseDesign designs an acquisition, an ACQUISITION_DESIGN task the
    #: fingerprinting profile does not support, and
    #: CrossScannerMRFHarmonisation has no Task member describing harmonisation.
    capabilities: ClassVar[StrategyCapabilities] = StrategyCapabilities(
        workflows=frozenset({Regime.FINGERPRINTING}),
        tasks=frozenset({Task.PARAMETER_MAPPING}),
    )

    def __init__(self, env: TrainingEnvironment, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        cfg = getattr(self.config.training, "conformal_mrf_dictless_recon", None)
        self.lambda_c = float(getattr(cfg, "lambda_conformality", 1e-2)) if cfg else 1e-2
        # KD-tree matcher is lazily built once the dictionary is supplied at
        # inference time via :meth:`build_matcher`.
        self._matcher: MRFDictlessMatcher | None = None

    def build_matcher(
        self,
        fingerprints: torch.Tensor,
        parameters: torch.Tensor,
        *,
        batch_size: int = 256,
        normalise: bool = True,
    ) -> MRFDictlessMatcher:
        """Embed the dictionary and build the KD-tree matcher.

        Replaces the brute-force ``O(V·D·N)`` cosine-similarity inner loop
        of vanilla MRF matching with an ``O(V·log D)`` KD-tree lookup over
        the 5-D conformal embedding learned by the strategy. The matcher
        is cached on the strategy so subsequent :meth:`match` calls reuse
        the tree.

        Args:
            fingerprints: ``[D, N]`` real-valued dictionary entries.
            parameters: ``[D, 5]`` Bloch parameters paired to ``fingerprints``.
            batch_size: Embedding batch size for the dictionary forward pass.
            normalise: L2-normalise embeddings so KD-tree distance
                corresponds to cosine distance.

        Returns:
            The constructed :class:`MRFDictlessMatcher` (also cached as
            ``self._matcher``).
        """
        matcher = MRFDictlessMatcher(self.generator_model, device=self.device, normalise=normalise)
        matcher.build_index(fingerprints, parameters, batch_size=batch_size)
        self._matcher = matcher
        return matcher

    def match(
        self,
        fingerprints: torch.Tensor,
        *,
        k: int = 1,
        batch_size: int = 256,
    ) -> dict[str, Any]:
        """Look up in-vivo fingerprints against the cached KD-tree.

        Args:
            fingerprints: ``[V, N]`` real-valued query fingerprints.
            k: Number of nearest neighbours to return.
            batch_size: Embedding batch size for the query forward pass.

        Returns:
            Dict with ``indices`` ``[V, k]``, ``parameters`` ``[V, k, 5]``,
            and embedding-space ``distances`` ``[V, k]``.

        Raises:
            RuntimeError: if :meth:`build_matcher` has not been called yet.
        """
        if self._matcher is None:
            raise RuntimeError(
                "ConformalMRFDictlessReconStrategy.match: call "
                "build_matcher(dict_fp, dict_params) before match(...)"
            )
        return self._matcher.query(fingerprints, k=k, batch_size=batch_size)

    @staticmethod
    def _pairwise_cos(x: torch.Tensor) -> torch.Tensor:
        nrm = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        x_n = x / nrm
        return x_n @ x_n.transpose(-1, -2)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch
        embedded = self.generator_model(input_batch)
        cos_emb = self._pairwise_cos(embedded)
        cos_raw = self._pairwise_cos(input_batch)
        agree = (cos_emb - cos_raw).pow(2).mean()
        # Optional conformality penalty when the model exposes Jacobian.
        jac = None
        if hasattr(self.generator_model, "jacobian_at"):
            jac = self.generator_model.jacobian_at(input_batch[: min(2, input_batch.shape[0])])
        if jac is not None and jac.dim() >= 3:
            sv = torch.linalg.svdvals(jac)
            frob_sq = jac.pow(2).sum(dim=(-1, -2))
            target_val = jac.shape[-2] * sv[..., 0].pow(2)
            conf_loss = (frob_sq - target_val).pow(2).mean()
        else:
            conf_loss = embedded.new_zeros(())
        total = agree + self.lambda_c * conf_loss
        return {
            "g_total_loss": total,
            "conformal_mrf_cos_agree": agree.detach(),
            "conformal_mrf_jacobian_penalty": conf_loss.detach(),
        }


class CRLBMRFPulseDesignStrategy(BaseTrainingStrategy):
    """Beltrami-CRLB-optimal MRF pulse-sequence design (MRF §4).

    The generator (``mrf_pulse_design``) emits one ``[T, 2]`` (flip angle rad, TR ms) train,
    broadcast over the batch. Each step projects it onto the declared ``mrf`` SAR and slew
    proxies and minimises the expected relative CRLB (T1/T2 block, M0 a nuisance) of an
    IR-FISP EPG fingerprint over a log-uniform T1/T2 prior drawn by the strategy
    (:mod:`...physics.acquisition.mrf_crlb`).
    The data batch only clocks the steps: the prior, not the image, is the design's input.
    """

    # The CRLB is the whole objective; a declared image loss would be a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset()
    folds_image_losses: ClassVar[bool | None] = False
    checkpointed_streams: ClassVar[tuple[str, ...]] = ("_prior_rng",)
    reports_train_metrics: ClassVar[bool] = False

    #: ``validation_step`` writes these itself on a fixed prior grid; the metrics registry
    #: cannot, since they need the generator's design and the Bloch model.
    capabilities: ClassVar[StrategyCapabilities] = StrategyCapabilities(
        emitted_metrics=frozenset(
            {"crlb_objective", "crlb_objective_initial", "pulse_sar_proxy", "pulse_slew_proxy"}
        ),
    )

    def __init__(self, env: TrainingEnvironment, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        cfg = getattr(self.config.training, "crlb_mrf_pulse_design", None)
        self.lambda_beltrami = float(getattr(cfg, "lambda_beltrami", 1e-3))
        self.t1_range_ms = tuple(getattr(cfg, "t1_range_ms", (100.0, 3000.0)))
        self.t2_range_ms = tuple(getattr(cfg, "t2_range_ms", (20.0, 500.0)))
        self.prior_samples = int(getattr(cfg, "prior_samples", 64))
        self.te_ms = float(getattr(cfg, "te_ms", 2.0))
        self.bound = self._resolve_bound(getattr(self.config, "mrf", None))
        self._prior_rng = rank_seeded_generator(int(self.config.run.seed))
        self._check_te_below_tr(unwrap_model(self.generator_model), self.te_ms)

    @staticmethod
    def _check_te_below_tr(design: Any, te_ms: float) -> None:
        """The EPG reads at TE inside every TR; checked once against the reachable TR floor."""
        if not isinstance(design, MRFPulseDesign):
            raise TypeError(
                "crlb_mrf_pulse_design needs the mrf_pulse_design generator; got "
                f"{type(design).__name__}"
            )
        if te_ms >= design.min_tr_ms:
            raise ValueError(
                f"crlb_mrf_pulse_design.te_ms={te_ms} must be below the shortest reachable TR "
                f"({design.min_tr_ms:.3f} ms = tr_init_ms * (1 - k_max / 2))"
            )

    @staticmethod
    def _resolve_bound(mrf_cfg: Any) -> AcquisitionBound:
        sar = getattr(mrf_cfg, "sar_max_w_per_kg", None)
        slew = getattr(mrf_cfg, "gradient_slew_rate_t_per_m_per_s", None)
        if sar is None or slew is None:
            raise ValueError(
                "crlb_mrf_pulse_design needs mrf.sar_max_w_per_kg and "
                "mrf.gradient_slew_rate_t_per_m_per_s: the design is projected onto them."
            )
        return AcquisitionBound(float(sar), float(slew))

    def declared_metric_keys(self) -> frozenset[str]:
        return frozenset(
            {"crlb_objective", "pulse_beltrami_smoothness", "pulse_sar_proxy", "pulse_slew_proxy"}
        )

    def _design(self, model_out: torch.Tensor) -> torch.Tensor:
        """The single ``[T, 2]`` design from the generator's ``[B, T, 2]`` broadcast."""
        if model_out.dim() != 3 or model_out.shape[-1] != 2:
            raise ValueError(
                "crlb_mrf_pulse_design needs a [B, T, 2] pulse-train generator "
                f"(model_type mrf_pulse_design); got output {tuple(model_out.shape)}"
            )
        return model_out[0]

    def _crlb_objective(self, pulse: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Expected relative CRLB of the bound-projected ``pulse``; shared by train and val."""
        legal = project_pulse_to_bound(pulse, self.bound)
        return expected_relative_crlb(legal, t1, t2, te_ms=self.te_ms)

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Registry metrics the arm requests, plus the design's CRLB on a fixed prior grid."""
        metrics = super().validation_step(input_batch, target_batch, **kwargs)
        generator = self.generator_model
        was_training = generator.training
        generator.eval()
        try:
            pulse = self._design(generator(input_batch.to(self.device, non_blocking=True)))
            t1, t2 = log_grid_prior(
                self.prior_samples, self.t1_range_ms, self.t2_range_ms, device=pulse.device
            )
            crlb = self._crlb_objective(pulse, t1, t2)
            values = {"crlb_objective": crlb}
            design_module = unwrap_model(generator)
            if not isinstance(design_module, MRFPulseDesign):
                raise TypeError(
                    "crlb_objective_initial needs the mrf_pulse_design generator; got "
                    f"{type(design_module).__name__}"
                )
            values["crlb_objective_initial"] = self._crlb_objective(
                design_module.initial_design(), t1, t2
            )
            values["pulse_sar_proxy"] = sar_estimate(pulse)
            values["pulse_slew_proxy"] = slew_estimate(pulse)
        finally:
            generator.train(was_training)
        # Validation is outside the training loop; one host transfer per pass.
        metrics.update({k: float(v) for k, v in values.items()})
        return metrics

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch, target_batch
        pulse = self._design(self.generator_model(input_batch))
        t1, t2 = sample_log_uniform_prior(
            self.prior_samples, self.t1_range_ms, self.t2_range_ms, generator=self._prior_rng
        )
        t1 = t1.to(pulse.device, non_blocking=True)
        t2 = t2.to(pulse.device, non_blocking=True)
        crlb = self._crlb_objective(pulse, t1, t2)
        smooth = pulse.diff(dim=0).pow(2).mean()
        total = crlb + self.lambda_beltrami * smooth
        return {
            "g_total_loss": total,
            "crlb_objective": crlb.detach(),
            "pulse_beltrami_smoothness": smooth.detach(),
            "pulse_sar_proxy": sar_estimate(pulse).detach(),
            "pulse_slew_proxy": slew_estimate(pulse).detach(),
        }


class CrossScannerMRFHarmonisationStrategy(BaseTrainingStrategy):
    """Per-scanner time reparameterisation for cross-vendor MRF
    harmonisation (MRF §5).

    Caller supplies a batch dict with per-sample ``scanner_id`` (long
    tensor) and ``f_canonical`` (canonical-scanner fingerprint of the
    same parameters); the strategy reparameterises ``f_canonical`` by
    the scanner-specific τ and matches it to the observed
    ``input_batch``.
    """

    def __init__(self, env: TrainingEnvironment, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        cfg = getattr(self.config.training, "cross_scanner_mrf_harmonisation", None)
        # NOTE: the Beltrami squash `eps` lives in the model
        # (`ScannerTimeReparam.forward` -> `enforce_1d_beltrami_constraint(
        # raw_b, eps=self.eps)`), driven by the model's own `eps`. A
        # strategy-level `tau_eps` knob would be a silent no-op (it never
        # reaches the model), so it is deliberately not read here; wiring it
        # would require a model-construction change (2026-05-31 audit).
        self.lambda_anchor = float(getattr(cfg, "lambda_anchor", 1e-3)) if cfg else 1e-3
        expose = getattr(getattr(self.config, "data", None), "expose", None)
        self.requires_scanner_id = bool(getattr(expose, "scanner_id", False))

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch
        # generator_model is a ScannerTimeReparam: forward returns τ[B, T].
        b = input_batch.shape[0]
        # Per-scanner conditioning: pass scanner_id (when the batch provides it)
        # so the model indexes its per-scanner τ table instead of a single
        # global vector. Absent is allowed only when the arm does not declare
        # data.expose.scanner_id; otherwise every vendor would share scanner 0's τ.
        resolved = self._resolve_legacy_batch(input_batch, kwargs)
        scanner_id = resolved.get("scanner_id") if isinstance(resolved, dict) else None
        if scanner_id is None:
            scanner_id = kwargs.get("scanner_id")
        if scanner_id is None and self.requires_scanner_id:
            raise ValueError(
                "cross_scanner_mrf_harmonisation: data.expose.scanner_id is declared but the "
                "batch carries no scanner_id, so every scanner would share one τ."
            )
        tau = self.generator_model(batch_size=b, scanner_id=scanner_id)
        # Warp the canonical fingerprint by τ; ``input_batch`` is the observed.
        warped = apply_time_reparam(target_batch, tau)
        residual = F.mse_loss(warped, input_batch)
        # Anchor τ near identity using SRV regulariser surrogate (L2 on raw).
        anchor = (
            (tau - torch.linspace(0, tau.shape[-1] - 1, tau.shape[-1], device=tau.device))
            .pow(2)
            .mean()
        )
        total = residual + self.lambda_anchor * anchor
        return {
            "g_total_loss": total,
            "harmonisation_residual": residual.detach(),
            "harmonisation_tau_anchor": anchor.detach(),
        }


__all__ = [
    "CRLBMRFPulseDesignStrategy",
    "ConformalMRFDictlessReconStrategy",
    "CrossScannerMRFHarmonisationStrategy",
]
