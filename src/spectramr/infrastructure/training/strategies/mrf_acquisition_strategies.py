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
  expected Cramér-Rao bound over a target parameter prior (MRF §4).
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
from spectramr.domain.services.mrf_dictless_matching import (
    MRFDictlessMatcher,
)
from spectramr.infrastructure.physics.manifolds.time_reparameterisation import (
    apply_time_reparam,
)
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.models.capabilities import StrategyCapabilities


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
    #: CRLBMRFPulseDesign is a self-admitted toy Fisher surrogate whose
    #: ACQUISITION_DESIGN task the fingerprinting profile does not support, and
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

    Treats ``input_batch[B, T, 2]`` as the current (α, TR) pulse-sequence
    representation and ``target_batch[B, 5]`` as the parameter prior
    sample. Inner Fisher trace is approximated by the squared
    sensitivity of a toy Bloch surrogate to first-order; full
    integration with ``differentiable_bloch`` is left as an inference-time
    step.
    """

    #: ``validation_step`` writes the CRLB objective on the held-out prior
    #: samples itself; the metrics registry cannot, since it needs the generator
    #: and autograd through the Bloch surrogate.
    capabilities: ClassVar[StrategyCapabilities] = StrategyCapabilities(
        emitted_metrics=frozenset({"crlb_objective"}),
    )

    def __init__(self, env: TrainingEnvironment, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        cfg = getattr(self.config.training, "crlb_mrf_pulse_design", None)
        self.lambda_beltrami = float(getattr(cfg, "lambda_beltrami", 1e-3)) if cfg else 1e-3

    @staticmethod
    def _toy_fisher_trace(pulse: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """First-order Fisher-trace surrogate: ``Σ_n (∂_θ s_n)²`` for a
        closed-form Bloch signal stand-in."""
        alpha = pulse[..., 0]
        TR = pulse[..., 1].clamp_min(1e-3)
        T1, T2 = params[..., 0].clamp_min(1e-3), params[..., 1].clamp_min(1e-3)
        signal = (
            alpha.sin().unsqueeze(0)
            * (1.0 - (-TR.unsqueeze(0) / T1.unsqueeze(-1)).exp())
            * (-TR.unsqueeze(0) / T2.unsqueeze(-1)).exp()
        )
        # create_graph=True keeps the second-order graph: signal depends on
        # `pulse` (alpha=pulse[...,0], TR=pulse[...,1]), so ∂signal/∂θ still
        # carries a path back to the generator output. With create_graph=False
        # the Fisher trace (and thus -CRLB) was detached from `pulse`, so the
        # generator learned ONLY the smoothness term (2026-05-31 audit).
        d_T1 = torch.autograd.grad(
            signal.sum(),
            T1,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        d_T2 = torch.autograd.grad(
            signal.sum(),
            T2,
            create_graph=True,
            allow_unused=True,
        )[0]
        zero = T1.new_zeros(())
        sq_T1 = d_T1.pow(2).sum() if d_T1 is not None else zero
        sq_T2 = d_T2.pow(2).sum() if d_T2 is not None else zero
        return sq_T1 + sq_T2

    @classmethod
    def _crlb_objective(cls, pulse: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """``-Fisher`` (CRLB ≈ 1/Fisher, minimised) of ``pulse`` over the prior ``params``.

        The one definition training and validation share. It differentiates the
        signal with respect to ``params``, so callers under ``no_grad`` must
        re-enable gradients around it.
        """
        params_req = params.detach().clone().requires_grad_(True)
        return -cls._toy_fisher_trace(pulse, params_req)

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Registry metrics the arm requests, plus ``crlb_objective`` on the val batch."""
        metrics = super().validation_step(input_batch, target_batch, **kwargs)

        generator = self.generator_model
        was_training = generator.training
        generator.eval()
        try:
            pulse = generator(input_batch.to(self.device, non_blocking=True)).detach()
            # The caller validates under no_grad; the Fisher trace needs ∂s/∂θ.
            with torch.enable_grad():
                crlb = self._crlb_objective(pulse, target_batch.to(pulse.device))
        finally:
            generator.train(was_training)
        metrics["crlb_objective"] = float(crlb.detach())
        return metrics

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch
        pulse = self.generator_model(input_batch)  # [B, T, 2]
        crlb = self._crlb_objective(pulse, target_batch)
        # Beltrami-style smoothness: penalise large per-step pulse change.
        smooth = (pulse.diff(dim=1)).pow(2).mean()
        total = crlb + self.lambda_beltrami * smooth
        return {
            "g_total_loss": total,
            "crlb_objective": crlb.detach(),
            "pulse_beltrami_smoothness": smooth.detach(),
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
        # global vector (audit 2026-06). Absent → scanner 0 broadcast (legacy).
        resolved = self._resolve_legacy_batch(input_batch, kwargs)
        scanner_id = resolved.get("scanner_id") if isinstance(resolved, dict) else None
        if scanner_id is None:
            scanner_id = kwargs.get("scanner_id")
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
