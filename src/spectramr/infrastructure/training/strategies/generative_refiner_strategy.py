"""Generative-iterative refiner: the shared Track-A chassis (MRIxFields2026).

The saturation analysis of the cohort found that the one arm to break the SSIM ceiling
(b22, ``ulf_dps``, +0.32) did so not by a wider receptive field but by being a
generative prior + iterative sampler + non-saturating eps-matching objective, with its
distinctive coupling entering as a per-step data-consistency guidance term. This module
generalizes that recipe into a reusable chassis:

- **training** = DDPM eps-matching on the field-conditioned target (the learned prior);
  the objective is the generic :func:`compute_ulf_dps_loss`, whose gradient does not
  vanish as the reconstruction sharpens (unlike L1), so it keeps training past the
  pixel-loss plateau;
- **inference** = a normalized DDIM/DPS reverse loop (:func:`refiner_sample`) whose
  per-step guidance ``(zeta / ||r||) * grad`` is driven by a **pluggable forward
  operator** ``A`` (the arm's mechanism); ``r = A(x0_hat) - y``.

An arm keeps its identity by choosing / overriding the forward operator; its one-knob
ablation is ``guidance_weight = 0`` (the unconditional prior). Selecting the operator is
a dict dispatch (no ``if/elif`` chains); an unknown operator raises (pitfall #9), which
the closed ``Literal`` in :class:`GenerativeRefinerConfig` already guards at load time.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, ClassVar

import torch

from .field_guided_diffusion_strategy import make_alphas_cumprod
from .reconstruction import ReconstructionTrainingStrategy
from .ulf_dps_strategy import compute_ulf_dps_loss
from .ulf_lowpass_operator import UlfLowpassOperator, build_ulf_lowpass_operator

# ── Forward-operator dispatch table (the arm's mechanism, as A(x)) ───────────
# Signature: (x0_hat, op_kwargs) -> A(x0_hat). Registry-style dispatch, never if/elif.
ForwardOp = Callable[[torch.Tensor, dict[str, Any]], torch.Tensor]


def _field_lowpass_op(x: torch.Tensor, kw: dict[str, Any]) -> torch.Tensor:
    """b22 recipe: the ULF low-pass ``A`` (``kw['forward_op']``, or a ``blur_cutoff``)."""
    op = kw.get("forward_op")
    if op is None:
        if "blur_cutoff" not in kw:
            raise ValueError("field_lowpass needs op_kwargs 'forward_op' or 'blur_cutoff'.")
        op = UlfLowpassOperator.constant_cutoff(float(kw["blur_cutoff"]))
    return op.apply(x, kw.get("contrast_id"))


def _wiener_recoverable_noise(source_field_tesla: float) -> float:
    """N(b) = softplus(-softplus(0) * log10(b)); the b27 Wiener noise-to-signal ratio.

    Reproduces :meth:`FieldWienerNet.noise_level` at the model's identity init
    (slope = -softplus(0) = -ln 2, c = 0): the field slope is negative by construction,
    so N rises as the source field drops (lower field, more noise, narrower band).
    """
    slope = -math.log1p(math.exp(0.0))  # -softplus(0) = -ln 2
    arg = slope * math.log10(max(float(source_field_tesla), 1e-3))
    return math.log1p(math.exp(arg))  # softplus(arg)


def _wiener_band_op(x: torch.Tensor, kw: dict[str, Any]) -> torch.Tensor:
    """b27 mechanism as guidance: the field's Wiener recoverable-band low-pass response.

    Applies the field-dependent Wiener gain on the canonical reference spectrum
    ``S_ref(r) = 1/(1+(r/0.15)^2)`` at the source-field noise ratio ``N(b)`` (the same
    ``S_ref`` and ``N(b)`` as :meth:`FieldWienerNet.recoverable_band`), DC-normalized so
    ``G(0) = 1``. This is a DC-preserving low-pass whose rolloff IS the recoverable band
    (steeper at lower field), so the DPS consistency term pins the band the source field
    could physically measure while the prior fills the rest. The soft gain avoids the hard
    ``>0.5`` band, which at ULF (``N > 1``) would be empty and inert. The gain is constant
    in ``x`` (linear operator) so the DPS gradient flows cleanly; rfft2/irfft2 on a real
    magnitude image is exempt from the complex-k-space fft2c rule (physics.md).
    """
    h, w = int(x.shape[-2]), int(x.shape[-1])
    dev = x.device
    fy = torch.fft.fftfreq(h, device=dev).abs()[:, None]
    fx = torch.fft.rfftfreq(w, device=dev).abs()[None, :]
    r = torch.sqrt(fy**2 + fx**2)
    s_ref = 1.0 / (1.0 + (r / 0.15) ** 2)
    n = _wiener_recoverable_noise(float(kw.get("source_field", 0.1)))
    gain = s_ref / (s_ref + n)
    gain = (gain / gain[0, 0].clamp_min(1e-8)).clamp(0.0, 1.0).to(x.dtype)  # DC-normalized
    xf = torch.fft.rfft2(x, norm="ortho")
    return torch.fft.irfft2(gain * xf, s=(h, w), norm="ortho")


#: Operators that apply the shared ULF low-pass A, so the arm must declare training.ulf_lowpass.
_OPERATORS_USING_ULF_FORWARD = frozenset({"field_lowpass"})

_FORWARD_OPERATORS: dict[str, ForwardOp] = {
    # Field-dependent low-pass degradation (the b22 recipe).
    "field_lowpass": _field_lowpass_op,
    # b27 Wiener recoverable-band projection (the field-dependent SNR crossover).
    "wiener_band": _wiener_band_op,
}


def resolve_forward_operator(name: str) -> ForwardOp:
    """Resolve the DPS forward operator by name, raising on an unknown mechanism.

    A silent fallback here would let an arm advertise a mechanism the sampler never
    applies (pitfall #16 at the sampling layer); the closed ``Literal`` on the schema
    plus this raise make the guidance operator a wired, validated knob (pitfall #15).
    """
    try:
        return _FORWARD_OPERATORS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown generative-refiner forward_operator {name!r}. "
            f"Registered: {sorted(_FORWARD_OPERATORS)}."
        ) from exc


def refiner_sample(
    model: Any,
    y: torch.Tensor,
    b_t: torch.Tensor,
    *,
    alphas_cumprod: torch.Tensor,
    forward_op: ForwardOp,
    op_kwargs: dict[str, Any] | None = None,
    n_steps: int = 50,
    guidance_weight: float = 1.0,
    contrast_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalized DDIM/DPS reverse sampling with a pluggable forward operator.

    ``contrast_id`` (optional, per-sample ``[B]``) conditions the widened score net's
    FiLM (only when the net's ``use_contrast_conditioning`` is on).

    Generalizes :func:`ulf_dps_sample`: the only arm-specific piece is ``forward_op``,
    which maps the current clean-image estimate ``x0_hat`` to its predicted measurement
    ``A(x0_hat)`` compared against ``y``. Each step predicts ``eps`` -> ``x0_hat``
    (clamped to the data range), takes the deterministic DDIM (eta=0) prior step, then
    subtracts the residual-normalized guidance ``(zeta / ||r||) * grad`` (Chung et al.
    2023, Alg-1). ``guidance_weight = 0`` (zeta) => plain prior sampling (the ablation).
    Autograd through the score net is forced even under an outer ``no_grad``.
    """
    op_kwargs = op_kwargs or {}
    timesteps = alphas_cumprod.shape[0]
    n = max(1, min(int(n_steps), timesteps))
    idx = torch.linspace(timesteps - 1, 0, n, device=y.device).round().long()
    b_t_vec = b_t.reshape(-1).to(y.device)
    zeta = float(guidance_weight)
    x = torch.randn_like(y)
    with torch.enable_grad():
        for i in range(n):
            t = int(idx[i])
            x = x.detach().requires_grad_(True)
            t_b = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            eps = model(x, timesteps=t_b, field_strength=b_t_vec, contrast_id=contrast_id)
            a_t = alphas_cumprod[t]
            x0_hat = (x - (1.0 - a_t).sqrt() * eps) / a_t.sqrt().clamp_min(1e-3)
            x0_hat = x0_hat.clamp(-1.0, 2.0)  # data range guard (images are [0,1])
            residual = forward_op(x0_hat, {**op_kwargs, "contrast_id": contrast_id}) - y
            if i < n - 1:
                a_next = alphas_cumprod[int(idx[i + 1])]
                x_prior = a_next.sqrt() * x0_hat + (1.0 - a_next).sqrt() * eps
            else:
                x_prior = x0_hat
            if zeta > 0.0:
                grad = torch.autograd.grad(residual.pow(2).sum(), x)[0]
                rnorm = (
                    residual.detach()
                    .pow(2)
                    .flatten(1)
                    .sum(1)
                    .sqrt()
                    .clamp_min(1e-8)
                    .view(-1, 1, 1, 1)
                )
                x = (x_prior.detach() - (zeta / rnorm) * grad).detach()
            else:
                x = x_prior.detach()
    return x.detach().clamp(0.0, 1.0)


class GenerativeRefinerStrategy(ReconstructionTrainingStrategy):
    """Train a field-conditioned diffusion prior; sample with mechanism-guided DPS.

    The shared Track-A chassis. The distinctive mechanism enters as the DPS forward
    operator (``config.training.generative_refiner.forward_operator``); the ablation sets
    ``guidance_weight = 0``. Subclasses may override :meth:`_forward_operator` to supply
    an arm-specific mechanism that is not a pure config knob.
    """

    #: Loss ownership (mrixfields review 2026-09-03): computes its objective inline and folds NOTHING else: any other declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset()
    folds_image_losses: ClassVar[bool | None] = False

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("generative_refiner", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "generative_refiner", None)
        self._gr_timesteps = int(getattr(cfg, "timesteps", 1000)) if cfg else 1000
        self._gr_schedule = str(getattr(cfg, "beta_schedule", "cosine")) if cfg else "cosine"
        self._gr_sampling_steps = int(getattr(cfg, "sampling_steps", 50)) if cfg else 50
        self._gr_guidance_weight = float(getattr(cfg, "guidance_weight", 1.0)) if cfg else 1.0
        self._gr_operator_name = (
            str(getattr(cfg, "forward_operator", "field_lowpass")) if cfg else "field_lowpass"
        )
        self._gr_op_kwargs: dict[str, Any] = {
            "source_field": float(getattr(cfg, "source_field", 0.1)) if cfg else 0.1,
        }
        if self._gr_operator_name in _OPERATORS_USING_ULF_FORWARD:
            self._gr_op_kwargs["forward_op"] = build_ulf_lowpass_operator(self.config)
        # Resolve now so an unknown operator fails at setup, not mid-validation.
        self._gr_forward_op = self._forward_operator()
        self._acp_cache: dict[tuple, torch.Tensor] = {}

    def _forward_operator(self) -> ForwardOp:
        """The arm's mechanism as A(x). Override for a non-config-knob mechanism."""
        return resolve_forward_operator(self._gr_operator_name)

    def _alphas_cumprod(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (self._gr_timesteps, self._gr_schedule, str(device), str(dtype))
        acp = self._acp_cache.get(key)
        if acp is None:
            acp = make_alphas_cumprod(
                self._gr_timesteps, self._gr_schedule, device=device, dtype=dtype
            )
            self._acp_cache[key] = acp
        return acp

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if batch is None or not hasattr(batch, "get"):
            raise ValueError(
                "GenerativeRefinerStrategy requires a mapping batch (dict/TrainingBatch) "
                f"from the mrixfields dataset; got {type(batch)!r}."
            )
        acp = self._alphas_cumprod(batch["target"].device, batch["target"].dtype)
        return compute_ulf_dps_loss(self.env.generator, batch, alphas_cumprod=acp)

    def validation_step(
        self,
        input_batch: Any,
        target_batch: Any,
        field_strength_target: Any = None,
        contrast_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        bc = kwargs.pop("batch_context", None) or {}
        if field_strength_target is not None:
            bc["field_strength_target"] = field_strength_target
        if contrast_id is not None:
            bc["contrast_id"] = contrast_id
        return super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation = mechanism-guided DPS sampling."""
        y = batch_context.get("input", input_batch)
        b_t = batch_context.get("field_strength_target", kwargs.get("field_strength_target"))
        if b_t is None:
            raise ValueError(
                "GenerativeRefinerStrategy validation requires per-sample "
                "'field_strength_target' (set data.expose_field_strength_target on the "
                "mrixfields dataset). Got field_strength_target=None."
            )
        acp = self._alphas_cumprod(y.device, y.dtype)
        gen = self.env.generator
        b_t = b_t if torch.is_tensor(b_t) else torch.as_tensor(b_t, device=y.device)
        cid = batch_context.get("contrast_id", kwargs.get("contrast_id"))
        if cid is not None and not torch.is_tensor(cid):
            cid = torch.as_tensor(cid, device=y.device)
        val_cfg = getattr(self.config, "validation", None)
        chunk = int((val_cfg.loader.chunk_size if val_cfg else 0) or 0)

        def _sample(
            y_c: torch.Tensor, b_c: torch.Tensor, cid_c: torch.Tensor | None
        ) -> torch.Tensor:
            return refiner_sample(
                gen,
                y_c,
                b_c,
                alphas_cumprod=acp,
                forward_op=self._gr_forward_op,
                op_kwargs=self._gr_op_kwargs,
                n_steps=self._gr_sampling_steps,
                guidance_weight=self._gr_guidance_weight,
                contrast_id=cid_c,
            )

        # The DPS reverse loop retains a per-step autograd graph, so micro-batch when set.
        if chunk > 0 and y.shape[0] > chunk:
            outs = [
                _sample(
                    y[i : i + chunk],
                    b_t[i : i + chunk],
                    None if cid is None else cid[i : i + chunk],
                )
                for i in range(0, y.shape[0], chunk)
            ]
            return torch.cat(outs, dim=0)
        return _sample(y, b_t, cid)
