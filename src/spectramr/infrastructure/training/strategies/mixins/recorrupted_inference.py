r"""Monte-Carlo recorrupted inference for the R2R family (#2329, ``PAPER.md`` Theorem 4).

An R2R-trained network is the minimum-mean-square denoiser for a *recorrupted*
input, whose noise is :math:`(1+\tau^2)\Sigma`, not :math:`\Sigma`. Scoring it
on the raw repetition validates it at a noise level it was never trained at.
The estimator R2R prescribes instead feeds :math:`J` fresh recorruptions and
averages the outputs,

.. math::

    \hat x_J = \frac1J\sum_{j=1}^{J} f\bigl(y + \tau z_j\bigr),\qquad
    z_j\sim\mathcal{CN}(0,\Sigma),

or, for ``nullspace_r2r``, the input half of :math:`J` coil-null-space splits
with fresh null directions :math:`v_j`. ``validation.scoring.recorrupted_inference``
turns it on; this module resolves the knob against the arm's training target,
builds the per-batch recorruption once, and runs the :math:`J` forwards one at a
time with a running sum on the device (a ``J``-fold batch would multiply the
activation memory, which is what binds on these arms).

**Units.** The validation input has been divided by the subject's
``kspace_scale``, while the committed :math:`\Sigma` is in raw scanner units, so
the draw is divided by the same per-sample scale. Missing that is a silent
recorruption at the wrong level (about 20x on M4Raw), so an absent scale raises.
The null-space split needs no scale: it is linear and its whitener needs only
the shape of :math:`\Sigma`.

**Randomness.** Draws come from dedicated generators, so turning the knob on
does not shift the training RNG stream. A call carrying the validation batch
index reseeds them from ``(run.seed, index)``, so every epoch scores batch ``i``
under the same recorruptions; ``pred_alt`` passes no index and continues the
stream, so its draws are independent of the prediction's (identical draws would
cancel out of ``rep_variance``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from spectramr.core.metrics.coil_manifold import (
    as_complex_coils,
    as_interleaved,
    require_complex_maps,
)
from spectramr.data.batch_types import align_scale_to_batch, read_batch_field

from .validation_context import coil_maps, fold_depth, unfold_depth

logger = logging.getLogger(__name__)

__all__ = [
    "ELIGIBLE_TARGET_MODES",
    "RecorruptedInference",
    "RecorruptionSpec",
    "draw_r2r_noise",
    "resolve_recorrupted_inference",
]

#: Training targets whose network is an estimator for a recorrupted input.
ELIGIBLE_TARGET_MODES: frozenset[str] = frozenset({"r2r", "nullspace_r2r"})


@dataclass(frozen=True)
class RecorruptionSpec:
    """The resolved knob: ``J``, the strength, and where the noise model comes from."""

    target_mode: str
    samples: int
    alpha: float | None
    angle_deg: float | None
    covariance_source: str | None
    whitening: str | None
    seed: int

    def describe(self) -> str:
        if self.target_mode == "r2r":
            return f"J={self.samples}, r2r alpha={self.alpha}, Sigma={self.covariance_source}"
        return (
            f"J={self.samples}, nullspace_r2r angle={self.angle_deg} deg, "
            f"whitening={self.whitening}"
        )


def resolve_recorrupted_inference(config: Any) -> RecorruptionSpec | None:
    """The spec for ``validation.scoring.recorrupted_inference``, or ``None`` when off.

    Raises:
        ValueError: the arm's ``data.target_mode`` is not an R2R mode (averaging
            recorruptions of an input the network was trained on unrecorrupted
            would score something no estimator defines), or the knob names the
            other mode's strength.
    """
    knob = config.validation.scoring.recorrupted_inference
    if knob is None:
        return None
    data = config.data
    mode = data.target_mode
    if mode not in ELIGIBLE_TARGET_MODES:
        raise ValueError(
            "validation.scoring.recorrupted_inference averages the model over "
            f"recorrupted inputs, which only an R2R-trained network is an estimator "
            f"for; data.target_mode is {mode!r} (eligible: {sorted(ELIGIBLE_TARGET_MODES)})."
        )
    foreign_field, foreign_message = _OTHER_MODES_STRENGTH[mode]
    if getattr(knob, foreign_field) is not None:
        raise ValueError(foreign_message)
    run = getattr(config, "run", None)
    fields: dict[str, Any] = {
        "alpha": None,
        "angle_deg": None,
        "covariance_source": None,
        "whitening": None,
    }
    fields.update(_MODE_FIELDS[mode](knob, data))
    return RecorruptionSpec(
        target_mode=mode,
        samples=int(knob.samples),
        seed=int(getattr(run, "seed", 0) or 0),
        **fields,
    )


def _r2r_fields(knob: Any, data: Any) -> dict[str, Any]:
    alpha = knob.alpha if knob.alpha is not None else data.r2r_alpha
    return {"alpha": float(alpha), "covariance_source": data.r2r_covariance_source}


def _nullspace_r2r_fields(knob: Any, data: Any) -> dict[str, Any]:
    angle = knob.angle_deg if knob.angle_deg is not None else data.r2r_split_angle_deg
    return {"angle_deg": float(angle), "whitening": data.r2r_whitening}


#: The spec fields each eligible ``target_mode`` fills (the others stay ``None``). The keys
#: are ``ELIGIBLE_TARGET_MODES``.
_MODE_FIELDS = {"r2r": _r2r_fields, "nullspace_r2r": _nullspace_r2r_fields}

#: Per eligible mode, the knob field that belongs to the OTHER mode and the refusal for it.
_OTHER_MODES_STRENGTH = {
    "r2r": (
        "angle_deg",
        "recorrupted_inference.angle_deg is the nullspace_r2r split angle; under "
        "target_mode 'r2r' the strength is recorrupted_inference.alpha.",
    ),
    "nullspace_r2r": (
        "alpha",
        "recorrupted_inference.alpha is the r2r draw strength; under target_mode "
        "'nullspace_r2r' the strength is recorrupted_inference.angle_deg.",
    ),
}


def draw_r2r_noise(
    chol: torch.Tensor, shape: tuple[int, ...], generator: torch.Generator
) -> torch.Tensor:
    """``z = L w`` with ``w`` circular white, per sample: ``(B, C, C)`` x ``(B, C, H, W)``."""
    b, c, h, w = shape
    real = torch.randn((b, c, h * w), generator=generator, device=chol.device, dtype=torch.float64)
    imag = torch.randn((b, c, h * w), generator=generator, device=chol.device, dtype=torch.float64)
    white = torch.complex(real, imag) / (2.0**0.5)
    return (chol @ white).reshape(b, c, h, w)


def _support(kspace: torch.Tensor) -> torch.Tensor:
    """``(B, 1, 1, W)`` bool: the phase-encode columns each sample acquired."""
    return (kspace.abs().sum(dim=(1, 2)) > 0)[:, None, None, :]


def _batch_covariance(batch_data: Any, source: str | None, like: torch.Tensor) -> torch.Tensor:
    """``(B, C, C)`` complex128 Sigma_n in raw units, from the arm's declared source."""
    from spectramr.infrastructure.physics.m4raw_noise import single_repetition_covariance

    b, c = like.shape[0], like.shape[1]
    if source in (None, "none", "committed"):
        cov = single_repetition_covariance(device=like.device, dtype=torch.complex128)
        if cov.shape[-1] != c:
            raise ValueError(
                f"the committed covariance describes {cov.shape[-1]} coils, the input {c}."
            )
        return cov.expand(b, c, c)
    cov = read_batch_field(batch_data, "noise_covariance")
    if not isinstance(cov, torch.Tensor):
        raise ValueError(
            f"recorrupted inference draws from the scan's own Sigma_n ({source}) and the "
            "validation batch carries no `noise_covariance`; the M4Raw loader publishes it "
            "on the validation split of a self-calibrated arm."
        )
    cov = cov.to(device=like.device, dtype=torch.complex128).reshape(-1, c, c)
    if cov.shape[0] != b:
        cov = cov.repeat_interleave(b // cov.shape[0], dim=0)
    return cov


class RecorruptedInference:
    """Runs a forward on ``J`` recorruptions of each validation input and averages."""

    def __init__(self, spec: RecorruptionSpec) -> None:
        self.spec = spec
        self._generators: dict[str, torch.Generator] = {}
        self._batch_key: int | None = None

    def _reseed_for(self, batch_key: int | None) -> None:
        """Restart every stream at ``(run.seed, batch_key)``; ``None`` continues."""
        if batch_key is None:
            return
        self._batch_key = int(batch_key)
        for generator in self._generators.values():
            generator.manual_seed(self._seed())

    def _generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        if key not in self._generators:
            self._generators[key] = torch.Generator(device=device).manual_seed(self._seed())
        return self._generators[key]

    def _seed(self) -> int:
        if self._batch_key is None:
            return self.spec.seed
        return self.spec.seed * 1_000_003 + self._batch_key

    def _r2r(self, input_batch: torch.Tensor, batch_data: Any) -> Callable[[], torch.Tensor]:
        y = as_complex_coils(input_batch, name="input")
        scale = read_batch_field(batch_data, "kspace_scale")
        if scale is None:
            raise ValueError(
                "recorrupted inference needs the per-sample `kspace_scale` to bring the raw-"
                "unit Sigma_n to the normalized input; the batch carries none."
            )
        scale = align_scale_to_batch(scale, y.shape[0], field="kspace_scale", device=y.device)
        cov = _batch_covariance(batch_data, self.spec.covariance_source, y)
        chol = torch.linalg.cholesky(cov) / scale.reshape(-1, 1, 1).to(torch.float64)
        support = _support(y)
        alpha = float(self.spec.alpha or 1.0)
        generator = self._generator(y.device)

        def recorrupt() -> torch.Tensor:
            z = draw_r2r_noise(chol, tuple(y.shape), generator).to(y.dtype)
            return as_interleaved(y + alpha * z * support)

        return recorrupt

    def _nullspace(self, input_batch: torch.Tensor, batch_data: Any) -> Callable[[], torch.Tensor]:
        from spectramr.infrastructure.physics.coil_split_r2r import (
            prepare_coil_split,
            split_kspace_prepared,
            whitener,
        )
        from spectramr.infrastructure.physics.fft_ops import ifft2c
        from spectramr.infrastructure.physics.m4raw_noise import single_repetition_covariance

        y = as_complex_coils(input_batch, name="input")
        maps, _ = coil_maps(batch_data, input_batch)
        maps = require_complex_maps(maps, y, metric="recorrupted_inference")
        support = _support(y)
        coils = y.shape[1]
        # One whitener for the batch, except a self-calibrated one, which is per
        # scan and so per sample; the split takes one whitener per call.
        if self.spec.whitening == "none":
            whiteners = [whitener(None, coils)]
        elif self.spec.whitening == "committed":
            whiteners = [whitener(single_repetition_covariance(), coils)]
        else:
            cov = _batch_covariance(batch_data, self.spec.whitening, y)
            whiteners = [whitener(cov[b], coils) for b in range(y.shape[0])]
        if "cpu" not in self._generators:
            self._generators["cpu"] = torch.Generator().manual_seed(self._seed())
        generator = self._generators["cpu"]
        angle = float(self.spec.angle_deg or 45.0)

        # Everything but the draw of v is fixed per slice; compute it once for all J samples.
        images = ifft2c(y)
        prepared = (
            [
                prepare_coil_split(images[b : b + 1], maps[b : b + 1], whitening=w)
                for b, w in enumerate(whiteners)
            ]
            if len(whiteners) > 1
            else [prepare_coil_split(images, maps, whitening=whiteners[0])]
        )

        def recorrupt() -> torch.Tensor:
            if len(prepared) == 1:
                first = split_kspace_prepared(
                    prepared[0], support, angle_deg=angle, generator=generator
                )
                return as_interleaved(first)
            halves = [
                split_kspace_prepared(p, support[b : b + 1], angle_deg=angle, generator=generator)
                for b, p in enumerate(prepared)
            ]
            return as_interleaved(torch.cat(halves, dim=0))

        return recorrupt

    def predict(
        self,
        forward: Callable[[torch.Tensor], torch.Tensor],
        input_batch: torch.Tensor,
        batch_data: Any,
        *,
        batch_key: int | None = None,
    ) -> torch.Tensor:
        """``mean_j forward(recorrupt_j(input_batch))``, J forwards run sequentially.

        ``batch_key`` (the validation batch index) reseeds the draws; ``None``
        continues the current stream.
        """
        self._reseed_for(batch_key)
        build = self._r2r if self.spec.target_mode == "r2r" else self._nullspace
        # The loop may hand over the TorchIO (B, C, H, W, D) layout; the draw is
        # a per-slice coil operation, so depth folds into the batch and unfolds
        # again before the forward, which expects the layout it was given.
        four_d, layout = fold_depth(input_batch)
        recorrupt = build(four_d, batch_data)
        total: torch.Tensor | None = None
        for _ in range(self.spec.samples):
            out = forward(unfold_depth(recorrupt(), layout).to(input_batch.dtype))
            total = out if total is None else total + out
        assert total is not None  # samples >= 1 by schema
        return total / self.spec.samples
