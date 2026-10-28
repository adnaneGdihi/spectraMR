"""Quantitative maps from noise-unit T1w/T2w/FLAIR with header protocols (exp 32, direction 1).

``bloch_parameter_bottleneck`` predicts (rho, T1, T2) from the co-registered triplet; each
series is rendered with its own header protocol (``tse_forward``) and compared with the
acquired series. ``gain_mode: calibrated`` fixes the per-series gain ratios measured by the
series-gain gate and learns one global scale; ``projected`` fits a gain per image instead,
the image-domain control.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import torch
import torch.nn as nn

from spectramr.data.batch_types import read_batch_field
from spectramr.infrastructure.distributed.distributed_training import RankUtility
from spectramr.infrastructure.distributed.owned_modules import OWNED_CALL, OWNED_PARAM_GROUP
from spectramr.infrastructure.physics.tse_forward import tse_signal
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.owned_param_groups import add_owned_param_group
from spectramr.infrastructure.training.strategies.qmap_availability import (
    CONTRAST_KEYS,
    draw_availability,
    read_series_protocol,
    unroll_protocol,
    validate_contrast_dropout,
)
from spectramr.infrastructure.training.strategies.qmap_conformal import SubjectResiduals
from spectramr.infrastructure.training.strategies.qmap_gauge_prior import (
    gain_ratio_penalty,
    parse_gain_ratio_prior,
)
from spectramr.infrastructure.training.strategies.qmap_heldout import heldout_pass
from spectramr.infrastructure.training.strategies.qmap_rep_split import repetition_risk
from spectramr.infrastructure.training.strategies.qmap_scores import (
    coefficient_of_variation,
    white_matter_medians,
)
from spectramr.models.losses.registry import create_loss

CONTRASTS = ("T1", "T2", "FLAIR")
_SIGNAL_FLOOR = 3.0  # noise units: a voxel above 3 sigma carries signal


class CalibratedQMapStrategy(BaseTrainingStrategy):
    # parallel_compat: The global gain is a bare group on opt_g, and every training step renders
    # through generator.render_triplet / predict_parameters, outside the wrapped forward.
    parallel_compat = frozenset({"replicated"})

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        own = (("generator.render_triplet", OWNED_CALL), ("_log_global_gain", OWNED_PARAM_GROUP))
        return (*super().owned_module_declarations(config), *own)

    #: The target is the acquired T1/T2/FLAIR triplet this strategy renders, not the
    #: model's 1-channel output.
    renders_target_series: ClassVar[bool] = True
    #: The model_kwargs keys this strategy reads (not the model); the advisory skips them.
    reads_model_kwargs: ClassVar[frozenset[str]] = frozenset(
        {"gain_mode", "series_gain", "contrast_dropout", "gain_ratio_prior"}
    )
    #: Owns its objective (render loss + gauge prior): an empty built stack is by design.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"calibrated_gain_bloch"})
    folds_image_losses: ClassVar[bool | None] = False

    def __init__(self, env: Any = None, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        self._configure_calibrated_qmap()

    def _configure_calibrated_qmap(self) -> None:
        mk = dict(self.config.model.model_kwargs or {})
        mode = mk.get("gain_mode")
        if mode not in ("calibrated", "projected"):
            raise ValueError(
                f"model_kwargs.gain_mode={mode!r}; expected 'calibrated' or 'projected'"
            )
        self._loss = create_loss("calibrated_gain_bloch", gain_mode=mode)
        self._dropout = validate_contrast_dropout(mk.get("contrast_dropout", 0.0))
        self._gauge_prior = parse_gain_ratio_prior(mk.get("gain_ratio_prior"))
        self._residuals = SubjectResiduals()
        if self._gauge_prior is not None and mode != "projected":  # calibrated fixes the ratios
            raise ValueError("model_kwargs.gain_ratio_prior applies to gain_mode='projected'")
        self._gains: torch.Tensor | None = None
        self._log_global_gain: nn.Parameter | None = None
        self._gain_initialised = False
        self._protocol: list[dict[str, float]] | None = None
        self._wm: dict[str, list[tuple[float, float]]] = {}
        if mode != "calibrated":
            return
        if RankUtility.get_world_size() > 1:
            # The global gain is a bare parameter outside the wrapped generator, so DDP and
            # FSDP never reduce its gradient and each rank would learn a different scale.
            raise ValueError(
                "gain_mode='calibrated' is single-process: its global gain is not synchronised "
                "across ranks. Use gain_mode='projected' or run on one process."
            )
        gains = dict(mk.get("series_gain") or {})
        if set(gains) != set(CONTRASTS):
            raise ValueError(f"model_kwargs.series_gain must name {CONTRASTS}, got {sorted(gains)}")
        if any(float(v) <= 0 for v in gains.values()):
            raise ValueError("model_kwargs.series_gain values must be > 0")
        self._gains = torch.tensor([float(gains[c]) for c in CONTRASTS], device=self.device)
        # The gate fixes gain ratios; apparent PD is bounded to [0, 1] while the images
        # are in noise units, so one global scale is learned (the gauge PD keeps anyway).
        # Set from the first batch's data scale; decay would pull it back towards 1.
        self._log_global_gain = nn.Parameter(torch.zeros((), device=self.device))
        add_owned_param_group(
            self,
            self.env.opt_g,
            {"params": [self._log_global_gain], "name": "global_gain", "weight_decay": 0.0},
            owner="its global gain",
        )

    @staticmethod
    def _rows(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            if x.shape[-1] != 1:
                raise ValueError(
                    f"the triplet must arrive one slice at a time, got depth {x.shape[-1]}"
                )
            x = x[..., 0]
        return x.float()

    def _series_protocol(self, batch: Any) -> list[dict[str, float]]:
        """Host-side protocol per series, read once; the dataset refuses mixed protocols."""
        if self._protocol is None:
            self._protocol = read_series_protocol(batch)
        return self._protocol

    def _require_same_protocol(self, batch: Any) -> None:
        """Validation renders with the cached train protocol, so a differing split must raise."""
        if self._protocol is not None and read_series_protocol(batch) != self._protocol:
            raise ValueError(  # rendering it with the training protocol would corrupt val_psnr
                f"the validation batch's acquisition protocol differs from training's: "
                f"{read_series_protocol(batch)} vs {self._protocol}"
            )

    def _render(
        self, x: torch.Tensor, batch: Any, available: torch.Tensor | None = None
    ) -> tuple[dict, list[torch.Tensor]]:
        """Maps and one render per series; ``available`` hides contrasts from the model."""
        generator = self.inner_for_call(self.env.generator, "generator.render_triplet")
        if getattr(generator, "takes_protocol", False):
            protocol = unroll_protocol(self._series_protocol(batch))
            return generator.render_triplet(x, protocol, available)
        if available is not None:  # a model without the protocol sees zeros for a hidden series
            x = x * available.view(*available.shape, 1, 1).to(x.dtype)
        maps = generator.predict_parameters(x)
        renders = [
            tse_signal(
                maps["rho"],
                maps["t1"],
                maps["t2"],
                family=int(p["sequence_family"]),
                tr=p["acq_tr"],
                te=p["acq_te"],
                ti=p["acq_ti"],
                esp=p["acq_esp"],
                etl=int(p["acq_etl"]),
                refocus_deg=p["acq_refocus_deg"],
            )
            for p in self._series_protocol(batch)
        ]
        return maps, renders

    def _gain(self, k: int) -> torch.Tensor | None:
        if self._gains is None:
            return None
        return self._log_global_gain.exp() * self._gains[k]

    @torch.no_grad()
    def _initialise_gain(
        self, render_t1: torch.Tensor, y_t1: torch.Tensor, mask: torch.Tensor
    ) -> None:
        """Start the global gain at the T1 series' least-squares scale (T1's ratio is 1).

        Bounded apparent PD against noise-unit targets otherwise starts the gain about two
        orders of magnitude low, and the maps bend to make up the scale while it climbs.
        """
        w = mask.to(render_t1.dtype)
        scale = (render_t1 * y_t1 * w).sum() / (render_t1 * render_t1 * w).sum().clamp_min(1e-12)
        self._log_global_gain.copy_(scale.clamp_min(1e-6).log())
        self._gain_initialised = True

    @staticmethod
    def _signal_masks(y: torch.Tensor) -> list[torch.Tensor]:
        """One signal mask per series: voxels above the floor in THAT series.

        A T1-only mask, applied to all three, dropped the voxels that are dark in T1 but
        bright in T2 or FLAIR (CSF, lesions), so the T2 fit never saw the tissue that
        constrains it and the T2 map drifted to its range bound there.
        """
        return [y[:, k : k + 1] > _SIGNAL_FLOOR for k in range(len(CONTRASTS))]

    def _compute_losses_impl(self, input_batch, target_batch, epoch, **kwargs) -> dict:
        x, y = self._rows(input_batch), self._rows(target_batch)
        # Hidden series are still supervised: the loss below covers all three.
        available = draw_availability(x.shape[0], self._dropout, x.device)
        _, renders = self._render(x, kwargs.get("batch"), available)
        masks = self._signal_masks(y)
        if self._gains is not None and not self._gain_initialised:
            self._initialise_gain(renders[0], y[:, :1], masks[0])
        total = sum(
            self._loss(
                None, y[:, k : k + 1], context={"render": r, "mask": masks[k], "gain": self._gain(k)}
            )
            for k, r in enumerate(renders)
        )
        if self._gauge_prior is not None:  # closes the two gain directions (qmap_gauge_prior)
            ys = [y[:, k : k + 1] for k in range(len(CONTRASTS))]
            total = total + gain_ratio_penalty(renders, ys, masks, self._gauge_prior)
        return {"g_total_loss": total}

    def load_strategy_state_dict(
        self, state: dict[str, Any] | None, *, load_optimizer: bool = True
    ) -> None:
        """Restore the owned gain, and keep it: the first batch must not re-seed a resumed run.

        ``load_optimizer`` is resume's gate (#2964); this strategy owns no optimizer.
        """
        super().load_strategy_state_dict(state, load_optimizer=load_optimizer)
        if self._log_global_gain is not None and "_log_global_gain" in (state or {}).get(
            "params", {}
        ):
            self._gain_initialised = True

    def _psnr(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        computed = self._get_validation_metrics_computer(self.config).compute(pred, target)
        return torch.as_tensor(computed["psnr"])

    def _scaled(
        self, render: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, k: int
    ) -> torch.Tensor:
        gain = self._gain(k)
        if gain is None:  # projected control: the per-image least-squares gain
            w = mask.to(render.dtype)
            num = (render * target * w).flatten(1).sum(1)
            gain = (num / (render * render * w).flatten(1).sum(1).clamp_min(1e-12)).view(
                -1, 1, 1, 1
            )
        return gain * render

    @torch.no_grad()
    def validation_step(self, input_batch, target_batch, batch_data: Any = None, **kwargs) -> dict:
        self.env.generator.eval()
        self._require_same_protocol(batch_data)
        x, y = self._rows(input_batch), self._rows(target_batch)
        maps, renders = self._render(x, batch_data)
        masks = self._signal_masks(y)
        psnr = torch.stack(
            [
                self._psnr(self._scaled(r, y[:, k : k + 1], masks[k], k), y[:, k : k + 1])
                for k, r in enumerate(renders)
            ]
        )
        subjects = read_batch_field(batch_data, "subject_id")
        if subjects is None:
            raise ValueError(
                "validation needs each sample's subject_id for the across-subject spread"
            )
        if maps:  # a physics-free control (masked_contrast_unet) has no tissue maps
            wm = white_matter_medians(maps, y)
            self._record_white_matter(list(subjects), wm["t1"], wm["t2"])
        loo = heldout_pass(
            lambda available: self._render(x, batch_data, available)[1],
            y,
            masks,
            fitted_prediction=lambda j, r: self._scaled(r, y[:, j : j + 1], masks[j], j),
            psnr=lambda pred, target: float(self._psnr(pred, target)),
            prior=self._gauge_prior,
            residuals=self._residuals,
            subjects=subjects,
        )
        noise_power = read_batch_field(batch_data, "noise_power")
        if noise_power is not None:  # repetition split (R1): a reference-free risk
            loo.update(repetition_risk(renders, x, y, noise_power))
        return {"val_psnr": float(psnr.mean()), **loo}

    def _record_white_matter(self, subjects: list[str], t1: torch.Tensor, t2: torch.Tensor) -> None:
        """Keep each sample's white-matter medians under its subject until the pass ends."""
        for subject, a, b in zip(subjects, t1.tolist(), t2.tolist(), strict=True):
            if math.isfinite(a) and math.isfinite(b):  # an empty mask yields NaN: skip the slice
                self._wm.setdefault(str(subject), []).append((a, b))

    def reset_validation_metrics(self) -> None:
        super().reset_validation_metrics()
        self._wm = {}
        self._residuals.reset()

    def finalize_validation(self) -> dict[str, float]:
        """The pre-registered quantity: spread across subjects of per-subject WM medians."""
        out = dict(super().finalize_validation())
        if self._wm:
            # Interpolating medians: torch.median takes the lower middle on even counts.
            per_subject = torch.stack(
                [torch.quantile(torch.tensor(v), 0.5, dim=0) for v in self._wm.values()]
            )
            out["val_median_t1_wm"] = float(torch.quantile(per_subject[:, 0], 0.5))
            out["val_median_t2_wm"] = float(torch.quantile(per_subject[:, 1], 0.5))
            out["val_cv_t1_wm"] = float(coefficient_of_variation(per_subject[:, 0]))
            out["val_cv_t2_wm"] = float(coefficient_of_variation(per_subject[:, 1]))
        # Subject-level conformal intervals of the prior-gain synthesis (qmap_conformal).
        out.update(self._residuals.summary(alpha=0.1, beta=0.1, n_splits=5, names=CONTRAST_KEYS))
        return out
