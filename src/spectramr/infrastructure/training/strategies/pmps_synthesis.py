"""PMPS Stage C: the sampling loop that renders and writes the synthetic paired corpus.

One pass: reverse-sample tissue maps from the frozen Phase-1 prior, render them at a
``(P_ULF, P_HF)`` protocol pair through :class:`PairedSynthesisGenerator` with the frozen Phase-2
operators, score each pair, and write it through :mod:`spectramr.data.pmps_shard_writer`.
``PairedSynthesisStrategy`` resolves the knobs and the frozen modules; this module owns the loop.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from spectramr.config.schemas.training.pmps import (
    AcquisitionParam,
    ProtocolSamplingConfig,
    TrainingConfigPairedSynthesis,
)
from spectramr.data.pmps_shard_writer import write_manifest, write_pair
from spectramr.infrastructure.checkpointing.restore import weights
from spectramr.infrastructure.training.schedulers.diffusion_scheduler import DiffusionScheduler

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_RENDERS_PER_PAIR",
    "ULF_B0_CEILING_T",
    "SynthesisGeometry",
    "load_frozen",
    "protocol_pairs",
    "require_paired_synthesis_config",
    "run_paired_synthesis",
    "sample_tissue_prior",
]

#: Protocols below this field are the ULF side of a pair, the rest the HF side.
ULF_B0_CEILING_T = 1.0
#: Renders allowed per requested accepted pair before the run gives up.
MAX_RENDERS_PER_PAIR = 10


@dataclass(frozen=True)
class SynthesisGeometry:
    """Batch size and in-plane size of the rendered tissue maps."""

    batch_size: int
    height: int
    width: int


def require_paired_synthesis_config(training: Any) -> TrainingConfigPairedSynthesis:
    """Validate the arm's ``training`` extras against the Stage-C schema, refusing unknown keys.

    ``training:`` loads through ``TrainingStrategyConfigSchema`` (``extra="allow"``), so the
    Stage-C keys arrive unvalidated; this is where they are read and checked.
    """
    extras = dict(getattr(training, "model_extra", None) or {})
    unknown = sorted(set(extras) - set(TrainingConfigPairedSynthesis.model_fields))
    if unknown:
        raise ValueError(f"PairedSynthesisStrategy does not read training.{unknown}")
    return TrainingConfigPairedSynthesis.model_validate(extras)


def _protocol(p: AcquisitionParam) -> dict[str, float]:
    if p.contrast_type != "gradient_echo" or p.TI != -1.0:
        raise ValueError(
            f"protocol {p.name!r}: the generator renders spoiled gradient echo only, got "
            f"contrast_type={p.contrast_type!r}, TI={p.TI}; declare gradient_echo with TI -1"
        )
    return {"te": p.TE, "tr": p.TR, "fa_rad": math.radians(p.FA), "b0_t": p.B0}


def protocol_pairs(cfg: ProtocolSamplingConfig) -> list[tuple[str, dict, str, dict]]:
    """The ``(ulf_name, p_ulf, hf_name, p_hf)`` pairs synthesis cycles through, in order."""
    if cfg.mode != "fixed":
        raise ValueError(
            f"protocol_sampling.mode={cfg.mode!r} draws protocols from a per-site acquisition "
            "corpus (the real paired manifest's sidecars), which no arm supplies yet; declare "
            "mode: fixed with fixed_protocols"
        )
    if cfg.site_weights:
        raise ValueError("protocol_sampling.site_weights is read only by mode site_balanced")
    ulf = [p for p in cfg.fixed_protocols if p.B0 < ULF_B0_CEILING_T]
    hf = [p for p in cfg.fixed_protocols if p.B0 >= ULF_B0_CEILING_T]
    if not ulf or not hf:
        raise ValueError(
            f"protocol_sampling.fixed_protocols needs at least one protocol below and one at or "
            f"above {ULF_B0_CEILING_T} T; got {len(ulf)} ULF and {len(hf)} HF"
        )
    return [(u.name, _protocol(u), h.name, _protocol(h)) for u, h in itertools.product(ulf, hf)]


def load_frozen(module: nn.Module, checkpoint: Path, part: str, device: torch.device) -> str:
    """Load ``part`` of ``checkpoint`` (a file, or a run directory's best) exactly; freeze."""
    ref: Any = {"run_dir": checkpoint, "select": "best"} if checkpoint.is_dir() else checkpoint
    report = weights(ref, part, module, policy="exact")
    module.to(device).eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return str(report.source)


@torch.no_grad()
def sample_tissue_prior(
    prior: nn.Module,
    scheduler: DiffusionScheduler,
    shape: tuple[int, ...],
    *,
    prediction_type: str,
    b0_t: float,
) -> torch.Tensor:
    """Full ancestral DDPM chain on the training schedule; returns the t = 0 sample."""
    device = scheduler.device
    batch = shape[0]
    x = torch.randn(shape, device=device)
    b0 = torch.full((batch,), b0_t, device=device)
    sqrt_ab = scheduler.sqrt_alphas_cumprod.to(device)
    sqrt_1m_ab = scheduler.sqrt_one_minus_alphas_cumprod.to(device)
    for t in range(scheduler.num_timesteps - 1, -1, -1):
        t_vec = torch.full((batch,), t, dtype=torch.long, device=device)
        out = prior(x, timesteps=t_vec, b0=b0)
        if out.shape != x.shape:
            raise ValueError(f"prior returned {tuple(out.shape)}, expected {tuple(x.shape)}")
        # A sample-predicting prior gives x0; the step takes the eps that x0 implies.
        eps = out if prediction_type == "epsilon" else (x - sqrt_ab[t] * out) / sqrt_1m_ab[t]
        x = scheduler.denoise_step(x, eps, t_vec)
    return x


def run_paired_synthesis(
    generator: nn.Module,
    prior: nn.Module,
    scheduler: DiffusionScheduler,
    residual_loss: nn.Module,
    cfg: TrainingConfigPairedSynthesis,
    geometry: SynthesisGeometry,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Render until ``cfg.n_synthetic_pairs`` pairs pass the gate, write them, return a summary.

    Stage D's cells count usable synthetic pairs, so ``n_synthetic_pairs`` counts accepted ones;
    the run raises after ``MAX_RENDERS_PER_PAIR`` renders per requested pair. Each batch shares
    one protocol pair, cycling through :func:`protocol_pairs`. A pair is accepted when the
    generator's noise-standardised residual is under the threshold; the declared
    ``per_pair_consistency_residual`` (L1 of each corrupted image against its clean render, the
    larger of the two) is recorded beside it. Rejected pairs are written only with
    ``write_rejected``. Accepted pairs fill ``parallel_shards`` contiguous shards in order.
    """
    combos = protocol_pairs(cfg.protocol_sampling)
    root = Path(cfg.output_root)
    n = cfg.n_synthetic_pairs
    budget = MAX_RENDERS_PER_PAIR * n
    per_pair = torch.vmap(residual_loss)
    records: list[dict[str, Any]] = []
    n_accepted = 0
    batch_idx = 0
    preview: dict[str, torch.Tensor] = {}
    while n_accepted < n:
        if len(records) >= budget:
            write_manifest(root, records, provenance)
            raise ValueError(
                f"PMPS Stage C accepted {n_accepted} of {len(records)} rendered pairs at "
                f"defensibility_threshold={cfg.defensibility_threshold}, short of "
                f"n_synthetic_pairs={n}; the manifest under {root} lists their residuals"
            )
        size = min(geometry.batch_size, budget - len(records))
        ulf_name, p_ulf, hf_name, p_hf = combos[batch_idx % len(combos)]
        batch_idx += 1
        theta = sample_tissue_prior(
            prior,
            scheduler,
            (size, 3, geometry.height, geometry.width),
            prediction_type=cfg.tissue_prior.prediction_type,
            b0_t=p_hf["b0_t"],
        )
        with torch.no_grad():
            out = generator.generate(theta, p_ulf=p_ulf, p_hf=p_hf)
            ppcr = torch.maximum(
                per_pair(out["ulf"], out["ulf_clean"]), per_pair(out["hf"], out["hf_clean"])
            )
        # One host copy per batch.
        host = {k: out[k].detach().float().cpu().numpy() for k in ("ulf", "hf", "theta")}
        scores = {
            "residual": out["residual"],
            "residual_ulf": out["residual_ulf"],
            "residual_hf": out["residual_hf"],
            "per_pair_consistency_residual": ppcr,
        }
        scores_host = {k: v.detach().float().cpu().tolist() for k, v in scores.items()}
        flags = out["defensibility_flag"].cpu().tolist()
        for i in range(size):
            accepted = bool(flags[i]) and n_accepted < n
            index = len(records)
            record: dict[str, Any] = {
                "index": index,
                "shard": min(n_accepted, n - 1) * cfg.parallel_shards // n,
                "accepted": accepted,
                "p_ulf": ulf_name,
                "p_hf": hf_name,
                **{k: v[i] for k, v in scores_host.items()},
                "files": [],
            }
            n_accepted += int(accepted)
            if accepted and not preview:
                preview = {k: out[k][i : i + 1].detach().float().cpu() for k in ("ulf", "hf")}
            if accepted or (cfg.write_rejected and not flags[i]):
                record["files"] = write_pair(
                    root,
                    shard=record["shard"],
                    index=index,
                    accepted=accepted,
                    arrays={k: v[i] for k, v in host.items()},
                    fmt=cfg.output_format,
                )
            records.append(record)
    manifest = write_manifest(root, records, provenance)
    kept = [r["per_pair_consistency_residual"] for r in records if r["accepted"]]
    summary = {
        "n_rendered": len(records),
        "n_accepted": n_accepted,
        "acceptance_rate": n_accepted / len(records),
        "per_pair_consistency_residual": sum(kept) / len(kept),
        "manifest": str(manifest),
    }
    logger.info("PMPS Stage C: %s", summary)
    # The first accepted (x_ULF, x_HF), shown by validation in place of a per-batch forward.
    return {**summary, "preview": preview}
