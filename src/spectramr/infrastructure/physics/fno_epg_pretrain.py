"""Fit, store and load the FNO-EPG surrogate that arms with ``use_fno_epg`` render through.

The surrogate stands in for the analytic EPG (:func:`simulate_differentiable_epg_fse`). A
surrogate with its default random weights is not that: arm 51 built one, loaded nothing and
gave its parameters to no optimiser, so its render was a frozen random function of the tissue
maps. Here it is fitted to the analytic signal at the effective echo over the T1, T2, B1+ and
echo-spacing ranges the render meets, stored with its measured fidelity, and loaded frozen.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from spectramr.infrastructure.physics.epg import simulate_differentiable_epg_fse
from spectramr.infrastructure.physics.fno_epg import FNOEPGSurrogate
from spectramr.shared.utils.safe_io import safe_torch_load

logger = logging.getLogger(__name__)

#: Echoes in the rendered train; the signal is read at the middle one.
ECHO_TRAIN_LENGTH = 16

#: Ranges the training set covers (ms, except B1+ which is a ratio).
T1_RANGE = (100.0, 5000.0)
T2_RANGE = (10.0, 2500.0)
B1_RANGE = (0.5, 1.5)  # the generator's B1+ head is sigmoid + 0.5
#: Echo spacing = TE / (ETL // 2): TE of 10..150 ms over 8 gaps.
TE_SPACING_RANGE = (1.0, 20.0)

_FORMAT = "fno_epg_surrogate/1"


def _log_uniform(lo: float, hi: float, n: int, gen: torch.Generator) -> torch.Tensor:
    u = torch.rand(n, generator=gen)
    return torch.exp(
        torch.log(torch.tensor(lo))
        + u * (torch.log(torch.tensor(hi)) - torch.log(torch.tensor(lo)))
    )


def sample_training_set(n: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """``n`` voxels with the analytic effective-echo signal at unit proton density.

    Proton density multiplies the surrogate's output outside the network, so the target is
    taken at ``rho = 1``. Each tensor is ``(n, 1, 1, 1)``.
    """
    gen = torch.Generator().manual_seed(seed)
    t1 = _log_uniform(*T1_RANGE, n, gen)
    t2 = torch.minimum(_log_uniform(*T2_RANGE, n, gen), t1)
    b1 = B1_RANGE[0] + torch.rand(n, generator=gen) * (B1_RANGE[1] - B1_RANGE[0])
    ts = TE_SPACING_RANGE[0] + torch.rand(n, generator=gen) * (
        TE_SPACING_RANGE[1] - TE_SPACING_RANGE[0]
    )
    out = {
        k: v.view(n, 1, 1, 1) for k, v in {"t1": t1, "t2": t2, "b1": b1, "te_spacing": ts}.items()
    }
    out["rho"] = torch.ones(n, 1, 1, 1)
    out["target"] = simulate_differentiable_epg_fse(
        rho=out["rho"],
        t1=out["t1"],
        t2=out["t2"],
        excitation_fa=90.0 * out["b1"],
        refocusing_fa=180.0 * out["b1"],
        echo_train_length=ECHO_TRAIN_LENGTH,
        te_spacing=out["te_spacing"],
    )
    return out


def _predict(net: FNOEPGSurrogate, d: Mapping[str, torch.Tensor]) -> torch.Tensor:
    out: torch.Tensor = net(
        d["rho"],
        d["t1"],
        d["t2"],
        d["b1"],
        te_spacing=d["te_spacing"],
        echo_train_length=ECHO_TRAIN_LENGTH,
    )
    return out


def fit(
    net: FNOEPGSurrogate,
    data: Mapping[str, torch.Tensor],
    steps: int,
    batch_size: int = 4096,
    lr: float = 2e-3,
    seed: int = 0,
) -> list[float]:
    """Minimise the squared error to the analytic signal; returns the loss of every step."""
    gen = torch.Generator().manual_seed(seed)
    n = data["target"].shape[0]
    device = next(net.parameters()).device
    data = {k: v.to(device) for k, v in data.items()}
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    losses = []
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),), generator=gen).to(device)
        batch = {k: v[idx] for k, v in data.items()}
        loss = torch.mean((_predict(net, batch) - batch["target"]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        losses.append(loss.detach())
    return [float(v) for v in torch.stack(losses)]


def fidelity(net: FNOEPGSurrogate, data: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Absolute error of the surrogate against the analytic signal (a signal in [0, 1])."""
    device = next(net.parameters()).device
    was_training = net.training
    net.eval()
    with torch.no_grad():
        batch = {k: v.to(device) for k, v in data.items()}
        err = (_predict(net, batch) - batch["target"]).abs().flatten().cpu()
    net.train(was_training)
    return {
        "mae": float(err.mean()),
        "rmse": float(err.pow(2).mean().sqrt()),
        "p99_abs": float(torch.quantile(err, 0.99)),
        "max_abs": float(err.max()),
    }


def _config(net: FNOEPGSurrogate) -> dict[str, int]:
    lift = net.lift[0]
    assert isinstance(lift, torch.nn.Linear)
    return {
        "tissue_channels": lift.in_features,
        "hidden_width": lift.out_features,
        "n_layers": len(net.spectral_layers),
        "max_etl": net.max_etl,
        "seq_param_dim": net.seq_param_dim,
    }


def save_fno_epg_surrogate(
    net: FNOEPGSurrogate, path: str | os.PathLike[str], measured_fidelity: Mapping[str, float]
) -> None:
    """Write the weights, the architecture to rebuild them, and the fidelity they reached."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": _FORMAT,
            "config": _config(net),
            "fidelity": dict(measured_fidelity),
            "echo_train_length": ECHO_TRAIN_LENGTH,
            "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        },
        path,
    )


def load_fno_epg_surrogate(
    path: str | os.PathLike[str], device: torch.device | str = "cpu"
) -> FNOEPGSurrogate:
    """Rebuild a stored surrogate, frozen and in eval mode (it is physics, not a learner)."""
    payload: Any = safe_torch_load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != _FORMAT:
        raise ValueError(f"{path} is not an FNO-EPG surrogate written by save_fno_epg_surrogate")
    if payload.get("echo_train_length") != ECHO_TRAIN_LENGTH:
        raise ValueError(
            f"{path} was fitted for an echo train of {payload.get('echo_train_length')} echoes; "
            f"the render reads a train of {ECHO_TRAIN_LENGTH}. Refit it."
        )
    net = FNOEPGSurrogate(**payload["config"])
    net.load_state_dict(payload["state_dict"])
    net.requires_grad_(False)
    return net.to(device).eval()


def check_protocol_within_fit(protocol: Mapping[str, Mapping[str, float]]) -> None:
    """Refuse a declared protocol whose echo spacings the surrogate was not fitted on.

    The render spaces the train at ``TE / (ETL // 2)``; outside ``TE_SPACING_RANGE`` the
    surrogate extrapolates, which nothing would report.
    """
    lo, hi = TE_SPACING_RANGE
    for contrast, params in protocol.items():
        spacing = float(params["TE"]) / max(1, ECHO_TRAIN_LENGTH // 2)
        if not lo <= spacing <= hi:
            raise ValueError(
                f"bloch_protocol_by_contrast[{contrast!r}] TE={params['TE']} ms gives an echo "
                f"spacing of {spacing:.2f} ms, outside the spacings the FNO-EPG surrogate was "
                f"fitted on ({lo}-{hi} ms)"
            )


def build_fno_epg_surrogate(
    model_kwargs: Mapping[str, Any], device: torch.device | str
) -> FNOEPGSurrogate | None:
    """The surrogate an arm declares, or ``None`` for an arm that does not use one.

    Raises:
        ValueError: ``use_fno_epg`` without ``fno_epg_weights`` (a random network is not an
            EPG), or ``fno_epg_weights`` without ``use_fno_epg`` (a knob nothing reads).
    """
    weights = model_kwargs.get("fno_epg_weights")
    if not model_kwargs.get("use_fno_epg", False):
        if weights:
            raise ValueError(
                "model_kwargs.fno_epg_weights is set but use_fno_epg is not, so nothing reads it"
            )
        return None
    if not weights:
        raise ValueError(
            "use_fno_epg needs model_kwargs.fno_epg_weights, a surrogate fitted to the analytic "
            "EPG; a default-initialised one is a random function. Write one with "
            "scripts/physics/pretrain_fno_epg.py."
        )
    net = load_fno_epg_surrogate(weights, device)
    # The weights are not part of the run checkpoint, so name the exact file that rendered:
    # a regenerated surrogate under the same path otherwise changes a resumed run unseen.
    digest = hashlib.sha256(Path(weights).read_bytes()).hexdigest()
    logger.info("FNO-EPG surrogate loaded from %s (sha256 %s)", weights, digest)
    return net
