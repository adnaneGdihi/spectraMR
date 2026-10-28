"""Precomputed self-calibrated ``Sigma_n`` for the M4Raw repetition route (#3117).

``r2r_covariance_source: self_calibrated`` and ``r2r_whitening: self_calibrated`` fit
each group's noise covariance from its coil null space (ESPIRiT on the central
slices of repetition 0). The fit lived in a per-process cache, so every DataLoader
worker repeated it for every group it touched: with four workers the first epoch
paid four fits per group and was bound by them. A store holds the fits once:
``<root>/noise_covariance.json``, keyed by the group (:func:`covariance_key`),
written by ``scripts/precompute_m4raw_noise_covariance.py`` with the same
:func:`~spectramr.data.datasets.m4raw_dataset.fit_scan_noise_covariance` the loader
falls back to. The header records everything that changes the fit; a store made
with other settings is refused.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

import torch

from spectramr.data.datasets.m4raw_coil_maps import map_key

__all__ = [
    "VALID_NOISE_COVARIANCE_MISSING",
    "NoiseCovarianceMismatchError",
    "NoiseCovarianceMissingError",
    "NoiseCovarianceStore",
    "covariance_key",
    "fit_header",
    "read_covariances",
    "write_covariances",
]

#: ``raise`` stops on a group absent from the store; ``fit`` runs the loader's own fit.
VALID_NOISE_COVARIANCE_MISSING: frozenset[str] = frozenset({"raise", "fit"})

_FILENAME = "noise_covariance.json"
_FORMAT = 1


class NoiseCovarianceMismatchError(ValueError):
    """The store was fitted with other settings, or on another file, than the loader's."""


class NoiseCovarianceMissingError(KeyError):
    """A group the loader fits has no entry in the store."""


def covariance_key(fitted_file: str | Path) -> str:
    """The store key of a group, from the repetition its ``Sigma_n`` is fitted on."""
    return map_key(Path(fitted_file).stem)


def fit_header(support_policy: str) -> dict[str, object]:
    """Every setting the stored fit depends on; a store is read only when they all match."""
    from spectramr.data.datasets.m4raw_dataset import _SIGMA_FIT_SLICES
    from spectramr.infrastructure.physics.scan_coil_calibration import DEFAULT_CALIBRATION

    return {
        "format": _FORMAT,
        "fit_slices": _SIGMA_FIT_SLICES,
        "support_policy": support_policy,
        "calibration": {k: DEFAULT_CALIBRATION[k] for k in sorted(DEFAULT_CALIBRATION)},
    }


def _encode(covariance: torch.Tensor) -> dict[str, list[list[float]]]:
    c = covariance.to(torch.complex64)
    return {"re": c.real.tolist(), "im": c.imag.tolist()}


def _decode(entry: Mapping[str, object]) -> torch.Tensor:
    return torch.complex(
        torch.tensor(entry["re"], dtype=torch.float32),
        torch.tensor(entry["im"], dtype=torch.float32),
    )


def read_covariances(root: str | Path, support_policy: str) -> dict[str, tuple[str, torch.Tensor]]:
    """``{group key: (fitted file name, Sigma_n)}`` under ``root``; ``{}`` when none was written."""
    path = Path(root) / _FILENAME
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    expected = fit_header(support_policy)
    header = {k: payload.get(k) for k in expected}
    if header != expected:
        raise NoiseCovarianceMismatchError(
            f"{path} was fitted with {header}, this loader fits with {expected}; write the "
            "store again with scripts/precompute_m4raw_noise_covariance.py."
        )
    return {k: (str(v["file"]), _decode(v)) for k, v in payload["groups"].items()}


def write_covariances(
    root: str | Path, groups: Mapping[str, tuple[str, torch.Tensor]], support_policy: str
) -> Path:
    """Write ``groups`` to ``root`` atomically, so a loader never reads a half-written file."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / _FILENAME
    payload = {
        **fit_header(support_policy),
        "groups": {k: {"file": f, **_encode(c)} for k, (f, c) in sorted(groups.items())},
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)
    return path


class NoiseCovarianceStore:
    """Read side of the store: one lookup per group, loaded once with the dataset."""

    def __init__(self, groups: Mapping[str, tuple[str, torch.Tensor]], missing: str) -> None:
        if missing not in VALID_NOISE_COVARIANCE_MISSING:
            raise ValueError(
                f"[M4Raw] Unknown noise_covariance_missing: {missing!r}. "
                f"Valid: {sorted(VALID_NOISE_COVARIANCE_MISSING)}"
            )
        self._groups = dict(groups)
        self._missing = missing

    @classmethod
    def declared(cls, root: str | Path, missing: str, support_policy: str) -> NoiseCovarianceStore:
        """The store under ``root``; an empty one is an error, not an empty cache."""
        groups = read_covariances(root, support_policy)
        if not groups:
            raise FileNotFoundError(
                f"[M4Raw] data.noise_covariance_root={str(root)!r} holds no {_FILENAME}; run "
                "scripts/precompute_m4raw_noise_covariance.py --out that directory."
            )
        return cls(groups, missing)

    def get(self, fitted_file: str | Path) -> torch.Tensor | None:
        """The group's ``Sigma_n``, or ``None`` under ``fit`` when it has none; raises under ``raise``."""
        key = covariance_key(fitted_file)
        entry = self._groups.get(key)
        if entry is None:
            if self._missing == "raise":
                raise NoiseCovarianceMissingError(
                    f"[M4Raw] no precomputed Sigma_n for group {key!r}; extend the store with "
                    "scripts/precompute_m4raw_noise_covariance.py or set "
                    "data.noise_covariance_missing: fit."
                )
            return None
        stored_file, covariance = entry
        if stored_file != Path(fitted_file).name:
            raise NoiseCovarianceMismatchError(
                f"[M4Raw] the store fitted group {key!r} on {stored_file}, the loader fits it "
                f"on {Path(fitted_file).name}."
            )
        return covariance.clone()
