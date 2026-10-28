"""Fail when a code change alters any 2-D degradation operator's output.

Splitting the degradation registry into a package and adding 3-D support both
claim to leave every 2-D output bit-identical. This hashes each registry key
(plus ``DigitalTwinSimulator`` and ``tio_physics.DigitalTwinDegradation``) on
CPU float32 and compares against a committed fixture.

Regenerate ONLY on review, alongside an intentional behaviour change, and name
the operators that moved in the commit. A new registry axis uses ``--add``: it
writes only the ADDED cases and refuses if any existing case changed or vanished. Hashes are only comparable on the reference
machine: an environment-fingerprint mismatch raises instead of passing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ci.degradation_golden_inputs import (  # noqa: E402
    SHAPES,
    case_shapes,
    declared_exemptions,
    exemption_errors,
    required_kwargs,
    validate_required_inputs,
)

FIXTURE = REPO_ROOT / "tests/fixtures/degradation_golden_2d.json"

THETAS = (0.0, 0.37, 1.0)
SEEDS = (0, 7)
KWARG_VARIANTS = {
    "chemical_shift": {"b0_T": 1.5, "bw_per_pixel_hz": 200},
    "susceptibility": {"slice_plane": "axial"},
    "partial_fourier": {"axis": "kx"},
    "t2star_blur": {"echo": "gradient"},
}
SIM_CONFIGS = {
    "default_nomotion_nob0_nob1": {
        "enable_motion": False,
        "enable_b0": False,
        "enable_b1": False,
    },
    "undersampling_r4": {
        "enable_motion": False,
        "enable_b0": False,
        "enable_b1": False,
        "enable_undersampling": True,
        "acceleration": 4.0,
    },
    "b1_gibbs_spike": {
        "enable_motion": False,
        "enable_b0": False,
        "enable_b1": True,
        "enable_gibbs": True,
        "enable_spike_noise": True,
    },
}


def _phantom(shape: tuple[int, ...], is_complex: bool, seed: int = 123) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    re = torch.randn(shape, generator=gen, dtype=torch.float32)
    if not is_complex:
        return re
    return torch.complex(re, torch.randn(shape, generator=gen, dtype=torch.float32))


def _digest(out: torch.Tensor) -> dict[str, object]:
    out = out.detach().cpu().contiguous()
    return {
        "sha256": hashlib.sha256(out.numpy().tobytes()).hexdigest(),
        "shape": list(out.shape),
        "dtype": str(out.dtype),
    }


def _record(fn, *args, **kwargs) -> dict[str, object]:
    return _digest(fn(*args, **kwargs))


def uncovered_axes(registry: Iterable[str], covered: Iterable[str]) -> set[str]:
    """Registry axes with no golden case, volume-only ones excepted."""
    return set(registry) - declared_exemptions()["volume-only"] - set(covered)


def _registry_cases() -> dict[str, dict[str, object]]:
    from spectramr.infrastructure.physics.degradations import (
        DEGRADATION_REGISTRY,
        apply_degradation,
    )

    inputs = {shape: _phantom(shape, cplx) for shape, cplx in SHAPES}
    cases: dict[str, dict[str, object]] = {}
    for name in sorted(DEGRADATION_REGISTRY):
        variants = {"": {}}
        if name in KWARG_VARIANTS:
            variants["+kw"] = KWARG_VARIANTS[name]
        for tag, kwargs in variants.items():
            for shape in case_shapes(name):
                x = inputs[shape]
                for theta in THETAS:
                    for seed in SEEDS:
                        key = f"{name}{tag}|{'x'.join(map(str, shape))}|t{theta}|s{seed}"
                        torch.manual_seed(0)
                        cases[key] = _record(
                            apply_degradation,
                            name,
                            x.clone(),
                            theta,
                            seed,
                            **{**required_kwargs(name, x), **kwargs},
                        )
    return cases


def _simulate(sim, x: torch.Tensor) -> torch.Tensor:
    return sim(x, snr_target=20.0, degradation_only=True, seed=3)[0]


def _simulator_cases() -> dict[str, dict[str, object]]:
    from spectramr.infrastructure.physics.digital_twin_simulator import DigitalTwinSimulator

    x = _phantom((1, 1, 32, 32), True)
    cases = {}
    for label, cfg in SIM_CONFIGS.items():
        torch.manual_seed(0)
        sim = DigitalTwinSimulator(im_size=(32, 32), **cfg)
        cases[f"simulator|{label}"] = _record(_simulate, sim, x.clone())
    return cases


def _tio_cases() -> dict[str, dict[str, object]]:
    import torchio as tio

    from spectramr.data.transforms.tio_physics import DigitalTwinDegradation
    from spectramr.infrastructure.physics.digital_twin_simulator import DigitalTwinSimulator

    def run() -> torch.Tensor:
        torch.manual_seed(0)
        sim = DigitalTwinSimulator(
            im_size=(32, 32), enable_motion=False, enable_b0=False, enable_b1=True
        )
        raw = _phantom((2, 32, 32, 2), False)
        subject = tio.Subject(kspace_raw=tio.ScalarImage(tensor=raw))
        return DigitalTwinDegradation(sim, pe_axis="W")(subject)["input"].data

    return {"tio_digital_twin_degradation": _record(run)}


def _torchkbnufft_version() -> str | None:
    """The NUFFT library behind the offres_blur_noncartesian hashes; ``None`` when absent."""
    try:
        import torchkbnufft
    except ImportError:
        return None
    return str(torchkbnufft.__version__)


def fingerprint() -> dict[str, object]:
    """Environment fields that decide FFT/BLAS/NUFFT bit patterns."""
    return {
        "torch_version": torch.__version__.split("+")[0],
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "mkl": torch.backends.mkl.is_available(),
        "mkldnn": torch.backends.mkldnn.is_available(),
        "torchkbnufft_version": _torchkbnufft_version(),
    }


def compute_cases() -> dict[str, dict[str, object]]:
    previous = torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        return {**_registry_cases(), **_simulator_cases(), **_tio_cases()}
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(previous)


def fingerprint_mismatches(fixture: dict) -> list[str]:
    """Name each fingerprint field that differs between the fixture and this machine."""
    now, stamped = fingerprint(), fixture.get("fingerprint", {})
    return [f"{k}: fixture={stamped.get(k)!r} now={v!r}" for k, v in now.items() if stamped.get(k) != v]


def compare(current: dict, fixture: dict) -> dict[str, list[str]]:
    """Return ADDED / REMOVED / CHANGED case keys; raise on a fingerprint mismatch."""
    bad = fingerprint_mismatches(fixture)
    if bad:
        raise RuntimeError(
            "golden fixture was generated in a different environment ("
            + "; ".join(bad)
            + "). Hashes are only comparable on the reference machine; regenerate with "
            "--regenerate only on review, on that machine."
        )
    base = fixture["cases"]
    return {
        "ADDED": sorted(set(current) - set(base)),
        "REMOVED": sorted(set(base) - set(current)),
        "CHANGED": sorted(k for k in set(current) & set(base) if current[k] != base[k]),
    }


def add_cases(current: dict, fixture: dict) -> dict:
    """``fixture`` plus the cases only ``current`` has; raise if any existing case moved."""
    diff = compare(current, fixture)
    moved = {label: diff[label] for label in ("CHANGED", "REMOVED") if diff[label]}
    if moved:
        raise RuntimeError(
            "--add only appends new cases, but existing ones moved: "
            + "; ".join(f"{label}: {', '.join(keys)}" for label, keys in moved.items())
        )
    return {**fixture, "cases": {**fixture["cases"], **{k: current[k] for k in diff["ADDED"]}}}


def _write(payload: dict) -> None:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--regenerate", action="store_true")
    group.add_argument("--add", action="store_true", help="append new cases only")
    args = parser.parse_args(argv)

    from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY

    try:
        validate_required_inputs(DEGRADATION_REGISTRY)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    errors = exemption_errors()
    if errors:
        print("declared exemptions disagree with the registry: " + "; ".join(errors), file=sys.stderr)
        return 1
    current = compute_cases()
    if args.regenerate:
        _write({"fingerprint": fingerprint(), "cases": current})
        print(f"fixture written: {len(current)} cases")
        return 0
    if not FIXTURE.exists():
        print(f"no fixture at {FIXTURE}; run with --regenerate", file=sys.stderr)
        return 1
    if args.add:
        fixture = json.loads(FIXTURE.read_text())
        merged = add_cases(current, fixture)
        _write(merged)
        print(f"fixture extended: {len(merged['cases']) - len(fixture['cases'])} cases added")
        return 0

    diff = compare(current, json.loads(FIXTURE.read_text()))
    for label, keys in diff.items():
        for key in keys:
            print(f"{label}: {key}")
    if any(diff.values()):
        print(
            "\n2-D degradation output differs from the golden fixture. New axes only: "
            "rerun with --add. An intended change: --regenerate, naming the operators.",
            file=sys.stderr,
        )
        return 1
    print(f"all {len(current)} cases match the golden fixture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
