#!/usr/bin/env python
"""Report what a checkpoint would restore, and what no checkpoint carries.

Run this against a real run directory before trusting a 120 h chain to it::

    python scripts/verification/inspect_checkpoint.py \\
        --checkpoint experiments/results/<arm>/checkpoints/checkpoint_epoch_0003_step_040000.pt

Exit code is 0 when every group a resume needs is present, 1 when one is
missing. The point is that "the resume works" stops being a claim about the
code and becomes a reading off the artifact the cluster actually wrote — the
two have disagreed before, because the periodic writer
(``CheckpointDirector``) and the fallback writer (``CheckpointService``) put
different key sets on the wire.

Reads only; needs no GPU and no config. Writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

#: Groups a resume reads, and what each one costs when it is absent.
#:
#: ``required`` is what makes the exit code non-zero: a checkpoint without
#: ``generator`` or ``optimizer_g`` cannot continue a run at all, while a
#: missing discriminator is simply an arm with no critic.
EXPECTED: tuple[tuple[str, bool, str], ...] = (
    ("generator", True, "the model weights"),
    ("optimizer_g", True, "Adam moments; absent means a zeroed optimizer"),
    ("global_step", True, "the iteration the curriculum difficulty is read from"),
    ("discriminator", False, "critic weights (arms without a critic: absent)"),
    ("optimizer_d", False, "critic moments"),
    ("scheduler_g", False, "LR schedule position; absent restarts the ramp"),
    ("scheduler_d", False, "critic LR schedule position"),
    ("scaler_state", False, "AMP dynamic scale (fp16 arms only)"),
    ("ema_state", False, "EMA shadow weights and their warmup counter"),
    ("counter_state", False, "step/epoch counters"),
    ("strategy_state", False, "strategy-owned heads, DC lambdas, learnable params"),
    ("rng_state", False, "the stochastic stream: timesteps, masks, dropout"),
    ("metrics", False, "the loss history carried for reporting"),
)

#: Stated on every report because it is the half a key list cannot show.
NOT_CARRIED: tuple[tuple[str, str], ...] = (
    (
        "dataloader position",
        "the epoch is recomputed from global_step, but the sampler restarts at "
        "that epoch's beginning, so a mid-epoch resume replays samples",
    ),
    (
        "the config",
        "no fingerprint is stored and nothing compares the resumed config to "
        "the saved one, so an edited YAML resumes silently",
    ),
)


def _summarize(key: str, value: Any) -> str:
    """One line describing what was actually stored under ``key``."""
    if key == "rng_state" and isinstance(value, dict):
        streams = [k for k in ("torch", "numpy", "python") if k in value]
        cuda = value.get("cuda")
        if cuda is not None:
            streams.append(f"cuda[{len(cuda)}]")
        return ", ".join(streams) or "empty"
    if key == "strategy_state" and isinstance(value, dict):
        modules = sorted(value.get("modules", {}))
        params = sorted(value.get("params", {}))
        if not modules and not params:
            return "no owned state"
        return f"modules={modules or '[]'} params={params or '[]'}"
    if key.startswith("optimizer") and isinstance(value, dict):
        groups = value.get("param_groups") or []
        state = value.get("state") or {}
        lr = groups[0].get("lr") if groups else None
        return f"{len(groups)} group(s), {len(state)} tensor state(s), lr={lr}"
    if key.startswith("scheduler") and isinstance(value, dict):
        return f"last_epoch={value.get('last_epoch')} _step_count={value.get('_step_count')}"
    if key in {"generator", "discriminator", "ema_state"} and isinstance(value, dict):
        tensors = [v for v in value.values() if torch.is_tensor(v)]
        params = sum(int(t.numel()) for t in tensors)
        return f"{len(tensors)} tensors, {params / 1e6:.1f}M parameters"
    if isinstance(value, dict):
        return f"{len(value)} entries"
    return str(value)


def _load(path: Path) -> dict[str, Any]:
    """Read the envelope.

    ``weights_only=False`` matches ``CheckpointDirector.load_from``: the
    envelope carries the numpy RNG tuple and plain-Python metadata, which the
    weights-only loader refuses. This reads the same files the resume reads, so
    it inherits the same trust domain and nothing is gained by diverging.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise TypeError(
            f"{path} holds {type(blob).__name__}, not a checkpoint envelope. A "
            "consolidated DeepSpeed export is a bare state_dict and can only be "
            "read by the strategy that wrote it."
        )
    return blob


def report(path: Path) -> int:
    blob = _load(path)
    size_mb = path.stat().st_size / (1024 * 1024)

    print(f"\nCheckpoint : {path.name}  ({size_mb:.1f} MB)")
    print(f"Position   : epoch={blob.get('epoch')}  global_step={blob.get('global_step')}")

    print("\nRESTORED BY A RESUME")
    missing_required: list[str] = []
    for key, required, why in EXPECTED:
        if key in blob and blob[key] is not None:
            print(f"  {key:<20} PRESENT  {_summarize(key, blob[key])}")
        elif required:
            missing_required.append(key)
            print(f"  {key:<20} MISSING  {why}")
        else:
            print(f"  {key:<20} absent   {why}")

    unexpected = sorted(set(blob) - {k for k, _, _ in EXPECTED})
    if unexpected:
        print(f"\n  also on the wire: {', '.join(unexpected)}")

    print("\nNOT CARRIED BY ANY CHECKPOINT")
    for name, why in NOT_CARRIED:
        print(f"  {name:<20} {why}")

    if missing_required:
        print(f"\nFAIL: {', '.join(missing_required)} missing — this cannot resume.\n")
        return 1
    print("\nOK: every group a resume needs is on the wire.\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path, help="path to a .pt checkpoint")
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error(f"no such checkpoint: {args.checkpoint}")
    return report(args.checkpoint)


if __name__ == "__main__":
    sys.exit(main())
