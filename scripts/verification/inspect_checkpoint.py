#!/usr/bin/env python
"""Report what a checkpoint would restore, and what no checkpoint carries.

Run this against a real run directory before trusting a 120 h chain to it::

    python scripts/verification/inspect_checkpoint.py \\
        --checkpoint experiments/results/<arm>/checkpoints/step_00040000.safetensors

Exit code is 0 when every part a resume needs is present, 1 when one is missing, and 2
when the file is not a schema-1 RunState (convert the run first). The point is that
"the resume works" stops being a claim about the code and becomes a reading off the
artifact the cluster actually wrote.

Reads only; needs no GPU and no config. Writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch

from spectramr.infrastructure.checkpointing import codecs
from spectramr.infrastructure.checkpointing.payload import SchemaVersionError

#: RunState parts beyond ``meta`` and ``models``, and what each costs when absent.
OPTIONAL_PARTS: tuple[tuple[str, str], ...] = (
    ("sched", "LR schedule position; absent restarts the ramp"),
    ("scaler", "AMP dynamic scale (fp16 arms only)"),
    ("ema", "EMA shadow weights and their warmup counter"),
    ("strategy", "strategy-owned heads, DC lambdas, learnable params"),
    ("early_stopping", "patience counter and best value"),
    ("rng", "the stochastic stream: timesteps, masks, dropout"),
)

#: Stated on every report because it is the half a part list cannot show.
NOT_CARRIED: tuple[tuple[str, str], ...] = (
    (
        "dataloader position",
        "the epoch is recomputed from meta.step, but the sampler restarts at "
        "that epoch's beginning, so a mid-epoch resume replays samples",
    ),
    (
        "the config",
        "meta.config_fingerprint records it, but no resume compares it to the "
        "resumed config, so an edited YAML resumes silently",
    ),
)

_WIDTH = 22


def _tensors(value: Any) -> Iterator[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for inner in value.values():
            yield from _tensors(inner)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            yield from _tensors(inner)


def _size(value: Any) -> str:
    """Tensor count and bytes held under ``value``, at any depth."""
    tensors = list(_tensors(value))
    nbytes = sum(t.numel() * t.element_size() for t in tensors)
    return f"{len(tensors)} tensors, {nbytes} bytes"


def _line(name: str, status: str, detail: str) -> None:
    print(f"  {name:<{_WIDTH}} {status:<8} {detail}")


def report(path: Path) -> int:
    """Print the file's meta and parts; 0 when it can resume a run, 1 when it cannot.

    Raises:
        SchemaVersionError: The file is not a schema-1 RunState.
    """
    state = codecs.read(path)
    meta = state["meta"]
    size_mb = path.stat().st_size / (1024 * 1024)

    print(f"\nCheckpoint : {path.name}  ({size_mb:.1f} MB)")
    print(
        f"Meta       : role={meta['role']}  reason={meta['reason']}  epoch={meta['epoch']}  "
        f"step={meta['step']}  format={meta['format']}"
    )
    print(
        f"             topology={meta['topology']}  native_tag={meta['native_tag']}  "
        f"best_metric={meta['best_metric']}"
    )
    print(
        f"             config_fingerprint={meta['config_fingerprint']}  "
        f"spectramr={meta['spectramr_version']}  git_sha={meta['git_sha']}"
    )

    print("\nRESTORED BY A RESUME")
    missing: list[str] = []
    models = state["models"]
    if "generator" not in models:
        missing.append("models.generator")
        _line("models.generator", "MISSING", "the model weights")
    for key in sorted(models):
        _line(f"models.{key}", "PRESENT", _size(models[key]))

    optim = state.get("optim")
    if optim:
        _line("optim", "PRESENT", f"{sorted(optim)}; {_size(optim)}")
    elif optim is not None and meta["native_tag"] is not None:
        _line("optim", "PRESENT", f"held by the engine under native tag {meta['native_tag']}")
    else:
        missing.append("optim")
        _line("optim", "MISSING", "optimizer moments; absent means a zeroed optimizer")

    for part, why in OPTIONAL_PARTS:
        if state.get(part):
            _line(part, "PRESENT", _size(state[part]))
        else:
            _line(part, "absent", why)

    print("\nNOT CARRIED BY ANY CHECKPOINT")
    for name, why in NOT_CARRIED:
        print(f"  {name:<{_WIDTH}} {why}")

    if missing:
        print(f"\nFAIL: {', '.join(missing)} missing — this cannot resume.\n")
        return 1
    print("\nOK: every part a resume needs is in the file.\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", required=True, type=Path, help="path to a step or best checkpoint file"
    )
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error(f"no such checkpoint: {args.checkpoint}")
    try:
        return report(args.checkpoint)
    except SchemaVersionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
