"""``spectramr campaign infer``: grade a finished campaign's arms on their held-out test split."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def attach(campaign_sub) -> None:
    """Register ``campaign infer`` on the ``campaign`` sub-parsers."""
    ci = campaign_sub.add_parser(
        "infer",
        help="Re-run each arm's held-out test pass from its best checkpoint into <arm>/inference",
    )
    ci.add_argument("campaign_dir", type=Path, help="Campaign output directory")
    ci.add_argument("--base-dir", type=Path, default=None, help="Project root (default: .)")
    ci.add_argument("--only", action="append", default=[], help="Arm name or dir stem; repeatable")
    ci.add_argument("--force", action="store_true", help="Redo an arm that already has inference/")
    ci.add_argument("--concurrency", type=int, default=None, help="Array %%N throttle")
    ci.add_argument("--slurm", action="append", default=[], metavar="KEY=VALUE")
    ci.add_argument("--time", default=None, help="Sugar for --slurm time=...")
    ci.add_argument("--dry-run", action="store_true", help="Plan and print; submit nothing")
    ci.add_argument(
        "--check",
        action="store_true",
        help="Compare each arm's inference/ with its training-time test pass; submit nothing",
    )
    ci.add_argument("--rtol", type=float, default=1e-3, help="--check relative tolerance")
    ci.add_argument("--atol", type=float, default=1e-6, help="--check absolute tolerance")
    ci.set_defaults(func=campaign_infer)


def campaign_infer(args: argparse.Namespace) -> int:
    from spectramr.infrastructure.orchestration.campaign_inference import (
        plan_inference,
        submit_inference,
    )

    only = [s for raw in args.only for s in raw.split(",") if s]
    base_dir = args.base_dir or Path(".")
    try:
        if args.check:
            return _check(args, only)
        plans = plan_inference(args.campaign_dir, base_dir, only=only, force=args.force)
        for p in plans:
            print(
                f"{p.name}: "
                + (
                    f"SKIP {p.skip_reason}"
                    if p.skip_reason
                    else f"{p.rule} {p.checkpoint} | test split {p.test_manifest}"
                )
            )
        overrides = _slurm(args)
        failed = [p for p in plans if p.skip_reason and not p.already_done]
        # A request that cannot cover every arm is refused whole; --only is how a subset is asked for.
        if failed and not only:
            print(
                f"refusing to submit a partial array: {len(failed)} arm(s) cannot be graded (above)"
            )
            return 1
        if not any(p.skip_reason is None for p in plans):
            print("nothing to grade (every arm is already done or cannot be graded)")
            return 1 if failed else 0
        job = submit_inference(
            plans,
            args.campaign_dir,
            base_dir,
            dry_run=args.dry_run,
            concurrency=args.concurrency,
            slurm_overrides=overrides,
        )
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        logger.error("campaign infer failed: %s", exc)
        return 1
    print("dry run: nothing submitted" if args.dry_run else f"submitted array job {job}")
    return 1 if failed else 0


def _slurm(args: argparse.Namespace) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in args.slurm:
        if "=" not in raw:
            raise ValueError(f"invalid --slurm {raw!r}: expected key=value")
        key, value = raw.split("=", 1)
        out[key.strip()] = value.strip()
    if args.time is not None:
        if out.get("time", args.time) != args.time:
            raise ValueError("conflicting --time and --slurm time=...")
        out["time"] = args.time
    return out


def _check(args: argparse.Namespace, only: list[str]) -> int:
    from spectramr.core.run_artifacts import INFERENCE_SUBDIR, write_parity
    from spectramr.infrastructure.orchestration.campaign_inference import plan_inference
    from spectramr.infrastructure.orchestration.inference_parity import check_arm

    bad = 0
    for plan in plan_inference(
        args.campaign_dir, args.base_dir or Path("."), only=only, force=True
    ):
        res = check_arm(plan.run_dir, rtol=args.rtol, atol=args.atol)
        if (Path(plan.run_dir) / INFERENCE_SUBDIR).is_dir():
            write_parity(plan.run_dir, ok=res.ok, reason=res.reason, rtol=args.rtol, atol=args.atol)
        worst = max((m[2] for m in res.metrics.values()), default=float("nan"))
        print(
            f"{plan.name}: {'OK' if res.ok else 'MISMATCH'}  worst rel diff {worst:.3g}"
            + ("" if res.ok else f"  ({res.reason})")
        )
        bad += not res.ok
    return 1 if bad else 0
