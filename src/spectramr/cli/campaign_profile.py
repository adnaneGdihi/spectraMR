"""``spectramr campaign profile``: profile every arm of a campaign into ``<campaign>/<arm>/profile``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from spectramr.cli.campaign_infer import _slurm

logger = logging.getLogger(__name__)


def attach(campaign_sub) -> None:
    """Register ``campaign profile`` on the ``campaign`` sub-parsers."""
    cp = campaign_sub.add_parser(
        "profile",
        help="Profile each arm (one SLURM array) into <campaign>/<arm>/profile/<run_id>",
    )
    cp.add_argument("config", type=Path, help="Path to campaign YAML")
    cp.add_argument("--base-dir", type=Path, default=None, help="Project root (default: .)")
    cp.add_argument(
        "--only", action="append", default=[], help="Arm name or run-dir name; repeatable"
    )
    cp.add_argument("--concurrency", type=int, default=None, help="Array %%N throttle")
    cp.add_argument("--slurm", action="append", default=[], metavar="KEY=VALUE")
    cp.add_argument("--time", default=None, help="Sugar for --slurm time=... (default 00:45:00)")
    cp.add_argument(
        "--dry-run", action="store_true", help="Plan and print the script; submit nothing"
    )
    cp.add_argument(
        "profile_args",
        nargs="*",
        help="`spectramr profile` flags after `--`, appended to the array preset "
        "(--mode cpu-only --val-batches 4 --single-process) so they win",
    )
    cp.set_defaults(func=campaign_profile)


def campaign_profile(args: argparse.Namespace) -> int:
    from spectramr.infrastructure.orchestration.campaign_profile import (
        DEFAULT_PROFILE_ARGS,
        plan_profile,
        submit_profile,
    )

    only = [s for raw in args.only for s in raw.split(",") if s]
    profile_args = [*DEFAULT_PROFILE_ARGS, *args.profile_args]
    try:
        cfg, campaign_dir, plans = plan_profile(args.config, args.base_dir or Path("."), only=only)
        for p in plans:
            print(
                f"{p.name}: "
                + (f"SKIP {p.skip_reason}" if p.skip_reason else f"{p.run_dir}/profile")
            )
        failed = [p for p in plans if p.skip_reason]
        # A request that cannot cover every arm is refused whole; --only is how a subset is asked for.
        if failed and not only:
            print(f"refusing to submit a partial array: {len(failed)} arm(s) cannot be profiled")
            return 1
        job, script = submit_profile(
            cfg,
            campaign_dir,
            plans,
            args.base_dir or Path("."),
            profile_args=profile_args,
            dry_run=args.dry_run,
            concurrency=args.concurrency,
            slurm_overrides=_slurm(args),
        )
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        logger.error("campaign profile failed: %s", exc)
        return 1
    print(f"profile flags: {' '.join(profile_args)}")
    if args.dry_run:
        print(script)
        print("dry run: nothing submitted")
    else:
        print(f"submitted array job {job}")
    return 1 if failed else 0
