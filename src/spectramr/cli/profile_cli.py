"""``spectramr profile`` — run a pipeline verb under Scalene.

Scalene is a *whole-process* sampling profiler: it is launched as
``scalene run <script.py> --- <args>`` and its ``scalene_profiler.start()`` /
``stop()`` API is inert unless the process is already running under Scalene. So
this verb is a **subprocess wrapper**, not loop instrumentation — which is also
the only shape non-negotiable 9 permits, since a hook inside the training loop
would be exactly the per-step overhead that rule forbids.

Two mechanisms carry the "profile the training loop / the validation loop"
intent without touching the loop: ``--target`` chooses which verb actually
*runs*, and ``--focus`` narrows what Scalene *reports*. Both vocabularies, and
the argv they produce, live in :mod:`spectramr.cli.profile_command`; the artifact
layout lives in :mod:`spectramr.cli.profile_paths`.

Everything a run produces — the profile, the manifest, the child's log, and the
profiled run's own checkpoints/logs/metrics — is written under
``experiments/results/<experiment>/profiles/<run_id>/``, never *at* the arm's
own results directory, which a profiling run must not overwrite (see
:class:`~spectramr.cli.profile_paths.ProfilePaths`). Checks that refuse a run
before Scalene starts live in :mod:`spectramr.cli.profile_preflight`.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from spectramr.cli.profile_command import (
    FOCUS_PRESETS,
    PROFILE_MODES,
    PROFILE_TARGETS,
    build_scalene_command,
    require_scalene,
)
from spectramr.cli.profile_diagnostics import explain_failure
from spectramr.cli.profile_paths import (
    ProfilePaths,
    build_profile_manifest,
    resolve_experiment_name,
    resolve_profile_paths,
)
from spectramr.cli.profile_preflight import run_preflight

from spectramr.cli.val_batches import VAL_BATCHES_OVERRIDE_KEY

logger = logging.getLogger(__name__)

#: Targets whose child is redirected with ``--override`` AND that actually run a
#: training-time validation loop, i.e. the ones ``--val-batches`` can act on.
#: ``infer`` is redirected with ``--output`` and runs no validation loop, so the
#: override would have nothing to act on; :func:`profile` raises rather than
#: dropping it, because a silently ignored knob reads as an applied one (#3).
_VAL_BATCH_TARGETS = frozenset({"train", "sanity_check"})


def _load_arm(config_path: Path) -> Any:
    """Load the arm through the real loader, once.

    Deliberately not a raw ``yaml.safe_load``: the schema's defaults and
    validators are what the *child* will see, so re-parsing the YAML here would
    make this a second resolver free to disagree with the run it is profiling
    (non-negotiable 17). Loaded once and threaded to all three consumers — the
    declared ``output_dir`` and both pre-flight checks — for the same reason.
    """
    from spectramr.config.settings import TrainingSettings

    return TrainingSettings.from_yaml(str(config_path))


def _write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _run_child(argv: list[str], log_path: Path) -> int:
    """Run the profiled child, teeing its output to the terminal and the log.

    Teed rather than captured: a profiled training run is long, and swallowing
    its progress output until exit would make a working run look hung.
    """
    with (
        log_path.open("w", encoding="utf-8") as log_fh,
        subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as proc,
    ):
        if proc.stdout is not None:
            for line in proc.stdout:
                sys.stdout.write(line)
                log_fh.write(line)
        return proc.wait()


def _split_phases(paths: ProfilePaths) -> dict[str, Any]:
    """Post-process the written profile into per-phase reports.

    Fail-open by design: the expensive artifact — the full profile — already
    exists and is valid, so a splitter problem degrades one derived view rather
    than failing a run that may have taken hours. The reason is recorded in the
    manifest instead of being inferred from a missing directory.
    """
    from spectramr.cli.profile_phases import write_phase_reports

    try:
        profile_json = json.loads(paths.outfile.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Phase split skipped — could not read the profile: %s", exc)
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    return write_phase_reports(paths.profile_dir, profile_json)


def profile(args: argparse.Namespace) -> int:
    """``spectramr profile`` entry point."""
    # Argument validation first: a bad flag combination is the operator's to fix
    # and is true regardless of whether scalene happens to be installed here.
    if args.val_batches is not None and args.target not in _VAL_BATCH_TARGETS:
        raise ValueError(
            f"--val-batches is meaningless for --target {args.target}: it caps "
            f"the training-time validation loop, which that target does not run. "
            f"Valid with: {', '.join(sorted(_VAL_BATCH_TARGETS))}."
        )

    scalene_version: str | None = None
    if args.dry_run:
        import importlib.util

        if importlib.util.find_spec("scalene") is None:
            logger.warning(
                "scalene is not installed — the printed command cannot be run "
                "here. Install it with: pip install -e '.[profile]'"
            )
    else:
        scalene_version = require_scalene()

    started_at = datetime.now().astimezone()
    settings = _load_arm(args.config)
    # Before the run directory is resolved: a refusal leaves nothing behind.
    run_preflight(args, settings)
    declared = getattr(settings.training, "output_dir", None)
    experiment = resolve_experiment_name(
        declared, explicit=args.experiment, config_path=args.config
    )

    from spectramr.infrastructure.logging.provenance import git_provenance, make_run_id

    run_id = make_run_id(
        f"{args.target}-{args.mode}-{args.focus}",
        started_at,
        git_provenance().get("sha_short"),
    )
    paths = resolve_profile_paths(experiment, run_id)
    argv = build_scalene_command(args, child_run_dir=paths.child_run_dir, outfile=paths.outfile)

    if args.dry_run:
        logger.info("Would profile into %s", paths.profile_dir)
        print(" ".join(argv))
        return 0

    paths.mkdirs()
    manifest = build_profile_manifest(
        paths=paths,
        run_id=run_id,
        started_at=started_at,
        target=args.target,
        mode=args.mode,
        focus=args.focus,
        mode_flags=PROFILE_MODES[args.mode],
        focus_filters=FOCUS_PRESETS[args.focus],
        config_path=args.config,
        declared_output_dir=declared,
        device=args.device,
        argv=argv,
        scalene_version=scalene_version,
    )
    # Written BEFORE the run: a profiled arm can take hours and be killed by the
    # scheduler, and a directory whose modes are unrecoverable is not evidence.
    _write_manifest(paths.manifest, manifest)

    logger.info("Profiling '%s' -> %s", args.target, paths.profile_dir)
    logger.info("Run outputs (throwaway) -> %s", paths.child_run_dir)
    exit_code = _run_child(argv, paths.log)

    manifest["outcome"] = {
        "exit_code": exit_code,
        "duration_s": round((datetime.now().astimezone() - started_at).total_seconds(), 3),
        "outfile_written": paths.outfile.exists(),
        "diagnosis": explain_failure(exit_code, paths.log) if exit_code != 0 else None,
    }
    _write_manifest(paths.manifest, manifest)

    if exit_code != 0:
        return exit_code
    if not paths.outfile.exists():
        # Green child, no profile: the failure mode --program-path exists to
        # prevent. Loud and non-zero, because the run's whole cost is already
        # paid and a silent success here reads as "nothing was slow".
        logger.error(
            "Run succeeded but Scalene wrote no profile to %s. "
            "Re-run with --focus all to widen the reporting scope.",
            paths.outfile,
        )
        return 1
    logger.info("Profile:  %s", paths.outfile)
    if args.no_phase_split:
        manifest["phase_split"] = {"status": "disabled", "reason": "--no-phase-split"}
    else:
        logger.info("CPU share by phase:")
        manifest["phase_split"] = _split_phases(paths)
    _write_manifest(paths.manifest, manifest)
    logger.info("Manifest: %s", paths.manifest)
    return 0


def attach_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``profile`` verb on the top-level CLI parser."""
    parser = subparsers.add_parser(
        "profile",
        help="Profile a pipeline verb with Scalene (CPU/GPU/memory); the "
        "profile and the run's own outputs land under experiments/results/.",
    )
    parser.add_argument(
        "--config", "-c", type=Path, required=True, help="Path to the arm's config YAML"
    )
    parser.add_argument(
        "--target",
        default="train",
        choices=PROFILE_TARGETS,
        help="Which pipeline verb to run under the profiler (default: train).",
    )
    parser.add_argument(
        "--mode",
        default="full",
        choices=tuple(PROFILE_MODES),
        help="What Scalene samples. 'full' = CPU+GPU+memory (default); "
        "'cpu-only' is the fastest. Note cpu+gpu excludes memory and vice "
        "versa — Scalene's flags narrow the set, they do not add to it.",
    )
    parser.add_argument(
        "--focus",
        default="all",
        choices=tuple(FOCUS_PRESETS),
        help="Narrow the REPORT to a set of FILES (Scalene --profile-only). "
        "This does NOT separate training from validation — they share their "
        "callees, so use the automatic phases/ split for that.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Override the <experiment> segment of experiments/results/<experiment>. "
        "Defaults to the one implied by the config's training.output_dir.",
    )
    parser.add_argument(
        "--device",
        "-d",
        default=None,
        help="Device for the PROFILED RUN (cuda/cpu/auto), passed through to the "
        "target verb. Unrelated to --mode, which is the profiler's own scope.",
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=None,
        metavar="N",
        help="Cap the validation loop at N batches for this run (--val-batches 8 "
        f"applies -O {VAL_BATCHES_OVERRIDE_KEY}=8), so a slow validation pass "
        "does not dominate a profile aimed at training. Note validation.enabled "
        "is NOT read by the framework (issue #673), so it cannot be used to skip "
        "validation outright.",
    )
    parser.add_argument(
        "--no-phase-split",
        action="store_true",
        help="Skip writing the per-phase train/validation reports under phases/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the scalene command that would run, without executing it.",
    )
    parser.add_argument(
        "extra",
        nargs="*",
        help="Passthrough args for the target verb after a '--' separator "
        "(e.g. ... -- --checkpoint best.pt --input data/test/ for infer).",
    )
    parser.set_defaults(func=profile)


__all__ = ["attach_subparsers", "profile"]
