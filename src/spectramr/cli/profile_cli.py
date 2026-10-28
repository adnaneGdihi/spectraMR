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
import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from spectramr.cli.profile_backends import (
    BACKEND_CHOICES,
    TORCH_MIN_ITERATIONS,
    TORCH_SUMMARY_FILE,
    BackendChoice,
    build_torch_command,
    cuda_visible,
    select_backend,
    stamp_backend,
    torch_window,
)
from spectramr.cli.profile_command import (
    DEFAULT_PROFILE_ITERATIONS,
    DEFAULT_PROFILE_VAL_BATCHES,
    FOCUS_PRESETS,
    PROFILE_CHILD_OVERRIDES,
    PROFILE_MODES,
    PROFILE_TARGETS,
    build_child_argv,
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
from spectramr.cli.profile_preflight import run_preflight, single_process_overrides
from spectramr.cli.val_batches import VAL_BATCHES_OVERRIDE_KEY

logger = logging.getLogger(__name__)

#: Targets whose child is redirected with ``--override`` AND that actually run a
#: training-time validation loop, i.e. the ones ``--val-batches`` can act on.
#: ``infer`` is redirected with ``--output`` and runs no validation loop, so the
#: override would have nothing to act on; :func:`profile` raises rather than
#: dropping it, because a silently ignored knob reads as an applied one (#3).
_VAL_BATCH_TARGETS = frozenset({"train", "sanity_check"})

#: The one crash signature (``profile_diagnostics.KNOWN_FAILURES``) that is
#: timing-dependent rather than deterministic, so the only one worth a rerun.
_INTERMITTENT_CRASH = "scalene_torch_profiler_segfault"
MAX_INTERMITTENT_RETRIES = 1

#: Wall-clock cap on the profiled child. A profile that has not finished by then is
#: not going to: under ``--gpu`` the Scalene child was measured at 12-25x the
#: unprofiled time, then host-OOM-killed or left holding an idle GPU for hours.
DEFAULT_TIMEOUT_MIN = 30
#: Seconds between the interrupt (lets Scalene write what it has) and the kill.
TIMEOUT_GRACE_S = 120
#: ``timeout(1)``'s exit code, so a capped run is distinguishable from a crash.
TIMEOUT_EXIT_CODE = 124


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


def _run_child(argv: list[str], log_path: Path, *, timeout_s: float | None = None) -> int:
    """Run the profiled child, teeing its output to the terminal and the log.

    Teed rather than captured: a profiled training run is long, and swallowing
    its progress output until exit would make a working run look hung. When
    ``timeout_s`` elapses the child's whole process group is interrupted (so
    Scalene can still write its profile), killed after :data:`TIMEOUT_GRACE_S`, and
    :data:`TIMEOUT_EXIT_CODE` is returned.
    """
    expired = threading.Event()
    with (
        log_path.open("w", encoding="utf-8") as log_fh,
        subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            # Its own group, so a grandchild holding the pipe is signalled with it.
            start_new_session=True,
        ) as proc,
    ):

        def _signal_group(sig: signal.Signals) -> None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, sig)

        def _terminate() -> None:
            _signal_group(signal.SIGINT)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=TIMEOUT_GRACE_S)
            _signal_group(signal.SIGKILL)

        def _stop() -> None:
            expired.set()
            _terminate()

        timer = threading.Timer(timeout_s, _stop) if timeout_s else None
        if timer is not None:
            timer.daemon = True
            timer.start()
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    log_fh.write(line)
            code = proc.wait()
        except BaseException:
            # A new session no longer receives the terminal's Ctrl-C; pass it on.
            _terminate()
            raise
        finally:
            if timer is not None:
                timer.cancel()
        return TIMEOUT_EXIT_CODE if expired.is_set() else code


def _diagnose(exit_code: int, log_path: Path, timeout_min: float) -> dict[str, Any] | None:
    """The failure record for ``outcome.diagnosis``; a capped run names the cap."""
    if exit_code == 0:
        return None
    if exit_code == TIMEOUT_EXIT_CODE:
        logger.error("Profiled child exceeded --timeout-min=%s and was stopped.", timeout_min)
        return {
            "status": "recognized",
            "exit_code": exit_code,
            "signature": "profile_timeout",
            "summary": f"the profiled child did not finish within {timeout_min} min and was stopped.",
            "remedy": (
                "Profile with --mode cpu-only (Scalene's --gpu mode was measured at 12-25x the "
                "unprofiled time), lower --iterations or --val-batches (validation is not capped "
                "by --iterations), or raise --timeout-min deliberately."
            ),
        }
    return explain_failure(exit_code, log_path)


def _build_argv(args: argparse.Namespace, choice: BackendChoice, paths: ProfilePaths) -> list[str]:
    """The command the resolved backend runs; the verb argv inside it is the same for both."""
    if choice.applied == "torch":
        child = build_child_argv(args, child_run_dir=paths.child_run_dir)
        return build_torch_command(child, out_dir=paths.profile_dir, iterations=args.iterations)
    return build_scalene_command(args, child_run_dir=paths.child_run_dir, outfile=paths.outfile)


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


def _write_hotspots(paths: ProfilePaths) -> dict[str, Any]:
    """Derive ``hotspots.json`` from the written profile; fail-open like the split."""
    from spectramr.cli.profile_hotspots import write_hotspots

    try:
        return write_hotspots(paths.outfile, paths.profile_dir / "hotspots.json")
    except (OSError, ValueError) as exc:
        logger.warning("Hotspot report skipped: %s", exc)
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


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

    if args.iterations is not None and args.target not in _VAL_BATCH_TARGETS:
        raise ValueError(
            f"--iterations is meaningless for --target {args.target}: it caps "
            f"training.max_iterations, which that target does not run. "
            f"Valid with: {', '.join(sorted(_VAL_BATCH_TARGETS))}."
        )
    if args.iterations is None and args.target in _VAL_BATCH_TARGETS:
        args.iterations = DEFAULT_PROFILE_ITERATIONS
    if args.val_batches is None and args.target in _VAL_BATCH_TARGETS:
        args.val_batches = DEFAULT_PROFILE_VAL_BATCHES
    if args.iterations is not None and args.iterations < 1:
        raise ValueError(f"--iterations must be >= 1, got {args.iterations}.")

    timeout_min = getattr(args, "timeout_min", DEFAULT_TIMEOUT_MIN)
    if timeout_min < 0:
        raise ValueError(f"--timeout-min must be >= 0 (0 disables the cap), got {timeout_min}.")
    timeout_s = timeout_min * 60 or None

    import importlib.util

    scalene_installed = importlib.util.find_spec("scalene") is not None
    choice = select_backend(
        args.backend,
        mode_flags=PROFILE_MODES[args.mode],
        cuda_available=cuda_visible(),
        scalene_available=scalene_installed,
    )
    logger.info("Profiler backend: %s (%s)", choice.applied, choice.reason)
    scalene_version: str | None = None
    if choice.applied == "scalene":
        if args.dry_run:
            if not scalene_installed:
                logger.warning(
                    "scalene is not installed — the printed command cannot be run "
                    "here. Install it with: pip install -e '.[profile]'"
                )
        else:
            scalene_version = require_scalene()

    started_at = datetime.now().astimezone()
    settings = _load_arm(args.config)
    parallel_off = (
        single_process_overrides(settings) if getattr(args, "single_process", False) else []
    )
    if parallel_off:
        logger.warning(
            "--single-process: profiling with parallel.strategy=none instead of the "
            "arm's %r; the optimizer path measured is a plain single rank.",
            settings.parallel.strategy,
        )
        args.extra = [*parallel_off, *(args.extra or [])]
    # Before the run directory is resolved: a refusal leaves nothing behind.
    run_preflight(args, settings)
    declared = getattr(settings.training, "output_dir", None)
    experiment = resolve_experiment_name(
        declared, explicit=args.experiment, config_path=args.config
    )

    from spectramr.infrastructure.logging.provenance import git_provenance, make_run_id

    # The arm is part of the id: arms of one cohort share <experiment>, and array
    # tasks starting in the same second otherwise wrote into one directory.
    run_id = make_run_id(
        f"{Path(args.config).stem}-{args.target}-{args.mode}-{args.focus}",
        started_at,
        git_provenance().get("sha_short"),
    )
    paths = resolve_profile_paths(experiment, run_id)
    argv = _build_argv(args, choice, paths)

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
        iterations=args.iterations,
        argv=argv,
        scalene_version=scalene_version,
    )
    manifest["modes"]["timeout_min"] = timeout_min
    stamp_backend(
        manifest,
        choice,
        window=torch_window(args.iterations) if choice.applied == "torch" else None,
    )
    manifest["modes"]["val_batches"] = args.val_batches
    manifest["modes"]["single_process"] = parallel_off[1::2]
    manifest["modes"]["child_overrides"] = (
        list(PROFILE_CHILD_OVERRIDES) if args.target in _VAL_BATCH_TARGETS else []
    )
    # Written BEFORE the run: a profiled arm can take hours and be killed by the
    # scheduler, and a directory whose modes are unrecoverable is not evidence.
    _write_manifest(paths.manifest, manifest)

    logger.info("Profiling '%s' -> %s", args.target, paths.profile_dir)
    logger.info("Run outputs (throwaway) -> %s", paths.child_run_dir)
    deadline = time.monotonic() + timeout_s if timeout_s else None
    exit_code = _run_child(argv, paths.log, timeout_s=timeout_s)
    diagnosis = _diagnose(exit_code, paths.log, timeout_min)
    retries = 0
    # Scalene's torch profiler crashes intermittently under --gpu; a rerun of the
    # same command usually lands, which an unattended array cannot do by hand.
    while (
        diagnosis is not None
        and diagnosis.get("signature") == _INTERMITTENT_CRASH
        and retries < MAX_INTERMITTENT_RETRIES
    ):
        # --timeout-min caps the whole profile, retries included.
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            break
        retries += 1
        logger.warning("Known intermittent crash; rerunning (%d/%d).", retries, MAX_INTERMITTENT_RETRIES)
        exit_code = _run_child(argv, paths.log, timeout_s=remaining)
        diagnosis = _diagnose(exit_code, paths.log, timeout_min)

    artifact = paths.profile_dir / TORCH_SUMMARY_FILE if choice.applied == "torch" else paths.outfile
    manifest["outcome"] = {
        "exit_code": exit_code,
        "duration_s": round((datetime.now().astimezone() - started_at).total_seconds(), 3),
        "outfile_written": artifact.exists(),
        "retries": retries,
        "diagnosis": diagnosis,
    }
    _write_manifest(paths.manifest, manifest)

    if not artifact.exists():
        if exit_code != 0:
            return exit_code
        # Green child, no profile: the failure mode --program-path exists to
        # prevent. Loud and non-zero, because the run's whole cost is already
        # paid and a silent success here reads as "nothing was slow".
        logger.error(
            "Run succeeded but the %s backend wrote no profile to %s. "
            "Re-run with --focus all to widen the reporting scope.",
            choice.applied,
            artifact,
        )
        return 1
    # A child that died after Scalene wrote its profile (a late crash, a failed
    # final eval) still measured the loop; deriving the reports is what makes
    # that profile readable, so it must not depend on the exit code.
    logger.info("Profile:  %s", artifact)
    if choice.applied == "torch":
        # stamp_backend recorded phase_split and hotspots as unavailable, with the reason.
        _write_manifest(paths.manifest, manifest)
        logger.info("Manifest: %s", paths.manifest)
        return exit_code
    if args.no_phase_split:
        manifest["phase_split"] = {"status": "disabled", "reason": "--no-phase-split"}
    else:
        logger.info("CPU share by phase:")
        manifest["phase_split"] = _split_phases(paths)
    manifest["hotspots"] = _write_hotspots(paths)
    _write_manifest(paths.manifest, manifest)
    logger.info("Manifest: %s", paths.manifest)
    return exit_code


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
        "--timeout-min",
        type=float,
        default=DEFAULT_TIMEOUT_MIN,
        help=f"Stop the profiled child after this many minutes (default {DEFAULT_TIMEOUT_MIN}; "
        "0 = no cap). The child is interrupted so Scalene can write a partial profile.",
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
        "--backend",
        default="auto",
        choices=BACKEND_CHOICES,
        help="Which profiler runs. 'auto' (default) uses torch.profiler instead of Scalene when "
        "the mode would sample the GPU through Scalene (12-25x slower, crashes intermittently) "
        "or Scalene is not installed, and says why. 'torch' records GPU kernel time only: no "
        "phase split, no hotspots.json, needs CUDA and --iterations >= "
        f"{TORCH_MIN_ITERATIONS}. 'scalene' forces Scalene.",
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
        "does not dominate a profile aimed at training. Default "
        f"{DEFAULT_PROFILE_VAL_BATCHES} for train/sanity_check, because --iterations does "
        "not shorten validation. Note validation.enabled "
        "is NOT read by the framework (issue #673), so it cannot be used to skip "
        "validation outright.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        metavar="N",
        help="Cap the profiled run at N training iterations "
        f"(default {DEFAULT_PROFILE_ITERATIONS} for train/sanity_check). A cap, not "
        "a forced value: an arm declaring fewer is not inflated. The validation "
        "interval is shrunk with it so one validation pass still runs.",
    )
    parser.add_argument(
        "--single-process",
        action="store_true",
        help="Profile a ddp/fsdp/deepspeed arm as one plain rank (parallel.strategy=none "
        "and its sub-block disabled), recorded in the manifest. A no-op for arms that "
        "need no process group, so it is safe on a whole folder.",
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
        "(e.g. ... -- --checkpoint best.safetensors --input data/test/ for infer).",
    )
    parser.set_defaults(func=profile)


__all__ = ["attach_subparsers", "profile"]
