"""Where ``spectramr profile`` writes, and what it records about how it ran.

Two responsibilities, both deliberately kept out of :mod:`spectramr.cli.profile_cli`
so they are testable without building an argparse namespace:

1. **Path resolution.** Every artifact a profiling run produces — the profiled
   run's own checkpoints/logs/metrics *and* the Scalene profile beside them —
   lands under ``<arm run dir>/profile/<run_id>/``. The arm run dir is
   ``experiments/results/<experiment>`` for a standalone run and the
   ``--output-dir`` that ``campaign profile`` passes (``<campaign>/<arm>``). That
   root is not a new convention invented here: ``config_health_checker`` already
   *requires* ``training.output_dir`` to be ``experiments/results/
   <experiment_name>``, so this module derives from that owner rather than
   becoming a second one (non-negotiable 17). ``resolve_experiment_name`` is the
   single place the ``<experiment>`` segment is decided.

   The ``profile/<run_id>/`` nesting is load-bearing rather than cosmetic: a
   profiling run is a THROWAWAY run of a real arm, and pointing it at the arm's
   own results directory made it overwrite the artifacts it was supposed to be
   measuring. See :class:`ProfilePaths` for the two roots that keeps apart.

2. **The manifest.** A profile that cannot be traced back to the modes that
   produced it is not evidence. ``cpu-only`` and ``full`` produce legitimately
   different numbers for the same arm, and ``--focus`` changes *which lines are
   reported at all* — so a bare ``scalene-profile.json`` sitting in a directory
   is ambiguous in exactly the way non-negotiable 14 exists to prevent. Every
   run therefore stamps ``profile_manifest.json`` recording the declared knobs
   beside what was actually executed (the literal child argv).

The provenance halves (git identity, interpreter/runtime environment) are
delegated to :mod:`spectramr.infrastructure.logging.provenance` — the framework's
existing owner for that — never re-derived here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from spectramr.infrastructure.run_layout import PROFILE_SUBDIR

#: The path convention ``config_health_checker`` enforces on
#: ``training.output_dir``: ``experiments/results/<experiment_name>``. A
#: statement about the repo's layout, NOT a knob — :func:`resolve_experiment_name`
#: matches against this to find the ``<experiment>`` segment.
CONVENTION_PARTS: tuple[str, str] = ("experiments", "results")

#: Root every profiling artifact is WRITTEN to. The same location, but a
#: distinct concept: tests redirect this into ``tmp_path``, and doing so must not
#: change how an experiment NAME is parsed out of a declared ``output_dir``.
#: Deriving name-matching from this constant is exactly the coupling that made
#: every redirected run fall through to the config-stem fallback.
RESULTS_ROOT = Path(*CONVENTION_PARTS)


@dataclass(frozen=True)
class ProfilePaths:
    """Resolved destinations for one profiling run.

    Two distinct roots, and the distinction is the point:

    ``run_dir`` is the arm's REAL results directory — ``experiments/results/
    <experiment>`` — and is used only to *locate* the profile beneath it. Nothing
    is written there.

    ``child_run_dir`` is what the profiled child is told to write into. It sits
    at ``<profile_dir>/run/``, so a profiling run's checkpoints, logs, metrics
    and TensorBoard events are filed with the profile that produced them.

    They were the same path until this was fixed, which meant a 300-iteration
    profiling run wrote ``checkpoints/`` (``best.<format>`` among them), metrics CSVs
    and provenance directly on top of the arm's real results. Nothing announced
    it: the child's own config-health run reported the injected path as
    on-convention (``check_output_dir_convention`` is a *prefix* test), so a
    profiling run and a real run were indistinguishable after the fact. Keeping
    the two as separate fields means a future redirect has to say which one it
    means rather than picking up whichever is in scope.
    """

    experiment: str
    run_dir: Path
    profile_dir: Path
    child_run_dir: Path
    outfile: Path
    manifest: Path
    log: Path

    def mkdirs(self) -> None:
        """Create the profile and child-run directories up front.

        Scalene writes ``--outfile`` itself and will not create intermediate
        directories; without this the child dies at exit, *after* the whole
        profiled run has been paid for.
        """
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.child_run_dir.mkdir(parents=True, exist_ok=True)


def resolve_experiment_name(
    output_dir: str | None,
    *,
    explicit: str | None,
    config_path: Path,
) -> str:
    """Decide ``<experiment>`` in ``experiments/results/<experiment>``: the arm's run dir.

    A relative path that may span several segments (``<cohort>/<arm>``), so the
    profile is filed inside the arm's own run directory rather than its cohort's.

    Precedence, most authoritative first:

    1. ``explicit`` — an operator-supplied ``--experiment``. Always wins, so a
       one-off profiling run can be filed separately from the arm's real results.
    2. ``output_dir`` — ``training.output_dir`` from the loaded config, when it
       already points under ``experiments/results/``: everything after that pair.
       This is the common case and the reason a profiling run files itself inside
       the arm's own run directory.
    3. The config file's stem.

    Case 3 is reached when an arm's ``training.output_dir`` is the schema default
    (``./training_output``) or otherwise off-convention. That is a *derivation*,
    not a silent substitution of a declared value: the config declared a location
    that is not addressable under the results root, so the name is taken from the
    only other identifier the arm has. The caller surfaces which case fired, and
    the manifest records both the declared ``training.output_dir`` and the
    ``run_dir`` actually used, so a divergence is visible rather than inferred.
    """
    if explicit:
        return explicit.strip("/")
    if output_dir:
        candidate = Path(output_dir)
        parts = candidate.parts
        # Match on the ('experiments', 'results') pair anywhere in the path so an
        # absolute output_dir (/scratch/.../experiments/results/foo) resolves the
        # same as the repo-relative spelling the health checker enforces. Anchored
        # on CONVENTION_PARTS, never on the injectable RESULTS_ROOT.
        anchor = CONVENTION_PARTS
        for i in range(len(parts) - len(anchor)):
            if parts[i : i + len(anchor)] == anchor and i + len(anchor) < len(parts):
                return str(Path(*parts[i + len(anchor) :]))
    return config_path.stem


def resolve_profile_paths(
    experiment: str,
    run_id: str,
    *,
    results_root: Path | None = None,
) -> ProfilePaths:
    """Build the artifact layout for one profiling run.

    ``results_root`` defaults to the repo-relative :data:`RESULTS_ROOT`, the path
    the config health checker enforces and ``discover_run_dirs`` walks. A
    campaign's ``--output-dir`` is split into the two, so the profile lands in the
    same ``<campaign>/<arm>`` directory its training run writes.
    """
    root = results_root if results_root is not None else RESULTS_ROOT
    run_dir = root / experiment
    profile_dir = run_dir / PROFILE_SUBDIR / run_id
    return ProfilePaths(
        experiment=experiment,
        run_dir=run_dir,
        profile_dir=profile_dir,
        # Under profile_dir, never at run_dir: the profiled child must not write
        # into the arm's real results. Still under `experiments/results/`, which
        # is what `check_output_dir_convention` requires of the injected value.
        child_run_dir=profile_dir / "run",
        outfile=profile_dir / "scalene-profile.json",
        manifest=profile_dir / "profile_manifest.json",
        log=profile_dir / "scalene.log",
    )


def build_profile_manifest(
    *,
    paths: ProfilePaths,
    run_id: str,
    started_at: datetime,
    target: str,
    mode: str,
    focus: str,
    mode_flags: tuple[str, ...],
    focus_filters: tuple[str, ...],
    config_path: Path,
    declared_output_dir: str | None,
    device: str | None,
    iterations: int | None,
    argv: list[str],
    scalene_version: str | None,
) -> dict[str, Any]:
    """Assemble the record written to ``profile_manifest.json``.

    Structured as *declared* beside *applied*, the same shape the debug-snapshot
    contract uses: ``modes`` says what was asked for in this run's own
    vocabulary, ``scalene`` says which flags that became, and ``command`` is the
    literal argv — so a reader can re-run the profile without reconstructing it
    from prose. ``declared_output_dir`` is recorded next to ``run_dir`` precisely
    so the case-3 derivation in :func:`resolve_experiment_name` is auditable.

    Fail-open on provenance, matching ``provenance.collect_run_provenance``: a
    missing ``git`` binary degrades one field, never the manifest.
    """
    from spectramr.infrastructure.logging.provenance import (
        git_provenance,
        runtime_environment,
    )

    return {
        "run_id": run_id,
        "started_at": started_at.isoformat(timespec="seconds"),
        "experiment": paths.experiment,
        "modes": {
            "target": target,
            "mode": mode,
            "focus": focus,
            "device": device,
            "iterations": iterations,
        },
        "scalene": {
            "version": scalene_version,
            "mode_flags": list(mode_flags),
            "profile_only": list(focus_filters),
        },
        "config": {
            "path": str(config_path),
            "declared_training_output_dir": declared_output_dir,
        },
        "paths": {
            "run_dir": str(paths.run_dir),
            "profile_dir": str(paths.profile_dir),
            # What the child was actually told to write into. Recorded beside
            # `run_dir` and `declared_training_output_dir` so all three are
            # readable at once: what the arm declared, where its real results
            # live, and where THIS run's outputs went instead.
            "child_run_dir": str(paths.child_run_dir),
            "outfile": str(paths.outfile),
            "log": str(paths.log),
        },
        "command": argv,
        "git": git_provenance(),
        "runtime": runtime_environment(),
    }


__all__ = [
    "CONVENTION_PARTS",
    "RESULTS_ROOT",
    "ProfilePaths",
    "build_profile_manifest",
    "resolve_experiment_name",
    "resolve_profile_paths",
]
