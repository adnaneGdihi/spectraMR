"""What ``spectramr profile`` runs: the target allowlist, the modes, and the argv.

Kept separate from :mod:`spectramr.cli.profile_cli` so the vocabulary (which verbs
may be profiled, what a ``--mode`` means, what a ``--focus`` narrows to) and the
argv it produces are pure data and a pure function — assertable in a unit test
without a subprocess, and feedable to the real child parser to prove the child
accepts what this builds.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from spectramr.cli.profile_phases import PHASE_MARKERS
from spectramr.cli.val_batches import val_batches_override

#: Verbs this command can profile. An allowlist, because Scalene must be able to
#: see the work: it profiles the process it launches and *not* that process's
#: children. Deliberately EXCLUDED, each for a reason that would otherwise make
#: the option a lie (pitfall #15):
#:   * ``train-distributed`` / ``ablation`` / ``hpo`` / ``experiment`` — these
#:     fan out into torchrun workers or per-trial subprocesses. Scalene would
#:     faithfully report the launcher sitting in ``wait()`` and nothing else.
#:   * ``predict`` — takes ``--model``/``--input`` and no ``--config``; ``infer``
#:     is the config-driven inference verb.
#:   * ``audit`` — takes its config positionally and does no sustained compute.
#: Regression-tested against the real parser, the way ``launch.PIPELINE_VERBS``
#: is, so a verb whose flags change cannot silently stay on this list.
PROFILE_TARGETS = ("train", "sanity_check", "infer")

#: ``--mode`` -> the Scalene flags it becomes.
#:
#: These are NOT additive enablers, which the ``--help-advanced`` text does not
#: make clear. Verified against ``ScaleneParseArgs`` in scalene 2.3.0 and pinned
#: by ``test_profile_command.py::test_mode_flags_match_scalene_semantics``:
#:     (no flag)    -> cpu=True  gpu=True   memory=True
#:     --cpu-only   -> cpu=True  gpu=False  memory=False
#:     --gpu        -> cpu=True  gpu=True   memory=False   <- turns memory OFF
#:     --memory     -> cpu=True  gpu=False  memory=True    <- turns GPU OFF
#: So passing ``--gpu`` to "add GPU profiling" would silently drop memory
#: profiling. Each mode below names the set it actually produces.
PROFILE_MODES: dict[str, tuple[str, ...]] = {
    "full": (),
    "cpu-only": ("--cpu-only",),
    "cpu+gpu": ("--gpu",),
    "cpu+memory": ("--memory",),
}

#: Training iterations a profile runs by default: past warm-up and loader
#: start-up, short enough for one validation pass to land inside the run.
DEFAULT_PROFILE_ITERATIONS = 100

#: Validation batches per pass a profile runs by default. ``--iterations`` shortens training
#: but not validation, so an arm declaring a short eval interval walked its whole split every
#: few steps (the cs_mno arms: five full passes with LPIPS inside 40 min, each one a timeout).
DEFAULT_PROFILE_VAL_BATCHES = 4

#: Which module defines each phase's driver — read off the marker table instead
#: of restated here. The two declarations DID diverge while both read plausibly:
#: this module asserted ``_run_validation`` sat in ``pipelines/training_loop.py``,
#: so ``--focus val-loop`` narrowed the report to files that do not contain the
#: validation loop at all. One owner, and it is :data:`PHASE_MARKERS`, which is
#: already pinned to the source by ``test_profile_phases.py`` (non-negotiable 17).
_PHASE_FILE: dict[str, str] = {m.phase: m.file_suffix for m in PHASE_MARKERS}

#: ``--focus`` -> ``--profile-only`` substrings. A lookup table rather than an
#: ``if/elif`` chain (non-negotiable 20); ``all`` maps to the empty tuple, which
#: omits ``--profile-only`` entirely and leaves ``--program-path`` as the scope.
#:
#: These match on FILENAME, so they narrow which *frames* are reported and can
#: never separate the training phase from the validation phase — the two share
#: their callees. The ``phases/`` split (:mod:`spectramr.cli.profile_phases`) is the
#: mechanism for that. A loop preset is correspondingly coarse: ``val-loop``
#: admits the whole of ``pipelines/train.py``, not only ``_run_validation``.
#: ``infrastructure/validation/`` is absent by design — that package is *config*
#: validation, not the validation loop.
FOCUS_PRESETS: dict[str, tuple[str, ...]] = {
    "all": (),
    "train-loop": (_PHASE_FILE["train"], "infrastructure/training/"),
    "val-loop": (_PHASE_FILE["validation"], "core/metrics/"),
    "data": ("data/",),
    "model": ("models/",),
    "losses": ("models/losses/",),
}

#: How each target is told to write into this profiling run's OWN directory,
#: ``<arm run dir>/profile/<run_id>/run/``.
#:
#: The formatting key is ``child_run_dir`` and not ``run_dir`` for a reason worth
#: stating: these templates used to be handed the arm's real results directory,
#: so profiling ``experiment_11_attention_none`` for 300 iterations wrote
#: ``checkpoints/`` (``best.<format>`` among them), metrics and TensorBoard events over
#: the arm's actual run. The child reported it as healthy, because the injected
#: path satisfied ``check_output_dir_convention`` either way.
#:
#: ``validation.enabled`` is deliberately absent from every redirect: it switches
#: validation off rather than capping it, which is not what ``--val-batches`` asks
#: for, so the cap goes to ``validation.loader.num_batches`` (``pipelines/train.py``).
#: ``train``/``sanity_check`` accept ``--override`` and are redirected through
#: the config SSOT; ``infer`` does not, and takes ``--output`` instead. A table,
#: so adding a target forces its redirect to be stated rather than forgotten —
#: an unlisted target would otherwise scatter its artifacts into the CWD.
OUTPUT_INJECTION: dict[str, tuple[str, str]] = {
    "train": ("--override", "training.output_dir={child_run_dir}"),
    "sanity_check": ("--override", "training.output_dir={child_run_dir}"),
    "infer": ("--output", "{child_run_dir}/inference"),
}


#: Overrides every ``--override`` target's child runs with. Scalene re-arms itself
#: in forked DataLoader workers without reporting them, and a worker freeing the
#: parent's CUDA memory aborts the run (sabine, 2026-10-21, every mode). Loading in
#: the main process both avoids that and puts the data path in the profile. The
#: held-out test pass is not the loop being profiled and aborted the same way.
PROFILE_CHILD_OVERRIDES: tuple[str, ...] = (
    "data.loader.num_workers=0",
    "validation.loader.num_workers=0",
    "data.source.test_index_path=null",
)


def require_scalene() -> str | None:
    """Raise unless Scalene is importable; return its version.

    A profiler that silently degraded to ``cProfile`` would report plausible
    CPU numbers with no memory or GPU columns, which is worse than not running
    at all (non-negotiable 3). Absent is a state to report, never one to infer.
    """
    if importlib.util.find_spec("scalene") is None:
        raise RuntimeError(
            "scalene is not installed in this environment. Install the extra:\n"
            "    pip install -e '.[profile]'\n"
            "spectramr profile does not fall back to another profiler."
        )
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("scalene")
    except PackageNotFoundError:  # pragma: no cover - importable, no metadata
        return None


def entry_script() -> Path:
    """The script Scalene launches: ``src/spectramr/cli/__main__.py``.

    Scalene 2.3.0's ``run`` takes a script path and has no ``-m module`` form,
    so the package's ``__main__`` is addressed by path. It delegates to
    ``spectramr.cli.app.main``, i.e. exactly what ``python -m spectramr.cli`` does.
    """
    return Path(__file__).with_name("__main__.py")


def program_path() -> Path:
    """The directory Scalene should consider "the program": ``src/spectramr``.

    **This must always be passed explicitly.** Scalene defaults
    ``--program-path`` to the directory of the profiled script, which here would
    be ``src/spectramr/cli/`` — putting ``pipelines/`` and ``infrastructure/`` out
    of scope and yielding a green run with an empty profile.
    """
    return Path(__file__).parents[1]


def build_child_argv(args: argparse.Namespace, *, child_run_dir: Path) -> list[str]:
    """The verb argv both backends profile: ``<verb> --config ... --override ...``.

    Pure: no filesystem or process side effects.
    """
    flag, template = OUTPUT_INJECTION[args.target]
    child: list[str] = [
        args.target,
        "--config",
        str(args.config),
        flag,
        template.format(child_run_dir=child_run_dir),
    ]
    if args.device:
        child += ["--device", args.device]
    if flag == "--override":
        # The log sink is its own key; left alone it names the arm's experiment, not this run.
        child += ["--override", f"logging.sinks.dir={child_run_dir / 'logs'}"]
        child += [tok for ov in PROFILE_CHILD_OVERRIDES for tok in ("--override", ov)]
    # `--override` is `action="append"` on the child parser, so this composes
    # with the output redirect above rather than replacing it. Guarded upstream:
    # `profile_cli` raises for a target that runs no validation loop.
    if args.val_batches is not None:
        child += ["--override", val_batches_override(args.val_batches)]
    if args.iterations is not None:
        # The dispatcher's smoke cap is the one owner: it also shrinks the
        # validation interval, so the short run still reaches a validation pass.
        from spectramr.cli.manifest_dispatch import compute_iter_cap_overrides

        child += compute_iter_cap_overrides(args.config, args.iterations)[0]
    child += list(args.extra or [])
    return child


def build_scalene_command(
    args: argparse.Namespace,
    *,
    child_run_dir: Path,
    outfile: Path,
) -> list[str]:
    """Assemble the full ``scalene run ... --- <verb> ...`` argv.

    Pure: no filesystem or process side effects.
    """
    scalene_opts: list[str] = [
        "--outfile",
        str(outfile),
        "--program-path",
        str(program_path()),
        *PROFILE_MODES[args.mode],
    ]
    focus_filters = FOCUS_PRESETS[args.focus]
    if focus_filters:
        scalene_opts += ["--profile-only", ",".join(focus_filters)]

    return [
        sys.executable,
        "-m",
        "scalene",
        "run",
        *scalene_opts,
        str(entry_script()),
        "---",
        *build_child_argv(args, child_run_dir=child_run_dir),
    ]


__all__ = [
    "DEFAULT_PROFILE_ITERATIONS",
    "DEFAULT_PROFILE_VAL_BATCHES",
    "FOCUS_PRESETS",
    "OUTPUT_INJECTION",
    "PROFILE_CHILD_OVERRIDES",
    "PROFILE_MODES",
    "PROFILE_TARGETS",
    "build_child_argv",
    "build_scalene_command",
    "entry_script",
    "program_path",
    "require_scalene",
]
