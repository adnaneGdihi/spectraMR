"""Per-array-task dispatch for the SLURM experiment array (WS-4).

This is the Python SSOT for the fragile body that used to live inline in
``scripts/training/dispatch_experiments.sbatch``: resolve the config for THIS
array index from a manifest, optionally cap the iteration budget (smoke runs),
run the Tier-0/1 audit pre-flight, then launch training with the verb the arm's
own ``parallel.strategy`` requires.

Two launch verbs, not one
-------------------------
``spectramr train`` is a single process and never calls ``setup_distributed``,
so an arm declaring ddp / fsdp / deepspeed builds its whole training
environment and then raises out of ``_require_process_group`` at Stage B —
correctly, because the framework refuses to quietly run single-process, but
only after the audit and the model build have been paid for. That took 36 of
the 45 tasks in array 8589967 (2026-09-12), every one of them a kspace_filling
arm declaring ``parallel.strategy: deepspeed``. Such arms are launched here
under ``torch.distributed.run`` with the ``train-distributed`` verb instead;
:func:`build_launch_argv` owns that choice and
:func:`requires_process_group` owns the predicate behind it.

Why it moved out of bash
------------------------
The ``TRAIN_ITERS`` cap was a ``grep | sed | … || true`` pipeline under
``set -euo pipefail``. It aborted whole arrays in production **twice**: a config
that omits ``eval_interval`` makes the ``grep`` exit 1, and ``set -e`` then kills
the task right after the header, before training (job 7145522 — all 56
mrixfields tasks). Reimplemented here, the cap is a plain YAML parse:
``yaml.safe_load`` strips inline comments natively (no ``sed``) and
``dict.get → None`` handles a missing key explicitly (no ``set -e`` foot-gun).

Torch-free by design
--------------------
This module imports only stdlib + PyYAML, and ``spectramr.cli`` package init is
torch-free, so ``python -m spectramr.cli.manifest_dispatch`` runs on a torch-less
interpreter. The dispatcher's ``--dry-run`` path (the unit tests) therefore
needs no GPU/torch; the real audit + train are shelled out to
``python -m spectramr.cli`` AFTER the venv is activated, so the heavy imports
happen in the right interpreter.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from spectramr.core.env_names import SPECTRAMR_WALL_CLOCK_MARKER

#: Default per-arm log/audit root, relative to the repo root (mirrors the old
#: ``DISPATCH_DIR`` default in the .sbatch).
_DEFAULT_DISPATCH_DIR = "experiments/results/dispatch"


def resolve_manifest_config(manifest_path: str | Path, index: int) -> str:
    """Return the config path on line ``index`` (0-based) of ``manifest_path``.

    Blank lines are ignored (a trailing newline is not a phantom arm). Raises
    :class:`IndexError` with an "out of range" message when ``index`` is outside
    ``[0, N)`` — the dispatcher turns that into a non-zero exit so SLURM marks
    the task failed rather than silently training nothing.
    """
    text = Path(manifest_path).read_text()
    configs = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n = len(configs)
    if index < 0 or index >= n:
        raise IndexError(
            f"SLURM_ARRAY_TASK_ID={index} out of range [0, {n}) for manifest {manifest_path}"
        )
    return configs[index]


def _read_capped_key(
    cfg: dict,
    section: str,
    key: str,
    *,
    block: str | None = None,
    legacy_key: str | None = None,
) -> int | None:
    """Read a declared int out of ``cfg[section]``, or ``None`` if absent.

    ``(cfg.get(section) or {})`` tolerates a section that is missing OR present
    but null (``validation:`` with no body) — the two shapes that broke the bash
    grep. A present-but-non-int value is a config error and is left to surface
    downstream rather than silently coerced.

    ``block`` is the sub-block the leaf lives in today and is read first;
    ``legacy_key`` is its pre-decomposition flat spelling, read only if the
    canonical location is absent. Both are needed because this is a raw YAML
    read *on purpose* — the cap must see the DECLARATION, before the loader
    supplies a default — so nothing folds the legacy spelling on our behalf.

    ``validation.eval_interval`` became ``validation.schedule.interval_steps``
    on 2026-07-31, after which a migrated arm answered ``None`` here. That is the
    safe direction for the cap itself (absent reads as "cap it"), but it left the
    "already fires within the capped run" branch unreachable and turned the
    documented cap-not-inflate contract into an INFLATE for any arm declaring a
    smaller interval than the cap.
    """
    body = cfg.get(section) or {}
    if block is not None:
        nested = (body.get(block) or {}).get(key)
        if nested is not None:
            return int(nested)
    value = body.get(key)
    if value is None and legacy_key is not None:
        value = body.get(legacy_key)
    return None if value is None else int(value)


def compute_iter_cap_overrides(
    config_path: str | Path, train_iters: int
) -> tuple[list[str], list[str]]:
    """Compute the ``--override`` args for a ``TRAIN_ITERS`` smoke cap.

    Caps BOTH ``training.max_iterations`` AND ``validation.eval_interval`` so a
    short run actually reaches a validation step — capping iterations alone is a
    trap (most arms set ``eval_interval`` to 2500, so a 100-iter run never
    validates; pitfall #10). Both are CAPS, not forced values: an arm declaring a
    smaller budget (a 1-shot calibration) is never inflated.

    Returns ``(override_args, messages)`` — ``override_args`` is a flat
    ``["--override", "k=v", …]`` list ready to splat into the train command;
    ``messages`` are human ``[TRAIN] …`` lines explaining each cap decision.
    """
    cfg = yaml.safe_load(Path(config_path).read_text()) or {}
    cfg_iters = _read_capped_key(cfg, "training", "max_iterations")
    cfg_eval = _read_capped_key(
        cfg,
        "validation",
        "interval_steps",
        block="schedule",
        legacy_key="eval_interval",
    )

    overrides: list[str] = []
    messages: list[str] = []

    # max_iterations: cap-not-inflate.
    if cfg_iters is None or cfg_iters > train_iters:
        eff_iters = train_iters
        overrides += ["--override", f"training.max_iterations={eff_iters}"]
    else:
        eff_iters = cfg_iters
        messages.append(
            f"[TRAIN] keeping config max_iterations={cfg_iters} (<= cap "
            f"{train_iters}); short/calibration arm not inflated"
        )

    # eval_interval: shrink to the effective iteration count so validation fires
    # at least once (on the final step), unless it already fires within the run.
    if cfg_eval is None or cfg_eval > eff_iters:
        overrides += ["--override", f"validation.eval_interval={eff_iters}"]
        shown = "unset" if cfg_eval is None else cfg_eval
        messages.append(
            f"[TRAIN] capping validation.eval_interval {shown}->{eff_iters} so "
            "validation runs within the capped run"
        )
    else:
        messages.append(
            f"[TRAIN] keeping config eval_interval={cfg_eval} (<= {eff_iters}); "
            "already fires within the capped run"
        )
    return overrides, messages


def read_parallel_strategy(config_path: str | Path) -> str:
    """Return the arm's DECLARED ``parallel.strategy``, or the schema default.

    A raw YAML read on purpose, for the same reason the iteration cap is one:
    the launch verb must be decided from what the file says, in a torch-free
    interpreter, before anything loads the config. An absent ``parallel`` block
    (or a present-but-null one, the shape that broke the old bash grep) reads as
    ``"none"`` — which is the schema default, so the two agree.
    """
    cfg = yaml.safe_load(Path(config_path).read_text()) or {}
    parallel = cfg.get("parallel") or {}
    strategy = parallel.get("strategy") if isinstance(parallel, dict) else None
    return "none" if strategy is None else str(strategy)


def requires_process_group(strategy: str) -> bool:
    """True when *strategy* cannot run without an initialised process group.

    Asked of the strategy plugins themselves, through the same registry query
    ``spectramr.cli.profile_preflight`` uses, so a backend added tomorrow
    answers for itself. Spelling the set here as ``{"ddp", "fsdp", "deepspeed"}``
    would be a second owner of a fact the registry already holds
    (non-negotiable 17), and wrong in a way nothing would report: ``dp`` is
    ``nn.DataParallel`` and genuinely single-process, so the tempting
    ``strategy != "none"`` shorthand would send it to torchrun for no reason.

    Raises :class:`ImportError` on a torch-less interpreter — the caller decides
    whether that is fatal (it is, for a real launch) or merely unreportable (the
    dry run).
    """
    from spectramr.cli.profile_preflight import process_group_strategies

    return strategy in process_group_strategies()


def resolve_rendezvous_port(env: dict[str, str] | None = None) -> int:
    """A rendezvous port no sibling array task on this node will also pick.

    Derived from ``SLURM_JOB_ID``, which is **unique per array task** — the id
    shared by the whole array is ``SLURM_ARRAY_JOB_ID``. That matters because
    two tasks do land on one node (31 and 35 of array 8589967 both ran on
    compute-2-5), and a port derived from the shared id would have them fight
    over one socket.

    Falls back to a port the OS picks when there is no scheduler, and also when
    the derived one is already taken — a derived port is collision-free against
    other *array tasks*, not against whatever else is on the node.
    """
    import socket

    job_id = (env if env is not None else os.environ).get("SLURM_JOB_ID", "")
    if job_id.isdigit():
        candidate = 20000 + (int(job_id) % 20000)
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", candidate))
            except OSError:
                pass
            else:
                return candidate
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def build_launch_argv(
    config: str,
    overrides: list[str],
    resume: bool,
    *,
    distributed: bool,
    nproc_per_node: int = 1,
    master_port: int = 29500,
    verb: str = "train",
    verb_args: list[str] | None = None,
) -> list[str]:
    """The complete argv for this arm's launch.

    ``verb`` is the pipeline the array runs on every arm — ``train`` unless the
    submitter named another one (``spectramr <verb>`` in the wrapper's command
    form). It must accept the config as a ``--config`` FLAG, which is what
    ``launch.PIPELINE_VERBS`` enumerates and :func:`validate_verb_plan` checks;
    the process-group upgrade below is ``train``-only, so pairing it with any
    other verb is a caller bug rather than a config the launch should guess at.

    ``verb_args`` are the submitter's own trailing flags and land LAST, after
    the overrides, so a flag the submitter spells wins the argparse last-one-wins
    contest against anything this function put there.

    Single-process arms keep ``spectramr.cli train``. A ``distributed`` arm gets
    ``torch.distributed.run`` + the ``train-distributed`` verb, which is the only
    spelling that calls ``setup_distributed`` and therefore the only one under
    which ddp / fsdp / deepspeed reach ``adopt`` with a process group to use.

    ``sys.executable -m torch.distributed.run``, never a bare ``torchrun`` on
    PATH: the venv is the torch SSOT, and a PATH ``torchrun`` can belong to a
    different interpreter — the same shadowing class of failure the .sbatch's
    FreeSurfer scrub exists for.

    The rendezvous is **static, on loopback**, not ``--standalone``. Each array
    task is ``--nodes=1``, so there is nothing to rendezvous across — and
    ``--standalone`` selects the c10d *dynamic* backend, whose store server
    resolves ``socket.gethostname()`` no matter what endpoint it is handed. On a
    host whose own name is not resolvable that hangs for the full 300 s store
    timeout and then dies with ``DistNetworkError: client socket has timed out
    … trying to connect to (<hostname>, <port>)``, which names neither the cause
    nor the fix. Observed here, and an unnecessary dependency in any case: a
    single-node launch needs loopback and nothing more. ``127.0.0.1`` is always
    resolvable. :func:`resolve_rendezvous_port` keeps two tasks on one node
    apart.

    ``--resume`` and the overrides are spelled identically for both verbs;
    ``train-distributed`` takes the same flags. The spelling is ``if-present``,
    not ``auto``: an array task that is requeued at the wall clock re-runs this
    exact command, so the first attempt has no checkpoint to find and ``auto``
    would fail it outright.
    """
    if distributed and verb != "train":
        raise ValueError(
            f"the process-group upgrade is train-only, got verb {verb!r} with "
            "distributed=True"
        )
    if not distributed:
        args = [sys.executable, "-m", "spectramr.cli", verb, "--config", config]
    else:
        args = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            f"--nproc_per_node={nproc_per_node}",
            "--master_addr=127.0.0.1",
            f"--master_port={master_port}",
            "-m",
            "spectramr.cli",
            "train-distributed",
            "--config",
            config,
        ]
    if resume:
        args += ["--resume", "if-present"]
    return args + overrides + list(verb_args or [])




def output_dir_override(output_base: str | Path, config: str | Path) -> list[str]:
    """Return the ``--override training.output_dir=<base>/<stem>`` args.

    Routes THIS task's checkpoints / CSVs into a campaign tree keyed by the
    config's file stem, so the campaign orchestrator's ``_discover_results``
    (which globs ``<output_dir>/checkpoints/*.pt``) finds them. The orchestrator
    records the identical ``<base>/<stem>`` as the arm's ``output_dir``; the two
    MUST agree, so the path is computed the same way on both sides.
    """
    stem = Path(config).stem
    return ["--override", f"training.output_dir={Path(output_base) / stem}"]



def yield_marker_path(dispatch_dir: str | Path, name: str, index: int) -> Path:
    """Where THIS task's run announces that it stopped at the wall.

    The launcher names it and exports it, and the run obeys — so there is no
    second derivation to disagree with (non-negotiable 17). Keyed by arm and
    array index because one dispatch directory serves the whole array.
    """
    return Path(dispatch_dir) / f"yield_{name}_{index}.json"


def requeue_this_task(marker: Path) -> bool:
    """Ask Slurm to re-run this array task so training continues.

    Returns ``True`` when Slurm accepted the requeue. A refusal is REPORTED with
    the manual chain command rather than swallowed: a run that yielded and was
    not requeued has stopped for good, and looks identical to one that finished.
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        print(
            f"[WallClock] {marker} present but SLURM_JOB_ID is unset — not under "
            "Slurm, so nothing to requeue. Re-run the same command to continue "
            "from the checkpoint.",
            file=sys.stderr,
        )
        return False

    print(f"[WallClock] Yield marker found; requeueing job {job_id} to continue training.")
    completed = subprocess.run(
        ["scontrol", "requeue", job_id], capture_output=True, text=True
    )
    if completed.returncode == 0:
        return True
    print(
        f"FATAL: `scontrol requeue {job_id}` failed (rc={completed.returncode}): "
        f"{completed.stderr.strip() or completed.stdout.strip()}\n"
        "       This run YIELDED at the wall clock and is NOT finished. Some "
        "clusters disallow requeue; chain it explicitly instead:\n"
        f"         sbatch --dependency=afterany:{job_id} ...  (see "
        "scripts/training/submit_experiment_array.sh)",
        file=sys.stderr,
    )
    return False


def read_overrides_file(path: str | Path) -> list[str]:
    """Read a frozen ``key=value`` override list into ``--override`` args.

    The list is snapshotted to a file at submit time rather than passed through
    ``sbatch --export``, which splits its value on commas: an override such as
    ``data.transforms=[a,b]`` would arrive truncated, and a truncated override
    still validates and still trains (pitfall 9).

    Blank lines and ``#`` comments are skipped. A line with no ``=``, or an
    empty key, raises rather than being dropped — a silently-ignored override is
    indistinguishable from one that took effect.
    """
    args: list[str] = []
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line or not line.split("=", 1)[0].strip():
            raise ValueError(
                f"{path}:{lineno}: override must be 'key=value', got {raw!r}"
            )
        args += ["--override", line]
    return args


def read_verb_args_file(path: str | Path) -> list[str]:
    """Read the submitter's frozen verb args, one per line, in order.

    A file rather than an ``--export`` knob for the same reason as the override
    list: ``sbatch --export`` splits its value on commas, so ``--metrics
    psnr,ssim`` would arrive truncated and the verb would still run (pitfall 9).
    Each line is one argv element and is passed VERBATIM — no splitting, no
    unquoting — so a value keeps its spaces. Blank lines and ``#`` comments are
    dropped.
    """
    return [
        raw
        for raw in Path(path).read_text().splitlines()
        if raw.strip() and not raw.lstrip().startswith("#")
    ]


def verb_option_strings(verb: str) -> frozenset[str]:
    """Option strings the real ``spectramr`` parser accepts for ``verb``.

    Asked of :func:`spectramr.cli.app.build_parser` rather than tabulated here:
    which flags a verb takes is the parser's fact to state, and a second copy
    would keep answering confidently after the first one moved
    (non-negotiable 17). ``build_parser`` is deliberately torch-free, so this
    stays runnable on the torch-less interpreter the dry run and the submit-time
    pre-screen use. An unknown verb yields an empty set.
    """
    from spectramr.cli.app import build_parser

    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            sub_parser = action.choices.get(verb)
            if sub_parser is None:
                return frozenset()
            return frozenset(
                opt for a in sub_parser._actions for opt in a.option_strings
            )
    return frozenset()


def validate_verb_plan(verb: str, *, resume: bool, overrides: bool) -> str | None:
    """Why ``verb`` cannot run this array, or ``None`` when it can.

    The array injects the config as ``--config <path>`` and appends the knobs it
    was asked for, so a verb that takes neither is not a preference question: it
    is an argparse error that every task would hit identically, after its queue
    wait and its audit. Reported once here instead — the submit wrapper runs this
    same check before the array is even sized.

    ``launch.PIPELINE_VERBS`` is the membership owner because it enumerates
    exactly the verbs whose config arrives as a flag; ``train-distributed`` is
    absent from it on purpose, being the spelling :func:`build_launch_argv`
    resolves for a process-group arm rather than one a submitter names.
    """
    from spectramr.cli.launch import PIPELINE_VERBS

    if verb not in PIPELINE_VERBS:
        return (
            f"verb {verb!r} cannot be dispatched per array task. The array hands "
            "each arm its config as `--config <path>`, so the verb must accept "
            f"that flag; the ones that do are: {', '.join(PIPELINE_VERBS)}. "
            "`train-distributed` is not named here because it is resolved "
            "automatically for an arm whose parallel.strategy needs a process "
            "group -- ask for `train`. `audit` takes its config positionally and "
            "already runs as the per-task pre-flight; `predict` takes "
            "--model/--input, so use `infer` for config-driven inference."
        )

    options = verb_option_strings(verb)
    if overrides and "--override" not in options:
        return (
            f"verb {verb!r} takes no --override, so the config overrides this "
            "submission carries (-O / TRAIN_ITERS / --output-base) cannot be "
            "applied to it. Drop them, or run a verb that accepts them (train, "
            "sanity_check, experiment)."
        )
    if resume and "--resume" not in options:
        return (
            f"verb {verb!r} takes no --resume, so RESUME=1 / --prod cannot be "
            "honoured. Drop the resume knob, or run `train`."
        )
    return None


def _override_keys(override_args: list[str]) -> list[str]:
    """Extract the dotted keys from a flat ``["--override", "k=v", ...]`` list."""
    return [a.split("=", 1)[0] for a in override_args if a != "--override"]


def _dispatch(
    *,
    manifest: str,
    index: int,
    train_iters: int | None,
    resume: bool,
    no_audit: bool,
    dispatch_dir: str,
    dry_run: bool,
    prod: bool = False,
    output_base: str | None = None,
    nproc_per_node: int = 1,
    overrides_file: str | None = None,
    verb: str = "train",
    verb_args_file: str | None = None,
) -> int:
    # --prod runs the actual experiment (FULL config max_iterations). A smoke cap
    # (--train-iters) is the opposite intent, so combining them is a contradiction
    # — fail loud rather than silently let one win (CLAUDE.md #9/#15).
    if prod and train_iters is not None:
        print(
            "FATAL: --prod is mutually exclusive with --train-iters "
            f"(got --train-iters={train_iters}). --prod runs the FULL config "
            "max_iterations (the actual experiment); drop the smoke cap to run "
            "production.",
            file=sys.stderr,
        )
        return 2

    # The verb decides whether the knobs above can be spelled at all, so it is
    # settled before the manifest is even read: an impossible plan is the same
    # refusal on every task and costs nothing to report first.
    plan_error = validate_verb_plan(
        verb,
        resume=resume or prod,
        overrides=train_iters is not None
        or output_base is not None
        or overrides_file is not None,
    )
    if plan_error is not None:
        print(f"FATAL: {plan_error}", file=sys.stderr)
        return 2

    verb_args = read_verb_args_file(verb_args_file) if verb_args_file else []

    try:
        config = resolve_manifest_config(manifest, index)
    except IndexError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    name = Path(config).stem
    print(f"[manifest_dispatch] task {index}: {config}")
    print(f"[VERB] spectramr {verb}" + (f" {' '.join(verb_args)}" if verb_args else ""))
    # Run-mode provenance (stamped into the SLURM .out — #15): make the
    # smoke-vs-production decision explicit and auditable, not inferred from the
    # absence of a cap.
    if prod:
        print("[MODE] PROD — actual experiment: full config max_iterations, no smoke cap")
        # A production arm does not fit in one 120h allocation, so a production
        # run is a CHAIN by definition and every link must continue the last.
        # Without this a yielded task requeues, starts from iteration 0, yields
        # again, and burns the allocation forever without ever finishing.
        if not resume:
            print("[MODE] PROD implies --resume if-present (wall-clock chaining)")
            resume = True
    elif train_iters is not None:
        print(f"[MODE] SMOKE — capped at TRAIN_ITERS={train_iters}")
    else:
        print("[MODE] default — full config max_iterations (no --prod, no smoke cap)")

    if not Path(config).is_file():
        print(f"FATAL: config file missing: {config}", file=sys.stderr)
        return 1

    overrides: list[str] = []
    if train_iters is not None:
        overrides, messages = compute_iter_cap_overrides(config, train_iters)
        for msg in messages:
            print(msg)

    # Route output into the campaign tree (campaign array mode). Appended after
    # any smoke-cap overrides so both land; absent (None) leaves the config's own
    # output_dir untouched (standalone array dispatcher path).
    if output_base is not None:
        out_override = output_dir_override(output_base, config)
        overrides += out_override
        print(f"[OUTPUT] routing into {out_override[-1].split('=', 1)[1]}")

    # The caller's explicit -O list lands last. A key already set by the smoke
    # cap is a contradiction of intent, not a precedence question, so it is
    # refused the way --prod/--train-iters is rather than resolved silently.
    if overrides_file is not None:
        user_overrides = read_overrides_file(overrides_file)
        clash = sorted(set(_override_keys(user_overrides)) & set(_override_keys(overrides)))
        if clash:
            print(
                f"FATAL: --overrides-file sets {', '.join(clash)}, which the smoke "
                "cap already set. Drop --train-iters and set every key explicitly, "
                "or drop those keys from the overrides file.",
                file=sys.stderr,
            )
            return 2
        overrides += user_overrides
        print(f"[OVERRIDE] {len(user_overrides) // 2} from {overrides_file}")

    strategy = read_parallel_strategy(config)

    if dry_run:
        print(f"[DRYRUN] would audit + {verb}: {config}")
        print(f"[DRYRUN] declared parallel.strategy={strategy!r}")
        if overrides:
            print(f"[DRYRUN] train overrides: {' '.join(overrides)}")
        # The predicate is the registry's, and the registry needs torch. The dry
        # run is documented to work on a torch-less interpreter, so say which of
        # the two happened rather than inferring a verb (non-negotiable 18).
        try:
            distributed = requires_process_group(strategy)
        except ImportError as exc:
            print(
                f"[DRYRUN] parallel-strategy registry unavailable ({exc.name} not "
                "importable); the launch verb is resolved in the real run, after "
                "the venv is active."
            )
        else:
            argv = build_launch_argv(
                config,
                overrides,
                resume,
                distributed=distributed and verb == "train",
                nproc_per_node=nproc_per_node,
                master_port=resolve_rendezvous_port(),
                verb=verb,
                verb_args=verb_args,
            )
            print(f"[DRYRUN] launch: {' '.join(argv)}")
        return 0

    py = [sys.executable, "-m", "spectramr.cli"]

    if not no_audit:
        Path(dispatch_dir).mkdir(parents=True, exist_ok=True)
        audit_json = Path(dispatch_dir) / f"audit_{name}_{index}.json"
        print(f"[AUDIT] spectramr.cli audit {config} --strict")
        with audit_json.open("w") as fh:
            audit_rc = subprocess.run([*py, "audit", config, "--strict"], stdout=fh).returncode
        if audit_rc != 0:
            print(
                f"AUDIT FAILED for {name} - skipping train (see {audit_json})",
                file=sys.stderr,
            )
            return 2
        print("[AUDIT] passed")

    # The upgrade is train-only: `parallel.strategy` describes how the arm
    # TRAINS, and inference or a sweep verb running single-process under it is
    # correct rather than a downgrade. Said out loud so a log reader is not left
    # inferring why a deepspeed arm ran without torchrun.
    needs_group = requires_process_group(strategy)
    distributed = needs_group and verb == "train"
    if distributed:
        print(
            f"[PARALLEL] parallel.strategy={strategy!r} requires a process group "
            f"-> torchrun --nproc_per_node={nproc_per_node}, verb 'train-distributed'"
        )
    elif needs_group:
        print(
            f"[PARALLEL] parallel.strategy={strategy!r} needs a process group to "
            f"TRAIN; verb {verb!r} is single-process and runs as declared"
        )
    else:
        print(f"[PARALLEL] parallel.strategy={strategy!r} -> single-process {verb!r}")

    argv = build_launch_argv(
        config,
        overrides,
        resume,
        distributed=distributed,
        nproc_per_node=nproc_per_node,
        master_port=resolve_rendezvous_port(),
        verb=verb,
        verb_args=verb_args,
    )
    # Name the marker and hand the location to the run, so both sides agree by
    # construction rather than by two derivations that must be kept in step.
    marker = yield_marker_path(dispatch_dir, name, index)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    os.environ[SPECTRAMR_WALL_CLOCK_MARKER] = str(marker)

    if verb == "train":
        print(f"[TRAIN] {' '.join(argv)}")
    else:
        print(f"[LAUNCH] {verb}: {' '.join(argv)}")
    train_rc = subprocess.run(argv).returncode

    if not marker.is_file():
        return train_rc

    # A wall-clock yield is a SUCCESSFUL partial run: the loop saved and stopped
    # on purpose, so the task is requeued to pick up where it left off.
    #
    # Except when the launch carried no --resume. The requeued task would then
    # re-run this same command, start at iteration 0, yield at the next wall and
    # repeat -- an allocation burned forever without ever passing 120h of
    # training. Refuse loudly instead; the checkpoint on disk is still good.
    if not resume:
        print(
            f"FATAL: {name} yielded at the wall clock but was launched WITHOUT "
            "resume, so requeueing it would restart it from iteration 0 and loop "
            "forever. Not requeueing. Resubmit with RESUME=1 (or --prod, which "
            "implies it) to continue from the checkpoint just written.",
            file=sys.stderr,
        )
        return 1
    return 0 if requeue_this_task(marker) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="manifest_dispatch",
        description="Resolve + audit + train one experiment from a SLURM array manifest.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Newline-separated config list. Required unless --check-plan.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="0-based array task index. Required unless --check-plan.",
    )
    parser.add_argument(
        "--verb",
        default="train",
        help=(
            "Pipeline verb to run on every arm (default: train). Must accept the "
            "config as a --config flag -- see launch.PIPELINE_VERBS."
        ),
    )
    parser.add_argument(
        "--verb-args-file",
        default=None,
        help=(
            "File of trailing args for the verb, one argv element per line, "
            "appended last so they win. Frozen at submit time; a file rather "
            "than an env var because sbatch --export splits on commas."
        ),
    )
    parser.add_argument(
        "--check-plan",
        action="store_true",
        help=(
            "Validate the verb against the knobs given (--resume / --prod / "
            "--train-iters / --overrides-file) and exit, without a manifest. The "
            "submit wrapper runs this before sizing the array so an impossible "
            "plan costs one line instead of one failing task per arm."
        ),
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=None,
        help="Smoke cap on BOTH training.max_iterations and validation.eval_interval.",
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        help=(
            "Run the actual experiment: the FULL config max_iterations with no "
            "smoke cap. Mutually exclusive with --train-iters (errors if both given)."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true", help="Append --resume if-present."
    )
    parser.add_argument("--no-audit", action="store_true", help="Skip the audit pre-flight.")
    parser.add_argument("--dispatch-dir", default=_DEFAULT_DISPATCH_DIR)
    parser.add_argument(
        "--output-base",
        default=None,
        help=(
            "Route this task's training.output_dir to <output-base>/<config-stem> "
            "(campaign array mode). Default None keeps the config's own output_dir."
        ),
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help=(
            "Ranks to fork for an arm whose parallel.strategy needs a process "
            "group. This is the per-task GPU grant (the .sbatch passes the "
            "derived GPUS_PER_NODE); single-process arms ignore it."
        ),
    )
    parser.add_argument(
        "--overrides-file",
        default=None,
        help=(
            "File of key=value config overrides, one per line, applied to every "
            "task. Frozen at submit time; a file rather than an env var because "
            "sbatch --export splits on commas."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config + planned cap/command, then exit 0 (no torch).",
    )
    args = parser.parse_args(argv)

    if args.check_plan:
        error = validate_verb_plan(
            args.verb,
            resume=args.resume or args.prod,
            overrides=args.train_iters is not None
            or args.output_base is not None
            or args.overrides_file is not None,
        )
        if error is not None:
            print(f"FATAL: {error}", file=sys.stderr)
            return 2
        print(f"[VERB] spectramr {args.verb}: plan accepted")
        return 0

    if args.manifest is None or args.index is None:
        parser.error(
            "--manifest and --index are required to dispatch a task "
            "(pass --check-plan to validate a verb on its own)"
        )

    return _dispatch(
        manifest=args.manifest,
        index=args.index,
        overrides_file=args.overrides_file,
        train_iters=args.train_iters,
        resume=args.resume,
        no_audit=args.no_audit,
        dispatch_dir=args.dispatch_dir,
        dry_run=args.dry_run,
        prod=args.prod,
        output_base=args.output_base,
        nproc_per_node=args.nproc_per_node,
        verb=args.verb,
        verb_args_file=args.verb_args_file,
    )


if __name__ == "__main__":  # pragma: no cover - module CLI entry
    raise SystemExit(main())
