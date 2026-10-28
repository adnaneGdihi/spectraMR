"""Unified launcher — ``spectramr launch`` and the in-process ``LocalBackend``.

``launch`` is the single front door over the execution-mode hypercube: it picks
a backend by ``--where`` (local / docker / apptainer / slurm), builds a
pipeline-agnostic :class:`SpectraMRInvocation` from ``--pipeline`` + ``--config``,
resolves a :class:`ResourceSpec`, and runs it. The existing dedicated commands
(``spectramr train``, ``sbatch …``, ``spectramr campaign submit``) all keep working
— ``launch`` is additive.

This lives in the ``cli`` layer (not ``infrastructure``) because the in-process
:class:`LocalBackend` calls into the CLI dispatch (``cli`` may import rightward;
``infrastructure`` may not import ``cli``/``pipelines``). The shell-out backends
(Docker/Apptainer/SLURM) are the pure infra ones in
:mod:`spectramr.infrastructure.execution`.
"""

from __future__ import annotations

import argparse
import logging

from spectramr.infrastructure.execution import (
    ApptainerBackend,
    DockerBackend,
    ExecutionBackend,
    SpectraMRInvocation,
    ResourceSpec,
    RunHandle,
    SlurmBackend,
    export_launch_env,
)

logger = logging.getLogger(__name__)

#: Pipeline verbs ``launch`` can dispatch. ``launch`` injects the config as a
#: ``--config`` FLAG (``SpectraMRInvocation.to_cli_args`` → ``[verb, "--config",
#: cfg, *extra]``), so only verbs whose subparser accepts a ``--config`` flag
#: belong here. Deliberately EXCLUDED — each would fail at argparse time, so
#: advertising it is a CLI lie (pitfall #15). The membership is regression-
#: tested against the real parser in ``tests/unit/cli/test_launch.py``:
#:   * ``predict`` — takes ``--model``/``--input``, no ``--config`` (use
#:     ``infer``, which is the config-driven inference verb);
#:   * ``audit`` — takes the config POSITIONALLY (``spectramr audit X.yaml``);
#:   * ``report`` — drives an output dir via ``--exp-dir``, no ``--config``;
#:   * ``meta-evaluate`` — consumes a metric set / ``--input``, no ``--config``;
#:   * ``infer-dataset`` — a ``[DEPRECATED]`` alias for ``infer``.
#: The verb's OTHER required args (``infer``'s ``--checkpoint``/``--input``,
#: ``experiment``'s ``--experiment``, ``hpo``'s ``--model-type``) are supplied
#: via the ``--`` passthrough (``SpectraMRInvocation.extra_args``).
PIPELINE_VERBS = (
    "train",
    "sanity_check",
    "infer",
    "hpo",
    "ablation",
    "experiment",
    "profile",
)

WHERE_CHOICES = ("local", "docker", "apptainer", "slurm")
FANOUT_CHOICES = ("single", "campaign")


class LocalBackend:
    """Run the invocation in THIS process via the CLI dispatch (no subprocess).

    Reuses the full ``spectramr`` command dispatch in-process, so every verb
    (train/infer/…) runs through the same code path as the dedicated command.
    """

    name = "local"

    def run(
        self,
        invocation: SpectraMRInvocation,
        resources: ResourceSpec,
        *,
        dry_run: bool = False,
    ) -> RunHandle:
        argv = invocation.to_cli_args()
        if dry_run:
            return RunHandle(self.name, command=["spectramr", *argv])
        # Lazy import: cli.app imports this module to register the subparser, so
        # importing app at module load would cycle.
        from spectramr.cli.app import main

        returncode = main(argv)
        return RunHandle(self.name, returncode=returncode, command=["spectramr", *argv])


_BACKENDS: dict[str, type] = {
    "local": LocalBackend,
    "docker": DockerBackend,
    "apptainer": ApptainerBackend,
    "slurm": SlurmBackend,
}


def resolve_backend(where: str) -> ExecutionBackend:
    """Resolve a ``--where`` value to a backend instance (raise on unknown)."""
    try:
        backend_cls = _BACKENDS[where]
    except KeyError:
        raise ValueError(f"Unknown --where '{where}'. Choose from {sorted(_BACKENDS)}.") from None
    return backend_cls()  # type: ignore[return-value]


def _resources_from_args(args: argparse.Namespace) -> ResourceSpec:
    """Build a ResourceSpec from CLI flags, omitting unset ones (use defaults)."""
    kwargs = {
        k: v
        for k, v in {
            "account": getattr(args, "account", None),
            "partition": getattr(args, "partition", None),
            "mem": getattr(args, "mem", None),
            "gpus": getattr(args, "gpus", None),
            "time": getattr(args, "time", None),
            "nodes": getattr(args, "nodes", None),
        }.items()
        if v is not None
    }
    return ResourceSpec(**kwargs)


def _print_dry_run(handle: RunHandle) -> None:
    """Print what a dry run *would* execute (command or sbatch script)."""
    if handle.script:
        print(handle.script)
    elif handle.command:
        print(" ".join(handle.command))


def launch(args: argparse.Namespace) -> int:
    """``spectramr launch`` entry point."""
    if args.fanout == "campaign":
        return _launch_campaign(args)

    invocation = SpectraMRInvocation(
        verb=args.pipeline,
        config=str(args.config) if args.config else None,
        extra_args=tuple(getattr(args, "extra", None) or ()),
    )
    resources = _resources_from_args(args)
    backend = resolve_backend(args.where)

    # Hand the resolved backend + resources to the child run via SPECTRAMR_LAUNCH_*
    # so its run_summary.json records what it ran under (pitfall #15c). Done
    # before backend.run so the in-process / SLURM / container paths all inherit
    # or forward it (and a --dry-run shows the forwarded --env in the command).
    export_launch_env(args.where, resources)

    logger.info(
        "launch: pipeline=%s where=%s fanout=single%s",
        args.pipeline,
        args.where,
        " (dry-run)" if args.dry_run else "",
    )
    handle = backend.run(invocation, resources, dry_run=args.dry_run)
    if args.dry_run:
        _print_dry_run(handle)
        return 0
    return handle.returncode or 0


def _launch_campaign(args: argparse.Namespace) -> int:
    """Delegate ``--fanout campaign`` to the campaign submitter with ``--where``.

    SLURM (default) submits each arm as an sbatch job; ``docker`` / ``apptainer``
    run each arm in a container (synchronously, parallel-mode campaigns only).
    ``local`` is not supported for campaigns (the in-process backend can't drive a
    multi-arm campaign) and fails loudly rather than silently degrading (#9).
    """
    if args.where == "local":
        raise NotImplementedError(
            "--fanout campaign does not support --where local; use --where "
            "slurm/docker/apptainer, or run a single config with `spectramr train`."
        )
    from spectramr.cli.app import main

    argv = ["campaign", "submit", str(args.config), "--where", args.where]
    if args.dry_run:
        argv.append("--dry-run")
    return main(argv)
