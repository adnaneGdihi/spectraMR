"""Pipeline-agnostic execution backends — *where* a ``spectramr`` run executes.

A backend answers exactly one question: "run this ``spectramr <verb> --config …``
invocation **here**" — local, in a Docker/Apptainer container, or as a SLURM
job. It knows nothing about training, inference, or any specific pipeline; the
verb is opaque. This is the WHERE axis of the execution-mode model, orthogonal
to WHAT (the config) and WHICH-PIPELINE (the verb).

The shell-out backends here (Docker / Apptainer / SLURM) are **pure**: they
build a command list / sbatch script from a :class:`SpectraMRInvocation` +
:class:`ResourceSpec` (the ``build_*`` methods, fully testable without spawning
anything) and only ``run`` executes it. They import **no** inward ``spectramr``
layers, so they live in ``infrastructure`` and can be shared by the campaign
orchestrator (WS-E). The in-process ``LocalBackend`` lives in ``cli`` instead,
because it must call into the pipeline (a leftward import forbidden here).
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

#: Default container artifact names (overridable via env).
_DEFAULT_DOCKER_IMAGE = "spectramr:v6.1"
_DEFAULT_APPTAINER_SIF = "spectramr.sif"

#: Env-var prefix by which ``spectramr launch`` hands the resolved ResourceSpec to
#: the child run, so its provenance records the backend + resources it ran
#: under (pitfall #15c). Inherited in-process and by SLURM (``--export=ALL``),
#: and forwarded into containers via ``--env``.
_LAUNCH_ENV_PREFIX = "SPECTRAMR_LAUNCH_"
#: ``(env-suffix, ResourceSpec attr, is_int)`` for the resource fields stamped.
_LAUNCH_RESOURCE_FIELDS: tuple[tuple[str, str, bool], ...] = (
    ("ACCOUNT", "account", False),
    ("PARTITION", "partition", False),
    ("MEM", "mem", False),
    ("GPUS", "gpus", True),
    ("TIME", "time", False),
    ("NODES", "nodes", True),
    ("NTASKS", "ntasks", True),
    ("CPUS_PER_TASK", "cpus_per_task", True),
)


@dataclass(frozen=True)
class SpectraMRInvocation:
    """A single ``spectramr`` command to execute, independent of *where*.

    Attributes:
        verb: The CLI subcommand (``train``, ``infer``, ``hpo``, …).
        config: Path to the config YAML (``--config``), if the verb takes one.
        extra_args: Verb-specific passthrough flags (e.g. ``--resume auto`` for
            train, ``--checkpoint/--input/--output`` for infer).
    """

    verb: str
    config: str | None = None
    extra_args: tuple[str, ...] = ()

    def to_cli_args(self) -> list[str]:
        """Render the invocation as ``spectramr`` CLI arguments (no program name)."""
        args = [self.verb]
        if self.config:
            args += ["--config", self.config]
        args += list(self.extra_args)
        return args


@dataclass(frozen=True)
class ResourceSpec:
    """Compute resources for a run (centralized, env-defaulted, stampable).

    ``account`` / ``partition`` fall back to the ``SPECTRAMR_SLURM_ACCOUNT`` /
    ``SPECTRAMR_SLURM_PARTITION`` env vars when not given (pitfall #15 —
    resolved + readable). Neither carries a built-in default, because an
    allocation name is site-specific: there is no value this tree could ship
    that is right anywhere but one cluster. An unresolved account is reported
    by :meth:`SlurmBackend.render_directives`, not guessed.
    """

    account: str | None = None
    partition: str | None = None
    mem: str = "128G"
    #: GPU COUNT. Default 1 (rendered as SLURM's generic ``--gpus=1``, no GPU
    #: type pinned so the chosen account/partition/cluster picks the hardware).
    #: The sentinel ``0`` means "all visible GPUs" (the container backends'
    #: unrestricted mode); a negative value is rejected (pitfall #9 — no silent
    #: fallback).
    gpus: int = 1
    time: str = "24:00:00"
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int = 8
    #: SLURM job-array spec, e.g. ``"0-9"``, ``"0-19%4"``, ``"1,3,5"``. ``None``
    #: (default) emits no ``#SBATCH --array`` line, so every existing caller +
    #: the campaign golden stay byte-identical. Set it to fan one script out over
    #: a manifest of arms (the SSOT replacement for the hand-written array
    #: ``.sbatch`` files).
    array: str | None = None
    #: Allow SLURM to re-run this job. Required for wall-clock chaining: the run
    #: yields a margin before its allocation ends and asks to be requeued, which
    #: SLURM refuses outright without this. ``False`` (default) emits no
    #: directive, so every existing caller and the campaign golden stay
    #: byte-identical.
    requeue: bool = False
    #: ``--open-mode``. ``"append"`` goes with :attr:`requeue`: a requeued task
    #: otherwise TRUNCATES its predecessor's ``.out``, destroying the record of
    #: why the previous link stopped. ``None`` (default) emits no directive.
    open_mode: str | None = None

    def __post_init__(self) -> None:
        if self.gpus < 0:
            raise ValueError(
                f"ResourceSpec.gpus must be >= 0 (0 = all visible GPUs); got {self.gpus}."
            )
        if self.array is not None and not self.array.strip():
            # An empty/whitespace array is a mistake, not "unset" — raise rather
            # than silently emit an invalid ``#SBATCH --array=`` (pitfall #15).
            raise ValueError(
                "ResourceSpec.array, when set, must be a non-empty SLURM array "
                "spec (e.g. '0-9', '0-19%4'); use None to disable arrays."
            )
        if self.account is None:
            env_acct = os.environ.get("SPECTRAMR_SLURM_ACCOUNT")
            if env_acct:
                object.__setattr__(self, "account", env_acct)
        if self.partition is None:
            env_part = os.environ.get("SPECTRAMR_SLURM_PARTITION")
            if env_part:
                object.__setattr__(self, "partition", env_part)


@dataclass
class RunHandle:
    """Reference to a launched run (or, for a dry run, the command that would run)."""

    backend: str
    id: str | None = None
    returncode: int | None = None
    command: list[str] | None = None
    script: str | None = None
    meta: dict = field(default_factory=dict)


@runtime_checkable
class ExecutionBackend(Protocol):
    """Where a ``spectramr`` invocation executes (pipeline-agnostic)."""

    name: str

    def run(
        self,
        invocation: SpectraMRInvocation,
        resources: ResourceSpec,
        *,
        dry_run: bool = False,
    ) -> RunHandle: ...


def export_launch_env(where: str, resources: ResourceSpec) -> None:
    """Export ``SPECTRAMR_LAUNCH_*`` into ``os.environ`` for the child run.

    Called by ``spectramr launch`` before ``backend.run``. The in-process backend
    inherits these directly; SLURM inherits via ``sbatch --export=ALL``; the
    container backends forward them with :func:`_launch_env_pairs`. The child's
    :func:`collect_run_provenance` reads them back via
    :func:`resolve_launch_provenance` and stamps them into ``run_summary.json``.
    """
    os.environ[f"{_LAUNCH_ENV_PREFIX}BACKEND"] = where
    for suffix, attr, _is_int in _LAUNCH_RESOURCE_FIELDS:
        val = getattr(resources, attr)
        if val is not None:
            os.environ[f"{_LAUNCH_ENV_PREFIX}{suffix}"] = str(val)


def resolve_launch_provenance() -> dict[str, object]:
    """Read ``SPECTRAMR_LAUNCH_*`` back into a provenance dict (pitfall #15c).

    Mirrors :func:`spectramr.plugins.resolve_plugin_provenance`. Returns ``{}``
    when ``SPECTRAMR_LAUNCH_BACKEND`` is absent — a plain ``spectramr train`` (not
    launched via ``spectramr launch``) is a strict no-op. Raises ``ValueError`` on
    an unparseable integer field (pitfall #9 — never silently drop a knob).
    """
    backend = os.environ.get(f"{_LAUNCH_ENV_PREFIX}BACKEND")
    if not backend:
        return {}
    record: dict[str, object] = {"backend": backend}
    for suffix, attr, is_int in _LAUNCH_RESOURCE_FIELDS:
        raw = os.environ.get(f"{_LAUNCH_ENV_PREFIX}{suffix}")
        if raw is None:
            continue
        if is_int:
            try:
                record[attr] = int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"{_LAUNCH_ENV_PREFIX}{suffix}={raw!r} is not an integer."
                ) from exc
        else:
            record[attr] = raw
    return record


def _launch_env_pairs() -> list[str]:
    """``--env KEY=VALUE`` pairs for each ``SPECTRAMR_LAUNCH_*`` var present, so a
    container forwards the launch provenance to the in-container pipeline."""
    pairs: list[str] = []
    for key in sorted(os.environ):
        if key.startswith(_LAUNCH_ENV_PREFIX):
            pairs.extend(["--env", f"{key}={os.environ[key]}"])
    return pairs


def _docker_gpu_args(gpus: int) -> list[str]:
    """Docker ``--gpus`` flag honoring the COUNT; ``0`` = all visible GPUs.

    Docker re-numbers the allocated devices to ``0..N-1`` inside the container,
    so the run sees exactly ``gpus`` GPUs (pitfall #15 — the knob does something).
    """
    return ["--gpus", "all" if gpus == 0 else str(gpus)]


def _apptainer_gpu_args(gpus: int) -> list[str]:
    """Apptainer GPU flags honoring the COUNT; ``0`` = all visible GPUs.

    Apptainer has no per-count ``--gpus`` flag, so ``--nv`` enables CUDA and
    ``CUDA_VISIBLE_DEVICES=0,…,N-1`` restricts the container to ``gpus`` devices.
    """
    args = ["--nv"]
    if gpus >= 1:
        visible = ",".join(str(i) for i in range(gpus))
        args += ["--env", f"CUDA_VISIBLE_DEVICES={visible}"]
    return args


class _ContainerBackend:
    """Shared logic for the OCI/Apptainer container backends."""

    name = "container"

    def build_command(
        self, invocation: SpectraMRInvocation, resources: ResourceSpec
    ) -> list[str]:  # pragma: no cover - overridden
        raise NotImplementedError

    def run(
        self,
        invocation: SpectraMRInvocation,
        resources: ResourceSpec,
        *,
        dry_run: bool = False,
    ) -> RunHandle:
        cmd = self.build_command(invocation, resources)
        if dry_run:
            return RunHandle(self.name, command=cmd)
        proc = subprocess.run(cmd, check=False)
        return RunHandle(self.name, returncode=proc.returncode, command=cmd)


class DockerBackend(_ContainerBackend):
    """Run the invocation inside a Docker container (``docker run``)."""

    name = "docker"

    def __init__(self, image: str | None = None) -> None:
        self._image = image

    @property
    def image(self) -> str:
        return self._image or os.environ.get("SPECTRAMR_DOCKER_IMAGE", _DEFAULT_DOCKER_IMAGE)

    def build_command(self, invocation: SpectraMRInvocation, resources: ResourceSpec) -> list[str]:
        workdir = os.getcwd()
        return [
            "docker",
            "run",
            *_docker_gpu_args(resources.gpus),
            *_launch_env_pairs(),  # forward SPECTRAMR_LAUNCH_* for provenance
            "-v",
            f"{workdir}:/workspace",
            self.image,
            *invocation.to_cli_args(),
        ]


class ApptainerBackend(_ContainerBackend):
    """Run the invocation inside an Apptainer container (``apptainer run --nv``)."""

    name = "apptainer"

    def __init__(self, sif: str | None = None) -> None:
        self._sif = sif

    @property
    def sif(self) -> str:
        return self._sif or os.environ.get("SPECTRAMR_APPTAINER_SIF", _DEFAULT_APPTAINER_SIF)

    def build_command(self, invocation: SpectraMRInvocation, resources: ResourceSpec) -> list[str]:
        workdir = os.getcwd()
        return [
            "apptainer",
            "run",
            *_apptainer_gpu_args(resources.gpus),
            *_launch_env_pairs(),  # forward SPECTRAMR_LAUNCH_* for provenance
            "--bind",
            f"{workdir}:/workspace",
            self.sif,
            *invocation.to_cli_args(),
        ]


class SlurmBackend:
    """Submit the invocation as a SLURM batch job (``sbatch``).

    Generates one ``#SBATCH`` script from the :class:`ResourceSpec`. This is the
    single SLURM-script generator the campaign orchestrator consolidates onto in
    WS-E (so the two generators stop drifting).
    """

    name = "slurm"

    def render_directives(
        self,
        resources: ResourceSpec,
        *,
        job_name: str,
        output: str | None = None,
        error: str | None = None,
        mail_type: str | None = None,
        mail_user: str | None = None,
    ) -> str:
        """Render the ``#SBATCH`` directive block — the single source of truth.

        Used by :meth:`build_script` (the launcher) AND the campaign
        orchestrator's ``generate_job_script`` (WS-E consolidation), so the
        directive set, line order, and "partition only when set" rule live in
        ONE place rather than drifting across two generators. The optional
        ``output`` / ``error`` / ``mail_*`` lines are emitted only when supplied
        (the campaign provides them; the launcher does not).
        """
        if resources.account is None:
            # Not a silent skip and not a guess (non-negotiable 3). Rendering
            # ``--account=None`` submits a job that sbatch rejects with an
            # error naming neither the knob nor the env var, which is how a
            # missing allocation turns into ten minutes of confusion.
            raise ValueError(
                "SLURM submission needs an allocation account and this tree "
                "ships no default -- the value is site-specific. Set "
                "$SPECTRAMR_SLURM_ACCOUNT (see .env.example) or pass "
                "ResourceSpec(account=...)."
            )
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
        ]
        # Account, like partition below, is emitted ONLY when set. Rendering
        # ``--account=None`` (or a placeholder) submits a job that SLURM rejects;
        # omitting the directive lets the scheduler apply the submitter's default
        # allocation, which is the correct behavior on any cluster but this one.
        if resources.account:
            lines.append(f"#SBATCH --account={resources.account}")
        if output:
            lines.append(f"#SBATCH --output={output}")
        if error:
            lines.append(f"#SBATCH --error={error}")
        lines += [
            f"#SBATCH --time={resources.time}",
            f"#SBATCH --nodes={resources.nodes}",
            f"#SBATCH --ntasks={resources.ntasks}",
            f"#SBATCH --cpus-per-task={resources.cpus_per_task}",
            f"#SBATCH --mem={resources.mem}",
            # ``--gpus=N`` (a generic COUNT) rather than ``--gres=gpu:<type>:N``:
            # pinning a GPU type (e.g. ``ada``) hardwires the hardware and only
            # schedules on nodes carrying it. A plain count lets the caller's
            # chosen environment — account / partition / cluster — decide which
            # GPU the job lands on.
            f"#SBATCH --gpus={resources.gpus}",
        ]
        # Job-array line directly after --gpus, only when set — so the default
        # (None) leaves the directive block byte-identical for every non-array
        # caller and the campaign golden.
        if resources.array:
            lines.append(f"#SBATCH --array={resources.array}")
        if resources.partition:
            lines.append(f"#SBATCH --partition={resources.partition}")
        # After --partition and before the mail lines; the committed .sbatch
        # headers are regenerated from this order and compared line by line.
        if resources.requeue:
            lines.append("#SBATCH --requeue")
        if resources.open_mode:
            lines.append(f"#SBATCH --open-mode={resources.open_mode}")
        if mail_type:
            lines.append(f"#SBATCH --mail-type={mail_type}")
        if mail_user:
            lines.append(f"#SBATCH --mail-user={mail_user}")
        return "\n".join(lines) + "\n"

    def build_script(self, invocation: SpectraMRInvocation, resources: ResourceSpec) -> str:
        header = self.render_directives(resources, job_name=f"spectramr-{invocation.verb}")
        cli = "python -m spectramr.cli " + " ".join(shlex.quote(a) for a in invocation.to_cli_args())
        return header + "\nset -euo pipefail\n" + f"{cli}\n"

    def submit_script(
        self,
        script: str,
        *,
        dependency_job_ids: list[int] | None = None,
        timeout: int = 30,
    ) -> int:
        """Submit a script string to SLURM via ``sbatch`` (stdin) → job id.

        The single canonical sbatch primitive: both the launcher's :meth:`run`
        and the campaign orchestrator's ``SLURMBackend.submit_job`` route through
        here, so there is one submission path (one place that invokes ``sbatch``,
        supports ``--dependency=afterok`` chaining, and parses the job id).

        Raises:
            RuntimeError: if ``sbatch`` fails or its output is unparseable.
        """
        cmd = ["sbatch"]
        if dependency_job_ids:
            dep = "afterok:" + ":".join(str(j) for j in dependency_job_ids)
            cmd.extend(["--dependency", dep])
        result = subprocess.run(
            cmd,
            input=script,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed (exit {result.returncode}): {result.stderr.strip()}")
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse sbatch output: {result.stdout.strip()}")
        return int(match.group(1))

    def run(
        self,
        invocation: SpectraMRInvocation,
        resources: ResourceSpec,
        *,
        dry_run: bool = False,
    ) -> RunHandle:
        script = self.build_script(invocation, resources)
        if dry_run:
            return RunHandle(self.name, script=script)
        job_id = self.submit_script(script)
        return RunHandle(self.name, id=str(job_id), returncode=0, script=script)
