#!/usr/bin/env python3
"""Run the blocking PR lane locally, DERIVED from ``.github/workflows/pr-required.yml``.

GitHub Actions is disabled on this repository, so ``pr-required.yml`` describes a lane
that never executes. This runner executes it by hand. It **parses the workflow** rather
than restating it, because a hand-written twin would be a second owner of the same
invariant (non-negotiable 17): both plausible, both green, diverging silently. Adding a
job to the workflow must change what this runs, without editing this file. Same shape as
``_derive_target_methods()`` (PR #1410), which walks ``base.py`` instead of enumerating
it -- and, like it, **raises** when the derivation comes back empty rather than reporting
an empty audited set as a pass.

Three states, never two
-----------------------
Every step is ``PASS``, ``FAIL`` or ``UNRUNNABLE(reason)``, and **UNRUNNABLE is never
folded into PASS**. That is non-negotiable 18's rule -- *absent is a state to report,
never a state to infer* -- and this repository has already been bitten by the opposite:
``docs/known_limitations.rst`` records that ``spectramr audit`` prints a check which
*declined to run* with the same green tick as one that passed, which is why a green audit
is not coverage. A local runner that reported "pip-audit is not installed" as a pass would
rebuild that blindness one level up.

What is deliberately not executed
---------------------------------
``uses:`` steps. They are environment provisioning -- ``actions/checkout``,
``actions/setup-python``, and ``./.github/actions/setup-env``, which does
``pip install torch --index-url .../cpu`` followed by ``pip install -e ".[dev]"``. The
already-provisioned local interpreter plays that role. On an aarch64 / sm_110 box that
substitution is also a *safety* requirement, not a convenience: re-resolving would pull
cu129 wheels over a working cu130 pair and break ``populate_model_registry()`` through the
``torchvision::nms`` ABI trap.

A ``run:`` step that only installs packages is treated the same way -- skipped as
provisioning -- but the packages it names become **requirements for the rest of that
job**, checked before the job runs. That is how ``pip install ruff`` teaches the runner
that ``lint-diff`` needs ``ruff``, with no hand-maintained tool list to drift.

The workflow path and repo root are **parameters, not module constants**, so a planted
workflow can be pointed at (D01#16: a scanner whose scan root is a constant cannot be
aimed at a planted tree, so its ``0`` is never interrogated).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PASS, FAIL, UNRUNNABLE = "PASS", "FAIL", "UNRUNNABLE"

#: Jobs that exist only to aggregate other jobs' ``result`` contexts. They have no
#: local meaning -- this runner *is* the aggregation.
AGGREGATOR_JOBS = frozenset({"required"})

_PIP_INSTALL = re.compile(r"^\s*(?:pip|pip3|python -m pip)\s+install\s+(?P<args>.+)$")
_SUPPORTED_IF = re.compile(r"^steps\.(?P<step>[\w-]+)\.outputs\.(?P<key>[\w-]+)\s*==\s*'(?P<val>[^']*)'$")


@dataclass
class StepResult:
    job: str
    name: str
    state: str
    detail: str = ""


@dataclass
class JobPlan:
    name: str
    steps: list[dict] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)


def load_workflow(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "jobs" not in data:
        raise SystemExit(f"{path}: not a workflow (no `jobs:` mapping)")
    return data


def _pip_packages(run_body: str) -> list[str] | None:
    """Package names if EVERY line of ``run_body`` is a pip install, else ``None``.

    Returning ``None`` for a mixed body is deliberate: a step that installs *and* checks
    must still run its check. Only a pure-provisioning step may be skipped.
    """
    packages: list[str] = []
    for line in run_body.strip().splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        m = _PIP_INSTALL.match(line)
        if not m:
            return None
        for token in m.group("args").split():
            if token.startswith("-"):
                continue
            name = re.split(r"[<>=!~\[]", token.strip("'\""), 1)[0]
            if name:
                packages.append(name)
    return packages or None


def _is_available(package: str) -> bool:
    """A package counts as present if it is on PATH *or* importable.

    Both, because the two provisioning steps in this lane name different kinds of thing:
    ``ruff`` and ``pip-audit`` are executables, ``setuptools`` is import-only.
    """
    if shutil.which(package):
        return True
    module = package.replace("-", "_")
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def plan_jobs(workflow: dict, only: set[str] | None = None) -> list[JobPlan]:
    """Derive the executable plan. Raises if a selected job yields no runnable step."""
    plans: list[JobPlan] = []
    for name, job in workflow["jobs"].items():
        if name in AGGREGATOR_JOBS or (only and name not in only):
            continue
        plan = JobPlan(name=name)
        for step in job.get("steps", []) or []:
            if "run" not in step:
                continue  # `uses:` -- provisioning, see module docstring
            packages = _pip_packages(step["run"])
            if packages is not None:
                plan.requires.extend(packages)
                continue
            plan.steps.append(step)
        if not plan.steps:
            # NOT "0 steps, all green". A job whose derivation comes back empty is a
            # broken derivation, and reporting it as a pass is the exact defect this
            # runner exists to avoid.
            raise SystemExit(
                f"job {name!r}: derived 0 executable steps from the workflow. "
                "Either the parser is wrong or the job is provisioning-only; "
                "neither may be reported as a pass."
            )
        plans.append(plan)
    if not plans:
        raise SystemExit("no jobs selected -- refusing to report an empty lane as green")
    return plans


def _resolve_env(key: str, raw: str, base_env: dict[str, str]) -> str:
    """Resolve a step ``env:`` value, substituting a locally-derived one for ``${{ }}``.

    The lane's two diff-scoped jobs declare ``BASE``/``HEAD`` as
    ``${{ github.event.pull_request.*.sha }}``. There is no pull-request context here, so
    the caller's git-derived values stand in. Passing the literal expression through is
    what a naive ``dict.update`` does, and it fails deep inside the checker as an opaque
    ``git diff ... returned non-zero exit status 128`` rather than as a missing value.
    """
    if "${{" not in raw:
        return raw
    if key in base_env:
        return base_env[key]
    raise SystemExit(
        f"step env {key}={raw!r} is a workflow expression with no local equivalent. "
        "Derive one explicitly; substituting the literal would fail obscurely downstream."
    )


def _should_run(step: dict, outputs: dict[str, dict[str, str]]) -> bool:
    """Evaluate a step ``if:``. An unsupported expression RAISES, never silently runs."""
    expr = str(step.get("if", "")).strip()
    if not expr:
        return True
    m = _SUPPORTED_IF.match(expr)
    if not m:
        raise SystemExit(
            f"unsupported step `if:` expression {expr!r}. Add support explicitly -- "
            "guessing would either skip a real check or run one out of context."
        )
    return outputs.get(m["step"], {}).get(m["key"]) == m["val"]


def run_job(plan: JobPlan, repo: Path, base_env: dict[str, str], echo: bool) -> list[StepResult]:
    missing = sorted({p for p in plan.requires if not _is_available(p)})
    if missing:
        return [
            StepResult(plan.name, step.get("name") or "(unnamed)", UNRUNNABLE,
                       f"job requires {', '.join(missing)}; not installed")
            for step in plan.steps
        ]

    results: list[StepResult] = []
    outputs: dict[str, dict[str, str]] = {}
    for step in plan.steps:
        label = step.get("name") or "(unnamed)"
        if not _should_run(step, outputs):
            results.append(StepResult(plan.name, label, PASS, "condition false; step not applicable"))
            continue

        env = dict(base_env)
        env.update({k: _resolve_env(k, str(v), base_env) for k, v in (step.get("env") or {}).items()})
        with tempfile.NamedTemporaryFile("w+", suffix=".out", delete=False) as fh:
            out_path = Path(fh.name)
        env["GITHUB_OUTPUT"] = str(out_path)

        # The header prints in BOTH modes; --quiet suppresses the step's own output, not
        # the fact that a step is running. A multi-minute lane that prints nothing until
        # it finishes is indistinguishable from a hung one.
        print(f":: {plan.name} / {label}", flush=True)
        proc = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
            cwd=repo, env=env,
            capture_output=not echo, text=True,
        )
        if step.get("id"):
            outputs[step["id"]] = dict(
                line.split("=", 1) for line in out_path.read_text().splitlines() if "=" in line
            )
        out_path.unlink(missing_ok=True)

        if proc.returncode == 0:
            results.append(StepResult(plan.name, label, PASS))
        elif proc.returncode == 127:
            # `command not found`: the step could not run, which is not the same fact as
            # the step having failed, and must not be reported as one.
            results.append(StepResult(plan.name, label, UNRUNNABLE, "a command in this step is not installed"))
        else:
            tail = ((proc.stderr or proc.stdout or "").strip().splitlines() or [""])[-1] if not echo else ""
            results.append(StepResult(plan.name, label, FAIL, f"exit {proc.returncode} {tail}".strip()))
    return results


def report(results: list[StepResult], allow_unrunnable: bool) -> int:
    failed = [r for r in results if r.state == FAIL]
    unrunnable = [r for r in results if r.state == UNRUNNABLE]
    passed = [r for r in results if r.state == PASS]

    print("\n" + "=" * 72)
    print(f"PASS {len(passed)}   FAIL {len(failed)}   UNRUNNABLE {len(unrunnable)}")
    print("=" * 72)
    for title, rows in (("FAILED", failed), ("COULD NOT RUN", unrunnable)):
        if rows:
            print(f"\n{title}:")
            for r in rows:
                print(f"  {r.job} / {r.name}\n      {r.detail}")
    if failed:
        return 1
    if unrunnable and not allow_unrunnable:
        print("\nUNRUNNABLE steps are not passes. Install the tool, or re-run with "
              "--allow-unrunnable to state that you accept the gap.")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    repo_default = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, default=repo_default)
    ap.add_argument("--workflow", type=Path, default=None,
                    help="workflow to derive from (default: <repo>/.github/workflows/pr-required.yml)")
    ap.add_argument("--jobs", default="", help="comma-separated subset of job names")
    ap.add_argument("--allow-unrunnable", action="store_true",
                    help="exit 0 when the only non-passes are steps whose tools are absent")
    ap.add_argument("--list", action="store_true", help="print the derived plan and exit")
    ap.add_argument("--quiet", action="store_true", help="capture step output instead of streaming it")
    args = ap.parse_args(argv)

    repo = args.repo.resolve()
    workflow_path = args.workflow or repo / ".github" / "workflows" / "pr-required.yml"
    if not workflow_path.is_file():
        raise SystemExit(f"{workflow_path}: no such workflow")

    workflow = load_workflow(workflow_path)
    only = {j.strip() for j in args.jobs.split(",") if j.strip()} or None
    plans = plan_jobs(workflow, only)

    if args.list:
        for p in plans:
            print(f"{p.name}  ({len(p.steps)} steps, requires: {', '.join(p.requires) or '-'})")
            for s in p.steps:
                print(f"    - {s.get('name') or '(unnamed)'}")
        return 0

    env = dict(os.environ)
    env.update({k: str(v) for k, v in (workflow.get("env") or {}).items()})
    # The venv's bin must win: `python`, `pytest`, `ruff` and `spectramr` in the step
    # bodies mean THIS interpreter's, not whatever the login shell resolves.
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    env.setdefault("BASE", _merge_base(repo))
    env.setdefault("HEAD", _rev(repo, "HEAD"))

    results: list[StepResult] = []
    for plan in plans:
        results.extend(run_job(plan, repo, env, echo=not args.quiet))
    return report(results, args.allow_unrunnable)


def _rev(repo: Path, ref: str) -> str:
    return subprocess.run(["git", "rev-parse", ref], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.strip()


def _merge_base(repo: Path) -> str:
    """The PR base. Falls back to HEAD~1 off a branch, so the diff-scoped steps still run."""
    for ref in ("origin/dev", "dev"):
        proc = subprocess.run(["git", "merge-base", ref, "HEAD"], cwd=repo,
                              capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    return _rev(repo, "HEAD~1")


if __name__ == "__main__":
    sys.exit(main())
