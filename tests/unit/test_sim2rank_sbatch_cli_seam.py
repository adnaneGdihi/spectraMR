"""The sbatch -> argparse seam for the sim2rank launchers.

Two real defects motivated this gate, and both are invisible to every
unit test in the suite because neither side is individually wrong:

* ``--severity-yardstick`` landed on ``sim2rank.py`` *and*
  ``run_all_generations.py`` (PR #462) and was passed by no launcher, so the
  iso-distortion calibration could not be reached from the cluster at all --
  pitfall #15 relocated from the config layer to the launcher layer;
* the brain launcher passed ``--max-contrasts 3`` to ``sim2rank.py`` and nothing
  to ``run_all_generations.py``, whose parser defaults it to 1, so the two
  ranking surfaces were built on different cohorts and the report then rendered
  "inter-ranker agreement" by joining them.

So this module asserts three things about the seam rather than about either
side of it: every flag a launcher emits exists in the target parser; the
cohort- and severity-defining flags agree across the two ranker sweeps in one
job; and the sweep knobs each launcher advertises are validated before the
expensive work starts.

Flags are read statically (``ast``) rather than by shelling out to ``--help``:
the targets import torch, and a gate that costs 30 s of CUDA-capable import per
run is a gate people disable.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests.utils.repo_scripts import require_repo_file

REPO = Path(__file__).resolve().parents[2]
SIM2RANK = REPO / "scripts" / "sim2rank"

#: Launchers under test.
SBATCHES = (
    "run_fastmri_brain_pipeline.sbatch",
    "run_full_pipeline.sbatch",
    "submit_sim2rank.sbatch",
)


def _is_boolean_optional(node: ast.Call) -> bool:
    """True for ``action=argparse.BooleanOptionalAction``.

    That action synthesises a ``--no-<name>`` counterpart at parse time, so the
    negation is a real accepted flag even though no literal declares it. Missing
    this reads ``--no-unified-degradations`` as unknown when it is not.
    """
    for kw in node.keywords:
        if kw.arg != "action":
            continue
        value = kw.value
        attr = value.attr if isinstance(value, ast.Attribute) else None
        ident = value.id if isinstance(value, ast.Name) else None
        if "BooleanOptionalAction" in {attr, ident}:
            return True
    return False


def _declared_flags(script: Path) -> set[str]:
    """Every ``--flag`` the script's parser accepts.

    Literal option strings, plus the ``--no-x`` counterparts argparse
    synthesises for ``BooleanOptionalAction``. Explicit ``--no-x`` pairs (the
    ``store_false`` + ``dest=`` idiom this codebase also uses) come through as
    ordinary literals.
    """
    tree = ast.parse(script.read_text())
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name != "add_argument":
            continue
        literals = [
            a.value
            for a in node.args
            if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("--")
        ]
        flags.update(literals)
        if _is_boolean_optional(node):
            flags.update(f"--no-{lit[2:]}" for lit in literals)
    return flags


def _flag_var_assignments(text: str) -> dict[str, str]:
    """``VAR="--flag ..."`` assignments, so ``${VAR}`` in a call can be resolved.

    The launchers build optional flags into shell variables
    (``MG_FLAG="--multi-gen-json ${MG_JSON}"``). Expanding them is what makes
    this check see the flags a run actually emits rather than only the
    hard-coded ones.
    """
    out: dict[str, str] = {}
    for m in re.finditer(r'^\s*([A-Z_][A-Z0-9_]*)="(--[^"]*)"', text, re.MULTILINE):
        out[m.group(1)] = m.group(2)
    for m in re.finditer(
        r'^\s*\[\[[^\]]*\]\]\s*&&\s*([A-Z_][A-Z0-9_]*)="(--[^"]*)"',
        text,
        re.MULTILINE,
    ):
        out[m.group(1)] = m.group(2)
    return out


def _invocations(text: str) -> list[tuple[str, str]]:
    """``(target_script, flag_block)`` for every ``python scripts/...py`` call."""
    calls: list[tuple[str, str]] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.search(r"python\s+(?:\S*/)?(\w+\.py)\b", line)
        if not m:
            continue
        block = [line]
        j = i
        while block[-1].rstrip().endswith("\\") and j + 1 < len(lines):
            j += 1
            block.append(lines[j])
        calls.append((m.group(1), "\n".join(block)))
    return calls


def _emitted_flags(block: str, variables: dict[str, str]) -> set[str]:
    """Flags in one invocation, with ``${VAR}`` flag-holders expanded."""
    expanded = block
    for var, value in variables.items():
        expanded = expanded.replace(f"${{{var}}}", value)
    # Drop any remaining unresolved expansions so their contents are not mined
    # for false flags.
    expanded = re.sub(r"\$\{[^}]*\}", " ", expanded)
    return set(re.findall(r"(?<![\w-])(--[a-zA-Z][\w-]*)", expanded))


def _read(name: str) -> str:
    """A launcher's text, or an explained skip where sim2rank does not ship.

    Not ``SIM2RANK / name`` directly: the whole subsystem is denied by the public
    allowlist, so in the export these reads raised ``FileNotFoundError``. The
    existence check inside :func:`_sbatch_targets` stays a plain ``.exists()`` --
    there, absence means "this launcher calls a helper outside sim2rank/", which
    is a real condition in BOTH trees and not a publication boundary.
    """
    return require_repo_file(f"scripts/sim2rank/{name}").read_text()


def _sbatch_targets(name: str):
    text = _read(name)
    variables = _flag_var_assignments(text)
    for target, block in _invocations(text):
        script = SIM2RANK / target
        if not script.exists():
            continue  # a helper outside scripts/sim2rank/ (data prep, export)
        declared = _declared_flags(script)
        if not declared:
            continue  # not an argparse entry point
        yield target, block, declared, variables


@pytest.mark.parametrize("name", SBATCHES)
def test_every_emitted_flag_exists_in_the_target_parser(name: str) -> None:
    """A renamed or dropped CLI flag must fail here, not silently at 3 a.m."""
    problems: list[str] = []
    for target, block, declared, variables in _sbatch_targets(name):
        for flag in sorted(_emitted_flags(block, variables) - declared):
            problems.append(f"{name} -> {target}: unknown flag {flag}")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("name", SBATCHES)
def test_at_least_one_invocation_was_actually_checked(name: str) -> None:
    """Guard the guard: a parser change that breaks discovery must not pass."""
    assert list(_sbatch_targets(name)), f"{name}: no sim2rank invocation discovered"


def _brain_invocation(target: str) -> str:
    text = _read("run_fastmri_brain_pipeline.sbatch")
    blocks = [b for t, b in _invocations(text) if t == target]
    assert blocks, f"no {target} invocation in the brain launcher"
    return "\n".join(blocks)


@pytest.mark.parametrize("flag", ["--severity-yardstick", "--max-contrasts"])
def test_both_ranker_sweeps_receive_the_cohort_defining_flags(flag: str) -> None:
    """The two ranking surfaces must be built on one cohort and one severity grid.

    ``per_axis_figures.py --multi-gen-json`` and ``consensus_figures.py`` join
    ``sim2rank_results.json`` with ``run_all_generations_results.json`` into
    agreement and consensus panels. Those panels only mean "these rankers
    disagree" if both rankers scored the same data at the same severities.
    """
    for target in ("sim2rank.py", "run_all_generations.py"):
        assert flag in _brain_invocation(target), (
            f"{target} is not passed {flag} by the brain launcher — its parser "
            f"default will silently diverge from the other ranking surface"
        )


def test_leaderboard_sweep_declares_its_axis_bank() -> None:
    """#414: a run must say which degradation bank produced its leaderboard."""
    assert "--axis-bank" in _brain_invocation("sim2rank.py")
    text = _read("run_full_pipeline.sbatch")
    assert "--axis-bank" in text


@pytest.mark.parametrize(
    "name,knobs",
    [
        ("run_fastmri_brain_pipeline.sbatch", ("AXIS_BANK", "SEVERITY_YARDSTICK")),
        ("run_full_pipeline.sbatch", ("AXIS_BANK", "SEVERITY_YARDSTICK")),
    ],
)
def test_sweep_knobs_are_validated_against_their_advertised_set(
    name: str, knobs: tuple[str, ...]
) -> None:
    """Pitfall #15: an advertised knob is read, validated, and stamped.

    Both launchers reach a GPU allocation and a multi-hour cohort load before
    argparse ever sees these values, so an unknown one has to fail in the
    preflight rather than after the expensive part.
    """
    text = _read(name)
    for knob in knobs:
        assert f'{knob}="${{' in text, f"{name}: {knob} is not an overridable knob"
        assert re.search(rf'case "\$\{{{knob}\}}" in', text), (
            f"{name}: {knob} is accepted but never validated against its choices"
        )


def test_the_seam_check_would_catch_an_unknown_flag() -> None:
    """Meta-test: the extraction actually resolves flags, including via vars."""
    block = "python scripts/sim2rank/sim2rank.py \\\n  --timesteps 20 ${MG_FLAG}"
    flags = _emitted_flags(block, {"MG_FLAG": "--multi-gen-json /tmp/x.json"})
    assert flags == {"--timesteps", "--multi-gen-json"}
