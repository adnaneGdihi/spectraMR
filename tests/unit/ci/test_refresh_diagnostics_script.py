"""Regression tests for scripts/ci/refresh_diagnostics.sh.

2026-07-02: refreshing sabine_diagnostics by hand hit a silent-drop bug —
pointing run_forensics.py's --out directly at compile_diagnostics.py's
<out>/forensics collapses the staging step, so the *__contact.png copy
into forensics/contact_sheets/ becomes a same-path no-op and
compile_diagnostics.py reports "contact sheets: 0" even though the PNGs
were generated. refresh_diagnostics.sh exists to make that mistake
structurally impossible (mktemp -d staging, always distinct from --out).
These tests pin the properties that prevent a regression back to the bug,
following the tests/unit/ci/test_smoke_exclude_vf.py convention (read the
script text / bash -n it rather than executing the full pipeline).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.utils.repo_scripts import require_repo_file

REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_REL = "scripts/ci/refresh_diagnostics.sh"
MAKEFILE = REPO_ROOT / "Makefile"


def _script() -> Path:
    """The wrapper, or an explained skip where scripts/ci/ ships only in part.

    Resolved per-test rather than as a module constant: the two Makefile tests
    below have no scripts/ subject and must keep running in the export. This file
    is why the second census pass exists -- it sat in the export failing 12 of 14
    without appearing in the first census at all, because it names its subject
    through a REPO_ROOT alias rather than an inline path chain.
    """
    return require_repo_file(_SCRIPT_REL)


def test_script_exists_and_is_executable() -> None:
    script = _script()
    # Presence is _script()'s to answer -- it distinguishes "denied by the
    # allowlist" from "deleted", which a bare .exists() here cannot.
    assert script.stat().st_mode & 0o111, f"{script} is not executable (chmod +x)"


def test_script_is_valid_bash() -> None:
    result = subprocess.run(["bash", "-n", str(_script())], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_forensics_staged_in_mktemp_not_compile_out() -> None:
    """The gotcha guard: run_forensics.py --out must be `mktemp -d`, never the
    compile_diagnostics.py --out/forensics path directly — same-path source
    and destination silently drops every contact sheet during the copy."""
    text = _script().read_text()
    assert "mktemp -d" in text, (
        "refresh_diagnostics.sh must stage run_forensics.py output under "
        "mktemp -d, distinct from compile_diagnostics.py's --out — "
        "pointing run_forensics.py straight at <out>/forensics makes the "
        "contact-sheet copy a same-path no-op (see docs/validation_image_audit.rst"
        "#diagnostics_make_targets)."
    )
    assert '--out "${stage}"' in text
    assert '--forensics-src "${stage}"' in text


def test_both_known_trees_are_refreshed() -> None:
    text = _script().read_text()
    assert "refresh_one tests_experiments tests_experiments/diagnostics" in text
    assert "refresh_one sabine_tests_experiments sabine_diagnostics" in text


def test_missing_tree_is_skipped_not_fatal() -> None:
    """sabine_tests_experiments only exists once a cluster tree has been
    downloaded — the wrapper must not abort when it (or, in principle,
    tests_experiments) is absent."""
    text = _script().read_text()
    assert '-d "${root_abs}"' in text
    assert "set -e" not in text, "must not hard-abort on a missing optional tree"


def test_an_absolute_root_is_not_prefixed_with_the_repo_root() -> None:
    """A downloaded tree lives wherever rsync put it. Prefixing REPO_ROOT at
    every use turned ``--out /tmp/x`` into ``<repo>/tmp/x`` -- the bundle was
    written, just not where it was asked for."""
    text = _script().read_text()
    assert "abspath()" in text
    assert '/*) printf \'%s\' "$1" ;;' in text
    body = text.split("refresh_one() {", 1)[1].split("\n}", 1)[0]
    assert '"${REPO_ROOT}/${root}' not in body
    assert '"${REPO_ROOT}/${out}' not in body


def test_no_forensics_flag_skips_the_image_pass() -> None:
    text = _script().read_text()
    assert "--no-forensics" in text
    assert "RUN_FORENSICS=0" in text


def test_report_cases_png_render_is_wired_and_unconditional() -> None:
    """The report_cases/*.npz -> PNG pass must run inside refresh_one (so it fires
    for BOTH `diagnostics` and `diagnostics-fast`), and be non-fatal — it is cheap
    (numpy+matplotlib) and independent of the heavy forensics pass, but a raw .npz
    hides a collapse / magnitude blow-up until someone loads numpy by hand."""
    text = _script().read_text()
    assert "scripts/diagnostics/render_report_cases.py" in text
    body = text.split("refresh_one() {", 1)[1].split("\n}", 1)[0]
    assert "render_report_cases.py" in body, "renderer must run inside refresh_one"
    render_line = next(ln for ln in text.splitlines() if "render_report_cases.py" in ln)
    assert "RUN_FORENSICS" not in render_line, "render must not be gated on forensics"


def test_forensics_script_path_is_overridable() -> None:
    """run_forensics.py lives in the spectramr-image-forensics *skill*, not this
    repo, so its path is host-local and must be overridable via env var
    rather than hardcoded with no escape hatch."""
    text = _script().read_text()
    assert "SPECTRAMR_FORENSICS_SCRIPT" in text


def test_makefile_wires_diagnostics_targets() -> None:
    text = MAKEFILE.read_text()
    assert "diagnostics:\n\t./scripts/ci/refresh_diagnostics.sh $(DIAG_ARGS)\n" in text
    fast = "diagnostics-fast:\n\t./scripts/ci/refresh_diagnostics.sh "
    assert fast in text
    fast_recipe = text.split(fast, 1)[1].splitlines()[0]
    for flag in ("--no-forensics", "--no-probe", "--no-mosaic", "$(DIAG_ARGS)"):
        assert flag in fast_recipe, f"diagnostics-fast must pass {flag}"
    assert "diagnostics" in text.split(".PHONY:", 1)[1].splitlines()[0]


def test_makefile_exposes_the_tree_and_cohort_selectors() -> None:
    """ROOT / OUT / COHORT are the whole point of the targets being parameterised;
    a selector that is defined but not forwarded is an unread knob (pitfall #15)."""
    text = MAKEFILE.read_text()
    diag = text.split("DIAG_ARGS :=", 1)[1].splitlines()[0]
    for var, flag in (("ROOT", "--root"), ("OUT", "--out"), ("COHORT", "--cohort")):
        assert f"{var} ?=" in text, f"{var} must be an overridable Make variable"
        assert f"$(if $({var}),{flag} $({var}))" in diag, f"{var} must reach {flag}"


@pytest.mark.parametrize(
    "pass_script",
    [
        "scripts/diagnostics/render_report_cases.py",
        "scripts/diagnostics/debug_snapshot_audit.py",
        "scripts/generate_validation_mosaic.py",
        "scratch/batch_scientific_audit.py",
        "scratch/batch_probe.py",
        "scratch/compile_diagnostics.py",
    ],
)
def test_every_diagnostic_pass_runs_inside_refresh_one(pass_script: str) -> None:
    """`make diagnostics` is the one command, so each pass has to be IN it.

    Before this, the wrapper ran three of the six: the Tier-0/1 audit and the
    Tier-2 probe were never called, so a refresh republished the previous run's
    scientific_audit.json / probe_results.json under today's date stamp, and
    nothing said so."""
    body = _script().read_text().split("refresh_one() {", 1)[1].split("\n}", 1)[0]
    assert pass_script in body, f"{pass_script} must run inside refresh_one"


def test_root_and_out_select_one_tree() -> None:
    text = _script().read_text()
    for flag in ("--root)", "--root=*)", "--out)", "--out=*)"):
        assert flag in text, f"wrapper must accept {flag.rstrip(')*=')}"
    # That the no-argument path still refreshes both known trees is
    # test_both_known_trees_are_refreshed's assertion, not a second copy here.
    assert 'refresh_one "${ROOT}" "${OUT:-${ROOT}/diagnostics}"' in text


def test_out_without_root_is_refused_rather_than_applied_to_both_trees() -> None:
    """--out names ONE bundle; silently applying it to both trees would make the
    second overwrite the first."""
    text = _script().read_text()
    assert '-n "${OUT}" && -z "${ROOT}"' in text


def test_cohort_reaches_every_pass_that_can_filter() -> None:
    text = _script().read_text()
    assert "--cohort)" in text and "--cohort=*)" in text
    assert 'cohort_args=(--cohort "${COHORTS[@]}")' in text
    body = text.split("refresh_one() {", 1)[1].split("\n}", 1)[0]
    # One per pass that takes --cohort: snapshots, audit, probe, compile.
    assert body.count('"${cohort_args[@]}"') >= 4


def test_cohort_names_are_resolved_by_the_one_owner_not_a_bash_alias_table() -> None:
    """A second alias table in bash would drift from the python passes' -- the
    same name would then select different arms in different passes."""
    text = _script().read_text()
    assert "scripts/cohort_membership.py" in text
    for alias in ("mrixfields2026", "hilbert_mamba", "ldm_two_stage_ulf_to_hf"):
        assert alias not in text, f"{alias} must not be spelled out in the wrapper"


def test_slow_passes_are_individually_skippable() -> None:
    """diagnostics-fast is composed of these, so each must exist on its own."""
    text = _script().read_text()
    for flag, var in (("--no-forensics", "RUN_FORENSICS=0"),
                      ("--no-snapshots", "RUN_SNAPSHOTS=0"),
                      ("--no-mosaic", "RUN_MOSAIC=0"),
                      ("--no-audit", "RUN_AUDIT=0"),
                      ("--no-probe", "RUN_PROBE=0")):
        assert flag in text and var in text, f"{flag} must set {var}"


def test_a_cohort_matching_no_arm_stops_the_arm_level_passes() -> None:
    """The scope-widening shape, planted as an assertion because it is invisible
    at the call site: ``generate_validation_mosaic.py`` with no ``--filter``
    means EVERY arm, so resolving a cohort to an empty arm list and then omitting
    the flag renders the whole tree under a cohort-scoped invocation. Observed
    once here -- `--cohort mamba` against a tree holding only kspace_filling arms
    started mosaicking all 68."""
    text = _script().read_text()
    body = text.split("refresh_one() {", 1)[1].split("\n}", 1)[0]
    assert 'arms_resolved' in body, "the wrapper must distinguish 'no selection' from 'empty selection'"
    assert body.count('"${arms_resolved}" == "0" && -z "${arms}"') >= 2, (
        "both arm-level passes (mosaic, forensics) must refuse an empty resolution"
    )
    assert "cohort matched no arm" in body


def test_cohort_filters_the_image_passes_too_not_only_the_config_passes() -> None:
    """--cohort that filtered five passes and silently not the sixth would put a
    whole-tree contact sheet next to a one-cohort summary."""
    body = _script().read_text().split("refresh_one() {", 1)[1].split("\n}", 1)[0]
    assert "filter_args=(--filter ${arms})" in body, "mosaic must be filtered"
    assert "arm_args=(--arms" in body, "forensics must be filtered"


def test_the_arm_list_is_resolved_once_per_tree() -> None:
    """Two resolutions are two chances to disagree; the window resolver above is
    pinned the same way."""
    body = _script().read_text().split("refresh_one() {", 1)[1].split("\n}", 1)[0]
    assert body.count("cohort_membership.py") == 1


def test_a_skipped_tier_pass_is_reported_not_silent() -> None:
    """The republished-stale-JSON trap: skipping the audit/probe leaves the
    previous run's verdicts in the bundle, so the skip has to be visible."""
    text = _script().read_text()
    assert "scientific audit: skipped" in text
    assert "forward probe: skipped" in text


def test_makefile_does_not_invoke_bash_by_bare_name() -> None:
    """Regression pin: `bash <path>` fails inside a recipe because the
    pre-existing .env-loading block (lines 8-11) mis-parses .env's bash
    `export VAR=$(pwd)` syntax as Make syntax, corrupting the PATH Make
    exports to recipe shells. Invoking the script by its own executable path
    sidesteps this — the kernel resolves the #!/bin/bash shebang directly,
    with no PATH lookup by the calling shell."""
    diag_section = MAKEFILE.read_text().split("diagnostics:", 1)[1]
    assert "\tbash " not in diag_section.split("install-topology", 1)[0]


def test_run_since_is_wired_through_to_the_compiler() -> None:
    """The run-window knob must reach ``compile_diagnostics.py``, not be advertised
    and dropped (pitfall #15): the wrapper is how ``make diagnostics`` reaches it."""
    text = _script().read_text()
    assert "--run-since)" in text, "wrapper must accept --run-since <value>"
    assert "--run-since=*)" in text, "wrapper must accept --run-since=<value>"
    assert "window_args=(--run-since" in text
    assert '"${window_args[@]}"' in text, "the parsed window must be passed on"


def test_arg_parsing_shifts_so_a_two_token_flag_cannot_be_misread() -> None:
    """``--run-since 2026-07-21`` is two tokens; a ``for arg in "$@"`` loop would
    read the value as an unknown arg and exit 2."""
    text = _script().read_text()
    assert "while [[ $# -gt 0 ]]" in text
    assert "shift 2" in text


def test_image_passes_get_the_same_window_as_the_compile() -> None:
    """Forensics renders the contact sheets BEFORE the compile runs, so it needs
    the window too. Without it the sheets are picked by ``sorted(glob)[-1]`` over
    names that begin with the epoch NUMBER, which returns a previous run's image.
    """
    text = _script().read_text()
    assert "--print-run-window" in text, "wrapper must ask the compiler for the window"
    assert "since_args=(--since" in text, "the window must reach run_forensics.py"
    assert '"${since_args[@]}"' in text


def test_the_window_is_resolved_once_not_reimplemented_in_bash() -> None:
    """One resolver (compile_diagnostics.resolve_run_window). A bash re-derivation
    would be free to drift from the compiler's — the pitfall #13b shape."""
    text = _script().read_text()
    assert text.count("--print-run-window") == 1
    # no second implementation: the wrapper must not scan dispatch/ dirs itself
    assert "dispatch/" not in text.split("refresh_one()")[1]
