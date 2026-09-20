"""The DDP launcher must not guess its own world size.

``GPUS_PER_NODE`` feeds ``torchrun --nproc_per_node`` directly, so it *is* the
per-node world size. The derivation used to end in ``|| echo 1``:

    GPUS_PER_NODE="$(echo "${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}" \
        | grep -oE '[0-9]+$' || echo 1)"

Any input that produced no trailing digits -- both variables unset, or a
type-only value like ``ada`` -- resolved to ``1``. A four-GPU allocation then
forked a single rank: training ran to completion, the checkpoint was valid, the
metrics were real, and nothing in the logs said three quarters of the requested
hardware was never touched. The run did not fail; it silently became a smaller,
slower, differently-batched experiment than the one that was submitted. That is
non-negotiable #9 in its most expensive form, because the evidence of it is
absent rather than wrong.

Compounding it, the comment above the block advertised a "count visible devices"
fallback that did not exist in the code -- a facade (pitfall #16) that made the
``1`` look like a last resort after a real search.

Two halves here, and the second is the one that matters:

* **Shape** -- the ``|| echo 1`` idiom must not return, and the block must stay
  extractable.
* **Behaviour** -- the committed block is EXTRACTED FROM THE SCRIPT and executed
  in a real ``bash`` against controlled environments. Asserting on the script's
  text alone would pass on any transcription-equivalent rewrite that reinstates
  the guess, and shell precedence around ``&&``/``||``/``$( )`` is exactly where
  such a rewrite goes wrong.

The same guess lived in two sibling submitters, spelled differently -- both
``NPROC="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}"``, both feeding the same
``torchrun --nproc_per_node``. They are covered here by shape guards over
``RANK_DERIVING_SCRIPTS`` rather than a second execution harness: the construct
is identical and the refusal is proven behaviourally once, against the
derivation itself.

The derivation now lives in ``scripts/common/gpu_count.sh`` and is *sourced* by
the two launchers that fork ranks -- ``train_distributed.sbatch`` and, since the
array dispatcher learned to launch process-group arms,
``dispatch_experiments.sbatch``. Copying it into the second launcher would have
been a second owner of a rule that has already cost one silent single-rank run
(non-negotiable 17), so the behaviour tests below execute the HELPER and the
shape guards accept either spelling: a launcher may derive the count inline, or
delegate to the helper, and :func:`_rank_count_is_guarded` requires the fatal to
be reachable in whichever case it chose.

Scope note: this covers the derivation only. Whether Slurm's count matches the
GPUs the job can actually open is the cluster's business, not the script's.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from tests.utils.repo_scripts import require_repo_file

REPO = Path(__file__).resolve().parents[3]
_SBATCH_REL = "scripts/training/train_distributed.sbatch"
_DISPATCH_REL = "scripts/training/dispatch_experiments.sbatch"

#: Every launcher that derives a rank count from the environment and hands it to
#: ``torchrun --nproc_per_node``. All three carried the same ``:-1`` guess; the
#: two ablation submitters get the shape guards below rather than a second
#: extraction harness, because they use the identical construct and the raise
#: behaviour is proven once, behaviourally, against the main launcher.
#: Repo-relative, not ``Path``: each is resolved through ``require_repo_file`` at
#: use, so the two ablation submitters -- which the public allowlist denies --
#: skip with a reason in the export instead of failing, while a launcher that
#: MOVED still fails loudly in every other tree.
RANK_DERIVING_SCRIPTS = [
    _SBATCH_REL,
    _DISPATCH_REL,
    "scripts/training/submit_exp11_fpk_ablation.sbatch",
    "scripts/training/submit_exp11_ema_warmup_ablation.sbatch",
]

#: The shared derivation and its entry point. A launcher that sources this and
#: calls the function inherits its refusal; one that spells the derivation
#: itself must carry its own.
_HELPER_REL = "scripts/common/gpu_count.sh"
_HELPER_FN = "derive_gpus_per_node"

_BEGIN = "# --- gpu-count-derivation"
_END = "# --- end gpu-count-derivation"


def _executable_lines(text: str) -> str:
    """Comment lines dropped.

    Each fix quotes the idiom it removed in order to explain what it cost, so a
    substring search over the raw file flags the explanation as the defect.
    """
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


#: A PATH with the ordinary utilities the block calls (``grep``), so the harness
#: below can run under ``env -i`` without inheriting the developer's shell.
_BASE_PATH = "/usr/local/bin:/usr/bin:/bin"


@pytest.fixture(scope="module")
def script() -> str:
    return require_repo_file(_SBATCH_REL).read_text()


@pytest.fixture(scope="module")
def block() -> str:
    """The committed derivation, sliced out of the HELPER between its sentinels.

    Read from ``scripts/common/gpu_count.sh`` rather than from either launcher:
    that file is the one owner, so executing it is what proves the behaviour for
    every launcher that sources it. Extraction is asserted non-empty in its own
    test below -- a silently empty slice would make every behaviour test pass
    against nothing, which is the same shape of bug this file exists to catch.
    """
    lines = require_repo_file(_HELPER_REL).read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(_BEGIN))
    end = next(i for i, ln in enumerate(lines) if ln.startswith(_END))
    return "\n".join(lines[start + 1 : end])


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", RANK_DERIVING_SCRIPTS, ids=lambda p: Path(p).stem)
def test_script_is_valid_bash(path: str) -> None:
    """A syntax error here is invisible until the job is queued on the cluster."""
    result = subprocess.run(
        ["bash", "-n", str(require_repo_file(path))], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("path", RANK_DERIVING_SCRIPTS, ids=lambda p: Path(p).stem)
def test_no_launcher_defaults_its_rank_count_to_one(path: str) -> None:
    """``${SLURM_GPUS*:-1}`` is the sibling spelling of the same guess.

    The main launcher's version ended in ``|| echo 1``; the two ablation
    submitters wrote ``${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}``. Different
    syntax, identical consequence -- one rank on a multi-GPU allocation, with a
    successful-looking run to show for it.
    """
    code = _executable_lines(require_repo_file(path).read_text())
    offenders = re.findall(r"\$\{SLURM_GPUS[A-Z_]*:-1\}", code)
    assert not offenders, (
        f"{path.name} defaults its rank count to 1 ({offenders}); a job whose "
        "GPU variables are unset would silently run one rank"
    )


def _rank_count_is_guarded(text: str) -> str | None:
    """``None`` when *text* refuses an underivable rank count; else why not.

    Two spellings are legitimate, and both are judged here rather than in two
    tests so that a launcher switching from one to the other cannot fall between
    them:

    * **delegated** — sources ``scripts/common/gpu_count.sh`` and calls
      ``derive_gpus_per_node``, inheriting the refusal the behaviour tests below
      execute directly. Sourcing without calling is the interesting failure:
      ``GPUS_PER_NODE`` is then simply unset, which is the guess this whole file
      exists to forbid, wearing a ``source`` line as camouflage.
    * **inline** — derives the count itself, in which case an ``exit 1`` must sit
      BETWEEN the derivation and the ``torchrun`` that consumes it. A bare
      ``"exit 1" in script`` looked equivalent and was vacuous: both ablation
      submitters already carried an unrelated ``exit 1`` (a missing-CONFIG
      guard) before the fix, so that assertion passed on the buggy versions.

    Returns a reason rather than asserting, so the planted violations below can
    check that each shape is rejected for the right reason.
    """
    lines = _executable_lines(text).splitlines()

    called = next((i for i, ln in enumerate(lines) if _HELPER_FN in ln and "(" not in ln), None)
    if called is not None:
        # Both launchers source through a variable (`source "${GPU_COUNT_HELPER}"`),
        # so the helper PATH and the `source` that consumes it land on different
        # lines. Requiring the literal path on the source line itself would reject
        # both correct launchers -- and, worse, would push the next one toward
        # inlining the path just to satisfy a test.
        named = next((i for i, ln in enumerate(lines) if "gpu_count.sh" in ln), None)
        if named is None:
            return f"calls {_HELPER_FN} without sourcing {_HELPER_REL}"
        sourced = next(
            (i for i, ln in enumerate(lines[named:], named) if re.match(r"\s*(source|\.)\s", ln)),
            None,
        )
        if sourced is None:
            return f"calls {_HELPER_FN} and names {_HELPER_REL} but never sources it"
        if sourced > called:
            return f"calls {_HELPER_FN} at line ~{called} before sourcing it at ~{sourced}"
        return None

    derived = next(
        (i for i, ln in enumerate(lines) if re.match(r"\s*(NPROC|GPUS_PER_NODE)=", ln)),
        None,
    )
    if derived is None:
        return "no rank-count derivation and no call to the shared helper"
    used = next((i for i, ln in enumerate(lines[derived:], derived) if "torchrun" in ln), None)
    if used is None:
        return "derivation never reaches a torchrun"
    if not [ln for ln in lines[derived:used] if "exit 1" in ln]:
        return (
            f"derives its rank count at line ~{derived} and hands it to torchrun "
            f"at ~{used} with no fatal check in between"
        )
    return None


@pytest.mark.parametrize("path", RANK_DERIVING_SCRIPTS, ids=lambda p: Path(p).stem)
def test_every_launcher_refuses_rather_than_guesses(path: str) -> None:
    """Removing the ``:-1`` is only half a fix if what replaced it carries on."""
    reason = _rank_count_is_guarded(require_repo_file(path).read_text())
    assert reason is None, f"{Path(path).name}: {reason}"


@pytest.mark.parametrize(
    ("planted", "expected"),
    [
        pytest.param(
            "source scripts/common/gpu_count.sh\ntorchrun --nproc_per_node=1\n",
            "no rank-count derivation",
            id="sources-but-never-calls",
        ),
        pytest.param(
            f"{_HELPER_FN}\ntorchrun --nproc_per_node=1\n",
            "without sourcing",
            id="calls-but-never-sources",
        ),
        pytest.param(
            f'H="scripts/common/gpu_count.sh"\n{_HELPER_FN}\ntorchrun --nproc_per_node=1\n',
            "never sources it",
            id="names-the-helper-but-never-sources-it",
        ),
        pytest.param(
            f'{_HELPER_FN}\nH="scripts/common/gpu_count.sh"\nsource "$H"\n',
            "before sourcing it",
            id="calls-before-sourcing",
        ),
        pytest.param(
            "source scripts/common/gpu_count.sh\n",
            "no rank-count derivation",
            id="sources-only-no-consumer",
        ),
        pytest.param(
            'GPUS_PER_NODE="${SLURM_GPUS_ON_NODE}"\ntorchrun --nproc_per_node=1\n',
            "no fatal check in between",
            id="inline-derivation-with-no-fatal",
        ),
        pytest.param(
            'GPUS_PER_NODE=2\n# exit 1 lives only in a comment\ntorchrun --nproc=1\n',
            "no fatal check in between",
            id="fatal-only-in-a-comment",
        ),
    ],
)
def test_the_guard_rejects_each_unguarded_shape(planted: str, expected: str) -> None:
    """A gate is only a gate for the violation shape it has been watched to fail
    on (non-negotiable 15).

    Each case is a launcher that would reach ``--nproc_per_node`` with a count
    nothing vouched for. ``sources-but-never-calls`` is the one the extraction
    itself introduced: before the derivation moved into a function, the mere
    presence of the code WAS the call.
    """
    reason = _rank_count_is_guarded(planted)
    assert reason is not None, f"guard passed a launcher that guesses: {planted!r}"
    assert expected in reason, reason


def test_the_guard_accepts_both_legitimate_shapes() -> None:
    """Anti-vacuity for the rejections above: the guard must not simply fail
    everything it is shown."""
    delegated = f"source scripts/common/gpu_count.sh\n{_HELPER_FN}\ntorchrun --nproc_per_node=1\n"
    inline = (
        'GPUS_PER_NODE="${SLURM_GPUS_ON_NODE}"\n'
        'if [[ -z "${GPUS_PER_NODE}" ]]; then exit 1; fi\n'
        "torchrun --nproc_per_node=1\n"
    )
    assert _rank_count_is_guarded(delegated) is None
    assert _rank_count_is_guarded(inline) is None


def test_the_derivation_block_is_extractable(block: str) -> None:
    """Anti-vacuity for every behaviour test below."""
    assert block.strip(), "sentinels present but the slice between them is empty"
    assert "GPUS_PER_NODE" in block


def test_the_guess_idiom_is_gone(script: str) -> None:
    """The literal defect. Kept as a text assertion because it is a specific
    idiom a well-meaning simplification would reintroduce.

    Comments are stripped first: the fix's own commentary quotes the old idiom
    verbatim to explain what it cost, and a naive substring search over the raw
    file flags that quotation. Searching executable lines only keeps the
    regression guard honest without forbidding the script from documenting its
    own history.
    """
    assert "|| echo 1" not in _executable_lines(script), (
        "the GPU count falls back to a guess of 1; a multi-GPU allocation would "
        "silently run one rank"
    )


def test_the_failure_path_exits_nonzero(block: str) -> None:
    """A warn-and-continue here is the original bug wearing a log line."""
    assert "exit 1" in block


def test_the_advertised_fallback_is_implemented(block: str) -> None:
    """The old comment promised a visible-device count and the code had none.

    Documentation and behaviour diverging is what let the guess read as a last
    resort; this pins them together.
    """
    assert "nvidia-smi" in block


# ---------------------------------------------------------------------------
# Behaviour: the committed block, executed
# ---------------------------------------------------------------------------


#: Resolved once, from the real environment. The sandboxed PATH handed to the
#: child does not contain ``bash``, and ``subprocess`` resolves the executable
#: through the CHILD's PATH -- so naming it bare fails with FileNotFoundError
#: before the block ever runs.
_BASH = shutil.which("bash") or "/bin/bash"


def _run(block: str, env: dict[str, str], path: str = _BASE_PATH):
    """Execute the extracted block under the launchers' own shell options.

    The block now *defines* ``derive_gpus_per_node`` rather than running inline,
    so the harness calls it -- exactly as both launchers do after sourcing the
    helper. Without the call every assertion below would run against a shell
    that defined a function and did nothing.
    """
    body = (
        f"set -euo pipefail\n{block}\n{_HELPER_FN}\n"
        'echo "RESULT=${GPUS_PER_NODE}"'
    )
    return subprocess.run(
        [_BASH, "-c", body],
        capture_output=True,
        text=True,
        env={"PATH": path, **env},
    )


def _resolved(block: str, env: dict[str, str], path: str = _BASE_PATH) -> str:
    proc = _run(block, env, path)
    assert proc.returncode == 0, f"block failed: {proc.stderr}"
    match = re.search(r"^RESULT=(.*)$", proc.stdout, re.M)
    assert match, proc.stdout
    return match.group(1)


def _sandbox_path(tmp_path: Path) -> Path:
    """A PATH directory holding ONLY what the block legitimately calls.

    It must REPLACE the system PATH rather than prepend to it. Prepending an
    empty directory does not hide ``/usr/bin/nvidia-smi``, so on any developer
    machine with a GPU the "no source at all" case quietly resolved to that
    machine's real device count -- 1 on a single-GPU laptop, which is exactly
    the value under test. The assertion would have passed for the wrong reason.
    """
    sandbox = tmp_path / "path"
    sandbox.mkdir(exist_ok=True)
    for tool in ("grep",):
        found = next((p for d in _BASE_PATH.split(":") if (p := Path(d) / tool).exists()), None)
        assert found, f"{tool} not found on {_BASE_PATH}; the harness cannot run"
        target = sandbox / tool
        if not target.exists():
            target.symlink_to(found)
    return sandbox


@pytest.fixture()
def no_nvidia_smi(tmp_path: Path) -> str:
    """A PATH with the utilities but genuinely no ``nvidia-smi``."""
    return str(_sandbox_path(tmp_path))


def _stub_nvidia_smi(tmp_path: Path, gpus: int) -> str:
    """A PATH whose ``nvidia-smi -L`` lists exactly ``gpus`` devices."""
    sandbox = _sandbox_path(tmp_path)
    stub = sandbox / "nvidia-smi"
    # ``printf`` rather than a ``cat`` heredoc: the sandbox PATH holds only the
    # tools the block itself needs, and ``cat`` is not one of them. A builtin
    # keeps the stub working wherever the sandbox is pointed.
    args = " ".join(f"'GPU {i}: Stub (UUID: GPU-{i})'" for i in range(gpus))
    body = f"printf '%s\\n' {args}" if gpus else ":"
    stub.write_text(f"#!/bin/bash\n{body}\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return str(sandbox)


def test_slurm_gpus_on_node_is_the_first_source(block: str) -> None:
    assert _resolved(block, {"SLURM_GPUS_ON_NODE": "4"}) == "4"


def test_a_type_labelled_count_resolves_to_its_digits(block: str) -> None:
    """``--gpus=ada:2`` is what reaches SLURM_GPUS_PER_NODE on this cluster.

    The request SPELLING is not parsed anywhere -- this is the whole reason
    ``--gres=gpu:ada:2`` and ``--gpus=ada:2`` behave identically.
    """
    assert _resolved(block, {"SLURM_GPUS_PER_NODE": "ada:2"}) == "2"


def test_a_bare_count_in_the_second_source_also_resolves(block: str) -> None:
    assert _resolved(block, {"SLURM_GPUS_PER_NODE": "8"}) == "8"


def test_the_explicit_override_wins(block: str) -> None:
    """The documented escape hatch must beat both Slurm variables."""
    resolved = _resolved(block, {"GPUS_PER_NODE": "3", "SLURM_GPUS_ON_NODE": "8"})
    assert resolved == "3"


def test_a_job_total_is_divided_by_the_node_count(block: str) -> None:
    """`--gpus=N` populates only SLURM_GPUS, and that is a JOB total.

    On a 2-node allocation SLURM_GPUS=8 means 4 per node, so handing it to
    --nproc_per_node undivided over-forks by the node count. The framework's own
    idle-device guard splits these into ALLOC_GPU_ENV_PER_NODE vs _JOB for the
    same reason -- and it is the guard this must agree with, because it REFUSES
    a launch whose grant exceeds the ranks started.
    """
    assert _resolved(block, {"SLURM_GPUS": "8", "SLURM_NNODES": "2"}) == "4"


def test_a_single_node_job_total_is_the_per_node_count(block: str) -> None:
    """The array dispatcher's own case: `--gpus=1` with `--nodes=1`.

    This source was missing, so an allocation that populated only SLURM_GPUS
    fell through to the nvidia-smi fallback -- the node's PHYSICAL device count,
    which can EXCEED the grant, i.e. the one path in this derivation that can
    over-fork rather than under-fork.
    """
    assert _resolved(block, {"SLURM_GPUS": "1"}) == "1"


def test_the_per_node_sources_still_win_over_the_job_total(block: str) -> None:
    """Anti-vacuity: a per-node count must not be overridden by a job total."""
    assert _resolved(block, {"SLURM_GPUS_ON_NODE": "2", "SLURM_GPUS": "8"}) == "2"


def test_visible_devices_are_counted_when_slurm_is_silent(block: str, tmp_path: Path) -> None:
    """The third source -- promised by the old comment, absent from the old code."""
    assert _resolved(block, {}, _stub_nvidia_smi(tmp_path, 4)) == "4"


def test_no_source_at_all_raises(block: str, no_nvidia_smi: str) -> None:
    """The headline behaviour change: refuse, do not assume 1."""
    proc = _run(block, {}, no_nvidia_smi)
    assert proc.returncode != 0
    assert "could not determine GPUs per node" in proc.stderr
    assert "GPUS_PER_NODE" in proc.stderr, "the message must name the escape hatch"


def test_a_type_only_value_raises_rather_than_resolving_to_one(
    block: str, no_nvidia_smi: str
) -> None:
    """``ada`` has no trailing digits. The old idiom turned that into ``1``."""
    proc = _run(block, {"SLURM_GPUS_PER_NODE": "ada"}, no_nvidia_smi)
    assert proc.returncode != 0


def test_a_host_with_no_gpus_raises(block: str, tmp_path: Path) -> None:
    """``nvidia-smi`` present but listing nothing is a zero, not a one."""
    proc = _run(block, {}, _stub_nvidia_smi(tmp_path, 0))
    assert proc.returncode != 0


# ---------------------------------------------------------------------------
# Anti-vacuity: the old idiom, executed, to prove the bug was real
# ---------------------------------------------------------------------------

_OLD_IDIOM = (
    'GPUS_PER_NODE="$(echo "${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}" '
    "| grep -oE '[0-9]+$' || echo 1)\"\n"
    'echo "RESULT=${GPUS_PER_NODE}"'
)


@pytest.mark.parametrize(
    "env",
    [
        pytest.param({}, id="nothing-set"),
        pytest.param({"SLURM_GPUS_PER_NODE": "ada"}, id="type-only"),
    ],
)
def test_the_old_idiom_really_did_resolve_to_one(env: dict[str, str]) -> None:
    """Without this, the tests above could be asserting a distinction that the
    original code already made."""
    proc = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{_OLD_IDIOM}"],
        capture_output=True,
        text=True,
        env={"PATH": _BASE_PATH, **env},
    )
    assert proc.returncode == 0, proc.stderr
    assert "RESULT=1" in proc.stdout


def test_the_repo_still_has_the_launcher_where_the_docs_say() -> None:
    """The header of this file cites the path; a move would strand every skip
    above as a silent green."""
    # require_repo_file, not `SBATCH.is_file()`: both answers must stay
    # distinguishable. A launcher the allowlist denies is a publication boundary
    # (skip); a launcher that MOVED is the defect this test exists for, and it
    # must still fail loudly in any tree that is not the export.
    assert os.access(require_repo_file(_SBATCH_REL), os.R_OK)
