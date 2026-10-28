"""Explain a profiling run that died, instead of handing back a bare exit code.

A profiled run is expensive — the arm in issue #1486 was pacing at 59 s/iteration
— so when one dies the operator has already paid for it. ``Profiled run exited
245`` tells them nothing about whether to re-run the same command, change a flag,
or fix the arm, and the answer is not guessable: 245 is not a signal number, not
a Scalene exit code documented anywhere, and not stable across Scalene versions.

This module reads the failure out of the child's own teed log and names it. It is
a *classifier over observed evidence*, never a predictor: nothing here refuses a
run or changes what executes, so a signature that fails to match degrades the
error message and nothing else. That is deliberate — the known failure below is
**nondeterministic** (the same ``--mode cpu+gpu`` command on the same arm
succeeded the previous night), so a pre-flight refusal would block a mode that
demonstrably works.

**Anchors are library symbols, not human-readable banners.** The crash that
motivated this prints ``!!!!!!! Segfault encountered !!!!!!!`` on the cluster, and
that string appears nowhere in scalene 2.3.0 — the version pinned here — so the
banner's wording is a property of whichever Scalene build the run used. The
anchors below are C++ symbols owned by the *crashing* library (``torch``), which
is the half that does not move when Scalene is upgraded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: How much of the log tail to search. A fatal fault terminates the run, so the
#: evidence is always at the end; a profiled training run's log is unbounded, and
#: reading it whole to diagnose a failure would risk a MemoryError while handling
#: one. Bounded reads mean a signature planted only at the *start* of a huge log
#: is missed by design — an acceptable trade for a diagnostic that must never
#: itself fail.
TAIL_BYTES = 256 * 1024


@dataclass(frozen=True)
class FailureSignature:
    """One recognizable way a profiled child dies.

    ``anchors`` must **all** appear in the searched text. Requiring a conjunction
    rather than any-of is what keeps a signature from firing on a passing mention:
    a log that merely names a profiler is not a profiler crash.
    """

    name: str
    anchors: tuple[str, ...]
    summary: str
    remedy: str
    #: Exit codes this signature is restricted to. Empty means "any code, judge
    #: by text alone". A signature may discriminate on the code INSTEAD of on
    #: text, because the most common cluster death -- the OOM killer -- leaves
    #: no message at all: the log simply stops mid-progress-bar. Requiring a
    #: text anchor there would classify the commonest failure as unrecognized.
    exit_codes: tuple[int, ...] = ()

    def matches(self, text: str, exit_code: int | None = None) -> bool:
        if self.exit_codes:
            if exit_code is None or exit_code not in self.exit_codes:
                return False
        elif not self.anchors:
            # Neither discriminator: this would match every log. Refuse rather
            # than let a malformed signature swallow every future failure.
            return False
        return all(anchor in text for anchor in self.anchors)


#: Recognized failures, most specific first. A table rather than an ``if/elif``
#: chain (non-negotiable 6/20), so adding a signature is data.
KNOWN_FAILURES: tuple[FailureSignature, ...] = (
    FailureSignature(
        name="scalene_torch_profiler_segfault",
        # Both frames in the observed stack -- `_Sp_counted_ptr<...Result*>` and
        # `vector<shared_ptr<...Result>>::~vector` -- contain this one substring,
        # so a single anchor covers both shapes the crash presents.
        anchors=("torch::profiler::impl::Result",),
        summary=(
            "the child died inside PyTorch's profiler while releasing its event "
            "buffer (torch::profiler::impl::Result). spectramr never starts a torch "
            "profiler; Scalene does. Its TorchProfiler restarts a "
            "torch.profiler.profile(with_stack=True) every 100 sampling steps and "
            "keeps only a fraction of the windows, so each restart abandons a "
            "profiler holding a large vector<shared_ptr<Result>>. Collecting one "
            "of those inside a Python signal handler overruns the C stack. The "
            "crash is timing-dependent, not deterministic -- the same command can "
            "and does succeed."
        ),
        remedy=(
            "Scalene starts that profiler only when its own `gpu` flag is on, "
            "which --mode full (the default) and --mode cpu+gpu both set. Re-run "
            "with --mode cpu-only (fastest, no GPU or memory columns) or "
            "--mode cpu+memory to avoid it entirely, or simply retry the same "
            "mode -- it is intermittent. See docs/SCALENE_USAGE.md."
        ),
    ),
    FailureSignature(
        name="arm_refuses_to_run",
        anchors=("Pipeline failed: metadata.status=",),
        summary=(
            "the arm's own metadata.status says its mechanism cannot produce a result today "
            "(needs_implementation / inert), so the pipeline refused to start it."
        ),
        remedy=(
            "Not a defect in the run: the guard is deliberate. Profile another arm, or resolve "
            "the status the arm names before profiling it."
        ),
    ),
    FailureSignature(
        name="data_root_missing",
        anchors=("Data root not found:",),
        summary="the arm's data root does not exist on this machine.",
        remedy=(
            "Put the dataset at the printed path, or point data.source.root at what exists on "
            "this cluster. The profile cannot start without data."
        ),
    ),
    FailureSignature(
        name="cuda_out_of_memory",
        anchors=("torch.OutOfMemoryError: CUDA out of memory",),
        summary=(
            "the run exhausted GPU memory (the log's allocated/capacity figures say how far). "
            "A 'Validation lost N of M batch(es)' line next to it means the step trained and "
            "validation did not fit."
        ),
        remedy=(
            "Lower validation.loader.chunk_size or the batch size, or run on a larger GPU; "
            "metrics computed on a whole folded volume (LPIPS) are the usual extra allocation."
        ),
    ),
    FailureSignature(
        name="child_killed_sigkill",
        # No text anchor ON PURPOSE. A cgroup OOM kill is silent: the kernel
        # SIGKILLs the process and the log just stops. Anchoring on a message
        # would miss the common case and recognize only the chatty one.
        anchors=(),
        # -9 is Popen.wait()'s own report; 137 is the shell's 128+signal
        # convention; 247 is what Scalene exits with when it observes the kill
        # (256-9), which is the code the 2026-08-24 cluster run actually
        # returned. All three mean the same thing.
        exit_codes=(-9, 137, 247),
        summary=(
            "the child was SIGKILLed. Nothing in the process chose this: SIGKILL "
            "cannot be caught, so the sender was outside -- on a cluster that is "
            "the cgroup OOM killer (host RAM, not GPU memory) or the scheduler "
            "enforcing a time limit. Check the scheduler's own log for an "
            "oom_kill count; it is printed by slurmstepd AFTER this process "
            "exits and so is never captured here."
        ),
        remedy=(
            "Confirm which it was: SLURM prints 'oom_kill events' for memory and "
            "'DUE TO TIME LIMIT' for the clock. If memory, start with the run's "
            "own footprint. The largest lever is data.loader.num_workers: each "
            "worker holds whole volumes in flight, so halving it roughly halves "
            "peak host RAM. data.sampling.queue_length is the next one. Raising "
            "the job's --mem is the other half. On the profiler's own overhead: "
            "its restart cycle was MEASURED to PLATEAU rather than grow without "
            "limit (+0.56 GB over 300 windows on a small model, against 0.00 for "
            "an unprofiled control) -- but that measures the SHAPE, not the "
            "level, and the plateau scales with the profiled model. So this does "
            "not exonerate the profiler for your arm; rule it out with a control "
            "run at --mode cpu-only, which starts no torch profiler at all. "
            "See docs/SCALENE_USAGE.md."
        ),
    ),
)


def read_log_tail(log_path: Path, limit: int = TAIL_BYTES) -> str | None:
    """Return the last ``limit`` bytes of ``log_path``, or ``None`` if unreadable.

    ``None`` is a reportable state, not an inferred one: the caller records
    "log unreadable" rather than "no known failure", because the two justify
    different next actions.
    """
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as fh:
            if size > limit:
                fh.seek(size - limit)
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None


def classify_failure(text: str, exit_code: int | None = None) -> FailureSignature | None:
    """First signature that matches ``text`` (and ``exit_code``, when it discriminates).

    Order is significance, not specificity of match: an OOM kill during a
    segfaulting run is still reported as the segfault, because that is the
    finding with an action attached.
    """
    for signature in KNOWN_FAILURES:
        if signature.matches(text, exit_code):
            return signature
    return None


def explain_failure(exit_code: int, log_path: Path) -> dict[str, object]:
    """Log a human explanation for a failed run and return it for the manifest.

    Total function: every path returns a record and none raises. A diagnostic
    that threw while explaining a crash would replace the operator's real failure
    with its own, which is the one outcome that would make this worse than the
    bare exit code it replaces.

    ``status`` distinguishes the three outcomes deliberately -- ``recognized``,
    ``unrecognized`` and ``log_unavailable`` are different facts, and collapsing
    the last two would let a missing file read as a clean run.
    """
    logger.error("Profiled run exited %s — see %s", exit_code, log_path)
    text = read_log_tail(log_path)
    if text is None:
        logger.error(
            "Could not read %s to diagnose the failure — the exit code (%s) is "
            "all the evidence there is.",
            log_path,
            exit_code,
        )
        return {"status": "log_unavailable", "exit_code": exit_code}

    signature = classify_failure(text, exit_code)
    if signature is None:
        return {"status": "unrecognized", "exit_code": exit_code}

    logger.error("Known failure '%s': %s", signature.name, signature.summary)
    logger.error("What to do: %s", signature.remedy)
    return {
        "status": "recognized",
        "exit_code": exit_code,
        "signature": signature.name,
        "summary": signature.summary,
        "remedy": signature.remedy,
    }


__all__ = [
    "KNOWN_FAILURES",
    "TAIL_BYTES",
    "FailureSignature",
    "classify_failure",
    "explain_failure",
    "read_log_tail",
]
