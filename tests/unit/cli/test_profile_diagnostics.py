"""Tests for the failed-profiling-run classifier.

The point of a classifier is that it *discriminates*, so every recognition test
here is paired with a near-miss that must NOT match. A signature matching its own
motivating log proves nothing on its own — a matcher hardcoded to return the one
known failure would pass that test and be worse than useless on the next crash.

The recognized fixture is a verbatim excerpt of the crash as it was actually
emitted (a cluster run of ``experiment_11_attention_none`` on 2026-08-24), not a
paraphrase: a fixture written from the docstring agrees with the docstring by
construction and stops testing the producer.
"""

from __future__ import annotations

import pytest

from spectramr.cli.profile_diagnostics import (
    KNOWN_FAILURES,
    TAIL_BYTES,
    FailureSignature,
    classify_failure,
    explain_failure,
    read_log_tail,
)

# Verbatim from the failing run's teed log. Both frame shapes the crash presents
# are kept -- the `_Sp_counted_ptr<...Result*>` dispose frame and the
# `vector<shared_ptr<...Result>>::~vector` frame -- because a signature that only
# matched one of them would be blind to half the stacks this crash produces.
REAL_CRASH_LOG = """\
Training:   0%|          | 1/300 [00:59<4:54:47, 59.15s/it, generator_loss=0.5748]\
!!!!!!! Segfault encountered !!!!!!!
  File "<unknown>", line 0, in std::_Sp_counted_base<(__gnu_cxx::_Lock_policy)2>::_M_release()
  File "<unknown>", line 0, in std::_Sp_counted_ptr<torch::profiler::impl::Result*, \
(__gnu_cxx::_Lock_policy)2>::_M_dispose()
  File "<unknown>", line 0, in std::_Sp_counted_base<(__gnu_cxx::_Lock_policy)2>::_M_release_last_use_cold()
  File "<unknown>", line 0, in std::vector<std::shared_ptr<torch::profiler::impl::Result>, \
std::allocator<std::shared_ptr<torch::profiler::impl::Result> > >::~vector()
"""

# --------------------------------------------------------------- recognition


def test_the_real_crash_is_recognized():
    sig = classify_failure(REAL_CRASH_LOG)
    assert sig is not None
    assert sig.name == "scalene_torch_profiler_segfault"


def test_the_remedy_names_the_modes_that_avoid_it():
    """The remedy must be actionable. Scalene starts the crashing profiler only
    when its ``gpu`` flag is set, so the escape is a ``--mode`` that clears it —
    naming the mechanism without naming the flag leaves the operator stuck."""
    sig = classify_failure(REAL_CRASH_LOG)
    assert "cpu-only" in sig.remedy
    assert "cpu+memory" in sig.remedy


def test_the_summary_does_not_claim_the_crash_is_deterministic():
    """The same command succeeded the night before. A summary implying the mode
    is always broken would push operators off a mode that works."""
    sig = classify_failure(REAL_CRASH_LOG)
    assert "not deterministic" in sig.summary or "intermittent" in sig.remedy


# ------------------------------------------------------------ discrimination
# One near-miss per way this could over-match. Each of these is a real failure a
# profiled run can hit; none is THIS failure.


@pytest.mark.parametrize(
    "label,text",
    [
        ("empty", ""),
        (
            "segfault_elsewhere",
            "!!!!!!! Segfault encountered !!!!!!!\n"
            '  File "<unknown>", line 0, in c10::cuda::CUDACachingAllocator::malloc()',
        ),
        (
            "mentions_profiler_without_crashing",
            "Starting torch.profiler.profile(with_stack=True) for the train loop",
        ),
        (
            "python_traceback",
            'Traceback (most recent call last):\n  File "train.py", line 1\nValueError: bad',
        ),
    ],
)
def test_unrelated_failures_are_not_misattributed(label, text):
    assert classify_failure(text) is None, f"{label} was wrongly classified"


def test_a_partial_symbol_does_not_match():
    """``torch::profiler`` alone is not the anchor — the crash is specifically in
    ``Result`` teardown, and a looser anchor would swallow unrelated profiler
    frames."""
    assert classify_failure("in torch::profiler::impl::kineto_client_interface") is None


# -------------------------------------------------------------- bounded read


def test_a_signature_at_the_end_of_a_huge_log_is_found(tmp_path):
    """The evidence for a fatal fault is always at the tail, and the tail is what
    a bounded read keeps."""
    log = tmp_path / "scalene.log"
    log.write_text("filler line\n" * 200_000 + REAL_CRASH_LOG, encoding="utf-8")
    assert log.stat().st_size > TAIL_BYTES
    assert classify_failure(read_log_tail(log)) is not None


def test_a_signature_before_the_tail_window_is_missed_by_design(tmp_path):
    """Documents the trade rather than pretending it does not exist: a bounded
    read cannot see a signature buried megabytes above the end. Stated as a test
    so a future widening of the window is a deliberate change, not a surprise."""
    log = tmp_path / "scalene.log"
    log.write_text(REAL_CRASH_LOG + "filler line\n" * 200_000, encoding="utf-8")
    assert classify_failure(read_log_tail(log)) is None


def test_read_tail_returns_the_whole_file_when_small(tmp_path):
    log = tmp_path / "scalene.log"
    log.write_text(REAL_CRASH_LOG, encoding="utf-8")
    assert read_log_tail(log) == REAL_CRASH_LOG


def test_undecodable_bytes_do_not_raise(tmp_path):
    """A profiled child's log carries tqdm control bytes and can be cut mid-UTF-8
    sequence by a fatal fault."""
    log = tmp_path / "scalene.log"
    log.write_bytes(b"\xff\xfe partial " + REAL_CRASH_LOG.encode())
    assert classify_failure(read_log_tail(log)) is not None


# ----------------------------------------------------- total-function contract


def test_missing_log_is_reported_not_inferred(tmp_path):
    """``log_unavailable`` and ``unrecognized`` are different facts: collapsing
    them would let a missing file read as a run with no known problem."""
    out = explain_failure(245, tmp_path / "absent.log")
    assert out["status"] == "log_unavailable"
    assert out["exit_code"] == 245


def test_an_unreadable_path_does_not_raise(tmp_path):
    """A directory where a file is expected — the diagnostic must not replace the
    operator's real failure with its own."""
    out = explain_failure(1, tmp_path)
    assert out["status"] == "log_unavailable"


def test_recognized_failure_returns_the_full_record(tmp_path):
    log = tmp_path / "scalene.log"
    log.write_text(REAL_CRASH_LOG, encoding="utf-8")
    out = explain_failure(245, log)
    assert out["status"] == "recognized"
    assert out["signature"] == "scalene_torch_profiler_segfault"
    assert out["summary"] and out["remedy"]


def test_explain_failure_forwards_the_exit_code_to_the_classifier(tmp_path):
    """The production path, not the helper.

    ``REAL_OOM_LOG`` carries no text a signature anchors on -- the whole point of
    ``child_killed_sigkill`` is that it discriminates on the code. So this can
    only pass if ``explain_failure`` actually hands its ``exit_code`` down.
    Reverting that one call to ``classify_failure(text)`` leaves every
    classifier-level test green, which is precisely why this test exists here.
    """
    log = tmp_path / "scalene.log"
    log.write_text(REAL_OOM_LOG, encoding="utf-8")
    out = explain_failure(247, log)
    assert out["status"] == "recognized"
    assert out["signature"] == "child_killed_sigkill"


def test_explain_failure_forwards_the_actual_exit_code_not_a_constant(tmp_path):
    """Same log, a code no signature claims -> unrecognized.

    Without this, a call site that forwarded some fixed value (or that the
    signature matched on text after all) would satisfy the test above.
    """
    log = tmp_path / "scalene.log"
    log.write_text(REAL_OOM_LOG, encoding="utf-8")
    out = explain_failure(3, log)
    assert out["status"] == "unrecognized"


def test_unrecognized_failure_still_returns_a_record(tmp_path):
    log = tmp_path / "scalene.log"
    log.write_text("ValueError: something else entirely", encoding="utf-8")
    out = explain_failure(3, log)
    assert out["status"] == "unrecognized"
    assert out["exit_code"] == 3


def test_the_exit_code_is_always_logged(tmp_path, caplog):
    """The classifier took over the "run exited N" line from ``profile_cli``; if
    it stopped emitting it the operator would lose the exit code entirely."""
    log = tmp_path / "scalene.log"
    log.write_text("nothing recognizable", encoding="utf-8")
    with caplog.at_level("ERROR"):
        explain_failure(245, log)
    assert any("245" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------- table


def test_every_signature_has_at_least_one_discriminator_and_advice():
    """A signature with neither anchors nor exit codes matches every log
    (``all([])`` is True), which would misattribute every future failure to it.

    Anchors are no longer the only discriminator -- an OOM kill is silent, so
    ``child_killed_sigkill`` discriminates on the exit code instead. The
    invariant is therefore "at least one of the two", not "anchors".
    """
    assert KNOWN_FAILURES
    for sig in KNOWN_FAILURES:
        assert sig.anchors or sig.exit_codes, f"{sig.name} would match everything"
        assert sig.summary and sig.remedy


def test_a_signature_with_no_discriminator_never_fires():
    """The guard, exercised directly -- the invariant above only checks the
    signatures that happen to be in the table today."""
    empty = FailureSignature(name="x", anchors=(), summary="s", remedy="r")
    assert not empty.matches("literally anything", 1)
    assert not empty.matches("", None)


# --------------------------------------------------------------- sigkill / oom

#: Verbatim tail of the 2026-08-24 cluster run that was OOM-killed at
#: iteration 150/300. Note what it does NOT contain: no "oom_kill", no
#: "Out Of Memory" -- slurmstepd prints those AFTER the child exits, so they
#: never reach the child's own log. That absence is the point of the test.
REAL_OOM_LOG = """\
Training:  50%|#####     | 149/300 [10:55<05:17,  2.10s/it, loss=0.4746]
Training:  50%|#####     | 150/300 [10:57<04:55,  1.97s/it, loss=0.3576]\
Scalene error: received signal SIGKILL
terminate called after throwing an instance of 'std::system_error'
  what():  Invalid argument
"""


@pytest.mark.parametrize("code", [-9, 137, 247])
def test_every_sigkill_exit_code_is_recognized(code):
    """-9 (Popen), 137 (shell 128+n) and 247 (Scalene's 256-n) all mean SIGKILL.
    247 is the one the real cluster run returned."""
    sig = classify_failure(REAL_OOM_LOG, code)
    assert sig is not None and sig.name == "child_killed_sigkill"


def test_a_silent_oom_with_no_message_is_still_recognized():
    """The common case: the kernel kills the process and the log just stops."""
    sig = classify_failure("Training:  50%|#####     | 150/300 [10:57<04:55]\n", -9)
    assert sig is not None and sig.name == "child_killed_sigkill"


@pytest.mark.parametrize("code", [1, 2, 0, 130])
def test_the_same_text_on_a_non_sigkill_exit_is_not_an_oom(code):
    """Discrimination: the signature must key on the code, not on the presence
    of a progress bar. 130 is SIGINT -- a user Ctrl-C is not an OOM."""
    sig = classify_failure(REAL_OOM_LOG, code)
    assert sig is None or sig.name != "child_killed_sigkill"


def test_exit_code_unknown_does_not_fire_an_exit_code_signature():
    """``classify_failure(text)`` with no code must not guess SIGKILL."""
    assert classify_failure(REAL_OOM_LOG) is None


def test_a_segfault_that_was_also_killed_reports_the_segfault():
    """Ordering: both signatures can match one log. The segfault is the finding
    with an action attached, so it must win."""
    both = REAL_CRASH_LOG + "\n" + REAL_OOM_LOG
    sig = classify_failure(both, 247)
    assert sig is not None and sig.name == "scalene_torch_profiler_segfault"


def test_the_oom_remedy_names_the_actual_levers():
    """The advice must be actionable: name the knobs, and name the control run.

    The measurement behind this signature bounds the SHAPE of the profiler's
    overhead (it plateaus) and not its LEVEL (the plateau scales with the model),
    so the remedy must not read as an exoneration. ``--mode cpu-only`` is the
    escape hatch that settles it for a given arm, and is pinned here because
    dropping it would leave the operator with an opinion and no experiment.
    """
    sig = next(s for s in KNOWN_FAILURES if s.name == "child_killed_sigkill")
    assert "num_workers" in sig.remedy
    assert "--mem" in sig.remedy
    assert "queue_length" in sig.remedy
    assert "cpu-only" in sig.remedy


def test_the_oom_summary_does_not_claim_gpu_memory():
    """A cgroup OOM is host RAM. Reading it as CUDA OOM sends the user to
    batch_size, which would not have helped this run."""
    sig = next(s for s in KNOWN_FAILURES if s.name == "child_killed_sigkill")
    assert "host RAM" in sig.summary
    assert "not GPU memory" in sig.summary


@pytest.mark.parametrize(
    ("line", "name"),
    [
        (
            "ERROR spectramr.main: Pipeline failed: metadata.status='needs_implementation': this arm",
            "arm_refuses_to_run",
        ),
        (
            "ERROR spectramr.main: Pipeline failed: Data root not found: /p/hcp/T1w (resolved: /p)",
            "data_root_missing",
        ),
        (
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 994.00 MiB.",
            "cuda_out_of_memory",
        ),
    ],
)
def test_pipeline_failure_families_are_recognised_from_their_real_log_lines(line, name) -> None:
    sig = classify_failure(f"...progress...\n{line}\n", exit_code=1)
    assert sig is not None
    assert sig.name == name
    assert sig.name != "scalene_torch_profiler_segfault"


def test_a_pipeline_log_that_only_mentions_oom_in_passing_is_not_classified() -> None:
    assert classify_failure("setting expandable_segments to avoid CUDA out of memory", 1) is None
