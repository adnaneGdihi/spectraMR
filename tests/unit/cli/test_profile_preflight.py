"""Pre-flight refusals for ``spectramr profile`` — one planted violation per shape.

Both checks here replace a failure that already happened, correctly, minutes
later. So the thing these tests must prove is not "it raises" but "it raises for
exactly the cases the runtime would have raised for, and stays silent for the
rest" — a guard that over-refuses sends the operator editing an arm that was
fine, which is worse than the slow failure it replaced.

Every negative case below is therefore a real shape, not filler:

* ``dp`` is the shape a plausible-but-wrong implementation (``strategy !=
  "none"``) blocks. ``nn.DataParallel`` is single-process and profiles fine.
* ``--target sanity_check`` is exempt from the interval check because that mode
  overwrites ``max_iterations`` *after* overrides are applied.
* ``on_epoch`` and an epoch-derived budget are undecidable without a built
  train loader, so silence is the correct answer, not a guess.

And the five ``--override`` spellings are five separate shapes of one rule: a
spelling the extractor misses does not fail loudly, it makes the whole interval
check silently pass — which is the exact defect it exists to prevent.
"""

from __future__ import annotations

import argparse

import pytest

from spectramr.cli import profile_preflight as pf
from spectramr.config.settings import TrainingSettings

# --------------------------------------------------------------------- helpers

#: `strategy` alone does not validate: `ParallelismConfigSchema` requires the
#: sub-block flag to agree, which is also why the refusal message tells the
#: operator to flip both.
_SUBBLOCK = {
    "fsdp": {"fsdp": {"enabled": True}},
    "deepspeed": {"deepspeed": {"enabled": True}},
}


def settings(*, strategy="none", max_iterations=None, interval_steps=100, on_epoch=False):
    """A REAL TrainingSettings, never a stub.

    A SimpleNamespace would make the override tests vacuous: `apply_overrides`
    would raise on it, `effective_settings` would catch that and fall back to
    declared values, and a test asserting "the override made the guard fire"
    would pass without the override ever being applied.
    """
    return TrainingSettings(
        model={},
        data={},
        optimization={},
        logging={},
        training={"max_iterations": max_iterations},
        validation={"schedule": {"interval_steps": interval_steps, "on_epoch": on_epoch}},
        parallel={"strategy": strategy, **_SUBBLOCK.get(strategy, {})},
    )


def args(**over):
    base = {"target": "train", "extra": [], "config": "arm.yaml"}
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _arm_trains(monkeypatch):
    """Default: the arm's model is a network; parameter-free tests override this."""
    monkeypatch.setattr(
        "spectramr.cli.manifest_dispatch.declares_parameter_free_arm", lambda config: False
    )


# ------------------------------------------------- guard 1: process-group strategies


@pytest.mark.parametrize("strategy", ["ddp", "fsdp", "deepspeed"])
@pytest.mark.parametrize("target", ["train", "sanity_check"])
def test_process_group_strategy_is_refused(strategy, target):
    """One plant per blocked backend, and per target that applies parallelism."""
    with pytest.raises(ValueError, match="cannot be profiled"):
        pf.check_parallel_strategy(settings=settings(strategy=strategy), target=target)


@pytest.mark.parametrize("strategy", ["none", "dp"])
def test_single_process_strategies_are_allowed(strategy):
    """`dp` is the negative that a `strategy != 'none'` check would fail."""
    pf.check_parallel_strategy(settings=settings(strategy=strategy), target="train")


def test_infer_is_exempt_because_it_applies_no_parallelism():
    """`pipelines/infer.py` resolves no strategy, so refusing it would over-block."""
    pf.check_parallel_strategy(settings=settings(strategy="deepspeed"), target="infer")


def test_refusal_names_the_single_process_flag():
    with pytest.raises(ValueError) as exc:
        pf.check_parallel_strategy(settings=settings(strategy="deepspeed"), target="train")
    assert "--single-process" in str(exc.value)


@pytest.mark.parametrize("strategy", ["fsdp", "deepspeed"])
def test_single_process_overrides_validate_and_clear_the_guard(strategy):
    """Applied through the real override parser, so #1480's 'none' -> None regresses red."""
    argv = pf.single_process_overrides(settings(strategy=strategy))
    assert argv == [
        "--override",
        "parallel.strategy=none",
        "--override",
        f"parallel.{strategy}.enabled=false",
    ]
    resolved = pf.effective_settings(settings(strategy=strategy), argv)
    assert resolved.parallel.strategy == "none"
    pf.check_parallel_strategy(settings=resolved, target="train")


@pytest.mark.parametrize("strategy", ["none", "dp"])
def test_single_process_overrides_are_a_noop_without_a_process_group(strategy):
    assert pf.single_process_overrides(settings(strategy=strategy)) == []


def test_process_group_strategies_matches_the_registry():
    assert set(pf.process_group_strategies()) == {"ddp", "fsdp", "deepspeed"}


# ------------------------------------------------------- override extraction


@pytest.mark.parametrize(
    "argv",
    [
        ["--override", "a=1"],
        ["--override=a=1"],
        ["-O", "a=1"],
        ["-Oa=1"],
        ["-O=a=1"],
    ],
    ids=["long-split", "long-joined", "short-split", "short-glued", "short-eq"],
)
def test_every_spelling_argparse_accepts_is_extracted(argv):
    """Each spelling verified against argparse itself, not against belief."""
    child = argparse.ArgumentParser()
    child.add_argument("--override", "-O", action="append", default=[])
    assert pf.extract_overrides(argv) == child.parse_args(argv).override == ["a=1"]


def test_unrelated_passthrough_args_are_not_mistaken_for_overrides():
    assert pf.extract_overrides(["--resume", "auto", "--checkpoint", "best.pt"]) == []


def test_a_trailing_override_flag_with_no_value_is_dropped_not_crashed():
    assert pf.extract_overrides(["--override"]) == []


def test_overrides_keep_their_order_so_the_last_one_wins():
    assert pf.extract_overrides(["-O", "a=1", "--override=a=2"]) == ["a=1", "a=2"]


# ------------------------------------------------ guard 2: validation interval


def test_interval_exceeding_the_declared_budget_is_refused():
    with pytest.raises(ValueError, match="can NEVER fire"):
        pf.check_validation_interval(
            settings=settings(max_iterations=300, interval_steps=5000), target="train"
        )


def test_an_interval_that_fits_the_budget_passes():
    pf.check_validation_interval(
        settings=settings(max_iterations=300, interval_steps=150), target="train"
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--override", "training.max_iterations=300"],
        ["--override=training.max_iterations=300"],
        ["-O", "training.max_iterations=300"],
        ["-Otraining.max_iterations=300"],
        ["-O=training.max_iterations=300"],
    ],
    ids=["long-split", "long-joined", "short-split", "short-glued", "short-eq"],
)
def test_a_budget_arriving_by_override_is_checked(argv):
    """The motivating case: the arm is consistent, the LAUNCH is not.

    experiment_11 declares max_iterations=30000 / interval_steps=5000 — fine.
    `-O training.max_iterations=300` is what made the gate unreachable, and it
    arrives in `extra`, not in the YAML. One plant per spelling.
    """
    declared = settings(max_iterations=30000, interval_steps=5000)
    pf.check_validation_interval(settings=declared, target="train")  # arm alone is fine

    with pytest.raises(ValueError, match="whole budget of 300 iterations"):
        pf.run_preflight(args(extra=argv), declared)


def test_a_matching_interval_override_clears_the_refusal():
    """The remediation the message gives must actually work."""
    pf.run_preflight(
        args(
            extra=[
                "--override",
                "training.max_iterations=300",
                "--override",
                "validation.schedule.interval_steps=150",
            ]
        ),
        settings(max_iterations=30000, interval_steps=5000),
    )


def test_the_message_reproduces_the_loops_own_remediation():
    with pytest.raises(ValueError) as exc:
        pf.check_validation_interval(
            settings=settings(max_iterations=300, interval_steps=5000), target="train"
        )
    # max(1, 300 // 2) -- the same arithmetic training_loop.py prints
    assert "interval_steps=150" in str(exc.value)


def test_sanity_check_is_exempt_because_the_mode_overwrites_the_budget():
    """training_loop.py overwrites max_iterations with 5000 AFTER overrides."""
    pf.check_validation_interval(
        settings=settings(max_iterations=300, interval_steps=5000), target="sanity_check"
    )


def test_infer_is_exempt_from_the_interval_check():
    pf.check_validation_interval(
        settings=settings(max_iterations=300, interval_steps=5000), target="infer"
    )


def test_on_epoch_is_undecidable_up_front_so_it_stays_silent():
    """The epoch gate can rescue the run, but only len(train_loader) knows."""
    pf.check_validation_interval(
        settings=settings(max_iterations=300, interval_steps=5000, on_epoch=True),
        target="train",
    )


def test_an_epoch_derived_budget_stays_silent():
    """max_iterations=None -> epochs x len(train_loader), unknown pre-flight."""
    pf.check_validation_interval(
        settings=settings(max_iterations=None, interval_steps=5000), target="train"
    )


# ------------------------------------------------------------- degradation


def test_an_unappliable_override_reports_rather_than_infers(caplog):
    """Non-negotiable 18: absent is a state to report, never one to infer."""
    import logging

    with caplog.at_level(logging.WARNING, logger=pf.__name__):
        out = pf.effective_settings(
            settings(max_iterations=300), ["--override", "no_such.key=1"]
        )
    assert out.training.max_iterations == 300
    assert any("could not apply" in r.getMessage() for r in caplog.records)


def test_run_preflight_runs_both_checks():
    """A dry run is checked too — printing an argv that cannot work is the trap."""
    with pytest.raises(ValueError, match="cannot be profiled"):
        pf.run_preflight(args(), settings(strategy="ddp"))
    with pytest.raises(ValueError, match="can NEVER fire"):
        pf.run_preflight(args(), settings(max_iterations=300, interval_steps=5000))


# ------------------------------------------------- guard 3: parameter-free arms


@pytest.mark.parametrize("target", ["train", "sanity_check"])
def test_a_parameter_free_arm_is_refused_for_training_targets(monkeypatch, target):
    monkeypatch.setattr(
        "spectramr.cli.manifest_dispatch.declares_parameter_free_arm", lambda config: True
    )
    with pytest.raises(ValueError, match="nothing to train"):
        pf.run_preflight(args(target=target), settings())


def test_a_parameter_free_arm_may_still_be_inferred(monkeypatch):
    monkeypatch.setattr(
        "spectramr.cli.manifest_dispatch.declares_parameter_free_arm", lambda config: True
    )
    pf.run_preflight(args(target="infer"), settings())


def test_a_trainable_arm_passes_the_parameter_free_check():
    pf.check_parameter_free(config="arm.yaml", target="train")


def test_a_torchless_interpreter_reports_instead_of_guessing(monkeypatch, caplog):
    def no_torch(config):
        raise ImportError("no torch", name="torch")

    monkeypatch.setattr("spectramr.cli.manifest_dispatch.declares_parameter_free_arm", no_torch)
    pf.check_parameter_free(config="arm.yaml", target="train")
    assert any("check skipped" in r.getMessage() for r in caplog.records)
