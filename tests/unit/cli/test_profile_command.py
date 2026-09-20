"""Tests for the ``spectramr profile`` vocabulary and argv construction.

The three tests that earn their keep here are the ones pinning facts that are
true *outside* this repo and would otherwise rot silently:

* ``--program-path`` really resolves to the package root, not ``cli/``;
* Scalene's mode flags narrow rather than add;
* the argv we build is one the real child parser accepts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from spectramr.cli.profile_command import (
    FOCUS_PRESETS,
    OUTPUT_INJECTION,
    PROFILE_MODES,
    PROFILE_TARGETS,
    build_scalene_command,
    entry_script,
    program_path,
    require_scalene,
)
from spectramr.cli.val_batches import VAL_BATCHES_OVERRIDE_KEY


def _ns(**over) -> argparse.Namespace:
    base = {
        "target": "train",
        "mode": "full",
        "focus": "all",
        "config": Path("experiments/validated/dummy_gan.yaml"),
        "device": None,
        "val_batches": None,
        "extra": [],
    }
    base.update(over)
    return argparse.Namespace(**base)


def _child_argv(argv: list[str]) -> list[str]:
    """The part after Scalene's ``---`` separator."""
    return argv[argv.index("---") + 1 :]


# ---------------------------------------------------------------- the landmine


def test_program_path_is_package_root_not_cli_dir():
    """Scalene defaults --program-path to the profiled script's directory.

    That default would be ``src/spectramr/cli/``, which excludes
    ``pipelines/training_loop.py`` — a green run with an empty profile. This
    asserts the override points at the package root and that the loop is under
    it, which is the property that actually matters.
    """
    root = program_path()
    assert root.name == "spectramr"
    assert entry_script().parent == root / "cli"
    assert (root / "pipelines" / "training_loop.py").is_file()
    assert entry_script().parent != root, "the naive default would scope to cli/"


def test_build_always_sets_program_path_and_outfile():
    argv = build_scalene_command(_ns(), child_run_dir=Path("rd"), outfile=Path("rd/p.json"))
    assert "--program-path" in argv
    assert argv[argv.index("--program-path") + 1] == str(program_path())
    assert argv[argv.index("--outfile") + 1] == "rd/p.json"


# ------------------------------------------------------- scalene mode semantics


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("full", {"cpu": True, "gpu": True, "memory": True}),
        ("cpu-only", {"cpu": True, "gpu": False, "memory": False}),
        ("cpu+gpu", {"cpu": True, "gpu": True, "memory": False}),
        ("cpu+memory", {"cpu": True, "gpu": False, "memory": True}),
    ],
)
def test_mode_flags_match_scalene_semantics(mode, expected, monkeypatch):
    """Every ``--mode`` produces exactly the sampling set its name claims.

    Scalene's flags NARROW: ``--gpu`` alone turns memory off, ``--memory`` alone
    turns GPU off. The ``--help-advanced`` text reads like they add. If a future
    scalene changes this, the name ``cpu+gpu`` becomes a lie and this goes red.
    """
    pytest.importorskip("scalene")
    from scalene.scalene_parseargs import ScaleneParseArgs

    monkeypatch.setattr("sys.argv", ["scalene", "run", *PROFILE_MODES[mode], "prog.py"])
    args, _ = ScaleneParseArgs.parse_args()
    assert {k: getattr(args, k) for k in expected} == expected


# ------------------------------------------------------------ table invariants


def test_every_target_has_an_output_injection():
    """A target with no redirect would scatter its artifacts into the CWD."""
    assert set(OUTPUT_INJECTION) == set(PROFILE_TARGETS)


def test_focus_presets_name_paths_that_exist():
    """A stale substring silently narrows the report to nothing."""
    root = program_path()
    for focus, filters in FOCUS_PRESETS.items():
        for frag in filters:
            assert (root / frag).exists(), f"--focus {focus}: {frag} is gone"


#: ``--focus`` preset -> the driver function a reader of that name expects to see
#: in the report. Kept as source-level truth, NOT read back off ``PHASE_MARKERS``:
#: the presets are *derived* from that table, so comparing the two would assert a
#: tautology and stay green through a mis-wired derivation (train's file bound to
#: ``val-loop``). Asking the source where the function is defined is the only form
#: of this check that can fail.
LOOP_PRESET_DRIVERS = {
    "train-loop": "_execute_training_loop",
    "val-loop": "_run_validation",
}


@pytest.mark.parametrize(("focus", "driver"), sorted(LOOP_PRESET_DRIVERS.items()))
def test_each_loop_preset_names_the_file_that_defines_its_driver(focus, driver):
    """An existing path is not the RIGHT path — the shape that shipped broken.

    ``test_focus_presets_name_paths_that_exist`` only proves each fragment is
    still on disk, so ``val-loop`` naming ``pipelines/training_loop.py`` passed
    it while excluding ``_run_validation`` (which lives in ``pipelines/train.py``)
    from the very report the preset is named after. Existence is not membership.
    """
    root = program_path()
    named = [
        (root / frag).read_text(encoding="utf-8")
        for frag in FOCUS_PRESETS[focus]
        if (root / frag).is_file()
    ]
    assert any(f"def {driver}(" in text for text in named), (
        f"--focus {focus} names {FOCUS_PRESETS[focus]}, none of which defines "
        f"{driver}() — the preset would filter the report to files that do not "
        f"contain the loop it is named after."
    )


def test_focus_all_omits_profile_only():
    argv = build_scalene_command(_ns(focus="all"), child_run_dir=Path("rd"), outfile=Path("o.json"))
    assert "--profile-only" not in argv


def test_focus_train_loop_passes_comma_joined_filters():
    argv = build_scalene_command(
        _ns(focus="train-loop"), child_run_dir=Path("rd"), outfile=Path("o.json")
    )
    assert argv[argv.index("--profile-only") + 1] == (
        "pipelines/training_loop.py,infrastructure/training/"
    )


# ------------------------------------------------------- the child accepts it


@pytest.mark.parametrize("target", PROFILE_TARGETS)
def test_built_argv_is_accepted_by_the_real_child_parser(target):
    """The argv we hand Scalene must parse under the CLI the child actually runs.

    This is the test that would have caught ``infer`` not accepting
    ``--override`` — the reason OUTPUT_INJECTION is per-target at all.
    """
    from spectramr.cli.app import build_parser

    extra = ["--checkpoint", "ckpt.pt", "--input", "data/test/"] if target == "infer" else []
    argv = build_scalene_command(
        _ns(target=target, extra=extra),
        child_run_dir=Path("experiments/results/x"),
        outfile=Path("o.json"),
    )
    parsed = build_parser().parse_args(_child_argv(argv))
    assert parsed.command == target


@pytest.mark.parametrize("verb", ["predict", "audit", "hpo", "ablation"])
def test_excluded_verbs_are_excluded_for_a_real_reason(verb):
    """Each exclusion is justified in PROFILE_TARGETS' comment; prove it holds.

    ``predict``/``audit`` have no ``--config`` FLAG; ``hpo``/``ablation`` fan out
    into subprocesses Scalene cannot see. Only the mechanical half is checkable
    here, so check that and keep the list honest.
    """
    from spectramr.cli.app import build_parser

    assert verb not in PROFILE_TARGETS
    sub = next(a for a in build_parser()._actions if a.dest == "command")
    assert verb in sub.choices, f"{verb} vanished; the exclusion note is stale"


def test_device_is_passed_through_to_the_child_not_to_scalene():
    argv = build_scalene_command(_ns(device="cpu"), child_run_dir=Path("rd"), outfile=Path("o.json"))
    child = _child_argv(argv)
    assert child[child.index("--device") + 1] == "cpu"
    assert "--device" not in argv[: argv.index("---")]


def test_extra_args_land_after_the_injected_flags():
    argv = build_scalene_command(
        _ns(extra=["--resume", "auto"]), child_run_dir=Path("rd"), outfile=Path("o.json")
    )
    child = _child_argv(argv)
    assert child[-2:] == ["--resume", "auto"]


# ------------------------------------------------------------- no silent fallback


def test_require_scalene_raises_when_absent(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match=r"scalene is not installed"):
        require_scalene()


def test_require_scalene_returns_version_when_present():
    pytest.importorskip("scalene")
    assert require_scalene()


# ------------------------------------------------------------- --val-batches


def test_val_batches_appends_an_override_without_losing_the_output_redirect():
    """`--override` is `action="append"`, so the two must coexist, not replace."""
    argv = build_scalene_command(
        _ns(val_batches=4), child_run_dir=Path("experiments/results/exp_11"), outfile=Path("p.json")
    )
    child = _child_argv(argv)
    overrides = [child[i + 1] for i, tok in enumerate(child) if tok == "--override"]
    assert "training.output_dir=experiments/results/exp_11" in overrides
    assert f"{VAL_BATCHES_OVERRIDE_KEY}=4" in overrides


def test_no_val_batches_adds_no_override():
    child = _child_argv(build_scalene_command(_ns(), child_run_dir=Path("r"), outfile=Path("p.json")))
    assert not any("num_batches" in tok for tok in child)


def test_val_batches_targets_a_key_the_framework_actually_reads():
    """Guards against re-pointing this at `validation.enabled`, which is inert.

    Issue #673: 1006 arms set `validation.enabled`, 8 to `false`, and validation
    runs regardless. A cap wired to it would report success and do nothing.
    """
    child = _child_argv(
        build_scalene_command(_ns(val_batches=2), child_run_dir=Path("r"), outfile=Path("p.json"))
    )
    assert not any("validation.enabled" in tok for tok in child)
    assert f"{VAL_BATCHES_OVERRIDE_KEY}=2" in child


# --------------------------------------------------------------------------
# Output isolation, asserted on the argv the child actually receives.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["train", "sanity_check", "infer"])
def test_no_target_is_told_to_write_into_the_arms_real_results_dir(target):
    """Every target, because the redirect table is per-target.

    `infer` shipped the same defect through a different flag (`--output`), so a
    fix applied only to the `--override` rows would have left it corrupting
    `experiments/results/<exp>/inference/`.
    """
    from spectramr.cli.profile_paths import resolve_profile_paths

    paths = resolve_profile_paths("exp_11", "run-1")
    argv = build_scalene_command(
        _ns(target=target), child_run_dir=paths.child_run_dir, outfile=paths.outfile
    )
    child = _child_argv(argv)

    injected = [a for a in child if "exp_11" in a]
    assert injected, f"nothing in the child argv addresses the run dir: {child}"
    for value in injected:
        assert "profiles/run-1/run" in value
        assert not value.endswith("experiments/results/exp_11")


def test_every_output_injection_template_uses_the_isolated_key():
    """A new target added with the old `{run_dir}` key must not slip through.

    `str.format` ignores unused kwargs, so a stale `{run_dir}` template would
    not raise — it would raise KeyError only because `run_dir` is no longer
    passed. Asserting on the templates says the intent directly.
    """
    for target, (_flag, template) in OUTPUT_INJECTION.items():
        assert "{child_run_dir}" in template, target
        assert "{run_dir}" not in template.replace("{child_run_dir}", ""), target
