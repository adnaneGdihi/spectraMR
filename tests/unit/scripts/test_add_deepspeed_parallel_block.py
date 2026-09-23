"""``--block-file``: the block is data, the oracle is not.

The rollout's insertion point and its resolve/diff oracle are one
implementation, but the *rationale* a cohort carries is cohort-specific -- an
fp32 cohort and a bf16 one owe different explanations, and a Mamba arm owes the
reason DeepCompile is off. So the block text became a parameter while the check
that nothing outside ``parallel`` moved did not.

That split is only safe while a supplied block is still rooted at a top-level
``parallel:``. Two invariants depend on it and neither announces itself: the
idempotency guard recognises an already-migrated file by that key, and the
oracle tolerates movement only under that path. A block rooted anywhere else
would be re-inserted on every run and trip the diff on the first. The guard is
planted below rather than argued (non-negotiable 15).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = "scripts/migrations/add_deepspeed_parallel_block.py"


def _load():
    spec = importlib.util.spec_from_file_location("_add_ds_block", _REPO_ROOT / _SCRIPT)
    assert spec and spec.loader, _SCRIPT
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


_ARM = "config_version: '1.0'\nmodel:\n  model_type: unet\ndata:\n  batch_size: 4\n"


def test_the_builtin_block_is_still_the_default(mod) -> None:
    out = mod.insert_block(_ARM)
    assert out is not None
    assert "strategy: deepspeed" in out
    assert out.index("parallel:") < out.index("data:"), "block goes above the data: anchor"


def test_a_supplied_block_replaces_the_builtin_one(mod) -> None:
    out = mod.insert_block(_ARM, "parallel:\n  strategy: ddp\n")
    assert out is not None
    assert "strategy: ddp" in out
    assert "deepspeed" not in out, "the built-in block must not leak through"


def test_the_anchor_contract_is_unchanged_by_the_parameter(mod) -> None:
    """Zero or many top-level ``data:`` keys is still a refusal, not a guess."""
    assert mod.insert_block("model:\n  model_type: unet\n", "parallel:\n") is None
    assert mod.insert_block("data:\na: 1\ndata:\n", "parallel:\n") is None


def test_a_block_without_a_top_level_parallel_key_is_refused(mod, tmp_path, capsys) -> None:
    """Planted: the violation the guard exists for.

    ``optimization:`` is a plausible thing to hand this flag and is exactly the
    shape that would be re-inserted forever while tripping the oracle, so it is
    the block the refusal is measured against.
    """
    bad = tmp_path / "block.txt"
    bad.write_text("optimization:\n  compile:\n    enabled: true\n")
    arm = tmp_path / "arm.yaml"
    arm.write_text(_ARM)

    rc = mod.main([str(arm), "--block-file", str(bad), "--apply"])

    assert rc == 1
    assert "no top-level 'parallel:' key" in capsys.readouterr().err
    assert arm.read_text() == _ARM, "a refused run must not touch the corpus"


def test_an_indented_parallel_key_does_not_satisfy_the_guard(mod, tmp_path, capsys) -> None:
    """The second shape the same rule takes: nested, so not a top-level key.

    The MESSAGE is the assertion, not the exit code. With the guard removed this
    block still exits 1 -- the resolve oracle rejects the stray top-level key a
    moment later -- so an exit-code-only test passes either way and pins
    nothing. Only the guard produces this text.
    """
    bad = tmp_path / "block.txt"
    bad.write_text("deepspeed:\n  parallel:\n    strategy: deepspeed\n")
    arm = tmp_path / "arm.yaml"
    arm.write_text(_ARM)

    assert mod.main([str(arm), "--block-file", str(bad), "--apply"]) == 1
    assert "no top-level 'parallel:' key" in capsys.readouterr().err
    assert arm.read_text() == _ARM
