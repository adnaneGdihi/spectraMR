"""The checkpoint switch-over is done when no pre-1.0 checkpoint name or symbol is left.

Code (src/, scripts/, runners/, and experiments/ scripts) may not spell a retired file
name, symbol or envelope key; the corpus YAMLs may not spell a name that cannot be a v1
name; and every corpus checkpoint path must name its producer's best file
(drain_checkpoint_paths_v1 --check). ``best.pt`` is only banned in code: it is the v1
best file of a ``format: pt`` arm, so in YAML the drainer's producer check owns it.

Only **tracked (indexed) files** are scanned; ``git add`` a new file before trusting a
green local run. ``tests/`` is not scanned: the converter's tests and
``tests/fixtures/checkpoints_pre_v1/`` must spell the pre-1.0 names, and so must the
plants below. An allowlist entry permits named labels in one file, never the file.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

from ._legacy_checkpoint_names import (
    ALLOWLIST,
    CODE_NEEDLES,
    DELETED,
    REPO,
    YAML_NEEDLES,
    YAML_SUFFIXES,
    code_files,
    code_offenders,
    corpus_yamls,
    git_ls,
    parse_ls_files,
    read_all,
    scan_text,
    scan_yaml_text,
    tracked,
)

pytestmark = pytest.mark.architecture

__all__ = ["ALLOWLIST", "scan_text", "scan_yaml_text"]


def test_no_code_file_spells_a_pre_v1_name() -> None:
    assert code_offenders(read_all(code_files())) == {}


def test_no_corpus_yaml_spells_a_pre_v1_name() -> None:
    texts = read_all(corpus_yamls())
    offenders = {f: hits for f, text in texts.items() if (hits := scan_yaml_text(text))}
    assert offenders == {}, offenders


def test_every_corpus_checkpoint_path_names_its_producers_best_file() -> None:
    path = REPO / "scripts" / "migrations" / "drain_checkpoint_paths_v1.py"
    spec = importlib.util.spec_from_file_location("drain_checkpoint_paths_v1", path)
    drain = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drain)
    assert drain.check() == []  # tracked files only


@pytest.mark.parametrize("path", DELETED)
def test_the_retired_modules_stay_deleted(path: str) -> None:
    assert not (REPO / path).exists()


@pytest.mark.parametrize("path", sorted(ALLOWLIST))
def test_every_allowlisted_label_is_still_needed(path: str) -> None:
    """A stale permission is a hole a new violation of that label could hide in."""
    permitted, reason = ALLOWLIST[path]
    assert reason and permitted
    assert permitted <= set(scan_text((REPO / path).read_text())), path


def test_a_foreign_needle_in_an_allowlisted_file_is_red() -> None:
    path = "src/spectramr/models/classifiers/guidance_free_classifier.py"
    text = "torch.save({'model_state_dict': sd}, p)\nCheckpointService(cfg)\n"
    assert code_offenders({path: text}) == {path: ["CheckpointService"]}
    assert code_offenders({path: "torch.save({'model_state_dict': sd}, p)"}) == {}


@pytest.mark.parametrize("path", sorted(ALLOWLIST))
def test_no_allowlist_entry_permits_every_needle(path: str) -> None:
    assert ALLOWLIST[path][0] < set(CODE_NEEDLES)


def test_every_listed_path_names_a_file() -> None:
    """A C-quoted name names no file and fails the suffix filter, so it is skipped silently.

    Checked on the raw listing, before the suffix filter can drop it.
    """
    listed = tracked("src", "scripts", "runners", "experiments", "tests/fixtures/configs")
    assert [f for f in listed if not os.path.lexists(REPO / f)] == []


def test_the_corpus_scan_reaches_top_level_files() -> None:
    """Derived from the disk and one unfiltered listing, not from ``tracked``."""
    listing = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    everything = set(parse_ls_files(listing))
    top = {
        f"experiments/{p.name}"
        for p in (REPO / "experiments").iterdir()
        if p.suffix in YAML_SUFFIXES and f"experiments/{p.name}" in everything
    }
    if not top:
        pytest.skip("no top-level corpus YAML is tracked")
    assert top <= set(corpus_yamls())


def test_experiment_shell_scripts_are_scanned() -> None:
    shells = [f for f in tracked("experiments") if f.endswith((".sh", ".sbatch"))]
    if not shells:
        pytest.skip("no experiments/ shell script is tracked")
    assert set(shells) <= set(code_files())


def test_an_unstaged_deletion_is_not_listed(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "src").mkdir()
    for name in ("kept.py", "gone.py"):
        (tmp_path / "src" / name).write_text("x = 1\n")
    git("add", "src")
    (tmp_path / "src" / "gone.py").unlink()
    assert git_ls("--", "src", cwd=tmp_path) == ["src/gone.py", "src/kept.py"]
    assert tracked("src", cwd=tmp_path) == ["src/kept.py"]
    assert code_offenders(read_all(code_files(tmp_path), tmp_path)) == {}


def test_ls_files_names_survive_quotes_and_non_ascii() -> None:
    out = 'experiments/a/"q"_x.yaml\0experiments/b/café.yaml\0'
    assert parse_ls_files(out) == ['experiments/a/"q"_x.yaml', "experiments/b/café.yaml"]


# Planted violations: one per needle, one per shape it can take (NN15).
PLANTS: dict[str, tuple[str, ...]] = {
    "checkpoint_best": (
        'p = run / "checkpoints" / "checkpoint_best.pt"',
        'p = run / "checkpoint_best.pth"',
        "name = 'checkpoint_best.safetensors'",
        "# the trainer writes checkpoint_best.pt",
    ),
    "bare best.pt": (
        'ckpt = f"{out}/checkpoints/best.pt"',
        'name = "best.pt"',
        "--model best.pt --input d/",
        "``best.pt``",
    ),
    "checkpoint_epoch_": ('name = f"checkpoint_epoch_{e:04d}_step_{s:06d}.pt"',),
    "checkpoint_last": ('name = "checkpoint_last.pth"', "glob('checkpoint_last.*')"),
    "checkpoint_step_": (
        'name = f"checkpoint_step_{s}.safetensors"',
        "checkpoints/checkpoint_step_10.pt",
    ),
    "best_ema": ('SIDE = "checkpoint" + "_best_ema.pt"', "r/checkpoints/best_ema.safetensors"),
    "PERIODIC_CHECKPOINT_RE": ("from x import PERIODIC_CHECKPOINT_RE",),
    "periodic_checkpoint_name/tag": (
        "tag = periodic_checkpoint_tag(e, s)",
        "periodic_checkpoint_name(e, s)",
    ),
    "discover_best_checkpoint": ("ckpt = discover_best_checkpoint(d)",),
    "find_latest_checkpoint": ("svc.find_latest_checkpoint(d)",),
    "latest_periodic_checkpoint": ("latest_periodic_checkpoint(d)",),
    "scan_periodic_checkpoints": ("scan_periodic_checkpoints(d).ignored",),
    "CHECKPOINT_STATE_KEYS": (
        "for k in CHECKPOINT_STATE_KEYS: pass",
        "from spectramr.core.module_utils import CHECKPOINT_STATE_KEYS",
    ),
    "resolve_state_dict": ("sd = resolve_state_dict(ckpt)", "from m import resolve_state_dict"),
    "CheckpointDirector": (
        "CheckpointDirector(config).save()",
        "the ``CheckpointDirector`` writes",
        "_CheckpointDirector(cfg)",
        "CheckpointDirectorFactory()",
    ),
    "CheckpointService": (
        "svc = CheckpointService(cfg)",
        "resolve_service(CheckpointService)",
        "_CheckpointService(cfg)",
        "CheckpointServiceFactory()",
    ),
    "ICheckpointManager": (
        "class M(ICheckpointManager): pass",
        "class X(_ICheckpointManager): pass",
    ),
    "CheckpointingMixin": (
        "class S(CheckpointingMixin): pass",
        "class S(_CheckpointingMixin): pass",
    ),
    "sync_ema_sidecar": ("sync_ema_sidecar(d, s, m, supported=True)",),
    "_publish_best_alias": ("_publish_best_alias(p)",),
    "quoted envelope key": (
        "sd = state['model_state_dict']",
        'sd = state["model_state_dict"]',
        "sd = state['generator_state_dict']",
        'sd = state["ema_state_dict"]',
    ),
    "envelope f-string prefix": ('k = f"ema_state_dict_{name}"', "k = f'generator_state_dict_{n}'"),
}
_PLANT_CASES = [(label, text) for label, texts in sorted(PLANTS.items()) for text in texts]


@pytest.mark.parametrize(("label", "text"), _PLANT_CASES)
def test_each_needle_turns_the_gate_red(label: str, text: str) -> None:
    assert label in scan_text(text)


@pytest.mark.parametrize(
    "text",
    [
        "x = ckpt.with_name('best.pth')",
        "p = layout.best_path('pt')",
        "class ICheckpointService: pass",
        "class G(GradientCheckpointingMixin): pass",
        'state["models"]["generator"]',
        "checkpoints/best.safetensors",
        "checkpoints/step_00000010.pt",
        "my_best.pt",
        "state['model_state']",
    ],
)
def test_v1_spellings_stay_green(text: str) -> None:
    assert scan_text(text) == []


_YAML_CASES = [(label, text) for label, text in _PLANT_CASES if label in YAML_NEEDLES]


@pytest.mark.parametrize(("label", "text"), _YAML_CASES)
def test_each_yaml_needle_turns_the_gate_red(label: str, text: str) -> None:
    assert label in scan_yaml_text(f"model:\n  checkpoint_path: {text}\n")


def test_best_pt_is_legal_in_yaml() -> None:
    assert scan_yaml_text("model:\n  checkpoint_path: r/checkpoints/best.pt\n") == []


def test_the_needles_and_the_plants_cover_each_other() -> None:
    assert set(PLANTS) == set(CODE_NEEDLES)
    assert all(PLANTS.values())
