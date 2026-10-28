"""The scanner behind ``test_no_legacy_checkpoint_names``: needles, allowlist, file listing.

It reads **tracked (indexed) files only**: a file that is not yet ``git add``-ed is not
scanned, so a local run before staging can be falsely green. CI and committed trees see
every file.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Class needles take a left boundary that admits ``_`` (``_CheckpointDirector``) but
# not a letter (``ICheckpointService``, ``GradientCheckpointingMixin``), and no right
# boundary (``CheckpointDirectorFactory``).
_CLASS = r"(?<![A-Za-z0-9])"
CODE_NEEDLES: dict[str, str] = {
    "checkpoint_best": r"checkpoint_best",
    "bare best.pt": r"(?<![\w])best\.pt\b",
    "checkpoint_epoch_": r"checkpoint_epoch_",
    "checkpoint_last": r"checkpoint_last",
    "checkpoint_step_": r"checkpoint_step_",
    "best_ema": r"best_ema",
    "PERIODIC_CHECKPOINT_RE": r"PERIODIC_CHECKPOINT_RE",
    "periodic_checkpoint_name/tag": r"periodic_checkpoint_(?:name|tag)",
    "discover_best_checkpoint": r"discover_best_checkpoint",
    "find_latest_checkpoint": r"find_latest_checkpoint",
    "latest_periodic_checkpoint": r"latest_periodic_checkpoint",
    "scan_periodic_checkpoints": r"scan_periodic_checkpoints",
    "CHECKPOINT_STATE_KEYS": r"CHECKPOINT_STATE_KEYS",
    "resolve_state_dict": r"resolve_state_dict",
    "CheckpointDirector": _CLASS + r"CheckpointDirector",
    "CheckpointService": _CLASS + r"CheckpointService",
    "ICheckpointManager": _CLASS + r"ICheckpointManager",
    "CheckpointingMixin": _CLASS + r"CheckpointingMixin",
    "sync_ema_sidecar": r"sync_ema_sidecar",
    "_publish_best_alias": r"_publish_best_alias",
    "quoted envelope key": r"""["'](?:model_state_dict|generator_state_dict|ema_state_dict)["']""",
    "envelope f-string prefix": r"""f["'](?:model_state_dict|generator_state_dict|ema_state_dict)_""",
}
YAML_LABELS = (
    "checkpoint_best",
    "checkpoint_epoch_",
    "checkpoint_step_",
    "checkpoint_last",
    "best_ema",
)
YAML_NEEDLES: dict[str, str] = {k: CODE_NEEDLES[k] for k in YAML_LABELS}
DELETED = (
    "src/spectramr/infrastructure/builders/directors/checkpoint_director.py",
    "src/spectramr/infrastructure/builders/directors/checkpoint_retention.py",
    "src/spectramr/infrastructure/services/checkpoint_service.py",
    "src/spectramr/infrastructure/services/persistence/checkpoint_delegate.py",
    "src/spectramr/infrastructure/distributed/best_ema_sidecar.py",
)

_CONVERTER = "the pre-1.0 converter is the one owner of the retired names; it must read them"
_OWN_FORMAT = "a self-contained script format nothing in src reads"
_ENVELOPE = frozenset({"quoted envelope key"})
#: path -> (the labels it may spell, why). Any other needle in the file is still a finding.
ALLOWLIST: dict[str, tuple[frozenset[str], str]] = {
    "scripts/migrations/convert_checkpoints_v1.py": (
        frozenset({"checkpoint_best", "checkpoint_epoch_", "checkpoint_last", "checkpoint_step_"}),
        f"{_CONVERTER} (entry point: the renames)",
    ),
    "scripts/migrations/_checkpoint_v1_envelopes.py": (
        frozenset({"checkpoint_best", "quoted envelope key"}),
        f"{_CONVERTER} (envelope mapper)",
    ),
    "scripts/migrations/_checkpoint_v1_load.py": (
        frozenset({"CheckpointService", "best_ema", "checkpoint_best", "quoted envelope key"}),
        f"{_CONVERTER} (pre-1.0 reader: the service's safetensors shape, the EMA sidecar)",
    ),
    "scripts/migrations/_checkpoint_v1_plan.py": (
        frozenset({"checkpoint_best"}),
        f"{_CONVERTER} (run classifier)",
    ),
    "scripts/migrations/_checkpoint_v1_state.py": (
        frozenset({"checkpoint_last", "quoted envelope key"}),
        f"{_CONVERTER} (file to RunState)",
    ),
    "scripts/migrations/drain_checkpoint_paths_v1.py": (
        frozenset({"best_ema", "checkpoint_best"}),
        "matches pre-1.0 names in the corpus to rewrite them",
    ),
    "src/spectramr/infrastructure/checkpointing/payload.py": (
        _ENVELOPE,
        "recognises a pre-1.0 envelope only to tell it from a foreign file; never reads one",
    ),
    "src/spectramr/models/classifiers/guidance_free_classifier.py": (
        _ENVELOPE,
        "its own classifier file, written and read by itself, not a training checkpoint",
    ),
    "scripts/physics/pretrain_qmap_generator.py": (_ENVELOPE, _OWN_FORMAT),
    "scripts/training/latent_diffusion_pipeline.py": (_ENVELOPE, _OWN_FORMAT),
}
_CODE_SUFFIXES = (".py", ".sh", ".sbatch", ".yaml", ".yml")
_EXPERIMENT_CODE_SUFFIXES = (".py", ".sh", ".sbatch")
YAML_SUFFIXES = (".yaml", ".yml")
_CODE_RE = {label: re.compile(rx) for label, rx in CODE_NEEDLES.items()}
_YAML_RE = {label: re.compile(rx) for label, rx in YAML_NEEDLES.items()}


def scan_text(text: str) -> list[str]:
    return sorted(label for label, rx in _CODE_RE.items() if rx.search(text))


def scan_yaml_text(text: str) -> list[str]:
    return sorted(label for label, rx in _YAML_RE.items() if rx.search(text))


def code_offenders(
    texts: Mapping[str, str],
    allowlist: Mapping[str, tuple[frozenset[str], str]] = ALLOWLIST,
) -> dict[str, list[str]]:
    """``path -> labels`` found, less the labels that path's allowlist entry permits."""
    found = {}
    for path, text in texts.items():
        permitted = allowlist.get(path, (frozenset(), ""))[0]
        if hits := [h for h in scan_text(text) if h not in permitted]:
            found[path] = hits
    return found


def parse_ls_files(out: str) -> list[str]:
    """Split ``git ls-files -z`` output; without ``-z`` git C-quotes unusual names."""
    return [name for name in out.split("\0") if name]


def git_ls(*args: str, cwd: Path = REPO) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout
    return parse_ls_files(out)


def tracked(*roots: str, cwd: Path = REPO) -> list[str]:
    # Directory roots, never a ``**`` glob: in git's default pathspec mode
    # ``experiments/**/*.yaml`` needs two slashes and skips top-level files. A tracked
    # file deleted from the working tree but not yet staged is not there to scan.
    deleted = set(git_ls("--deleted", "--", *roots, cwd=cwd))
    return [f for f in git_ls("--", *roots, cwd=cwd) if f not in deleted]


def code_files(cwd: Path = REPO) -> list[str]:
    files = [f for f in tracked("src", "scripts", "runners", cwd=cwd) if f.endswith(_CODE_SUFFIXES)]
    return files + [
        f for f in tracked("experiments", cwd=cwd) if f.endswith(_EXPERIMENT_CODE_SUFFIXES)
    ]


def corpus_yamls(cwd: Path = REPO) -> list[str]:
    return [
        f
        for f in tracked("experiments", "tests/fixtures/configs", cwd=cwd)
        if f.endswith(YAML_SUFFIXES)
    ]


def read_all(
    paths: Iterable[str], cwd: Path = REPO, *, reader: Callable[[Path], str] | None = None
) -> dict[str, str]:
    read = reader or (lambda p: p.read_text(errors="replace"))
    return {f: read(cwd / f) for f in paths}
