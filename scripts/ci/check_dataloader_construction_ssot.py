"""Pre-commit / PR gate: a ``DataLoader`` is constructed in ONE place.

Non-negotiable #7 (data SSOT): File->Dataset/DataLoader code lives under
``src/spectramr/data/``; everything else consumes via ``DataPipelineDirector`` and
never calls ``DataLoader(...)`` itself. Before the 2026-08 data-layer audit
(finding D22) there were four construction sites under ``src/spectramr/``, and they
had drifted apart in exactly the way duplicated construction always does: one read
a plain ``dict`` instead of the frozen config, two omitted the per-worker
``worker_init_fn`` seeding, and three hardcoded ``prefetch_factor`` /
``persistent_workers`` past the schema fields that exist to set them.

Usage:
    python scripts/ci/check_dataloader_construction_ssot.py

Exit codes:
    0 -- every loader construction under ``src/spectramr/`` is allow-listed.
    1 -- a new construction site appeared, or an allow-list entry went stale.

Why AST and not grep
--------------------
A line scanner for ``DataLoader(`` is wrong in both directions, and the count IS
the gate. **False positives**: ``class IDataLoader(ABC)`` is a class definition
and ``consolidated_dataset_factory.py`` names the symbol inside a docstring
mermaid diagram -- neither constructs anything. **False negatives**: the symbol
is reachable under many spellings (bare, fully-qualified, and the aliased
``_DataLoader(`` in ``data/builders/data_loader_builder.py``); a pattern
enumerating two of the three reports "clean" while the third walks past.

Why subclasses, and why a generated vocabulary (#1362)
------------------------------------------------------
Until 2026-08 this gate matched the *name* ``DataLoader``, so ``tio.SubjectsLoader``
-- a real ``DataLoader`` subclass, four construction sites under ``data/`` -- was
invisible, and with it the one site that never seeds its workers. Fixing that by
adding ``SubjectsLoader`` to a hardcoded list would have re-committed the exact
mistake this docstring warns about one paragraph up: the vocabulary would again
be an enumeration of the spellings someone happened to think of. It is instead
**discovered** -- ``refresh_dataloader_binding_names.py`` imports the repo's
dependencies, walks the transitive ``DataLoader.__subclasses__()`` closure, and
records every attribute name those classes are reachable under, into
``dataloader_binding_names.txt``. Reading that file needs no torch, so the gate
still fits the ``guards`` job's dependency-free contract (checkout +
setup-python, no pip install).

That the vocabulary must be discovered is not a hypothetical. Hand-reasoning it
produced ``{DataLoader, SubjectsLoader, ThreadDataLoader}``; discovery returns
five, because monai re-exports the torch loader as ``TorchDataLoader`` and
``_TorchDataLoader``, names that appear in no class ``__name__``.

Known false-positive mode, deliberately kept
--------------------------------------------
An attribute call whose head cannot be resolved (``anything.DataLoader(...)``)
matches on the final name alone. The corpus has no such site today, and the
polarity is the one this repo wants: loud-wrong beats silently-blind.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

#: The generated vocabulary this gate matches against. Lives beside the gate and
#: is rewritten by ``refresh_dataloader_binding_names.py`` -- which imports THIS
#: module for the reader, never the reverse: the gate must keep zero sibling
#: imports so that a bare ``python scripts/ci/<gate>.py`` in the dependency-free
#: ``guards`` job cannot fail on a path-resolution detail.
VOCABULARY_FILENAME = "dataloader_binding_names.txt"
_VOCABULARY_PATH = Path(__file__).resolve().parent / VOCABULARY_FILENAME


def read_vocabulary(path: Path | None = None) -> set[str]:
    """Parse the generated vocabulary; RAISE rather than default on absent/empty.

    A missing file must not fall back to ``{"DataLoader"}``: that silently
    restores the pre-#1362 blind spot and prints "OK" while doing it. An unknown
    vocabulary is a state to report, never one to infer (non-negotiables #3, #18).
    """
    path = path or _VOCABULARY_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Regenerate it with "
            f"`python scripts/ci/refresh_dataloader_binding_names.py`."
        )
    names = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not names:
        raise ValueError(f"{path} is empty; the gate would match nothing.")
    return names


#: The sanctioned construction sites, as ``path -> ((enclosing function, reason), ...)``.
#:
#: Several per path, because a builder legitimately constructs in more than one
#: method (train vs. val queue). The tuple is the unit the staleness check keys
#: on: an entry that matches nothing is reported, so a rename or a deletion
#: cannot leave a permanently-satisfied exemption behind.
#:
#: The first entry sits under ``infrastructure/`` while non-negotiable #7 names
#: ``data/``. That is a KNOWN, deferred tension, not a silent exemption: the leaf
#: builder is where the live train/val/inference path actually constructs, and
#: moving it is a file move, which the data-layer plan sequences LAST (unit 6d)
#: precisely because it relocates code other units are still registering. Until
#: then this allow-list encodes today's truth rather than pretending otherwise.
#: Keys are ``Class.method`` where the site sits in a class, and a bare function
#: name where it does not -- the spelling :func:`_qualified_spans` produces. A bare
#: method name here would sanction that name on EVERY class in the file (``D05#8``).
_ALLOWED: dict[str, tuple[tuple[str, str], ...]] = {
    "src/spectramr/infrastructure/builders/leaf/data_builders.py": (
        (
            "DataLoaderBuilder.build",
            "The single construction site for the live train / val / inference paths.",
        ),
    ),
    "src/spectramr/data/builders/data_loader_builder.py": (
        (
            "wrap_with_distributed_sampler",
            "DDP rewrap: PyTorch cannot swap a sampler onto a built loader, so the "
            "canonical pattern is to reconstruct from the source loader's own knobs.",
        ),
    ),
    # The four tio.SubjectsLoader sites this gate could not see before #1362.
    # Recorded, not forgiven -- each states WHY it is tolerable today, and the
    # one that is not says so.
    "src/spectramr/data/builders/torchio_queue_builder.py": (
        (
            "TorchIOQueueBuilder.build_train_queue",
            "SubjectsLoader over a tio.Queue with num_workers=0 hardcoded: the "
            "queue owns its own workers, so the builder's worker seeding has "
            "nothing to seed here.",
        ),
        (
            "TorchIOQueueBuilder.build_val_queue",
            "Same, val side. Constructs twice (queue-backed and dataset-backed); "
            "the dataset-backed call already passes worker_init_fn.",
        ),
    ),
    "src/spectramr/data/builders/prefetching_queue.py": (
        (
            "PrefetchingQueue._get_subjects_iterable",
            "The train queue's internal subject loader, overriding tio.Queue's own so "
            "it takes a prefetch_factor (#2673). No worker_init_fn, as in TorchIO: "
            "torch seeds random/numpy/torch per worker, and seed_worker would reseed "
            "NumPy and change existing runs' augmentation draws.",
        ),
    ),
    "src/spectramr/data/datasets/preprocessed_dataset.py": (
        (
            "create_preprocessed_dataloader",
            "KNOWN DEBT (#1362): the one loader with caller-controlled num_workers "
            "and no worker_init_fn -- per-worker seeds are left to chance. "
            "Allow-listed to make the gate honest about today, NOT to bless it; "
            "routing it through the leaf builder is the fix.",
        ),
    ),
}


def _dotted(node: ast.AST) -> str | None:
    """Flatten ``a.b.c`` attribute/name chains to ``"a.b.c"``; None if not one."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _loader_aliases(tree: ast.AST, vocabulary: set[str]) -> set[str]:
    """Local names this module bound to a loader class, via any import spelling.

    Covers ``from torch.utils.data import DataLoader``, the aliased
    ``... import DataLoader as _DataLoader``, and ``from torchio import
    SubjectsLoader``. Matching is on the ORIGINAL name, so an alias cannot hide a
    loader, while a module that never imports one contributes nothing -- a
    same-named local class still cannot trip the gate.
    """
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in vocabulary:
                    aliases.add(alias.asname or alias.name)
    return aliases


def _qualified_spans(tree: ast.AST) -> list[tuple[int, int, str]]:
    """Every function in ``tree`` as ``(start, end, qualified_name)``.

    Qualified by its enclosing **classes**, not just its own name -- ``D05#8``.
    A bare ``node.name`` makes the allow-list ambiguous exactly where it is used:
    ``infrastructure/builders/leaf/data_builders.py`` defines ``build`` on TWO
    classes, ``DatasetBuilder`` (:165) and ``DataLoaderBuilder`` (:404), and the
    entry naming ``"build"`` sanctioned both. Its own reason string already said
    ``DataLoaderBuilder.build``, so the allow-list *documented* one method and
    *exempted* two -- a loader constructed in ``DatasetBuilder.build`` would have
    been accepted in silence.

    Not a one-file quirk: **580 files under ``src/spectramr`` define a same-named
    method on two or more classes**, so any future entry naming a common verb
    (``build``, ``forward``, ``__init__``) inherits the same widening.

    Descent, not ``ast.walk``, because the prefix has to accumulate along the
    path. A function nested inside a method still reports its own name last
    (``DataLoaderBuilder.build.helper``), preserving the property the previous
    implementation was careful about: a nested helper must NOT inherit the
    allow-list entry of the method containing it.
    """
    spans: list[tuple[int, int, str]] = []

    def descend(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                descend(child, f"{prefix}{child.name}.")
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                qualified = f"{prefix}{child.name}"
                spans.append((child.lineno, child.end_lineno or child.lineno, qualified))
                descend(child, f"{qualified}.")
            else:
                # An ``if`` / ``try`` / ``with`` block does not change the qualified
                # path, but a def inside one is still a def.
                descend(child, prefix)

    descend(tree, "")
    return spans


def _enclosing_functions(tree: ast.AST) -> dict[int, str]:
    """Map every line inside a function body to its INNERMOST enclosing function.

    Resolved by span width rather than by visit order. Relying on the traversal to
    put inner defs last happens to work today, but it is the same order-dependence
    that has bitten import-time tables in this repo: a nested helper inside an
    allow-listed method must report the HELPER's name, or the allow-list silently
    widens to everything defined under it.

    Names are **class-qualified** -- see :func:`_qualified_spans`.
    """
    spans = _qualified_spans(tree)
    # Widest first, so narrower (more deeply nested) spans overwrite them.
    spans.sort(key=lambda s: s[1] - s[0], reverse=True)

    owner: dict[int, str] = {}
    for start, end, name in spans:
        for line in range(start, end + 1):
            owner[line] = name
    return owner


def find_construction_sites(
    source: str, vocabulary: set[str] | None = None
) -> list[tuple[int, str]]:
    """Every loader construction in ``source`` as ``(lineno, enclosing_func)``."""
    vocabulary = read_vocabulary() if vocabulary is None else vocabulary
    tree = ast.parse(source)
    aliases = _loader_aliases(tree, vocabulary)
    owners = _enclosing_functions(tree)

    sites: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        hit = False
        if isinstance(func, ast.Name):
            hit = func.id in aliases
        elif isinstance(func, ast.Attribute):
            hit = func.attr in vocabulary
        if hit:
            sites.append((node.lineno, owners.get(node.lineno, "<module>")))
    return sites


def find_unsanctioned(
    root: Path, vocabulary: set[str] | None = None
) -> tuple[list[str], set[tuple[str, str]]]:
    """Scan the tree; return ``(violations, allow-list entries that matched)``.

    Split from :func:`check` so the two failure modes can be planted
    independently: a violation test needs a tree containing ONLY the planted
    file, which would make every real allow-list entry look stale.
    """
    vocabulary = read_vocabulary() if vocabulary is None else vocabulary
    violations: list[str] = []
    matched: set[tuple[str, str]] = set()

    for path in sorted((root / "src" / "spectramr").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        try:
            sites = find_construction_sites(path.read_text(encoding="utf-8"), vocabulary)
        except SyntaxError as exc:  # a file that does not parse is a real problem
            violations.append(f"{rel}: could not parse ({exc})")
            continue
        allowed_funcs = {func for func, _ in _ALLOWED.get(rel, ())}
        for lineno, func in sites:
            if func in allowed_funcs:
                matched.add((rel, func))
            else:
                violations.append(
                    f"{rel}:{lineno}: loader constructed in {func!r}. "
                    "Route through the leaf DataLoaderBuilder "
                    "(infrastructure/builders/leaf/data_builders.py) instead — "
                    "it supplies collate selection, worker seeding, and the "
                    "prefetch/persistent knobs a hand-rolled call drops."
                )
    return violations, matched


def find_stale_entries(matched: set[tuple[str, str]]) -> list[str]:
    """Allow-list entries that matched no construction site.

    A sanctioned site that no longer exists is a DETECTOR regression, not a
    tidy-up: the exemption stays satisfied forever, and silently covers any
    future call site that happens to reuse the function name. This is also the
    only signal that catches a shape no plant anticipated -- if widening the
    vocabulary ever stops resolving a site the gate used to see, the entry for
    it goes stale rather than the gate going quiet.
    """
    return [
        f"{rel}: allow-list entry {func!r} matched no construction site. It was "
        "renamed, moved or deleted — drop the entry rather than leaving a "
        "permanently-satisfied exemption."
        for rel, entries in sorted(_ALLOWED.items())
        for func, _reason in entries
        if (rel, func) not in matched
    ]


def check(root: Path, vocabulary: set[str] | None = None) -> list[str]:
    """Return one violation string per unsanctioned site or stale allow-list entry."""
    violations, matched = find_unsanctioned(root, vocabulary)
    return violations + find_stale_entries(matched)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root (defaults to the checkout this script lives in).",
    )
    args = parser.parse_args()

    vocabulary = read_vocabulary()
    violations = check(args.root, vocabulary)
    if violations:
        print("DataLoader construction SSOT violated (CLAUDE.md non-negotiable #7):")
        for v in violations:
            print(f"  {v}")
        print(f"\n{len(violations)} problem(s).")
        return 1

    sanctioned = sum(len(e) for e in _ALLOWED.values())
    print(
        f"DataLoader construction SSOT: OK ({sanctioned} allow-listed site(s), "
        f"no others; {len(vocabulary)} loader binding name(s) matched against)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
