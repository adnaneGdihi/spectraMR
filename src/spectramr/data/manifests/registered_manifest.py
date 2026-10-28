"""Check that a manifest of a registered M4Raw mirror really serves the registered files.

``contrast_registration.register_split`` writes a mirror of each split directory in
which the reference contrast is registered and every other file is a symlink to the
original. A manifest of that mirror is only useful if the loader, reading it, reaches
the registered copies: one that still reaches an original reference trains on the
unaligned file, and nothing downstream can tell.

The check therefore goes through the loader's own resolver, ``parse_fastmri_index``:
every record must resolve to an existing file, and every registered reference to its
copy inside the mirror. The other files legitimately resolve to the originals, since
the resolver follows the mirror's symlinks.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

from spectramr.data.datasets.universal_dataset import parse_fastmri_index


def verify_registered_manifest(manifest: Path, mirror: Path, registered: Collection[str]) -> int:
    """Raise unless ``manifest`` resolves ``registered`` (``<stem>.h5``) inside ``mirror``.

    Returns the number of records. A manifest holding no registered reference at all is
    refused too: it indexes the wrong tree, or a split nothing was registered in.
    """
    root = mirror.resolve()
    paths = [Path(r["primary_path"]) for r in parse_fastmri_index(str(manifest))]
    ours = [p for p in paths if p.name in registered]
    stray = [
        p
        for p in paths
        if not p.exists() or (p.name in registered and not p.resolve().is_relative_to(root))
    ]
    if stray or not ours:
        shown = "\n  ".join(map(str, stray[:5])) or "(no registered reference in it)"
        raise ValueError(
            f"{manifest}: {len(stray)} of {len(paths)} records are missing, or are a "
            f"registered reference resolving outside {mirror}:\n  {shown}"
        )
    return len(paths)
