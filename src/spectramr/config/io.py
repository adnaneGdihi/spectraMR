"""Raw YAML I/O helpers for code paths that must mutate config dicts.

The repo's SSOT for training configuration is
:class:`spectramr.config.settings.TrainingSettings` (frozen Pydantic v2). For
training and inference, code must consume the already-parsed
``TrainingSettings`` — never re-parse the YAML downstream
(`TODO/backlog_ssot_and_layering_cleanup.md` Phase 1).

But there's a small class of build-time tools that legitimately need
the *raw, mutable* dict — they generate new YAMLs from a template,
not consume one for training. Examples:

- :mod:`spectramr.infrastructure.orchestration.ablation_config_generator` —
  reads a base YAML, applies axis overrides, dumps N new YAMLs.
- HPO sub-process spawners — each subprocess legitimately loads its
  own YAML (parallel entry points).

Those tools route through :func:`read_yaml_dict` here rather than
calling ``yaml.safe_load`` directly. Keeping the helper inside
``src/config/`` means the ``tests/unit/test_no_yaml_reparse.py``
guard still passes (config/ is the allowed subtree for raw YAML I/O).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError


class DuplicateKeyError(ConstructorError):
    """A mapping in a config YAML repeats a key; PyYAML would keep the last one silently."""


class StrictSafeLoader(yaml.SafeLoader):
    """``SafeLoader`` that raises on a repeated mapping key at any depth.

    Each mapping is checked once, as written, before merging rewrites it in place: an
    override of a merged key (``<<: *anchor`` then the key) stays legal at any depth.
    Unhashable keys are left to PyYAML's own ``ConstructorError``.
    """

    def __init__(self, stream: Any) -> None:
        super().__init__(stream)
        self._checked: set[int] = set()

    def flatten_mapping(self, node: yaml.MappingNode) -> None:
        if id(node) not in self._checked:
            self._checked.add(id(node))
            self._reject_duplicates(node)
        super().flatten_mapping(node)

    def _reject_duplicates(self, node: yaml.MappingNode) -> None:
        seen: dict[Any, int] = {}
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node)
            line = key_node.start_mark.line + 1
            try:
                repeated = key in seen
            except TypeError:
                continue  # unhashable: construct_mapping raises PyYAML's error for it
            if repeated:
                raise DuplicateKeyError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r} on line {line} (first on line {seen[key]})",
                    key_node.start_mark,
                )
            seen[key] = line


def load_yaml_strict(stream: Any) -> Any:
    """Parse YAML like ``yaml.safe_load`` but raise :class:`DuplicateKeyError` on a repeated key."""
    return yaml.load(stream, Loader=StrictSafeLoader)


def read_yaml_dict(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return the raw mutable dict.

    Use only when the caller intends to mutate the result or dump it
    to a new YAML. For training/inference, prefer
    :meth:`spectramr.config.settings.TrainingSettings.from_yaml`.

    Args:
        path: YAML file path.

    Returns:
        Parsed dict (or empty dict if the file is empty).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {p}")
    with p.open() as f:
        data = load_yaml_strict(f)
    return data or {}


def write_yaml_dict(data: dict[str, Any], path: str | Path) -> Path:
    """Dump a dict to a YAML file with stable ordering.

    Args:
        data: Mapping to serialise.
        path: Output path; parent directory created on demand.

    Returns:
        Absolute path of the written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return out


__all__ = [
    "DuplicateKeyError",
    "StrictSafeLoader",
    "load_yaml_strict",
    "read_yaml_dict",
    "write_yaml_dict",
]
