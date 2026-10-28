"""Anti-rot for the shared ``data:`` config stand-in.

A stand-in only earns trust while it matches the schema. These tests resolve
every name it claims against the live schema, so the stub cannot quietly drift
into a shape nothing produces -- which is the exact failure it exists to prevent
(31 tests stayed green over phase-9a-broken code because the stand-ins agreed
with the broken guards).
"""

from __future__ import annotations

import pytest

from tests.utils.data_config_stub import (
    FLAT_TO_CANONICAL,
    DataConfigStub,
    nested_blocks,
)


def test_every_block_is_a_real_schema_instance() -> None:
    """The stub must build blocks from the schema, never hand-write them."""
    import spectramr.config.schemas.data as data_mod
    from spectramr.config.schemas.data import DataConfigSchema

    blocks = nested_blocks()
    assert blocks, "the stub declares no sub-blocks"
    for name, instance in blocks.items():
        assert name in DataConfigSchema.model_fields, (
            f"the stub supplies a `{name}` block the real DataConfigSchema does "
            "not mount -- it is a shape no config produces"
        )
        declared = DataConfigSchema.model_fields[name].annotation
        assert isinstance(instance, declared), (
            f"stub `{name}` is {type(instance).__name__}, schema declares "
            f"{getattr(declared, '__name__', declared)}"
        )
        # The class schemas.data mounts (some, e.g. augmentation, live in a sibling module).
        assert getattr(data_mod, type(instance).__name__, None) is type(instance)


def test_the_stub_covers_every_sub_block_the_schema_mounts() -> None:
    """A block added by a later phase must be added here, not discovered later.

    Without this, the next tranche's sub-block would simply be absent from the
    stub and every test using it would fail with ``no attribute`` -- the same
    hunt this module exists to end.
    """
    from pydantic import BaseModel

    from spectramr.config.schemas.data import DataConfigSchema

    mounted = {
        name
        for name, f in DataConfigSchema.model_fields.items()
        if isinstance(f.annotation, type) and issubclass(f.annotation, BaseModel)
    }
    # Component blocks that predate the phase-9 decomposition are out of scope:
    # they were always nested, so no stand-in ever carried them flat.
    phase9 = set(nested_blocks())
    missing = {
        n
        for n in mounted
        if n
        in {
            "loader",
            "coils",
            "expose",
            "mrixfields",
            "split",
            "source",
            "sampling",
            "processing",
            "domain",
            # named `pairing`, not the plan's `contrast` -- see
            # DataPairingConfigSchema's docstring. A stale name here does not
            # fail; it makes `missing` empty and the check vacuous.
            "pairing",
        }
    } - phase9
    assert not missing, (
        f"DataConfigSchema mounts phase-9 sub-block(s) {sorted(missing)} that "
        "the stub does not supply -- add them to _BLOCK_SCHEMAS"
    )


@pytest.mark.parametrize("flat", sorted(FLAT_TO_CANONICAL))
def test_every_flat_alias_resolves_to_a_real_field(flat: str) -> None:
    """``FLAT_TO_CANONICAL`` must point at fields that exist."""
    block, leaf = FLAT_TO_CANONICAL[flat]
    blocks = nested_blocks()
    assert block in blocks, f"{flat} routes to unknown block {block!r}"
    assert leaf in type(blocks[block]).model_fields, (
        f"{flat} routes to {block}.{leaf}, which is not a field on "
        f"{type(blocks[block]).__name__}"
    )


def test_flat_aliases_agree_with_the_rename_ssot() -> None:
    """Where a fold record exists, the stub must route to the SAME place.

    Two tables that disagree is the problem the rename SSOT was created to
    remove; a test-only copy is no more exempt than a production one.
    """
    from spectramr.config.schemas.renames import RENAMES

    for legacy, rec in RENAMES.items():
        if rec.posture != "fold" or not rec.canonical.startswith("data."):
            continue
        leaf = legacy.split(".")[-1]
        if leaf not in FLAT_TO_CANONICAL:
            continue  # the stub need not cover every record, only agree
        block, target = FLAT_TO_CANONICAL[leaf]
        assert rec.canonical == f"data.{block}.{target}", (
            f"stub routes {leaf!r} to data.{block}.{target}, but the rename "
            f"SSOT folds it to {rec.canonical}"
        )


def test_a_flat_kwarg_lands_on_the_canonical_path() -> None:
    cfg = DataConfigStub(batch_size=2, split_strategy="random", validation_split=0.3)
    assert cfg.loader.batch_size == 2
    assert cfg.split.type == "random"
    assert cfg.split.validation_fraction == 0.3
    # ...and does NOT survive as a second, flat spelling.
    assert not hasattr(cfg, "batch_size")
    assert not hasattr(cfg, "split_strategy")


def test_unmapped_kwargs_pass_through() -> None:
    cfg = DataConfigStub(dataset_type="kspace")
    assert cfg.dataset_type == "kspace"


def test_a_mapped_flat_kwarg_lands_canonically_and_not_flat() -> None:
    """`data_root` IS mapped (``FLAT_TO_CANONICAL`` -> ``source.root``).

    The test above used to pass it and assert ``cfg.data_root == "/tmp"``, which
    is the one thing the stub deliberately refuses to do: routing rather than
    also setting the flat name is what stops a stand-in growing a second
    spelling that silently disagrees with the one the reader walks.
    """
    cfg = DataConfigStub(data_root="/tmp")
    assert cfg.source.root == "/tmp"
    assert not hasattr(cfg, "data_root")


def test_defaults_come_from_the_schema_not_from_literals() -> None:
    """Pins rule 1: restating a default is how `int(None)` happened."""
    from spectramr.config.schemas.data import (
        DataLoaderConfigSchema,
        MRIxFieldsDataConfigSchema,
    )

    cfg = DataConfigStub()
    assert (
        cfg.loader.prefetch_factor
        == DataLoaderConfigSchema.model_fields["prefetch_factor"].default
    )
    assert (
        cfg.mrixfields.max_resident_volumes
        == MRIxFieldsDataConfigSchema.model_fields["max_resident_volumes"].default
    )
