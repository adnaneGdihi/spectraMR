"""Which validation metrics are averaged on the reciprocal scale.

A leaf module so ``pipelines/train.py`` can ask without importing the reporting package.
"""

#: Metrics averaged on the reciprocal scale (``val_`` prefix stripped).
RECIPROCAL: frozenset[str] = frozenset({"nex_equivalent"})


def is_reciprocal(column: str) -> bool:
    return column.lower().removeprefix("val_") in RECIPROCAL
