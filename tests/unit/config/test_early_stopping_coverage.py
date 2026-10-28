"""Early Stopping Schema Coverage Tests.

PURPOSE:
Verifies that EarlyStoppingConfigSchema fields are actually consumed by
the EarlyStoppingService.
"""

from __future__ import annotations

import inspect

import pytest

from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema
from spectramr.infrastructure.services.early_stopping import EarlyStoppingService


class TestEarlyStoppingSchemaCoverage:
    """Ensure EarlyStoppingService reads its config parameters."""

    # Fields that EarlyStoppingService actively reads in its source code
    EXPECTED_CONSUMED_FIELDS = {
        "enabled",
        "patience",
        "patience_min_iterations",
        "min_delta",
        "metric",
        "mode",
    }

    # Technical debt: Schema fields defined but never referenced by the service
    KNOWN_SCHEMA_ONLY_FIELDS = {
        "restore_best_weights",  # Defined in schema but service does not implement this behavior
    }

    def test_all_fields_classified(self):
        """All EarlyStoppingConfigSchema fields must be either consumed or documented as schema-only."""
        schema_fields = set(EarlyStoppingConfigSchema.model_fields.keys())
        all_known = self.EXPECTED_CONSUMED_FIELDS | self.KNOWN_SCHEMA_ONLY_FIELDS
        unclassified = schema_fields - all_known

        assert not unclassified, (
            f"Found unclassified EarlyStopping config fields: {unclassified}\n"
            "Either add them to EXPECTED_CONSUMED_FIELDS (if used) or KNOWN_SCHEMA_ONLY_FIELDS."
        )

    # NB: sort the set before parametrizing. A bare set yields a
    # PYTHONHASHSEED-dependent iteration order, so each pytest-xdist worker
    # (a separate process with its own hash seed) collects these cases in a
    # different order. xdist then aborts the whole run with "Different tests
    # were collected between gw0 and gwN" (with an *empty* set-difference,
    # since only the order differs). sorted() makes collection deterministic.
    @pytest.mark.parametrize("field_name", sorted(EXPECTED_CONSUMED_FIELDS))
    def test_consumed_fields_in_source(self, field_name: str):
        """Verify that expected consumed fields are actually in the service source code."""
        src = inspect.getsource(EarlyStoppingService)
        assert field_name in src, (
            f"EarlyStoppingService does not reference '{field_name}'. "
            "Has the implementation changed?"
        )

    def test_known_schema_only_fields_not_in_source(self):
        """Verify schema-only fields remain unused to prevent silent implementation drift."""
        src = inspect.getsource(EarlyStoppingService)
        for field in self.KNOWN_SCHEMA_ONLY_FIELDS:
            if field in src:
                pytest.fail(
                    f"Field '{field}' is now referenced in EarlyStoppingService! "
                    "Remove from KNOWN_SCHEMA_ONLY_FIELDS and add to EXPECTED_CONSUMED_FIELDS."
                )

    def test_schema_only_restore_best_weights(self):
        """Document the restore_best_weights gap."""
        src = inspect.getsource(EarlyStoppingService)
        if "restore_best_weights" not in src:
            pytest.xfail(
                reason="KNOWN GAP: EarlyStoppingConfigSchema.restore_best_weights is true by default "
                "but EarlyStoppingService does not implement weight restoration. "
                "Currently, best weights must be restored manually from the checkpoint service."
            )

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
