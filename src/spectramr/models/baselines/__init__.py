"""The contract every prior-method baseline adapter satisfies.

The adapters themselves -- each wrapping an upstream vendored under
``external/baselines/<name>/`` -- live out of tree in the
``spectramr_baselines`` plugin (``plugins/spectramr_baselines/``). This package
keeps only what the framework reads back from them: the declarative
attributes that ``ConfigHealthChecker`` and ``baseline_provenance`` inspect.

See ``TODO/backlog_baseline_replication_experiment_11.md`` for the
replication contract and ``BaselineAdapter`` for the declarative
attributes every adapter must expose.
"""

from spectramr.models.baselines._base import BaselineAdapter, CoilHandling, FFTNorm

__all__ = ["BaselineAdapter", "CoilHandling", "FFTNorm"]
