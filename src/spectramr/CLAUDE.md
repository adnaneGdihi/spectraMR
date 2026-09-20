# Local rules: `src/spectramr/`

You are in the package. This is where the layering, SSOT and registry rules apply literally.

- **Imports point inward only** — `cli/ → pipelines/ → application/ → infrastructure/ → models/,
  domain/ → core/, config/`. Never leftward, and **never import from `scripts/`** (non-negotiable 5).
  The layering gate is anchored at `^`, so a *function-local* upward import is invisible to it — check
  by hand.
- **Register, do not `if/elif`** (non-negotiable 6). New models go in `models/registry.py`; the registry
  is populated by `models/init_registry.py`.
- **k-space goes through `fft2c`/`ifft2c`** (`infrastructure/physics/fft_ops.py`), never raw
  `torch.fft.*` (non-negotiable 2). This is the one module `models/`, `domain/` and `core/` may import
  upward.
- **File loading lives in `data/`** (non-negotiable 7). No `torch.load`/`h5py.File`/`nib.load`/
  `DataLoader(...)` in higher layers.
- **Devices resolve through `spectramr.core.compute_device`** (non-negotiable 9b). Never
  `"cuda" if torch.cuda.is_available() else "cpu"`.
- **A new file lands under 300 LOC at inheritance depth ≤ 2** (non-negotiable 20).
- **Config paths are nested and frozen** (non-negotiable 1). Resolve an uncertain path through
  `config/schemas/renames.py`.
- **`tests/unit/<area>/test_<mod>.py` ships in the same commit** (non-negotiable 10).

→ `.claude/rules/architecture.md`, `physics.md`, `data.md`, `configuration.md`, `performance.md`
