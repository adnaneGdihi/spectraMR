# Local rules: `tests/`

- **Source↔test pairing is per-commit** (non-negotiable 10): `src/spectramr/<area>/<mod>.py` ↔
  `tests/unit/<area>/test_<mod>.py`, enforced by `scripts/ci/check_test_paired_with_source.py`.
- **A detector ships with planted violations that turn it red** (non-negotiable 15) — one per rule it
  claims to enforce, **and one per shape that rule can take** (top-of-file *and* function-local, direct
  name *and* subclass). Committed as tests, not run once by hand.
- **A baseline regeneration must not raise a ceiling** (non-negotiable 20). `tests/architecture/baselines/`
  keys on identity alone, so growth inside an already-baselined file is invisible; read the baseline diff
  and reject any entry whose measurement went *up*. Never regenerate to make a gate green.
- Markers, `filterwarnings` and the conftest torch-shim are declared in `pyproject.toml` /
  `tests/conftest.py`.

Run one test: `pytest tests/unit/<area>/test_<mod>.py -v` (or `::test_name`).

→ `.agent/rules/unit-test.md`, `.agent/rules/detectors.md`, `.claude/rules/change-discipline.md`
