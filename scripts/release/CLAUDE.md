# Local rules: `scripts/release/`

This is the publish lane. Non-negotiable 21 governs it: **nothing reaches the public repository except
through the export script.**

- **Two write paths exist and only two** — `public_allowlist.txt` (fail-closed: a forgotten allowance
  loses a file, a forgotten denial publishes one) and the overlay under `public_overlay/`. **Adding a
  third is the change that makes the export unreviewable.**
- **Assert before exporting.** The exporting branch must carry the current tooling; pin the
  *mechanism* (`def read_overlay`), never a constant — a guard pinned to the old `OVERLAY_PREFIX` name
  reported STOP on correct branches, and a guard that cannot pass stops being run.
- **Publishing replaces the tree.** `git rm -r .` first; overlaying a new export on the old checkout
  leaves denied paths published forever and nothing reports it.
- **The repo has been public since 2026-08-26** — a bad export is fixed *before* it ships, never after.
- The four settings files are denied from the export on purpose: they are the export's **policy**, not
  its product.

**Do not run the procedure from memory.** → `.claude/rules/publishing.md` (the full lane and the seven
ways it has silently gone wrong), `docs/versioning.rst`, `docs/contributing/public_export.rst`
