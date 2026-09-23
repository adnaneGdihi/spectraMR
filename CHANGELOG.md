# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`dev` ships builds.** `.github/workflows/dev-publish.yml` publishes
  `X.Y.B.dev<run number>` to PyPI when a `dev-build-*` tag is pushed on the public
  repository's `dev` head, so a change is installable with `pip install --pre
  spectramr` without waiting for the next release. It is tag-triggered because a
  branch push is autonomous CI (`test_push_triggers_are_tag_only` admits a `push:`
  only when tag-scoped), while `schedule:` and `workflow_dispatch` register from
  the default branch, where this file will not be until a release export puts it
  there -- a dispatch returned 404. The `pypi` job is gated on the
  **commit**: it uploads only when the build is of public `dev`'s current head, so
  a tag anywhere else rehearses the build and uploads nothing. The counter is
  `github.run_number` because it is the only monotonic one available and PyPI
  never re-issues a filename; the lane refuses to run from a tree carrying a
  release version, since a `0.1.3.dev<n>` wheel would sort before an 0.1.3 that
  has already shipped. The file ships to both repositories -- the export
  allowlist selects `.github/` wholesale and the overlay is replace-only -- so
  every job carries `if: github.repository == 'adnaneGdihi/spectraMR'`. The lane
  is not yet switched on: no PyPI trusted publisher names this file, so PyPI
  still receives only tagged releases.
- **The branch model is written down.** `docs/versioning.rst` states what `main`,
  `dev` and `nightly` each are and what moves them, and records that nothing
  currently moves `nightly`: on 2026-09-06 it sat 10 commits behind `main`,
  carrying a documentation set `main` had already corrected.
- **A `docs` branch, documenting the other three.** Read the Docs builds the
  public repository and maps one version to one ref, so with `latest` reading
  `main` the published site could only ever describe the release: a docs
  correction was unreadable until the next release carried it, and restoring
  `main` to a release tree takes those corrections back off the site. `latest`
  now reads `docs`, which is *moved* to a snapshot whose documentation has been
  reviewed. It is a whole snapshot rather than a `docs/` directory, because
  without an importable `spectramr` Sphinx drops 77 of the 105 rendered
  `spectramr.*` signatures and seven of the nine autodoc pages' API sections
  and still exits 0 (`conf.py` suppresses `autodoc.import_object`, so
  `fail_on_warning` has nothing to fail on). What separates it from `dev` is the
  mover: `dev` advances on every export, `docs` only on a reviewed one. It lives
  in the published repository only -- the research repository already carries refs
  under `refs/heads/docs/` (40 of them on 2026-09-06), and git cannot hold a ref at
  `docs` and refs beneath `docs/` at the same time.

- **Non-Cartesian k-space.** Golden-angle radial acquisition, gridding by
  attention, a phase-exact per-annulus gain with one owner for the radial
  partition, and per-annulus latent tokens for the outer band.
- **Multi-coil operators.** A coil-subspace residual loss, a multi-coil
  equivariant-imaging operator and an ESPIRiT sensitivity transform.
- **A run survives the end of its cluster time allocation.** The allocation's end is derived
  once; training stops a margin ahead of it, saves, requeues and resumes with the
  random-number stream it left off on, rather than restarting the augmentation
  sequence.
- **Compilation and sharding.** Regional `torch.compile` with an explicit opt-out
  for complex-valued modules, placement chosen per parallel strategy, and bf16
  gated on the device's native compute capability rather than on a version
  string. The n2n and two Mamba cohorts shard on DeepSpeed ZeRO-2.
- **Cold diffusion can end its reverse loop at t=0**, as a terminal step, and
  error is attributed to the k-space band that wrote it.
- **Launch.** Process-group arms launch under `torchrun`, `-O` overrides reach
  every task of an experiment array, and the array wrapper accepts sbatch flags,
  a launch verb and node exclusion. One diagnostics make target runs every pass,
  with cohort and tree selection.

### Changed
- **Coil sensitivity maps are normalised after they are resized**, not before.
  Bilinear interpolation between unit-modulus complex values returns the chord,
  so maps normalised on the calibration grid and then upsampled fell to a mean
  squared norm of 0.58 between grid nodes; they now hold 1.0 inside the support.
  The SENSE projection and every coil-weighted loss and metric move with them, so
  results from before this change are not comparable with results after it.
- **Hard data consistency replaces only the acquired k-space bins**, and
  `null_space_content` is enabled across the k-space-filling experiments, so the
  unsampled bins receive a gradient that the post-consistency losses cannot
  supply. Metrics from before this change are not comparable either.
- **Loss weights have one declaration surface.** The domain-grouped loss lists
  build the loss modules and the weight table is derived from them; a `lambda_*`
  scalar that names the same loss at a different weight raises.
- **GAN critics score the prepared target in the domain they were built for**, and
  the composite-GAN weight is read from the configuration instead of a hardcoded
  10.0.
- **`IModel` is an `nn.Module`**, gradient reversal has one implementation, and
  model capabilities are read from one table.

### Deprecated
- `CheckpointDirector.cleanup_old_checkpoints`, which no code path calls.

### Fixed
- **The dev series could not be built at all.** `build_dist.py` compared
  `CHANGELOG.md`'s newest dated heading against the wheel's version as strings,
  but `bump_version.py nightly` deliberately writes a dev build no heading -- so
  on the 0.1.3 series the changelog read `0.1.2`, the wheel read `0.1.3.dev1`,
  and the build failed on a tree that was correct. `docs/versioning.rst`
  meanwhile described `nightly` as the branch that "exists to be installed and
  tried". The changelog is now judged by shape: a release must carry its own
  dated heading, a dev build must carry an open `[Unreleased]` section and must
  not name a version already released.
- **Two comparators owned "do the version sources agree".** `bump_version.py
  show` compared `len(set(values)) != 1`, which cannot express a dev build, and
  so reported `DISAGREEMENT` on every tree its own `nightly` mode had just
  written. Both now call `build_dist.version_disagreements`, which takes the
  reference version and the raw changelog text (non-negotiable 17).

- **Training produced NaN weights at the first optimizer step** on arms using the
  k-space magnitude clamp: the square root was taken before a lower bound was
  applied to the squared modulus.
- **Coil maps were dropped at the Fourier bridge**, so every arm declaring the
  coil-subspace residual crashed at the first iteration; and validation raised on
  `null_space_content` because the t=0 probe did not pass its mask on.
- **Three prior-method baselines did not run their authors' methods**, EMA
  blended zero tensors on 75 arms with a validation swap that could not restore
  the weights, and five backbone arms could not build a model.
- **The SSL k-space adjoint rendered k-space as an image**, so the n2n cohort
  did not train.
- The k-space mask no longer writes in place, a severed autograd graph in one
  strategy is reconnected, and `contrast_idx` is aligned to the flattened batch.

### Security
- **Six open vulnerability alerts on the published repository, closed by three
  lockfile bumps.** `urllib3` 2.5.0 -> 2.7.0 (four *high* advisories: cross-origin
  forwarding of sensitive headers through a proxy, and three decompression-bomb
  paths patched in 2.6.0 / 2.6.3 / 2.7.0), `idna` 3.13 -> 3.15 (specification
  deviation) and `pydantic-settings` 2.14.1 -> 2.14.2 (`NestedSecretsSettingsSource`
  follows symlinks). Dependabot raised all three against `adnaneGdihi/spectraMR`
  (#11, #12, #13); they are applied here because an export snapshot **replaces** the
  published tree, so a bump that lands only there is reverted by the next publish
  (#1878). `pydantic-settings>=2.14` in `pyproject.toml` already admits 2.14.2, and
  the other two are transitive, so no declared constraint changes.

## [0.1.2] - 2026-09-05

### Fixed
- **The Read the Docs build, which had failed on every version since 0.1.0.** Every
  build died in `pre_install` at ~22 s -- `latest`, `stable` and the Dependabot PR
  builds alike -- so the sphinx phase had never once executed there. `build.jobs`
  runs before Read the Docs upgrades pip, so the CPU-torch line executed under the
  `pip==23.1` that virtualenv seeds; that pip compares a wheel's metadata `Name:`
  against the index's project name without PEP 503 normalisation, discarded its own
  valid `typing_extensions` wheel, fell back to the sdist, and could not resolve the
  sdist's `flit_core` build dependency because `--index-url` replaces PyPI rather
  than adding to it. `pip install --upgrade pip` now runs first in `pre_install`.
- **The published documentation announced the wrong version.** `docs/conf.py` set
  `release` from a literal `"0.1.0"` -- a fifth version declaration that
  `build_dist.py` does not reconcile against the other four -- so every rendered page
  still said `0.1.0` after `v0.1.1` was cut. `conf.py` now parses `__version__` out of
  `src/spectramr/__init__.py`, the single writer, as text rather than by importing the
  package, and raises rather than falling back to a placeholder.
- **`MetricsRegistry` could answer with a wrong value instead of an error.** `clear()`,
  `snapshot()` and `restore()` were three independent enumerations of the mutable
  registration tables, and one of them was incomplete: the test-isolation helper
  restored three of the six tables `clear()` wipes, so `_needs` stayed empty for the
  rest of the process and every later `needs()` returned `()` -- which the caller
  cannot tell apart from "declares nothing". Under `pytest-xdist` that made the
  failure set depend on which test files shared a worker. All three helpers now
  iterate one declared tuple, and a test in `test_registry.py` fails if a table is
  added without listing it.

### Changed
- The export no longer ships 20 test files that could only fail here, because they
  exercise internal scripts and experiment cohorts this repository does not carry.
  `tests/utils/repo_scripts.py` gains `require_repo_file` and `skip_if_public_export`,
  so a test needing a file the export omits skips with a reason instead of erroring.
  The published tree goes from 4790 files to 4771.
- `make fuzz` and `make mutate` use the `$(VENV)` prefix the rest of the Makefile
  already uses, rather than spelling `. .venv/bin/activate` twice per target.
- Dependencies: `monai` 1.5.2 -> 1.6.0, `pillow` 12.2.0 -> 12.3.0, `setuptools`
  81.0.0 -> 83.0.0.


## [0.1.1] - 2026-09-04

### Added
- `scripts/release/zenodo_deposit.py` can deposit a **new version** of the published
  Zenodo record (`actions/newversion`) instead of only minting a new record, and
  discards the files Zenodo inherits from the previous version before uploading.
- README: a "Bring your own code" guide covering the three plugin-discovery layers
  (entry points, `SPECTRAMR_PLUGINS`, `plugins.paths`) and the collision rules.
- `.readthedocs.yaml` pre-installs the CPU torch wheel, so a docs build does not
  resolve the 5.1 GB CUDA stack it never uses.
- The **version policy** is written down and mechanised: `docs/versioning.rst` states the
  MAJOR.MINOR.BUILD scheme and the branch-to-version mapping (`main` carries stable
  releases, `nightly` the latest build, `dev` integration), and
  `scripts/release/bump_version.py` is the sole **writer** of the four independent
  version statements that `build_dist.py` compares -- a writer with its own idea of
  where the version lives is the second owner non-negotiable 17 forbids.
- The published repository's settings are a **declared, diffable model**
  (`public_repo_settings.yaml` in the research tree's release lane, which the export
  deliberately withholds) with `public_settings_diff.py` to compare it against the live
  repository and `public_settings_apply.py` to converge it, rather than console state
  nobody can review.
- The published repository runs a **two-lane CI**: a blocking `pr-required` lane and an
  advisory lane, shipped through the export overlay, plus a `manual-full-suite` workflow
  for the tiers too long to gate a PR on.
- The release lane -- build, test suite, GitHub release -- runs on the self-hosted
  `thor` runner.

### Changed
- The Zenodo concept-DOI check is now a gate that raises **before** `actions/publish`
  rather than a note printed after it; the weaker copy in `report_badge` is gone.
- README badges: the licence badge moves to the dynamic form now that the repository
  is public, and ~90 lines of commented-out badge archaeology were removed.
- Every GitHub Actions reference is **SHA-pinned**, and zizmor's advisory rules are
  absolute rather than best-effort.
- `gudhi` and `scalene` are marked off `linux/aarch64`, so `pip install -e ".[dev]"`
  resolves on ARM64. The two are excluded for *different* reasons: gudhi publishes
  neither an aarch64 wheel nor an sdist, so there is nothing to install; scalene ships
  an sdist but no linux-aarch64 wheel, so it would compile four native objects as a
  side effect of installing the test suite. macOS arm64 keeps its scalene wheel.
- `CubicalPHWassersteinLoss`'s `ImportError` now names **which** package is missing
  rather than the union of both, and on `linux/aarch64` says that the `[topology]`
  extra cannot supply gudhi there -- pointing at an extra that cannot help sends the
  reader round a loop that ends back at the same error.

### Removed
- README: the "What pip actually resolves" section. The load-bearing part -- cu126 is
  the last wheel lane shipping `sm_70`, so a V100 needs it -- is now under
  Installation as "Pinning the CUDA build".

### Fixed
- The public export ships `tests/unit/release/conftest.py`. A dropped `conftest.py` is a
  different failure shape from a dropped test subject: pytest resolves the bare name
  against the **root** `conftest.py`, fails to find the fixture, and aborts collection
  for the whole directory -- so the published tree lost far more than the one file, and
  only in the export. `test_export_public_tree.py` now gates that shape.

## [0.1.0] - 2026-09-04

<!-- Cut from `[Unreleased]` in preparation for the `v0.1.0` tag, which is the
     action that actually publishes. The heading is not cosmetic: it is one of the
     four version declarations `scripts/release/build_dist.py` reconciles, and
     `changelog_version()` deliberately SKIPS `[Unreleased]`, so while this section
     carried that heading the reconciler reported `CHANGELOG.md: unreadable` and
     `release.yml` would have failed at the build step -- before reaching PyPI --
     for every tag pushed. Verified by calling the reconciler directly rather than
     by reading it.

     The date must equal the day the tag is pushed. If that slips, change it here
     first; the tag is what makes the claim public, and until it is pushed this
     heading claims nothing that a reader can see. The earlier version of this file
     dated a released 0.1.0 at 2026-08-01 while no tag, no GitHub release and no
     PyPI distribution existed, which is the failure this note exists to prevent. -->

### Added
- Initial public release of spectraMR.
- Multi-paradigm training framework: 206 `training_mode` keys resolving to 153
  training-strategy classes.
- 586 registered models, reachable from a cold import after
  `populate_model_registry()`.
- 217 registered losses and 213 registered metrics, across image, k-space,
  complex, physics, adversarial, latent, distillation and virtual-fiducial
  domains.

<!-- Every count above is a REGISTRATION count, produced by calling
     populate_model_registry() and reading len(MODEL_REGISTRY) -- not a
     decorator-site grep. The two
     are different questions and nothing in CI compares them: a grep counts every
     textual occurrence, including the ones in comments, docstrings and tests, while
     the registry is what a config can actually reach. They also drift, in both
     directions -- the metric count above was 211 here and measured 213, and the
     model count of 590 above is an earlier reading than the rest. Re-run the snippet
     before changing a number here; do not carry one forward, and do not let a
     verified number vouch for its neighbour. -->
- Three-tier audit ladder (Tier 0 schema, Tier 1 static cross-validation,
  Tier 2 synthetic forward probe).
- MRI physics single source of truth under `src/spectramr/infrastructure/physics/`.
- Sphinx documentation and a Read the Docs build configuration
  (`.readthedocs.yaml`). The hosted site is not live as of v0.1.0.
- Apache-2.0 licence + clinical-use disclaimer.
- PyPI Trusted Publishing pipeline (`release.yml`) producing wheel + sdist on tag
  push. `spectramr` 0.1.0 is published, carrying both artefacts
  (`spectramr-0.1.0-py3-none-any.whl` and `spectramr-0.1.0.tar.gz`).
- GitHub Actions CI on pull requests: a blocking `pr-required` lane (changed-line
  lint, repository guards, architecture fitness functions, unit-test collection,
  physics tests, dependency and secret scanning) aggregated behind a single
  `required` check; an advisory `pr-advisory` lane (full lint, mypy, pre-commit,
  docs build, dependency review, zizmor); and CodeQL. The blocking lane runs the
  physics and architecture suites and **collects** the unit suite — it does not
  execute the full unit suite, which is too long for per-PR CI.
- Issue templates (bug / feature / paradigm-proposal) and PR template with DCO
  sign-off checklist.

<!-- Both links resolve as of 2026-09-04: `v0.1.0` is pushed and the GitHub Release
     is published. Verified by status, and the probe was checked against a tag that
     does not exist first -- `releases/tag/v9.9.9` and `compare/v9.9.9...HEAD` both
     return 404, so the 200s here mean something. The `[0.1.0]` heading date above
     and `date-released` in CITATION.cff are two further declarations of the same
     day, and nothing reconciles the three; change them together. -->
[Unreleased]: https://github.com/adnaneGdihi/spectramr/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/adnaneGdihi/spectramr/releases/tag/v0.1.2
[0.1.1]: https://github.com/adnaneGdihi/spectramr/releases/tag/v0.1.1
[0.1.0]: https://github.com/adnaneGdihi/spectramr/releases/tag/v0.1.0
